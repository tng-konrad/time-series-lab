"""
HALT: Hallucination Assessment via Log-probs as Time Series
Paper: Shapiro, Taneja & Goel (2026) — arXiv:2602.02888

Architecture:
    Input  (B, T, 25)  — top-20 log-probs + 5 engineered features
    → LayerNorm + 2-layer GELU MLP → (B, T, 128)
    → BiGRU (hidden=256, layers=5, dropout=0.4) → (B, T, 512)
    → Top-q pooling (q=0.15) → (B, 512)
    → Linear → (B, 1) logit
"""

import math
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class HALTConfig:
    # Feature dimensions
    top_k: int = 20              # number of top log-probs per token
    n_eng_features: int = 5      # engineered uncertainty features
    input_dim: int = 25          # top_k + n_eng_features

    # Projection MLP
    proj_dim: int = 128

    # BiGRU encoder
    hidden_dim: int = 256        # per-direction hidden size
    n_gru_layers: int = 5
    gru_dropout: float = 0.4
    bidirectional: bool = True

    # Top-q pooling
    top_q: float = 0.15

    # Classifier
    out_norm: bool = False       # disabled in best reported setting

    # Training
    batch_size: int = 512
    lr: float = 4.41e-4
    weight_decay: float = 2.34e-6
    max_epochs: int = 100
    early_stop_patience: int = 15
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    grad_clip_max_norm: float = 1.0

    @property
    def gru_output_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)


# =============================================================================
# FEATURE EXTRACTION  (Section 3.2, Equations 4–13)
# =============================================================================

def safe_entropy(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Shannon entropy of a probability vector along last dim."""
    return -(probs * (probs + eps).log()).sum(dim=-1)


def renorm_top_k(logprobs: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable softmax over top-k log-probs (Eq. 4).
    logprobs : (T, k) → p_tilde : (T, k)
    """
    m = logprobs.max(dim=-1, keepdim=True).values
    exp_l = (logprobs - m).exp()
    return exp_l / exp_l.sum(dim=-1, keepdim=True)


def extract_features(logprobs: torch.Tensor) -> torch.Tensor:
    """
    Extract 5 engineered uncertainty features for each token step.

    Args:
        logprobs : (T, k)  column 0 = selected token log-prob

    Returns:
        features : (T, 5)  [avg_logprob, rank_proxy, entropy_overall,
                             entropy_alts, dec_entropy_delta]
    """
    T, k = logprobs.shape

    # 1. Average log-probability (Eq. 5)
    avg_logprob = logprobs.mean(dim=-1, keepdim=True)               # (T, 1)

    # 2. Rank proxy (Eq. 6)
    selected_lp = logprobs[:, 0:1]                                  # (T, 1)
    alts_lp     = logprobs[:, 1:]                                   # (T, k-1)
    rank_proxy  = 1.0 + (alts_lp > selected_lp).float().sum(dim=-1, keepdim=True)  # (T, 1)

    # Renormalised distribution (Eq. 4)
    p_tilde = renorm_top_k(logprobs)                                # (T, k)

    # 3. Overall entropy (Eq. 7)
    entropy_overall = safe_entropy(p_tilde).unsqueeze(-1)           # (T, 1)

    # 4. Alternatives-only entropy (Eq. 8-9)
    p_alts      = p_tilde[:, 1:]                                    # (T, k-1)
    p_alts_norm = p_alts / (p_alts.sum(dim=-1, keepdim=True) + 1e-9)
    entropy_alts = safe_entropy(p_alts_norm).unsqueeze(-1)          # (T, 1)

    # 5. Decision entropy delta (Eq. 10-13)
    best_alt_lp = alts_lp.max(dim=-1).values                       # (T,)
    pc = selected_lp.squeeze().exp() / (
        selected_lp.squeeze().exp() + best_alt_lp.exp() + 1e-9
    )                                                               # (T,)
    h_dec = -(pc * (pc + 1e-9).log() + (1 - pc) * (1 - pc + 1e-9).log())  # (T,)
    delta_h_dec = torch.cat([
        torch.zeros(1, device=logprobs.device),
        h_dec[1:] - h_dec[:-1]
    ]).unsqueeze(-1)                                                # (T, 1)

    return torch.cat([avg_logprob, rank_proxy, entropy_overall,
                      entropy_alts, delta_h_dec], dim=-1)          # (T, 5)


def build_input_sequence(logprobs: torch.Tensor) -> torch.Tensor:
    """
    Build enriched feature sequence l̃_{1:T} (Section 3.2).

    Args:
        logprobs : (T, k)
    Returns:
        x_tilde  : (T, k+5=25)
    """
    eng = extract_features(logprobs)                # (T, 5)
    return torch.cat([eng, logprobs], dim=-1)        # (T, 25)


# =============================================================================
# DATASET
# =============================================================================

class HALTDataset(Dataset):
    """
    samples : list of (logprobs_tensor (T_i, top_k), label int)
    """
    def __init__(self, samples: List[Tuple[torch.Tensor, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        logprobs, label = self.samples[idx]
        x_tilde = build_input_sequence(logprobs)    # (T_i, 25)
        length  = x_tilde.shape[0]
        return x_tilde, torch.tensor(label, dtype=torch.float32), length


def halt_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        x_padded : (B, T_max, 25)
        labels   : (B,)
        lengths  : (B,)
        mask     : (B, T_max)  True at valid positions
    """
    x_list, labels, lengths = zip(*batch)
    lengths  = torch.tensor(lengths, dtype=torch.long)
    labels   = torch.stack(labels)
    x_padded = pad_sequence(x_list, batch_first=True)              # (B, T_max, 25)
    T_max    = x_padded.shape[1]
    mask     = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T_max)
    return x_padded, labels, lengths, mask


def make_dataloader(samples, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        HALTDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=halt_collate_fn,
        drop_last=False,
    )


def make_synthetic_dataset(
    n: int = 1000, top_k: int = 20, min_len: int = 10, max_len: int = 150
) -> List[Tuple[torch.Tensor, int]]:
    """
    Synthetic data: hallucinated responses contain injected entropy spikes.
    Replace this with real LLM log-prob extraction for actual use.
    """
    samples = []
    for _ in range(n):
        T     = random.randint(min_len, max_len)
        label = random.randint(0, 1)
        lp    = torch.zeros(T, top_k)
        for i in range(top_k):
            lp[:, i] = -float(i) * (0.5 + torch.rand(T) * 0.5)
        if label == 1:
            n_spikes  = random.randint(1, max(1, T // 10))
            spike_pos = random.sample(range(T), min(n_spikes, T))
            for pos in spike_pos:
                lp[pos] = -torch.rand(top_k) * 2.0
        samples.append((lp, label))
    return samples


# =============================================================================
# MODEL
# =============================================================================

class InputProjection(nn.Module):
    """
    LayerNorm + 2-layer GELU MLP.
    (B, T, input_dim) → (B, T, proj_dim)
    """
    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class TopQPooling(nn.Module):
    """
    Average the top-q fraction of timesteps by ℓ₂ norm.
    Ignores padded positions via boolean mask.

    (B, T, D), (B, T) → (B, D)
    """
    def __init__(self, q: float = 0.15):
        super().__init__()
        self.q = q

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = H.shape
        scores  = H.norm(dim=-1)                                    # (B, T)
        scores  = scores.masked_fill(~mask, float("-inf"))

        valid_lengths = mask.sum(dim=1).float()                     # (B,)
        K = int((self.q * valid_lengths).ceil().clamp(min=1).max().item())

        _, top_idx  = scores.topk(K, dim=1)                        # (B, K)
        idx_exp     = top_idx.unsqueeze(-1).expand(-1, -1, D)       # (B, K, D)
        top_states  = H.gather(1, idx_exp)                          # (B, K, D)
        return top_states.mean(dim=1)                               # (B, D)


class HALT(nn.Module):
    """
    HALT: Hallucination Assessment via Log-probs as Time Series.

    forward(x, lengths, mask) → logit (B,)
        x       : (B, T, 25)  padded enriched features
        lengths : (B,)        actual sequence lengths
        mask    : (B, T)      True at valid positions
    """
    def __init__(self, config: HALTConfig):
        super().__init__()
        self.config = config

        self.projection = InputProjection(config.input_dim, config.proj_dim)

        self.gru = nn.GRU(
            input_size=config.proj_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_gru_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.gru_dropout if config.n_gru_layers > 1 else 0.0,
        )

        self.pool     = TopQPooling(q=config.top_q)
        self.out_norm = (nn.LayerNorm(config.gru_output_dim)
                         if config.out_norm else nn.Identity())
        self.classifier = nn.Linear(config.gru_output_dim, 1)

    def forward(
        self,
        x: torch.Tensor,       # (B, T, 25)
        lengths: torch.Tensor, # (B,)
        mask: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:         # (B,)

        x_proj = self.projection(x)                                # (B, T, 128)

        packed         = pack_padded_sequence(
            x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_packed, _  = self.gru(packed)
        gru_out, _     = pad_packed_sequence(gru_packed, batch_first=True)  # (B, T', 512)

        # Re-pad to original T if pad_packed_sequence truncates
        T_padded = x.shape[1]
        if gru_out.shape[1] < T_padded:
            pad_size = T_padded - gru_out.shape[1]
            gru_out  = F.pad(gru_out, (0, 0, 0, pad_size))

        pooled = self.pool(gru_out, mask)                          # (B, 512)
        pooled = self.out_norm(pooled)
        logit  = self.classifier(pooled).squeeze(-1)               # (B,)
        return logit


# =============================================================================
# METRICS
# =============================================================================

def macro_f1(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (logits.sigmoid() >= threshold).long()
    y     = labels.long()
    f1s   = []
    for cls in [0, 1]:
        tp = ((preds == cls) & (y == cls)).sum().float()
        fp = ((preds == cls) & (y != cls)).sum().float()
        fn = ((preds != cls) & (y == cls)).sum().float()
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1s.append((2 * p * r / (p + r + 1e-9)).item())
    return float(np.mean(f1s))


def accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    return ((logits.sigmoid() >= threshold).long() == labels.long()).float().mean().item()


# =============================================================================
# TRAIN / EVAL STEPS
# =============================================================================

def train_step(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    dev: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for x, labels, lengths, mask in loader:
        x, labels         = x.to(dev), labels.to(dev)
        lengths, mask     = lengths.to(dev), mask.to(dev)

        optimizer.zero_grad()
        logits = model(x, lengths, mask)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return total_loss / len(all_labels), macro_f1(all_logits, all_labels)


@torch.no_grad()
def eval_step(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    dev: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for x, labels, lengths, mask in loader:
        x, labels         = x.to(dev), labels.to(dev)
        lengths, mask     = lengths.to(dev), mask.to(dev)

        logits = model(x, lengths, mask)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    n = len(all_labels)
    return total_loss / n, macro_f1(all_logits, all_labels), accuracy(all_logits, all_labels)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    config: HALTConfig,
    dev: torch.device,
) -> Dict:
    best_val_f1      = -1.0
    epochs_no_imprv  = 0
    best_state       = None
    history          = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, config.max_epochs + 1):
        tr_loss, tr_f1          = train_step(model, train_loader, optimizer,
                                             criterion, config.grad_clip_max_norm, dev)
        va_loss, va_f1, va_acc  = eval_step(model, val_loader, criterion, dev)
        scheduler.step(va_f1)

        history["train_loss"].append(tr_loss)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_f1"].append(va_f1)

        print(
            f"Epoch {epoch:3d}/{config.max_epochs} | "
            f"Train loss={tr_loss:.4f} F1={tr_f1:.4f} | "
            f"Val loss={va_loss:.4f} F1={va_f1:.4f} Acc={va_acc:.4f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1     = va_f1
            epochs_no_imprv = 0
            best_state      = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_imprv += 1
            if epochs_no_imprv >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint  (val macro-F1={best_val_f1:.4f})")

    return history


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def predict_hallucination(
    logprobs: torch.Tensor,
    model: nn.Module,
    dev: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """
    Predict hallucination for a single response.

    Args:
        logprobs  : (T, 20)  raw top-20 log-probs from the LLM
        model     : trained HALT instance
        dev       : torch device
        threshold : decision boundary

    Returns:
        dict with hallucination_probability and is_hallucinated flag
    """
    model.eval()
    x_tilde = build_input_sequence(logprobs)                        # (T, 25)
    x_b     = x_tilde.unsqueeze(0).to(dev)                         # (1, T, 25)
    lengths = torch.tensor([x_tilde.shape[0]], device=dev)
    mask    = torch.ones(1, x_tilde.shape[0], dtype=torch.bool, device=dev)

    logit = model(x_b, lengths, mask)
    prob  = logit.sigmoid().item()
    return {
        "hallucination_probability": prob,
        "is_hallucinated": prob >= threshold,
        "threshold": threshold,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HALT hallucination detector")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--n_train",    type=int,   default=800)
    parser.add_argument("--n_val",      type=int,   default=100)
    parser.add_argument("--n_test",     type=int,   default=100)
    parser.add_argument("--save",       type=str,   default="halt_checkpoint.pt")
    args = parser.parse_args()

    print(f"Device: {device}\n")

    # Config
    cfg             = HALTConfig()
    cfg.max_epochs  = args.epochs
    cfg.batch_size  = args.batch_size

    # Data
    print("Building synthetic dataset …")
    train_samples = make_synthetic_dataset(n=args.n_train)
    val_samples   = make_synthetic_dataset(n=args.n_val)
    test_samples  = make_synthetic_dataset(n=args.n_test)

    train_loader  = make_dataloader(train_samples, cfg.batch_size, shuffle=True)
    val_loader    = make_dataloader(val_samples,   cfg.batch_size, shuffle=False)
    test_loader   = make_dataloader(test_samples,  cfg.batch_size, shuffle=False)

    # Model
    model     = HALT(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  ({n_params/1e6:.2f}M)\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )

    # Train
    history = train(model, train_loader, val_loader,
                    optimizer, scheduler, criterion, cfg, device)

    # Test
    test_loss, test_f1, test_acc = eval_step(model, test_loader, criterion, device)
    print(f"\n=== Test Results ===")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Macro-F1  : {test_f1:.4f}")
    print(f"  Accuracy  : {test_acc:.4f}")

    # Save
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config":               cfg,
        "history":              history,
        "test_macro_f1":        test_f1,
    }, args.save)
    print(f"\nCheckpoint saved → {args.save}")

    # Demo inference
    sample_lp, sample_label = test_samples[0]
    result = predict_hallucination(sample_lp, model, device)
    print(f"\n--- Inference demo ---")
    print(f"True label               : {'hallucinated' if sample_label == 1 else 'faithful'}")
    print(f"Hallucination probability: {result['hallucination_probability']:.4f}")
    print(f"Predicted                : {'hallucinated' if result['is_hallucinated'] else 'faithful'}")


if __name__ == "__main__":
    main()
