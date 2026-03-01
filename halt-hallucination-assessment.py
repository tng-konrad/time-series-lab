#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HALT: Hallucination Assessment via Log-probs as Time Series
===========================================================
Consolidated PyTorch implementation of:
  "HALT: Hallucination Assessment via Log-probs as Time series"
  Shapiro, Taneja, Goel — Georgia Tech, 2026

Key features:
- Numerically stable feature extraction (Eq. 4–13)
- Bidirectional GRU encoder with projection + dropout
- Top-q pooling on L2-saliency timesteps (Sec 3.2)
- Full training pipeline: early stopping, LR scheduling, grad clipping
- Feature importance via Gradient×Input (Appendix C.2)
- Reproducible, well-documented, production-ready
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


from haltvisualise import (plot_feature_importance, plot_t_sne_embeddings,
                plot_feature_correlation, plot_sensitivity_analysis)



# ──────────────────────────────────────────────────────────────────────
# Reproducibility & device setup
# ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class HALTConfig:
    # Feature dimensions (Sec 3.2 + Fig.1)
    top_k: int = 20
    n_stat_features: int = 5  # [avg_logp, rank_proxy, h_overall, h_alts, delta_h_dec]
    input_dim: int = 25       # top_k + n_stat_features

    # Projection MLP (Sec 3.2, Appendix B)
    proj_dim: int = 128

    # Bidirectional GRU (Appendix B)
    hidden_dim: int = 256
    num_layers: int = 5
    dropout: float = 0.4

    # Top-q salient pooling (Eq Sec 3.2)
    top_q: float = 0.15

    # Training (Appendix B)
    batch_size: int = 512
    max_epochs: int = 5 # 100
    lr: float = 4.41e-4
    weight_decay: float = 2.34e-6
    lr_patience: int = 3
    lr_factor: float = 0.5
    early_stop_patience: int = 15
    max_grad_norm: float = 1.0


# ──────────────────────────────────────────────────────────────────────
# Feature Extraction (Sec 3.2, numerically stable)
# ──────────────────────────────────────────────────────────────────────
EPS = 1e-9


def compute_engineered_features(logprobs_k: torch.Tensor) -> torch.Tensor:
    """
    Compute 5 token-level engineered features from top-k log-probs (Sec 3.2).

    Args:
        logprobs_k: [T, K] — top-k log-probs; col 0 = selected token.

    Returns:
        [T, 5] — engineered features
            [avg_logp, rank_proxy, h_overall, h_alts, delta_h_dec]
    """
    T, K = logprobs_k.shape

    # Stable truncated distribution over top-k (Eq 4)
    mt = logprobs_k.max(dim=1, keepdim=True).values
    exp_s = torch.exp(logprobs_k - mt)
    p_tilde = exp_s / (exp_s.sum(dim=1, keepdim=True) + EPS)

    # 1. AvgLogP (Eq 5)
    avg_logp = logprobs_k.mean(dim=1)  # [T]

    # 2. RankProxy (Eq 6)
    sel = logprobs_k[:, :1]            # [T, 1]
    rank_proxy = 1.0 + (logprobs_k[:, 1:] > sel).float().sum(dim=1)  # [T]

    # 3. Overall entropy over truncated distribution (Eq 7)
    h_overall = -(p_tilde * torch.log(p_tilde + EPS)).sum(dim=1)  # [T]

    # 4. Alternatives-only entropy (Eq 8–9)
    p_alts = p_tilde[:, 1:]                            # [T, K-1]
    p_alts_n = p_alts / (p_alts.sum(dim=1, keepdim=True) + EPS)
    h_alts = -(p_alts_n * torch.log(p_alts_n + EPS)).sum(dim=1)   # [T]

    # 5. Temporal delta of binary decision entropy (Eq 10–13)
    best_alt = logprobs_k[:, 1:].max(dim=1).values     # [T]
    sel_lp = logprobs_k[:, 0]                          # [T]

    pc_num = torch.exp(sel_lp)
    pc_den = torch.exp(sel_lp) + torch.exp(best_alt)   # [T]
    pc = pc_num / (pc_den + EPS)
    pc = pc.clamp(EPS, 1.0 - EPS)                      # avoid log(0)
    h_dec = -(pc * torch.log(pc) + (1 - pc) * torch.log(1 - pc))  # [T]

    delta_h_dec = h_dec.clone()
    delta_h_dec[1:] = h_dec[1:] - h_dec[:-1]

    return torch.stack([avg_logp, rank_proxy, h_overall, h_alts, delta_h_dec], dim=1)


def build_halt_input(logprobs_k: torch.Tensor) -> torch.Tensor:
    """
    Concatenate engineered stats with raw log-probs → [T, 25].

    Order: [stats (5) || raw_logprobs (20)] as in paper Fig. 1 / Sec 3.2.
    """
    stats = compute_engineered_features(logprobs_k)     # [T, 5]
    return torch.cat([stats, logprobs_k], dim=1)        # [T, 25]


# ──────────────────────────────────────────────────────────────────────
# Synthetic Dataset Generator (Sec 4.1)
# ──────────────────────────────────────────────────────────────────────
def _synthetic_logprobs(
    seq_len: int,
    top_k: int,
    hallucinated: bool,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Synthesize realistic top-k log-prob matrix for hallucinated vs correct tokens.

    Hallucinated: flatter distribution (small gaps, high noise)
    Correct: peaked distribution (large gaps, low noise)

    Returns:
        [seq_len, top_k] float32 log-probs, sorted descending per row.
    """
    result = np.zeros((seq_len, top_k), dtype=np.float32)
    for t in range(seq_len):
        if hallucinated:
            base = rng.uniform(-4.0, -0.3)      # lower mean (more uncertainty)
            gap  = rng.uniform(0.05, 0.3)       # narrow gaps (indistinguishable)
        else:
            base = rng.uniform(-1.5, -0.1)      # higher mean (confident)
            gap  = rng.uniform(0.5, 2.0)        # large gaps (confident ranking)
        # Alternatives: base - gap*(1..K-1) + small noise
        alts = base - gap * np.arange(1, top_k)
        alts += rng.uniform(-0.5, 0.5, top_k - 1)
        result[t] = np.concatenate([[base], alts]).astype(np.float32)
    return result


class SyntheticLogProbDataset(Dataset):
    """
    Synthetic dataset: (feature_sequence, label) pairs.
    Each sample = one LLM response as top-k log-prob time series.
    """
    def __init__(
        self,
        n_samples: int = 4000,
        top_k: int = 20,
        min_len: int = 10,
        max_len: int = 150,
        hallucination_rate: float = 0.5,
        seed: int = SEED,
    ):
        rng = np.random.default_rng(seed)
        self.samples: List[Tuple[torch.Tensor, int]] = []
        for _ in range(n_samples):
            label   = int(rng.random() < hallucination_rate)
            seq_len = int(rng.integers(min_len, max_len + 1))
            raw     = _synthetic_logprobs(seq_len, top_k, bool(label), rng)
            x       = build_halt_input(torch.from_numpy(raw))
            self.samples.append((x, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        x, label = self.samples[idx]
        return x, label, x.shape[0]   # tensor, label, length


def collate_fn(batch):
    xs, labels, lengths = zip(*batch)
    
    # Convert list of tensors to padded tensor (requires_grad kept)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    idx       = lengths_t.argsort(descending=True)
    
    xs_sorted   = [xs[i] for i in idx]
    lengths_s   = lengths_t[idx]
    
    # 🔑关键：pad_sequence preserves requires_grad if inputs have it
    padded = pad_sequence(xs_sorted, batch_first=True, padding_value=0.0)
    
    # If needed, ensure grad flag is preserved:
    # padded.retain_grad()  # optional, not needed here
    
    labels_t = torch.tensor([labels[i] for i in idx], dtype=torch.float)
    
    return padded, labels_t, lengths_s



# ──────────────────────────────────────────────────────────────────────
# Model Blocks (Sec 3.2 + Appendix B)
# ──────────────────────────────────────────────────────────────────────
class InputProjection(nn.Module):
    """
    LayerNorm → Linear → GELU → Linear  (Sec 3.2)
    [B,T,25] → [B,T,128]
    """
    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1  = nn.Linear(input_dim, proj_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(proj_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class TopQPooling(nn.Module):
    """
    Average top-q salient timesteps (Sec 3.2).
    [B,T,D], lengths → [B,D]
    """
    def __init__(self, q: float = 0.15):
        super().__init__()
        self.q = q

    def forward(self, H: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, D = H.shape
        mask   = torch.arange(T, device=H.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = H.norm(dim=2).masked_fill(~mask, -1e9)  # L2 norm + padding mask
        k_vals = (lengths.float() * self.q).ceil().long().clamp(min=1)

        pooled = torch.zeros(B, D, device=H.device)
        for b in range(B):
            top_idx = scores[b].topk(k_vals[b].item()).indices
            pooled[b] = H[b, top_idx].mean(dim=0)
        return pooled


class HALT(nn.Module):
    """
    HALT: Hallucination Assessment via Log-probs as Time series
    (Sec 3.2 + Appendix B)

    Input:
        x:       [B, T, D=25] padded feature sequences
        lengths: [B]          original sequence lengths
    Output:
        logits:  [B]          raw hallucination scores (apply sigmoid)
    """
    def __init__(self, config: HALTConfig):
        super().__init__()
        self.cfg = config
        self.projection = InputProjection(config.input_dim, config.proj_dim)

        # Bidirectional GRU encoder (Appendix B)
        self.gru = nn.GRU(
            input_size=config.proj_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        gru_out_dim = config.hidden_dim * 2   # 512
        self.pooler = TopQPooling(q=config.top_q)
        self.classifier = nn.Linear(gru_out_dim, 1)

        # Proper weight initialization (Appendix B)
        self._init_weights()

    def _init_weights(self):
        # GRU weights: xavier for input-hidden, orthogonal for hidden-hidden
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        proj    = self.projection(x)                          # [B,T,128]
        lengths_cpu = lengths.cpu()  # pack_padded_sequence requires CPU

        # 🔑 Sort lengths descending (required by pack_padded_sequence)
        idx_sorted = torch.sort(lengths_cpu, descending=True).indices
        lengths_sorted = lengths_cpu[idx_sorted]
        proj_sorted    = proj[idx_sorted]

        packed  = pack_padded_sequence(proj_sorted, lengths_sorted,
                                       batch_first=True, enforce_sorted=True)
        out_pk, _ = self.gru(packed)
        H, _    = pad_packed_sequence(out_pk, batch_first=True,
                                      total_length=T)        # [B,T,512]

        # Restore original order for pooling & loss
        inv_idx = idx_sorted.argsort()
        H_orig  = H[inv_idx]

        pooled  = self.pooler(H_orig, lengths)               # [B,512]
        logits  = self.classifier(pooled).squeeze(1)         # [B]
        return logits


# ──────────────────────────────────────────────────────────────────────
# Training & Evaluation Utilities (Appendix B)
# ──────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, max_grad_norm: float):
    model.train()
    total_loss, n = 0.0, 0
    for x_pad, labels, lengths in loader:
        x_pad = x_pad.to(DEVICE)
        labels = labels.to(DEVICE)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_pad, lengths)
        loss   = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for x_pad, labels, lengths in loader:
        x_pad = x_pad.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(x_pad, lengths)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        n += 1
        all_logits.extend(logits.cpu().tolist())
        all_labels.extend(labels.cpu().int().tolist())

    probs    = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds    = (probs >= threshold).astype(int)
    labels_np = np.array(all_labels)

    macro_f1  = f1_score(labels_np, preds, average="macro", zero_division=0)
    accuracy  = (preds == labels_np).mean()
    try:
        auroc   = roc_auc_score(labels_np, probs)
    except ValueError:  # e.g., single class
        auroc   = 0.5

    return {
        "loss": total_loss / max(n, 1),
        "macro_f1": macro_f1,
        "auroc": auroc,
        "accuracy": accuracy,
    }


def train_halt(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    cfg: HALTConfig,
) -> float:
    """Training loop with early stopping & LR scheduling."""
    best_f1, best_state, patience_ctr = -float("inf"), None, 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 cfg.max_grad_norm)

        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_metrics["macro_f1"])

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{cfg.max_epochs} | "
                f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f} | "
                f"macro-F1={val_metrics['macro_f1']:.4f}, AUROC={val_metrics['auroc']:.4f}"
            )

        # Early stopping based on macro-F1
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best val macro-F1: {best_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1


# ──────────────────────────────────────────────────────────────────────
# Inference & Interpretability (Appendix C.2)
# ──────────────────────────────────────────────────────────────────────
FEATURE_NAMES = (
    ["avg_logp", "rank_proxy", "h_overall", "h_alts", "delta_h_dec"]
    + [f"logprob_{i+1}" for i in range(20)]
)


def predict_hallucination(logprobs_matrix: np.ndarray, model: nn.Module,
                          device: str = DEVICE, threshold: float = 0.5) -> dict:
    """
    Predict hallucination for a single LLM response.

    Args:
        logprobs_matrix: [T, top_k] — top-k log-probs per token.
                         Column 0 = selected token, rest = alternatives sorted desc.

    Returns:
        dict with prob_hallucinated, prediction, logit
    """
    model.eval()
    x = build_halt_input(torch.from_numpy(logprobs_matrix.astype(np.float32)))
    x = x.unsqueeze(0).to(device)                      # [1,T,25]
    lengths = torch.tensor([logprobs_matrix.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        logit = model(x, lengths).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()

    return {"prob_hallucinated": prob, "prediction": int(prob >= threshold), "logit": logit}

def compute_feature_importance(model, loader, device, n_features=25) -> Dict[str, float]:
    """
    Feature importance via Gradient × Input (Appendix C.2).
    Safely handles model grad state.
    """
    # 🔑 Ensure model is in train mode AND parameters require grad
    model.train()
    
    for p in model.parameters():
        if not p.requires_grad:
            print("⚠️  Enabling grad on parameter:", p.shape)
        p.requires_grad_(True)

    feat_imp = torch.zeros(n_features, device=device)

    for x_pad, labels, lengths in loader:
        # Ensure x_pad requires grad
        x_pad = torch.nn.Parameter(x_pad.to(device), requires_grad=True)
        labels = labels.to(device)

        logits = model(x_pad, lengths)  # [B]
        
        # Compute gradient of sum (scalar) w.r.t. inputs
        loss = logits.sum()
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=x_pad,
            retain_graph=False,
            create_graph=False
        )[0]

        B, T, D = grads.shape
        mask    = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(2)
        grad_input = (grads.abs() * x_pad.abs()) * mask

        feat_imp += grad_input.sum(dim=(0, 1))
    
    model.eval()
    feat_imp /= feat_imp.sum() + EPS
    importance_dict = dict(zip(FEATURE_NAMES, feat_imp.tolist()))
    return dict(sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True))




# ──────────────────────────────────────────────────────────────────────
# Shape sanity & backward pass check (debug/coverage)
# ──────────────────────────────────────────────────────────────────────
def _shape_sanity_check(model, cfg: HALTConfig):
    B_t, T_t = 4, 80
    x   = torch.randn(B_t, T_t, cfg.input_dim).to(DEVICE)
    lens = torch.tensor([80, 65, 40, 20], dtype=torch.long).to(DEVICE)
    lbls = torch.zeros(B_t).to(DEVICE)

    out = model(x, lens)
    loss = nn.BCEWithLogitsLoss()(out, lbls)
    loss.backward()

    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Input shape:   {list(x.shape)}")
    print(f"  Output shape:  {list(out.shape)}")
    print(f"  Loss:          {loss.item():.4f}")
    print(f"  Gradients OK? {has_grads} ✓")


# ──────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("HALT: Hallucination Assessment via Log-probs as Time series")
    print("Consolidated PyTorch Implementation (v1.0)")
    print("=" * 72)

    cfg = HALTConfig()
    print(f"\n⚙️ Configuration:\n{cfg}")
    print(f"💾 Device: {DEVICE}")

    # ── Dataset
    full_ds = SyntheticLogProbDataset(n_samples=4000, top_k=cfg.top_k, seed=SEED)
    n_total = len(full_ds)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    loader_kw = dict(collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size,
                              shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_set,  batch_size=cfg.batch_size,
                              shuffle=False, **loader_kw)

    print(f"\n📊 Dataset split:")
    print(f"  Train: {len(train_set):>5} | Val: {len(val_set):>5} | Test: {len(test_set):>5}")

    # ── Model
    model = HALT(cfg).to(DEVICE)

    for p in model.parameters():
        p.requires_grad_(True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: HALT ({n_params:,} params ≈ {n_params/1e6:.1f}M)")

    # ── Optimizer & scheduler
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
#         verbose=False
    )

    # ── Training loop
    print("\n" + "=" * 72)
    print("🚀 Training HALT")
    print("=" * 72)

    best_val_f1 = train_halt(model, train_loader, val_loader,
                             optimizer, scheduler, criterion, cfg)

    print(f"\n🏆 Best validation macro-F1: {best_val_f1:.4f}")

    # ── Final test evaluation
    print("\n" + "=" * 72)
    print("🧪 Final Test Evaluation")
    print("=" * 72)

    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    print(f"  Macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  AUROC    : {test_metrics['auroc']:.4f}")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Loss     : {test_metrics['loss']:.4f}")

    # ── Example inference
    print("\n" + "=" * 72)
    print("🔍 Example Inference")
    print("=" * 72)

    rng_demo = np.random.default_rng(123)
    for label_str, is_hall in [("Hallucinated", True), ("Correct", False)]:
        demo = _synthetic_logprobs(60, cfg.top_k, is_hall, rng_demo)
        res  = predict_hallucination(demo, model, DEVICE)
        pred_str = "HALLUCINATED" if res["prediction"] else "NOT HALLUCINATED"
        print(f"[{label_str:>13}]  P(hallucinated)={res['prob_hallucinated']:.4f} "
              f"→ {pred_str}")

    # ── Feature importance (Appendix C.2)
    print("\n" + "=" * 72)
    print("📊 Feature Importance (Gradient×Input, top-10)")
    print("=" * 72)

    print("🔍 Testing gradient flow...")
    dummy_x = torch.randn(2, 5, cfg.input_dim, device=DEVICE, requires_grad=True)
    dummy_lens = torch.tensor([5, 3], device=DEVICE)

    model.eval()
    logits = model(dummy_x, dummy_lens)
    print(f" logits.requires_grad? {logits.requires_grad}")
    loss = nn.BCEWithLogitsLoss()(logits, torch.zeros(2, device=DEVICE))
    loss.backward()
    print(f"dummy x.grad exists? {dummy_x.grad is not None}")
    assert dummy_x.grad is not None, "Break in gradient flow!"
    print("✅ Gradient test PASSED")



    importance = compute_feature_importance(model, val_loader, DEVICE,
                                            n_features=cfg.input_dim)
    print(f"{'Feature':<22}  {'Importance':>10}")
    print("-" * 36)
    for name, score in list(importance.items())[:10]:
        print(f"{name:<22}  {score:>10.4f}")

    # ── Shape & backward pass check
    print("\n" + "=" * 72)
    print("🔍 Shape Sanity Check & Backward Pass")
    print("=" * 72)
    _shape_sanity_check(model, cfg)

    # ── Save model (optional)
    torch.save(model.state_dict(), "halt_model.pt")
    print("\n💾 Model saved to halt_model.pt")


    # plotting

    importance = compute_feature_importance(model, val_loader, DEVICE)
    plot_feature_importance(importance, output_path="feature_importance.png")

    plot_feature_correlation(model, val_loader, DEVICE)


    plot_t_sne_embeddings(
        model, val_loader, DEVICE,
        output_path="tsne_embeddings.png",
        n_samples=1500
    )

    plot_sensitivity_analysis(model, val_loader, DEVICE)


if __name__ == "__main__":
    main()
