"""
HALT: Hallucination Assessment via Log-probs as Time Series
===========================================================
PyTorch implementation of:
  "HALT: Hallucination Assessment via Log-probs as Time series"
  Shapiro, Taneja, Goel — Georgia Tech, 2026

Usage:
  python halt.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, roc_auc_score

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class HALTConfig:
    # Feature dimensions
    top_k: int = 20              # top-k log-probs per token step
    n_stat_features: int = 5     # AvgLogP, RankProxy, Hoverall, Halts, ΔHdec
    input_dim: int = 25          # top_k + n_stat_features

    # Projection MLP
    proj_dim: int = 128

    # Bidirectional GRU encoder
    hidden_dim: int = 256        # per-direction hidden size
    num_layers: int = 5
    dropout: float = 0.4

    # Top-q salient pooling
    top_q: float = 0.15          # fraction of timesteps to average

    # Training
    batch_size: int = 512
    max_epochs: int = 100
    lr: float = 4.41e-4
    weight_decay: float = 2.34e-6
    lr_patience: int = 3
    lr_factor: float = 0.5
    early_stop_patience: int = 15
    max_grad_norm: float = 1.0


# ── Engineered Feature Extraction ─────────────────────────────────────────────
def compute_engineered_features(logprobs_k: torch.Tensor) -> torch.Tensor:
    """
    Compute 5 token-level features from top-k log-probs (Sec 3.2).

    Args:
        logprobs_k: (T, K)  — top-k log-probs; column 0 = selected token.
    Returns:
        (T, 5)  — [AvgLogP, RankProxy, Hoverall, Halts, ΔHdec]
    """
    EPS = 1e-9
    T, K = logprobs_k.shape

    # Proximal renormalised distribution over top-k
    mt = logprobs_k.max(dim=1, keepdim=True).values
    exp_s = torch.exp(logprobs_k - mt)
    p_tilde = exp_s / (exp_s.sum(dim=1, keepdim=True) + EPS)           # (T, K)

    # 1. AvgLogP
    avg_logp = logprobs_k.mean(dim=1)                                   # (T,)

    # 2. RankProxy — rank of selected token within top-k window
    sel = logprobs_k[:, 0:1]                                            # (T, 1)
    rank_proxy = 1.0 + (logprobs_k[:, 1:] > sel).float().sum(dim=1)    # (T,)

    # 3. Hoverall — entropy over full top-k distribution
    h_overall = -(p_tilde * (p_tilde + EPS).log()).sum(dim=1)           # (T,)

    # 4. Halts — entropy over alternatives only
    p_alts = p_tilde[:, 1:]
    p_alts_n = p_alts / (p_alts.sum(dim=1, keepdim=True) + EPS)
    h_alts = -(p_alts_n * (p_alts_n + EPS).log()).sum(dim=1)           # (T,)

    # 5. ΔHdec — temporal delta of binary decision entropy
    best_alt = logprobs_k[:, 1:].max(dim=1).values
    sel_lp = logprobs_k[:, 0]
    pc = torch.exp(sel_lp) / (torch.exp(sel_lp) + torch.exp(best_alt) + EPS)
    pc = pc.clamp(EPS, 1.0 - EPS)
    h_dec = -(pc * pc.log() + (1 - pc) * (1 - pc).log())
    delta_h_dec = h_dec - torch.cat([h_dec[:1], h_dec[:-1]])            # (T,)

    return torch.stack([avg_logp, rank_proxy, h_overall, h_alts, delta_h_dec], dim=1)


def build_halt_input(logprobs_k: torch.Tensor) -> torch.Tensor:
    """
    Concatenate engineered stats with raw log-probs → (T, 25).
    Order: [stats (5) || raw_logprobs (20)]  (Fig. 1 / Sec 3.2)
    """
    stats = compute_engineered_features(logprobs_k)        # (T, 5)
    return torch.cat([stats, logprobs_k], dim=1)            # (T, 25)


# ── Synthetic Dataset ─────────────────────────────────────────────────────────
def _synthetic_logprobs(
    seq_len: int,
    top_k: int,
    hallucinated: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Synthetic top-k log-prob matrix mimicking real LLM output.
    Hallucinated: flatter, noisier. Correct: peaked, stable.
    Returns: (seq_len, top_k) float32, sorted descending per row.
    """
    result = np.zeros((seq_len, top_k), dtype=np.float32)
    for t in range(seq_len):
        if hallucinated:
            base = rng.uniform(-4.0, -0.3)
            gap  = rng.uniform(0.05, 0.3)
        else:
            base = rng.uniform(-1.5, -0.1)
            gap  = rng.uniform(0.5, 2.0)
        lp_alts = base - gap * np.arange(1, top_k) - rng.uniform(0, 0.5, top_k - 1)
        result[t] = np.concatenate([[base], lp_alts]).astype(np.float32)
    return result


class SyntheticLogProbDataset(Dataset):
    """
    Synthetic dataset of (feature_sequence, label) pairs.
    Each sample = one LLM response as a top-k log-prob time series.
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
            x       = build_halt_input(torch.from_numpy(raw))   # (T, 25)
            self.samples.append((x, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        x, label = self.samples[idx]
        return x, label, x.shape[0]   # tensor, label, length


def collate_fn(batch):
    """
    Pad variable-length sequences; sort by length descending for pack_padded_sequence.
    Returns: padded_x (B, T, D), labels (B,), lengths (B,)
    """
    xs, labels, lengths = zip(*batch)
    lengths_t   = torch.tensor(lengths, dtype=torch.long)
    idx         = lengths_t.argsort(descending=True)
    xs_sorted   = [xs[i] for i in idx]
    lens_sorted = lengths_t[idx]
    lbls_sorted = torch.tensor([labels[i] for i in idx], dtype=torch.float)
    padded      = pad_sequence(xs_sorted, batch_first=True, padding_value=0.0)  # (B,T,D)
    return padded, lbls_sorted, lens_sorted


# ── Model Blocks ──────────────────────────────────────────────────────────────
class InputProjection(nn.Module):
    """
    LayerNorm → Linear → GELU → Linear
    (B, T, 25) → (B, T, proj_dim)
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
    Average the top-q fraction of GRU timesteps ranked by ℓ₂ norm.
    (B, T, D), lengths (B,) → (B, D)
    """
    def __init__(self, q: float = 0.15):
        super().__init__()
        self.q = q

    def forward(self, H: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, D = H.shape
        mask   = torch.arange(T, device=H.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = H.norm(dim=2).masked_fill(~mask, -1e9)            # (B, T)
        k_vals = (lengths.float() * self.q).ceil().long().clamp(min=1)
        pooled = torch.zeros(B, D, device=H.device)
        for b in range(B):
            top_idx   = scores[b].topk(k_vals[b].item()).indices
            pooled[b] = H[b, top_idx].mean(dim=0)
        return pooled


# ── HALT Model ────────────────────────────────────────────────────────────────
class HALT(nn.Module):
    """
    HALT: Hallucination Assessment via Log-probs as Time series (Sec 3.2 + App B).

    Input:
        x:       (B, T, 25)   padded feature sequences
        lengths: (B,)         original sequence lengths
    Output:
        logits:  (B,)         raw hallucination scores (apply sigmoid for probability)
    """
    def __init__(self, config: HALTConfig):
        super().__init__()
        self.cfg        = config
        self.projection = InputProjection(config.input_dim, config.proj_dim)
        self.gru        = nn.GRU(
            input_size  = config.proj_dim,
            hidden_size = config.hidden_dim,
            num_layers  = config.num_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = config.dropout if config.num_layers > 1 else 0.0,
        )
        gru_out_dim     = config.hidden_dim * 2          # 512
        self.pool       = TopQPooling(q=config.top_q)
        self.classifier = nn.Linear(gru_out_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:  nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias" in name:     nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        proj    = self.projection(x)                              # (B, T, 128)
        packed  = pack_padded_sequence(proj, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out_pk, _ = self.gru(packed)
        H, _    = pad_packed_sequence(out_pk, batch_first=True, total_length=T)  # (B,T,512)
        pooled  = self.pool(H, lengths)                           # (B, 512)
        return self.classifier(pooled).squeeze(1)                 # (B,)


# ── Training & Evaluation ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, max_grad_norm, device):
    model.train()
    total, n = 0.0, 0
    for x_pad, labels, lengths in loader:
        x_pad, labels, lengths = x_pad.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_pad, lengths), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    all_logits, all_labels, total, n = [], [], 0.0, 0
    with torch.no_grad():
        for x_pad, labels, lengths in loader:
            x_pad, labels, lengths = x_pad.to(device), labels.to(device), lengths.to(device)
            logits = model(x_pad, lengths)
            total += criterion(logits, labels).item(); n += 1
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().int().tolist())

    probs    = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds    = (probs >= threshold).astype(int)
    labels_np = np.array(all_labels)

    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    accuracy  = (preds == labels_np).mean()
    try:    auroc = roc_auc_score(labels_np, probs)
    except ValueError: auroc = 0.5

    return {"loss": total / max(n, 1), "macro_f1": macro_f1, "auroc": auroc, "accuracy": accuracy}


def train_halt(model, train_loader, val_loader, optimizer, scheduler, criterion, cfg, device):
    best_f1, best_state, patience_ctr = -1.0, None, 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg.max_grad_norm, device)
        val        = evaluate(model, val_loader, criterion, device)
        scheduler.step(val["macro_f1"])

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{cfg.max_epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val['loss']:.4f}"
                f"  val_macro_f1={val['macro_f1']:.4f}"
                f"  val_auroc={val['auroc']:.4f}"
                f"  acc={val['accuracy']:.4f}"
            )

        if val["macro_f1"] > best_f1:
            best_f1      = val["macro_f1"]
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best val macro-F1: {best_f1:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_f1


# ── Inference Helper ──────────────────────────────────────────────────────────
def predict_hallucination(logprobs_matrix: np.ndarray, model: nn.Module, device: str,
                          threshold: float = 0.5) -> dict:
    """
    Predict hallucination for a single LLM response.

    Args:
        logprobs_matrix: (T, top_k) — top-k log-probs per token.
                         Column 0 = selected token, rest = alternatives sorted desc.
    Returns:
        dict with prob_hallucinated, prediction, logit.
    """
    model.eval()
    x       = build_halt_input(torch.from_numpy(logprobs_matrix.astype(np.float32)))
    x       = x.unsqueeze(0).to(device)                                   # (1, T, 25)
    lengths = torch.tensor([logprobs_matrix.shape[0]], dtype=torch.long).to(device)
    with torch.no_grad():
        logit = model(x, lengths).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    return {"prob_hallucinated": prob, "prediction": int(prob >= threshold), "logit": logit}


# ── Feature Attribution (Gradient × Input, Appendix C.2) ─────────────────────
FEATURE_NAMES = (
    ["avg_logp", "rank_proxy", "h_overall", "h_alts", "delta_h_dec"]
    + [f"logprob_{i+1}" for i in range(20)]
)

def compute_feature_importance(model, loader, device, n_features=25):
    model.eval()
    feat_imp = torch.zeros(n_features, device=device)
    for x_pad, labels, lengths in loader:
        x_pad   = x_pad.to(device).requires_grad_(True)
        labels  = labels.to(device)
        lengths = lengths.to(device)
        logits  = model(x_pad, lengths)
        nn.BCEWithLogitsLoss()(logits, labels).backward()
        grads  = x_pad.grad                                           # (B, T, D)
        B, T, D = grads.shape
        mask   = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(2)
        feat_imp += ((grads * x_pad).abs() * mask).sum(dim=(0, 1))
        x_pad.grad = None
    feat_imp /= feat_imp.sum() + 1e-9
    return dict(sorted(zip(FEATURE_NAMES, feat_imp.tolist()), key=lambda kv: kv[1], reverse=True))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = HALTConfig()
    print(f"Device: {device}")

    # ── Dataset
    full_ds = SyntheticLogProbDataset(n_samples=4000, top_k=cfg.top_k, seed=SEED)
    n_total = len(full_ds)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    loader_kw = dict(collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=False, **loader_kw)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    # ── Model
    model = HALT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HALT parameters: {n_params:,}  (~{n_params/1e6:.1f}M)")

    # ── Optimiser & scheduler
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience, verbose=False,
    )

    # ── Train
    print("\n" + "=" * 60)
    print("Training HALT")
    print("=" * 60)
    best_val_f1 = train_halt(
        model, train_loader, val_loader, optimizer, scheduler, criterion, cfg, device
    )
    print(f"\nBest validation macro-F1: {best_val_f1:.4f}")

    # ── Test
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    test_m = evaluate(model, test_loader, criterion, device)
    print(f"Test macro-F1 : {test_m['macro_f1']:.4f}")
    print(f"Test AUROC    : {test_m['auroc']:.4f}")
    print(f"Test Accuracy : {test_m['accuracy']:.4f}")
    print(f"Test Loss     : {test_m['loss']:.4f}")

    # ── Example inference
    print("\n" + "=" * 60)
    print("Example Inference")
    print("=" * 60)
    rng_demo = np.random.default_rng(123)
    for label_str, is_hall in [("Hallucinated", True), ("Correct", False)]:
        demo = _synthetic_logprobs(60, cfg.top_k, is_hall, rng_demo)
        res  = predict_hallucination(demo, model, device)
        print(
            f"[{label_str:>13}]  P(hallucinated)={res['prob_hallucinated']:.4f}"
            f"  → {'HALLUCINATED' if res['prediction'] else 'NOT HALLUCINATED'}"
        )

    # ── Feature importance
    print("\n" + "=" * 60)
    print("Feature Importance (Gradient × Input, top-10)")
    print("=" * 60)
    importance = compute_feature_importance(model, val_loader, device, n_features=cfg.input_dim)
    print(f"{'Feature':<22}  {'Importance':>10}")
    print("-" * 36)
    for name, score in list(importance.items())[:10]:
        print(f"{name:<22}  {score:>10.4f}")

    # ── Shape sanity check
    print("\n" + "=" * 60)
    print("Shape Sanity Check")
    print("=" * 60)
    B_t, T_t = 4, 80
    x_t   = torch.randn(B_t, T_t, cfg.input_dim).to(device)
    lens_t = torch.tensor([80, 65, 40, 20], dtype=torch.long).to(device)
    lbls_t = torch.zeros(B_t).to(device)
    out   = model(x_t, lens_t)
    loss_t = nn.BCEWithLogitsLoss()(out, lbls_t)
    loss_t.backward()
    print(f"  Input:   {list(x_t.shape)}")
    print(f"  Output:  {list(out.shape)}")
    print(f"  Loss:    {loss_t.item():.4f}")
    print(f"  Grads OK: {all(p.grad is not None for p in model.parameters() if p.requires_grad)}")
    print("Shape check PASSED ✓")


if __name__ == "__main__":
    main()