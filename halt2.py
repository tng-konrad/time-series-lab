"""
HALT: Hallucination Assessment via Log-probs as Time Series
Single-file PyTorch implementation.

This script implements:

- Feature extraction from top-k log-probabilities
- Bidirectional GRU encoder
- Top-q pooling
- Binary hallucination classification

The script uses a synthetic dataset so it runs end-to-end.
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# Device
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================================================
# Config
# =========================================================

class Config:
    # data
    batch_size = 8
    max_seq_len = 64
    min_seq_len = 16

    # logprob settings
    top_k = 20
    stats_dim = 5
    input_dim = top_k + stats_dim

    # model
    proj_dim = 128
    hidden_dim = 128
    num_layers = 2
    bidirectional = True
    top_q = 0.2

    # training
    lr = 1e-3
    epochs = 3


cfg = Config()

# =========================================================
# Dataset
# =========================================================

class SyntheticLogProbDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        T = random.randint(cfg.min_seq_len, cfg.max_seq_len)

        # simulate realistic log-prob ranges
        log_probs = torch.randn(T, cfg.top_k) * 1.5 - 2.0

        label = torch.randint(0, 2, (1,)).float()
        return log_probs, label


def collate_fn(batch):
    logprob_list, label_list = zip(*batch)

    lengths = [x.shape[0] for x in logprob_list]
    max_len = max(lengths)

    padded = torch.zeros(len(batch), max_len, cfg.top_k)

    for i, x in enumerate(logprob_list):
        padded[i, :x.shape[0]] = x

    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(label_list).view(-1, 1)

    return padded, lengths, labels


# =========================================================
# Model
# =========================================================

class HALTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = cfg.input_dim

        self.proj = nn.Linear(self.input_dim, cfg.proj_dim)

        self.gru = nn.GRU(
            input_size=cfg.proj_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        gru_out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.classifier = nn.Linear(gru_out_dim, 1)

    # -----------------------------------------------------
    # Feature Extraction
    # -----------------------------------------------------

    def compute_features(self, log_probs: torch.Tensor):
        """
        log_probs: (B, T, K)
        return: (B, T, K+5)
        """
        B, T, K = log_probs.shape

        # stable truncated distribution
        max_lp, _ = log_probs.max(dim=-1, keepdim=True)
        probs = torch.exp(log_probs - max_lp)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 1. AvgLogP
        avg_logp = log_probs.mean(dim=-1, keepdim=True)

        # 2. RankProxy
        selected = log_probs[:, :, 0:1]
        alternatives = log_probs[:, :, 1:]
        rank_proxy = 1 + (alternatives > selected).float().sum(dim=-1, keepdim=True)

        # 3. Overall entropy
        hoverall = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)

        # 4. Alternatives-only entropy
        probs_alts = probs[:, :, 1:]
        probs_alts = probs_alts / (probs_alts.sum(dim=-1, keepdim=True) + 1e-8)
        halts = -(probs_alts * (probs_alts + 1e-8).log()).sum(dim=-1, keepdim=True)

        # 5. Delta decision entropy
        best_alt_idx = alternatives.argmax(dim=-1, keepdim=True)
        best_alt = torch.gather(alternatives, -1, best_alt_idx)

        pc_num = torch.exp(selected)
        pc_den = torch.exp(selected) + torch.exp(best_alt)
        pc = pc_num / (pc_den + 1e-8)

        hdec = -(pc * (pc + 1e-8).log() + (1 - pc) * (1 - pc + 1e-8).log())

        delta_hdec = torch.zeros_like(hdec)
        delta_hdec[:, 1:] = hdec[:, 1:] - hdec[:, :-1]

        stats = torch.cat(
            [avg_logp, rank_proxy, hoverall, halts, delta_hdec], dim=-1
        )

        enriched = torch.cat([stats, log_probs], dim=-1)
        return enriched

    # -----------------------------------------------------
    # Top-q Pooling
    # -----------------------------------------------------

    def top_q_pool(self, hidden: torch.Tensor, lengths: torch.Tensor):
        """
        hidden: (B, T, H)
        """
        B, T, H = hidden.shape
        pooled = []

        for b in range(B):
            L = lengths[b].item()
            h = hidden[b, :L]

            norms = torch.norm(h, dim=-1)
            k = max(1, int(cfg.top_q * L))
            top_idx = torch.topk(norms, k=k, dim=0).indices

            pooled_vec = h[top_idx].mean(dim=0)
            pooled.append(pooled_vec)

        pooled = torch.stack(pooled, dim=0)
        return pooled

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    def forward(self, log_probs: torch.Tensor, lengths: torch.Tensor):
        """
        log_probs: (B, T, 20)
        lengths: (B,)
        """
        x = self.compute_features(log_probs)
        x = self.proj(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.gru(packed)

        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        pooled = self.top_q_pool(hidden, lengths)

        logits = self.classifier(pooled)
        return logits


# =========================================================
# Training Utilities
# =========================================================

def train_step(model, optimizer, criterion, batch):
    model.train()

    log_probs, lengths, labels = batch
    log_probs = log_probs.to(device)
    lengths = lengths.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    logits = model(log_probs, lengths)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == labels).float().mean()

    return loss.item(), acc.item()


@torch.no_grad()
def eval_step(model, criterion, batch):
    model.eval()

    log_probs, lengths, labels = batch
    log_probs = log_probs.to(device)
    lengths = lengths.to(device)
    labels = labels.to(device)

    logits = model(log_probs, lengths)
    loss = criterion(logits, labels)

    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == labels).float().mean()

    return loss.item(), acc.item()


# =========================================================
# Main
# =========================================================

def main():

    train_dataset = SyntheticLogProbDataset(128)
    val_dataset = SyntheticLogProbDataset(32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = HALTModel().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Training Loop
    for epoch in range(cfg.epochs):

        train_losses = []
        train_accs = []

        for batch in train_loader:
            loss, acc = train_step(model, optimizer, criterion, batch)
            train_losses.append(loss)
            train_accs.append(acc)

        val_losses = []
        val_accs = []

        for batch in val_loader:
            loss, acc = eval_step(model, criterion, batch)
            val_losses.append(loss)
            val_accs.append(acc)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss {sum(train_losses)/len(train_losses):.4f} "
            f"Train Acc {sum(train_accs)/len(train_accs):.4f} | "
            f"Val Loss {sum(val_losses)/len(val_losses):.4f} "
            f"Val Acc {sum(val_accs)/len(val_accs):.4f}"
        )

    # Example Forward Pass
    example_batch = next(iter(train_loader))
    log_probs, lengths, _ = example_batch
    logits = model(log_probs.to(device), lengths.to(device))
    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    main()