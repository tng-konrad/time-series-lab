import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Feature Dimensions (Section 3.2)
    top_k = 20  
    d_stats = 5 
    input_dim = top_k + d_stats  # 25  
    
    # Model Architecture (Appendix B)
    proj_dim = 128  
    hidden_dim = 256 
    num_layers = 5  
    bidirectional = True  
    dropout = 0.4 
    q_fraction = 0.15  # Top-q pooling fraction 
    
    # Training Hyperparameters (Appendix B)
    batch_size = 512  
    lr = 4.41e-4  
    weight_decay = 2.34e-6  
    epochs = 100  

# ==========================================
# FEATURE EXTRACTION (Section 3.2)
# ==========================================
def compute_halt_features(raw_logprobs):
    """
    Extracts the 5 engineered uncertainty features and concatenates with raw log-probs.
    Input: (B, T, 20) tensor where index 0 is the selected token.
    Output: (B, T, 25) enriched feature vector.
    """
    B, T, K = raw_logprobs.shape
    
    # Probabilities restricted to top-k support (Eq. 4) 
    probs = torch.softmax(raw_logprobs, dim=-1)
    
    # 1. Average Log-probability (Eq. 5) 
    avg_logp = raw_logprobs.mean(dim=-1, keepdim=True) #  
    
    # 2. Rank Proxy (Eq. 6): 1 + count of alternatives scoring higher than selected 
    selected_logp = raw_logprobs[:, :, 0:1]
    alternatives_logp = raw_logprobs[:, :, 1:]
    rank_proxy = 1.0 + (alternatives_logp > selected_logp).sum(dim=-1, keepdim=True).float() #  
    
    # 3. Overall Entropy on truncated distribution (Eq. 7) 
    h_overall = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1, keepdim=True) 
    
    # 4. Alternatives-only Entropy (Eq. 8-9)  
    p_alts = torch.softmax(raw_logprobs[:, :, 1:], dim=-1)
    h_alts = -torch.sum(p_alts * torch.log(p_alts + 1e-9), dim=-1, keepdim=True)  
    
    # 5. Temporal Change in Binary Decision Entropy (Eq. 10-13)  
    best_alt_logp = torch.max(alternatives_logp, dim=-1, keepdim=True)[0]  
    combined = torch.cat([selected_logp, best_alt_logp], dim=-1)
    p_c_dist = torch.softmax(combined, dim=-1)
    p_c = p_c_dist[:, :, 0:1]  
    h_dec = -(p_c * torch.log(p_c + 1e-9) + (1-p_c) * torch.log(1-p_c + 1e-9))  
    
    # Temporal Delta (t - (t-1)) 
    h_dec_prev = torch.roll(h_dec, shifts=1, dims=1)
    h_dec_prev[:, 0, :] = h_dec[:, 0, :]
    delta_h_dec = h_dec - h_dec_prev #  
    
    # Concatenate [Raw Log-probs || Stats] 
    return torch.cat([raw_logprobs, avg_logp, rank_proxy, h_overall, h_alts, delta_h_dec], dim=-1)

# ==========================================
# ARCHITECTURE (Section 3.2 & Appendix B)
# ==========================================
class TopQPooling(nn.Module):
    """Averages the top-q fraction of salient timesteps (Eq. Section 3.2)."""  
    def __init__(self, q_fraction=0.15):
        super().__init__()
        self.q = q_fraction

    def forward(self, x, lengths):
        B, T, D = x.shape
        norms = torch.norm(x, p=2, dim=-1)  # L2 norm for saliency 
        
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
        norms = norms.masked_fill(~mask, -1e9)
        
        pooled_outputs = []
        for i in range(B):
            seq_len = lengths[i].item()
            num_top = max(1, int(math.ceil(self.q * seq_len)))  
            _, indices = torch.topk(norms[i, :seq_len], k=num_top)  
            pooled_outputs.append(x[i, indices].mean(dim=0))  
            
        return torch.stack(pooled_outputs)

class HALTModel(nn.Module):
    """The 5M-parameter Bidirectional GRU Hallucination Detector."""  
    def __init__(self, config):
        super().__init__()
        # Input Projection 
        self.input_ln = nn.LayerNorm(config.input_dim)  
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.proj_dim),
            nn.GELU(),
            nn.Linear(config.proj_dim, config.proj_dim),
            nn.GELU()
        )
        
        # Bidirectional GRU Encoder 
        self.gru = nn.GRU(
            input_size=config.proj_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout
        )
        
        self.pooling = TopQPooling(q_fraction=config.q_fraction)  
        self.classifier = nn.Linear(config.hidden_dim * 2, 1) 

    def forward(self, raw_logprobs, lengths):
        # 1. Feature Engineering  
        x = compute_halt_features(raw_logprobs)
        
        # 2. Input Processing  
        x = self.input_ln(x)
        x = self.projection(x)
        
        # 3. Temporal Encoding  
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)  
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  
        
        # 4. Pooling & Classification 
        pooled = self.pooling(out, lengths)
        logit = self.classifier(pooled)  
        return logit.squeeze(-1)

# ==========================================
# DATASET & MAIN
# ==========================================
class SyntheticLogProbDataset(Dataset):
    """Synthetic dataset generating sequences of top-20 log-probs."""
    def __init__(self, size=100):
        self.data = [torch.randn(np.random.randint(20, 100), 20).sort(descending=True)[0] for _ in range(size)]
        self.labels = torch.randint(0, 2, (size,)).float()

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.data[i], self.labels[i]

def collate_fn(batch):
    data, labels = zip(*batch)
    lengths = torch.tensor([d.shape[0] for d in data])
    padded_data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    return padded_data, torch.stack(labels), lengths

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HALTModel(Config).to(device)
    
    # Test Pass
    test_ds = SyntheticLogProbDataset(10)
    loader = DataLoader(test_ds, batch_size=2, collate_fn=collate_fn)
    
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels, batch_lens in loader:
            logits = model(batch_data.to(device), batch_lens.to(device))
            scores = torch.sigmoid(logits)  
            print(f"Hallucination Scores: {scores.cpu().numpy()}")