# visualize.py — Enhanced visualization utilities for HALT (compatible with corrected compute_feature_importance)

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale


plt.switch_backend('agg')  # Non-interactive backend


def plot_feature_importance(
    importance_dict: dict,
    output_path: str = "feature_importance.png",
    top_k: int = 10,
    title: str = "Feature Importance (Gradient × Input)",
):
    """Bar plot of feature importance."""
    features = list(importance_dict.keys())
    scores   = list(importance_dict.values())

    # Sort descending
    indices = np.argsort(scores)[::-1][:top_k]
    features_top = [features[i] for i in indices]
    scores_top   = [scores[i] for i in indices]

    # Normalize
    scores_top = minmax_scale(scores_top, feature_range=(0.2, 1.0))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(features_top[::-1], scores_top[::-1],
                   color="#2C7BB6", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Normalized Importance Score")
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis='y', labelsize=10)

    # Annotate
    for i, v in enumerate(scores_top[::-1]):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved feature importance plot → {output_path}")
    plt.close()


def plot_feature_correlation(
    model,
    loader,
    device,
    output_path: str = "feature_correlation.png"
):
    """
    Correlation heatmap between engineered features and model logits.
    Uses gradient-safe input path via compute_feature_importance pattern.
    """
    # Temporarily enable grad & eval to get logits
    model.train()
    X, Y = [], []

    with torch.enable_grad():
        for x_pad, labels, lengths in loader:
            # Wrap input in Parameter (like compute_feature_importance)
            x_pad = torch.nn.Parameter(x_pad.to(device), requires_grad=True)
            logits = model(x_pad, lengths)

            # Use only first 5 engineered features (avg over time)
            feat_means = x_pad[:, :, :5].mean(dim=1).detach().cpu().numpy()
            X.append(feat_means)

            Y.extend(logits.detach().cpu().numpy())

    model.eval()

    X = np.vstack(X)
    Y = np.array(Y)

    corr_matrix = []
    for i in range(5):
        feat_i = X[:, i]
        corr = np.corrcoef(feat_i, Y)[0, 1]
        corr_matrix.append(corr)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    im = ax.imshow([corr_matrix], cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_yticks([0])
    ax.set_yticklabels(["Logits"], fontsize=10)
    ax.set_xticks(range(5))
    ax.set_xticklabels([
        "AvgLogP", "RankProxy",
        "Hoverall", "Halts", "ΔHdec"
    ], fontsize=10)
    ax.set_title("Feature ↔ Hallucination Logits Correlation", pad=8)

    cbar = plt.colorbar(im, shrink=0.75)
    cbar.set_label("Pearson Correlation", fontsize=9)

    # Annotations
    for i in range(5):
        color = "black" if abs(corr_matrix[i]) < 0.7 else "white"
        ax.text(i, 0, f"{corr_matrix[i]:+.2f}", ha="center", va="center",
                color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved correlation heatmap → {output_path}")
    plt.close()


def extract_embeddings_for_t_sne(model, loader, device, n_samples=2000):
    """
    Extract embeddings — only if they are detachable.
    Uses same forward as compute_feature_importance to ensure consistency.
    """
    model.train()  # Ensure grad enabled
    all_embs, all_labels = [], []

    with torch.no_grad():  # Safe: embeddings don’t need gradients
        for x_pad, labels, lengths in loader:
            # Use same input as feature importance: wrap in Parameter but detach
            x_pad = torch.nn.Parameter(x_pad.to(device), requires_grad=False)  # no grad needed here
            logits = model(x_pad, lengths)

            # Extract pooled embeddings via same pipeline:
            proj = model.projection(x_pad)
            lengths_cpu = lengths.cpu()
            idx_sorted  = torch.sort(lengths_cpu, descending=True).indices
            proj_sorted = proj[idx_sorted]
            lengths_sorted = lengths_cpu[idx_sorted]

            packed  = torch.nn.utils.rnn.pack_padded_sequence(
                proj_sorted, lengths_sorted, batch_first=True, enforce_sorted=True
            )
            out_pk, _ = model.gru(packed)
            H, _    = torch.nn.utils.rnn.pad_packed_sequence(out_pk, batch_first=True)
            inv_idx = idx_sorted.argsort()
            H_orig  = H[inv_idx]

            pooled = model.pooler(H_orig, lengths)
            all_embs.append(pooled.detach().cpu().numpy())
            # Use true labels from dataset (not prediction)
            all_labels.extend(labels.int().cpu().tolist())

            if len(all_embs) * loader.batch_size >= n_samples:
                break

    model.eval()
    embeddings = np.vstack(all_embs)
    labels_arr   = np.array(all_labels[:len(embeddings)])
    return embeddings, labels_arr


def plot_t_sne_embeddings(
    model,
    loader,
    device,
    output_path: str = "tsne_embeddings.png",
    n_samples: int = 1500,
    perplexity: float = 30.0
):
    """
    T-SNE visualization of GRU embeddings, colored by **true class** (hallucinated vs correct).
    Uses same data loading and embedding extraction logic as feature importance.
    """
    print(f"🧠 Extracting {n_samples} embeddings for T-SNE...")
    embeddings, labels = extract_embeddings_for_t_sne(model, loader, device, n_samples=n_samples)

    # Subsample if needed (T-SNE O(N^2))
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), size=n_samples, replace=False)
        embeddings = embeddings[idx]
        labels     = labels[idx]

    # T-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=500, metric="euclidean")
    embs_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 5))

    class_names = ["Hallucinated", "Correct"]
    colors      = ["#D7191C", "#2B83BA"]

    for cls in [0, 1]:
        mask = labels == cls
        ax.scatter(
            embs_2d[mask, 0], embs_2d[mask, 1],
            c=colors[cls], label=class_names[cls],
            alpha=0.6, s=25, edgecolor="none"
        )

    ax.set_xlabel("T-SNE Component 1", fontsize=10)
    ax.set_ylabel("T-SNE Component 2", fontsize=10)
    ax.set_title("GRU Embedding Space (T-SNE Projection)", fontsize=12, pad=10)
    ax.legend(loc="best", fontsize=9)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved T-SNE plot → {output_path}")
    plt.close()


def plot_sensitivity_analysis(
    model,
    loader,
    device,
    n_samples: int = 50,
    output_path: str = "sensitivity_analysis.png"
):
    """SHAP-like perturbation sensitivity on top-5 engineered features."""
    model.train()

    results = {f: [] for f in range(25)}

    with torch.enable_grad():
        for x_pad, _, lengths in loader:
            if len(results[0]) >= n_samples * 5:
                break

            # Detach & clone to avoid interfering with training
            x_clean = torch.nn.Parameter(x_pad.to(device).detach().clone(), requires_grad=False)

            # Compute baseline logits (without perturbation)
            with torch.no_grad():
                base_logits = model(x_pad.to(device), lengths.to(device))

            for feat_idx in range(5):
                # Create perturbed copy
                x_pert = x_pad.clone().detach()
                mean_feat = x_pert[:, :, feat_idx].mean(dim=1, keepdim=True)
                std_feat  = x_pert[:, :, feat_idx].std(dim=1, keepdim=True) + 1e-8
                noise = torch.randn_like(x_pert[:, :, feat_idx]) * std_feat * 0.5
                x_pert[:, :, feat_idx] += noise

                # Perturbed logits (no grad needed for sensitivity)
                with torch.no_grad():
                    pert_logits = model(x_pert.to(device), lengths.to(device))

                delta_logits = torch.abs(pert_logits - base_logits).cpu().numpy()
                results[feat_idx].extend(delta_logits.tolist())

    model.eval()

    mean_delta = [np.mean(results[i]) for i in range(5)]
    std_delta  = [np.std(results[i]) for i in range(5)]

    features_names = ["AvgLogP", "RankProxy", "Hoverall", "Halts", "ΔHdec"]
    fig, ax = plt.subplots(figsize=(8, 4))

    x_pos = np.arange(len(features_names))
    ax.bar(x_pos, mean_delta, yerr=std_delta,
           color="#1B9E77", capsize=5, alpha=0.8, edgecolor="black")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(features_names, fontsize=10)
    ax.set_ylabel("Logit Δ (Perturbation Sensitivity)", fontsize=10)
    ax.set_title("Sensitivity Analysis: Engineered Features", pad=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved sensitivity plot → {output_path}")
    plt.close()


def plot_training_curves(
    history: dict,
    output_path: str = "training_curves.png"
):
    """
    Multi-panel plot of training curves.
    
    Args:
        history: dict with keys ["train_loss", "val_loss", "macro_f1", "auroc"]
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Train loss
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#D7191C")
    if "val_loss" in history and len(history["val_loss"]):
        axes[0].plot(epochs, history["val_loss"], label="Val", color="#FDAE61")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Loss Curves")

    # Metrics
    if "macro_f1" in history or "auroc" in history:
        color_map = {"macro_f1": "#1A9641", "auroc": "#3182BD"}
        for key in ["macro_f1", "auroc"]:
            if key in history and len(history[key]):
                m_epochs = range(1, len(history[key]) + 1)
                axes[1].plot(m_epochs, history[key], label=key.replace("_", " ").title(),
                            color=color_map[key], linewidth=2)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_title("Validation Metrics")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved training curves → {output_path}")
    plt.close()

    