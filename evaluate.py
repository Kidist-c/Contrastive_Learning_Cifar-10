"""
evaluate.py
-----------
Representation-quality evaluation (paper section 10):

    - extract_embeddings         : run the encoder (h, NOT the projection head z)
                                    over a dataset and collect (embeddings, labels)
    - knn_accuracy                : k-NN classification using embeddings as features
    - similarity_gap               : S_same - S_different
    - retrieval_recall_at_k        : nearest-neighbor retrieval quality
    - collapse_diagnostics         : variance-based collapse detectors
    - plot_embeddings_2d           : PCA / t-SNE projection for visualization
"""

import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def extract_embeddings(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu",
                        normalize=True):
    """
    Runs the ENCODER (h), not the projection head, over a dataloader of
    (image, label) pairs and returns (embeddings [N, D] numpy, labels [N] numpy).

    Using h (not z) here matters: after training, h is the representation you
    actually deploy downstream (see paper section 6).
    """
    model.eval()
    all_h, all_labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        h, _ = model(x)
        if normalize:
            h = F.normalize(h, dim=1)
        all_h.append(h.cpu())
        all_labels.append(y)
    embeddings = torch.cat(all_h, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return embeddings, labels


def knn_accuracy(train_embeddings, train_labels, test_embeddings, test_labels, k=10):
    """
    Simple k-NN classifier using cosine similarity (embeddings assumed normalized).
    Implemented directly with numpy/torch to avoid extra dependencies.
    """
    train_t = torch.from_numpy(train_embeddings)
    test_t = torch.from_numpy(test_embeddings)
    train_labels_t = torch.from_numpy(train_labels)

    sims = test_t @ train_t.T  # [N_test, N_train] cosine similarity (already normalized)
    topk = sims.topk(k, dim=1).indices  # [N_test, k]
    neighbor_labels = train_labels_t[topk]  # [N_test, k]

    preds = []
    for row in neighbor_labels:
        values, counts = torch.unique(row, return_counts=True)
        preds.append(values[counts.argmax()].item())
    preds = np.array(preds)

    accuracy = (preds == test_labels).mean()
    return accuracy


def similarity_gap(embeddings, labels, max_samples_per_class=200, seed=0):
    """
    S_same  = average cosine similarity between pairs from the SAME class
    S_diff  = average cosine similarity between pairs from DIFFERENT classes
    gap     = S_same - S_diff   (bigger = better separated representation)
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    # Subsample per class for tractable pairwise computation
    idx_per_class = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) > max_samples_per_class:
            idx = rng.choice(idx, size=max_samples_per_class, replace=False)
        idx_per_class[c] = idx

    all_idx = np.concatenate(list(idx_per_class.values()))
    emb = torch.from_numpy(embeddings[all_idx])
    lab = labels[all_idx]
    sims = (emb @ emb.T).numpy()

    same_mask = (lab[:, None] == lab[None, :])
    np.fill_diagonal(same_mask, False)  # exclude self-similarity
    diff_mask = ~same_mask
    np.fill_diagonal(diff_mask, False)

    s_same = sims[same_mask].mean()
    s_diff = sims[diff_mask].mean()
    return {"S_same": float(s_same), "S_diff": float(s_diff), "gap": float(s_same - s_diff)}


def retrieval_recall_at_k(query_embeddings, query_labels, db_embeddings, db_labels, k=5):
    """
    For each query, retrieve top-k nearest database embeddings (cosine similarity)
    and check whether at least one of them shares the query's label.
    """
    query_t = torch.from_numpy(query_embeddings)
    db_t = torch.from_numpy(db_embeddings)
    sims = query_t @ db_t.T  # [N_query, N_db]
    topk = sims.topk(k, dim=1).indices.numpy()

    hits = 0
    for i, neighbors in enumerate(topk):
        retrieved_labels = db_labels[neighbors]
        if query_labels[i] in retrieved_labels:
            hits += 1
    return hits / len(query_labels)


def collapse_diagnostics(embeddings):
    """
    Detects representation collapse: if embeddings barely vary across samples,
    per-dimension variance and average pairwise similarity are the tell-tale signs.
    """
    emb = torch.from_numpy(embeddings)
    per_dim_var = emb.var(dim=0).mean().item()

    # average pairwise cosine similarity on a random subsample (for tractability)
    n = min(1000, emb.shape[0])
    idx = torch.randperm(emb.shape[0])[:n]
    sub = emb[idx]
    sims = sub @ sub.T
    mask = ~torch.eye(n, dtype=torch.bool)
    avg_pairwise_sim = sims[mask].mean().item()

    return {
        "mean_per_dim_variance": per_dim_var,
        "avg_pairwise_cosine_similarity": avg_pairwise_sim,
        "likely_collapsed": bool(per_dim_var < 1e-3 or avg_pairwise_sim > 0.95),
    }


def plot_embeddings_2d(embeddings, labels, method="tsne", title="Embedding space",
                        save_path=None, max_points=2000, seed=0):
    """
    Projects embeddings to 2D (PCA or t-SNE) and saves a scatter plot colored by label.
    Requires matplotlib and scikit-learn.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    if len(embeddings) > max_points:
        idx = rng.choice(len(embeddings), size=max_points, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    if method == "pca":
        proj = PCA(n_components=2).fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, init="pca", random_state=seed).fit_transform(embeddings)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Class", loc="best", fontsize=8)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()