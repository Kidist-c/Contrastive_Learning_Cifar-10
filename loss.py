"""
loss.py
-------
InfoNCE loss using in-batch negatives 

Given a batch of B images, we produce 2B embeddings (view1, view2 for each
image). For every embedding, its positive is the OTHER view of the SAME
image; every other one of the 2B-2 remaining embeddings acts as a negative
(see paper sections 8-9).

    L = -log [ exp(sim(z, z+)/tau) / sum_k exp(sim(z, z_k)/tau) ]
"""

import torch
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5):
    """
    z1, z2: [B, D] projected (or encoder) embeddings for view 1 / view 2 of each sample.
    Returns: scalar loss, plus some diagnostics useful for analysis.
    """
    batch_size = z1.shape[0]
    device = z1.device

    # 1) Normalize embeddings -> cosine similarity becomes a simple dot product
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # 2) Stack all 2B embeddings together: [2B, D]
    z = torch.cat([z1, z2], dim=0)

    # 3) Full pairwise similarity matrix [2B, 2B]
    sim_matrix = torch.matmul(z, z.T) / temperature

    # 4) Mask out self-similarity (diagonal) -- a sample is never its own negative/positive
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(self_mask, float("-inf"))

    # 5) Build positive-pair index: for i in [0, B) the positive is i+B, and vice versa
    positive_idx = torch.arange(2 * batch_size, device=device)
    positive_idx = (positive_idx + batch_size) % (2 * batch_size)

    # 6) Cross-entropy where the "correct class" is the positive's index among all 2B-1 candidates
    loss = F.cross_entropy(sim_matrix, positive_idx)

    # ---- Diagnostics (useful for Experiment 1 / geometry analysis) ----
    with torch.no_grad():
        pos_sim = sim_matrix[torch.arange(2 * batch_size, device=device), positive_idx] * temperature
        # average negative similarity: mask out -inf (self) and the positive entry, average the rest
        neg_mask = ~self_mask
        neg_mask[torch.arange(2 * batch_size, device=device), positive_idx] = False
        neg_sim = (sim_matrix * temperature).masked_select(neg_mask)
        diagnostics = {
            "mean_positive_similarity": pos_sim.mean().item(),
            "mean_negative_similarity": neg_sim.mean().item(),
            "similarity_gap": (pos_sim.mean() - neg_sim.mean()).item(),
        }

    return loss, diagnostics