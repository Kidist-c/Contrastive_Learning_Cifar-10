"""
train.py
--------
The contrastive training loop.

    Dataset -> Augmentation -> Two Views -> Encoder -> Projection Head
    -> Normalized Embeddings -> InfoNCE Loss -> Learned Representation
"""

import torch
from tqdm import tqdm

from loss import info_nce_loss


def train_contrastive(model, dataloader, epochs=10, lr=3e-4, temperature=0.5,
                       device="cuda" if torch.cuda.is_available() else "cpu",
                       log_every=1):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch_loss": [], "epoch_pos_sim": [], "epoch_neg_sim": [], "epoch_gap": []}

    for epoch in range(epochs):
        model.train()
        running_loss, running_pos, running_neg, running_gap, n_batches = 0.0, 0.0, 0.0, 0.0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for x1, x2, _labels in pbar:
            x1, x2 = x1.to(device), x2.to(device)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss, diag = info_nce_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_pos += diag["mean_positive_similarity"]
            running_neg += diag["mean_negative_similarity"]
            running_gap += diag["similarity_gap"]
            n_batches += 1

            pbar.set_postfix(loss=loss.item(), gap=diag["similarity_gap"])

        history["epoch_loss"].append(running_loss / n_batches)
        history["epoch_pos_sim"].append(running_pos / n_batches)
        history["epoch_neg_sim"].append(running_neg / n_batches)
        history["epoch_gap"].append(running_gap / n_batches)

        if (epoch + 1) % log_every == 0:
            print(f"[Epoch {epoch + 1}] loss={history['epoch_loss'][-1]:.4f} "
                  f"pos_sim={history['epoch_pos_sim'][-1]:.4f} "
                  f"neg_sim={history['epoch_neg_sim'][-1]:.4f} "
                  f"gap={history['epoch_gap'][-1]:.4f}")

    return history