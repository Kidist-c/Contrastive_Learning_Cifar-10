"""

--------------
Runs the three required controlled experiments (paper's assignment section 4):

    Experiment 1: Temperature          (0.05 vs 0.5 vs 1.0, weak aug, with proj head)
    Experiment 2: Augmentation strength (weak vs strong, fixed temperature, with proj head)
    Experiment 3: Projection head       (with vs without, fixed temperature + weak aug)

Each run keeps every other factor fixed, trains a fresh model, and reports:
    - training loss curve
    - final positive/negative similarity + gap
    - k-NN accuracy
    - retrieval Recall@5
    - collapse diagnostics

Results are collected into a single dict you can dump to JSON / a table.
"""

import json
import copy
import torch

from data import get_dataloaders
from model import ContrastiveModel
from train import train_contrastive
from evaluate import (extract_embeddings, knn_accuracy, similarity_gap,
                       retrieval_recall_at_k, collapse_diagnostics, plot_embeddings_2d)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_single_config(dataset, strength, temperature, use_projection_head,
                       subset_size=5000, epochs=10, batch_size=256,
                       embedding_dim=128, projection_dim=64,
                       run_name="run", plot=True):
    """Trains one model under a given configuration and returns a results dict."""
    in_channels = 1 if dataset == "mnist" else 3

    contrastive_loader, eval_train_loader, eval_test_loader = get_dataloaders(
        dataset=dataset, strength=strength, batch_size=batch_size, subset_size=subset_size)

    model = ContrastiveModel(in_channels=in_channels, embedding_dim=embedding_dim,
                              projection_dim=projection_dim,
                              use_projection_head=use_projection_head)

    history = train_contrastive(model, contrastive_loader, epochs=epochs,
                                 temperature=temperature, device=DEVICE)

    # ---- Evaluation: extract encoder embeddings h (not projection z) ----
    train_emb, train_labels = extract_embeddings(model, eval_train_loader, device=DEVICE)
    test_emb, test_labels = extract_embeddings(model, eval_test_loader, device=DEVICE)

    knn_acc = knn_accuracy(train_emb, train_labels, test_emb, test_labels, k=10)
    gap_stats = similarity_gap(test_emb, test_labels)
    recall5 = retrieval_recall_at_k(test_emb, test_labels, train_emb, train_labels, k=5)
    collapse = collapse_diagnostics(test_emb)

    if plot:
        plot_embeddings_2d(test_emb, test_labels, method="pca",
                            title=f"{run_name} (PCA)")

    result = {
        "config": {
            "dataset": dataset, "strength": strength, "temperature": temperature,
            "use_projection_head": use_projection_head, "epochs": epochs,
            "subset_size": subset_size, "batch_size": batch_size,
        },
        "final_train_loss": history["epoch_loss"][-1],
        "final_train_gap": history["epoch_gap"][-1],
        "loss_curve": history["epoch_loss"],
        "knn_accuracy": knn_acc,
        "test_similarity_gap": gap_stats,
        "retrieval_recall@5": recall5,
        "collapse_diagnostics": collapse,
    }
    return result


def experiment_temperature(dataset="cifar10", subset_size=5000, epochs=10):
    """Experiment 1: vary temperature, everything else fixed (weak aug, with projection head)."""
    results = {}
    for tau in [0.05, 0.5, 1.0]:
        name = f"exp1_temp_{tau}"
        print(f"\n=== Experiment 1: temperature={tau} ===")
        results[name] = run_single_config(
            dataset=dataset, strength="weak", temperature=tau, use_projection_head=True,
            subset_size=subset_size, epochs=epochs, run_name=name)
    return results


def experiment_augmentation(dataset="cifar10", subset_size=5000, epochs=10, temperature=0.5):
    """Experiment 2: weak vs strong augmentation, fixed temperature, with projection head."""
    results = {}
    for strength in ["weak", "strong"]:
        name = f"exp2_aug_{strength}"
        print(f"\n=== Experiment 2: augmentation={strength} ===")
        results[name] = run_single_config(
            dataset=dataset, strength=strength, temperature=temperature, use_projection_head=True,
            subset_size=subset_size, epochs=epochs, run_name=name)
    return results


def experiment_projection_head(dataset="cifar10", subset_size=5000, epochs=10, temperature=0.5):
    """Experiment 3: with vs without projection head, fixed temperature + weak augmentation."""
    results = {}
    for use_head in [True, False]:
        name = f"exp3_projhead_{use_head}"
        print(f"\n=== Experiment 3: use_projection_head={use_head} ===")
        results[name] = run_single_config(
            dataset=dataset, strength="weak", temperature=temperature, use_projection_head=use_head,
            subset_size=subset_size, epochs=epochs, run_name=name)
    return results


def run_all_experiments(dataset="cifar10", subset_size=5000, epochs=10):
    """Runs all three controlled experiments and saves a combined JSON report."""
    all_results = {}
    all_results.update(experiment_temperature(dataset, subset_size, epochs))
    all_results.update(experiment_augmentation(dataset, subset_size, epochs))
    all_results.update(experiment_projection_head(dataset, subset_size, epochs))

   

    print("\n\n==== SUMMARY ====")
    for name, r in all_results.items():
        print(f"{name:25s} | loss={r['final_train_loss']:.3f} | "
              f"knn_acc={r['knn_accuracy']:.3f} | gap={r['test_similarity_gap']['gap']:.3f} | "
              f"recall@5={r['retrieval_recall@5']:.3f} | "
              f"collapsed={r['collapse_diagnostics']['likely_collapsed']}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()