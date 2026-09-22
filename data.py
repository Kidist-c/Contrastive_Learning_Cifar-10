"""
data.py
-------
Dataset loading + augmentation pipeline for contrastive learning.

Key idea (see paper section 5):
    x -> t1(x), t2(x)  are two independently-augmented "views" of the SAME
    image. They form a positive pair because they originate from the same
    underlying sample.

We support MNIST and CIFAR-10, and two augmentation "strengths":
    - weak   : large crop, small jitter -> easy contrastive task
    - strong : cutout, grayscale, heavy jitter -> hard contrastive task
"""

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T


def build_transform(strength: str, dataset: str):
    """Returns a torchvision.transforms.Compose that maps PIL image -> augmented tensor."""
    is_mnist = dataset == "mnist"
    img_size = 28 if is_mnist else 32
    norm_mean = (0.1307,) if is_mnist else (0.4914, 0.4822, 0.4465)
    norm_std = (0.3081,) if is_mnist else (0.2470, 0.2435, 0.2616)

    if strength == "weak":
        pipeline = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5 if not is_mnist else 0.0),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std),
        ])
    elif strength == "strong":
        pipeline = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            T.RandomHorizontalFlip(p=0.5 if not is_mnist else 0.0),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std),
            T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")

    return pipeline


def build_eval_transform(dataset: str):
    """Deterministic transform (no augmentation) used for evaluation / k-NN / retrieval."""
    is_mnist = dataset == "mnist"
    img_size = 28 if is_mnist else 32
    norm_mean = (0.1307,) if is_mnist else (0.4914, 0.4822, 0.4465)
    norm_std = (0.3081,) if is_mnist else (0.2470, 0.2435, 0.2616)
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(norm_mean, norm_std),
    ])


class ContrastivePairDataset(Dataset):
    """
    Wraps a base image dataset. On each __getitem__, returns TWO independently
    augmented views of the SAME image (x1, x2) -> a positive pair -- plus the
    true label (label is NOT used for training, only kept for later evaluation).
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)  # independently sampled augmentation -> second view
        return x1, x2, label


def load_base_dataset(dataset: str, root: str = "./data", train: bool = True):
    """Loads the raw (PIL-returning) dataset, no transform applied yet."""
    if dataset == "mnist":
        return torchvision.datasets.MNIST(root=root, train=train, download=True)
    elif dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root=root, train=train, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_dataloaders(dataset="cifar10", strength="weak", batch_size=256, subset_size=None,
                     root="./data", num_workers=2):
    """
    Convenience builder.

    Returns:
        contrastive_train_loader: yields (x1, x2, label) batches for contrastive training
        eval_train_loader:        deterministic-transform loader over the (sub)training set,
                                   used as the retrieval/k-NN "database"
        eval_test_loader:         deterministic-transform loader over the test set,
                                   used as k-NN/retrieval "queries"
    """
    base_train = load_base_dataset(dataset, root=root, train=True)
    base_test = load_base_dataset(dataset, root=root, train=False)

    if subset_size is not None:
        idx = torch.randperm(len(base_train))[:subset_size]
        base_train = torch.utils.data.Subset(base_train, idx.tolist())

    train_transform = build_transform(strength, dataset)
    eval_transform = build_eval_transform(dataset)

    contrastive_train = ContrastivePairDataset(base_train, train_transform)

    # Wrap with eval transform too, for evaluation purposes (no augmentation)
    class EvalWrap(Dataset):
        def __init__(self, base, tfm):
            self.base = base
            self.tfm = tfm

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            return self.tfm(img), label

    eval_train = EvalWrap(base_train, eval_transform)
    eval_test = EvalWrap(base_test, eval_transform)

    contrastive_loader = torch.utils.data.DataLoader(
        contrastive_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    eval_train_loader = torch.utils.data.DataLoader(
        eval_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_test_loader = torch.utils.data.DataLoader(
        eval_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return contrastive_loader, eval_train_loader, eval_test_loader