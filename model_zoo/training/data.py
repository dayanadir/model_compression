"""DataLoader factory for image classification datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

if TYPE_CHECKING:
    from model_zoo.config import DatasetInfo


def _get_cifar10_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform) for CIFAR-10."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def _get_mnist_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform) for MNIST / FashionMNIST."""
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


_TRANSFORM_BUILDERS = {
    "cifar10": _get_cifar10_transforms,
    "cifar100": _get_cifar10_transforms,  # same normalization
    "mnist": _get_mnist_transforms,
    "fashion_mnist": _get_mnist_transforms,
}

_DATASET_CLASSES = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}


def build_dataloaders(
    dataset_info: DatasetInfo,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders (downloaded once, shared across all models).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if dataset_info.name not in _DATASET_CLASSES:
        raise ValueError(
            f"Unsupported dataset: {dataset_info.name}. "
            f"Supported: {list(_DATASET_CLASSES)}"
        )

    dataset_cls = _DATASET_CLASSES[dataset_info.name]
    train_tf, eval_tf = _TRANSFORM_BUILDERS[dataset_info.name](dataset_info.image_size)

    # Download once
    full_train = dataset_cls(
        root=dataset_info.data_dir, train=True, download=True, transform=train_tf
    )
    test_set = dataset_cls(
        root=dataset_info.data_dir, train=False, download=True, transform=eval_tf
    )

    # Also build an eval-transform version of training data for validation
    full_train_eval = dataset_cls(
        root=dataset_info.data_dir, train=True, download=False, transform=eval_tf
    )

    # Split training into train / val
    n_total = len(full_train)
    n_val = int(n_total * dataset_info.val_fraction)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )
    # Convert to plain lists of indices
    train_idx = list(train_indices)
    val_idx = list(val_indices)

    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train_eval, val_idx)

    common_loader_kwargs = {
        "batch_size": dataset_info.batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True
        common_loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        drop_last=False,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **common_loader_kwargs,
    )

    return train_loader, val_loader, test_loader
