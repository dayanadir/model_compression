"""Deterministic CIFAR-10 evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_zoo.training.evaluator import evaluate


@dataclass(frozen=True)
class EvalResult:
    loss: float
    accuracy: float
    per_class_accuracy: list[float]


def build_cifar10_test_loader(
    data_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
) -> DataLoader:
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf)
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_cifar10_accuracy(
    model: nn.Module,
    device: str,
    data_dir: str = "./data",
    batch_size: int = 512,
    num_workers: int = 0,
) -> EvalResult:
    loader = build_cifar10_test_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    result = evaluate(model=model, loader=loader, device=torch.device(device), num_classes=10)
    return EvalResult(
        loss=float(result["loss"]),
        accuracy=float(result["accuracy"]),
        per_class_accuracy=[float(x) for x in result["per_class_accuracy"]],
    )
