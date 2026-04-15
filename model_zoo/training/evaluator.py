"""Model evaluation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> dict:
    """Evaluate a model on a DataLoader.

    Returns:
        dict with keys: loss, accuracy, per_class_accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    use_autocast = device.type == "cuda"

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            target_counts = torch.bincount(targets, minlength=num_classes)
            correct_counts = torch.bincount(
                targets[predicted.eq(targets)],
                minlength=num_classes,
            )
            for idx in range(num_classes):
                class_total[idx] += int(target_counts[idx].item())
                class_correct[idx] += int(correct_counts[idx].item())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    per_class_accuracy = [
        class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        for c in range(num_classes)
    ]

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
    }
