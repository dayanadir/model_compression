"""Single-model training loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_zoo.families.base import TrainingHyperparams
from model_zoo.training.evaluator import evaluate

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Outcome of a single model training run."""

    final_train_loss: float = 0.0
    final_train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    val_per_class_accuracy: list[float] = field(default_factory=list)
    test_loss: float = 0.0
    test_acc: float = 0.0
    test_per_class_accuracy: list[float] = field(default_factory=list)
    epoch_metrics: list[dict] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    completed: bool = False
    error: str | None = None


def _build_optimizer(
    model: nn.Module, hparams: TrainingHyperparams
) -> torch.optim.Optimizer:
    """Create an optimizer from training hyperparameters."""
    name = hparams.optimizer.lower()
    kwargs = dict(lr=hparams.lr, weight_decay=hparams.weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


class Trainer:
    """Trains a single model on a classification dataset."""

    def __init__(
        self,
        device: torch.device,
        num_classes: int = 10,
        eval_every: int = 5,
    ) -> None:
        self.device = device
        self.num_classes = num_classes
        self.eval_every = eval_every
        self.use_autocast = device.type == "cuda"
        if self.use_autocast:
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        hparams: TrainingHyperparams,
        seed: int,
    ) -> TrainResult:
        """Run the full training loop and return results.

        Sets torch seeds for reproducibility, trains for ``hparams.epochs``
        epochs, evaluates periodically, and returns a ``TrainResult``.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = model.to(self.device)
        optimizer = _build_optimizer(model, hparams)
        criterion = nn.CrossEntropyLoss(label_smoothing=hparams.label_smoothing)

        result = TrainResult()
        start_time = time.time()

        try:
            for epoch in range(1, hparams.epochs + 1):
                # --- Train one epoch ---
                model.train()
                running_loss = 0.0
                running_correct = 0
                running_total = 0

                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_autocast,
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    running_total += targets.size(0)
                    running_correct += predicted.eq(targets).sum().item()

                epoch_loss = running_loss / running_total
                epoch_acc = running_correct / running_total

                epoch_record: dict = {
                    "epoch": epoch,
                    "train_loss": round(epoch_loss, 6),
                    "train_acc": round(epoch_acc, 6),
                }

                # --- Periodic validation ---
                if epoch % self.eval_every == 0 or epoch == hparams.epochs:
                    val_metrics = evaluate(
                        model, val_loader, self.device, self.num_classes
                    )
                    epoch_record["val_loss"] = round(val_metrics["loss"], 6)
                    epoch_record["val_acc"] = round(val_metrics["accuracy"], 6)

                result.epoch_metrics.append(epoch_record)

            # --- Final evaluation ---
            val_final = evaluate(model, val_loader, self.device, self.num_classes)
            test_final = evaluate(model, test_loader, self.device, self.num_classes)

            result.final_train_loss = round(epoch_loss, 6)
            result.final_train_acc = round(epoch_acc, 6)
            result.val_loss = round(val_final["loss"], 6)
            result.val_acc = round(val_final["accuracy"], 6)
            result.val_per_class_accuracy = [
                round(a, 6) for a in val_final["per_class_accuracy"]
            ]
            result.test_loss = round(test_final["loss"], 6)
            result.test_acc = round(test_final["accuracy"], 6)
            result.test_per_class_accuracy = [
                round(a, 6) for a in test_final["per_class_accuracy"]
            ]
            result.completed = True

        except Exception as e:
            logger.exception("Training failed")
            result.error = str(e)
            result.completed = False

        result.wall_time_seconds = round(time.time() - start_time, 2)
        return result
