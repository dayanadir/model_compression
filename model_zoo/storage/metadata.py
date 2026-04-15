"""Metadata construction for trained model artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch.nn as nn

from model_zoo import __version__
from model_zoo.families.base import ArchHyperparams, TrainingHyperparams
from model_zoo.training.trainer import TrainResult


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _get_layer_shapes(model: nn.Module) -> dict[str, list[int]]:
    return {name: list(p.shape) for name, p in model.named_parameters()}


def _compute_depth(model: nn.Module) -> int:
    """Count parameterized layers (conv, linear, etc.)."""
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            count += 1
    return count


class MetadataBuilder:
    """Assembles a standardized metadata dict for one trained model."""

    @staticmethod
    def build(
        model_id: str,
        family_name: str,
        arch_hparams: ArchHyperparams,
        training_hparams: TrainingHyperparams,
        train_result: TrainResult,
        model: nn.Module,
        seed: int,
        dataset_name: str,
        input_shape: list[int],
        output_shape: list[int],
        training_started_at: str | None = None,
    ) -> dict[str, Any]:
        """Build the full metadata dict.

        Returns:
            JSON-serializable dict with all model information.
        """
        num_params = _count_parameters(model)
        layer_shapes = _get_layer_shapes(model)
        depth = _compute_depth(model)
        arch_dict = arch_hparams.to_dict()
        now = datetime.now(timezone.utc).isoformat()

        return {
            "model_id": model_id,
            "family": family_name,
            "seed": seed,
            "architecture": arch_dict,
            "training": {
                **training_hparams.to_dict(),
                "dataset": dataset_name,
            },
            "results": {
                "final_train_loss": train_result.final_train_loss,
                "final_train_acc": train_result.final_train_acc,
                "val_loss": train_result.val_loss,
                "val_acc": train_result.val_acc,
                "val_per_class_accuracy": train_result.val_per_class_accuracy,
                "test_loss": train_result.test_loss,
                "test_acc": train_result.test_acc,
                "test_per_class_accuracy": train_result.test_per_class_accuracy,
                "wall_time_seconds": train_result.wall_time_seconds,
                "completed": train_result.completed,
                "error": train_result.error,
            },
            "model_info": {
                "num_params": num_params,
                "layer_shapes": layer_shapes,
                "input_shape": input_shape,
                "output_shape": output_shape,
            },
            "summary": {
                "model_id": model_id,
                "family": family_name,
                "num_params": num_params,
                "test_acc": train_result.test_acc,
                "val_acc": train_result.val_acc,
                "hidden_dim": arch_dict.get("hidden_dim"),
                "depth": depth,
            },
            "timestamps": {
                "training_started_at": training_started_at,
                "training_finished_at": now,
                "created_at": now,
            },
            "version": __version__,
        }
