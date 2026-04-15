"""Base classes for architecture families."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn

from model_zoo.config import DatasetInfo, sample_from_spec


@dataclass(frozen=True)
class ArchHyperparams:
    """Base for family-specific architecture hyperparameters.

    Each family defines its own frozen dataclass subclass.
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingHyperparams:
    """Training hyperparameters, shared across all families."""

    optimizer: str
    lr: float
    weight_decay: float
    label_smoothing: float
    epochs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelFamily(ABC):
    """Abstract base for an architecture family.

    Subclasses must set ``family_name`` as a class attribute and implement
    ``sample_arch_hyperparams`` and ``build_model``.
    """

    family_name: str  # set by each subclass

    def __init__(
        self,
        search_space: dict[str, Any],
        training_space: dict[str, Any],
        dataset_info: DatasetInfo,
    ) -> None:
        self._space = search_space
        self._training_space = training_space
        self._dataset_info = dataset_info

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def sample_arch_hyperparams(self, rng: random.Random) -> ArchHyperparams:
        """Sample architecture hyperparameters from the configured search space."""
        ...

    @abstractmethod
    def build_model(self, hparams: ArchHyperparams) -> nn.Sequential:
        """Construct an nn.Sequential model from sampled hyperparameters."""
        ...

    # ------------------------------------------------------------------
    # Shared default: training hyperparameter sampling
    # ------------------------------------------------------------------

    def sample_training_hyperparams(self, rng: random.Random) -> TrainingHyperparams:
        """Sample training hyperparameters from training_defaults config."""
        ts = self._training_space
        return TrainingHyperparams(
            optimizer=sample_from_spec(rng, ts.get("optimizer", "adam")),
            lr=sample_from_spec(rng, ts.get("lr", 1e-3)),
            weight_decay=sample_from_spec(rng, ts.get("weight_decay", 0.0)),
            label_smoothing=sample_from_spec(rng, ts.get("label_smoothing", 0.0)),
            epochs=sample_from_spec(rng, ts.get("epochs", 50)),
        )
