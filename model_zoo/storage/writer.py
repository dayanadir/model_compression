"""Artifact writer: persists model weights and metadata to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ModelArtifactWriter:
    """Writes one model's artifacts (weights.pt + metadata.json) to disk."""

    WEIGHTS_FILE = "weights.pt"
    METADATA_FILE = "metadata.json"

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def _model_dir(self, model_id: str) -> Path:
        return self.base_dir / model_id

    def save(
        self,
        model_id: str,
        model: nn.Module,
        metadata: dict[str, Any],
    ) -> Path:
        """Save model weights and metadata to ``base_dir/model_id/``.

        Returns:
            Path to the model directory.
        """
        model_dir = self._model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_path = model_dir / self.WEIGHTS_FILE
        torch.save(model.state_dict(), weights_path)

        # Save metadata
        metadata_path = model_dir / self.METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return model_dir

    def save_failure(
        self,
        model_id: str,
        metadata: dict[str, Any],
    ) -> Path:
        """Save metadata for a failed training run (no weights)."""
        model_dir = self._model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = model_dir / self.METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return model_dir

    def exists(self, model_id: str) -> bool:
        """Check if a model has been fully saved (both files present)."""
        model_dir = self._model_dir(model_id)
        return (
            (model_dir / self.WEIGHTS_FILE).exists()
            and (model_dir / self.METADATA_FILE).exists()
        )
