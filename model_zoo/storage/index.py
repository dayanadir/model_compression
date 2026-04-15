"""Dataset index: tracks which models have been completed or failed."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetIndex:
    """Tracks completed and failed model IDs for resumability.

    Uses simple append-only text files that survive crashes.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)
        self._completed_file = self._base_dir / ".completed_ids.txt"
        self._failed_file = self._base_dir / ".failed_ids.txt"
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load existing completion state from disk."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        if self._completed_file.exists():
            self._completed = set(self._completed_file.read_text().splitlines())
        if self._failed_file.exists():
            self._failed = set(self._failed_file.read_text().splitlines())
        logger.info(
            "Index loaded: %d completed, %d failed",
            len(self._completed),
            len(self._failed),
        )

    def is_complete(self, model_id: str) -> bool:
        return model_id in self._completed

    def is_failed(self, model_id: str) -> bool:
        return model_id in self._failed

    def mark_complete(self, model_id: str) -> None:
        if model_id not in self._completed:
            self._completed.add(model_id)
            with open(self._completed_file, "a") as f:
                f.write(model_id + "\n")

    def mark_failed(self, model_id: str) -> None:
        if model_id not in self._failed:
            self._failed.add(model_id)
            with open(self._failed_file, "a") as f:
                f.write(model_id + "\n")

    @property
    def num_completed(self) -> int:
        return len(self._completed)

    @property
    def num_failed(self) -> int:
        return len(self._failed)

    def completed_ids(self) -> set[str]:
        return set(self._completed)
