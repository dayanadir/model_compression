"""Selector interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch

from compression.axes import AxisGroup


@dataclass(frozen=True)
class CalibrationContext:
    """Calibration scores keyed by axis-group id."""

    group_scores: Mapping[str, torch.Tensor]


class Selector(ABC):
    """Selects teacher indices to keep for one axis group."""

    name: str
    requires_calibration: bool = False

    @abstractmethod
    def select(
        self,
        axis_group: AxisGroup,
        teacher_state: Mapping[str, torch.Tensor],
        calib: CalibrationContext | None,
        rng: torch.Generator,
    ) -> torch.LongTensor:
        raise NotImplementedError


def stable_topk_indices(scores: torch.Tensor, k: int) -> torch.LongTensor:
    """Top-k with deterministic lower-index tie-breaking."""

    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {tuple(scores.shape)}")
    if k <= 0 or k > scores.numel():
        raise ValueError(f"Invalid k={k} for {scores.numel()} scores")
    eps = 1e-12
    adjusted = scores - (eps * torch.arange(scores.numel(), device=scores.device))
    idx = torch.topk(adjusted, k=k, largest=True, sorted=True).indices
    return torch.sort(idx).values.to(dtype=torch.long, device="cpu")
