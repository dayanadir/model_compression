"""Activation-based structured selector baseline."""

from __future__ import annotations

from typing import Mapping

import torch

from compression.axes import AxisGroup
from compression.selectors.base import CalibrationContext, Selector, stable_topk_indices


class ActivationSelector(Selector):
    name = "activation"
    requires_calibration = True

    def select(
        self,
        axis_group: AxisGroup,
        teacher_state: Mapping[str, torch.Tensor],
        calib: CalibrationContext | None,
        rng: torch.Generator,
    ) -> torch.LongTensor:
        del teacher_state, rng
        if calib is None:
            raise ValueError("Activation selector requires calibration context")
        if axis_group.id not in calib.group_scores:
            raise ValueError(f"Missing activation scores for group {axis_group.id}")
        scores = calib.group_scores[axis_group.id].detach().cpu().double()
        return stable_topk_indices(scores, axis_group.target)
