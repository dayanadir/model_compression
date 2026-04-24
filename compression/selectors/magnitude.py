"""Structured magnitude selector baseline."""

from __future__ import annotations

from typing import Mapping

import torch

from compression.axes import AxisGroup
from compression.selectors._score_utils import sum_abs_per_unit
from compression.selectors.base import CalibrationContext, Selector, stable_topk_indices


class MagnitudeSelector(Selector):
    name = "magnitude"
    requires_calibration = False

    def select(
        self,
        axis_group: AxisGroup,
        teacher_state: Mapping[str, torch.Tensor],
        calib: CalibrationContext | None,
        rng: torch.Generator,
    ) -> torch.LongTensor:
        del calib, rng
        scores = sum_abs_per_unit(
            teacher_state=teacher_state,
            group=axis_group,
            include_affine=False,
        )
        return stable_topk_indices(scores, axis_group.target)
