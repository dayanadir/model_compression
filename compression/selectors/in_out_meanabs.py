"""In+Out mean-|W| selector baseline."""

from __future__ import annotations

from typing import Mapping

import torch

from compression.axes import AxisGroup
from compression.selectors._score_utils import mean_abs_per_unit
from compression.selectors.base import CalibrationContext, Selector, stable_topk_indices


class InOutMeanAbsSelector(Selector):
    name = "in_out_meanabs"
    requires_calibration = False

    def __init__(self, include_affine_gamma: bool = False) -> None:
        self.include_affine_gamma = include_affine_gamma

    def select(
        self,
        axis_group: AxisGroup,
        teacher_state: Mapping[str, torch.Tensor],
        calib: CalibrationContext | None,
        rng: torch.Generator,
    ) -> torch.LongTensor:
        del calib, rng
        in_score = mean_abs_per_unit(teacher_state, axis_group, side="in")
        out_score = mean_abs_per_unit(teacher_state, axis_group, side="out")
        scores = in_score + out_score
        if self.include_affine_gamma:
            affine_score = mean_abs_per_unit(teacher_state, axis_group, side="affine")
            scores = scores + affine_score
        return stable_topk_indices(scores, axis_group.target)
