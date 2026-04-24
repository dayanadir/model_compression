"""Uniform index selection baseline."""

from __future__ import annotations

from typing import Mapping

import torch

from compression.axes import AxisGroup
from compression.selectors.base import CalibrationContext, Selector


class UniformSelector(Selector):
    name = "uniform"
    requires_calibration = False

    def select(
        self,
        axis_group: AxisGroup,
        teacher_state: Mapping[str, torch.Tensor],
        calib: CalibrationContext | None,
        rng: torch.Generator,
    ) -> torch.LongTensor:
        del teacher_state, calib, rng
        h = axis_group.width
        hs = axis_group.target
        idx = torch.div(
            torch.arange(hs, dtype=torch.long) * h,
            hs,
            rounding_mode="floor",
        )
        return idx
