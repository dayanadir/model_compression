"""Random selection with consistency baseline."""

from __future__ import annotations

from typing import Mapping

import torch

from compression.axes import AxisGroup
from compression.selectors.base import CalibrationContext, Selector


class RandomConsistentSelector(Selector):
    name = "random_consistent"
    requires_calibration = False

    def select(
        self,
        axis_group: AxisGroup,
        teacher_state: Mapping[str, torch.Tensor],
        calib: CalibrationContext | None,
        rng: torch.Generator,
    ) -> torch.LongTensor:
        del teacher_state, calib
        perm = torch.randperm(axis_group.width, generator=rng)
        idx = torch.sort(perm[: axis_group.target]).values
        return idx.to(dtype=torch.long, device="cpu")
