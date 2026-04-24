"""Structured L1 selector baseline."""

from __future__ import annotations

from typing import Mapping

import torch

from compression.axes import AxisGroup
from compression.selectors._score_utils import _member_view
from compression.selectors.base import CalibrationContext, Selector, stable_topk_indices


class L1StructuredSelector(Selector):
    name = "l1_structured"
    requires_calibration = False

    def select(
        self,
        axis_group: AxisGroup,
        teacher_state: Mapping[str, torch.Tensor],
        calib: CalibrationContext | None,
        rng: torch.Generator,
    ) -> torch.LongTensor:
        del calib, rng
        rep = None
        for member in axis_group.members:
            if member.side != "out":
                continue
            tensor = teacher_state[member.param_key]
            if tensor.ndim >= 2:
                rep = member
                break
        if rep is None:
            raise ValueError(f"No outgoing multidim member for group {axis_group.id}")

        view = _member_view(teacher_state, rep, axis_group.width).double().abs()
        reduce_axes = tuple(i for i in range(view.ndim) if i != rep.axis)
        scores = view if not reduce_axes else view.sum(dim=reduce_axes)
        return stable_topk_indices(scores, axis_group.target)
