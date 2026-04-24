"""Slice-and-copy surgery engine."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch

from compression.axes import AxisGroup


def apply_axis_group_surgery(
    teacher_state: Mapping[str, torch.Tensor],
    student_state: Mapping[str, torch.Tensor],
    axis_groups: list[AxisGroup],
    selections: Mapping[str, torch.LongTensor],
) -> dict[str, torch.Tensor]:
    """Return a new student state initialized by teacher slices."""

    rules: dict[str, list[tuple[int, torch.LongTensor]]] = defaultdict(list)
    for group in axis_groups:
        if group.id not in selections:
            raise ValueError(f"Missing selection for axis group {group.id}")
        idx = selections[group.id].to(dtype=torch.long, device="cpu")
        for member in group.members:
            expanded = member.transform.expand(idx, group.width)
            rules[member.param_key].append((member.axis, expanded))

    out: dict[str, torch.Tensor] = {}
    for key, student_tensor in student_state.items():
        if key not in teacher_state:
            out[key] = student_tensor.clone()
            continue
        teacher_tensor = teacher_state[key]
        if key not in rules:
            if teacher_tensor.shape == student_tensor.shape:
                out[key] = teacher_tensor.detach().clone()
            else:
                # Keep student default for shape-mismatch tensors like SinPosEnc buffers.
                out[key] = student_tensor.detach().clone()
            continue

        transformed = teacher_tensor
        for axis, idx in sorted(rules[key], key=lambda x: x[0], reverse=True):
            transformed = torch.index_select(
                transformed, dim=axis, index=idx.to(teacher_tensor.device)
            )
        if transformed.shape != student_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {key}: got {tuple(transformed.shape)}, "
                f"expected {tuple(student_tensor.shape)}"
            )
        out[key] = transformed.detach().clone()
    return out
