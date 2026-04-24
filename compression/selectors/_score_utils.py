"""Shared score aggregation helpers for weight-based selectors."""

from __future__ import annotations

from typing import Mapping

import torch

from compression.axes import AxisGroup, AxisMember


def _member_view(
    teacher_state: Mapping[str, torch.Tensor],
    member: AxisMember,
    width: int,
) -> torch.Tensor:
    tensor = teacher_state[member.param_key]
    if member.transform.kind == "identity":
        return tensor
    if member.transform.kind == "tile":
        if tensor.shape[member.axis] != member.transform.tile_k * width:
            raise ValueError(
                f"Unexpected tiled axis width for {member.param_key}: "
                f"{tensor.shape[member.axis]} vs {member.transform.tile_k * width}"
            )
        pieces = torch.chunk(tensor, member.transform.tile_k, dim=member.axis)
        return torch.stack(pieces, dim=0).mean(dim=0)
    raise ValueError(f"Unsupported transform: {member.transform.kind}")


def sum_abs_per_unit(
    teacher_state: Mapping[str, torch.Tensor],
    group: AxisGroup,
    include_affine: bool = False,
) -> torch.Tensor:
    out = torch.zeros(group.width, dtype=torch.float64)
    for member in group.members:
        if member.side == "affine" and not include_affine:
            continue
        view = _member_view(teacher_state, member, group.width).double().abs()
        reduce_axes = tuple(i for i in range(view.ndim) if i != member.axis)
        contrib = view if not reduce_axes else view.sum(dim=reduce_axes)
        out += contrib.cpu()
    return out


def mean_abs_per_unit(
    teacher_state: Mapping[str, torch.Tensor],
    group: AxisGroup,
    side: str,
) -> torch.Tensor:
    parts = []
    for member in group.members:
        if member.side != side:
            continue
        view = _member_view(teacher_state, member, group.width).double().abs()
        reduce_axes = tuple(i for i in range(view.ndim) if i != member.axis)
        contrib = view if not reduce_axes else view.mean(dim=reduce_axes)
        parts.append(contrib.cpu())
    if not parts:
        return torch.zeros(group.width, dtype=torch.float64)
    return torch.stack(parts, dim=0).mean(dim=0)

