"""Common helpers for family axis-group construction."""

from __future__ import annotations

import torch.nn as nn

from compression.axes import AxisMember, HookSite, IndexTransform


def add_norm_members(members: list[AxisMember], module_name: str, module: nn.Module) -> None:
    members.append(
        AxisMember(
            param_key=f"{module_name}.weight",
            axis=0,
            transform=IndexTransform.identity(),
            side="affine",
        )
    )
    members.append(
        AxisMember(
            param_key=f"{module_name}.bias",
            axis=0,
            transform=IndexTransform.identity(),
            side="affine",
        )
    )
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        members.append(
            AxisMember(
                param_key=f"{module_name}.running_mean",
                axis=0,
                transform=IndexTransform.identity(),
                side="affine",
            )
        )
        members.append(
            AxisMember(
                param_key=f"{module_name}.running_var",
                axis=0,
                transform=IndexTransform.identity(),
                side="affine",
            )
        )


def activation_hook(module_name: str, axis: int) -> HookSite:
    return HookSite(module_name=module_name, axis=axis)
