"""Axis-group specification for deepsets."""

from __future__ import annotations

import torch.nn as nn

from gmn.graph_construct.layers import EquivSetLinear

from compression.axes import AxisGroup, AxisMember, IndexTransform
from compression.families.common import activation_hook, add_norm_members


def build_axis_groups(model: nn.Module, arch: dict) -> list[AxisGroup]:
    hidden = int(arch["hidden_dim"])
    target = hidden // 2
    members: list[AxisMember] = []
    hooks = []

    named_modules = dict(model.named_modules())
    conv_names: list[str] = []
    norm_names: list[str] = []
    equiv_names: list[str] = []
    fc_hidden_names: list[str] = []
    classifier_name = ""

    for name, module in named_modules.items():
        if not name:
            continue
        if isinstance(module, nn.Conv1d):
            conv_names.append(name)
        elif isinstance(module, EquivSetLinear):
            equiv_names.append(name)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
            norm_names.append(name)
        elif isinstance(module, nn.Linear):
            if module.out_features == 10:
                classifier_name = name
            elif module.in_features == hidden and module.out_features == hidden:
                fc_hidden_names.append(name)

    if not conv_names or not classifier_name:
        raise ValueError("Unexpected deepsets module layout")

    input_proj = conv_names[0]
    members.extend(
        [
            AxisMember(f"{input_proj}.weight", 0, IndexTransform.identity(), "out"),
            AxisMember(f"{input_proj}.bias", 0, IndexTransform.identity(), "out"),
        ]
    )

    for norm_name in norm_names:
        add_norm_members(members, norm_name, named_modules[norm_name])

    for eq_name in equiv_names:
        members.extend(
            [
                AxisMember(
                    f"{eq_name}.lin1.weight", 0, IndexTransform.identity(), "out"
                ),
                AxisMember(
                    f"{eq_name}.lin1.weight", 1, IndexTransform.identity(), "in"
                ),
                AxisMember(f"{eq_name}.lin1.bias", 0, IndexTransform.identity(), "out"),
                AxisMember(
                    f"{eq_name}.lin2.weight", 0, IndexTransform.identity(), "out"
                ),
                AxisMember(
                    f"{eq_name}.lin2.weight", 1, IndexTransform.identity(), "in"
                ),
            ]
        )

    for fc_name in fc_hidden_names:
        members.extend(
            [
                AxisMember(f"{fc_name}.weight", 0, IndexTransform.identity(), "out"),
                AxisMember(f"{fc_name}.weight", 1, IndexTransform.identity(), "in"),
                AxisMember(f"{fc_name}.bias", 0, IndexTransform.identity(), "out"),
            ]
        )

    members.append(
        AxisMember(f"{classifier_name}.weight", 1, IndexTransform.identity(), "in")
    )

    # Hook first-stage activation after input projection and each post-equiv activation.
    seq = list(model.named_children())
    for idx, (name, module) in enumerate(seq):
        if not isinstance(module, (nn.ReLU, nn.GELU)):
            continue
        prev_modules = [m for _, m in seq[:idx] if isinstance(m, (nn.Conv1d, EquivSetLinear))]
        if not prev_modules:
            continue
        prev = prev_modules[-1]
        if isinstance(prev, nn.Conv1d) and prev.in_channels == 3:
            hooks.append(activation_hook(name, axis=1))
        elif isinstance(prev, EquivSetLinear):
            hooks.append(activation_hook(name, axis=1))

    return [
        AxisGroup(
            id="deepsets:H",
            width=hidden,
            target=target,
            members=tuple(members),
            hook_sites=tuple(hooks),
        )
    ]
