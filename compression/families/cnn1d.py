"""Axis-group specification for cnn1d."""

from __future__ import annotations

import torch.nn as nn

from compression.axes import AxisGroup, AxisMember, IndexTransform
from compression.families.common import activation_hook, add_norm_members


def build_axis_groups(model: nn.Module, arch: dict) -> list[AxisGroup]:
    hidden = int(arch["hidden_dim"])
    target = hidden // 2
    members: list[AxisMember] = []
    hooks = []

    conv_names: list[str] = []
    norm_names: list[str] = []
    fc_hidden_names: list[str] = []
    classifier_name = ""

    named_modules = dict(model.named_modules())
    for name, module in named_modules.items():
        if not name:
            continue
        if isinstance(module, nn.Conv1d):
            conv_names.append(name)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
            norm_names.append(name)
        elif isinstance(module, nn.Linear):
            if module.out_features == 10:
                classifier_name = name
            elif module.in_features == hidden and module.out_features == hidden:
                fc_hidden_names.append(name)

    if not conv_names or not classifier_name:
        raise ValueError("Unexpected cnn1d module layout")

    first_conv = conv_names[0]
    members.extend(
        [
            AxisMember(f"{first_conv}.weight", 0, IndexTransform.identity(), "out"),
            AxisMember(f"{first_conv}.bias", 0, IndexTransform.identity(), "out"),
        ]
    )
    for conv_name in conv_names[1:]:
        members.extend(
            [
                AxisMember(f"{conv_name}.weight", 0, IndexTransform.identity(), "out"),
                AxisMember(f"{conv_name}.weight", 1, IndexTransform.identity(), "in"),
                AxisMember(f"{conv_name}.bias", 0, IndexTransform.identity(), "out"),
            ]
        )
    for norm_name in norm_names:
        add_norm_members(members, norm_name, named_modules[norm_name])

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

    seq = list(model.named_children())
    conv_seen = 0
    for idx, (name, module) in enumerate(seq):
        if isinstance(module, nn.Conv1d):
            conv_seen += 1
            for next_name, next_module in seq[idx + 1 : idx + 5]:
                if isinstance(next_module, (nn.ReLU, nn.GELU)):
                    hooks.append(activation_hook(next_name, axis=1))
                    break
        if conv_seen >= len(conv_names):
            break

    return [
        AxisGroup(
            id="cnn1d:H",
            width=hidden,
            target=target,
            members=tuple(members),
            hook_sites=tuple(hooks),
        )
    ]
