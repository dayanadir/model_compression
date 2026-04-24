"""Axis-group specification for cnn2d."""

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

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Conv2d):
            conv_names.append(name)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            norm_names.append(name)
        elif isinstance(module, nn.Linear):
            if module.out_features == 10:
                classifier_name = name
            elif module.in_features == hidden and module.out_features == hidden:
                fc_hidden_names.append(name)

    if not conv_names or not classifier_name:
        raise ValueError("Unexpected cnn2d module layout")

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
    for norm_name, norm_module in [
        (n, m)
        for n, m in model.named_modules()
        if n in norm_names
    ]:
        add_norm_members(members, norm_name, norm_module)

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

    # Hook activations following conv layers only.
    seq = list(model.named_children())
    conv_seen = 0
    for idx, (name, module) in enumerate(seq):
        if isinstance(module, nn.Conv2d):
            conv_seen += 1
            # activation is typically within next 1-3 modules (dropout/norm/activation)
            for next_name, next_module in seq[idx + 1 : idx + 5]:
                if isinstance(next_module, (nn.ReLU, nn.GELU)):
                    hooks.append(activation_hook(next_name, axis=1))
                    break
        if conv_seen >= len(conv_names):
            break

    return [
        AxisGroup(
            id="cnn2d:H",
            width=hidden,
            target=target,
            members=tuple(members),
            hook_sites=tuple(hooks),
        )
    ]
