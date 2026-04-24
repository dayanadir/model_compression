"""Axis-group specification for resnet."""

from __future__ import annotations

import torch.nn as nn

from gmn.graph_construct.layers import BasicBlock

from compression.axes import AxisGroup, AxisMember, HookSite, IndexTransform
from compression.families.common import add_norm_members


def build_axis_groups(model: nn.Module, arch: dict) -> list[AxisGroup]:
    hidden = int(arch["hidden_dim"])
    target = hidden // 2
    stem_width = hidden // 2
    stem_target = stem_width // 2

    named_modules = dict(model.named_modules())
    blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, BasicBlock)]
    classifier_name = None
    if not blocks:
        raise ValueError("Unexpected resnet module layout: no BasicBlock")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 10:
            classifier_name = name
            break
    if classifier_name is None:
        raise ValueError("Unexpected resnet module layout: classifier not found")

    members_stem: list[AxisMember] = []
    members_main: list[AxisMember] = []

    # Stem: 0(conv),1(bn),2(relu)
    members_stem.extend(
        [
            AxisMember("0.weight", 0, IndexTransform.identity(), "out"),
            AxisMember("0.bias", 0, IndexTransform.identity(), "out"),
        ]
    )
    add_norm_members(members_stem, "1", named_modules["1"])

    first_block_name, _ = blocks[0]
    members_stem.extend(
        [
            AxisMember(
                f"{first_block_name}.conv1.weight", 1, IndexTransform.identity(), "in"
            ),
            AxisMember(
                f"{first_block_name}.shortcut.0.weight",
                1,
                IndexTransform.identity(),
                "in",
            ),
        ]
    )

    # Main H group.
    for block_idx, (block_name, block_module) in enumerate(blocks):
        members_main.append(
            AxisMember(f"{block_name}.conv1.weight", 0, IndexTransform.identity(), "out")
        )
        if block_idx > 0:
            members_main.append(
                AxisMember(
                    f"{block_name}.conv1.weight", 1, IndexTransform.identity(), "in"
                )
            )
        members_main.extend(
            [
                AxisMember(
                    f"{block_name}.conv2.weight", 0, IndexTransform.identity(), "out"
                ),
                AxisMember(
                    f"{block_name}.conv2.weight", 1, IndexTransform.identity(), "in"
                ),
            ]
        )
        add_norm_members(members_main, f"{block_name}.bn1", block_module.bn1)
        add_norm_members(members_main, f"{block_name}.bn2", block_module.bn2)
        if len(block_module.shortcut) == 2:
            members_main.append(
                AxisMember(
                    f"{block_name}.shortcut.0.weight",
                    0,
                    IndexTransform.identity(),
                    "out",
                )
            )
            add_norm_members(
                members_main, f"{block_name}.shortcut.1", block_module.shortcut[1]
            )

    members_main.append(
        AxisMember(f"{classifier_name}.weight", 1, IndexTransform.identity(), "in")
    )

    hook_stem = (HookSite(module_name="2", axis=1),)
    hook_main = tuple(HookSite(module_name=block_name, axis=1) for block_name, _ in blocks)

    return [
        AxisGroup(
            id="resnet:stem",
            width=stem_width,
            target=stem_target,
            members=tuple(members_stem),
            hook_sites=hook_stem,
        ),
        AxisGroup(
            id="resnet:H",
            width=hidden,
            target=target,
            members=tuple(members_main),
            hook_sites=hook_main,
        ),
    ]
