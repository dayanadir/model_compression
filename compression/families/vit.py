"""Axis-group specification for vit."""

from __future__ import annotations

import torch.nn as nn

from gmn.graph_construct.layers import PositionwiseFeedForward, SelfAttention

from compression.axes import AxisGroup, AxisMember, HookSite, IndexTransform
from compression.families.common import add_norm_members


def build_axis_groups(model: nn.Module, arch: dict) -> list[AxisGroup]:
    hidden = int(arch["hidden_dim"])
    target = hidden // 2

    named_modules = dict(model.named_modules())
    members: list[AxisMember] = []
    hooks: list[HookSite] = []
    classifier_name = None

    # Patch embedding.
    members.extend(
        [
            AxisMember("0.weight", 0, IndexTransform.identity(), "out"),
            AxisMember("0.bias", 0, IndexTransform.identity(), "out"),
        ]
    )
    hooks.append(HookSite(module_name="2", axis=2))

    for name, module in named_modules.items():
        if not name:
            continue
        if isinstance(module, nn.LayerNorm):
            add_norm_members(members, name, module)
        elif isinstance(module, SelfAttention):
            members.extend(
                [
                    AxisMember(
                        f"{name}.attn.in_proj_weight", 0, IndexTransform.tile(3), "out"
                    ),
                    AxisMember(
                        f"{name}.attn.in_proj_weight", 1, IndexTransform.identity(), "in"
                    ),
                    AxisMember(
                        f"{name}.attn.in_proj_bias", 0, IndexTransform.tile(3), "out"
                    ),
                    AxisMember(
                        f"{name}.attn.out_proj.weight",
                        0,
                        IndexTransform.identity(),
                        "out",
                    ),
                    AxisMember(
                        f"{name}.attn.out_proj.weight", 1, IndexTransform.identity(), "in"
                    ),
                    AxisMember(
                        f"{name}.attn.out_proj.bias", 0, IndexTransform.identity(), "out"
                    ),
                ]
            )
            hooks.append(HookSite(module_name=name, axis=2))
        elif isinstance(module, PositionwiseFeedForward):
            members.extend(
                [
                    AxisMember(
                        f"{name}.lin1.weight", 0, IndexTransform.tile(4), "out"
                    ),
                    AxisMember(
                        f"{name}.lin1.weight", 1, IndexTransform.identity(), "in"
                    ),
                    AxisMember(f"{name}.lin1.bias", 0, IndexTransform.tile(4), "out"),
                    AxisMember(
                        f"{name}.lin2.weight", 0, IndexTransform.identity(), "out"
                    ),
                    AxisMember(f"{name}.lin2.weight", 1, IndexTransform.tile(4), "in"),
                    AxisMember(
                        f"{name}.lin2.bias", 0, IndexTransform.identity(), "out"
                    ),
                ]
            )
            hooks.append(HookSite(module_name=name, axis=2))
        elif isinstance(module, nn.Linear) and module.out_features == 10:
            classifier_name = name

    if classifier_name is None:
        raise ValueError("ViT classifier layer not found")
    members.append(AxisMember(f"{classifier_name}.weight", 1, IndexTransform.identity(), "in"))

    return [
        AxisGroup(
            id="vit:H",
            width=hidden,
            target=target,
            members=tuple(members),
            hook_sites=tuple(hooks),
        )
    ]
