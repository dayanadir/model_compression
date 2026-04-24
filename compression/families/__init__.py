"""Family-specific builders and axis-group factories."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import torch.nn as nn

try:
    from gmn.graph_construct.net_makers import (
        make_cnn,
        make_cnn_1d,
        make_deepsets,
        make_resnet,
        make_transformer,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    gmn_root = repo_root / "graph_metanetworks-main"
    if str(gmn_root) not in sys.path:
        sys.path.append(str(gmn_root))
    from gmn.graph_construct.net_makers import (
        make_cnn,
        make_cnn_1d,
        make_deepsets,
        make_resnet,
        make_transformer,
    )

from compression.axes import AxisGroup
from compression.families import cnn1d, cnn2d, deepsets, resnet, vit


def halve_architecture(arch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(arch)
    hidden = int(out["hidden_dim"])
    if hidden % 2 != 0:
        raise ValueError(f"hidden_dim must be even for H->H/2, got {hidden}")
    out["hidden_dim"] = hidden // 2
    return out


def build_model_for_family(family: str, arch: dict[str, Any]) -> nn.Sequential:
    if family == "cnn2d":
        return make_cnn(
            conv_layers=int(arch["conv_layers"]),
            fc_layers=int(arch["fc_layers"]),
            hidden_dim=int(arch["hidden_dim"]),
            in_dim=3,
            num_classes=10,
            activation=str(arch["activation"]),
            norm=str(arch["norm"]),
            dropout=float(arch["dropout"]),
        )
    if family == "cnn1d":
        return make_cnn_1d(
            conv_layers=int(arch["conv_layers"]),
            fc_layers=int(arch["fc_layers"]),
            hidden_dim=int(arch["hidden_dim"]),
            in_dim=3,
            num_classes=10,
            activation=str(arch["activation"]),
            norm=str(arch["norm"]),
            dropout=float(arch["dropout"]),
        )
    if family == "deepsets":
        return make_deepsets(
            conv_layers=int(arch["equivariant_layers"]),
            fc_layers=int(arch["fc_layers"]),
            hidden_dim=int(arch["hidden_dim"]),
            in_dim=3,
            num_classes=10,
            activation=str(arch["activation"]),
            norm=str(arch["norm"]),
            dropout=float(arch["dropout"]),
        )
    if family == "resnet":
        return make_resnet(
            conv_layers=int(arch["blocks"]),
            hidden_dim=int(arch["hidden_dim"]),
            in_dim=3,
            num_classes=10,
        )
    if family == "vit":
        return make_transformer(
            in_dim=3,
            hidden_dim=int(arch["hidden_dim"]),
            num_heads=int(arch["num_heads"]),
            out_dim=10,
            dropout=float(arch["dropout"]),
            num_layers=int(arch["num_layers"]),
            vit=True,
            patch_size=int(arch["patch_size"]),
        )
    raise ValueError(f"Unknown family: {family}")


def build_axis_groups(family: str, model: nn.Module, arch: dict[str, Any]) -> list[AxisGroup]:
    if family == "cnn2d":
        return cnn2d.build_axis_groups(model, arch)
    if family == "cnn1d":
        return cnn1d.build_axis_groups(model, arch)
    if family == "deepsets":
        return deepsets.build_axis_groups(model, arch)
    if family == "resnet":
        return resnet.build_axis_groups(model, arch)
    if family == "vit":
        return vit.build_axis_groups(model, arch)
    raise ValueError(f"Unknown family: {family}")
