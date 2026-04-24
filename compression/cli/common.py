"""Shared helpers for compression CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from compression.families import build_model_for_family


ALL_METHODS = [
    "uniform",
    "random_consistent",
    "l1_structured",
    "magnitude",
    "in_out_meanabs",
    "activation",
    "he_reinit",
]


def resolve_methods(methods_arg: str) -> list[str]:
    if methods_arg == "all":
        return list(ALL_METHODS)
    methods = [m.strip() for m in methods_arg.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")
    return methods


def load_model_bundle(model_dir: Path, map_location: str = "cpu") -> tuple[dict[str, Any], torch.nn.Module]:
    meta_path = model_dir / "metadata.json"
    weights_path = model_dir / "weights.pt"
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    family = metadata["family"]
    arch = metadata["architecture"]
    model = build_model_for_family(family=family, arch=arch)
    state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state, strict=True)
    return metadata, model


def num_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))
