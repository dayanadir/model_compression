"""Axis-group abstractions for consistent hidden-unit slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class IndexTransform:
    """How selected indices map onto a tensor axis."""

    kind: Literal["identity", "tile"]
    tile_k: int = 1

    @staticmethod
    def identity() -> "IndexTransform":
        return IndexTransform(kind="identity", tile_k=1)

    @staticmethod
    def tile(k: int) -> "IndexTransform":
        if k < 1:
            raise ValueError(f"tile factor must be >= 1, got {k}")
        return IndexTransform(kind="tile", tile_k=int(k))

    def expand(self, indices: torch.LongTensor, width: int) -> torch.LongTensor:
        if self.kind == "identity":
            return indices
        if self.kind == "tile":
            parts = [indices + (i * width) for i in range(self.tile_k)]
            return torch.cat(parts, dim=0)
        raise ValueError(f"Unknown transform kind: {self.kind}")


@dataclass(frozen=True)
class AxisMember:
    """One tensor axis tied to an axis group."""

    param_key: str
    axis: int
    transform: IndexTransform
    side: Literal["in", "out", "affine"]


@dataclass(frozen=True)
class HookSite:
    """Activation hook site mapped to an axis group."""

    module_name: str
    axis: int


@dataclass(frozen=True)
class AxisGroup:
    """A hidden-width bus that must use one consistent index subset."""

    id: str
    width: int
    target: int
    members: tuple[AxisMember, ...]
    hook_sites: tuple[HookSite, ...]

