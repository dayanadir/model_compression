"""Configuration dataclasses and YAML loader for the model zoo pipeline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dataset info (derived from dataset name)
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "cifar10": {"in_channels": 3, "num_classes": 10, "image_size": 32},
    "cifar100": {"in_channels": 3, "num_classes": 100, "image_size": 32},
    "mnist": {"in_channels": 1, "num_classes": 10, "image_size": 28},
    "fashion_mnist": {"in_channels": 1, "num_classes": 10, "image_size": 28},
}


@dataclass(frozen=True)
class DatasetInfo:
    """Static properties of the image classification dataset."""

    name: str
    in_channels: int
    num_classes: int
    image_size: int
    data_dir: str
    batch_size: int
    val_fraction: float

    @classmethod
    def from_config(cls, cfg: dict) -> DatasetInfo:
        name = cfg["name"]
        if name not in _DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. Supported: {list(_DATASET_REGISTRY)}"
            )
        info = _DATASET_REGISTRY[name]
        return cls(
            name=name,
            in_channels=info["in_channels"],
            num_classes=info["num_classes"],
            image_size=info["image_size"],
            data_dir=cfg.get("data_dir", "./data"),
            batch_size=cfg.get("batch_size", 128),
            val_fraction=cfg.get("val_fraction", 0.1),
        )


# ---------------------------------------------------------------------------
# Family config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyConfig:
    """Per-family configuration: count and search space."""

    count: int
    search_space: dict[str, Any]


# ---------------------------------------------------------------------------
# Run config (top-level)
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Top-level configuration for a model zoo generation run."""

    output_dir: Path
    base_seed: int
    device: str
    num_workers: int
    dataset: DatasetInfo
    training_defaults: dict[str, Any]
    families: dict[str, FamilyConfig]
    gpu_ids: list[int] = field(default_factory=list)

    def family_base_offsets(self) -> dict[str, int]:
        """Global ``model_index`` starts at this offset for each family (YAML order)."""
        offsets: dict[str, int] = {}
        offset = 0
        for name, fcfg in self.families.items():
            offsets[name] = offset
            offset += fcfg.count
        return offsets

    def model_index_for_family_slot(self, *, family: str, family_slot: int) -> int:
        """Map (family, within-family slot) to the canonical global ``model_index``."""
        if family not in self.families:
            raise ValueError(
                f"Unknown family '{family}'. Known: {list(self.families.keys())}"
            )
        fcfg = self.families[family]
        if not 0 <= family_slot < fcfg.count:
            raise ValueError(
                f"family_slot must be in [0, {fcfg.count}), got {family_slot} for family '{family}'"
            )
        return self.family_base_offsets()[family] + int(family_slot)

    def interleave_step_to_family_slot_index(self, step: int) -> tuple[str, int, int]:
        """Map a linear sweep step to round-robin (family, slot) and canonical ``model_index``.

        Trial order ``step = 0, 1, 2, …`` cycles families in YAML order, then advances slot:
        ``(F0,s0), (F1,s0), …, (F_{k-1},s0), (F0,s1), …``

        Requires every family to have the same ``count`` (same as W&B balanced grid).
        """
        names = list(self.families.keys())
        counts = [fc.count for fc in self.families.values()]
        if len(set(counts)) != 1:
            raise ValueError(
                "Round-robin interleave requires equal `count` for every family; got "
                f"{dict(zip(names, counts))}"
            )
        n_fam = len(names)
        per = counts[0]
        total = n_fam * per
        if not 0 <= step < total:
            raise ValueError(
                f"interleave_step must be in [0, {total}), got {step}"
            )
        fi = step % n_fam
        slot = step // n_fam
        family = names[fi]
        model_index = self.model_index_for_family_slot(family=family, family_slot=slot)
        return family, slot, model_index

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)

        run = raw["run"]
        dataset_info = DatasetInfo.from_config(raw["dataset"])

        families: dict[str, FamilyConfig] = {}
        for name, fcfg in raw["families"].items():
            families[name] = FamilyConfig(
                count=fcfg["count"],
                search_space=fcfg["search_space"],
            )

        return cls(
            output_dir=Path(run["output_dir"]),
            base_seed=run.get("base_seed", 42),
            device=run.get("device", "cuda"),
            gpu_ids=[int(x) for x in run.get("gpu_ids", [])],
            num_workers=run.get("num_workers", 4),
            dataset=dataset_info,
            training_defaults=raw.get("training_defaults", {}),
            families=families,
        )


# ---------------------------------------------------------------------------
# Sampling helpers (used by ModelFamily.sample_training_hyperparams)
# ---------------------------------------------------------------------------


def sample_from_spec(rng: random.Random, spec: Any) -> Any:
    """Sample a value from a config specification.

    Specs can be:
    - A plain list of discrete choices, e.g. [16, 32, 64]
    - A [min, max, "uniform"] triple for continuous uniform
    - A [min, max, "log_uniform"] triple for log-uniform
    - A scalar (returned as-is)
    """
    if not isinstance(spec, list):
        return spec

    if (
        len(spec) == 3
        and isinstance(spec[2], str)
        and spec[2] in ("uniform", "log_uniform")
    ):
        lo, hi, dist = spec
        if dist == "uniform":
            return rng.uniform(lo, hi)
        if dist == "log_uniform":
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))

    # Discrete choice list
    return rng.choice(spec)
