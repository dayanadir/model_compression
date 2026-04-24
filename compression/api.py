"""Public compression API."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import torch
import torch.nn as nn

from compression.activations.calibration import CalibrationData, get_calibration_data
from compression.activations.collector import collect_activation_scores
from compression.families import build_axis_groups, build_model_for_family, halve_architecture
from compression.reinit.he import apply_he_reinit
from compression.selectors import (
    ActivationSelector,
    InOutMeanAbsSelector,
    L1StructuredSelector,
    MagnitudeSelector,
    RandomConsistentSelector,
    UniformSelector,
)
from compression.selectors.base import CalibrationContext, Selector
from compression.surgery import apply_axis_group_surgery


@dataclass(frozen=True)
class CalibrationConfig:
    data_dir: str = "./data"
    num_images: int = 512
    batch_size: int = 128
    seed: int = 0
    num_workers: int = 0


@dataclass(frozen=True)
class CompressionConfig:
    seed: int = 0
    device: str = "cpu"
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    include_affine_gamma: bool = False


@dataclass
class CompressionReport:
    family: str
    method: str
    teacher_hidden_dim: int
    student_hidden_dim: int
    indices_per_group: dict[str, list[int]]
    seed: int | None
    calibration: dict[str, Any] | None = None


def _selector_from_method(method: str, cfg: CompressionConfig) -> Selector:
    if method == "uniform":
        return UniformSelector()
    if method == "random_consistent":
        return RandomConsistentSelector()
    if method == "l1_structured":
        return L1StructuredSelector()
    if method == "magnitude":
        return MagnitudeSelector()
    if method == "in_out_meanabs":
        return InOutMeanAbsSelector(include_affine_gamma=cfg.include_affine_gamma)
    if method == "activation":
        return ActivationSelector()
    raise ValueError(f"Unsupported method: {method}")


def _group_seed(master_seed: int, group_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{group_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _collect_calibration_if_needed(
    selector: Selector,
    teacher: nn.Module,
    axis_groups,
    cfg: CompressionConfig,
) -> tuple[CalibrationContext | None, CalibrationData | None]:
    if not selector.requires_calibration:
        return None, None

    calib_data = get_calibration_data(
        data_dir=cfg.calibration.data_dir,
        num_images=cfg.calibration.num_images,
        batch_size=cfg.calibration.batch_size,
        seed=cfg.calibration.seed,
        num_workers=cfg.calibration.num_workers,
    )
    calib_ctx = collect_activation_scores(
        teacher=teacher,
        axis_groups=axis_groups,
        calib_loader=calib_data.loader,
        device=torch.device(cfg.device),
    )
    return calib_ctx, calib_data


def compress(
    *,
    teacher: nn.Module,
    family: str,
    architecture: dict[str, Any],
    method: str,
    cfg: CompressionConfig | None = None,
) -> tuple[nn.Module, CompressionReport]:
    """Compress one teacher model from H to H/2 for the chosen method."""

    cfg = cfg or CompressionConfig()
    teacher_hidden = int(architecture["hidden_dim"])
    student_arch = halve_architecture(architecture)
    student_hidden = int(student_arch["hidden_dim"])
    student = build_model_for_family(family, student_arch)

    if method == "he_reinit":
        apply_he_reinit(student, seed=cfg.seed)
        report = CompressionReport(
            family=family,
            method=method,
            teacher_hidden_dim=teacher_hidden,
            student_hidden_dim=student_hidden,
            indices_per_group={},
            seed=cfg.seed,
        )
        return student, report

    selector = _selector_from_method(method, cfg)
    axis_groups = build_axis_groups(family, teacher, architecture)
    calib_ctx, calib_data = _collect_calibration_if_needed(
        selector=selector,
        teacher=teacher,
        axis_groups=axis_groups,
        cfg=cfg,
    )

    teacher_state = teacher.state_dict()
    selections: dict[str, torch.LongTensor] = {}
    for group in axis_groups:
        rng = torch.Generator().manual_seed(_group_seed(cfg.seed, group.id))
        selections[group.id] = selector.select(
            axis_group=group,
            teacher_state=teacher_state,
            calib=calib_ctx,
            rng=rng,
        )

    student_state = student.state_dict()
    compressed_state = apply_axis_group_surgery(
        teacher_state=teacher_state,
        student_state=student_state,
        axis_groups=axis_groups,
        selections=selections,
    )
    student.load_state_dict(compressed_state, strict=True)

    calib_info = None
    if calib_data is not None:
        calib_info = {
            "seed": cfg.calibration.seed,
            "num_images": cfg.calibration.num_images,
            "indices_sha256": calib_data.indices_sha256,
        }
    report = CompressionReport(
        family=family,
        method=method,
        teacher_hidden_dim=teacher_hidden,
        student_hidden_dim=student_hidden,
        indices_per_group={k: v.tolist() for k, v in selections.items()},
        seed=cfg.seed,
        calibration=calib_info,
    )
    return student, report
