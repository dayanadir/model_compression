"""Run compression baselines on a single model directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from compression.api import CalibrationConfig, CompressionConfig, compress
from compression.cli.common import load_model_bundle, num_params, resolve_methods
from compression.eval.cifar10 import evaluate_cifar10_accuracy


def run_one_model(
    model_dir: str | Path,
    *,
    methods: list[str],
    seed: int,
    calib_seed: int,
    calib_n: int,
    calib_batch_size: int,
    device: str,
    data_dir: str,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    metadata, teacher = load_model_bundle(model_path, map_location="cpu")
    teacher.eval()

    family = metadata["family"]
    arch = metadata["architecture"]
    teacher_params = num_params(teacher)
    teacher_acc = metadata.get("results", {}).get("test_acc")
    if teacher_acc is None:
        teacher_acc = evaluate_cifar10_accuracy(
            teacher,
            device=device,
            data_dir=data_dir,
        ).accuracy
    teacher_acc = float(teacher_acc)

    out_methods: dict[str, Any] = {}
    for method in methods:
        cfg = CompressionConfig(
            seed=seed,
            device=device,
            calibration=CalibrationConfig(
                data_dir=data_dir,
                num_images=calib_n,
                batch_size=calib_batch_size,
                seed=calib_seed,
            ),
            include_affine_gamma=False,
        )
        student, report = compress(
            teacher=teacher,
            family=family,
            architecture=arch,
            method=method,
            cfg=cfg,
        )
        student_acc = evaluate_cifar10_accuracy(
            student,
            device=device,
            data_dir=data_dir,
        ).accuracy
        payload: dict[str, Any] = {
            "test_acc": float(student_acc),
            "seed": report.seed,
        }
        if report.indices_per_group:
            payload["indices_per_group"] = report.indices_per_group
        if report.calibration is not None:
            payload["calibration"] = report.calibration
        out_methods[method] = payload

    # params reduction is architecture-dependent, not method-dependent (except same for all here).
    any_method = methods[0]
    sample_student, _ = compress(
        teacher=teacher,
        family=family,
        architecture=arch,
        method=any_method if any_method != "he_reinit" else "uniform",
        cfg=CompressionConfig(seed=seed, device=device),
    )
    student_params = num_params(sample_student)
    reduction = 100.0 * (1.0 - (student_params / teacher_params))

    result = {
        "model_id": metadata["model_id"],
        "family": family,
        "teacher_H": int(arch["hidden_dim"]),
        "student_H": int(arch["hidden_dim"]) // 2,
        "teacher_num_params": teacher_params,
        "student_num_params": student_params,
        "teacher_test_acc": teacher_acc,
        "methods": out_methods,
        "params_reduction_pct": reduction,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to dataset/model_xxxxxx")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--calib-n", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    methods = resolve_methods(args.methods)
    result = run_one_model(
        args.model,
        methods=methods,
        seed=args.seed,
        calib_seed=args.calib_seed,
        calib_n=args.calib_n,
        calib_batch_size=args.calib_batch_size,
        device=args.device,
        data_dir=args.data_dir,
    )
    out_path = Path(args.model) / "compression_result.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"written": str(out_path), "methods": methods}))


if __name__ == "__main__":
    main()
