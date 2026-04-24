#!/usr/bin/env python3
"""Run evaluate_cifar10_accuracy for each compression baseline on one zoo model.

Example:
  python scripts/evaluate_compression_methods.py --model dataset/model_000000 --device cuda
  python scripts/evaluate_compression_methods.py --model dataset/model_000000 --methods uniform,he_reinit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Repo root on sys.path for `compression` and `model_zoo`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compression.api import CalibrationConfig, CompressionConfig, compress
from compression.cli.common import ALL_METHODS, load_model_bundle, resolve_methods
from compression.eval.cifar10 import evaluate_cifar10_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR-10 test accuracy for each compression method."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a dataset directory (e.g. dataset/model_000000)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help=f"Comma-separated methods or 'all'. Choices: {', '.join(ALL_METHODS)}",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--calib-n", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=128)
    parser.add_argument(
        "--eval-teacher",
        action="store_true",
        help="Also run evaluate_cifar10_accuracy on the teacher (full width H).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="If set, write results to this JSON file.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    methods = resolve_methods(args.methods)

    metadata, teacher = load_model_bundle(model_dir, map_location="cpu")
    family = metadata["family"]
    arch = metadata["architecture"]
    device = args.device

    cfg = CompressionConfig(
        seed=args.seed,
        device=device,
        calibration=CalibrationConfig(
            data_dir=args.data_dir,
            num_images=args.calib_n,
            batch_size=args.calib_batch_size,
            seed=args.calib_seed,
            num_workers=args.num_workers,
        ),
    )

    results: dict[str, object] = {
        "model_id": metadata.get("model_id", model_dir.name),
        "family": family,
        "methods_requested": methods,
    }

    if args.eval_teacher:
        teacher.to(device)
        teacher.eval()
        t_eval = evaluate_cifar10_accuracy(
            teacher,
            device=device,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        results["teacher"] = {
            "test_loss": t_eval.loss,
            "test_accuracy": t_eval.accuracy,
        }
        teacher.cpu()

    per_method: dict[str, dict[str, float]] = {}
    for method in methods:
        student, _report = compress(
            teacher=teacher,
            family=family,
            architecture=arch,
            method=method,
            cfg=cfg,
        )
        student.to(device)
        student.eval()
        ev = evaluate_cifar10_accuracy(
            student,
            device=device,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        per_method[method] = {
            "test_loss": ev.loss,
            "test_accuracy": ev.accuracy,
        }
        student.cpu()
        del student
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results["compression"] = per_method

    # Human-readable table
    print(f"model: {results['model_id']}  family: {family}")
    if "teacher" in results:
        print(
            f"  teacher:  acc={results['teacher']['test_accuracy']:.4f}  "
            f"loss={results['teacher']['test_loss']:.4f}"
        )
    print("method".ljust(22), "test_acc", "test_loss", sep="\t")
    for method in methods:
        row = per_method[method]
        print(
            method.ljust(22),
            f"{row['test_accuracy']:.4f}",
            f"{row['test_loss']:.4f}",
            sep="\t",
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
