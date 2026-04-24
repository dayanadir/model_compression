#!/usr/bin/env python3
"""Evaluate compression + distillation KL trajectories on CIFAR-10 teachers.

Examples:
  python scripts/evaluate_compression_distillation.py --dataset-dir dataset --methods magnitude --wandb-mode disabled --json-out results/distill_smoke.json
  python scripts/evaluate_compression_distillation.py --dataset-dir dataset --methods activation --wandb-mode offline --wandb-project compression-distill --json-out results/distill_activation_offline.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compression.api import CalibrationConfig, CompressionConfig, compress
from compression.cli.common import load_model_bundle, num_params
from compression.eval.distillation import (
    build_cifar10_test_loader_cached,
    build_cifar10_train_loader,
    distill_with_eval_checkpoints,
    model_normalized_l1,
    normalized_l1_bucket,
    param_size_bucket,
    parse_eval_steps,
    seed_everything,
    split_teacher_model_ids,
)
from compression.eval.method_registry import MethodSpec, resolve_method_specs

_WANDB_RUN_NAME_MAX = 128


def _update_running_stats(state: dict[str, float], value: float) -> tuple[float, float]:
    """Update running stats and return (mean, std)."""
    n = int(state.get("n", 0)) + 1
    total = float(state.get("sum", 0.0)) + value
    total_sq = float(state.get("sum_sq", 0.0)) + (value * value)
    state["n"] = float(n)
    state["sum"] = total
    state["sum_sq"] = total_sq
    mean = total / float(n)
    var = max(0.0, (total_sq / float(n)) - (mean * mean))
    return mean, float(np.sqrt(var))


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "count": int(arr.shape[0]),
    }


def _list_model_dirs(dataset_dir: Path) -> dict[str, Path]:
    model_dirs: dict[str, Path] = {}
    for path in sorted(dataset_dir.glob("model_*")):
        if path.is_dir():
            model_dirs[path.name] = path
    return model_dirs


def _make_cfg(args: argparse.Namespace, spec: MethodSpec) -> CompressionConfig:
    calib_n = int(spec.calibration_overrides.get("calib_n", args.calib_n))
    calib_batch = int(spec.calibration_overrides.get("calib_batch_size", args.calib_batch_size))
    calib_seed = int(spec.calibration_overrides.get("calib_seed", args.calib_seed))
    include_affine_gamma = (
        spec.include_affine_gamma
        if spec.include_affine_gamma is not None
        else False
    )
    return CompressionConfig(
        seed=args.seed,
        device=args.device,
        calibration=CalibrationConfig(
            data_dir=args.data_dir,
            num_images=calib_n,
            batch_size=calib_batch,
            seed=calib_seed,
            num_workers=args.num_workers,
        ),
        include_affine_gamma=include_affine_gamma,
    )


def _flatten_for_summary(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_for_summary(full_key, value))
        else:
            out[full_key] = value
    return out


def _aggregate_method_stats(
    teacher_results: list[dict[str, Any]],
    method_names: list[str],
    eval_steps: list[int],
) -> dict[str, dict[str, dict[str, float | int]]]:
    per_method: dict[str, dict[str, dict[str, float | int]]] = {}
    for method_name in method_names:
        steps: dict[str, dict[str, float | int]] = {}
        for step in eval_steps:
            vals: list[float] = []
            key = str(step)
            for t in teacher_results:
                method_payload = t["methods"].get(method_name)
                if method_payload is None:
                    continue
                if key in method_payload["test_kl_by_step"]:
                    vals.append(float(method_payload["test_kl_by_step"][key]))
            steps[key] = _stats(vals)
        per_method[method_name] = steps
    return per_method


def _aggregate_breakdown(
    teacher_results: list[dict[str, Any]],
    breakdown_field: str,
    method_names: list[str],
    eval_steps: list[int],
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in teacher_results:
        key = str(item["teacher_metadata"][breakdown_field])
        grouped.setdefault(key, []).append(item)

    out: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for key, subset in grouped.items():
        out[key] = _aggregate_method_stats(subset, method_names, eval_steps)
    return out


def _make_wandb_run_name(args: argparse.Namespace, method_specs: list[MethodSpec]) -> str:
    """Build a concise W&B run name (single-method preferred)."""
    method_part = method_specs[0].run_method_name if len(method_specs) == 1 else "multi"
    raw_prefix = (args.wandb_run_prefix or "d").strip()
    prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_prefix).strip("_") or "distill"
    name = f"{prefix}-{method_part}-s{args.seed}-sp{args.split_seed}-d{args.distill_steps}"
    return name[:_WANDB_RUN_NAME_MAX]


def _init_wandb(args: argparse.Namespace, method_specs: list[MethodSpec], split_info: dict[str, Any]):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed. Install with: python3 -m pip install --user wandb"
        ) from exc

    run_name = _make_wandb_run_name(args, method_specs)
    group = (
        args.wandb_group
        or (args.tmux_session_name if args.tmux_session_name else None)
        or f"distill-split{args.split_seed}"
    )
    config_payload = {
        "dataset_dir": args.dataset_dir,
        "data_dir": args.data_dir,
        "device": args.device,
        "methods": [m.to_config() for m in method_specs],
        "split": split_info,
        "temperature": args.temperature,
        "distill_steps": args.distill_steps,
        "eval_steps": args.eval_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size_train": args.batch_size_train,
        "batch_size_test": args.batch_size_test,
        "seed": args.seed,
        "calib_seed": args.calib_seed,
        "calib_n": args.calib_n,
        "calib_batch_size": args.calib_batch_size,
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=(args.wandb_entity or None),
        group=group,
        name=run_name,
        mode=args.wandb_mode,
        config=config_payload,
        tags=(
            ["compression", "distillation", "cifar10", method_specs[0].compress_method]
            if len(method_specs) == 1
            else ["compression", "distillation", "cifar10"]
        ),
        job_type="compression_distillation_eval",
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate compression + distillation KL on CIFAR-10 teacher models."
    )
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--teacher-train-fraction", type=float, default=0.8)
    parser.add_argument(
        "--max-test-teachers",
        type=int,
        default=0,
        help="Optional cap on number of test teachers to evaluate (0 means all).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distill-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=str, default="0,5,10,20,50,100,150,200")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size-train", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--calib-n", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=128)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="compression-distill")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument(
        "--wandb-run-prefix",
        type=str,
        default="",
        help="Short prefix for W&B run names (method + hyperparams are appended automatically).",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="",
        help="Override W&B group (default: --tmux-session-name, else distill-split<seed>).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--tmux-session-name", type=str, default="")
    parser.add_argument(
        "--test-kl-ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor for live test_kl moving mean logging in W&B.",
    )
    args = parser.parse_args()

    if args.distill_steps < 0:
        raise SystemExit("--distill-steps must be non-negative")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if not 0.0 < args.test_kl_ema_alpha <= 1.0:
        raise SystemExit("--test-kl-ema-alpha must be in (0, 1].")

    seed_everything(args.seed)
    eval_steps = parse_eval_steps(args.eval_steps, args.distill_steps)
    device = torch.device(args.device)
    dataset_dir = Path(args.dataset_dir)
    model_dirs = _list_model_dirs(dataset_dir)
    model_ids = sorted(model_dirs.keys())
    if not model_ids:
        raise SystemExit(f"No model_* directories found in {dataset_dir}")

    split = split_teacher_model_ids(
        model_ids=model_ids,
        teacher_train_fraction=args.teacher_train_fraction,
        split_seed=args.split_seed,
    )
    split_info = {
        "split_seed": split.split_seed,
        "teacher_train_fraction": split.teacher_train_fraction,
        "train_model_ids": split.train_model_ids,
        "test_model_ids": split.test_model_ids,
        "num_train_teachers": len(split.train_model_ids),
        "num_test_teachers": len(split.test_model_ids),
    }
    if args.max_test_teachers > 0:
        split_info["test_model_ids"] = split_info["test_model_ids"][: args.max_test_teachers]
        split_info["num_test_teachers"] = len(split_info["test_model_ids"])
    method_specs = resolve_method_specs(args.methods)
    method_names = [spec.run_method_name for spec in method_specs]

    train_loader = build_cifar10_train_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size_train,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = build_cifar10_test_loader_cached(
        data_dir=args.data_dir,
        batch_size=args.batch_size_test,
        num_workers=args.num_workers,
    )

    wandb_run = _init_wandb(args=args, method_specs=method_specs, split_info=split_info)

    teacher_results: list[dict[str, Any]] = []
    ema_by_method_step: dict[tuple[str, int], float] = {}
    running_stats_by_method_step: dict[tuple[str, int], dict[str, float]] = {}
    total_pairs = len(split_info["test_model_ids"]) * len(method_specs)
    completed_pairs = 0
    run_start_time = time.time()
    for teacher_index, model_id in enumerate(split_info["test_model_ids"]):
        model_dir = model_dirs[model_id]
        metadata, teacher = load_model_bundle(model_dir=model_dir, map_location="cpu")
        family = str(metadata["family"])
        architecture = metadata["architecture"]
        teacher.eval()
        teacher_params = num_params(teacher)
        teacher_norm_l1 = model_normalized_l1(teacher)
        teacher_payload: dict[str, Any] = {
            "model_id": model_id,
            "teacher_metadata": {
                "family": family,
                "num_params": teacher_params,
                "normalized_l1": teacher_norm_l1,
                "size_bucket": param_size_bucket(teacher_params),
                "normalized_l1_bucket": normalized_l1_bucket(teacher_norm_l1),
                "raw_metadata": metadata,
            },
            "methods": {},
        }

        for spec in method_specs:
            cfg = _make_cfg(args, spec)
            student, report = compress(
                teacher=teacher,
                family=family,
                architecture=architecture,
                method=spec.compress_method,
                cfg=cfg,
            )
            train_log_steps = [s for s in eval_steps if s > 0]
            train_kl_by_step, test_kl_by_step = distill_with_eval_checkpoints(
                teacher=teacher,
                student=student,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                distill_steps=args.distill_steps,
                eval_steps=eval_steps,
                temperature=args.temperature,
                lr=args.lr,
                weight_decay=args.weight_decay,
                train_log_steps=train_log_steps,
            )
            record = {
                "compress_method": spec.compress_method,
                "variant_params": spec.to_config(),
                "compression_report": {
                    "family": report.family,
                    "method": report.method,
                    "teacher_hidden_dim": report.teacher_hidden_dim,
                    "student_hidden_dim": report.student_hidden_dim,
                    "indices_per_group": report.indices_per_group,
                    "seed": report.seed,
                    "calibration": report.calibration,
                },
                "pre_distill_test_kl": float(test_kl_by_step[0]),
                "test_kl_by_step": {str(k): float(v) for k, v in sorted(test_kl_by_step.items())},
                "train_kl_by_step": {str(k): float(v) for k, v in sorted(train_kl_by_step.items())},
            }
            teacher_payload["methods"][spec.run_method_name] = record

            if wandb_run is not None:
                import wandb

                for step, value in sorted(train_kl_by_step.items()):
                    wandb.log(
                        {
                            "teacher_model_id": model_id,
                            "teacher_family": family,
                            "teacher_num_params": teacher_params,
                            "teacher_size_bucket": teacher_payload["teacher_metadata"]["size_bucket"],
                            "teacher_normalized_l1_bucket": teacher_payload["teacher_metadata"]["normalized_l1_bucket"],
                            "method": spec.run_method_name,
                            "distill_step": int(step),
                            "train_distill_kl": float(value),
                        }
                    )
                for step, value in sorted(test_kl_by_step.items()):
                    method_key = (spec.run_method_name, int(step))
                    prev_ema = ema_by_method_step.get(method_key, float(value))
                    ema = (
                        args.test_kl_ema_alpha * float(value)
                        + (1.0 - args.test_kl_ema_alpha) * prev_ema
                    )
                    ema_by_method_step[method_key] = float(ema)
                    state = running_stats_by_method_step.setdefault(method_key, {})
                    running_mean, running_std = _update_running_stats(
                        state=state,
                        value=float(value),
                    )
                    wandb.log(
                        {
                            "teacher_model_id": model_id,
                            "teacher_family": family,
                            "teacher_num_params": teacher_params,
                            "teacher_size_bucket": teacher_payload["teacher_metadata"]["size_bucket"],
                            "teacher_normalized_l1_bucket": teacher_payload["teacher_metadata"]["normalized_l1_bucket"],
                            "method": spec.run_method_name,
                            "distill_step": int(step),
                            "test_kl": float(value),
                            "test_kl_ema": float(ema),
                            "test_kl_running_mean": float(running_mean),
                            "test_kl_running_std": float(running_std),
                            "is_pre_distill": int(step == 0),
                        }
                    )
                completed_pairs += 1
                elapsed = max(1e-6, time.time() - run_start_time)
                avg_pair_seconds = elapsed / float(completed_pairs)
                remaining_pairs = max(0, total_pairs - completed_pairs)
                eta_seconds = avg_pair_seconds * remaining_pairs
                wandb.log(
                    {
                        "progress/completed_pairs": completed_pairs,
                        "progress/total_pairs": total_pairs,
                        "progress/percent": 100.0 * completed_pairs / float(total_pairs),
                        "progress/remaining_pairs": remaining_pairs,
                        "progress/elapsed_minutes": elapsed / 60.0,
                        "progress/eta_minutes": eta_seconds / 60.0,
                        "progress/teacher_index": teacher_index + 1,
                        "progress/teacher_total": len(split_info["test_model_ids"]),
                    }
                )
            else:
                completed_pairs += 1

            student.cpu()
            del student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        teacher.cpu()
        teacher_results.append(teacher_payload)
        print(
            f"[{teacher_index + 1}/{len(split_info['test_model_ids'])}] "
            f"processed {model_id} ({family})"
        )

    summary_per_method = _aggregate_method_stats(
        teacher_results=teacher_results,
        method_names=method_names,
        eval_steps=eval_steps,
    )
    breakdowns = {
        "family": _aggregate_breakdown(
            teacher_results=teacher_results,
            breakdown_field="family",
            method_names=method_names,
            eval_steps=eval_steps,
        ),
        "size_bucket": _aggregate_breakdown(
            teacher_results=teacher_results,
            breakdown_field="size_bucket",
            method_names=method_names,
            eval_steps=eval_steps,
        ),
        "normalized_l1_bucket": _aggregate_breakdown(
            teacher_results=teacher_results,
            breakdown_field="normalized_l1_bucket",
            method_names=method_names,
            eval_steps=eval_steps,
        ),
    }

    result_payload = {
        "run_config": {
            "dataset_dir": args.dataset_dir,
            "methods_arg": args.methods,
            "resolved_method_specs": [spec.to_config() for spec in method_specs],
            "device": args.device,
            "data_dir": args.data_dir,
            "num_workers": args.num_workers,
            "split_seed": args.split_seed,
            "teacher_train_fraction": args.teacher_train_fraction,
            "max_test_teachers": args.max_test_teachers,
            "temperature": args.temperature,
            "distill_steps": args.distill_steps,
            "eval_steps": eval_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size_train": args.batch_size_train,
            "batch_size_test": args.batch_size_test,
            "seed": args.seed,
            "calib_seed": args.calib_seed,
            "calib_n": args.calib_n,
            "calib_batch_size": args.calib_batch_size,
            "test_kl_ema_alpha": args.test_kl_ema_alpha,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_run_prefix": args.wandb_run_prefix,
            "wandb_group": args.wandb_group,
            "wandb_mode": args.wandb_mode,
            "gpu_ids": args.gpu_ids,
            "tmux_session_name": args.tmux_session_name,
        },
        "split": split_info,
        "test_teachers": teacher_results,
        "summary": {
            "per_method": summary_per_method,
            "breakdowns": breakdowns,
        },
    }

    if wandb_run is not None:
        summary_flat = _flatten_for_summary("", result_payload["summary"])
        wandb_run.summary.update(summary_flat)
        wandb_run.summary.update(
            {
                "num_test_teachers": len(split_info["test_model_ids"]),
                "num_methods": len(method_specs),
            }
        )
        wandb_run.finish()

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)
        print(f"Wrote {out_path}")
    else:
        print(
            json.dumps(
                {"num_test_teachers": len(split_info["test_model_ids"]), "methods": method_names}
            )
        )


if __name__ == "__main__":
    main()
