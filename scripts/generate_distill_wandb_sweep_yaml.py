#!/usr/bin/env python3
"""Generate a W&B sweep YAML for distillation evaluation.

The sweep is a grid over `methods`, so each run evaluates exactly one
compression method end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compression.eval.method_registry import resolve_method_specs


def _methods_values(methods_arg: str) -> list[str]:
    specs = resolve_method_specs(methods_arg)
    values = [s.run_method_name for s in specs]
    if not values:
        raise SystemExit("No methods resolved for sweep.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distillation W&B sweep YAML.")
    parser.add_argument("--output", required=True, help="Output sweep YAML path.")
    parser.add_argument("--methods", default="all", help="Method list or 'all'.")
    parser.add_argument("--project", default="model_compression", help="W&B project.")
    parser.add_argument("--entity", default="", help="Optional W&B entity.")
    parser.add_argument("--group", default="", help="W&B group for runs.")
    parser.add_argument("--sweep-name", default="", help="Sweep display name.")
    parser.add_argument("--wandb-run-prefix", default="d", help="Run name prefix.")

    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--teacher-train-fraction", type=float, default=0.8)
    parser.add_argument("--max-test-teachers", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distill-steps", type=int, default=200)
    parser.add_argument("--eval-steps", default="0,5,10,20,50,100,150,200")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size-train", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--calib-n", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=128)
    parser.add_argument("--tmux-session-name", default="")
    args = parser.parse_args()

    methods = _methods_values(args.methods)
    group = args.group or "distill_sweep"
    sweep_name = args.sweep_name or f"distill-grid-{len(methods)}m"

    command = [
        "${env}",
        "python3",
        "scripts/evaluate_compression_distillation.py",
        "--dataset-dir",
        args.dataset_dir,
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--split-seed",
        str(args.split_seed),
        "--teacher-train-fraction",
        str(args.teacher_train_fraction),
        "--max-test-teachers",
        str(args.max_test_teachers),
        "--temperature",
        str(args.temperature),
        "--distill-steps",
        str(args.distill_steps),
        "--eval-steps",
        args.eval_steps,
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size-train",
        str(args.batch_size_train),
        "--batch-size-test",
        str(args.batch_size_test),
        "--seed",
        str(args.seed),
        "--calib-seed",
        str(args.calib_seed),
        "--calib-n",
        str(args.calib_n),
        "--calib-batch-size",
        str(args.calib_batch_size),
        "--wandb-project",
        args.project,
        "--wandb-mode",
        "online",
        "--wandb-group",
        group,
        "--wandb-run-prefix",
        args.wandb_run_prefix,
        "--tmux-session-name",
        args.tmux_session_name,
        "${args}",
    ]
    if args.entity:
        command.extend(["--wandb-entity", args.entity])

    sweep_cfg: dict[str, object] = {
        "name": sweep_name,
        "project": args.project,
        "method": "grid",
        "metric": {"name": "test_kl", "goal": "minimize"},
        "parameters": {
            "methods": {"values": methods},
        },
        "command": command,
    }
    if args.entity:
        sweep_cfg["entity"] = args.entity

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(sweep_cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out} with {len(methods)} method runs.")


if __name__ == "__main__":
    main()
