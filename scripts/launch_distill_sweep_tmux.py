#!/usr/bin/env python3
"""Launch compression distillation evaluation runs in tmux.

Examples:
  python scripts/launch_distill_sweep_tmux.py start --dataset-dir dataset --methods all --gpu-ids 0,1,2,3,4,5 --tmux-session-name distill_sweep
  python scripts/launch_distill_sweep_tmux.py status --tmux-session-name distill_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compression.eval.method_registry import resolve_method_specs


def _run_tmux(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["tmux", *args], check=False, text=True, capture_output=True)


def _session_exists(session_name: str) -> bool:
    result = _run_tmux(["has-session", "-t", session_name])
    return result.returncode == 0


def _split_csv(csv: str) -> list[str]:
    vals = [x.strip() for x in csv.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"CSV argument cannot be empty: {csv!r}")
    return vals


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug or "method"


def _base_command_parts(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python3",
        "scripts/evaluate_compression_distillation.py",
        "--dataset-dir",
        args.dataset_dir,
        "--device",
        args.device,
        "--data-dir",
        args.data_dir,
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
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
        "--tmux-session-name",
        args.tmux_session_name,
    ]
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if getattr(args, "wandb_run_prefix", ""):
        cmd.extend(["--wandb-run-prefix", args.wandb_run_prefix])
    if getattr(args, "wandb_group", ""):
        cmd.extend(["--wandb-group", args.wandb_group])
    return cmd


def _build_window_command(
    args: argparse.Namespace,
    method_token: str,
    gpu_id: str,
    log_file: Path,
) -> str:
    cmd_parts = _base_command_parts(args)
    cmd_parts.extend(["--methods", method_token])
    if args.json_out:
        base = Path(args.json_out)
        out_name = f"{base.stem}_{_safe_slug(method_token)}{base.suffix or '.json'}"
        out_path = base.parent / out_name
        cmd_parts.extend(["--json-out", str(out_path)])
    command = " ".join(shlex.quote(part) for part in cmd_parts)
    shell_cmd = (
        f"cd {shlex.quote(str(_REPO_ROOT))} && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)} "
        f"OMP_NUM_THREADS=${{OMP_NUM_THREADS:-1}} "
        f"{command} 2>&1 | tee -a {shlex.quote(str(log_file))}"
    )
    return shell_cmd


def start(args: argparse.Namespace) -> None:
    if _session_exists(args.tmux_session_name):
        print(f"Session '{args.tmux_session_name}' already exists.")
        print(f"Attach with: tmux attach -t {args.tmux_session_name}")
        return

    method_specs = resolve_method_specs(args.methods)
    method_tokens = [m.run_method_name for m in method_specs]
    gpu_ids = _split_csv(args.gpu_ids)
    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    launched = 0
    for index, method in enumerate(method_tokens):
        gpu = gpu_ids[index % len(gpu_ids)]
        window_name = _safe_slug(method)[:30]
        log_file = log_dir / f"{args.tmux_session_name}_{window_name}.log"
        tmux_cmd = _build_window_command(args=args, method_token=method, gpu_id=gpu, log_file=log_file)
        if index == 0:
            result = _run_tmux(
                ["new-session", "-d", "-s", args.tmux_session_name, "-n", window_name, tmux_cmd]
            )
        else:
            result = _run_tmux(["new-window", "-t", args.tmux_session_name, "-n", window_name, tmux_cmd])
        if result.returncode != 0:
            raise SystemExit(
                f"Failed to launch method {method!r} on GPU {gpu}: {result.stderr.strip()}"
            )
        launched += 1
        print(f"launched method={method} gpu={gpu} log={log_file}")

    print(f"Started tmux session '{args.tmux_session_name}' with {launched} method runs.")
    print(f"Attach: tmux attach -t {args.tmux_session_name}")


def status(args: argparse.Namespace) -> None:
    if not _session_exists(args.tmux_session_name):
        print(f"Session '{args.tmux_session_name}' is not running.")
        return
    result = _run_tmux(
        ["list-windows", "-t", args.tmux_session_name, "-F", "window=#{window_name} active=#{window_active}"]
    )
    print(f"Session '{args.tmux_session_name}' is running.")
    print(result.stdout.strip())


def attach(args: argparse.Namespace) -> None:
    raise SystemExit(subprocess.call(["tmux", "attach", "-t", args.tmux_session_name]))


def logs(args: argparse.Namespace) -> None:
    log_dir = _REPO_ROOT / "logs"
    if args.log_window:
        path = log_dir / f"{args.tmux_session_name}_{_safe_slug(args.log_window)[:30]}.log"
    else:
        path = log_dir / f"{args.tmux_session_name}.log"
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")
    raise SystemExit(subprocess.call(["tail", "-f", str(path)]))


def stop(args: argparse.Namespace) -> None:
    if not _session_exists(args.tmux_session_name):
        print(f"Session '{args.tmux_session_name}' is not running.")
        return
    result = _run_tmux(["kill-session", "-t", args.tmux_session_name])
    if result.returncode != 0:
        raise SystemExit(f"Failed to stop session: {result.stderr.strip()}")
    print(f"Stopped session '{args.tmux_session_name}'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch distillation evaluation sweep in tmux.")
    sub = parser.add_subparsers(dest="action", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--tmux-session-name", type=str, default="distill_sweep")
        subparser.add_argument("--dataset-dir", type=str, default="dataset")
        subparser.add_argument("--methods", type=str, default="all")
        subparser.add_argument("--device", type=str, default="cuda")
        subparser.add_argument("--data-dir", type=str, default="./data")
        subparser.add_argument("--num-workers", type=int, default=8)
        subparser.add_argument("--split-seed", type=int, default=0)
        subparser.add_argument("--teacher-train-fraction", type=float, default=0.8)
        subparser.add_argument("--max-test-teachers", type=int, default=0)
        subparser.add_argument("--temperature", type=float, default=1.0)
        subparser.add_argument("--distill-steps", type=int, default=200)
        subparser.add_argument("--eval-steps", type=str, default="0,5,10,20,50,100,150,200")
        subparser.add_argument("--lr", type=float, default=1e-3)
        subparser.add_argument("--weight-decay", type=float, default=0.0)
        subparser.add_argument("--batch-size-train", type=int, default=128)
        subparser.add_argument("--batch-size-test", type=int, default=1024)
        subparser.add_argument("--seed", type=int, default=0)
        subparser.add_argument("--calib-seed", type=int, default=0)
        subparser.add_argument("--calib-n", type=int, default=512)
        subparser.add_argument("--calib-batch-size", type=int, default=128)
        subparser.add_argument("--json-out", type=str, default="")
        subparser.add_argument("--wandb-project", type=str, default="compression-distill")
        subparser.add_argument("--wandb-entity", type=str, default="")
        subparser.add_argument(
            "--wandb-run-prefix",
            type=str,
            default="",
            help="Passed to evaluate_compression_distillation for informative W&B run names.",
        )
        subparser.add_argument(
            "--wandb-group",
            type=str,
            default="",
            help="Passed to evaluate_compression_distillation as W&B group override.",
        )
        subparser.add_argument(
            "--wandb-mode",
            type=str,
            default="online",
            choices=("online", "offline", "disabled"),
        )
        subparser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5")

    add_common(sub.add_parser("start", help="Start a tmux session with method runs."))
    add_common(sub.add_parser("status", help="Show tmux session status."))
    add_common(sub.add_parser("attach", help="Attach to running tmux session."))
    logs_parser = sub.add_parser("logs", help="Tail a log file.")
    logs_parser.add_argument("--tmux-session-name", type=str, default="distill_sweep")
    logs_parser.add_argument("--log-window", type=str, default="")
    stop_parser = sub.add_parser("stop", help="Stop tmux session.")
    stop_parser.add_argument("--tmux-session-name", type=str, default="distill_sweep")
    return parser


def main() -> None:
    if _run_tmux(["-V"]).returncode != 0:
        raise SystemExit("tmux is not installed or not found in PATH.")

    parser = build_parser()
    args = parser.parse_args()
    if args.action == "start":
        start(args)
        return
    if args.action == "status":
        status(args)
        return
    if args.action == "attach":
        attach(args)
        return
    if args.action == "logs":
        logs(args)
        return
    if args.action == "stop":
        stop(args)
        return
    raise SystemExit(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()
