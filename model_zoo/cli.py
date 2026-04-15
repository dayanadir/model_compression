"""Command-line entry point for the model zoo generator."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a diverse model zoo dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--shard-rank",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wandb-sweep-trial",
        action="store_true",
        help=(
            "Run one model as a W&B sweep trial (balanced sweeps: family + family_slot; "
            "legacy sweeps: model_index; or pass --model-index)."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="model_compression",
        help="W&B project name (default: model_compression).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group.",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=None,
        help="Optional explicit model index override (for debug/manual runs).",
    )
    parser.add_argument(
        "--wandb-log-artifact",
        action="store_true",
        help="If set, upload model artifacts to W&B.",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        type=str,
        default=None,
        help="Short prefix for W&B run name (default: derived from config filename).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from model_zoo.config import RunConfig
    from model_zoo.pipeline import DatasetBuilder

    if args.wandb_sweep_trial:
        from model_zoo.wandb_sweep import run_wandb_sweep_trial

        run_wandb_sweep_trial(
            config_path=args.config,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            model_index_override=args.model_index,
            log_artifact=args.wandb_log_artifact,
            run_name_prefix=args.wandb_run_prefix,
        )
        return

    config = RunConfig.from_yaml(args.config)
    is_parent_launcher = args.shard_rank == 0 and args.num_shards == 1
    if is_parent_launcher and len(config.gpu_ids) > 1:
        children: list[subprocess.Popen] = []
        for shard_rank, gpu_id in enumerate(config.gpu_ids):
            child_cmd = [
                sys.executable,
                "-m",
                "model_zoo",
                "--config",
                args.config,
                "--log-level",
                args.log_level,
                "--shard-rank",
                str(shard_rank),
                "--num-shards",
                str(len(config.gpu_ids)),
            ]
            child_env = os.environ.copy()
            # Each child process sees only one GPU; inside the process this is cuda:0.
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logging.info(
                "Launching shard %d/%d on physical GPU %s",
                shard_rank + 1,
                len(config.gpu_ids),
                gpu_id,
            )
            children.append(subprocess.Popen(child_cmd, env=child_env))

        nonzero = 0
        for p in children:
            code = p.wait()
            if code != 0:
                nonzero += 1
        if nonzero:
            raise SystemExit(f"{nonzero} shard process(es) failed")
        return

    builder = DatasetBuilder(
        config,
        shard_rank=args.shard_rank,
        num_shards=args.num_shards,
    )
    builder.run()


if __name__ == "__main__":
    main()
