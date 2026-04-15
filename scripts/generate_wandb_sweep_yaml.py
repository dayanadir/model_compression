#!/usr/bin/env python3
"""Generate a W&B sweep YAML for the model zoo.

By default uses **interleaved** scheduling: a single ``interleave_step`` list ``0 … N-1``
so W&B walks trials in order ``(fam0,slot0), (fam1,slot0), …`` round-robin across
families (requires equal ``count`` per family). This fixes two-parameter grids where
W&B may iterate one Cartesian dimension for a long time.

Optional ``balanced`` mode keeps a ``family`` × ``family_slot`` grid (dispatch order
not guaranteed). ``legacy`` is sequential ``model_index`` blocks by YAML family order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from model_zoo.config import RunConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Model-zoo YAML config path.")
    parser.add_argument("--output", required=True, help="Output sweep YAML path.")
    parser.add_argument(
        "--project",
        default="model_compression",
        help="W&B project name (default: model_compression).",
    )
    parser.add_argument("--entity", default=None, help="Optional W&B entity/team.")
    parser.add_argument(
        "--group",
        default=None,
        help="W&B group (default: zoo-<config-stem>-<total_models>).",
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep display name (default: zoo-<config-stem>-<n>models).",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        default=None,
        help="Optional run name prefix passed to each trial (default: auto from config).",
    )
    parser.add_argument(
        "--schedule",
        choices=("interleaved", "balanced", "legacy"),
        default="interleaved",
        help=(
            "interleaved: single interleave_step 0..N-1 (round-robin families; equal counts). "
            "balanced: grid family × family_slot (equal counts; W&B may reorder). "
            "legacy: model_index 0..N-1 sequential by family blocks in YAML."
        ),
    )
    args = parser.parse_args()

    run_cfg = RunConfig.from_yaml(args.config)
    total_models = sum(f.count for f in run_cfg.families.values())
    cfg_stem = Path(args.config).stem
    slug = "".join(c if c.isalnum() else "" for c in cfg_stem.lower())[:10] or "zoo"
    group = args.group or f"zoo-{slug}-{total_models}"
    sweep_name = args.sweep_name or f"zoo-{slug}-{total_models}m-grid-{args.schedule}"

    if args.schedule == "interleaved":
        counts = [fc.count for fc in run_cfg.families.values()]
        if len(set(counts)) != 1:
            raise SystemExit(
                "Interleaved sweep requires every family to have the same `count`. "
                f"Got counts={dict(zip(run_cfg.families.keys(), counts))}. "
                "Use --schedule legacy or equalize family counts."
            )
        parameters = {
            "interleave_step": {"values": list(range(total_models))},
        }
    elif args.schedule == "balanced":
        counts = [fc.count for fc in run_cfg.families.values()]
        if len(set(counts)) != 1:
            raise SystemExit(
                "Balanced sweep requires every family to have the same `count`. "
                f"Got counts={dict(zip(run_cfg.families.keys(), counts))}. "
                "Use --schedule legacy or equalize family counts."
            )
        slot_max = counts[0]
        if slot_max * len(run_cfg.families) != total_models:
            raise SystemExit("Internal error: balanced grid size mismatch.")
        family_names = list(run_cfg.families.keys())
        parameters = {
            "family": {"values": family_names},
            "family_slot": {"values": list(range(slot_max))},
        }
    else:
        parameters = {
            "model_index": {
                "values": list(range(total_models)),
            }
        }

    command = [
        "${env}",
        "python3",
        "-m",
        "model_zoo",
        "--config",
        args.config,
        "--wandb-sweep-trial",
        "--wandb-project",
        args.project,
        "--wandb-group",
        group,
    ]
    if args.entity:
        command.extend(["--wandb-entity", args.entity])
    if args.wandb_run_prefix:
        command.extend(["--wandb-run-prefix", args.wandb_run_prefix])

    sweep_cfg = {
        "name": sweep_name,
        "project": args.project,
        "method": "grid",
        "metric": {"name": "test_acc", "goal": "maximize"},
        "parameters": parameters,
        "command": command,
    }
    if args.entity:
        sweep_cfg["entity"] = args.entity

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(sweep_cfg, sort_keys=False))
    if args.schedule == "interleaved":
        print(
            f"Wrote {out} (interleaved: {total_models} interleave_step values, "
            f"{len(run_cfg.families)} families round-robin)."
        )
    elif args.schedule == "balanced":
        print(
            f"Wrote {out} (balanced grid: {len(run_cfg.families)} families × "
            f"{slot_max} slots = {total_models} trials)."
        )
    else:
        print(f"Wrote {out} with {total_models} legacy model_index values.")


if __name__ == "__main__":
    main()
