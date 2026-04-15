"""W&B sweep support: run one model-zoo trial per W&B run."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from model_zoo.config import RunConfig
from model_zoo.pipeline import DatasetBuilder

logger = logging.getLogger(__name__)

# W&B run names should stay short (UI + limits).
_RUN_NAME_MAX = 48


def _run_name_prefix(config_path: str, explicit: str | None) -> str:
    if explicit:
        s = "".join(c if c.isalnum() else "" for c in explicit.lower())
        return s[:12] or "zoo"
    stem = Path(config_path).stem
    s = "".join(c if c.isalnum() else "" for c in stem.lower())
    return (s[:10] or "zoo")


def _resolve_model_index(
    *,
    config: RunConfig,
    model_index_override: int | None,
    wandb_config,
) -> tuple[int, str | None, dict]:
    """Return ``(model_index, sweep_mode, extra)``.

    ``extra`` holds sweep-specific fields for logging (e.g. interleave_step, family).
    """
    if model_index_override is not None:
        return int(model_index_override), None, {}

    wc = dict(wandb_config) if wandb_config is not None else {}

    if "interleave_step" in wc:
        step = int(wc["interleave_step"])
        family, slot, model_index = config.interleave_step_to_family_slot_index(step)
        return (
            model_index,
            "interleaved",
            {"interleave_step": step, "family": family, "family_slot": slot},
        )

    if "family" in wc and "family_slot" in wc:
        family = str(wc["family"])
        family_slot = int(wc["family_slot"])
        mi = config.model_index_for_family_slot(
            family=family, family_slot=family_slot
        )
        return mi, "balanced", {"family": family, "family_slot": family_slot}

    if "model_index" in wc:
        return int(wc["model_index"]), "legacy", {}

    raise ValueError(
        "W&B run config must contain 'interleave_step' (recommended), "
        "('family', 'family_slot'), legacy 'model_index', or pass --model-index."
    )


def run_wandb_sweep_trial(
    *,
    config_path: str,
    project: str,
    entity: str | None,
    group: str | None,
    model_index_override: int | None,
    log_artifact: bool,
    run_name_prefix: str | None = None,
) -> None:
    try:
        import wandb
    except ImportError as e:
        raise SystemExit(
            "wandb is not installed. Install dependencies with "
            "`python3 -m pip install --user -r requirements-model-zoo.txt`."
        ) from e

    config = RunConfig.from_yaml(config_path)
    total_models = sum(f.count for f in config.families.values())

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        tags=["dataset_generation"],
        job_type="model_zoo_trial",
        config={"config_path": config_path, "total_models": total_models},
    )
    assert run is not None

    model_index, sweep_mode, sweep_extra = _resolve_model_index(
        config=config,
        model_index_override=model_index_override,
        wandb_config=run.config,
    )
    extra_cfg: dict = {"model_index": model_index, **sweep_extra}
    run.config.update(extra_cfg, allow_val_change=True)
    prefix = _run_name_prefix(config_path, run_name_prefix)
    # Concise, stable run id for tables: e.g. cifar10def-m00042
    run.name = f"{prefix}-m{model_index:05d}"[:_RUN_NAME_MAX]
    if sweep_mode == "interleaved":
        logger.info(
            "W&B trial run_name=%s interleave_step=%s model_index=%d family=%s family_slot=%d",
            run.name,
            sweep_extra.get("interleave_step"),
            model_index,
            sweep_extra.get("family"),
            sweep_extra.get("family_slot"),
        )
    elif sweep_mode == "balanced":
        logger.info(
            "W&B trial run_name=%s model_index=%d family=%s family_slot=%d",
            run.name,
            model_index,
            sweep_extra.get("family"),
            sweep_extra.get("family_slot"),
        )
    else:
        logger.info("W&B trial run_name=%s model_index=%d", run.name, model_index)

    builder = DatasetBuilder(config=config, shard_rank=0, num_shards=1)
    status = builder.run_single_model(model_index=model_index, allow_skip=True)
    model_id = builder._make_model_id(model_index)
    model_dir = Path(config.output_dir) / model_id
    metadata_path = model_dir / "metadata.json"
    weights_path = model_dir / "weights.pt"

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        summary = metadata.get("summary", {})
        results = metadata.get("results", {})
        log_payload = {
            "status": 1 if status == "completed" else 0,
            "skipped": 1 if status == "skipped" else 0,
            "failed": 1 if status == "failed" else 0,
            "model_index": model_index,
            "model_num_params": summary.get("num_params"),
            "val_acc": results.get("val_acc"),
            "test_acc": results.get("test_acc"),
            "wall_time_seconds": results.get("wall_time_seconds"),
        }
        if sweep_mode in ("balanced", "interleaved"):
            log_payload["family"] = extra_cfg.get("family")
            log_payload["family_slot"] = extra_cfg.get("family_slot")
        if sweep_mode == "interleaved":
            log_payload["interleave_step"] = extra_cfg.get("interleave_step")
        wandb.log(log_payload)
        sum_payload = {
            "model_id": model_id,
            "family": summary.get("family"),
            "num_params": summary.get("num_params"),
            "val_acc": results.get("val_acc"),
            "test_acc": results.get("test_acc"),
            "status": status,
            "artifact_dir": str(model_dir),
        }
        if sweep_mode in ("balanced", "interleaved"):
            sum_payload["sweep_family"] = extra_cfg.get("family")
            sum_payload["sweep_family_slot"] = extra_cfg.get("family_slot")
        if sweep_mode == "interleaved":
            sum_payload["interleave_step"] = extra_cfg.get("interleave_step")
        run.summary.update(sum_payload)

        # Keep metadata visible for easy browsing in W&B.
        metadata_art = wandb.Artifact(
            name=f"{model_id}-metadata",
            type="model_zoo_metadata",
        )
        metadata_art.add_file(str(metadata_path))
        run.log_artifact(metadata_art)

        # Optional full weights upload (can be large for 30k models).
        if log_artifact and weights_path.exists():
            model_art = wandb.Artifact(
                name=f"{model_id}-weights",
                type="model_zoo_weights",
            )
            model_art.add_file(str(weights_path))
            run.log_artifact(model_art)
    else:
        run.summary.update(
            {
                "model_id": model_id,
                "status": status,
                "error": "metadata_not_found",
            }
        )

    if status == "failed":
        wandb.finish(exit_code=1)
        raise SystemExit(1)
    wandb.finish()
