"""DatasetBuilder: orchestrates the full model zoo generation pipeline."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone

import torch
import torch.nn as nn

from model_zoo.config import RunConfig
from model_zoo.families.base import ModelFamily
from model_zoo.registry import get_family_cls
from model_zoo.storage.index import DatasetIndex
from model_zoo.storage.metadata import MetadataBuilder
from model_zoo.storage.writer import ModelArtifactWriter
from model_zoo.training.data import build_dataloaders
from model_zoo.training.trainer import Trainer

# Ensure all families are registered
import model_zoo.families  # noqa: F401

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Top-level orchestrator that generates the full model zoo dataset.

    For each model:
      1. Compute deterministic model_id and seed
      2. Skip if already completed (resumability)
      3. Sample architecture + training hyperparameters
      4. Build nn.Sequential model
      5. Train on shared DataLoaders
      6. Evaluate on val and test sets
      7. Persist weights.pt + metadata.json
      8. Mark as completed
    """

    def __init__(
        self, config: RunConfig, shard_rank: int = 0, num_shards: int = 1
    ) -> None:
        self.config = config
        self.shard_rank = shard_rank
        self.num_shards = num_shards
        if self.num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        if not (0 <= self.shard_rank < self.num_shards):
            raise ValueError(
                f"shard_rank must be in [0, {self.num_shards}), got {self.shard_rank}"
            )
        self.writer = ModelArtifactWriter(config.output_dir)
        self.index = DatasetIndex(config.output_dir)
        self.device = torch.device(config.device)
        self.trainer = Trainer(
            device=self.device,
            num_classes=config.dataset.num_classes,
        )

        # Instantiate family objects
        self.families: dict[str, ModelFamily] = {}
        for name, fcfg in config.families.items():
            cls = get_family_cls(name)
            self.families[name] = cls(
                search_space=fcfg.search_space,
                training_space=config.training_defaults,
                dataset_info=config.dataset,
            )

    @staticmethod
    def _make_model_id(counter: int) -> str:
        return f"model_{counter:06d}"

    def _total_models(self) -> int:
        return sum(fcfg.count for fcfg in self.config.families.values())

    def _build_loaders(self):
        logger.info("Building dataloaders...")
        return build_dataloaders(
            self.config.dataset,
            num_workers=self.config.num_workers,
            seed=self.config.base_seed,
        )

    def _input_output_shapes(self) -> tuple[list[int], list[int]]:
        input_shape = [
            self.config.dataset.in_channels,
            self.config.dataset.image_size,
            self.config.dataset.image_size,
        ]
        output_shape = [self.config.dataset.num_classes]
        return input_shape, output_shape

    def _run_one_model(
        self,
        *,
        model_index: int,
        total_models: int,
        family_name: str,
        family: ModelFamily,
        train_loader,
        val_loader,
        test_loader,
        input_shape: list[int],
        output_shape: list[int],
        allow_skip: bool = True,
    ) -> str:
        """Run exactly one model and return one of: completed/skipped/failed."""
        model_id = self._make_model_id(model_index)
        seed = self.config.base_seed + model_index
        progress_n = model_index + 1

        if allow_skip and self.index.is_complete(model_id):
            logger.info("[%d/%d] %s already complete -> skip", progress_n, total_models, model_id)
            return "skipped"

        # --- Sample hyperparameters (deterministic) ---
        rng = random.Random(seed)
        arch_hp = family.sample_arch_hyperparams(rng)
        train_hp = family.sample_training_hyperparams(rng)

        logger.info(
            "[%d/%d] %s | family=%s | arch=%s",
            progress_n,
            total_models,
            model_id,
            family_name,
            arch_hp,
        )

        # --- Build model ---
        try:
            model = family.build_model(arch_hp)
            assert isinstance(model, nn.Sequential), (
                f"build_model must return nn.Sequential, got {type(model)}"
            )
        except Exception as e:
            logger.error("Failed to build %s: %s", model_id, e)
            self.index.mark_failed(model_id)
            return "failed"

        # --- Train ---
        started_at = datetime.now(timezone.utc).isoformat()
        result = self.trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            hparams=train_hp,
            seed=seed,
        )

        # --- Build metadata ---
        metadata = MetadataBuilder.build(
            model_id=model_id,
            family_name=family_name,
            arch_hparams=arch_hp,
            training_hparams=train_hp,
            train_result=result,
            model=model,
            seed=seed,
            dataset_name=self.config.dataset.name,
            input_shape=input_shape,
            output_shape=output_shape,
            training_started_at=started_at,
        )

        # --- Persist artifacts ---
        if result.completed:
            self.writer.save(model_id, model, metadata)
            self.index.mark_complete(model_id)
            logger.info(
                "  -> completed | test_acc=%.4f | val_acc=%.4f | params=%d | %.1fs",
                result.test_acc,
                result.val_acc,
                metadata["model_info"]["num_params"],
                result.wall_time_seconds,
            )
            return "completed"

        self.writer.save_failure(model_id, metadata)
        self.index.mark_failed(model_id)
        logger.warning("  -> FAILED: %s", result.error)
        return "failed"

    def run(self) -> None:
        """Generate the entire model zoo dataset."""
        train_loader, val_loader, test_loader = self._build_loaders()

        # Compute total models for progress reporting
        total_models = self._total_models()
        logger.info(
            "Starting generation: %d models across %d families",
            total_models,
            len(self.config.families),
        )
        assigned_models = sum(
            1 for idx in range(total_models) if idx % self.num_shards == self.shard_rank
        )
        logger.info(
            "Shard %d/%d running on device=%s: assigned %d models",
            self.shard_rank + 1,
            self.num_shards,
            self.device,
            assigned_models,
        )

        global_counter = 0
        completed_count = 0
        skipped_count = 0

        input_shape, output_shape = self._input_output_shapes()

        for family_name, fcfg in self.config.families.items():
            family = self.families[family_name]
            logger.info(
                "Family '%s': generating %d models (starting at counter %d)",
                family_name,
                fcfg.count,
                global_counter,
            )

            for i in range(fcfg.count):
                model_index = global_counter
                assigned_to_this_shard = (
                    global_counter % self.num_shards == self.shard_rank
                )
                global_counter += 1
                if not assigned_to_this_shard:
                    continue

                status = self._run_one_model(
                    model_index=model_index,
                    total_models=total_models,
                    family_name=family_name,
                    family=family,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    allow_skip=True,
                )
                if status == "completed":
                    completed_count += 1
                elif status == "skipped":
                    skipped_count += 1

        logger.info(
            "Done. completed=%d, skipped=%d, failed=%d",
            completed_count,
            skipped_count,
            self.index.num_failed,
        )

    def run_single_model(self, model_index: int, allow_skip: bool = False) -> str:
        """Run a specific model index and return completed/skipped/failed."""
        if model_index < 0:
            raise ValueError("model_index must be >= 0")

        total_models = self._total_models()
        if model_index >= total_models:
            raise ValueError(
                f"model_index={model_index} is out of range [0, {total_models - 1}]"
            )

        train_loader, val_loader, test_loader = self._build_loaders()
        input_shape, output_shape = self._input_output_shapes()

        offset = 0
        for family_name, fcfg in self.config.families.items():
            if model_index < offset + fcfg.count:
                family = self.families[family_name]
                return self._run_one_model(
                    model_index=model_index,
                    total_models=total_models,
                    family_name=family_name,
                    family=family,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    allow_skip=allow_skip,
                )
            offset += fcfg.count

        # Should be unreachable due to range check above
        raise RuntimeError("Failed to resolve family for model_index")
