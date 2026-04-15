"""Vision Transformer (ViT) architecture family."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch.nn as nn

from gmn.graph_construct.net_makers import make_transformer
from model_zoo.config import sample_from_spec
from model_zoo.families.base import ArchHyperparams, ModelFamily
from model_zoo.registry import register_family


@dataclass(frozen=True)
class ViTArchHyperparams(ArchHyperparams):
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    patch_size: int


@register_family
class ViTFamily(ModelFamily):
    family_name = "vit"

    def sample_arch_hyperparams(self, rng: random.Random) -> ViTArchHyperparams:
        hidden_dim = rng.choice(self._space["hidden_dim"])
        # num_heads must divide hidden_dim
        valid_heads = [
            h for h in self._space["num_heads"] if hidden_dim % h == 0
        ]
        num_heads = rng.choice(valid_heads)
        return ViTArchHyperparams(
            hidden_dim=hidden_dim,
            num_layers=rng.choice(self._space["num_layers"]),
            num_heads=num_heads,
            dropout=sample_from_spec(rng, self._space["dropout"]),
            patch_size=rng.choice(self._space["patch_size"]),
        )

    def build_model(self, hparams: ViTArchHyperparams) -> nn.Sequential:
        return make_transformer(
            in_dim=self._dataset_info.in_channels,
            hidden_dim=hparams.hidden_dim,
            num_heads=hparams.num_heads,
            out_dim=self._dataset_info.num_classes,
            dropout=hparams.dropout,
            num_layers=hparams.num_layers,
            vit=True,
            patch_size=hparams.patch_size,
        )
