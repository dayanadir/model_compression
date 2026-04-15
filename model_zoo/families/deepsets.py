"""DeepSets architecture family."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch.nn as nn

from gmn.graph_construct.net_makers import make_deepsets
from model_zoo.config import sample_from_spec
from model_zoo.families.base import ArchHyperparams, ModelFamily
from model_zoo.registry import register_family


@dataclass(frozen=True)
class DeepSetsArchHyperparams(ArchHyperparams):
    hidden_dim: int
    equivariant_layers: int
    fc_layers: int
    norm: str
    dropout: float
    activation: str


@register_family
class DeepSetsFamily(ModelFamily):
    family_name = "deepsets"

    def sample_arch_hyperparams(self, rng: random.Random) -> DeepSetsArchHyperparams:
        return DeepSetsArchHyperparams(
            hidden_dim=rng.choice(self._space["hidden_dim"]),
            equivariant_layers=rng.choice(self._space["equivariant_layers"]),
            fc_layers=rng.choice(self._space["fc_layers"]),
            norm=rng.choice(self._space["norm"]),
            dropout=sample_from_spec(rng, self._space["dropout"]),
            activation=rng.choice(self._space["activation"]),
        )

    def build_model(self, hparams: DeepSetsArchHyperparams) -> nn.Sequential:
        return make_deepsets(
            conv_layers=hparams.equivariant_layers,
            fc_layers=hparams.fc_layers,
            hidden_dim=hparams.hidden_dim,
            in_dim=self._dataset_info.in_channels,
            num_classes=self._dataset_info.num_classes,
            activation=hparams.activation,
            norm=hparams.norm,
            dropout=hparams.dropout,
        )
