"""CNN2D architecture family."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch.nn as nn

from gmn.graph_construct.net_makers import make_cnn
from model_zoo.config import sample_from_spec
from model_zoo.families.base import ArchHyperparams, ModelFamily
from model_zoo.registry import register_family


@dataclass(frozen=True)
class CNN2DArchHyperparams(ArchHyperparams):
    hidden_dim: int
    conv_layers: int
    fc_layers: int
    norm: str
    dropout: float
    activation: str


@register_family
class CNN2DFamily(ModelFamily):
    family_name = "cnn2d"

    def sample_arch_hyperparams(self, rng: random.Random) -> CNN2DArchHyperparams:
        return CNN2DArchHyperparams(
            hidden_dim=rng.choice(self._space["hidden_dim"]),
            conv_layers=rng.choice(self._space["conv_layers"]),
            fc_layers=rng.choice(self._space["fc_layers"]),
            norm=rng.choice(self._space["norm"]),
            dropout=sample_from_spec(rng, self._space["dropout"]),
            activation=rng.choice(self._space["activation"]),
        )

    def build_model(self, hparams: CNN2DArchHyperparams) -> nn.Sequential:
        return make_cnn(
            conv_layers=hparams.conv_layers,
            fc_layers=hparams.fc_layers,
            hidden_dim=hparams.hidden_dim,
            in_dim=self._dataset_info.in_channels,
            num_classes=self._dataset_info.num_classes,
            activation=hparams.activation,
            norm=hparams.norm,
            dropout=hparams.dropout,
        )
