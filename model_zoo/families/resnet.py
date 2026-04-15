"""ResNet architecture family."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch.nn as nn

from gmn.graph_construct.net_makers import make_resnet
from model_zoo.families.base import ArchHyperparams, ModelFamily
from model_zoo.registry import register_family


@dataclass(frozen=True)
class ResNetArchHyperparams(ArchHyperparams):
    hidden_dim: int
    blocks: int


@register_family
class ResNetFamily(ModelFamily):
    family_name = "resnet"

    def sample_arch_hyperparams(self, rng: random.Random) -> ResNetArchHyperparams:
        return ResNetArchHyperparams(
            hidden_dim=rng.choice(self._space["hidden_dim"]),
            blocks=rng.choice(self._space["blocks"]),
        )

    def build_model(self, hparams: ResNetArchHyperparams) -> nn.Sequential:
        return make_resnet(
            conv_layers=hparams.blocks,
            hidden_dim=hparams.hidden_dim,
            in_dim=self._dataset_info.in_channels,
            num_classes=self._dataset_info.num_classes,
        )
