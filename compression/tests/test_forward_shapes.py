from __future__ import annotations

from copy import deepcopy
import unittest

import torch

from compression.api import CompressionConfig, compress
from compression.families import build_model_for_family


def _check_forward(family: str, arch: dict) -> None:
    teacher = build_model_for_family(family, arch)
    student, _ = compress(
        teacher=teacher,
        family=family,
        architecture=deepcopy(arch),
        method="uniform",
        cfg=CompressionConfig(seed=0, device="cpu"),
    )
    x = torch.randn(2, 3, 32, 32)
    y = student(x)
    assert y.shape == (2, 10)


class ForwardShapeTests(unittest.TestCase):
    def test_forward_shapes_all_families(self) -> None:
        _check_forward(
            "cnn2d",
            {
                "hidden_dim": 24,
                "conv_layers": 2,
                "fc_layers": 2,
                "norm": "bn",
                "dropout": 0.0,
                "activation": "relu",
            },
        )
        _check_forward(
            "cnn1d",
            {
                "hidden_dim": 24,
                "conv_layers": 2,
                "fc_layers": 2,
                "norm": "gn",
                "dropout": 0.0,
                "activation": "gelu",
            },
        )
        _check_forward(
            "deepsets",
            {
                "hidden_dim": 32,
                "equivariant_layers": 2,
                "fc_layers": 2,
                "norm": "bn",
                "dropout": 0.0,
                "activation": "relu",
            },
        )
        _check_forward("resnet", {"hidden_dim": 32, "blocks": 2})
        _check_forward(
            "vit",
            {
                "hidden_dim": 32,
                "num_layers": 2,
                "num_heads": 2,
                "dropout": 0.0,
                "patch_size": 4,
            },
        )
