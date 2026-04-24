from __future__ import annotations

from copy import deepcopy
import unittest

from compression.api import CalibrationConfig, CompressionConfig, compress
from compression.families import build_model_for_family


class DeterminismTests(unittest.TestCase):
    def test_random_selector_is_seed_deterministic(self) -> None:
        arch = {
            "hidden_dim": 32,
            "conv_layers": 2,
            "fc_layers": 2,
            "norm": "bn",
            "dropout": 0.0,
            "activation": "relu",
        }
        teacher = build_model_for_family("cnn2d", arch)
        cfg = CompressionConfig(seed=123, device="cpu")

        _, report_a = compress(
            teacher=teacher,
            family="cnn2d",
            architecture=deepcopy(arch),
            method="random_consistent",
            cfg=cfg,
        )
        _, report_b = compress(
            teacher=teacher,
            family="cnn2d",
            architecture=deepcopy(arch),
            method="random_consistent",
            cfg=cfg,
        )
        self.assertEqual(report_a.indices_per_group, report_b.indices_per_group)

    def test_activation_selector_is_seed_deterministic(self) -> None:
        arch = {
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "dropout": 0.0,
            "patch_size": 4,
        }
        teacher = build_model_for_family("vit", arch)
        cfg = CompressionConfig(
            seed=7,
            device="cpu",
            calibration=CalibrationConfig(
                data_dir="./data",
                num_images=8,
                batch_size=4,
                seed=99,
                num_workers=0,
            ),
        )
        _, report_a = compress(
            teacher=teacher,
            family="vit",
            architecture=deepcopy(arch),
            method="activation",
            cfg=cfg,
        )
        _, report_b = compress(
            teacher=teacher,
            family="vit",
            architecture=deepcopy(arch),
            method="activation",
            cfg=cfg,
        )
        self.assertEqual(report_a.indices_per_group, report_b.indices_per_group)
