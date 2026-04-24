from __future__ import annotations

from copy import deepcopy
import unittest

from compression.api import CalibrationConfig, CompressionConfig, compress
from compression.families import build_model_for_family


METHODS = [
    "uniform",
    "random_consistent",
    "l1_structured",
    "magnitude",
    "in_out_meanabs",
    "activation",
    "he_reinit",
]


def num_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def n_cnn(H: int, L: int, F: int) -> int:
    return (9 * L + F - 10) * H * H + (3 * L + F + 36) * H + 10


def n_deepsets(H: int, E: int, F: int) -> int:
    return (2 * E + F - 1) * H * H + (3 * E + F + 15) * H + 10


def n_resnet(H: int, B: int) -> int:
    return (18 * B - 4) * H * H + (4 * B + 51) * H + 10


def n_vit(H: int, T: int, P: int = 4) -> int:
    return 12 * T * H * H + (3 * P * P + 13 * T + 11) * H + 10


def _run_family(family: str, arch: dict, formula_fn) -> None:
    teacher = build_model_for_family(family, arch)
    h = int(arch["hidden_dim"])
    h2 = h // 2
    assert num_params(teacher) == formula_fn(h, arch)

    for method in METHODS:
        cfg = CompressionConfig(
            seed=0,
            device="cpu",
            calibration=CalibrationConfig(
                data_dir="./data",
                num_images=8,
                batch_size=4,
                seed=0,
                num_workers=0,
            ),
        )
        student, _ = compress(
            teacher=teacher,
            family=family,
            architecture=deepcopy(arch),
            method=method,
            cfg=cfg,
        )
        assert num_params(student) == formula_fn(h2, arch)


class ParamCountTests(unittest.TestCase):
    def test_param_counts_all_families_all_methods(self) -> None:
        _run_family(
            "cnn2d",
            {
                "hidden_dim": 32,
                "conv_layers": 3,
                "fc_layers": 2,
                "norm": "bn",
                "dropout": 0.0,
                "activation": "relu",
            },
            lambda h, a: n_cnn(h, a["conv_layers"], a["fc_layers"]),
        )
        _run_family(
            "cnn1d",
            {
                "hidden_dim": 28,
                "conv_layers": 2,
                "fc_layers": 2,
                "norm": "gn",
                "dropout": 0.25,
                "activation": "gelu",
            },
            lambda h, a: n_cnn(h, a["conv_layers"], a["fc_layers"]),
        )
        _run_family(
            "deepsets",
            {
                "hidden_dim": 64,
                "equivariant_layers": 4,
                "fc_layers": 2,
                "norm": "bn",
                "dropout": 0.0,
                "activation": "relu",
            },
            lambda h, a: n_deepsets(h, a["equivariant_layers"], a["fc_layers"]),
        )
        _run_family(
            "resnet",
            {"hidden_dim": 32, "blocks": 4},
            lambda h, a: n_resnet(h, a["blocks"]),
        )
        _run_family(
            "vit",
            {
                "hidden_dim": 48,
                "num_layers": 3,
                "num_heads": 2,
                "dropout": 0.0,
                "patch_size": 4,
            },
            lambda h, a: n_vit(h, a["num_layers"], a["patch_size"]),
        )
