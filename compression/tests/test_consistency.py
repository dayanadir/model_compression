from __future__ import annotations

from copy import deepcopy
import unittest

import torch

from compression.api import CompressionConfig, compress
from compression.families import build_model_for_family


class ConsistencyTests(unittest.TestCase):
    def test_consistent_indices_applied_to_in_and_out_axes(self) -> None:
        arch = {
            "hidden_dim": 32,
            "conv_layers": 3,
            "fc_layers": 1,
            "norm": "bn",
            "dropout": 0.0,
            "activation": "relu",
        }
        teacher = build_model_for_family("cnn2d", arch)
        student, report = compress(
            teacher=teacher,
            family="cnn2d",
            architecture=deepcopy(arch),
            method="random_consistent",
            cfg=CompressionConfig(seed=123, device="cpu"),
        )
        idx = torch.tensor(report.indices_per_group["cnn2d:H"], dtype=torch.long)
        teacher_conv = teacher.state_dict()["4.weight"]
        expected = teacher_conv.index_select(0, idx).index_select(1, idx)
        got = student.state_dict()["4.weight"]
        self.assertTrue(torch.allclose(got, expected))
