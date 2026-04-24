from __future__ import annotations

import unittest

from compression.cli.common import ALL_METHODS
from compression.eval.method_registry import parse_method_spec, resolve_method_specs


class MethodRegistryTests(unittest.TestCase):
    def test_all_methods_keeps_baseline_order(self) -> None:
        specs = resolve_method_specs("all")
        self.assertEqual([s.compress_method for s in specs], ALL_METHODS)
        self.assertEqual([s.run_method_name for s in specs], ALL_METHODS)

    def test_parse_variant_calibration_overrides(self) -> None:
        spec = parse_method_spec(
            "activation:calib_n=1024;calib_batch_size=64;calib_seed=7"
        )
        self.assertEqual(spec.compress_method, "activation")
        self.assertEqual(spec.calibration_overrides["calib_n"], 1024)
        self.assertEqual(spec.calibration_overrides["calib_batch_size"], 64)
        self.assertEqual(spec.calibration_overrides["calib_seed"], 7)

    def test_parse_include_affine_gamma_bool(self) -> None:
        spec = parse_method_spec("in_out_meanabs:include_affine_gamma=true")
        self.assertEqual(spec.compress_method, "in_out_meanabs")
        self.assertTrue(spec.include_affine_gamma)

    def test_parse_rejects_bad_variant_key(self) -> None:
        with self.assertRaises(ValueError):
            parse_method_spec("magnitude:unknown=5")

    def test_parse_rejects_malformed_variant(self) -> None:
        with self.assertRaises(ValueError):
            parse_method_spec("magnitude:calib_n")
