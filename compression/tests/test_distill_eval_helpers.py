from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compression.eval.distillation import (
    evaluate_teacher_student_kl,
    model_normalized_l1,
    normalized_l1_bucket,
    param_size_bucket,
    split_teacher_model_ids,
)


class DistillEvalHelperTests(unittest.TestCase):
    def test_teacher_split_is_deterministic(self) -> None:
        model_ids = [f"model_{i:06d}" for i in range(10)]
        a = split_teacher_model_ids(
            model_ids=model_ids, teacher_train_fraction=0.8, split_seed=13
        )
        b = split_teacher_model_ids(
            model_ids=model_ids, teacher_train_fraction=0.8, split_seed=13
        )
        c = split_teacher_model_ids(
            model_ids=model_ids, teacher_train_fraction=0.8, split_seed=14
        )
        self.assertEqual(a.train_model_ids, b.train_model_ids)
        self.assertEqual(a.test_model_ids, b.test_model_ids)
        self.assertNotEqual(a.train_model_ids, c.train_model_ids)
        self.assertEqual(len(a.train_model_ids), 8)
        self.assertEqual(len(a.test_model_ids), 2)

    def test_kl_is_finite_and_near_zero_for_identical_models(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(32, 3, 4, 4)
        y = torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
        teacher = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
        student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
        student.load_state_dict(teacher.state_dict())
        kl = evaluate_teacher_student_kl(
            teacher=teacher,
            student=student,
            loader=loader,
            device=torch.device("cpu"),
            temperature=1.0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(kl)))
        self.assertLess(abs(kl), 1e-7)

    def test_param_size_bucket_boundaries(self) -> None:
        self.assertEqual(param_size_bucket(9146), "Small")
        self.assertEqual(param_size_bucket(9147), "Medium")
        self.assertEqual(param_size_bucket(27306), "Medium")
        self.assertEqual(param_size_bucket(27307), "Large")

    def test_normalized_l1_computation_and_bucket_boundaries(self) -> None:
        layer = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            layer.weight.fill_(2.0)
        expected = 2.0
        actual = model_normalized_l1(layer)
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(normalized_l1_bucket(0.1504), "Small")
        self.assertEqual(normalized_l1_bucket(0.1505), "Medium")
        self.assertEqual(normalized_l1_bucket(3.6259), "Medium")
        self.assertEqual(normalized_l1_bucket(3.6260), "Large")
