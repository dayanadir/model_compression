"""Evaluation and reporting utilities."""

from compression.eval.cifar10 import EvalResult, evaluate_cifar10_accuracy
from compression.eval.distillation import (
    TeacherSplit,
    distill_with_eval_checkpoints,
    evaluate_teacher_student_kl,
    model_normalized_l1,
    normalized_l1_bucket,
    param_size_bucket,
    parse_eval_steps,
    split_teacher_model_ids,
)
from compression.eval.method_registry import MethodSpec, parse_method_spec, resolve_method_specs
from compression.eval.report import aggregate_results

__all__ = [
    "EvalResult",
    "MethodSpec",
    "TeacherSplit",
    "aggregate_results",
    "distill_with_eval_checkpoints",
    "evaluate_cifar10_accuracy",
    "evaluate_teacher_student_kl",
    "model_normalized_l1",
    "normalized_l1_bucket",
    "param_size_bucket",
    "parse_eval_steps",
    "parse_method_spec",
    "resolve_method_specs",
    "split_teacher_model_ids",
]
