"""Helpers for compression distillation evaluation on CIFAR-10."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from compression.eval.cifar10 import build_cifar10_test_loader


PARAM_BUCKET_SMALL_MAX = 9146
PARAM_BUCKET_MEDIUM_MAX = 27306

NORM_L1_BUCKET_SMALL_MAX = 0.1504
NORM_L1_BUCKET_MEDIUM_MAX = 3.6259


@dataclass(frozen=True)
class TeacherSplit:
    train_model_ids: list[str]
    test_model_ids: list[str]
    split_seed: int
    teacher_train_fraction: float


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_teacher_model_ids(
    model_ids: Sequence[str],
    teacher_train_fraction: float,
    split_seed: int,
) -> TeacherSplit:
    """Deterministically split model ids into train/test teacher sets."""
    if not 0.0 < teacher_train_fraction < 1.0:
        raise ValueError("teacher_train_fraction must be in the open interval (0, 1)")
    sorted_ids = sorted(model_ids)
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(sorted_ids))
    shuffled = [sorted_ids[i] for i in order]
    train_n = int(len(shuffled) * teacher_train_fraction)
    train_ids = sorted(shuffled[:train_n])
    test_ids = sorted(shuffled[train_n:])
    return TeacherSplit(
        train_model_ids=train_ids,
        test_model_ids=test_ids,
        split_seed=int(split_seed),
        teacher_train_fraction=float(teacher_train_fraction),
    )


def parse_eval_steps(eval_steps: str, distill_steps: int) -> list[int]:
    """Parse comma-separated distillation eval steps."""
    try:
        steps = sorted({int(x.strip()) for x in eval_steps.split(",") if x.strip()})
    except ValueError as exc:
        raise ValueError(f"Invalid --eval-steps value: {eval_steps!r}") from exc
    if not steps:
        raise ValueError("--eval-steps must contain at least one integer step")
    if steps[0] < 0:
        raise ValueError("Eval steps must be non-negative")
    if steps[-1] > distill_steps:
        raise ValueError(
            f"Max eval step {steps[-1]} cannot exceed distill_steps={distill_steps}"
        )
    if 0 not in steps:
        raise ValueError("Eval steps must include step 0")
    return steps


def build_cifar10_train_loader(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """Build CIFAR-10 training loader used for distillation."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_cifar10_test_loader_cached(
    data_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
) -> DataLoader:
    """Build CIFAR-10 test loader for KL evaluation."""
    return build_cifar10_test_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def model_num_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_normalized_l1(model: nn.Module) -> float:
    """Compute normalized L1 across all model parameters."""
    total_abs = 0.0
    total_params = 0
    with torch.no_grad():
        for p in model.parameters():
            total_abs += float(p.detach().abs().sum().item())
            total_params += int(p.numel())
    if total_params == 0:
        return 0.0
    return total_abs / float(total_params)


def param_size_bucket(num_params: int) -> str:
    if num_params <= PARAM_BUCKET_SMALL_MAX:
        return "Small"
    if num_params <= PARAM_BUCKET_MEDIUM_MAX:
        return "Medium"
    return "Large"


def normalized_l1_bucket(normalized_l1: float) -> str:
    if normalized_l1 <= NORM_L1_BUCKET_SMALL_MAX:
        return "Small"
    if normalized_l1 <= NORM_L1_BUCKET_MEDIUM_MAX:
        return "Medium"
    return "Large"


def _batch_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * (temperature * temperature)
    )


@torch.no_grad()
def evaluate_teacher_student_kl(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> float:
    """Compute sample-weighted mean KL(student || teacher) over loader."""
    teacher.eval()
    student.eval()
    total_kl = 0.0
    total_samples = 0
    for inputs, _targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        teacher_logits = teacher(inputs)
        student_logits = student(inputs)
        loss = _batch_kl(student_logits, teacher_logits, temperature)
        bsz = int(inputs.shape[0])
        total_kl += float(loss.item()) * bsz
        total_samples += bsz
    if total_samples == 0:
        return 0.0
    return total_kl / float(total_samples)


def run_distillation_steps(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    distill_steps: int,
    temperature: float,
    lr: float,
    weight_decay: float,
    train_log_steps: Iterable[int] | None = None,
) -> dict[int, float]:
    """Distill student from teacher for exactly distill_steps updates.

    Returns a map {step: train_kl} for recorded steps where step starts at 1.
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    student.to(device)
    teacher.to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    log_steps = set(train_log_steps or [])
    step_to_train_kl: dict[int, float] = {}
    iterator = iter(train_loader)
    for step in range(1, distill_steps + 1):
        try:
            inputs, _targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, _targets = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)
        loss = _batch_kl(student_logits, teacher_logits, temperature)
        loss.backward()
        optimizer.step()
        if step in log_steps:
            step_to_train_kl[step] = float(loss.item())
    return step_to_train_kl


def distill_with_eval_checkpoints(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    distill_steps: int,
    eval_steps: Sequence[int],
    temperature: float,
    lr: float,
    weight_decay: float,
    train_log_steps: Iterable[int] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Run distillation and return train/test KL keyed by step.

    `eval_steps` may include step 0. Test KL is evaluated exactly at the
    requested steps. Train KL is recorded at `train_log_steps`.
    """
    eval_set = set(int(s) for s in eval_steps)
    if 0 not in eval_set:
        raise ValueError("eval_steps must include 0")
    if max(eval_set) > distill_steps:
        raise ValueError("eval_steps cannot exceed distill_steps")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    teacher.to(device)
    student.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_log = set(train_log_steps or [])
    train_kl_by_step: dict[int, float] = {}
    test_kl_by_step: dict[int, float] = {
        0: evaluate_teacher_student_kl(
            teacher=teacher,
            student=student,
            loader=test_loader,
            device=device,
            temperature=temperature,
        )
    }

    iterator = iter(train_loader)
    for step in range(1, distill_steps + 1):
        try:
            inputs, _targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, _targets = next(iterator)

        inputs = inputs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)
        loss = _batch_kl(student_logits, teacher_logits, temperature)
        loss.backward()
        optimizer.step()

        if step in train_log:
            train_kl_by_step[step] = float(loss.item())
        if step in eval_set:
            test_kl_by_step[step] = evaluate_teacher_student_kl(
                teacher=teacher,
                student=student,
                loader=test_loader,
                device=device,
                temperature=temperature,
            )
            student.train()
    return train_kl_by_step, test_kl_by_step
