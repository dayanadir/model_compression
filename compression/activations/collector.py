"""Forward-hook activation score collection."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from compression.axes import AxisGroup, HookSite
from compression.selectors.base import CalibrationContext


def _to_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError(f"Unsupported hook output type: {type(output)}")


def collect_activation_scores(
    teacher: nn.Module,
    axis_groups: list[AxisGroup],
    calib_loader,
    device: torch.device,
) -> CalibrationContext:
    teacher = teacher.to(device)
    teacher.eval()

    module_dict = dict(teacher.named_modules())
    running_sum = defaultdict(lambda: None)
    counts = defaultdict(int)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(group_id: str, site: HookSite):
        def _hook(_module, _inp, output):
            x = _to_tensor(output).detach()
            axis = site.axis if site.axis >= 0 else x.ndim + site.axis
            reduce_dims = tuple(i for i in range(x.ndim) if i != axis)
            mean_abs = x.abs()
            if reduce_dims:
                mean_abs = mean_abs.mean(dim=reduce_dims)
            mean_abs = mean_abs.to(dtype=torch.float64, device="cpu")
            if running_sum[group_id] is None:
                running_sum[group_id] = mean_abs
            else:
                running_sum[group_id] += mean_abs
            counts[group_id] += 1

        return _hook

    for group in axis_groups:
        for site in group.hook_sites:
            if site.module_name not in module_dict:
                raise ValueError(
                    f"Hook module {site.module_name} not found for group {group.id}"
                )
            handles.append(
                module_dict[site.module_name].register_forward_hook(make_hook(group.id, site))
            )

    with torch.no_grad():
        for inputs, _targets in calib_loader:
            teacher(inputs.to(device, non_blocking=True))

    for handle in handles:
        handle.remove()

    scores: dict[str, torch.Tensor] = {}
    for group in axis_groups:
        if counts[group.id] == 0:
            raise ValueError(f"No activation samples collected for group {group.id}")
        scores[group.id] = running_sum[group.id] / counts[group.id]
    return CalibrationContext(group_scores=scores)
