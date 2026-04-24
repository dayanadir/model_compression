"""He/Kaiming-based random reinitialization baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from gmn.graph_construct.layers import EquivSetLinear


def apply_he_reinit(module: nn.Module, seed: int) -> None:
    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    torch.manual_seed(seed)

    for submodule in module.modules():
        if isinstance(submodule, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(submodule.weight, nonlinearity="relu")
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if submodule.weight is not None:
                nn.init.ones_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
            submodule.running_mean.zero_()
            submodule.running_var.fill_(1.0)
            submodule.num_batches_tracked.zero_()
        elif isinstance(submodule, nn.GroupNorm):
            if submodule.weight is not None:
                nn.init.ones_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.LayerNorm):
            if submodule.weight is not None:
                nn.init.ones_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.MultiheadAttention):
            submodule._reset_parameters()
        elif isinstance(submodule, EquivSetLinear):
            nn.init.kaiming_normal_(submodule.lin1.weight, nonlinearity="relu")
            if submodule.lin1.bias is not None:
                nn.init.zeros_(submodule.lin1.bias)
            nn.init.kaiming_normal_(submodule.lin2.weight, nonlinearity="relu")

    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
