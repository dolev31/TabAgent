"""A minimal LoRA (Hu et al., 2022), so the fine-tuned baselines need no ``peft``.

It follows peft's defaults for ``LoraConfig(r, lora_alpha, lora_dropout, target_modules)``: A is
Kaiming-uniform (a = sqrt(5)), B is zero, the update is scaled by alpha / r, dropout is applied to the adapter
input, and every base weight is frozen. Adapters are kept in float32 when the base model runs in bfloat16.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T) @ self.lora_B.T
        return out + (delta * self.scaling).to(out.dtype)


def apply_lora(
    model: nn.Module,
    targets: Iterable[str],
    r: int = 16,
    alpha: float = 32,
    dropout: float = 0.05,
    train_modules: Iterable[str] = (),
) -> list[nn.Parameter]:
    """Freeze ``model``, wrap every ``nn.Linear`` child named in ``targets`` with a LoRA adapter, and
    unfreeze the modules named in ``train_modules`` (for example a classification head). Returns the
    trainable parameters."""
    targets, train_modules = set(targets), set(train_modules)
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
                n += 1
    if n == 0:
        raise ValueError(f"no nn.Linear module named {sorted(targets)} in {type(model).__name__}")
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in train_modules:
            for p in module.parameters():
                p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]
