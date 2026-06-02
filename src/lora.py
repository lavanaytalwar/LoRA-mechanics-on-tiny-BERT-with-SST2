from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "query",
        "key",
        "value",
        "dense",
    )


class LoRALinear(nn.Module):
    """
    LoRA for nn.Linear: y = xW^T + b + (x * D) * (BA)^T * scaling
    where A: (r, in), B: (out, r), scaling = alpha / r.
    Base weight is frozen (requires_grad=False).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError("r must be > 0")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        # Frozen base params
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # LoRA params
        # A: down projection, B: up projection
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # As in LoRA paper: initialize A with Kaiming, B with zeros so delta starts at 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def lora_weight(self) -> torch.Tensor:
        # (out, in)
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base.weight.add_(self.lora_weight())
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.sub_(self.lora_weight())
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)

        result = self.base(x)
        x_d = self.dropout(x)
        # F.linear expects weight shape (out, in)
        delta = F.linear(x_d, self.lora_weight(), bias=None)
        return result + delta


def _iter_named_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        yield name, module


def mark_only_lora_as_trainable(model: nn.Module, train_classifier: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    if train_classifier:
        for n, p in model.named_parameters():
            if n.startswith("classifier.") or ".classifier." in n:
                p.requires_grad_(True)


def inject_lora(
    model: nn.Module,
    cfg: LoRAConfig,
    module_filter: Optional[Tuple[str, ...]] = None,
) -> int:
    """
    Replace selected nn.Linear modules with LoRALinear wrappers.
    Returns number of modules replaced.
    """
    targets = module_filter if module_filter is not None else cfg.target_modules
    replaced = 0

    # We need parent references to replace modules in-place.
    for full_name, module in list(_iter_named_modules(model)):
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.split(".")[-1]
        if leaf not in targets:
            continue

        parent = model
        parts = full_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]
        base = getattr(parent, child_name)
        if isinstance(base, LoRALinear):
            continue
        setattr(parent, child_name, LoRALinear(base=base, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout))
        replaced += 1

    return replaced


def merge_lora_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable

