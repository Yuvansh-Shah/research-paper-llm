"""LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

Reference: *LoRA: Low-Rank Adaptation of Large Language Models*
           (Hu et al., 2021 — https://arxiv.org/abs/2106.09685)

LoRA inserts low-rank trainable matrices into existing ``nn.Linear``
layers while keeping the pre-trained weights frozen.  For a weight
matrix ``W₀ ∈ ℝ^{d×k}`` the adapted forward pass becomes::

    h = W₀x + (B · A)x · (α / r)

where ``A ∈ ℝ^{r×k}`` and ``B ∈ ℝ^{d×r}`` are the low-rank matrices,
``r`` is the rank (typically 4–64), and ``α`` is a scaling factor.

Usage::

    from finetune.lora import apply_lora, get_lora_state_dict

    # Inject LoRA into the model's attention projections.
    apply_lora(model, rank=16, alpha=32.0, target_modules=["c_attn", "c_proj"])

    # Train — only LoRA parameters have requires_grad=True.
    # ...

    # Save only the tiny LoRA weights (~2–20 MB).
    torch.save(get_lora_state_dict(model), "lora_weights.pt")
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` with LoRA adaptation.

    The original weight is frozen; only the low-rank matrices ``A`` and
    ``B`` are trainable.

    Args:
        linear: The ``nn.Linear`` layer to wrap.
        rank: Low-rank dimension *r*.
        alpha: Scaling factor α.  The effective scaling is ``alpha / rank``.
        dropout: Dropout probability applied to the LoRA path.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features

        # Freeze original weight + bias.
        self.weight = linear.weight  # references original, no copy
        self.weight.requires_grad_(False)
        self.bias = linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank

        # Low-rank matrices (trainable).
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialise A with Kaiming uniform (same as nn.Linear default),
        # B initialised to zero so the initial LoRA delta is zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Original frozen path.
        base_out = nn.functional.linear(x, self.weight, self.bias)
        # LoRA delta path.
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.weight.shape[1]}, "
            f"out_features={self.weight.shape[0]}, "
            f"rank={self.rank}, scaling={self.scaling:.3f}"
        )


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def _get_submodule(model: nn.Module, name: str) -> nn.Module:
    """Retrieve a nested sub-module by dotted attribute name."""
    parts = name.split(".")
    mod = model
    for part in parts:
        mod = getattr(mod, part)
    return mod


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a nested sub-module by dotted attribute name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("c_attn", "c_proj"),
) -> nn.Module:
    """Inject LoRA layers into *model* in-place.

    All ``nn.Linear`` layers whose attribute name ends with one of the
    strings in *target_modules* are replaced with :class:`LoRALinear`.
    All other parameters are frozen.

    Args:
        model: The GPT model to adapt.
        rank: LoRA rank *r*.
        alpha: LoRA scaling factor α.
        dropout: Dropout applied on the LoRA path.
        target_modules: Iterable of name suffixes to match against
            linear layer names (e.g. ``"c_attn"``, ``"c_proj"``).

    Returns:
        The same *model* with LoRA injected (modified in-place).
    """
    target_set = set(target_modules)

    # First, freeze all parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # Then replace matching Linear layers with LoRALinear.
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Match on the final component of the dotted name.
        leaf_name = name.split(".")[-1]
        if leaf_name not in target_set:
            continue
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, lora_layer)
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA injected into {replaced} linear layers. "
        f"Trainable params: {trainable / 1e6:.2f} M / {total / 1e6:.1f} M total "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def get_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return a state dict containing only the LoRA trainable parameters.

    This produces a small file (typically 2–50 MB) that can be merged
    back into the base model at inference time.
    """
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_lora_state_dict(model: nn.Module, lora_state: dict[str, Tensor]) -> None:
    """Load LoRA weights into *model* (must already have LoRA injected)."""
    model.load_state_dict(lora_state, strict=False)
