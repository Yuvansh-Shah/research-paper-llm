"""Transformer block: attention + feed-forward with residual connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.attention import CausalSelfAttention
from model.config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm with an always-learnable scale and an optional bias.

    ``nn.LayerNorm`` ties both the scale (gamma) and bias (beta) to a single
    ``elementwise_affine`` flag, so disabling the bias would also remove the
    learnable scale. This variant keeps the scale learnable unconditionally
    and makes only the bias optional.

    Args:
        ndim: Number of features to normalise over.
        bias: Whether to include a learnable bias (beta) term.
    """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Two linear layers with a GELU activation in between, following the
    GPT-2 convention of expanding to 4× the embedding dimension.

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """A single Transformer decoder block.

    Uses **pre-LayerNorm** (LayerNorm applied before each sub-layer) for
    more stable training, as recommended in recent literature.

    Structure::

        x → LN → Attention → + → LN → FFN → +
        └──────────────────┘ └───────────────┘

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape ``(B, T, C)``.

        Returns:
            Output tensor of shape ``(B, T, C)``.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x
