"""Multi-head causal self-attention module."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Each token can only attend to positions at or before itself in the
    sequence (causal / autoregressive mask).  Uses a standard scaled
    dot-product attention implementation.

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.dropout_p = config.dropout

        # Fused QKV projection for efficiency.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — lower-triangular matrix of ones.
        # Registered as a buffer so it moves to the correct device automatically.
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.context_length, config.context_length, dtype=torch.bool)
            ).view(1, 1, config.context_length, config.context_length),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, C)`` where B is batch size,
               T is sequence length, and C is the embedding dimension.

        Returns:
            Output tensor of shape ``(B, T, C)``.
        """
        B, T, C = x.size()

        # Compute query, key, value projections.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_heads, T, head_dim).
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention.
        attention_scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / attention_scale  # (B, nh, T, T)

        # Apply causal mask — set future positions to -inf so softmax → 0.
        attn = attn.masked_fill(~self.bias[:, :, :T, :T], float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values.
        y = attn @ v  # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection.
        y = self.resid_dropout(self.c_proj(y))
        return y
