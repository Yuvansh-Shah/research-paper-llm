"""Full GPT-style decoder-only language model."""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from model.config import GPTConfig
from model.transformer import TransformerBlock


class GPT(nn.Module):
    """GPT-style decoder-only Transformer language model.

    Architecture overview::

        Token IDs
            ↓
        [Token Embedding]  +  [Positional Embedding]
            ↓
        [Transformer Block] × n_layers
            ↓
        [LayerNorm]
            ↓
        [LM Head — Linear: n_embd → vocab_size]
            ↓
        Logits

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.context_length, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layers)]
                ),
                "ln_f": nn.LayerNorm(config.n_embd, elementwise_affine=config.bias),
            }
        )

        # Language model head — no bias, weight tied to token embedding.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share token embedding and LM head weights.
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialise weights.
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2 * n_layers) as in GPT-2.
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Args:
            idx: Integer token indices of shape ``(B, T)``.
            targets: Optional target token indices of shape ``(B, T)``.
                     When provided, the cross-entropy loss is also returned.

        Returns:
            A tuple ``(logits, loss)`` where ``logits`` has shape
            ``(B, T, vocab_size)`` and ``loss`` is a scalar tensor or
            ``None`` when ``targets`` is not supplied.
        """
        B, T = idx.size()
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context window {self.config.context_length}"
        )
        device = idx.device

        # Token + positional embeddings.
        pos = torch.arange(T, device=device, dtype=torch.long)
        tok_emb = self.transformer["wte"](idx)   # (B, T, n_embd)
        pos_emb = self.transformer["wpe"](pos)   # (T, n_embd)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Transformer blocks.
        for block in self.transformer["h"]:
            x = block(x)

        # Final layer norm.
        x = self.transformer["ln_f"](x)

        if targets is not None:
            # Training: compute loss over all positions.
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute logits for the last token (faster).
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    # ------------------------------------------------------------------
    # Parameter count helper
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total number of (trainable) parameters."""
        params = (
            (p for p in self.parameters() if p.requires_grad)
            if trainable_only
            else self.parameters()
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def build_checkpoint_payload(
        self,
        step: int,
        optimizer_state: dict | None = None,
        *,
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        payload: dict = {
            "step": step,
            "config": self.config.__dict__,
            "model_state_dict": self.state_dict(),
        }
        if optimizer_state is not None:
            payload["optimizer_state_dict"] = optimizer_state
        if extra_metadata:
            payload.update(extra_metadata)
        return payload

    def save_checkpoint(
        self,
        path: str,
        step: int,
        optimizer_state: dict | None = None,
        *,
        extra_metadata: Optional[dict] = None,
    ) -> None:
        """Save model checkpoint to *path*."""
        torch.save(
            self.build_checkpoint_payload(
                step,
                optimizer_state,
                extra_metadata=extra_metadata,
            ),
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "GPT":
        """Load a GPT model from a checkpoint file."""
        payload = torch.load(path, map_location=device, weights_only=False)
        
        # Handle both dict and GPTConfig object formats
        config_data = payload["config"]
        if isinstance(config_data, GPTConfig):
            config = config_data
        elif isinstance(config_data, dict):
            config = GPTConfig(**config_data)
        else:
            raise ValueError(f"Unknown config type in checkpoint: {type(config_data)}")
        
        model = cls(config)
        model.load_state_dict(payload["model_state_dict"])
        return model

    # ------------------------------------------------------------------
    # Gradient-checkpointing toggle
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on all Transformer blocks to reduce VRAM usage."""
        import torch.utils.checkpoint as cp

        for block in self.transformer["h"]:
            original_forward = block.forward

            def make_checkpointed(fn):
                def checkpointed_forward(x):
                    return cp.checkpoint(fn, x, use_reentrant=False)
                return checkpointed_forward

            block.forward = make_checkpointed(original_forward)
