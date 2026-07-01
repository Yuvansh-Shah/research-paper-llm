"""Model hyperparameter configuration dataclass."""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class GPTConfig:
    """Configuration for a GPT-style decoder-only Transformer.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 BPE default: 50257).
        context_length: Maximum sequence / context window length.
        n_embd: Embedding / hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        dropout: Dropout probability applied throughout the model.
        bias: Whether to use bias in Linear and LayerNorm layers.
        use_flash_attn: Whether to use Flash Attention / SDPA when available.
    """

    vocab_size: int = 50_257
    context_length: int = 1024
    n_embd: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.0
    bias: bool = False  # Modern practice: no bias for cleaner training
    use_flash_attn: bool = True  # Use SDPA by default when available

    def __post_init__(self) -> None:
        if self.n_embd % self.n_heads != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_heads

    def num_params_estimate(self) -> int:
        """Estimate total parameter count (approximate)."""
        # Embeddings
        emb = self.vocab_size * self.n_embd + self.context_length * self.n_embd
        # Per block: attn (qkv + out) + ffn (up + down)
        per_block = (4 * self.n_embd * self.n_embd) + (8 * self.n_embd * self.n_embd)
        blocks = per_block * self.n_layers
        # LM head (shared with embeddings, so not counted)
        return emb + blocks

    # -------------------------------------------------------------------------
    # Model Size Presets
    # -------------------------------------------------------------------------
    # Naming convention:
    #   - debug: For smoke tests, CPU, minimal VRAM (<2GB)
    #   - small: For quick experiments, consumer GPU (4-8GB)
    #   - base: Production baseline, mid-range GPU (16-24GB)
    #   - large: High-quality, high-end GPU (40-48GB)
    #   - xl: Research-scale, multi-GPU / A100s (80GB+)

    @classmethod
    def debug(
        cls,
        *,
        vocab_size: int = 50_257,
        context_length: int = 64,
    ) -> "GPTConfig":
        """~16M params - Smoke tests, CPU, minimal VRAM.

        The default 50257-token vocab dominates the count (~13M embedding,
        ~3M non-embedding), so a smaller vocab_size shrinks this sharply.

        Use case: Unit tests, CI, quick iteration on code changes.
        VRAM: <2GB | Context: 64 tokens
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            n_embd=256,
            n_heads=4,
            n_layers=4,
            dropout=0.0,
            bias=False,
        )

    @classmethod
    def small(
        cls,
        *,
        vocab_size: int = 50_257,
        context_length: int = 256,
    ) -> "GPTConfig":
        """~45M params (~19M non-embedding) - Quick experiments, consumer GPU.

        Use case: Hyperparameter search, debugging training dynamics.
        VRAM: ~4-6GB | Context: 256 tokens
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            n_embd=512,
            n_heads=8,
            n_layers=6,
            dropout=0.1,
            bias=False,
        )

    @classmethod
    def base(
        cls,
        *,
        vocab_size: int = 50_257,
        context_length: int = 1024,
    ) -> "GPTConfig":
        """~124M params (GPT-2 small equivalent) - Production baseline.
        
        Use case: Meaningful capability experiments, fine-tuning.
        VRAM: ~16-24GB | Context: 1024 tokens
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            n_embd=768,
            n_heads=12,
            n_layers=12,
            dropout=0.0,
            bias=False,
        )

    @classmethod
    def large(
        cls,
        *,
        vocab_size: int = 50_257,
        context_length: int = 2048,
    ) -> "GPTConfig":
        """~356M params (GPT-2 medium equivalent) - High quality.
        
        Use case: Production models, capability-focused training.
        VRAM: ~40-48GB | Context: 2048 tokens
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            n_embd=1024,
            n_heads=16,
            n_layers=24,
            dropout=0.0,
            bias=False,
        )

    @classmethod
    def xl(
        cls,
        *,
        vocab_size: int = 50_257,
        context_length: int = 2048,
    ) -> "GPTConfig":
        """~775M params (GPT-2 large equivalent) - Research scale.
        
        Use case: Multi-GPU training, maximum capability.
        VRAM: ~80GB+ (multi-GPU recommended) | Context: 2048 tokens
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            n_embd=1280,
            n_heads=20,
            n_layers=36,
            dropout=0.0,
            bias=False,
        )

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        vocab_size: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> "GPTConfig":
        """Create config from preset name."""
        presets = {
            "debug": cls.debug,
            "small": cls.small,
            "base": cls.base,
            "large": cls.large,
            "xl": cls.xl,
            # Legacy aliases
            "medium": cls.base,
        }
        if name not in presets:
            valid = ", ".join(presets.keys())
            raise ValueError(f"Unknown model size '{name}'. Valid options: {valid}")
        kwargs = {}
        if vocab_size is not None:
            kwargs["vocab_size"] = vocab_size
        if context_length is not None:
            kwargs["context_length"] = context_length
        return presets[name](**kwargs)
