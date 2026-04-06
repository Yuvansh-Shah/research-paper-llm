"""Model hyperparameter configuration dataclass."""

from dataclasses import dataclass, field


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
    """

    vocab_size: int = 50_257
    context_length: int = 256
    n_embd: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_heads != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})"
            )

    @classmethod
    def small(cls) -> "GPTConfig":
        """60 M-parameter model suitable for an RTX 3050 4 GB GPU."""
        return cls(
            vocab_size=50_257,
            context_length=256,
            n_embd=512,
            n_heads=8,
            n_layers=6,
            dropout=0.1,
        )

    @classmethod
    def medium(cls) -> "GPTConfig":
        """~120 M-parameter model for systems with more VRAM."""
        return cls(
            vocab_size=50_257,
            context_length=512,
            n_embd=768,
            n_heads=12,
            n_layers=12,
            dropout=0.1,
        )
