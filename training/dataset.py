"""PyTorch Dataset for token-level language model pre-training.

Reads a memory-mapped binary file of uint16 token IDs produced by
``data/prepare.py`` and yields overlapping ``(input, target)`` pairs
of length ``context_length`` for next-token prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class MemmapDataset:
    """Simple uint16 memory-mapped token dataset."""

    path: Path | str

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(self.data[idx])


class TokenDataset(Dataset):
    """Dataset backed by a memory-mapped ``.bin`` file of uint16 token IDs.

    Args:
        bin_path: Path to the ``.bin`` file containing tokenised data.
        context_length: Length of each training sequence (number of tokens).
    """

    def __init__(self, bin_path: str, context_length: int) -> None:
        # Load as memory-mapped array for zero-copy random access.
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length

    def __len__(self) -> int:
        # Number of complete (input, target) pairs available.
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single (input, target) pair.

        Args:
            idx: Starting index in the token array.

        Returns:
            A tuple ``(x, y)`` where ``x`` is the input sequence of
            ``context_length`` tokens and ``y`` is the target sequence
            shifted by one position.
        """
        chunk = torch.from_numpy(
            self.data[idx : idx + self.context_length + 1].astype(np.int64)
        )
        x = chunk[:-1]  # input tokens
        y = chunk[1:]   # target tokens (shifted by 1)
        return x, y
