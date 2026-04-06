from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
