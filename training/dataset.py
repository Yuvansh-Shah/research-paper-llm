"""PyTorch datasets and manifest helpers for token-level language modeling."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from tokenization import TokenizerSpec, load_tokenizer_spec


def load_dataset_manifest(path: str | Path) -> Optional[dict[str, Any]]:
    resolved = Path(path)
    if resolved.suffix != ".json" or not resolved.exists():
        return None
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must contain a JSON object: {resolved}")
    return manifest


def resolve_tokenizer_spec_from_dataset(path: str | Path) -> Optional[TokenizerSpec]:
    resolved = Path(path)
    manifest = load_dataset_manifest(resolved)
    if manifest is None:
        return None
    if "splits" in manifest:
        return load_tokenizer_spec(
            manifest.get("tokenizer_spec") or manifest.get("tokenizer"),
            base_dir=resolved.parent,
        )
    if "shards" in manifest:
        tokenizer_payload = manifest.get("tokenizer_spec") or manifest.get("tokenizer")
        if tokenizer_payload in (None, "custom", "ascii-test", "test"):
            return None
        return load_tokenizer_spec(tokenizer_payload, base_dir=resolved.parent)
    return None


def resolve_document_index_path(path: str | Path) -> Optional[Path]:
    resolved = Path(path)
    manifest = load_dataset_manifest(resolved)
    if manifest is None:
        return None
    document_index = manifest.get("document_index")
    if not document_index:
        return None
    return (resolved.parent / document_index).resolve()


def _load_manifest_paths(path: Path) -> tuple[list[Path], Optional[dict[str, Any]]]:
    manifest = load_dataset_manifest(path)
    if manifest is None:
        raise ValueError(f"Unrecognized dataset manifest format: {path}")
    if "shards" in manifest:
        return [path.parent / shard["path"] for shard in manifest["shards"]], manifest
    if "splits" in manifest:
        raise ValueError(
            f"{path} is a corpus manifest. Pass a split manifest such as "
            "'train.manifest.json' or 'val.manifest.json' instead."
        )
    raise ValueError(f"Unrecognized dataset manifest format: {path}")


def _resolve_token_paths(path: str | Path) -> tuple[list[Path], Optional[dict[str, Any]]]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    if resolved.suffix == ".json":
        shard_paths, manifest = _load_manifest_paths(resolved)
        if not shard_paths:
            raise ValueError(f"Manifest has no shards: {resolved}")
        return shard_paths, manifest
    return [resolved], None


class TokenStore:
    """A contiguous token view backed by one or more memmapped shards."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.paths, self.manifest = _resolve_token_paths(path)
        self.document_index_path = resolve_document_index_path(self.path)
        self.tokenizer_spec = resolve_tokenizer_spec_from_dataset(self.path)
        self.memmaps = [np.memmap(shard_path, dtype=np.uint16, mode="r") for shard_path in self.paths]
        self.lengths = [int(memmap.shape[0]) for memmap in self.memmaps]
        self.offsets = [0]
        for length in self.lengths:
            self.offsets.append(self.offsets[-1] + length)
        self.total_tokens = self.offsets[-1]
        self.single_memmap = self.memmaps[0] if len(self.memmaps) == 1 else None

    def __len__(self) -> int:
        return self.total_tokens

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total_tokens:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[shard_index]
        return shard_index, local_index

    def token_at(self, index: int) -> int:
        shard_index, local_index = self._locate(index)
        return int(self.memmaps[shard_index][local_index])

    def read_range(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop < start or stop > self.total_tokens:
            raise IndexError((start, stop))
        if start == stop:
            return np.empty(0, dtype=np.uint16)

        start_shard, start_local = self._locate(start)
        end_shard, end_local = self._locate(stop - 1)
        if start_shard == end_shard:
            return self.memmaps[start_shard][start_local : end_local + 1]

        pieces: list[np.ndarray] = []
        first = self.memmaps[start_shard][start_local:]
        pieces.append(np.asarray(first))
        for shard_index in range(start_shard + 1, end_shard):
            pieces.append(np.asarray(self.memmaps[shard_index]))
        last = self.memmaps[end_shard][: end_local + 1]
        pieces.append(np.asarray(last))
        return np.concatenate(pieces)

    def close(self) -> None:
        for memmap in self.memmaps:
            mmap_obj = getattr(memmap, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()


@dataclass
class MemmapDataset:
    """Simple uint16 token dataset backed by one or more memmapped files."""

    path: Path | str

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.store = TokenStore(self.path)
        self.data = self.store.single_memmap

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int) -> int:
        return self.store.token_at(idx)

    def close(self) -> None:
        self.store.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class TokenDataset(Dataset):
    """Dataset backed by a token ``.bin`` file or a shard manifest JSON file."""

    def __init__(self, bin_path: str, context_length: int) -> None:
        self.store = TokenStore(bin_path)
        self.data = self.store.single_memmap
        self.context_length = context_length
        self.document_index_path = self.store.document_index_path
        self.tokenizer_spec = self.store.tokenizer_spec

    def __len__(self) -> int:
        return max(0, len(self.store) - self.context_length)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        chunk = self.store.read_range(idx, idx + self.context_length + 1)
        chunk_tensor = torch.from_numpy(np.asarray(chunk, dtype=np.int64))
        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]
        return x, y

    def close(self) -> None:
        self.store.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
