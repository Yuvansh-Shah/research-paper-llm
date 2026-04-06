"""Data download and pre-processing script.

Downloads a text corpus (Wikipedia by default), tokenises it with the
GPT-2 BPE tokeniser via ``tiktoken``, and writes train/val split as
memory-mapped uint16 ``.bin`` files ready for :class:`training.dataset.TokenDataset`.

Usage::

    # Download and tokenise English Wikipedia (subset)
    python data/prepare.py --source wikipedia --lang en --output_dir data/processed

    # Tokenise a local text file
    python data/prepare.py --source file --input_path my_corpus.txt \\
        --output_dir data/processed

    # Use a Hugging Face dataset
    python data/prepare.py --source hf --hf_dataset wikitext \\
        --hf_name wikitext-103-raw-v1 --output_dir data/processed
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tokeniser():
    """Return a tiktoken GPT-2 encoder (lazily imported)."""
    try:
        import tiktoken  # noqa: PLC0415
    except ImportError:
        sys.exit("tiktoken is required: pip install tiktoken")
    return tiktoken.get_encoding("gpt2")


def _encode_texts(texts: Iterable[str], enc) -> np.ndarray:
    """Encode an iterable of strings and concatenate into a uint16 array."""
    all_ids: list[int] = []
    eot = enc.eot_token  # end-of-text sentinel between documents
    for text in texts:
        all_ids.extend(enc.encode_ordinary(text))
        all_ids.append(eot)
    return np.array(all_ids, dtype=np.uint16)


def _write_bin(ids: np.ndarray, path: str) -> None:
    """Write a uint16 token array to *path* as a memory-mapped .bin file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fp = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(ids),))
    fp[:] = ids
    fp.flush()
    print(f"  Wrote {len(ids):,} tokens → {path}")


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

def _load_file(input_path: str) -> list[str]:
    with open(input_path, encoding="utf-8") as fh:
        return [fh.read()]


def _load_hf_dataset(hf_dataset: str, hf_name: str, split: str) -> list[str]:
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        sys.exit("datasets is required: pip install datasets")
    ds = load_dataset(hf_dataset, hf_name, split=split, trust_remote_code=False)
    # Most HF text datasets have a 'text' column.
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    return [row for row in ds[col] if row and row.strip()]


def _load_wikipedia(lang: str = "en", num_articles: int = 50_000) -> list[str]:
    """Stream *num_articles* articles from the Wikipedia HF dataset."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        sys.exit("datasets is required: pip install datasets")
    ds = load_dataset(
        "wikipedia",
        f"20220301.{lang}",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    texts: list[str] = []
    for i, row in enumerate(ds):
        if i >= num_articles:
            break
        texts.append(row["text"])
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenise text corpus for CustomLM pre-training")
    p.add_argument("--source", choices=["file", "hf", "wikipedia"], default="wikipedia",
                   help="Data source")
    # File source
    p.add_argument("--input_path", default=None,
                   help="Path to local text file (--source file)")
    # HF source
    p.add_argument("--hf_dataset", default="wikitext",
                   help="Hugging Face dataset name (--source hf)")
    p.add_argument("--hf_name", default="wikitext-103-raw-v1",
                   help="Hugging Face dataset config (--source hf)")
    # Wikipedia source
    p.add_argument("--lang", default="en",
                   help="Wikipedia language (--source wikipedia)")
    p.add_argument("--num_articles", type=int, default=50_000,
                   help="Number of Wikipedia articles to download")
    # Common
    p.add_argument("--output_dir", default="data/processed",
                   help="Directory for output .bin files")
    p.add_argument("--val_fraction", type=float, default=0.05,
                   help="Fraction of tokens reserved for validation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enc = _get_tokeniser()
    print(f"Tokeniser: {enc.name}  (vocab size {enc.n_vocab})")

    # ------------------------------------------------------------------
    # Load raw texts
    # ------------------------------------------------------------------
    print(f"Loading data source: {args.source}")
    if args.source == "file":
        if not args.input_path:
            sys.exit("--input_path required for --source file")
        texts = _load_file(args.input_path)
    elif args.source == "hf":
        texts = _load_hf_dataset(args.hf_dataset, args.hf_name, split="train")
    else:  # wikipedia
        texts = _load_wikipedia(args.lang, args.num_articles)

    print(f"Loaded {len(texts):,} documents")

    # ------------------------------------------------------------------
    # Tokenise
    # ------------------------------------------------------------------
    print("Tokenising …")
    ids = _encode_texts(texts, enc)
    print(f"Total tokens: {len(ids):,}")

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    n_val = max(1, int(len(ids) * args.val_fraction))
    train_ids = ids[:-n_val]
    val_ids = ids[-n_val:]

    # ------------------------------------------------------------------
    # Write .bin files
    # ------------------------------------------------------------------
    _write_bin(train_ids, os.path.join(args.output_dir, "train.bin"))
    _write_bin(val_ids, os.path.join(args.output_dir, "val.bin"))
    print("Done.")


if __name__ == "__main__":
    main()
