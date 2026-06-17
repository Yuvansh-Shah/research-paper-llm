#!/usr/bin/env python3
"""Optional data preparation using HuggingFace Datasets.

This module provides a general-purpose dataset pipeline that:
- Downloads and processes datasets from HuggingFace Hub
- Applies quality filtering and deduplication
- Creates proper train/val/test splits
- Reports dataset statistics
- Supports streaming for large datasets

Usage::

    # Use a small dataset for quick testing
    python data/prepare_hf.py --dataset wikitext --config wikitext-2-raw-v1

    # Use a larger dataset
    python data/prepare_hf.py --dataset openwebtext --max_samples 1000000

    # Use TinyStories for quick capability testing
    python data/prepare_hf.py --dataset roneneldan/TinyStories --text_field text

    # Custom filtering
    python data/prepare_hf.py --dataset wikipedia --config 20220301.en \\
        --min_length 100 --max_length 10000 --deduplicate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import tiktoken


# ---------------------------------------------------------------------------
# Text Quality Filtering
# ---------------------------------------------------------------------------

def is_quality_text(
    text: str,
    min_length: int = 50,
    max_length: int = 100_000,
    min_alpha_ratio: float = 0.7,
    max_numeric_ratio: float = 0.2,
    max_repeated_char_ratio: float = 0.2,
) -> bool:
    """Check if text passes quality filters.
    
    Filters:
    - Length bounds
    - Minimum alphabetic character ratio
    - Maximum numeric character ratio  
    - Maximum repeated character ratio (e.g., "aaaa" patterns)
    """
    if not text or len(text) < min_length or len(text) > max_length:
        return False
    
    # Character type ratios
    alpha_count = sum(c.isalpha() for c in text)
    numeric_count = sum(c.isdigit() for c in text)
    total = len(text)
    
    if alpha_count / total < min_alpha_ratio:
        return False
    
    if numeric_count / total > max_numeric_ratio:
        return False
    
    # Check for excessive repeated characters
    if len(text) > 100:
        char_counts = Counter(text.lower())
        max_char_count = max(char_counts.values())
        if max_char_count / len(text) > max_repeated_char_ratio:
            return False
    
    return True


def compute_hash(text: str) -> str:
    """Compute a hash for deduplication."""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()


def compute_minhash_signature(text: str, num_hashes: int = 100) -> tuple:
    """Compute MinHash signature for approximate deduplication.
    
    Uses word-level 3-grams for better dedup quality.
    """
    words = text.lower().split()
    if len(words) < 3:
        return tuple([0] * num_hashes)
    
    # Create 3-grams
    ngrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    
    # Compute MinHash signature
    signature = []
    for seed in range(num_hashes):
        min_hash = float('inf')
        for ngram in ngrams:
            h = hash(ngram + str(seed)) & 0xffffffff
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    
    return tuple(signature)


def jaccard_from_minhash(sig1: tuple, sig2: tuple) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_hf_dataset(
    dataset_name: str,
    config: Optional[str] = None,
    split: str = "train",
    text_field: str = "text",
    streaming: bool = False,
    max_samples: Optional[int] = None,
    trust_remote_code: bool = False,
) -> Iterator[str]:
    """Load dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'wikitext', 'openwebtext')
        config: Dataset configuration (e.g., 'wikitext-2-raw-v1')
        split: Dataset split to load
        text_field: Name of the text field in the dataset
        streaming: Use streaming mode for large datasets
        max_samples: Maximum number of samples to yield
        trust_remote_code: Allow executing code from the dataset repo
        
    Yields:
        Text samples from the dataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        sys.exit(1)
    
    print(f"Loading dataset: {dataset_name}" + (f" ({config})" if config else ""))
    
    ds = load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )
    
    count = 0
    for sample in ds:
        if text_field in sample and sample[text_field]:
            yield sample[text_field]
            count += 1
            if max_samples and count >= max_samples:
                break
    
    print(f"Loaded {count} samples from {dataset_name}")


def load_local_files(
    path: str,
    extensions: tuple = (".txt", ".md", ".json", ".jsonl"),
) -> Iterator[str]:
    """Load text from local files.
    
    Supports:
    - Single text file
    - Directory of text files
    - JSONL files with 'text' field
    """
    path = Path(path)
    
    if path.is_file():
        files = [path]
    else:
        files = [f for f in path.rglob("*") if f.suffix.lower() in extensions]
    
    for file in files:
        print(f"Loading: {file}")
        
        if file.suffix.lower() == ".jsonl":
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            yield obj["text"]
                    except json.JSONDecodeError:
                        continue
        else:
            with open(file, "r", encoding="utf-8") as f:
                yield f.read()


# ---------------------------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------------------------

def process_and_tokenize(
    texts: Iterator[str],
    output_dir: str,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    min_length: int = 50,
    max_length: int = 100_000,
    deduplicate: bool = False,
    dedup_threshold: float = 0.8,
    verbose: bool = True,
) -> dict:
    """Process texts, filter, deduplicate, tokenize, and save.
    
    Args:
        texts: Iterator of text samples
        output_dir: Directory to save processed files
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set  
        test_ratio: Fraction for test set
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
        deduplicate: Whether to perform deduplication
        dedup_threshold: MinHash Jaccard threshold for dedup
        verbose: Print progress
        
    Returns:
        Statistics dict
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Statistics
    stats = {
        "total_samples": 0,
        "filtered_samples": 0,
        "duplicate_samples": 0,
        "kept_samples": 0,
        "total_tokens": 0,
        "train_tokens": 0,
        "val_tokens": 0,
        "test_tokens": 0,
    }
    
    # Collect all valid texts with filtering and deduplication
    all_tokens = []
    seen_hashes = set()
    minhash_signatures = []
    
    for i, text in enumerate(texts):
        stats["total_samples"] += 1
        
        # Quality filtering
        if not is_quality_text(text, min_length=min_length, max_length=max_length):
            stats["filtered_samples"] += 1
            continue
        
        # Exact deduplication via hash
        text_hash = compute_hash(text)
        if text_hash in seen_hashes:
            stats["duplicate_samples"] += 1
            continue
        seen_hashes.add(text_hash)
        
        # Approximate deduplication via MinHash
        if deduplicate:
            sig = compute_minhash_signature(text)
            is_dup = False
            for prev_sig in minhash_signatures[-1000:]:  # Check last 1000 for efficiency
                if jaccard_from_minhash(sig, prev_sig) > dedup_threshold:
                    is_dup = True
                    break
            if is_dup:
                stats["duplicate_samples"] += 1
                continue
            minhash_signatures.append(sig)
        
        # Tokenize
        tokens = enc.encode_ordinary(text)
        all_tokens.extend(tokens)
        stats["kept_samples"] += 1
        
        if verbose and stats["total_samples"] % 10000 == 0:
            print(f"Processed {stats['total_samples']} samples, "
                  f"kept {stats['kept_samples']}, "
                  f"tokens so far: {len(all_tokens):,}")
    
    stats["total_tokens"] = len(all_tokens)
    
    if len(all_tokens) == 0:
        print("Error: No tokens after filtering!")
        return stats
    
    # Convert to numpy array for efficient slicing
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # Create splits
    n = len(all_tokens)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_tokens = all_tokens[:train_end]
    val_tokens = all_tokens[train_end:val_end]
    test_tokens = all_tokens[val_end:]
    
    stats["train_tokens"] = len(train_tokens)
    stats["val_tokens"] = len(val_tokens)
    stats["test_tokens"] = len(test_tokens)
    
    # Save binary files
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    test_path = os.path.join(output_dir, "test.bin")
    
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    test_tokens.tofile(test_path)
    
    # Save statistics
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples processed: {stats['total_samples']:,}")
    print(f"Filtered (quality):      {stats['filtered_samples']:,}")
    print(f"Filtered (duplicate):    {stats['duplicate_samples']:,}")
    print(f"Samples kept:            {stats['kept_samples']:,}")
    print(f"Total tokens:            {stats['total_tokens']:,}")
    print()
    print(f"Train tokens: {stats['train_tokens']:,} ({100*train_ratio:.0f}%)")
    print(f"Val tokens:   {stats['val_tokens']:,} ({100*val_ratio:.0f}%)")
    print(f"Test tokens:  {stats['test_tokens']:,} ({100*test_ratio:.0f}%)")
    print()
    print(f"Saved to: {output_dir}")
    print(f"  - train.bin ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"  - val.bin ({os.path.getsize(val_path) / 1e6:.1f} MB)")
    print(f"  - test.bin ({os.path.getsize(test_path) / 1e6:.1f} MB)")
    print(f"  - stats.json")
    print("=" * 60)
    
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare training data from HuggingFace datasets or local files"
    )
    
    # Data source
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="HuggingFace dataset name")
    source.add_argument("--local", help="Path to local file or directory")
    
    # HF dataset options
    p.add_argument("--config", default=None,
                   help="HuggingFace dataset config (e.g., 'wikitext-2-raw-v1')")
    p.add_argument("--split", default="train",
                   help="Dataset split to use (default: train)")
    p.add_argument("--text_field", default="text",
                   help="Name of text field in dataset (default: text)")
    p.add_argument("--streaming", action="store_true",
                   help="Use streaming mode for large datasets")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Trust remote code when loading dataset")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Maximum number of samples to process")
    
    # Output
    p.add_argument("--output_dir", default="data/processed",
                   help="Output directory for processed files")
    
    # Splits
    p.add_argument("--train_ratio", type=float, default=0.90,
                   help="Fraction for training set (default: 0.90)")
    p.add_argument("--val_ratio", type=float, default=0.05,
                   help="Fraction for validation set (default: 0.05)")
    p.add_argument("--test_ratio", type=float, default=0.05,
                   help="Fraction for test set (default: 0.05)")
    
    # Filtering
    p.add_argument("--min_length", type=int, default=50,
                   help="Minimum text length in characters (default: 50)")
    p.add_argument("--max_length", type=int, default=100000,
                   help="Maximum text length in characters (default: 100000)")
    p.add_argument("--deduplicate", action="store_true",
                   help="Enable approximate deduplication via MinHash")
    p.add_argument("--dedup_threshold", type=float, default=0.8,
                   help="MinHash Jaccard threshold for deduplication (default: 0.8)")
    
    # Quick testing presets
    p.add_argument("--smoke", action="store_true",
                   help="Use minimal settings for quick testing")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Smoke test preset
    if args.smoke:
        args.max_samples = 100
        args.min_length = 10
    
    # Load texts
    if args.dataset:
        texts = load_hf_dataset(
            args.dataset,
            config=args.config,
            split=args.split,
            text_field=args.text_field,
            streaming=args.streaming,
            max_samples=args.max_samples,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        texts = load_local_files(args.local)
    
    # Process and save
    process_and_tokenize(
        texts,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_length=args.min_length,
        max_length=args.max_length,
        deduplicate=args.deduplicate,
        dedup_threshold=args.dedup_threshold,
    )


if __name__ == "__main__":
    main()
