#!/usr/bin/env python3
"""Single-command corpus builder: locate raw corpora, train the tokenizer, tokenise shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_papers import (
    DEFAULT_OUTPUT_DIR,
    FormattingConfig,
    QualityConfig,
    SplitConfig,
    process_corpus,
)
from data.train_tokenizer import train_tokenizer
from tokenization import DEFAULT_RESEARCH_TOKENIZER_DIR
from tokenization.tokenizer import TOKENIZER_MANIFEST_NAME

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DBLP_FILENAME = "dblp-v10.csv"
ARXIV_FILENAME = "arxiv-metadata-oai-snapshot.json"
TOKENIZER_TRAINING_MAX_RECORDS = 200_000


def locate_corpus_file(raw_dir: Path, filename: str) -> Optional[Path]:
    candidate = raw_dir / filename
    if candidate.is_file():
        return candidate
    if raw_dir.is_dir():
        return next((path for path in raw_dir.rglob(filename) if path.is_file()), None)
    return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the research-paper training corpus end to end."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing {DBLP_FILENAME} and/or {ARXIV_FILENAME}.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for token shards, manifests, and stats.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_RESEARCH_TOKENIZER_DIR,
        help="Tokenizer directory. Training is skipped when a manifest already exists there.",
    )
    parser.add_argument("--vocab-size", type=int, default=32_000, help="BPE vocabulary size.")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Validation split fraction.")
    parser.add_argument("--test-fraction", type=float, default=0.05, help="Test split fraction.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit records read from each source (for quick runs).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    dblp_path = locate_corpus_file(args.raw_dir, DBLP_FILENAME)
    arxiv_path = locate_corpus_file(args.raw_dir, ARXIV_FILENAME)
    if dblp_path is None:
        print(
            f"[build_corpus] warning: {DBLP_FILENAME} not found under {args.raw_dir}, "
            "skipping the DBLP source.",
            file=sys.stderr,
        )
    if arxiv_path is None:
        print(
            f"[build_corpus] warning: {ARXIV_FILENAME} not found under {args.raw_dir}, "
            "skipping the arXiv source.",
            file=sys.stderr,
        )
    if dblp_path is None and arxiv_path is None:
        print(
            f"[build_corpus] error: no raw corpora found under {args.raw_dir}. "
            f"Expected {DBLP_FILENAME} and/or {ARXIV_FILENAME}.",
            file=sys.stderr,
        )
        return 1

    manifest_path = args.tokenizer_dir / TOKENIZER_MANIFEST_NAME
    if manifest_path.exists():
        print(f"[build_corpus] reusing existing tokenizer at {args.tokenizer_dir}")
    else:
        print(f"[build_corpus] training BPE tokenizer into {args.tokenizer_dir} ...")
        try:
            train_tokenizer(
                dblp_path=dblp_path,
                arxiv_path=arxiv_path,
                output_dir=args.tokenizer_dir,
                quality_config=QualityConfig(),
                formatting_config=FormattingConfig(),
                max_records_per_source=(
                    args.max_records
                    if args.max_records is not None
                    else TOKENIZER_TRAINING_MAX_RECORDS
                ),
                vocab_size=args.vocab_size,
                min_frequency=2,
                model_type="bpe",
                name=args.tokenizer_dir.name.replace("_", "-"),
                verbose=True,
            )
        except Exception as exc:  # pragma: no cover - CLI error path
            print(f"[build_corpus] tokenizer training error: {exc}", file=sys.stderr)
            return 1

    split_config = SplitConfig(
        train_ratio=1.0 - args.val_fraction - args.test_fraction,
        val_ratio=args.val_fraction,
        test_ratio=args.test_fraction,
    )
    try:
        result = process_corpus(
            dblp_path=dblp_path,
            arxiv_path=arxiv_path,
            output_dir=args.out_dir,
            split_config=split_config,
            max_records_per_source=args.max_records,
            tokenizer=args.tokenizer_dir,
            verbose=True,
        )
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"[build_corpus] corpus preparation error: {exc}", file=sys.stderr)
        return 1

    stats = result.stats
    dropped_records = stats.total_records_seen - stats.kept_records
    print("[build_corpus] corpus build complete")
    print(
        f"[build_corpus] records: seen={stats.total_records_seen:,} "
        f"kept={stats.kept_records:,} dropped={dropped_records:,}"
    )
    for split in ("train", "val", "test"):
        print(
            f"[build_corpus] {split}: {stats.records_by_split[split]:,} records, "
            f"{stats.tokens_by_split[split]:,} tokens"
        )
    print(f"[build_corpus] tokenizer: {result.tokenizer_name} ({args.tokenizer_dir})")
    print(f"[build_corpus] corpus manifest: {result.corpus_manifest_path}")
    print(f"[build_corpus] stats: {result.stats_path}")
    for split, manifest in sorted(result.split_manifests.items()):
        print(f"[build_corpus] {split} manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
