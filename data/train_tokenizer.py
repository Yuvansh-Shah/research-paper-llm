#!/usr/bin/env python3
"""Train a domain tokenizer on the local DBLP + arXiv paper corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config_defaults
from data.prepare_papers import (
    DEFAULT_ARXIV_PATH,
    DEFAULT_DBLP_PATH,
    DEFAULT_OUTPUT_DIR,
    FormattingConfig,
    PaperRecord,
    QualityConfig,
    drop_reason,
    format_paper_for_training,
    split_categories,
    stream_arxiv_jsonl,
    stream_dblp_csv,
)
from tokenization import (
    DEFAULT_RESEARCH_TOKENIZER_DIR,
    TokenizerSpec,
    save_tokenizer_manifest,
)


def _ensure_tokenizers_dependency() -> None:
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "The 'tokenizers' package is required. Install dependencies from requirements.txt."
        ) from exc


def iter_records(
    dblp_path: Optional[Path],
    arxiv_path: Optional[Path],
    max_records_per_source: Optional[int],
) -> Iterator[PaperRecord]:
    if dblp_path and dblp_path.exists():
        yield from stream_dblp_csv(dblp_path, max_records=max_records_per_source)
    if arxiv_path and arxiv_path.exists():
        yield from stream_arxiv_jsonl(arxiv_path, max_records=max_records_per_source)


def iter_training_texts(
    dblp_path: Optional[Path],
    arxiv_path: Optional[Path],
    *,
    quality_config: QualityConfig,
    formatting_config: FormattingConfig,
    max_records_per_source: Optional[int],
    verbose: bool,
) -> tuple[Iterator[str], dict[str, int]]:
    stats = {
        "seen_records": 0,
        "kept_records": 0,
        "dropped_records": 0,
    }
    seen_sources: Counter[str] = Counter()
    kept_sources: Counter[str] = Counter()

    def _iterator() -> Iterator[str]:
        for record in iter_records(dblp_path, arxiv_path, max_records_per_source):
            stats["seen_records"] += 1
            seen_sources[record.source] += 1
            reason = drop_reason(record, quality_config)
            if reason is not None:
                stats["dropped_records"] += 1
                continue
            stats["kept_records"] += 1
            kept_sources[record.source] += 1
            if verbose and stats["seen_records"] % 50_000 == 0:
                print(
                    f"[train_tokenizer] seen={stats['seen_records']:,} "
                    f"kept={stats['kept_records']:,}"
                )
            yield format_paper_for_training(record, formatting_config)

    stats["seen_by_source"] = seen_sources  # type: ignore[assignment]
    stats["kept_by_source"] = kept_sources  # type: ignore[assignment]
    return _iterator(), stats


def _build_tokenizer(model_type: str, vocab_size: int, min_frequency: int):
    _ensure_tokenizers_dependency()
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers  # noqa: PLC0415

    special_tokens = ["[UNK]", "<|endoftext|>"]

    if model_type == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
        tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never")
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            unk_token="[UNK]",
            special_tokens=special_tokens,
        )
        return tokenizer, trainer

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
    )
    return tokenizer, trainer


def train_tokenizer(
    *,
    dblp_path: Optional[Path],
    arxiv_path: Optional[Path],
    output_dir: Path,
    quality_config: QualityConfig,
    formatting_config: FormattingConfig,
    max_records_per_source: Optional[int],
    vocab_size: int,
    min_frequency: int,
    model_type: str,
    name: str,
    verbose: bool,
) -> tuple[TokenizerSpec, dict[str, object]]:
    tokenizer, trainer = _build_tokenizer(model_type=model_type, vocab_size=vocab_size, min_frequency=min_frequency)
    texts, stats = iter_training_texts(
        dblp_path,
        arxiv_path,
        quality_config=quality_config,
        formatting_config=formatting_config,
        max_records_per_source=max_records_per_source,
        verbose=verbose,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    vocab = tokenizer.get_vocab()
    eos_token_id = vocab.get("<|endoftext|>")
    if eos_token_id is None:
        raise ValueError("Trained tokenizer is missing the <|endoftext|> token.")

    spec = TokenizerSpec(
        name=name,
        backend="hf_tokenizers",
        vocab_size=tokenizer.get_vocab_size(),
        eos_token_id=int(eos_token_id),
        tokenizer_path=str(tokenizer_path.relative_to(output_dir)),
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
        special_tokens=["[UNK]", "<|endoftext|>"],
    )

    training_metadata = {
        "model_type": model_type,
        "vocab_size": spec.vocab_size,
        "min_frequency": min_frequency,
        "max_records_per_source": max_records_per_source,
        "quality_config": vars(quality_config),
        "formatting_config": vars(formatting_config),
        "stats": {
            "seen_records": int(stats["seen_records"]),
            "kept_records": int(stats["kept_records"]),
            "dropped_records": int(stats["dropped_records"]),
            "seen_by_source": dict(stats["seen_by_source"]),
            "kept_by_source": dict(stats["kept_by_source"]),
        },
        "inputs": {
            "dblp": str(dblp_path.resolve()) if dblp_path else None,
            "arxiv": str(arxiv_path.resolve()) if arxiv_path else None,
        },
    }
    save_tokenizer_manifest(output_dir, spec, training_metadata=training_metadata)
    return spec, training_metadata


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    config_path, defaults = load_config_defaults(argv, section="tokenizer")
    parser = argparse.ArgumentParser(description="Train a domain tokenizer on DBLP + arXiv.")
    parser.add_argument("--config", type=Path, default=config_path, help="Optional JSON preset / recipe file.")
    parser.add_argument("--dblp", type=Path, default=None, help="Path to dblp-v10.csv.")
    parser.add_argument("--arxiv", type=Path, default=None, help="Path to arxiv-metadata-oai-snapshot.json.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RESEARCH_TOKENIZER_DIR)
    parser.add_argument("--name", default="research-papers-bpe32k")
    parser.add_argument("--model_type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--max_records_per_source", type=int, default=200_000)
    parser.add_argument("--sample", type=int, default=None, help="Shortcut for --max_records_per_source.")
    parser.add_argument("--include_authors", action="store_true")
    parser.add_argument("--categories", default=None, help="Optional comma-separated arXiv category filter.")
    parser.add_argument("--min_title_chars", type=int, default=12)
    parser.add_argument("--min_abstract_chars", type=int, default=80)
    parser.add_argument("--max_abstract_chars", type=int, default=12_000)
    parser.add_argument("--min_alpha_ratio", type=float, default=0.60)
    parser.add_argument("--max_junk_ratio", type=float, default=0.12)
    parser.add_argument("--max_repeated_char_ratio", type=float, default=0.15)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    dblp_path = args.dblp or (DEFAULT_DBLP_PATH if DEFAULT_DBLP_PATH.exists() else None)
    arxiv_path = args.arxiv or (DEFAULT_ARXIV_PATH if DEFAULT_ARXIV_PATH.exists() else None)
    max_records_per_source = args.sample if args.sample is not None else args.max_records_per_source
    allowed_prefixes = ()
    if args.categories:
        allowed_prefixes = tuple(
            piece.strip() for piece in args.categories.split(",") if piece.strip()
        )

    quality_config = QualityConfig(
        min_title_chars=args.min_title_chars,
        min_abstract_chars=args.min_abstract_chars,
        max_abstract_chars=args.max_abstract_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        max_junk_ratio=args.max_junk_ratio,
        max_repeated_char_ratio=args.max_repeated_char_ratio,
        allowed_category_prefixes=allowed_prefixes,
    )
    formatting_config = FormattingConfig(include_authors=args.include_authors)

    try:
        spec, metadata = train_tokenizer(
            dblp_path=dblp_path,
            arxiv_path=arxiv_path,
            output_dir=args.output_dir,
            quality_config=quality_config,
            formatting_config=formatting_config,
            max_records_per_source=max_records_per_source,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            model_type=args.model_type,
            name=args.name,
            verbose=not args.quiet,
        )
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"[train_tokenizer] error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "tokenizer": spec.to_dict(),
                "training": metadata,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
