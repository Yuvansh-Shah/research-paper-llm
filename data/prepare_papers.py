#!/usr/bin/env python3
"""Streaming preparation pipeline for the local research-paper corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import struct
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config_defaults
from tokenization import (
    TokenizerSpec,
    default_gpt2_tokenizer_spec,
    load_runtime_tokenizer,
    load_tokenizer_spec,
    maybe_load_recommended_tokenizer_spec,
)

DEFAULT_DBLP_PATH = PROJECT_ROOT / "dblp-v10.csv"
DEFAULT_ARXIV_PATH = PROJECT_ROOT / "arxiv-metadata-oai-snapshot.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "papers"

MOJIBAKE_MARKERS = ("Ãƒ", "Ã¢", "Ã‚", "â‚¬", "â„¢", "Å“", "Å¾", "\ufffd")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
LATEX_INLINE_RE = re.compile(r"\$[^$]+\$")
LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+(?:\[[^\]]*\])?\{[^{}]*\}")
LATEX_TOKEN_RE = re.compile(r"\\[A-Za-z]+")
MULTISPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"(19|20)\d{2}")
BUCKET_HEADER = struct.Struct("<QII")


def _configure_csv_field_limit() -> None:
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:  # pragma: no cover - platform-specific
            max_size //= 10


_configure_csv_field_limit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash_int(text: str, digest_size: int = 16) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=digest_size)
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_key_text(text: str) -> str:
    text = _ascii_fold(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def maybe_fix_mojibake(text: str) -> str:
    if not text or not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    old_bad = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    new_bad = sum(repaired.count(marker) for marker in MOJIBAKE_MARKERS)
    return repaired if new_bad < old_bad else text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = maybe_fix_mojibake(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\\n", "\n").replace("\\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = LATEX_INLINE_RE.sub(" ", text)
    text = LATEX_CMD_RE.sub(" ", text)
    text = LATEX_TOKEN_RE.sub(" ", text)
    text = text.replace("~", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def split_categories(raw_categories: str) -> tuple[str, ...]:
    if not raw_categories:
        return ()
    parts = re.split(r"[\s,]+", raw_categories.strip())
    return tuple(part for part in parts if part)


def clean_year(value: str) -> str:
    if not value:
        return ""
    match = YEAR_RE.search(str(value))
    return match.group(0) if match else ""


def extract_arxiv_year(payload: dict[str, object]) -> str:
    update_date = clean_year(str(payload.get("update_date", "")))
    if update_date:
        return update_date
    versions = payload.get("versions")
    if isinstance(versions, list) and versions:
        first = versions[0]
        if isinstance(first, dict):
            return clean_year(str(first.get("created", "")))
    return ""


def compute_alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(char.isalpha() for char in text)
    return alpha / max(1, len(text))


def compute_junk_ratio(text: str) -> float:
    if not text:
        return 1.0
    allowed_punct = set(".,;:!?()[]{}\"'/-_@%&*=+<>")
    junk = sum(
        1
        for char in text
        if not (char.isalnum() or char.isspace() or char in allowed_punct)
    )
    return junk / max(1, len(text))


def compute_repeated_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    longest = 1
    current = 1
    previous = text[0]
    for char in text[1:]:
        if char == previous:
            current += 1
            longest = max(longest, current)
        else:
            previous = char
            current = 1
    return longest / max(1, len(text))


def title_signature(title: str) -> str:
    normalized = normalize_key_text(title)
    words = normalized.split()
    return " ".join(words[:24])


def abstract_signature(abstract: str, token_window: int = 160) -> str:
    normalized = normalize_key_text(abstract)
    words = normalized.split()
    return " ".join(words[:token_window])


def category_matches(categories: tuple[str, ...], allowed_prefixes: tuple[str, ...]) -> bool:
    if not allowed_prefixes:
        return True
    for category in categories:
        for prefix in allowed_prefixes:
            if category == prefix or category.startswith(f"{prefix}."):
                return True
    return False


def parse_weight_mapping(raw: Optional[str]) -> dict[str, float]:
    if not raw:
        return {}
    mapping: dict[str, float] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight mapping item: {item!r}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = float(value)
    return mapping


@dataclass(frozen=True)
class PaperRecord:
    source: str
    source_id: str
    title: str
    abstract: str
    authors: str = ""
    year: str = ""
    venue: str = ""
    categories: tuple[str, ...] = ()

    def dedup_key(self) -> int:
        title_key = title_signature(self.title) or normalize_key_text(self.title)
        abstract_key = abstract_signature(self.abstract)
        return _stable_hash_int(f"{title_key}\n{abstract_key}", digest_size=16)

    def leakage_group_key(self) -> int:
        title_key = title_signature(self.title)
        if title_key:
            return _stable_hash_int(title_key, digest_size=16)
        return self.dedup_key()

    @property
    def domains(self) -> tuple[str, ...]:
        domains = {category.split(".", 1)[0] for category in self.categories if category}
        return tuple(sorted(domains))

    @property
    def primary_domain(self) -> str:
        domains = self.domains
        return domains[0] if domains else ""


@dataclass
class QualityConfig:
    min_title_chars: int = 12
    min_abstract_chars: int = 80
    max_abstract_chars: int = 12_000
    min_alpha_ratio: float = 0.60
    max_junk_ratio: float = 0.12
    max_repeated_char_ratio: float = 0.15
    allowed_category_prefixes: tuple[str, ...] = ()


@dataclass
class FormattingConfig:
    include_year: bool = True
    include_venue: bool = True
    include_categories: bool = True
    include_authors: bool = False


@dataclass
class SplitConfig:
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {self.train_ratio} + "
                f"{self.val_ratio} + {self.test_ratio} = {total}"
            )


@dataclass
class MixingConfig:
    shuffle_buckets: int = 256
    source_weights: dict[str, float] = field(default_factory=dict)
    domain_weights: dict[str, float] = field(default_factory=dict)
    category_weights: dict[str, float] = field(default_factory=dict)
    cleanup_temp_files: bool = True

    def validate(self) -> None:
        if self.shuffle_buckets <= 0 or self.shuffle_buckets & (self.shuffle_buckets - 1):
            raise ValueError("shuffle_buckets must be a positive power of two.")
        for name, mapping in (
            ("source_weights", self.source_weights),
            ("domain_weights", self.domain_weights),
            ("category_weights", self.category_weights),
        ):
            for key, value in mapping.items():
                if value < 0.0:
                    raise ValueError(f"{name}[{key!r}] must be non-negative, got {value}.")

    @property
    def normalization_denominator(self) -> float:
        candidates = [1.0]
        candidates.extend(value for value in self.source_weights.values() if value > 0.0)
        candidates.extend(value for value in self.domain_weights.values() if value > 0.0)
        candidates.extend(value for value in self.category_weights.values() if value > 0.0)
        return max(candidates)

    def keep_probability(self, record: PaperRecord) -> float:
        weight = self.source_weights.get(record.source, 1.0)
        if record.domains:
            domain_weight = max(self.domain_weights.get(domain, 1.0) for domain in record.domains)
        else:
            domain_weight = 1.0
        if record.categories:
            category_weight = max(
                self.category_weights.get(category, 1.0) for category in record.categories
            )
        else:
            category_weight = 1.0
        keep_probability = (weight * domain_weight * category_weight) / self.normalization_denominator
        return max(0.0, min(1.0, keep_probability))


@dataclass
class ShardMetadata:
    path: str
    token_count: int


@dataclass
class ProcessingStats:
    started_at: str = field(default_factory=_now_iso)
    finished_at: str = ""
    total_records_seen: int = 0
    kept_records: int = 0
    seen_by_source: Counter[str] = field(default_factory=Counter)
    kept_by_source: Counter[str] = field(default_factory=Counter)
    dropped_by_reason: Counter[str] = field(default_factory=Counter)
    dropped_by_source: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    tokens_by_source: Counter[str] = field(default_factory=Counter)
    tokens_by_split: Counter[str] = field(default_factory=Counter)
    records_by_split: Counter[str] = field(default_factory=Counter)
    split_source_tokens: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    split_source_records: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    categories_seen: Counter[str] = field(default_factory=Counter)
    domains_seen: Counter[str] = field(default_factory=Counter)
    venues_seen: Counter[str] = field(default_factory=Counter)
    kept_probability_sum: Counter[str] = field(default_factory=Counter)

    def note_seen(self, source: str) -> None:
        self.total_records_seen += 1
        self.seen_by_source[source] += 1

    def note_drop(self, source: str, reason: str) -> None:
        self.dropped_by_reason[reason] += 1
        self.dropped_by_source[source][reason] += 1

    def note_keep(
        self,
        split: str,
        record: PaperRecord,
        token_count: int,
        keep_probability: float,
    ) -> None:
        self.kept_records += 1
        self.kept_by_source[record.source] += 1
        self.tokens_by_source[record.source] += token_count
        self.tokens_by_split[split] += token_count
        self.records_by_split[split] += 1
        self.split_source_tokens[split][record.source] += token_count
        self.split_source_records[split][record.source] += 1
        self.kept_probability_sum[record.source] += keep_probability
        if record.venue:
            self.venues_seen[record.venue] += 1
        for category in record.categories:
            self.categories_seen[category] += 1
            self.domains_seen[category.split(".", 1)[0]] += 1

    def finish(self) -> None:
        self.finished_at = _now_iso()

    def to_dict(self) -> dict[str, object]:
        def _nested(source: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
            return {key: dict(counter) for key, counter in source.items()}

        kept_probability_avg = {
            source: self.kept_probability_sum[source] / max(1, self.kept_by_source[source])
            for source in self.kept_by_source
        }
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_records_seen": self.total_records_seen,
            "kept_records": self.kept_records,
            "seen_by_source": dict(self.seen_by_source),
            "kept_by_source": dict(self.kept_by_source),
            "dropped_by_reason": dict(self.dropped_by_reason),
            "dropped_by_source": _nested(self.dropped_by_source),
            "tokens_by_source": dict(self.tokens_by_source),
            "tokens_by_split": dict(self.tokens_by_split),
            "records_by_split": dict(self.records_by_split),
            "split_source_tokens": _nested(self.split_source_tokens),
            "split_source_records": _nested(self.split_source_records),
            "avg_keep_probability_by_source": kept_probability_avg,
            "top_categories": dict(self.categories_seen.most_common(50)),
            "top_domains": dict(self.domains_seen.most_common(20)),
            "top_venues": dict(self.venues_seen.most_common(20)),
        }

    def render_report(self) -> str:
        kept_pct = 100.0 * self.kept_records / max(1, self.total_records_seen)
        lines = [
            "=" * 72,
            "RESEARCH CORPUS PREPROCESSING REPORT",
            "=" * 72,
            f"Total records seen: {self.total_records_seen:,}",
            f"Records kept:      {self.kept_records:,} ({kept_pct:.1f}%)",
            "",
            "Kept by source:",
        ]
        for source, count in sorted(self.kept_by_source.items()):
            avg_keep = self.kept_probability_sum[source] / max(1, count)
            lines.append(
                f"  - {source}: {count:,} records, {self.tokens_by_source[source]:,} tokens, "
                f"avg keep_prob={avg_keep:.2f}"
            )

        lines.append("")
        lines.append("Dropped by reason:")
        for reason, count in self.dropped_by_reason.most_common():
            lines.append(f"  - {reason}: {count:,}")

        lines.append("")
        lines.append("Split sizes:")
        for split in ("train", "val", "test"):
            lines.append(
                f"  - {split}: {self.records_by_split[split]:,} records, "
                f"{self.tokens_by_split[split]:,} tokens"
            )

        if self.categories_seen:
            lines.append("")
            lines.append("Top categories:")
            for category, count in self.categories_seen.most_common(12):
                lines.append(f"  - {category}: {count:,}")

        if self.domains_seen:
            lines.append("")
            lines.append("Top domains:")
            for domain, count in self.domains_seen.most_common(12):
                lines.append(f"  - {domain}: {count:,}")

        lines.append("=" * 72)
        return "\n".join(lines)


@dataclass
class CorpusBuildResult:
    stats: ProcessingStats
    split_manifests: dict[str, Path]
    stats_path: Path
    corpus_manifest_path: Path
    tokenizer_name: str
    tokenizer_spec: Optional[TokenizerSpec]


def get_row_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name])
    return ""


def stream_dblp_csv(path: str | Path, max_records: Optional[int] = None) -> Iterator[PaperRecord]:
    """Stream DBLP papers directly from the CSV corpus."""

    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if max_records is not None and index >= max_records:
                break
            yield PaperRecord(
                source="dblp",
                source_id=get_row_value(row, "id") or f"dblp:{index}",
                title=clean_text(get_row_value(row, "title")),
                abstract=clean_text(get_row_value(row, "abstract")),
                authors=clean_text(get_row_value(row, "authors")),
                year=clean_year(get_row_value(row, "year")),
                venue=clean_text(get_row_value(row, "venue")),
            )


def stream_arxiv_jsonl(
    path: str | Path, max_records: Optional[int] = None
) -> Iterator[PaperRecord]:
    """Stream arXiv metadata from newline-delimited JSON."""

    path = Path(path)
    emitted = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if max_records is not None and emitted >= max_records:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            emitted += 1
            yield PaperRecord(
                source="arxiv",
                source_id=str(payload.get("id") or f"arxiv:{line_number}"),
                title=clean_text(str(payload.get("title", ""))),
                abstract=clean_text(str(payload.get("abstract", ""))),
                authors=clean_text(str(payload.get("authors", ""))),
                year=extract_arxiv_year(payload),
                categories=split_categories(str(payload.get("categories", ""))),
            )


def drop_reason(record: PaperRecord, config: QualityConfig) -> Optional[str]:
    if not record.title.strip():
        return "missing_title"
    if len(record.title) < config.min_title_chars or compute_alpha_ratio(record.title) < 0.50:
        return "weak_title"
    if not record.abstract.strip():
        return "missing_abstract"
    if len(record.abstract) < config.min_abstract_chars:
        return "short_abstract"
    if len(record.abstract) > config.max_abstract_chars:
        return "overlong_abstract"
    if compute_alpha_ratio(record.abstract) < config.min_alpha_ratio:
        return "low_alpha_ratio"
    if compute_junk_ratio(record.abstract) > config.max_junk_ratio:
        return "high_junk_ratio"
    if compute_repeated_char_ratio(record.abstract) > config.max_repeated_char_ratio:
        return "repeated_characters"
    if config.allowed_category_prefixes and record.source == "arxiv":
        if not category_matches(record.categories, config.allowed_category_prefixes):
            return "category_filter"
    return None


def format_authors(raw_authors: str) -> str:
    if not raw_authors:
        return ""
    authors = raw_authors.strip()
    if authors.startswith("[") and authors.endswith("]"):
        authors = authors[1:-1]
    authors = authors.replace("', '", ", ").replace('"', "")
    authors = authors.replace("'", "").strip()
    return MULTISPACE_RE.sub(" ", authors)


def format_paper_for_training(record: PaperRecord, config: FormattingConfig) -> str:
    """Format papers into a consistent research-oriented training template."""

    lines = [f"Title: {record.title}"]

    if config.include_authors and record.authors:
        authors = format_authors(record.authors)
        if authors:
            lines.append(f"Authors: {authors}")
    if config.include_year and record.year:
        lines.append(f"Year: {record.year}")
    if config.include_venue and record.venue:
        lines.append(f"Venue: {record.venue}")
    if config.include_categories and record.categories:
        lines.append(f"Categories: {' '.join(record.categories)}")

    lines.append("")
    lines.append("Abstract:")
    lines.append(record.abstract)
    return "\n".join(lines).strip()


def build_gpt2_encoder() -> tuple[Callable[[str], list[int]], str, Optional[TokenizerSpec]]:
    spec = default_gpt2_tokenizer_spec()
    tokenizer = load_runtime_tokenizer(spec)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_eos=True)

    return encode, spec.name, spec


def assign_split(group_key: int, config: SplitConfig, seed: int) -> str:
    bucket = _stable_hash_int(f"{seed}:{group_key}", digest_size=8) / float(2**64)
    if bucket < config.train_ratio:
        return "train"
    if bucket < config.train_ratio + config.val_ratio:
        return "val"
    return "test"


class BucketedDocumentShuffle:
    """External, deterministic document shuffle that stays bounded in RAM."""

    def __init__(
        self,
        temp_dir: Path,
        split: str,
        bucket_count: int,
        seed: int,
    ) -> None:
        self.temp_dir = temp_dir
        self.split = split
        self.bucket_count = bucket_count
        self.bucket_bits = int(math.log2(bucket_count))
        self.seed = seed
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[int, object] = {}

    def _bucket_path(self, bucket_id: int) -> Path:
        return self.temp_dir / f"{self.split}-bucket-{bucket_id:04d}.bin"

    def _bucket_id_for_key(self, shuffle_key: int) -> int:
        return shuffle_key >> (64 - self.bucket_bits)

    def add_document(self, record: PaperRecord, tokens: list[int], metadata: dict[str, object]) -> None:
        shuffle_key = _stable_hash_int(
            f"{self.seed}:{record.source}:{record.source_id}:{record.dedup_key()}",
            digest_size=8,
        )
        bucket_id = self._bucket_id_for_key(shuffle_key)
        handle = self._handles.get(bucket_id)
        if handle is None:
            handle = self._bucket_path(bucket_id).open("ab")
            self._handles[bucket_id] = handle
        metadata_json = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        token_array = np.asarray(tokens, dtype=np.uint16)
        handle.write(BUCKET_HEADER.pack(shuffle_key, len(metadata_json), int(token_array.size)))
        handle.write(metadata_json)
        handle.write(token_array.tobytes())

    def iter_documents(self) -> Iterator[tuple[np.ndarray, dict[str, object]]]:
        self.close()
        for bucket_id in range(self.bucket_count):
            bucket_path = self._bucket_path(bucket_id)
            if not bucket_path.exists():
                continue
            documents: list[tuple[int, np.ndarray, dict[str, object]]] = []
            with bucket_path.open("rb") as handle:
                while True:
                    header = handle.read(BUCKET_HEADER.size)
                    if not header:
                        break
                    shuffle_key, metadata_len, token_count = BUCKET_HEADER.unpack(header)
                    metadata_payload = handle.read(metadata_len)
                    token_bytes = handle.read(token_count * 2)
                    metadata = json.loads(metadata_payload.decode("utf-8"))
                    tokens = np.frombuffer(token_bytes, dtype=np.uint16).copy()
                    documents.append((shuffle_key, tokens, metadata))

            documents.sort(key=lambda item: item[0])
            for _, tokens, metadata in documents:
                yield tokens, metadata

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def cleanup(self) -> None:
        self.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class SplitCorpusWriter:
    """Write shuffled documents into split-specific shards plus document metadata."""

    def __init__(
        self,
        output_dir: Path,
        split: str,
        shard_size_tokens: int = 50_000_000,
        buffer_tokens: int = 1_000_000,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.shard_size_tokens = shard_size_tokens
        self.buffer_tokens = buffer_tokens
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.document_index_path = self.output_dir / f"{self.split}.documents.jsonl"
        self.document_index_handle = self.document_index_path.open("w", encoding="utf-8")
        self.buffer_docs: list[tuple[np.ndarray, dict[str, object]]] = []
        self.buffer_token_count = 0
        self.total_tokens = 0
        self.document_count = 0
        self._current_shard_index = 0
        self._current_shard_tokens = 0
        self._shards: list[ShardMetadata] = []
        self._current_handle = None
        self._current_path: Optional[Path] = None

    def _current_shard_path(self) -> Path:
        return self.output_dir / f"{self.split}-{self._current_shard_index:05d}.bin"

    def _ensure_shard(self) -> None:
        if self._current_handle is not None:
            return
        self._current_path = self._current_shard_path()
        self._current_handle = self._current_path.open("wb")
        self._shards.append(ShardMetadata(path=self._current_path.name, token_count=0))

    def _rotate_shard(self) -> None:
        if self._current_handle is not None:
            self._current_handle.close()
        self._current_handle = None
        self._current_path = None
        self._current_shard_index += 1
        self._current_shard_tokens = 0

    def write_document(self, tokens: np.ndarray, metadata: dict[str, object]) -> int:
        token_array = np.asarray(tokens, dtype=np.uint16)
        self.buffer_docs.append((token_array, metadata))
        self.buffer_token_count += int(token_array.size)
        if self.buffer_token_count >= self.buffer_tokens:
            self._flush_buffer()
        return int(token_array.size)

    def _flush_buffer(self) -> None:
        if not self.buffer_docs:
            return

        for token_array, metadata in self.buffer_docs:
            token_count = int(token_array.size)
            if token_count == 0:
                continue
            if self._current_handle is not None and (
                self._current_shard_tokens + token_count > self.shard_size_tokens
            ):
                self._rotate_shard()

            self._ensure_shard()
            shard = self._shards[-1]
            shard_token_start = self._current_shard_tokens
            global_token_start = self.total_tokens

            token_array.tofile(self._current_handle)

            self._current_shard_tokens += token_count
            self.total_tokens += token_count
            shard.token_count += token_count
            self.document_count += 1

            document_payload = dict(metadata)
            document_payload.update(
                {
                    "split": self.split,
                    "token_count": token_count,
                    "global_token_start": global_token_start,
                    "global_token_end": self.total_tokens,
                    "shard": shard.path,
                    "shard_token_start": shard_token_start,
                    "shard_token_end": self._current_shard_tokens,
                }
            )
            self.document_index_handle.write(
                json.dumps(document_payload, ensure_ascii=False) + "\n"
            )

        self.buffer_docs.clear()
        self.buffer_token_count = 0

    def finalize(
        self,
        tokenizer_name: str,
        tokenizer_spec: Optional[TokenizerSpec],
    ) -> Path:
        self._flush_buffer()
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None
        self.document_index_handle.close()

        manifest = {
            "format": "token-shards/v2",
            "split": self.split,
            "dtype": "uint16",
            "tokenizer": tokenizer_name,
            "tokenizer_spec": tokenizer_spec.to_dict() if tokenizer_spec else None,
            "document_count": self.document_count,
            "token_count": self.total_tokens,
            "document_index": self.document_index_path.name,
            "shards": [asdict(shard) for shard in self._shards],
        }
        manifest_path = self.output_dir / f"{self.split}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def close(self) -> None:
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None
        if not self.document_index_handle.closed:
            self.document_index_handle.close()


def iter_sources(
    dblp_path: Optional[Path],
    arxiv_path: Optional[Path],
    max_records_per_source: Optional[int],
) -> list[tuple[str, Iterator[PaperRecord]]]:
    sources: list[tuple[str, Iterator[PaperRecord]]] = []
    if dblp_path and dblp_path.exists():
        sources.append(("dblp", stream_dblp_csv(dblp_path, max_records=max_records_per_source)))
    if arxiv_path and arxiv_path.exists():
        sources.append(("arxiv", stream_arxiv_jsonl(arxiv_path, max_records=max_records_per_source)))
    return sources


def _resolve_encoder(
    *,
    tokenizer: Optional[str | Path | TokenizerSpec],
    encode_fn: Optional[Callable[[str], list[int]]],
    tokenizer_name: Optional[str],
) -> tuple[Callable[[str], list[int]], str, Optional[TokenizerSpec]]:
    if encode_fn is not None:
        return encode_fn, tokenizer_name or "custom", None

    spec: Optional[TokenizerSpec]
    if tokenizer is not None:
        spec = load_tokenizer_spec(tokenizer)
    else:
        spec = maybe_load_recommended_tokenizer_spec()
        if spec is None:
            spec = default_gpt2_tokenizer_spec()

    runtime_tokenizer = load_runtime_tokenizer(spec)

    def encode(text: str) -> list[int]:
        return runtime_tokenizer.encode(text, add_eos=True)

    return encode, spec.name, spec


def process_corpus(
    dblp_path: Optional[str | Path] = None,
    arxiv_path: Optional[str | Path] = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split_config: SplitConfig | None = None,
    quality_config: QualityConfig | None = None,
    formatting_config: FormattingConfig | None = None,
    mixing_config: MixingConfig | None = None,
    deduplicate: bool = True,
    max_records_per_source: Optional[int] = None,
    shard_size_tokens: int = 50_000_000,
    buffer_tokens: int = 1_000_000,
    seed: int = 42,
    verbose: bool = True,
    encode_fn: Optional[Callable[[str], list[int]]] = None,
    tokenizer: Optional[str | Path | TokenizerSpec] = None,
    tokenizer_name: Optional[str] = None,
) -> CorpusBuildResult:
    """Stream, clean, deduplicate, sample, shuffle, split, and tokenise papers."""

    split_config = split_config or SplitConfig()
    split_config.validate()
    quality_config = quality_config or QualityConfig()
    formatting_config = formatting_config or FormattingConfig()
    mixing_config = mixing_config or MixingConfig()
    mixing_config.validate()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encode, tokenizer_name, tokenizer_spec = _resolve_encoder(
        tokenizer=tokenizer,
        encode_fn=encode_fn,
        tokenizer_name=tokenizer_name,
    )

    stats = ProcessingStats()
    split_writers = {
        split: SplitCorpusWriter(
            output_dir=output_dir,
            split=split,
            shard_size_tokens=shard_size_tokens,
            buffer_tokens=buffer_tokens,
        )
        for split in ("train", "val", "test")
    }
    bucket_dir = output_dir / ".shuffle"
    split_buckets = {
        split: BucketedDocumentShuffle(
            temp_dir=bucket_dir / split,
            split=split,
            bucket_count=mixing_config.shuffle_buckets,
            seed=seed,
        )
        for split in ("train", "val", "test")
    }

    dblp_path = Path(dblp_path) if dblp_path else None
    arxiv_path = Path(arxiv_path) if arxiv_path else None
    if dblp_path is not None and not dblp_path.exists():
        print(
            f"[prepare_papers] warning: DBLP input not found, skipping source: {dblp_path}",
            file=sys.stderr,
        )
        dblp_path = None
    if arxiv_path is not None and not arxiv_path.exists():
        print(
            f"[prepare_papers] warning: arXiv input not found, skipping source: {arxiv_path}",
            file=sys.stderr,
        )
        arxiv_path = None
    sources = iter_sources(dblp_path, arxiv_path, max_records_per_source)
    if not sources:
        raise FileNotFoundError(
            "No valid input datasets found. Provide --dblp and/or --arxiv, or place "
            "dblp-v10.csv and arxiv-metadata-oai-snapshot.json in the project root."
        )

    seen_document_keys: dict[int, str] = {}

    try:
        for source_name, records in sources:
            if verbose:
                print(f"[prepare_papers] scanning {source_name} ...")
            for record in records:
                stats.note_seen(record.source)

                reason = drop_reason(record, quality_config)
                if reason is not None:
                    stats.note_drop(record.source, reason)
                    continue

                document_key = record.dedup_key()
                previous_source = seen_document_keys.get(document_key)
                if deduplicate and previous_source is not None:
                    if previous_source == record.source:
                        stats.note_drop(record.source, "duplicate_within_source")
                    else:
                        stats.note_drop(record.source, "duplicate_across_sources")
                    continue
                seen_document_keys.setdefault(document_key, record.source)

                keep_probability = mixing_config.keep_probability(record)
                if keep_probability <= 0.0:
                    stats.note_drop(record.source, "weighted_sampling_drop")
                    continue
                keep_bucket = _stable_hash_int(
                    f"{seed}:sample:{record.source}:{record.source_id}:{document_key}",
                    digest_size=8,
                ) / float(2**64)
                if keep_bucket > keep_probability:
                    stats.note_drop(record.source, "weighted_sampling_drop")
                    continue

                split = assign_split(record.leakage_group_key(), split_config, seed)
                document_text = format_paper_for_training(record, formatting_config)
                token_ids = encode(document_text)
                if not token_ids:
                    stats.note_drop(record.source, "empty_after_tokenization")
                    continue

                metadata = {
                    "source": record.source,
                    "source_id": record.source_id,
                    "title": record.title,
                    "year": record.year,
                    "venue": record.venue,
                    "categories": list(record.categories),
                    "domains": list(record.domains),
                    "primary_domain": record.primary_domain,
                }
                split_buckets[split].add_document(record, token_ids, metadata)
                stats.note_keep(split, record, len(token_ids), keep_probability)

                if verbose and stats.total_records_seen % 50_000 == 0:
                    print(
                        f"[prepare_papers] seen={stats.total_records_seen:,} "
                        f"kept={stats.kept_records:,} "
                        f"train_tokens={stats.tokens_by_split['train']:,}"
                    )

        if verbose:
            print("[prepare_papers] finalising deterministic shuffle buckets ...")

        split_manifests: dict[str, Path] = {}
        for split in ("train", "val", "test"):
            if verbose:
                print(f"[prepare_papers] writing shuffled {split} split ...")
            for tokens, metadata in split_buckets[split].iter_documents():
                split_writers[split].write_document(tokens, metadata)
            split_manifests[split] = split_writers[split].finalize(
                tokenizer_name=tokenizer_name,
                tokenizer_spec=tokenizer_spec,
            )
    finally:
        for bucket in split_buckets.values():
            if mixing_config.cleanup_temp_files:
                bucket.cleanup()
            else:
                bucket.close()
        for writer in split_writers.values():
            writer.close()

    stats.finish()

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")

    corpus_manifest = {
        "format": "research-paper-corpus/v2",
        "created_at": _now_iso(),
        "tokenizer": tokenizer_name,
        "tokenizer_spec": tokenizer_spec.to_dict() if tokenizer_spec else None,
        "inputs": {
            "dblp": str(dblp_path.resolve()) if dblp_path else None,
            "arxiv": str(arxiv_path.resolve()) if arxiv_path else None,
        },
        "split_config": asdict(split_config),
        "quality_config": asdict(quality_config),
        "formatting_config": asdict(formatting_config),
        "mixing_config": asdict(mixing_config),
        "deduplicate": deduplicate,
        "shard_size_tokens": shard_size_tokens,
        "buffer_tokens": buffer_tokens,
        "stats_path": stats_path.name,
        "splits": {split: manifest.name for split, manifest in split_manifests.items()},
    }
    corpus_manifest_path = output_dir / "corpus_manifest.json"
    corpus_manifest_path.write_text(
        json.dumps(corpus_manifest, indent=2), encoding="utf-8"
    )

    if verbose:
        print(stats.render_report())
        print(f"[prepare_papers] split manifests written to {output_dir}")

    return CorpusBuildResult(
        stats=stats,
        split_manifests=split_manifests,
        stats_path=stats_path,
        corpus_manifest_path=corpus_manifest_path,
        tokenizer_name=tokenizer_name,
        tokenizer_spec=tokenizer_spec,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    config_path, defaults = load_config_defaults(argv, section="preprocess")
    parser = argparse.ArgumentParser(
        description="Prepare the local DBLP + arXiv research-paper corpus."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Optional JSON preset / recipe file.",
    )
    parser.add_argument(
        "--dblp",
        type=Path,
        default=None,
        help="Path to dblp-v10.csv (auto-detected from the repo root when omitted).",
    )
    parser.add_argument(
        "--arxiv",
        type=Path,
        default=None,
        help=(
            "Path to arxiv-metadata-oai-snapshot.json. The file is treated as "
            "newline-delimited JSON records."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for token shards, manifests, and stats.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer manifest, tokenizer.json, or directory. "
            "When omitted, the recommended research tokenizer is used if present, "
            "otherwise legacy GPT-2 is used."
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=0.90, help="Training split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Test split ratio.")
    parser.add_argument("--min_title_chars", type=int, default=12, help="Minimum cleaned title length.")
    parser.add_argument(
        "--min_abstract_chars",
        type=int,
        default=80,
        help="Minimum cleaned abstract length.",
    )
    parser.add_argument(
        "--max_abstract_chars",
        type=int,
        default=12_000,
        help="Maximum cleaned abstract length before dropping the record.",
    )
    parser.add_argument(
        "--min_alpha_ratio",
        type=float,
        default=0.60,
        help="Minimum alphabetic-character ratio for abstracts.",
    )
    parser.add_argument(
        "--max_junk_ratio",
        type=float,
        default=0.12,
        help="Maximum non-text junk ratio for abstracts.",
    )
    parser.add_argument(
        "--max_repeated_char_ratio",
        type=float,
        default=0.15,
        help="Maximum longest-run character ratio before dropping a record.",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Optional comma-separated arXiv category prefixes to keep, for example "
            "'cs,stat.ML,math,physics'."
        ),
    )
    parser.add_argument(
        "--include_authors",
        action="store_true",
        help="Include cleaned author strings in the training text template.",
    )
    parser.add_argument(
        "--no_deduplicate",
        action="store_true",
        help="Disable document-level deduplication.",
    )
    parser.add_argument(
        "--max_records_per_source",
        type=int,
        default=None,
        help="Limit the number of records read from each source.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Shortcut for --max_records_per_source N.",
    )
    parser.add_argument(
        "--source_weights",
        default=None,
        help="Relative keep weights such as 'dblp=1.0,arxiv=0.75'.",
    )
    parser.add_argument(
        "--domain_weights",
        default=None,
        help="Relative keep weights by domain prefix, for example 'cs=1.0,physics=0.6'.",
    )
    parser.add_argument(
        "--category_weights",
        default=None,
        help="Relative keep weights by exact arXiv category, for example 'cs.LG=1.0,stat.ML=0.9'.",
    )
    parser.add_argument(
        "--shuffle_buckets",
        type=int,
        default=256,
        help="Power-of-two number of external shuffle buckets.",
    )
    parser.add_argument(
        "--keep_temp_shuffle_files",
        action="store_true",
        help="Keep the intermediate external-shuffle files instead of deleting them.",
    )
    parser.add_argument(
        "--shard_size_tokens",
        type=int,
        default=50_000_000,
        help="Target number of tokens per shard.",
    )
    parser.add_argument(
        "--buffer_tokens",
        type=int,
        default=1_000_000,
        help="In-memory document-token buffer size before flushing to disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic split assignment and shuffling.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging.",
    )
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def resolve_default_inputs(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path]]:
    dblp = args.dblp
    arxiv = args.arxiv
    if dblp is None and DEFAULT_DBLP_PATH.exists():
        dblp = DEFAULT_DBLP_PATH
    if arxiv is None and DEFAULT_ARXIV_PATH.exists():
        arxiv = DEFAULT_ARXIV_PATH
    return dblp, arxiv


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    dblp_path, arxiv_path = resolve_default_inputs(args)

    max_records_per_source = (
        args.sample if args.sample is not None else args.max_records_per_source
    )
    allowed_prefixes = ()
    if args.categories:
        allowed_prefixes = tuple(
            part.strip() for part in args.categories.split(",") if part.strip()
        )

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
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
    mixing_config = MixingConfig(
        shuffle_buckets=args.shuffle_buckets,
        source_weights=parse_weight_mapping(args.source_weights),
        domain_weights=parse_weight_mapping(args.domain_weights),
        category_weights=parse_weight_mapping(args.category_weights),
        cleanup_temp_files=not args.keep_temp_shuffle_files,
    )

    try:
        process_corpus(
            dblp_path=dblp_path,
            arxiv_path=arxiv_path,
            output_dir=args.output_dir,
            split_config=split_config,
            quality_config=quality_config,
            formatting_config=formatting_config,
            mixing_config=mixing_config,
            deduplicate=not args.no_deduplicate,
            max_records_per_source=max_records_per_source,
            shard_size_tokens=args.shard_size_tokens,
            buffer_tokens=args.buffer_tokens,
            seed=args.seed,
            verbose=not args.quiet,
            tokenizer=args.tokenizer,
        )
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"[prepare_papers] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
