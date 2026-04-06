from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import tiktoken

try:
    import wikipediaapi
except ImportError as exc:  # pragma: no cover - import error path
    raise RuntimeError(
        "wikipedia-api is required. Install dependencies from requirements.txt"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RAW = RAW_DIR / "geography_wikipedia.txt"
TRAIN_BIN = PROCESSED_DIR / "train.bin"
VAL_BIN = PROCESSED_DIR / "val.bin"

CITATION_RE = re.compile(r"\[\d+\]")
TEMPLATE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

TOPIC_SEED_PAGES = [
    "Geography",
    "Physical geography",
    "Human geography",
    "Geology",
    "Climatology",
    "Hydrology",
    "Geomorphology",
    "Biogeography",
    "Cartography",
    "Earth",
    "Plate tectonics",
    "Atmosphere of Earth",
    "Oceanography",
    "Mountain",
    "River",
    "Desert",
    "Rainforest",
    "Climate change",
    "Coast",
    "India",
    "Amazon rainforest",
]


def clean_wikipedia_text(text: str) -> str:
    text = TEMPLATE_RE.sub(" ", text)
    text = CITATION_RE.sub("", text)
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"^\s*==+\s*(References|External links|See also|Notes)\s*==+\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def collect_geography_corpus(target_mb: int, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_chars = target_mb * 1024 * 1024

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="CustomLM-dataset-prep/1.0 (https://github.com/yuvansh12/Research-paper-maker-)",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    seen_titles: set[str] = set()
    chunks: list[str] = []
    total_chars = 0

    def add_page(title: str) -> None:
        nonlocal total_chars
        if title in seen_titles or total_chars >= target_chars:
            return
        page = wiki.page(title)
        if not page.exists():
            return
        seen_titles.add(title)
        cleaned = clean_wikipedia_text(page.text)
        if len(cleaned) < 200:
            return
        chunks.append(cleaned)
        total_chars += len(cleaned)
        print(f"[prepare] collected '{title}' ({len(cleaned)} chars), total={total_chars}")

    for seed in TOPIC_SEED_PAGES:
        add_page(seed)
        if total_chars >= target_chars:
            break
        page = wiki.page(seed)
        if page.exists():
            for link_title in page.links.keys():
                if total_chars >= target_chars:
                    break
                if "geograph" in link_title.lower() or "climate" in link_title.lower() or "earth" in link_title.lower() or "ocean" in link_title.lower() or "river" in link_title.lower() or "mountain" in link_title.lower() or "desert" in link_title.lower() or "forest" in link_title.lower():
                    add_page(link_title)

    combined = "\n\n".join(chunks)
    output_path.write_text(combined, encoding="utf-8")
    print(f"[prepare] raw corpus saved to {output_path} ({len(combined)} chars)")
    return combined


def _encode_with_gpt2_tiktoken(text: str) -> list[int]:
    try:
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode_ordinary(text)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tiktoken GPT-2 BPE encoder. "
            "Likely cause: first-run tokenizer asset download is blocked/offline. "
            "Fix: run once with internet access to cache tokenizer files, then rerun "
            "`python data/prepare.py`."
        ) from exc


def tokenize_and_split(
    text: str,
    train_path: Path,
    val_path: Path,
    encode_fn: Callable[[str], list[int]] | None = None,
) -> tuple[int, int]:
    token_ids = (encode_fn or _encode_with_gpt2_tiktoken)(text)
    tokens = np.array(token_ids, dtype=np.uint16)
    if tokens.size == 0:
        raise ValueError("No tokens produced from dataset text.")
    split_idx = int(tokens.size * 0.95)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    return int(train_tokens.size), int(val_tokens.size)


def verify_dataset_readable(train_path: Path, val_path: Path) -> None:
    from training.dataset import MemmapDataset

    train_ds = MemmapDataset(train_path)
    val_ds = MemmapDataset(val_path)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset verification failed: one split is empty.")
    _ = train_ds[0]
    _ = val_ds[0]
    print(f"[prepare] verification OK: train={len(train_ds)} val={len(val_ds)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Wikipedia geography dataset.")
    parser.add_argument("--target-mb", type=int, default=200, help="Approximate raw text size target in MB.")
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW, help="Path for cleaned raw text output.")
    parser.add_argument("--input-text-file", type=Path, default=None, help="Optional existing text file; skips download if provided.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[prepare] starting geography corpus preparation")
    if args.input_text_file is not None:
        text = args.input_text_file.read_text(encoding="utf-8")
        args.raw_output.write_text(clean_wikipedia_text(text), encoding="utf-8")
        text = args.raw_output.read_text(encoding="utf-8")
        print(f"[prepare] loaded existing text from {args.input_text_file}")
    else:
        try:
            text = collect_geography_corpus(target_mb=args.target_mb, output_path=args.raw_output)
        except Exception as exc:
            print(
                "[prepare] error: failed to download Wikipedia geography corpus. "
                "Likely cause: no internet access or Wikipedia API block. "
                "Fix: check connectivity, then rerun `python data/prepare.py --target-mb 200`.",
                file=sys.stderr,
            )
            raise
    try:
        train_count, val_count = tokenize_and_split(text, TRAIN_BIN, VAL_BIN)
    except RuntimeError as exc:
        print(f"[prepare] error: {exc}", file=sys.stderr)
        return 1
    print(f"[prepare] token counts: train={train_count} val={val_count}")
    verify_dataset_readable(TRAIN_BIN, VAL_BIN)
    print("[prepare] completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
