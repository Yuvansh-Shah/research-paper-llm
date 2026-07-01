"""Shared qualitative prompt suites for training snapshots and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


PROMPT_SUITES = {
    "generic": [
        "The capital of France is",
        "In 1969, humans first",
        "The theory of relativity states that",
        "To make a good cup of coffee,",
        "The most important thing in life is",
    ],
    "research": [
        "Title: Efficient Transformer Pretraining for Scientific Corpora\nAbstract:",
        "Title: A Study of Self-Supervised Learning for Research Documents\nAbstract:",
        "Title: Benchmarking Retrieval-Augmented Language Models on arXiv Abstracts\nAbstract:",
        "We propose a method for improving document-level language modeling on scientific abstracts by",
        "The main contribution of this paper is",
        "Prior work on large-scale pretraining for research literature has focused on",
        "Our experimental setup uses a held-out validation split to measure",
        "A limitation of the current approach is",
        "In this paper, we analyze the effect of dataset deduplication on",
        "Future work should investigate",
    ],
    "research_capability": [
        "Title: Scaling Tokenization Strategies for Scientific Abstract Modeling\nAbstract:",
        "Title: Domain-Adaptive Pretraining for Mixed DBLP and arXiv Corpora\nAbstract:",
        "Title: Improving Validation Protocols for Scientific Language Models\nAbstract:",
        "Title: A Controlled Study of Corpus Shuffling in Research-Paper Pretraining\nAbstract:",
        "Introduction:\nScientific language models trained on research corpora benefit from",
        "Related Work:\nRecent work on document-level pretraining for scientific text has",
        "Method:\nWe train a decoder-only transformer on a mixed DBLP and arXiv corpus using",
        "Results:\nHeld-out perplexity improves when the tokenizer is trained directly on",
        "Ablation Study:\nRemoving deterministic document shuffling causes",
        "Discussion:\nThe difference between DBLP and arXiv validation loss suggests",
        "Conclusion:\nThis work shows that capability-first scientific language modeling requires",
        "Future Work:\nThe most promising next experiment for this recipe is",
    ],
}


def load_prompts(path: str | Path) -> list[str]:
    prompts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def get_prompts(prompts_file: Optional[str], prompt_suite: str) -> tuple[list[str], str]:
    if prompts_file:
        if Path(prompts_file).exists():
            return load_prompts(prompts_file), f"file:{prompts_file}"
        print(
            f"WARNING: prompts file not found: {prompts_file}; "
            f"falling back to built-in suite '{prompt_suite}'",
            file=sys.stderr,
        )
    return PROMPT_SUITES[prompt_suite], prompt_suite
