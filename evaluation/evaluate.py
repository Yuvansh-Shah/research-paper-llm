#!/usr/bin/env python3
"""Evaluation module for research-paper language models."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config_defaults
from evaluation.prompt_suites import PROMPT_SUITES, get_prompts
from inference.generate import generate, load_model_bundle
from model.config import GPTConfig
from training.dataset import TokenStore
from tokenization import (
    load_runtime_tokenizer,
    load_tokenizer_spec,
    load_tokenizer_spec_from_checkpoint,
)


@dataclass
class PerplexityResult:
    split: str
    loss: float
    perplexity: float
    num_tokens: int
    num_batches: int
    time_seconds: float


@dataclass
class GenerationResult:
    prompt: str
    completion: str
    tokens_generated: int
    time_seconds: float


@dataclass
class EvaluationReport:
    checkpoint_path: str
    model_size: str
    num_parameters: int
    context_length: int
    tokenizer_name: str
    timestamp: str
    prompt_suite: str
    perplexity_results: list[dict]
    skipped_splits: dict[str, str]
    source_breakdowns: dict[str, dict]
    domain_breakdowns: dict[str, dict]
    category_breakdowns: dict[str, dict]
    generation_results: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate_perplexity(
    model,
    data_path: str,
    context_length: int,
    batch_size: int = 8,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
    split_name: str = "test",
) -> PerplexityResult:
    """Token-weighted perplexity over non-overlapping ``context_length`` windows.

    The token stream is walked in steps of ``context_length`` so every target
    token is scored exactly once (the standard evaluation protocol). The final
    partial window, if any, is scored on its own.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    store = TokenStore(data_path)

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    t0 = time.perf_counter()

    def _score_batch(windows: list[torch.Tensor]) -> None:
        nonlocal total_loss, total_tokens, num_batches
        batch = torch.stack(windows).to(device)
        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()
        _, loss = model(x, y)
        batch_tokens = y.numel()
        total_loss += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens
        num_batches += 1

    try:
        pending: list[torch.Tensor] = []
        for cursor in range(0, len(store) - 1, context_length):
            if max_batches and num_batches >= max_batches:
                pending = []
                break
            window = store.read_range(cursor, min(cursor + context_length + 1, len(store)))
            window_tensor = torch.from_numpy(window.astype("int64", copy=False))
            if window_tensor.numel() == context_length + 1:
                pending.append(window_tensor)
                if len(pending) == batch_size:
                    _score_batch(pending)
                    pending = []
            elif window_tensor.numel() >= 2:
                _score_batch([window_tensor])
        if pending and not (max_batches and num_batches >= max_batches):
            _score_batch(pending)
    finally:
        store.close()

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(avg_loss, 20))
    return PerplexityResult(
        split=split_name,
        loss=avg_loss,
        perplexity=perplexity,
        num_tokens=total_tokens,
        num_batches=num_batches,
        time_seconds=elapsed,
    )


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer_spec,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
) -> list[GenerationResult]:
    runtime_tokenizer = load_runtime_tokenizer(tokenizer_spec)
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []
    for prompt in prompts:
        t0 = time.perf_counter()
        prompt_ids = runtime_tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        elapsed = time.perf_counter() - t0
        generated_ids = output_ids[0].tolist()
        completion_ids = generated_ids[len(prompt_ids):]
        completion = runtime_tokenizer.decode(completion_ids, skip_special_tokens=False)
        results.append(
            GenerationResult(
                prompt=prompt,
                completion=completion,
                tokens_generated=len(completion_ids),
                time_seconds=elapsed,
            )
        )
    return results


def _score_document_tokens(model, token_tensor: torch.Tensor, context_length: int, device: torch.device) -> tuple[float, int]:
    if token_tensor.numel() < 2:
        return 0.0, 0
    total_loss = 0.0
    total_tokens = 0
    for cursor in range(0, token_tensor.numel() - 1, context_length):
        segment = token_tensor[cursor : cursor + context_length + 1]
        if segment.numel() < 2:
            continue
        x = segment[:-1].unsqueeze(0).to(device)
        y = segment[1:].unsqueeze(0).to(device)
        _, loss = model(x, y)
        tokens = int(y.numel())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    return total_loss, total_tokens


def _load_document_index(path: Path) -> list[dict]:
    documents = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


@torch.no_grad()
def evaluate_breakdowns(
    model,
    data_path: str,
    context_length: int,
    *,
    device: torch.device,
    category_limit: int = 12,
    max_documents: Optional[int] = None,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    store = TokenStore(data_path)
    if store.document_index_path is None or not store.document_index_path.exists():
        store.close()
        return {}, {}, {}

    documents = _load_document_index(store.document_index_path)
    if max_documents is not None:
        documents = documents[:max_documents]

    source_totals: dict[str, dict[str, float]] = {}
    domain_totals: dict[str, dict[str, float]] = {}
    category_totals: dict[str, dict[str, float]] = {}

    def _note(container: dict[str, dict[str, float]], key: str, loss_sum: float, token_count: int) -> None:
        bucket = container.setdefault(key, {"loss_sum": 0.0, "token_count": 0, "document_count": 0})
        bucket["loss_sum"] += loss_sum
        bucket["token_count"] += token_count
        bucket["document_count"] += 1

    try:
        for document in documents:
            start = int(document["global_token_start"])
            end = int(document["global_token_end"])
            token_array = store.read_range(start, end)
            loss_sum, token_count = _score_document_tokens(
                model,
                torch.from_numpy(token_array.astype("int64", copy=False)),
                context_length,
                device,
            )
            if token_count == 0:
                continue
            source = str(document.get("source") or "unknown")
            _note(source_totals, source, loss_sum, token_count)

            primary_domain = str(document.get("primary_domain") or "").strip()
            if primary_domain:
                _note(domain_totals, primary_domain, loss_sum, token_count)

            for category in document.get("categories") or []:
                category_name = str(category).strip()
                if category_name:
                    _note(category_totals, category_name, loss_sum, token_count)
    finally:
        store.close()

    def _finalize(container: dict[str, dict[str, float]]) -> dict[str, dict]:
        items = []
        for key, bucket in container.items():
            avg_loss = bucket["loss_sum"] / max(1, bucket["token_count"])
            items.append(
                (
                    key,
                    {
                        "loss": avg_loss,
                        "perplexity": math.exp(min(avg_loss, 20)),
                        "token_count": int(bucket["token_count"]),
                        "document_count": int(bucket["document_count"]),
                    },
                )
            )
        items.sort(key=lambda item: item[1]["token_count"], reverse=True)
        return dict(items)

    finalized_categories = _finalize(category_totals)
    if category_limit > 0:
        finalized_categories = dict(list(finalized_categories.items())[:category_limit])
    return _finalize(source_totals), _finalize(domain_totals), finalized_categories


def _infer_model_size(config) -> str:
    for size_name in ["debug", "small", "base", "large", "xl"]:
        try:
            size_config = GPTConfig.from_name(size_name)
        except Exception:
            continue
        if (
            size_config.n_embd == config.n_embd
            and size_config.n_layers == config.n_layers
            and size_config.n_heads == config.n_heads
        ):
            return size_name
    return "custom"


def run_evaluation(
    checkpoint_path: str,
    test_bin: Optional[str] = None,
    val_bin: Optional[str] = None,
    prompts_file: Optional[str] = None,
    prompt_suite: str = "research_capability",
    output_dir: Optional[str] = None,
    batch_size: int = 8,
    max_batches: Optional[int] = None,
    max_breakdown_documents: Optional[int] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: Optional[str] = None,
    run_generation: bool = True,
    tokenizer: Optional[str] = None,
    category_breakdown_limit: int = 12,
) -> EvaluationReport:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    print(f"Evaluating on device: {device_obj}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model, config, payload = load_model_bundle(checkpoint_path, device=str(device_obj))
    model = model.to(device_obj)
    model.eval()

    if tokenizer:
        tokenizer_spec = load_tokenizer_spec(tokenizer)
    else:
        tokenizer_spec = load_tokenizer_spec_from_checkpoint(checkpoint_path, payload=payload)

    num_params = model.num_parameters()
    print(
        f"Model loaded: {num_params / 1e6:.1f}M parameters, "
        f"context={config.context_length}, tokenizer={tokenizer_spec.name}"
    )

    perplexity_results: list[dict] = []
    skipped_splits: dict[str, str] = {}
    source_breakdowns: dict[str, dict] = {}
    domain_breakdowns: dict[str, dict] = {}
    category_breakdowns: dict[str, dict] = {}

    for split_name, split_path in (("val", val_bin), ("test", test_bin)):
        if not split_path:
            continue
        if not Path(split_path).exists():
            print(
                f"WARNING: {split_name} split file not found, skipping: {split_path}",
                file=sys.stderr,
            )
            skipped_splits[split_name] = str(split_path)
            continue
        print(f"\nEvaluating perplexity on {split_name}: {split_path}")
        result = evaluate_perplexity(
            model,
            split_path,
            config.context_length,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device_obj,
            split_name=split_name,
        )
        perplexity_results.append(asdict(result))
        print(f"  {split_name} loss={result.loss:.4f} | ppl={result.perplexity:.2f}")

        sources, domains, categories = evaluate_breakdowns(
            model,
            split_path,
            config.context_length,
            device=device_obj,
            category_limit=category_breakdown_limit,
            max_documents=max_breakdown_documents,
        )
        if sources:
            source_breakdowns[split_name] = sources
        if domains:
            domain_breakdowns[split_name] = domains
        if categories:
            category_breakdowns[split_name] = categories

    prompts: list[str] = []
    resolved_prompt_suite = prompt_suite
    generation_results: list[dict] = []
    if run_generation:
        prompts, resolved_prompt_suite = get_prompts(prompts_file, prompt_suite)
        print(f"\nGenerating completions for {len(prompts)} prompts...")
        generation_results = [
            asdict(result)
            for result in evaluate_generation(
                model,
                tokenizer_spec,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device_obj,
            )
        ]

    report = EvaluationReport(
        checkpoint_path=checkpoint_path,
        model_size=_infer_model_size(config),
        num_parameters=num_params,
        context_length=config.context_length,
        tokenizer_name=tokenizer_spec.name,
        timestamp=datetime.now().isoformat(),
        prompt_suite=resolved_prompt_suite if run_generation else "disabled",
        perplexity_results=perplexity_results,
        skipped_splits=skipped_splits,
        source_breakdowns=source_breakdowns,
        domain_breakdowns=domain_breakdowns,
        category_breakdowns=category_breakdowns,
        generation_results=generation_results,
    )

    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        report.save(output_dir_path / "evaluation_report.json")
        if run_generation:
            (output_dir_path / "prompts_used.txt").write_text(
                "\n".join(prompts) + ("\n" if prompts else ""),
                encoding="utf-8",
            )
            (output_dir_path / "generations.txt").write_text(
                "\n\n".join(
                    [
                        f"=== Prompt {index + 1} ===\nPROMPT: {item['prompt']}\nOUTPUT: {item['completion']}"
                        for index, item in enumerate(generation_results)
                    ]
                ),
                encoding="utf-8",
            )
            (output_dir_path / "generations.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in generation_results) + ("\n" if generation_results else ""),
                encoding="utf-8",
            )

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {report.model_size} ({report.num_parameters / 1e6:.1f}M params)")
    print(f"Context: {report.context_length}")
    print(f"Tokenizer: {report.tokenizer_name}")
    for item in report.perplexity_results:
        print(
            f"{item['split']:>5s} | loss {item['loss']:.4f} | ppl {item['perplexity']:.2f} "
            f"| tokens {item['num_tokens']:,}"
        )
    for split_name, split_path in report.skipped_splits.items():
        print(f"{split_name:>5s} | SKIPPED (file not found: {split_path})")
    for split_name, breakdown in report.source_breakdowns.items():
        print(f"\n{split_name} source breakdown:")
        for source, metrics in breakdown.items():
            print(
                f"  {source:>8s} | loss {metrics['loss']:.4f} | ppl {metrics['perplexity']:.2f} "
                f"| docs {metrics['document_count']:,}"
            )
    return report


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    config_path, defaults = load_config_defaults(argv, section="evaluation")
    parser = argparse.ArgumentParser(description="Evaluate a trained CustomLM checkpoint")
    parser.add_argument("--config", type=Path, default=config_path, help="Optional JSON preset / recipe file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer override.")
    parser.add_argument("--test_bin", default=None)
    parser.add_argument("--val_bin", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_breakdown_documents", type=int, default=None)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--prompt_suite", default="research_capability", choices=sorted(PROMPT_SUITES))
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--category_breakdown_limit", type=int, default=12)
    parser.add_argument("--no_generation", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--device", default=None)
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    if args.save_outputs and not output_dir:
        output_dir = str(PROJECT_ROOT / "evaluation" / "results")

    run_evaluation(
        checkpoint_path=args.checkpoint,
        test_bin=args.test_bin,
        val_bin=args.val_bin,
        prompts_file=args.prompts,
        prompt_suite=args.prompt_suite,
        output_dir=output_dir,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_breakdown_documents=args.max_breakdown_documents,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        run_generation=not args.no_generation,
        tokenizer=args.tokenizer,
        category_breakdown_limit=args.category_breakdown_limit,
    )


if __name__ == "__main__":
    main()
