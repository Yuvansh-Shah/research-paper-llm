"""Text generation CLI for research-paper-llm checkpoints."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config_defaults
from model.config import GPTConfig
from model.gpt import GPT
from tokenization import (
    load_runtime_tokenizer,
    load_tokenizer_spec,
    load_tokenizer_spec_from_checkpoint,
)


def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    if k <= 0:
        return logits
    top_k_vals, _ = torch.topk(logits, k)
    threshold = top_k_vals[:, -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs - sorted_probs > p
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
    return logits.scatter(1, sorted_indices, sorted_logits)


def load_model_bundle(checkpoint_path: str | Path, device: str = "cpu") -> tuple[GPT, GPTConfig, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config_data = payload["config"]
    if not isinstance(config_data, dict):
        raise ValueError(f"Unknown config type in checkpoint: {type(config_data)}")
    config = GPTConfig(**config_data)

    model = GPT(config)
    model.load_state_dict(payload["model_state_dict"])
    return model, config, payload


def _sample_next_token(
    model,
    idx: Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
) -> Tensor:
    idx_cond = idx[:, -model.config.context_length :]
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :]

    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    eos_token_id: Optional[int] = None,
) -> Tensor:
    for _ in range(max_new_tokens):
        next_token = _sample_next_token(
            model,
            idx,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break
        idx = torch.cat([idx, next_token], dim=1)

    return idx


@torch.no_grad()
def stream_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    device: Optional[torch.device] = None,
    stop_at_eos: bool = True,
):
    """Stream a completion for ``prompt`` as incremental decoded text pieces.

    One token is sampled per step; after each step the full completion so far
    is decoded and only the suffix relative to the previously yielded text is
    emitted. Decoding the whole completion each step (rather than each token
    in isolation) keeps multi-token unicode sequences intact; pieces whose
    decode still ends in an incomplete byte sequence (U+FFFD) are held back
    until a later token completes them. Concatenating all yielded pieces
    reproduces the full decoded completion.

    Args:
        model: A ``GPT`` model in eval mode.
        tokenizer: A ``RuntimeTokenizer`` (``encode``/``decode`` plus an
            ``eos_token_id`` attribute).
        prompt: Text used to seed generation; not included in the output.
        max_new_tokens: Maximum number of tokens to sample.
        temperature: Softmax temperature (ignored when ``greedy``).
        top_k: Keep only the top-k logits per step (0 disables).
        top_p: Nucleus sampling threshold (1.0 disables).
        greedy: Use argmax decoding instead of sampling.
        device: Device for the prompt tensor; defaults to the model's device.
        stop_at_eos: Stop once the tokenizer's EOS token is produced. The EOS
            token itself is never emitted.

    Yields:
        Non-empty text pieces that concatenate to the decoded completion.
    """
    if device is None:
        device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, add_eos=False)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    completion_ids: list[int] = []
    emitted = ""

    for _ in range(max_new_tokens):
        next_token = _sample_next_token(
            model,
            idx,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        if stop_at_eos and int(next_token.item()) == tokenizer.eos_token_id:
            break
        idx = torch.cat([idx, next_token], dim=1)
        completion_ids.append(int(next_token.item()))

        decoded = tokenizer.decode(completion_ids, skip_special_tokens=False)
        if decoded.endswith("\ufffd"):
            continue
        piece = decoded[len(emitted):]
        if piece:
            emitted = decoded
            yield piece

    decoded = tokenizer.decode(completion_ids, skip_special_tokens=False)
    piece = decoded[len(emitted):]
    if piece:
        yield piece


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    config_path, defaults = load_config_defaults(argv, section="inference")
    parser = argparse.ArgumentParser(description="Generate text with a research-paper-llm checkpoint")
    parser.add_argument("--config", type=Path, default=config_path, help="Optional JSON preset / recipe file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer override.")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--ignore-eos", action="store_true", help="Keep sampling past the EOS token.")
    parser.add_argument("--device", default=None)
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading checkpoint: {args.checkpoint}")
    model, _, payload = load_model_bundle(args.checkpoint, device=str(device))
    model.eval()
    model.to(device)

    if args.tokenizer:
        tokenizer_spec = load_tokenizer_spec(args.tokenizer)
    else:
        tokenizer_spec = load_tokenizer_spec_from_checkpoint(args.checkpoint, payload=payload)
    runtime_tokenizer = load_runtime_tokenizer(tokenizer_spec)

    print(
        f"Model loaded - {model.num_parameters() / 1e6:.1f} M params | "
        f"context {model.config.context_length} | tokenizer {tokenizer_spec.name}"
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"\nPrompt: {args.prompt!r}\n{'-' * 60}")
    print(args.prompt, end="", flush=True)
    t0 = time.perf_counter()
    completion_parts: list[str] = []
    for piece in stream_generate(
        model,
        runtime_tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        device=device,
        stop_at_eos=not args.ignore_eos,
    ):
        completion_parts.append(piece)
        print(piece, end="", flush=True)
    elapsed = time.perf_counter() - t0

    completion = "".join(completion_parts)
    new_tokens = len(runtime_tokenizer.encode(completion, add_eos=False)) if completion else 0
    print(
        f"\n{'-' * 60}\n"
        f"Generated ~{new_tokens} tokens in {elapsed:.2f}s "
        f"({new_tokens / max(elapsed, 1e-6):.1f} tok/s)"
    )


if __name__ == "__main__":
    main()
