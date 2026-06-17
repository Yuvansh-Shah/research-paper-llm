"""Text generation CLI for CustomLM checkpoints."""

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
    default_gpt2_tokenizer_spec,
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
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_data = payload["config"]
    if isinstance(config_data, GPTConfig):
        config = config_data
    elif isinstance(config_data, dict):
        config = GPTConfig(**config_data)
    else:
        raise ValueError(f"Unknown config type in checkpoint: {type(config_data)}")

    model = GPT(config)
    model.load_state_dict(payload["model_state_dict"])
    return model, config, payload


@torch.no_grad()
def generate(
    model,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
) -> Tensor:
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if greedy:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            logits = _top_k_filter(logits, top_k)
            logits = _top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    config_path, defaults = load_config_defaults(argv, section="inference")
    parser = argparse.ArgumentParser(description="Generate text with a CustomLM checkpoint")
    parser.add_argument("--config", type=Path, default=config_path, help="Optional JSON preset / recipe file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer override.")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_special_tokens", action="store_true")
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
    tokenizer_spec = tokenizer_spec or default_gpt2_tokenizer_spec()
    runtime_tokenizer = load_runtime_tokenizer(tokenizer_spec)

    print(
        f"Model loaded - {model.num_parameters() / 1e6:.1f} M params | "
        f"context {model.config.context_length} | tokenizer {tokenizer_spec.name}"
    )

    token_ids = runtime_tokenizer.encode(args.prompt, add_eos=False)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    print(f"\nPrompt: {args.prompt!r}\n{'-' * 60}")
    t0 = time.perf_counter()
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )
    elapsed = time.perf_counter() - t0
    new_tokens = out.shape[1] - idx.shape[1]

    generated_ids = out[0].tolist()
    text = runtime_tokenizer.decode(generated_ids, skip_special_tokens=args.skip_special_tokens)
    print(text)
    print(
        f"\n{'-' * 60}\n"
        f"Generated {new_tokens} tokens in {elapsed:.2f}s "
        f"({new_tokens / max(elapsed, 1e-6):.1f} tok/s)"
    )


if __name__ == "__main__":
    main()
