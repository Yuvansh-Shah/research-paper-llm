"""Text generation CLI for CustomLM.

Usage::

    python inference/generate.py \\
        --checkpoint checkpoints/final.pt \\
        --prompt "The history of artificial intelligence" \\
        --max_tokens 200 \\
        --temperature 0.8 \\
        --top_p 0.9

Sampling strategies supported
------------------------------
* **greedy** — always picks the highest-probability token
* **temperature** — scales logits before sampling
* **top-k** — keeps only the top-k tokens before sampling
* **top-p (nucleus)** — keeps the smallest set of tokens whose cumulative
  probability ≥ p, then samples from that set
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F
from torch import Tensor


def _load_tokeniser():
    try:
        import tiktoken  # noqa: PLC0415
    except ImportError:
        sys.exit("tiktoken is required: pip install tiktoken")
    return tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Zero out all logits except the top-k."""
    if k <= 0:
        return logits
    top_k_vals, _ = torch.topk(logits, k)
    threshold = top_k_vals[:, -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Zero out all logits outside the nucleus (top-p set)."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative prob above threshold *after* including current.
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
    # Scatter back to original indexing.
    return logits.scatter(1, sorted_indices, sorted_logits)


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

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
    """Autoregressively generate tokens from *model*.

    Args:
        model: A :class:`model.gpt.GPT` instance in eval mode.
        idx: Seed token IDs of shape ``(1, T)``.
        max_new_tokens: Number of new tokens to generate.
        temperature: Logit scaling factor.  Values < 1 make the
            distribution sharper; values > 1 make it flatter.
        top_k: If > 0, restrict sampling to the top-k tokens.
        top_p: Nucleus sampling probability threshold.
        greedy: If ``True``, always pick the argmax token (ignores
            temperature / top-k / top-p).

    Returns:
        Tensor of shape ``(1, T + max_new_tokens)`` containing the full
        sequence including the seed.
    """
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        # Crop to the model's context window.
        idx_cond = idx[:, -context_length:]

        # Forward pass — returns logits for the last position only.
        logits, _ = model(idx_cond)          # (1, 1, vocab_size)
        logits = logits[:, -1, :]            # (1, vocab_size)

        if greedy:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            # Apply temperature scaling.
            logits = logits / max(temperature, 1e-8)
            # Apply top-k and top-p filtering.
            logits = _top_k_filter(logits, top_k)
            logits = _top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a CustomLM checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--prompt", default="Once upon a time",
                   help="Seed text for generation")
    p.add_argument("--max_tokens", type=int, default=200,
                   help="Maximum number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (default 0.8; lower → more focused)")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-k filtering (0 = disabled)")
    p.add_argument("--top_p", type=float, default=0.9,
                   help="Nucleus sampling probability (1.0 = disabled)")
    p.add_argument("--greedy", action="store_true",
                   help="Use greedy decoding instead of sampling")
    p.add_argument("--device", default=None,
                   help="Device to run on (default: cuda if available, else cpu)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load model.
    from model.gpt import GPT  # noqa: PLC0415 — deferred to keep import fast
    print(f"Loading checkpoint: {args.checkpoint}")
    model = GPT.from_checkpoint(args.checkpoint, device=str(device))
    model.eval()
    model.to(device)
    print(
        f"Model loaded — {model.num_parameters() / 1e6:.1f} M params | "
        f"context {model.config.context_length} tokens"
    )

    # Tokenise prompt.
    enc = _load_tokeniser()
    token_ids = enc.encode(args.prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Generate.
    print(f"\nPrompt: {args.prompt!r}\n{'─' * 60}")
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
    t1 = time.perf_counter()
    elapsed = t1 - t0
    new_tokens = out.shape[1] - idx.shape[1]

    # Decode and print.
    generated_ids = out[0].tolist()
    text = enc.decode(generated_ids)
    print(text)
    print(
        f"\n{'─' * 60}\n"
        f"Generated {new_tokens} tokens in {elapsed:.2f}s "
        f"({new_tokens / elapsed:.1f} tok/s)"
    )


if __name__ == "__main__":
    main()
