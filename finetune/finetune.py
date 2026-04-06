"""Fine-tuning script for CustomLM.

Supports two modes:
* **Domain fine-tuning** — same next-token prediction objective as
  pre-training but on a focused corpus and with a lower learning rate.
* **Instruction fine-tuning** — trains on instruction-response pairs
  formatted as ``### Instruction:\\n{prompt}\\n\\n### Response:\\n{response}``.

LoRA is available for both modes via ``--use_lora`` to keep VRAM usage
minimal on an RTX 3050.

Usage — domain fine-tuning::

    python finetune/finetune.py \\
        --base_checkpoint checkpoints/final.pt \\
        --train_bin data/processed/domain_train.bin \\
        --val_bin   data/processed/domain_val.bin \\
        --out_dir   checkpoints/finetune \\
        --mode      domain \\
        --use_lora

Usage — instruction fine-tuning from a JSONL file::

    python finetune/finetune.py \\
        --base_checkpoint checkpoints/final.pt \\
        --instruction_data data/raw/instructions.jsonl \\
        --out_dir checkpoints/finetune_instruct \\
        --mode instruction \\
        --use_lora
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.gpt import GPT
from training.dataset import TokenDataset
from training.lr_schedule import CosineScheduler


# ---------------------------------------------------------------------------
# Instruction dataset
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning from a JSONL file.

    Each line of the file must be a JSON object with at minimum an
    ``instruction`` and ``output`` (or ``response``) field, following the
    Alpaca format::

        {"instruction": "...", "input": "...", "output": "..."}

    The ``input`` field is optional context appended to the instruction.

    Args:
        jsonl_path: Path to the JSONL instruction file.
        enc: A ``tiktoken`` encoder.
        context_length: Maximum token length per example.
    """

    def __init__(self, jsonl_path: str, enc, context_length: int) -> None:
        self.enc = enc
        self.context_length = context_length
        self.examples: list[list[int]] = []

        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                instruction = obj.get("instruction", "")
                inp = obj.get("input", "")
                response = obj.get("output") or obj.get("response", "")

                if inp:
                    instruction = f"{instruction}\n\n{inp}"

                text = INSTRUCTION_TEMPLATE.format(
                    instruction=instruction, response=response
                )
                tokens = enc.encode_ordinary(text)
                # Truncate to context_length + 1 (need one extra for target).
                tokens = tokens[: context_length + 1]
                if len(tokens) >= 2:
                    self.examples.append(tokens)

        print(f"  Loaded {len(self.examples)} instruction examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        tokens = self.examples[idx]
        # Pad to context_length + 1.
        pad_len = (self.context_length + 1) - len(tokens)
        tokens = tokens + [-1] * pad_len
        chunk = torch.tensor(tokens, dtype=torch.long)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a CustomLM checkpoint")
    p.add_argument("--base_checkpoint", required=True,
                   help="Path to pre-trained base checkpoint (.pt)")
    p.add_argument("--out_dir", default="checkpoints/finetune",
                   help="Directory for fine-tuning checkpoints")
    p.add_argument("--mode", choices=["domain", "instruction"], default="domain",
                   help="Fine-tuning mode")
    # Domain mode
    p.add_argument("--train_bin", default=None,
                   help="Tokenised training data (.bin) — required for domain mode")
    p.add_argument("--val_bin", default=None,
                   help="Tokenised validation data (.bin) — required for domain mode")
    # Instruction mode
    p.add_argument("--instruction_data", default=None,
                   help="JSONL file with instruction-response pairs")
    # LoRA
    p.add_argument("--use_lora", action="store_true",
                   help="Use LoRA for parameter-efficient fine-tuning")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Training hyper-parameters
    p.add_argument("--micro_batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_lr", type=float, default=3e-5,
                   help="Lower LR than pre-training (default 3e-5)")
    p.add_argument("--min_lr", type=float, default=3e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--checkpoint_interval", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device, autocast_ctx) -> float:
    model.eval()
    total, steps = 0.0, 0
    for x, y in val_loader:
        if steps >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)
        total += loss.item()
        steps += 1
    model.train()
    return total / max(1, steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on device: {device}")

    # ------------------------------------------------------------------
    # Load base model
    # ------------------------------------------------------------------
    print(f"Loading base checkpoint: {args.base_checkpoint}")
    model = GPT.from_checkpoint(args.base_checkpoint, device=str(device))
    model.to(device)

    # ------------------------------------------------------------------
    # (Optionally) inject LoRA
    # ------------------------------------------------------------------
    if args.use_lora:
        from finetune.lora import apply_lora  # noqa: PLC0415
        apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        # Full fine-tune — all parameters trainable.
        for p in model.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.2f} M")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    context_length = model.config.context_length

    if args.mode == "domain":
        if not args.train_bin or not args.val_bin:
            sys.exit("--train_bin and --val_bin required for domain mode")
        train_ds = TokenDataset(args.train_bin, context_length)
        val_ds = TokenDataset(args.val_bin, context_length)
    else:  # instruction
        if not args.instruction_data:
            sys.exit("--instruction_data required for instruction mode")
        try:
            import tiktoken  # noqa: PLC0415
        except ImportError:
            sys.exit("tiktoken is required: pip install tiktoken")
        enc = tiktoken.get_encoding("gpt2")
        train_ds = InstructionDataset(args.instruction_data, enc, context_length)
        val_ds = train_ds  # simple split; a proper split can be added later

    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ------------------------------------------------------------------
    # Mixed precision + scheduler
    # ------------------------------------------------------------------
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16)

    scheduler = CosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad()

    for step in range(1, args.max_steps + 1):
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        optimizer.zero_grad()

        if step % args.log_interval == 0:
            ppl = math.exp(min(accum_loss, 20))
            print(f"step {step:5d} | loss {accum_loss:.4f} | ppl {ppl:.1f} | lr {lr:.2e}")

        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, args.eval_steps, device, autocast_ctx)
            print(f"  → val loss {val_loss:.4f} | val ppl {math.exp(min(val_loss,20)):.1f}")

        if step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"ft_step_{step:06d}.pt")
            if args.use_lora:
                from finetune.lora import get_lora_state_dict  # noqa: PLC0415
                torch.save({"step": step, "lora_state": get_lora_state_dict(model)},
                           ckpt_path)
            else:
                model.save_checkpoint(ckpt_path, step, optimizer.state_dict())
            print(f"  ✓ Checkpoint saved → {ckpt_path}")

    # Final checkpoint.
    final_path = os.path.join(args.out_dir, "ft_final.pt")
    if args.use_lora:
        from finetune.lora import get_lora_state_dict  # noqa: PLC0415
        torch.save({"step": args.max_steps, "lora_state": get_lora_state_dict(model)},
                   final_path)
    else:
        model.save_checkpoint(final_path, args.max_steps, optimizer.state_dict())
    print(f"Fine-tuning complete. Final checkpoint → {final_path}")


if __name__ == "__main__":
    main()
