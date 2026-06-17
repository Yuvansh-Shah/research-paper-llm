"""Fine-tuning script for CustomLM."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config_defaults
from model.gpt import GPT
from tokenization import (
    load_runtime_tokenizer,
    load_tokenizer_spec,
    load_tokenizer_spec_from_checkpoint,
)
from training.dataset import TokenDataset, resolve_tokenizer_spec_from_dataset
from training.lr_schedule import CosineScheduler


INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


class InstructionDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        runtime_tokenizer,
        context_length: int,
        split: str = "all",
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        import random

        self.context_length = context_length
        all_examples: list[list[int]] = []
        with open(jsonl_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                instruction = obj.get("instruction", "")
                inp = obj.get("input", "")
                response = obj.get("output") or obj.get("response", "")
                if inp:
                    instruction = f"{instruction}\n\n{inp}"
                text = INSTRUCTION_TEMPLATE.format(instruction=instruction, response=response)
                tokens = runtime_tokenizer.encode(text, add_eos=True)
                tokens = tokens[: context_length + 1]
                if len(tokens) >= 2:
                    all_examples.append(tokens)

        rng = random.Random(seed)
        indices = list(range(len(all_examples)))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_examples) * val_ratio))
        if split == "train":
            chosen = indices[n_val:]
        elif split == "val":
            chosen = indices[:n_val]
        else:
            chosen = indices
        self.examples = [all_examples[index] for index in chosen]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        tokens = self.examples[idx]
        pad_len = (self.context_length + 1) - len(tokens)
        tokens = tokens + [-1] * pad_len
        chunk = torch.tensor(tokens, dtype=torch.long)
        return chunk[:-1], chunk[1:]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    config_path, defaults = load_config_defaults(argv, section="finetune")
    parser = argparse.ArgumentParser(description="Fine-tune a CustomLM checkpoint")
    parser.add_argument("--config", type=Path, default=config_path, help="Optional JSON preset / recipe file.")
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--out_dir", default="checkpoints/finetune")
    parser.add_argument("--mode", choices=["domain", "instruction"], default="domain")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer override.")
    parser.add_argument("--train_bin", default=None)
    parser.add_argument("--val_bin", default=None)
    parser.add_argument("--instruction_data", default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


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
        total += float(loss.item())
        steps += 1
    model.train()
    return total / max(1, steps)


def resolve_finetune_tokenizer(args: argparse.Namespace):
    base_spec = load_tokenizer_spec_from_checkpoint(args.base_checkpoint)
    if args.tokenizer:
        explicit_spec = load_tokenizer_spec(args.tokenizer)
        if explicit_spec.vocab_size != base_spec.vocab_size:
            raise ValueError("Explicit tokenizer override does not match the base checkpoint tokenizer.")
        return explicit_spec
    if args.mode == "domain":
        manifest_spec = resolve_tokenizer_spec_from_dataset(args.train_bin or "")
        if manifest_spec and manifest_spec.vocab_size != base_spec.vocab_size:
            raise ValueError("Domain fine-tuning dataset tokenizer does not match the base checkpoint.")
    return base_spec


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on device: {device}")

    print(f"Loading base checkpoint: {args.base_checkpoint}")
    model = GPT.from_checkpoint(args.base_checkpoint, device=str(device))
    model.to(device)
    tokenizer_spec = resolve_finetune_tokenizer(args)
    runtime_tokenizer = load_runtime_tokenizer(tokenizer_spec)

    if args.use_lora:
        from finetune.lora import apply_lora  # noqa: PLC0415

        apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.2f} M | tokenizer {tokenizer_spec.name}")

    context_length = model.config.context_length
    if args.mode == "domain":
        if not args.train_bin or not args.val_bin:
            sys.exit("--train_bin and --val_bin are required for domain mode")
        train_ds = TokenDataset(args.train_bin, context_length)
        val_ds = TokenDataset(args.val_bin, context_length)
    else:
        if not args.instruction_data:
            sys.exit("--instruction_data is required for instruction mode")
        train_ds = InstructionDataset(
            args.instruction_data,
            runtime_tokenizer,
            context_length,
            split="train",
            seed=args.seed,
        )
        val_ds = InstructionDataset(
            args.instruction_data,
            runtime_tokenizer,
            context_length,
            split="val",
            seed=args.seed,
        )

    num_workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ctx = (
        torch.amp.autocast(device.type, enabled=use_amp, dtype=torch.float16)
        if use_amp
        else nullcontext()
    )
    scheduler = CosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

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
            accum_loss += float(loss.item())

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            ppl = math.exp(min(accum_loss, 20))
            print(f"step {step:5d} | loss {accum_loss:.4f} | ppl {ppl:.1f} | lr {lr:.2e}")

        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, args.eval_steps, device, autocast_ctx)
            print(f"  -> val loss {val_loss:.4f} | val ppl {math.exp(min(val_loss, 20)):.1f}")

        if step % args.checkpoint_interval == 0:
            ckpt_path = Path(args.out_dir) / f"ft_step_{step:06d}.pt"
            if args.use_lora:
                from finetune.lora import get_lora_state_dict  # noqa: PLC0415

                torch.save(
                    {
                        "step": step,
                        "lora_state": get_lora_state_dict(model),
                        "tokenizer": tokenizer_spec.to_dict(),
                        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                    },
                    ckpt_path,
                )
            else:
                model.save_checkpoint(
                    str(ckpt_path),
                    step,
                    optimizer.state_dict(),
                    extra_metadata={
                        "tokenizer": tokenizer_spec.to_dict(),
                        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                    },
                )
            print(f"  [saved] Checkpoint -> {ckpt_path}")

    final_path = Path(args.out_dir) / "ft_final.pt"
    if args.use_lora:
        from finetune.lora import get_lora_state_dict  # noqa: PLC0415

        torch.save(
            {
                "step": args.max_steps,
                "lora_state": get_lora_state_dict(model),
                "tokenizer": tokenizer_spec.to_dict(),
                "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            },
            final_path,
        )
    else:
        model.save_checkpoint(
            str(final_path),
            args.max_steps,
            optimizer.state_dict(),
            extra_metadata={
                "tokenizer": tokenizer_spec.to_dict(),
                "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            },
        )
    print(f"Fine-tuning complete. Final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
