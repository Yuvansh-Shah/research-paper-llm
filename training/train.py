"""Main pre-training loop for CustomLM.

Usage::

    python training/train.py \\
        --train_bin data/processed/train.bin \\
        --val_bin   data/processed/val.bin \\
        --out_dir    checkpoints \\
        --model_size small          # or 'medium'

Key features
------------
* Mixed-precision training (FP16) via ``torch.cuda.amp``
* Gradient accumulation to simulate large batch sizes on limited VRAM
* Gradient clipping to prevent exploding gradients
* Cosine LR scheduler with linear warm-up
* Checkpoint saving every ``--checkpoint_interval`` steps
* Optional TensorBoard / Weights & Biases logging
* Optional gradient checkpointing for additional VRAM savings
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import GPTConfig
from model.gpt import GPT
from training.dataset import TokenDataset
from training.lr_schedule import CosineScheduler


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train a GPT-style language model")
    p.add_argument("--train_bin", default="data/processed/train.bin",
                   help="Path to tokenised training data (.bin)")
    p.add_argument("--val_bin", default="data/processed/val.bin",
                   help="Path to tokenised validation data (.bin)")
    p.add_argument("--out_dir", default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--model_size", default="small", choices=["small", "medium"],
                   help="Model size preset")
    p.add_argument("--context_length", type=int, default=None,
                   help="Override context length from config")
    p.add_argument("--micro_batch_size", type=int, default=4,
                   help="Micro-batch size per GPU step (keep small for 4 GB VRAM)")
    p.add_argument("--grad_accum_steps", type=int, default=8,
                   help="Gradient accumulation steps (effective_batch = micro × accum)")
    p.add_argument("--max_steps", type=int, default=100_000,
                   help="Total number of optimiser steps")
    p.add_argument("--warmup_steps", type=int, default=2_000,
                   help="Number of linear LR warm-up steps")
    p.add_argument("--max_lr", type=float, default=3e-4,
                   help="Peak learning rate")
    p.add_argument("--min_lr", type=float, default=3e-5,
                   help="Minimum LR (end of cosine decay)")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping max norm (0 = disabled)")
    p.add_argument("--eval_interval", type=int, default=500,
                   help="Evaluate on validation set every N optimiser steps")
    p.add_argument("--eval_steps", type=int, default=50,
                   help="Number of validation batches to average")
    p.add_argument("--log_interval", type=int, default=10,
                   help="Print training loss every N optimiser steps")
    p.add_argument("--checkpoint_interval", type=int, default=1_000,
                   help="Save checkpoint every N optimiser steps")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM usage")
    p.add_argument("--use_wandb", action="store_true",
                   help="Log metrics to Weights & Biases")
    p.add_argument("--wandb_project", default="CustomLM",
                   help="W&B project name")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: GPT,
    val_loader: DataLoader,
    eval_steps: int,
    device: torch.device,
    autocast_ctx,
) -> float:
    """Run evaluation and return mean validation loss."""
    model.eval()
    total_loss = 0.0
    steps = 0
    for x, y in val_loader:
        if steps >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        steps += 1
    model.train()
    return total_loss / max(1, steps)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    cfg_cls = GPTConfig.small if args.model_size == "small" else GPTConfig.medium
    config: GPTConfig = cfg_cls()
    if args.context_length is not None:
        config.context_length = args.context_length

    model = GPT(config).to(device)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled.")
    print(f"Model parameters: {model.num_parameters() / 1e6:.1f} M")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_dataset = TokenDataset(args.train_bin, config.context_length)
    val_dataset = TokenDataset(args.val_bin, config.context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Optimiser — separate weight-decayed and non-decayed parameters
    # ------------------------------------------------------------------
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )

    # ------------------------------------------------------------------
    # Mixed-precision scaler
    # ------------------------------------------------------------------
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16)

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------
    scheduler = CosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
    )

    # ------------------------------------------------------------------
    # Optional: resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = payload.get("step", 0)
        scheduler.load_state_dict({"step": start_step})
        print(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------
    # Optional: W&B logging
    # ------------------------------------------------------------------
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()

    train_iter = iter(train_loader)
    optimizer.zero_grad()

    step = start_step
    t0 = time.perf_counter()

    while step < args.max_steps:
        # Accumulate gradients over micro-batches.
        accumulated_loss = 0.0
        for micro_step in range(args.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            with autocast_ctx:
                _, loss = model(x, y)
                # Scale loss to account for gradient accumulation.
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        # Gradient clipping.
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimiser + LR scheduler step.
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        optimizer.zero_grad()
        step += 1

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if step % args.log_interval == 0:
            t1 = time.perf_counter()
            dt = t1 - t0
            tokens_per_sec = (
                args.log_interval
                * args.grad_accum_steps
                * args.micro_batch_size
                * config.context_length
                / dt
            )
            ppl = math.exp(min(accumulated_loss, 20))  # cap to avoid overflow
            print(
                f"step {step:6d} | loss {accumulated_loss:.4f} | ppl {ppl:.1f} "
                f"| lr {lr:.2e} | {tokens_per_sec:.0f} tok/s"
            )
            t0 = t1
            if args.use_wandb:
                import wandb
                wandb.log({"train/loss": accumulated_loss, "train/ppl": ppl,
                           "train/lr": lr, "train/tokens_per_sec": tokens_per_sec},
                          step=step)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, args.eval_steps, device, autocast_ctx)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"  → val loss {val_loss:.4f} | val ppl {val_ppl:.1f}")
            if args.use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"step_{step:06d}.pt")
            model.save_checkpoint(ckpt_path, step, optimizer.state_dict())
            print(f"  ✓ Checkpoint saved → {ckpt_path}")

    # Final checkpoint.
    final_path = os.path.join(args.out_dir, "final.pt")
    model.save_checkpoint(final_path, step, optimizer.state_dict())
    print(f"Training complete. Final checkpoint → {final_path}")

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train(parse_args())
