"""Main pre-training loop for research-paper language models."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config_defaults
from evaluation.prompt_suites import get_prompts
from inference.generate import generate
from model.config import GPTConfig
from model.gpt import GPT
from tokenization import (
    TokenizerSpec,
    default_gpt2_tokenizer_spec,
    load_runtime_tokenizer,
    load_tokenizer_spec,
    load_tokenizer_spec_from_checkpoint,
)
from training.dataset import TokenDataset, resolve_tokenizer_spec_from_dataset
from training.lr_schedule import CosineScheduler


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def cleanup_ddp() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def _set_defaults_from_config(parser: argparse.ArgumentParser, defaults: dict) -> None:
    if defaults:
        parser.set_defaults(**defaults)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    config_path, defaults = load_config_defaults(argv, section="training")
    parser = argparse.ArgumentParser(description="Pre-train a GPT-style language model")
    parser.add_argument("--config", type=Path, default=config_path, help="Optional JSON preset / recipe file.")

    parser.add_argument("--train_bin", default="data/processed/train.bin")
    parser.add_argument("--val_bin", default="data/processed/val.bin")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer override.")

    parser.add_argument("--model_size", default="base", choices=["debug", "small", "base", "large", "xl"])
    parser.add_argument("--context_length", type=int, default=None)

    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1_000)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3)
    parser.add_argument("--eval_before_checkpoint", action="store_true")

    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from, or 'auto'.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--sample_prompt_suite", default="research_capability", choices=["generic", "research", "research_capability"])
    parser.add_argument("--sample_prompts", default=None, help="Optional prompt file for training snapshots.")
    parser.add_argument("--sample_count", type=int, default=3)
    parser.add_argument("--sample_max_new_tokens", type=int, default=96)
    parser.add_argument("--sample_temperature", type=float, default=0.8)
    parser.add_argument("--sample_top_k", type=int, default=40)
    parser.add_argument("--sample_top_p", type=float, default=0.95)
    parser.add_argument("--disable_training_samples", action="store_true")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="CustomLM")
    parser.add_argument("--wandb_run_name", default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    _set_defaults_from_config(parser, defaults)
    return parser.parse_args(argv)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    eval_steps: int,
    device: torch.device,
    autocast_ctx,
    rank: int,
    world_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0

    for x, y in val_loader:
        if steps >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            if hasattr(model, "module"):
                _, loss = model.module(x, y)
            else:
                _, loss = model(x, y)
        total_loss += float(loss.item())
        steps += 1

    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, steps], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(loss_tensor)
        total_loss = float(loss_tensor[0].item())
        steps = int(loss_tensor[1].item())

    model.train()
    return total_loss / max(1, steps)


def resolve_tokenizer_spec(args: argparse.Namespace) -> TokenizerSpec:
    dataset_specs = []
    for path in (args.train_bin, args.val_bin):
        spec = resolve_tokenizer_spec_from_dataset(path)
        if spec is not None:
            dataset_specs.append(spec)

    dataset_spec = dataset_specs[0] if dataset_specs else None
    if len(dataset_specs) >= 2:
        first = dataset_specs[0]
        for other in dataset_specs[1:]:
            if (
                other.vocab_size != first.vocab_size
                or other.eos_token_id != first.eos_token_id
                or other.name != first.name
            ):
                raise ValueError("Train/val manifests reference different tokenizers.")

    if args.tokenizer:
        explicit_spec = load_tokenizer_spec(args.tokenizer)
        if dataset_spec and (
            explicit_spec.vocab_size != dataset_spec.vocab_size
            or explicit_spec.eos_token_id != dataset_spec.eos_token_id
        ):
            raise ValueError("Explicit tokenizer override does not match dataset manifests.")
        return explicit_spec

    return dataset_spec or default_gpt2_tokenizer_spec()


def build_model_config(args: argparse.Namespace, tokenizer_spec: TokenizerSpec) -> GPTConfig:
    config = GPTConfig.from_name(
        args.model_size,
        vocab_size=tokenizer_spec.vocab_size,
        context_length=args.context_length,
    )
    return config


def find_resume_checkpoint(out_dir: str | Path) -> Optional[Path]:
    out_dir = Path(out_dir)
    latest_path = out_dir / "latest.pt"
    if latest_path.exists():
        return latest_path
    checkpoints = sorted(out_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def prune_old_checkpoints(out_dir: str | Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    out_dir = Path(out_dir)
    checkpoints = sorted(out_dir.glob("step_*.pt"))
    if len(checkpoints) <= keep_last_n:
        return
    for path in checkpoints[:-keep_last_n]:
        path.unlink(missing_ok=True)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def create_run_metadata(
    args: argparse.Namespace,
    config: GPTConfig,
    tokenizer_spec: TokenizerSpec,
    world_size: int,
    device: torch.device,
) -> dict:
    return {
        "created_at": _now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "world_size": world_size,
        "device": str(device),
        "model_size": args.model_size,
        "model_config": config.__dict__,
        "tokenizer": tokenizer_spec.to_dict(),
        "train_bin": str(Path(args.train_bin).resolve()),
        "val_bin": str(Path(args.val_bin).resolve()),
        "args": vars(args),
    }


def save_generation_snapshot(
    *,
    model: nn.Module,
    tokenizer_spec: TokenizerSpec,
    step: int,
    out_dir: str | Path,
    prompts_file: Optional[str],
    prompt_suite: str,
    prompt_count: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> None:
    runtime_tokenizer = load_runtime_tokenizer(tokenizer_spec)
    prompts, resolved_suite = get_prompts(prompts_file, prompt_suite)
    prompts = prompts[:prompt_count]
    raw_model = model.module if hasattr(model, "module") else model

    samples = []
    raw_model.eval()
    try:
        for prompt in prompts:
            prompt_ids = runtime_tokenizer.encode(prompt, add_eos=False)
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            output_ids = generate(
                model=raw_model,
                idx=idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            token_ids = output_ids[0].tolist()
            samples.append(
                {
                    "prompt": prompt,
                    "completion": runtime_tokenizer.decode(token_ids, skip_special_tokens=False),
                }
            )
    finally:
        raw_model.train()

    sample_dir = Path(out_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        sample_dir / f"step_{step:06d}.json",
        {
            "step": step,
            "prompt_suite": resolved_suite,
            "samples": samples,
        },
    )


def build_checkpoint_payload(
    *,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: CosineScheduler,
    step: int,
    args: argparse.Namespace,
    tokenizer_spec: TokenizerSpec,
    best_val_loss: Optional[float],
    last_val_loss: Optional[float],
    last_eval_step: int,
    run_metadata: dict,
) -> dict:
    return {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "config": raw_model.config.__dict__,
        "args": vars(args),
        "tokenizer": tokenizer_spec.to_dict(),
        "dataset": {
            "train_bin": str(Path(args.train_bin).resolve()),
            "val_bin": str(Path(args.val_bin).resolve()),
        },
        "training_state": {
            "best_val_loss": best_val_loss,
            "last_val_loss": last_val_loss,
            "last_eval_step": last_eval_step,
        },
        "run_metadata": run_metadata,
        "saved_at": _now_iso(),
    }


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = setup_ddp()
    is_main = is_main_process(rank)

    torch.manual_seed(args.seed + rank)

    if args.smoke:
        if is_main:
            print("=== SMOKE TEST MODE ===")
        args.model_size = "debug"
        args.max_steps = 50
        args.warmup_steps = 5
        args.log_interval = 5
        args.eval_interval = 25
        args.eval_steps = 5
        args.checkpoint_interval = 50
        args.grad_accum_steps = 2
        args.micro_batch_size = 2
        args.keep_last_n_checkpoints = max(args.keep_last_n_checkpoints, 1)
        if args.context_length is None:
            args.context_length = 64

    tokenizer_spec = resolve_tokenizer_spec(args)
    config = build_model_config(args, tokenizer_spec)
    model = GPT(config).to(device)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if is_main:
            print("Gradient checkpointing enabled.")

    if args.compile and hasattr(torch, "compile"):
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    raw_model = model.module if hasattr(model, "module") else model
    if is_main:
        print(f"Training on {world_size} worker(s), device: {device}")
        print(f"Tokenizer: {tokenizer_spec.name} | vocab={tokenizer_spec.vocab_size:,}")
        print(f"Model: {args.model_size} | Parameters: {raw_model.num_parameters() / 1e6:.1f}M")
        print(f"Context length: {config.context_length}")

    num_workers = 0 if platform.system() == "Windows" else 4
    train_dataset = TokenDataset(args.train_bin, config.context_length)
    val_dataset = TokenDataset(args.val_bin, config.context_length)
    if len(train_dataset) <= 0 or len(val_dataset) <= 0:
        raise ValueError("Training or validation dataset is empty for the selected context length.")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
    use_fused = device.type == "cuda"
    if use_fused:
        try:
            torch.optim.AdamW([torch.zeros(1, device=device)], fused=True)
        except (RuntimeError, TypeError):
            use_fused = False

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr,
        betas=(0.9, 0.95),
        fused=use_fused,
    )

    use_amp = device.type == "cuda" and args.dtype != "fp32"
    if args.dtype == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.dtype == "fp16" or (args.dtype == "bf16" and use_amp):
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    autocast_ctx = (
        torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype)
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

    run_metadata = create_run_metadata(args, raw_model.config, tokenizer_spec, world_size, device)
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "run_metadata.json", run_metadata)

    start_step = 0
    best_val_loss: Optional[float] = None
    last_val_loss: Optional[float] = None
    last_eval_step = 0

    resume_path = args.resume
    if resume_path == "auto":
        auto_path = find_resume_checkpoint(out_dir)
        resume_path = str(auto_path) if auto_path else None
        if resume_path and is_main:
            print(f"Auto-resume: found checkpoint {resume_path}")

    if resume_path and Path(resume_path).is_file():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scaler_state_dict" in payload and scaler.is_enabled():
            scaler.load_state_dict(payload["scaler_state_dict"])
        if "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_step = int(payload.get("step", 0))
        training_state = payload.get("training_state", {})
        best_val_loss = training_state.get("best_val_loss")
        last_val_loss = training_state.get("last_val_loss")
        last_eval_step = int(training_state.get("last_eval_step", 0))
        checkpoint_tokenizer = load_tokenizer_spec_from_checkpoint(resume_path, payload=payload)
        if checkpoint_tokenizer.vocab_size != tokenizer_spec.vocab_size:
            raise ValueError("Resume checkpoint tokenizer does not match current dataset/tokenizer.")
        if is_main:
            print(f"Resumed from step {start_step}")

    if args.use_wandb and is_main:
        import wandb  # noqa: PLC0415

        run_name = args.wandb_run_name or f"{args.model_size}-{tokenizer_spec.name}"
        wandb.init(project=args.wandb_project, name=run_name, config=run_metadata)

    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    t0 = time.perf_counter()

    effective_batch = args.micro_batch_size * args.grad_accum_steps * world_size
    if is_main:
        print(f"Effective batch size: {effective_batch}")
        print(f"Starting training from step {start_step}...")

    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(step // max(1, len(train_loader)))

        accumulated_loss = 0.0
        for micro_step in range(args.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            sync_context = nullcontext()
            if world_size > 1 and micro_step < args.grad_accum_steps - 1:
                sync_context = model.no_sync()

            with sync_context:
                with autocast_ctx:
                    if hasattr(model, "module"):
                        _, loss = model.module(x, y)
                    else:
                        _, loss = model(x, y)
                    loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()
                accumulated_loss += float(loss.item())

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_interval == 0 and is_main:
            elapsed = time.perf_counter() - t0
            tokens_per_sec = (
                args.log_interval
                * args.grad_accum_steps
                * args.micro_batch_size
                * config.context_length
                * world_size
                / max(elapsed, 1e-6)
            )
            ppl = math.exp(min(accumulated_loss, 20))
            print(
                f"step {step:6d} | loss {accumulated_loss:.4f} | ppl {ppl:.1f} "
                f"| lr {lr:.2e} | {tokens_per_sec:.0f} tok/s"
            )
            t0 = time.perf_counter()
            if args.use_wandb:
                import wandb  # noqa: PLC0415

                wandb.log(
                    {
                        "train/loss": accumulated_loss,
                        "train/ppl": ppl,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                    },
                    step=step,
                )

        should_eval = step % args.eval_interval == 0
        should_checkpoint = step % args.checkpoint_interval == 0
        if should_checkpoint and args.eval_before_checkpoint and last_eval_step != step:
            should_eval = True

        if should_eval:
            val_loss = evaluate(model, val_loader, args.eval_steps, device, autocast_ctx, rank, world_size)
            last_val_loss = val_loss
            last_eval_step = step
            is_best = best_val_loss is None or val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            if is_main:
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  -> val loss {val_loss:.4f} | val ppl {val_ppl:.1f}")
                metrics_payload = {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "best_val_loss": best_val_loss,
                    "evaluated_at": _now_iso(),
                }
                write_json(out_dir / "latest_metrics.json", metrics_payload)
                if args.use_wandb:
                    import wandb  # noqa: PLC0415

                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl, "val/best_loss": best_val_loss}, step=step)
                if not args.disable_training_samples:
                    save_generation_snapshot(
                        model=model,
                        tokenizer_spec=tokenizer_spec,
                        step=step,
                        out_dir=out_dir,
                        prompts_file=args.sample_prompts,
                        prompt_suite=args.sample_prompt_suite,
                        prompt_count=args.sample_count,
                        max_new_tokens=args.sample_max_new_tokens,
                        temperature=args.sample_temperature,
                        top_k=args.sample_top_k,
                        top_p=args.sample_top_p,
                        device=device,
                    )
                if is_best:
                    best_payload = build_checkpoint_payload(
                        raw_model=raw_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        step=step,
                        args=args,
                        tokenizer_spec=tokenizer_spec,
                        best_val_loss=best_val_loss,
                        last_val_loss=last_val_loss,
                        last_eval_step=last_eval_step,
                        run_metadata=run_metadata,
                    )
                    torch.save(best_payload, out_dir / "best.pt")
                    write_json(
                        out_dir / "best_checkpoint.json",
                        {"step": step, "val_loss": val_loss, "saved_at": _now_iso()},
                    )

        if should_checkpoint and is_main:
            payload = build_checkpoint_payload(
                raw_model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                step=step,
                args=args,
                tokenizer_spec=tokenizer_spec,
                best_val_loss=best_val_loss,
                last_val_loss=last_val_loss,
                last_eval_step=last_eval_step,
                run_metadata=run_metadata,
            )
            step_path = out_dir / f"step_{step:06d}.pt"
            torch.save(payload, step_path)
            torch.save(payload, out_dir / "latest.pt")
            prune_old_checkpoints(out_dir, args.keep_last_n_checkpoints)
            print(f"  [saved] Checkpoint -> {step_path}")

    if is_main:
        final_payload = build_checkpoint_payload(
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            step=step,
            args=args,
            tokenizer_spec=tokenizer_spec,
            best_val_loss=best_val_loss,
            last_val_loss=last_val_loss,
            last_eval_step=last_eval_step,
            run_metadata=run_metadata,
        )
        torch.save(final_payload, out_dir / "final.pt")
        torch.save(final_payload, out_dir / "latest.pt")
        write_json(
            out_dir / "training_summary.json",
            {
                "finished_at": _now_iso(),
                "step": step,
                "best_val_loss": best_val_loss,
                "last_val_loss": last_val_loss,
                "tokenizer": tokenizer_spec.to_dict(),
                "model_config": raw_model.config.__dict__,
            },
        )
        print(f"Training complete. Final checkpoint -> {out_dir / 'final.pt'}")
        if args.use_wandb:
            import wandb  # noqa: PLC0415

            wandb.finish()

    train_dataset.close()
    val_dataset.close()
    cleanup_ddp()


if __name__ == "__main__":
    train(parse_args())
