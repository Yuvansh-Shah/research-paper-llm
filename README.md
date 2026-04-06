# CustomLM — GPT-Style Language Model from Scratch

A small, domain-focused Large Language Model built entirely from scratch
using Python and PyTorch, designed to run on consumer-grade hardware
(NVIDIA RTX 3050 / 4 GB VRAM).

> **Goal:** Understand the full LLM pipeline — architecture, training,
> fine-tuning, and inference — while producing a functional model capable
> of coherent text generation.

---

## Project Structure

```
CustomLM/
├── data/
│   ├── raw/              # Downloaded raw text (git-ignored)
│   ├── processed/        # Tokenised .bin files (git-ignored)
│   └── prepare.py        # Data download + preprocessing script
├── model/
│   ├── config.py         # Model hyperparameter dataclass
│   ├── attention.py      # Multi-head causal self-attention
│   ├── transformer.py    # Transformer block (pre-LN + residuals)
│   └── gpt.py            # Full GPT model class
├── training/
│   ├── dataset.py        # PyTorch Dataset (memory-mapped .bin files)
│   ├── lr_schedule.py    # Cosine LR scheduler with linear warm-up
│   └── train.py          # Main pre-training loop
├── inference/
│   └── generate.py       # Text generation CLI
├── finetune/
│   ├── lora.py           # LoRA implementation
│   └── finetune.py       # Fine-tuning script (domain + instruction)
├── checkpoints/          # Saved model weights (git-ignored)
├── logs/                 # TensorBoard / W&B logs (git-ignored)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

CUDA 12.1 + PyTorch 2.x is recommended for GPU training:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare data

```bash
# Download and tokenise 50 k English Wikipedia articles
python data/prepare.py --source wikipedia --num_articles 50000 \
    --output_dir data/processed

# Or tokenise a local text file
python data/prepare.py --source file --input_path my_corpus.txt \
    --output_dir data/processed
```

### 3. Train

```bash
python training/train.py \
    --train_bin data/processed/train.bin \
    --val_bin   data/processed/val.bin \
    --out_dir   checkpoints \
    --model_size small            # 60 M params, fits in 4 GB VRAM
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--micro_batch_size` | 4 | Micro-batch size per GPU step |
| `--grad_accum_steps` | 8 | Gradient accumulation steps |
| `--gradient_checkpointing` | off | Save VRAM at cost of compute |
| `--max_lr` | 3e-4 | Peak learning rate |
| `--max_steps` | 100 000 | Total optimiser steps |
| `--use_wandb` | off | Log to Weights & Biases |

### 4. Generate text

```bash
python inference/generate.py \
    --checkpoint checkpoints/final.pt \
    --prompt "The history of machine learning" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_p 0.9
```

### 5. Fine-tune (domain adaptation)

```bash
python finetune/finetune.py \
    --base_checkpoint checkpoints/final.pt \
    --train_bin data/processed/domain_train.bin \
    --val_bin   data/processed/domain_val.bin \
    --mode domain \
    --use_lora \
    --max_lr 3e-5
```

### 6. Instruction fine-tuning (LoRA)

```bash
python finetune/finetune.py \
    --base_checkpoint checkpoints/final.pt \
    --instruction_data data/raw/instructions.jsonl \
    --mode instruction \
    --use_lora
```

The JSONL file should follow the Alpaca format:

```json
{"instruction": "Summarise the following text.", "input": "...", "output": "..."}
```

---

## Model Architecture

GPT-2-style decoder-only Transformer:

| Component | Specification |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters (small) | ~60 M |
| Parameters (medium) | ~120 M |
| Context window | 256 tokens (small) / 512 (medium) |
| Attention heads | 8 / 12 |
| Transformer layers | 6 / 12 |
| Embedding dimension | 512 / 768 |
| Vocabulary size | 50 257 (GPT-2 BPE) |
| Activation | GELU |
| Positional encoding | Learned |
| Normalisation | Pre-LayerNorm |

---

## VRAM Optimisations (RTX 3050 4 GB)

- **Mixed-precision FP16** via `torch.cuda.amp` — ~40–50 % VRAM reduction
- **Gradient checkpointing** via `--gradient_checkpointing`
- **Gradient accumulation** — effective batch size without extra VRAM
- **LoRA fine-tuning** — only 2–20 MB of trainable weights

---

## Development Milestones

- [x] Phase 1 — Environment setup
- [x] Phase 2 — Transformer architecture (`model/`)
- [x] Phase 3 — Training loop (`training/`)
- [x] Phase 6 — Inference engine (`inference/generate.py`)
- [x] Phase 7 — LoRA fine-tuning (`finetune/`)
- [ ] Phase 4 — First pretraining run (loss decreasing)
- [ ] Phase 5 — Full pretraining on curated corpus

---

## References

| Resource | Link |
|---|---|
| LLMs from Scratch — Sebastian Raschka | https://github.com/rasbt/LLMs-from-scratch |
| Andrej Karpathy — Neural Networks: Zero to Hero | https://youtube.com/@AndrejKarpathy |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 |
| LoRA paper | https://arxiv.org/abs/2106.09685 |
