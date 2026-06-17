# Research-paper LLM

A GPT-style language model built from scratch in PyTorch and trained on research-paper metadata, with the full pipeline around it: tokenizer, data prep, training, evaluation, generation, and fine-tuning.

## What it does

This is a decoder-only transformer (GPT-2 family) implemented from the ground up — custom causal self-attention, weight-tied embeddings, GPT-2-style initialization, and gradient checkpointing. Around the model is an end-to-end pipeline for modelling academic text:

- Data: streams, cleans, deduplicates, and tokenizes the DBLP and arXiv metadata corpora into shards, with deterministic source/domain weighting.
- Tokenizer: trains a domain BPE tokenizer (~32k vocab); GPT-2 BPE is kept as a fallback.
- Training: cosine LR schedule, gradient accumulation, mixed precision (bf16/fp16), multi-GPU DistributedDataParallel, checkpoint resume/pruning, and optional Weights & Biases logging.
- Evaluation: held-out perplexity with per-source and per-domain breakdowns.
- Inference: a sampling CLI with temperature, top-k, and top-p.
- Fine-tuning: LoRA-based domain and instruction tuning.

Model size presets in `model/config.py` range from a ~10M-parameter debug model up to a ~774M XL configuration.

## Built with

- Python, PyTorch
- Hugging Face tokenizers, tiktoken
- LoRA for fine-tuning
- Weights & Biases (optional)

## Running it

```
pip install -r requirements.txt

# smoke test of the full pipeline
python data/prepare.py --smoke
python training/train.py --smoke
python inference/generate.py --checkpoint checkpoints/final.pt --prompt "Geography is"
```

Config presets in `configs/*.json` drive the smoke/pilot/base/large recipes, and most scripts accept `--config`. The raw datasets and model checkpoints are large and are not committed — they're downloaded or generated locally.

Parts of this project were built with AI coding assistance.
