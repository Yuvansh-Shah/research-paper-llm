# research-paper-llm

A small GPT-style language model built **from scratch in PyTorch** — no pretrained weights, no Hugging Face `transformers` — and trained on the titles and abstracts of ~2.7 million arXiv papers. The repo contains the full pipeline: data ingestion and cleaning, a domain BPE tokenizer, training, evaluation, a sampling CLI, and a local web demo.

This is deliberately a **small, honestly-scoped project**: a 35.5M-parameter model trained for ~410M tokens on a single 4GB laptop GPU. It writes plausible-sounding research-paper prose and picks up the register and structure of academic abstracts across CS, math, and physics. It is not, and does not pretend to be, a general-purpose assistant — see [Limitations](#limitations).

## Demo

```
python server/app.py --checkpoint <path-to>/final.pt
# open http://127.0.0.1:7860
```

Type a prompt (e.g. `Title: Attention-based models for`), watch the model stream a completion, and open the About panel for the architecture, training data, and evaluation numbers.

## Model

| | |
|---|---|
| Architecture | decoder-only transformer (GPT-2 family), pre-LayerNorm |
| Parameters | 35.5M (weight-tied embeddings, counted once) |
| Layers / heads / width | 6 / 8 / 512 |
| Context length | 512 tokens |
| Positional encoding | learned absolute |
| Attention | fused-QKV causal self-attention (PyTorch SDPA, manual fallback) |
| FFN | 4× GELU |
| Tokenizer | byte-level BPE, 32,000 vocab, trained on this corpus |

The model code is ~600 lines in [model/](model/): `attention.py`, `transformer.py`, `gpt.py`, `config.py`. Presets from ~16M (`debug`) to ~775M (`xl`) exist in `config.py`; **only `small` has been trained here** — the larger presets are untested recipes, not claims.

## Training data

Source: the [arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) (titles + abstracts, ~2.71M records; also available via public Hugging Face mirrors). The pipeline ([data/prepare_papers.py](data/prepare_papers.py)):

- cleans text (mojibake repair, HTML unescape, NFKC, LaTeX stripping, whitespace),
- filters low-quality records (2,710,791 seen → 2,701,937 kept; drops: 3,662 short abstracts, 2,294 low-alpha, 1,430 weak titles, 1,423 within-source duplicates, 45 missing title/abstract),
- deduplicates by title+abstract signature,
- splits train/val/test **grouped by title signature** so near-duplicate papers cannot leak across splits,
- shuffles with a bounded-RAM external bucket shuffle and writes uint16 token shards with manifests.

Result: **566.2M train / 2.88M val / 2.84M test tokens.** The pipeline also supports DBLP-v10 CSV as a second source; this trained checkpoint is arXiv-only.

Documents are formatted as:

```
Title: <title>
Abstract: <abstract><|endoftext|>
```

## Training

Single NVIDIA RTX 3050 Laptop GPU (4GB). bf16 autocast, AdamW (weight-decay param groups), cosine LR schedule with linear warmup, gradient clipping, gradient accumulation (effective batch 32 × 512 tokens), checkpoint/resume. 25,000 steps ≈ 410M tokens ≈ 0.7 epochs — no data repetition.

```
python training/train.py --config configs/paper_small.json
```

The trainer is single-process by design; there is no multi-GPU path because this project never had multi-GPU hardware to test one on.

## Evaluation

_TODO_EVAL_SECTION_ (held-out perplexity, sample generations, and an honest capability summary — filled in from `evaluation/results/` after the training run)

```
python evaluation/evaluate.py --config configs/paper_small.json --checkpoint <path-to>/final.pt
```

Perplexity is computed over non-overlapping context-length windows (every token scored exactly once), with per-domain breakdowns from the document index.

## Running it yourself

```
pip install -r requirements.txt

# 1) offline smoke test of the whole loop (no data needed, CPU-friendly, ~1 min)
python training/train.py --smoke
python inference/generate.py --checkpoint checkpoints/smoke/final.pt --prompt "Title: Deep learning"

# 2) real corpus: download arxiv-metadata-oai-snapshot.json (see Training data)
#    into data/raw/, then one command builds tokenizer + shards:
python data/build_corpus.py

# 3) train (adjust micro_batch_size to your VRAM; this recipe fits in ~2.6GB)
python training/train.py --config configs/paper_small.json

# 4) evaluate + serve the demo
python evaluation/evaluate.py --config configs/paper_small.json --checkpoint checkpoints/papers_small/final.pt
python server/app.py --checkpoint checkpoints/papers_small/final.pt
```

Tests: `python -m unittest discover -s tests` (72 tests).

## Repo layout

```
model/        the transformer (attention, blocks, GPT, config presets)
data/         corpus download/build, cleaning pipeline, BPE tokenizer training
tokenization/ runtime tokenizer loading (domain BPE, GPT-2 fallback)
training/     training loop, LR schedule, memmapped token dataset
evaluation/   windowed perplexity + prompt-suite generations, report writer
inference/    sampling CLI and streaming generator API
server/       Flask demo UI (streaming generation + about panel)
tests/        unit + integration tests
configs/      training/eval recipes (paper_small is the trained one)
```

## Limitations

Honest ones, because they matter:

- **It completes prose; it doesn't know things.** At 35.5M parameters it produces fluent academic-sounding text, but named entities, citations, numbers, and "results" are frequently invented. It should never be used as a source of facts.
- **Trained on metadata only** — titles and abstracts, not full papers. It has never seen a proof, a method section, or a table.
- **512-token context.** Long prompts get cropped; it cannot write a paper, only abstract-length passages.
- **English-only, arXiv-flavored.** Out-of-domain prompts (dialogue, code, news) degrade quickly.
- _TODO_EVAL_LIMITATIONS_ (concrete failure modes observed in evaluation)

Parts of this project were built with AI coding assistance.
