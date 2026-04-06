# Research-paper-maker-

## Task 1: Dataset preparation (Wikipedia geography corpus)

Run from project root:

```bash
python data/prepare.py --target-mb 200
```

What it does:
- downloads English Wikipedia geography-oriented pages (~target MB of cleaned text),
- cleans text (citation brackets, simple table/template blocks, HTML-like tags, whitespace),
- tokenizes with GPT-2 BPE (`tiktoken`),
- writes `data/processed/train.bin` (95%) and `data/processed/val.bin` (5%) as `uint16`,
- verifies both `.bin` files are readable through `training/dataset.py`,
- prints final split token counts.

Optional offline/local-text mode:

```bash
python data/prepare.py --input-text-file /absolute/path/to/text.txt
```

If tokenizer assets are not yet cached and network is unavailable, the script prints:
- likely cause (offline/blocked `tiktoken` GPT-2 asset fetch),
- exact fix (run once with internet access, then rerun).
