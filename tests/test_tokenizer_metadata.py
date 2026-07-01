from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tokenization import default_gpt2_tokenizer_spec, load_tokenizer_spec
from training.dataset import resolve_document_index_path, resolve_tokenizer_spec_from_dataset


class TokenizerMetadataTests(unittest.TestCase):
    def test_load_gpt2_spec_from_string(self) -> None:
        spec = load_tokenizer_spec("gpt2")
        self.assertEqual(spec.name, "gpt2")
        self.assertEqual(spec.vocab_size, 50_257)

    def test_manifest_prefers_co_located_tokenizer_over_stale_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tokenizer_json = root / "tokenizer.json"
            tokenizer_json.write_text("{}", encoding="utf-8")
            stale_path = root / "elsewhere" / "tokenizer.json"
            manifest = root / "tokenizer_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "tokenizer": {
                            "name": "research-papers-bpe32k",
                            "backend": "hf_tokenizers",
                            "vocab_size": 32_000,
                            "eos_token_id": 1,
                            "tokenizer_path": str(stale_path),
                        }
                    }
                ),
                encoding="utf-8",
            )

            spec = load_tokenizer_spec(manifest)
            self.assertEqual(Path(spec.tokenizer_path), tokenizer_json.resolve())

    def test_manifest_rebases_relative_tokenizer_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tokenizer_json = root / "tokenizer.json"
            tokenizer_json.write_text("{}", encoding="utf-8")
            manifest = root / "tokenizer_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "tokenizer": {
                            "name": "research-papers-bpe32k",
                            "backend": "hf_tokenizers",
                            "vocab_size": 32_000,
                            "eos_token_id": 1,
                            "tokenizer_path": "tokenizer.json",
                        }
                    }
                ),
                encoding="utf-8",
            )

            spec = load_tokenizer_spec(manifest)
            self.assertEqual(Path(spec.tokenizer_path), tokenizer_json.resolve())

    def test_resolve_tokenizer_spec_and_document_index_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = root / "train-00000.bin"
            np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16).tofile(shard)
            document_index = root / "train.documents.jsonl"
            document_index.write_text(
                json.dumps(
                    {
                        "source": "dblp",
                        "global_token_start": 0,
                        "global_token_end": 6,
                        "categories": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            spec = default_gpt2_tokenizer_spec()
            manifest = root / "train.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "token-shards/v2",
                        "split": "train",
                        "dtype": "uint16",
                        "tokenizer": spec.name,
                        "tokenizer_spec": spec.to_dict(),
                        "document_index": document_index.name,
                        "document_count": 1,
                        "token_count": 6,
                        "shards": [{"path": shard.name, "token_count": 6}],
                    }
                ),
                encoding="utf-8",
            )

            resolved_spec = resolve_tokenizer_spec_from_dataset(manifest)
            resolved_index = resolve_document_index_path(manifest)

            self.assertIsNotNone(resolved_spec)
            self.assertEqual(resolved_spec.name, "gpt2")
            self.assertEqual(resolved_index, document_index.resolve())


if __name__ == "__main__":
    unittest.main()
