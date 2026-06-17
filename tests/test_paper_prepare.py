from __future__ import annotations

import csv
import gc
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.prepare_papers import (
    SplitConfig,
    QualityConfig,
    process_corpus,
    stream_arxiv_jsonl,
    stream_dblp_csv,
)
from training.dataset import TokenDataset


def _ascii_encoder(text: str) -> list[int]:
    return [ord(char) for char in text if ord(char) < 65536]


class PaperParserTests(unittest.TestCase):
    def test_stream_dblp_csv_parses_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "dblp.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "abstract",
                        "authors",
                        "title",
                        "venue",
                        "year",
                        "id",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "abstract": "A practical paper about efficient training.",
                        "authors": "['Ana Researcher', 'B. Scientist']",
                        "title": "JoaquÃ­n's Efficient Training",
                        "venue": "Conference on Useful Models",
                        "year": "2019",
                        "id": "dblp-1",
                    }
                )

            records = list(stream_dblp_csv(csv_path))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].source, "dblp")
            self.assertEqual(records[0].source_id, "dblp-1")
            self.assertIn("Joaqu", records[0].title)
            self.assertEqual(records[0].year, "2019")

    def test_stream_arxiv_jsonl_parses_ndjson(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "arxiv.json"
            rows = [
                {
                    "id": "1234.5678",
                    "title": "Learning from Research Corpora",
                    "abstract": "We study pretraining on scientific abstracts at scale.",
                    "authors": "A. Researcher and B. Scientist",
                    "categories": "cs.LG stat.ML",
                    "update_date": "2021-06-01",
                }
            ]
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            records = list(stream_arxiv_jsonl(jsonl_path))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].source, "arxiv")
            self.assertEqual(records[0].categories, ("cs.LG", "stat.ML"))
            self.assertEqual(records[0].year, "2021")


class PaperPipelineTests(unittest.TestCase):
    def test_process_corpus_writes_manifests_and_filters_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dblp_path = root / "dblp.csv"
            arxiv_path = root / "arxiv.json"
            output_dir = root / "processed"

            with dblp_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "abstract",
                        "authors",
                        "title",
                        "venue",
                        "year",
                        "id",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "abstract": "This paper studies document deduplication for scientific language models.",
                        "authors": "['A. One']",
                        "title": "Deduplication for Scientific Language Models",
                        "venue": "NeurIPS",
                        "year": "2022",
                        "id": "dblp-1",
                    }
                )
                writer.writerow(
                    {
                        "abstract": "tiny",
                        "authors": "['A. Two']",
                        "title": "Weak Abstract Example",
                        "venue": "Workshop",
                        "year": "2021",
                        "id": "dblp-2",
                    }
                )

            with arxiv_path.open("w", encoding="utf-8") as handle:
                for index in range(10):
                    row = {
                        "id": f"arxiv-{index}",
                        "title": f"Research Corpus Paper {index}",
                        "abstract": (
                            f"This abstract describes experiment {index} on research-paper "
                            f"pretraining and evaluation over scientific text corpora."
                        ),
                        "authors": "A. Researcher",
                        "categories": "cs.LG stat.ML",
                        "update_date": "2023-01-01",
                    }
                    handle.write(json.dumps(row) + "\n")
                handle.write(
                    json.dumps(
                        {
                            "id": "arxiv-dup",
                            "title": "Deduplication for Scientific Language Models",
                            "abstract": "This paper studies document deduplication for scientific language models.",
                            "authors": "A. One",
                            "categories": "cs.LG",
                            "update_date": "2023-01-01",
                        }
                    )
                    + "\n"
                )

            result = process_corpus(
                dblp_path=dblp_path,
                arxiv_path=arxiv_path,
                output_dir=output_dir,
                split_config=SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
                quality_config=QualityConfig(min_abstract_chars=20),
                shard_size_tokens=64,
                buffer_tokens=16,
                encode_fn=_ascii_encoder,
                tokenizer_name="ascii-test",
                verbose=False,
            )

            self.assertEqual(result.stats.total_records_seen, 13)
            self.assertEqual(result.stats.dropped_by_reason["short_abstract"], 1)
            self.assertEqual(result.stats.dropped_by_reason["duplicate_across_sources"], 1)
            self.assertEqual(result.stats.kept_records, 11)
            self.assertTrue(result.stats_path.exists())
            self.assertTrue(result.corpus_manifest_path.exists())
            for manifest in result.split_manifests.values():
                self.assertTrue(manifest.exists())

            best_split = max(
                result.stats.records_by_split,
                key=lambda split: result.stats.records_by_split[split],
            )
            dataset = TokenDataset(str(result.split_manifests[best_split]), context_length=8)
            self.assertGreater(len(dataset), 0)
            x, y = dataset[0]
            self.assertEqual(x.shape[0], 8)
            self.assertEqual(y.shape[0], 8)
            dataset.close()
            del dataset
            gc.collect()

    def test_token_dataset_reads_sharded_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_a = root / "train-00000.bin"
            shard_b = root / "train-00001.bin"
            np.array([0, 1, 2, 3, 4], dtype=np.uint16).tofile(shard_a)
            np.array([5, 6, 7, 8, 9], dtype=np.uint16).tofile(shard_b)

            manifest = root / "train.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "token-shards/v1",
                        "split": "train",
                        "dtype": "uint16",
                        "tokenizer": "test",
                        "document_count": 2,
                        "token_count": 10,
                        "shards": [
                            {"path": shard_a.name, "token_count": 5},
                            {"path": shard_b.name, "token_count": 5},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = TokenDataset(str(manifest), context_length=4)
            x, y = dataset[2]
            self.assertEqual(x.tolist(), [2, 3, 4, 5])
            self.assertEqual(y.tolist(), [3, 4, 5, 6])
            dataset.close()
            del dataset
            gc.collect()


if __name__ == "__main__":
    unittest.main()
