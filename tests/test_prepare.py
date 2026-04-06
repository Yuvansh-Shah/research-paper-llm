from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.prepare import clean_wikipedia_text, tokenize_and_split, verify_dataset_readable


class PreparePipelineTests(unittest.TestCase):
    def test_clean_wikipedia_text_removes_markup(self) -> None:
        raw = "Climate [1] <b>change</b> {| table |}   is real."
        cleaned = clean_wikipedia_text(raw)
        self.assertEqual(cleaned, "Climate change is real.")

    def test_tokenize_split_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train = tmp / "train.bin"
            val = tmp / "val.bin"
            train_n, val_n = tokenize_and_split(
                "The geography of India includes plains and plateaus.",
                train,
                val,
                encode_fn=lambda s: [ord(ch) % 65535 for ch in s],
            )
            self.assertGreater(train_n, 0)
            self.assertGreater(val_n, 0)
            self.assertEqual(np.memmap(train, dtype=np.uint16, mode="r").dtype, np.uint16)
            verify_dataset_readable(train, val)


if __name__ == "__main__":
    unittest.main()
