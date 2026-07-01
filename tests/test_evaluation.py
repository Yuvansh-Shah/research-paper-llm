"""Tests for perplexity evaluation and prompt suite resolution."""

from __future__ import annotations

import contextlib
import io
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from evaluation.evaluate import evaluate_perplexity
from evaluation.prompt_suites import PROMPT_SUITES, get_prompts
from model.config import GPTConfig
from model.gpt import GPT


class UniformLogitModel(nn.Module):
    """Model emitting all-zero logits so cross-entropy is exactly log(vocab_size)."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None):
        logits = torch.zeros(idx.shape[0], idx.shape[1], self.vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss


class TestEvaluatePerplexity(unittest.TestCase):
    """Tests for non-overlapping windowed perplexity."""

    VOCAB_SIZE = 50
    CONTEXT_LENGTH = 32
    NUM_TOKENS = 2000

    @classmethod
    def setUpClass(cls) -> None:
        """Write a tiny synthetic token .bin with uniform-random tokens."""
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="test_evaluation_"))
        rng = np.random.default_rng(seed=0)
        cls.tokens = rng.integers(0, cls.VOCAB_SIZE, size=cls.NUM_TOKENS, dtype=np.uint16)
        cls.bin_path = cls.tmp_dir / "tokens.bin"
        cls.tokens.tofile(cls.bin_path)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_uniform_model_perplexity_is_exactly_vocab_size(self) -> None:
        """A uniform-logits model must score ppl == vocab_size and loss == log(V)."""
        model = UniformLogitModel(self.VOCAB_SIZE)
        result = evaluate_perplexity(
            model,
            str(self.bin_path),
            self.CONTEXT_LENGTH,
            batch_size=8,
            device=torch.device("cpu"),
            split_name="test",
        )
        self.assertAlmostEqual(result.loss, math.log(self.VOCAB_SIZE), places=5)
        self.assertAlmostEqual(result.perplexity, float(self.VOCAB_SIZE), places=3)

    def test_every_token_scored_exactly_once(self) -> None:
        """Non-overlapping windows must score all tokens except the first."""
        model = UniformLogitModel(self.VOCAB_SIZE)
        result = evaluate_perplexity(
            model,
            str(self.bin_path),
            self.CONTEXT_LENGTH,
            batch_size=8,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.num_tokens, self.NUM_TOKENS - 1)

    def test_untrained_model_perplexity_near_vocab_size(self) -> None:
        """An untrained model on random tokens should land near ppl == vocab_size."""
        torch.manual_seed(0)
        config = GPTConfig(
            vocab_size=self.VOCAB_SIZE,
            context_length=self.CONTEXT_LENGTH,
            n_embd=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        model = GPT(config)
        model.eval()
        result = evaluate_perplexity(
            model,
            str(self.bin_path),
            self.CONTEXT_LENGTH,
            batch_size=8,
            device=torch.device("cpu"),
        )
        self.assertGreater(result.perplexity, self.VOCAB_SIZE / 2)
        self.assertLess(result.perplexity, self.VOCAB_SIZE * 2)

    def test_max_batches_caps_forward_passes(self) -> None:
        """max_batches must cap the number of scored batches."""
        model = UniformLogitModel(self.VOCAB_SIZE)
        result = evaluate_perplexity(
            model,
            str(self.bin_path),
            self.CONTEXT_LENGTH,
            batch_size=4,
            max_batches=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.num_batches, 2)
        self.assertEqual(result.num_tokens, 2 * 4 * self.CONTEXT_LENGTH)

    def test_file_shorter_than_context_window(self) -> None:
        """A split smaller than one full window is scored as a single partial window."""
        short_path = self.tmp_dir / "short.bin"
        self.tokens[:10].tofile(short_path)
        model = UniformLogitModel(self.VOCAB_SIZE)
        result = evaluate_perplexity(
            model,
            str(short_path),
            self.CONTEXT_LENGTH,
            batch_size=8,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.num_tokens, 9)
        self.assertEqual(result.num_batches, 1)
        self.assertAlmostEqual(result.perplexity, float(self.VOCAB_SIZE), places=3)


class TestGetPrompts(unittest.TestCase):
    """Tests for prompt suite resolution."""

    def test_existing_prompts_file_is_used(self) -> None:
        """An existing prompts file overrides the built-in suite."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts_path = Path(tmp_dir) / "prompts.txt"
            prompts_path.write_text("first prompt\n# comment\nsecond prompt\n", encoding="utf-8")
            prompts, resolved = get_prompts(str(prompts_path), "generic")
        self.assertEqual(prompts, ["first prompt", "second prompt"])
        self.assertEqual(resolved, f"file:{prompts_path}")

    def test_missing_prompts_file_warns_and_falls_back(self) -> None:
        """A missing prompts file must warn on stderr and fall back to the suite."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            prompts, resolved = get_prompts("does_not_exist.txt", "generic")
        self.assertEqual(prompts, PROMPT_SUITES["generic"])
        self.assertEqual(resolved, "generic")
        self.assertIn("WARNING", stderr.getvalue())
        self.assertIn("does_not_exist.txt", stderr.getvalue())

    def test_no_prompts_file_is_silent(self) -> None:
        """Without a prompts file the suite is used with no warning."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            prompts, resolved = get_prompts(None, "research")
        self.assertEqual(prompts, PROMPT_SUITES["research"])
        self.assertEqual(resolved, "research")
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
