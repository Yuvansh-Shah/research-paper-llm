"""Tests for text generation functionality."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from model.config import GPTConfig
from model.gpt import GPT
from inference.generate import (
    generate,
    parse_args,
    stream_generate,
    _top_k_filter,
    _top_p_filter,
)


class ScriptedModel(nn.Module):
    """Model rigged to emit a fixed token script, then EOS forever."""

    def __init__(self, config: GPTConfig, script: list[int], eos_token_id: int) -> None:
        super().__init__()
        self.config = config
        self.script = script
        self.eos_token_id = eos_token_id
        self.calls = 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None):
        if self.calls < len(self.script):
            token = self.script[self.calls]
        else:
            token = self.eos_token_id
        self.calls += 1
        logits = torch.full((idx.shape[0], idx.shape[1], self.config.vocab_size), -100.0)
        logits[:, -1, token] = 100.0
        return logits, None


class MockTokenizer:
    """Minimal tokenizer where token ``i`` decodes to ``<i>``."""

    eos_token_id = 0

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        token_ids = [1, 2, 3]
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids if token_id != self.eos_token_id]
        return "".join(f"<{token_id}>" for token_id in token_ids)


class TestSamplingFilters(unittest.TestCase):
    """Tests for sampling helper functions."""

    def test_top_k_filter(self) -> None:
        """Test top-k filtering keeps only top k values."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (1, 5)
        filtered = _top_k_filter(logits.clone(), k=2)

        # Only top 2 values (5.0, 4.0) should remain
        self.assertEqual(filtered[0, 4].item(), 5.0)  # Highest
        self.assertEqual(filtered[0, 3].item(), 4.0)  # Second highest
        self.assertEqual(filtered[0, 0].item(), float("-inf"))  # Filtered out

    def test_top_k_disabled_when_zero(self) -> None:
        """Test that k=0 disables filtering."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_k_filter(logits.clone(), k=0)
        self.assertTrue(torch.allclose(logits, filtered))

    def test_top_p_filter(self) -> None:
        """Test nucleus (top-p) sampling."""
        # Create logits where softmax gives clear probabilities
        logits = torch.tensor([[10.0, 5.0, 0.0, -5.0, -10.0]])  # (1, 5)
        filtered = _top_p_filter(logits.clone(), p=0.9)

        # Check that at least some values are kept and some filtered
        kept = (filtered[0] != float("-inf")).sum().item()
        self.assertGreater(kept, 0)
        self.assertLess(kept, 5)

    def test_top_p_disabled_when_one(self) -> None:
        """Test that p=1.0 disables filtering."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_p_filter(logits.clone(), p=1.0)
        self.assertTrue(torch.allclose(logits, filtered))


class TestGeneration(unittest.TestCase):
    """Tests for text generation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a small model for testing."""
        cls.config = GPTConfig(
            vocab_size=100,
            context_length=32,
            n_embd=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        cls.model = GPT(cls.config)
        cls.model.eval()

    def test_greedy_generation(self) -> None:
        """Test greedy decoding produces deterministic output."""
        torch.manual_seed(42)
        idx = torch.tensor([[1, 2, 3]])  # Seed tokens

        output1 = generate(self.model, idx.clone(), max_new_tokens=5, greedy=True)
        output2 = generate(self.model, idx.clone(), max_new_tokens=5, greedy=True)

        self.assertTrue(torch.equal(output1, output2))

    def test_generation_output_length(self) -> None:
        """Test that generation produces correct number of new tokens."""
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 10

        output = generate(self.model, idx, max_new_tokens=max_new_tokens, greedy=True)

        expected_length = idx.shape[1] + max_new_tokens
        self.assertEqual(output.shape[1], expected_length)

    def test_generation_with_temperature(self) -> None:
        """Test that very low temperature produces consistent output."""
        torch.manual_seed(42)
        idx = torch.tensor([[1, 2, 3]])

        # Very low temperature should produce nearly identical outputs
        # (effectively greedy sampling due to sharper distribution)
        torch.manual_seed(123)
        out1 = generate(self.model, idx.clone(), max_new_tokens=5, temperature=0.001)
        torch.manual_seed(123)
        out2 = generate(self.model, idx.clone(), max_new_tokens=5, temperature=0.001)

        # With same seed and very low temp, outputs should match
        self.assertTrue(torch.equal(out1, out2))

    def test_generation_respects_context_window(self) -> None:
        """Test that generation works with sequences at context limit."""
        # Create seed that fills context
        idx = torch.randint(0, self.config.vocab_size, (1, self.config.context_length - 1))

        # Should not raise even when seed is near context limit
        output = generate(self.model, idx, max_new_tokens=5, greedy=True)
        self.assertEqual(output.shape[1], idx.shape[1] + 5)


class TestEOSStopping(unittest.TestCase):
    """Tests for EOS early-stopping in generate and stream_generate."""

    def setUp(self) -> None:
        self.config = GPTConfig(
            vocab_size=100,
            context_length=32,
            n_embd=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        self.tokenizer = MockTokenizer()

    def test_generate_stops_at_eos(self) -> None:
        """generate must stop once EOS is produced and not emit the EOS token."""
        model = ScriptedModel(self.config, script=[5, 7], eos_token_id=0)
        idx = torch.tensor([[1, 2, 3]])

        output = generate(model, idx, max_new_tokens=10, greedy=True, eos_token_id=0)

        self.assertEqual(output[0].tolist(), [1, 2, 3, 5, 7])

    def test_generate_without_eos_runs_to_max_tokens(self) -> None:
        """Without eos_token_id, generation keeps emitting EOS tokens."""
        model = ScriptedModel(self.config, script=[5], eos_token_id=0)
        idx = torch.tensor([[1, 2, 3]])

        output = generate(model, idx, max_new_tokens=4, greedy=True)

        self.assertEqual(output[0].tolist(), [1, 2, 3, 5, 0, 0, 0])

    def test_stream_generate_stops_at_eos(self) -> None:
        """stream_generate must stop at EOS and never yield the EOS token."""
        model = ScriptedModel(self.config, script=[5, 7], eos_token_id=0)

        pieces = list(
            stream_generate(
                model,
                self.tokenizer,
                "prompt",
                max_new_tokens=10,
                greedy=True,
            )
        )

        self.assertEqual("".join(pieces), "<5><7>")

    def test_stream_generate_ignore_eos(self) -> None:
        """With stop_at_eos=False, EOS tokens are emitted like any other token."""
        model = ScriptedModel(self.config, script=[5], eos_token_id=0)

        pieces = list(
            stream_generate(
                model,
                self.tokenizer,
                "prompt",
                max_new_tokens=3,
                greedy=True,
                stop_at_eos=False,
            )
        )

        self.assertEqual("".join(pieces), "<5><0><0>")


class TestSeedReproducibility(unittest.TestCase):
    """Tests for seeded sampling as used by the --seed CLI flag."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = GPTConfig(
            vocab_size=100,
            context_length=32,
            n_embd=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        torch.manual_seed(0)
        cls.model = GPT(cls.config)
        cls.model.eval()

    def _sample(self, seed: int) -> list[int]:
        torch.manual_seed(seed)
        idx = torch.tensor([[1, 2, 3]])
        output = generate(self.model, idx, max_new_tokens=20, temperature=1.0)
        return output[0].tolist()

    def test_same_seed_produces_identical_output(self) -> None:
        """Two sampling runs with the same seed must match exactly."""
        self.assertEqual(self._sample(7), self._sample(7))

    def test_different_seeds_produce_different_output(self) -> None:
        """Sampling runs with different seeds must diverge."""
        self.assertNotEqual(self._sample(7), self._sample(8))

    def test_cli_accepts_seed_and_ignore_eos_flags(self) -> None:
        """parse_args must expose --seed and --ignore-eos."""
        args = parse_args(["--checkpoint", "dummy.pt", "--seed", "3", "--ignore-eos"])
        self.assertEqual(args.seed, 3)
        self.assertTrue(args.ignore_eos)

        defaults = parse_args(["--checkpoint", "dummy.pt"])
        self.assertIsNone(defaults.seed)
        self.assertFalse(defaults.ignore_eos)


class TestStreamGenerate(unittest.TestCase):
    """Tests for the streaming text generator."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = GPTConfig(
            vocab_size=100,
            context_length=32,
            n_embd=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        torch.manual_seed(0)
        cls.model = GPT(cls.config)
        cls.model.eval()
        cls.tokenizer = MockTokenizer()

    def test_stream_concatenation_matches_generate(self) -> None:
        """Concatenated stream pieces must equal the non-streaming completion."""
        torch.manual_seed(123)
        pieces = list(
            stream_generate(
                self.model,
                self.tokenizer,
                "prompt",
                max_new_tokens=15,
                temperature=0.9,
                top_k=10,
                top_p=0.95,
            )
        )

        torch.manual_seed(123)
        prompt_ids = self.tokenizer.encode("prompt", add_eos=False)
        idx = torch.tensor([prompt_ids], dtype=torch.long)
        output = generate(
            self.model,
            idx,
            max_new_tokens=15,
            temperature=0.9,
            top_k=10,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        completion_ids = output[0].tolist()[len(prompt_ids):]

        self.assertEqual("".join(pieces), self.tokenizer.decode(completion_ids))

    def test_stream_greedy_matches_generate(self) -> None:
        """Greedy streaming must reproduce greedy non-streaming output."""
        pieces = list(
            stream_generate(
                self.model,
                self.tokenizer,
                "prompt",
                max_new_tokens=10,
                greedy=True,
            )
        )

        prompt_ids = self.tokenizer.encode("prompt", add_eos=False)
        idx = torch.tensor([prompt_ids], dtype=torch.long)
        output = generate(
            self.model,
            idx,
            max_new_tokens=10,
            greedy=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        completion_ids = output[0].tolist()[len(prompt_ids):]

        self.assertEqual("".join(pieces), self.tokenizer.decode(completion_ids))


if __name__ == "__main__":
    unittest.main()
