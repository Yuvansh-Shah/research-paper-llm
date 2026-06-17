"""Tests for text generation functionality."""

from __future__ import annotations

import unittest

import torch

from model.config import GPTConfig
from model.gpt import GPT
from inference.generate import generate, _top_k_filter, _top_p_filter


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


if __name__ == "__main__":
    unittest.main()
