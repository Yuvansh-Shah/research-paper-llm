"""Tests for model architecture and checkpoint save/load."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from model.config import GPTConfig
from model.gpt import GPT


class TestGPTConfig(unittest.TestCase):
    """Tests for GPTConfig dataclass."""

    def test_small_config(self) -> None:
        config = GPTConfig.small()
        self.assertEqual(config.vocab_size, 50_257)
        self.assertEqual(config.context_length, 256)
        self.assertEqual(config.n_embd, 512)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.n_layers, 6)

    def test_base_config(self) -> None:
        """Test base config preset (~124M params)."""
        config = GPTConfig.base()
        self.assertEqual(config.n_embd, 768)
        self.assertEqual(config.n_heads, 12)
        self.assertEqual(config.n_layers, 12)
        self.assertEqual(config.context_length, 1024)

    def test_from_name(self) -> None:
        """Test creating config from name string."""
        for name in ["debug", "small", "base", "large", "xl"]:
            config = GPTConfig.from_name(name)
            self.assertIsInstance(config, GPTConfig)
        
        # Legacy alias
        config = GPTConfig.from_name("medium")
        self.assertEqual(config.n_embd, 768)  # Same as base

    def test_invalid_config_raises(self) -> None:
        with self.assertRaises(ValueError):
            # n_embd must be divisible by n_heads
            GPTConfig(n_embd=512, n_heads=7)


class TestGPTModel(unittest.TestCase):
    """Tests for GPT model forward pass and basic operations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a small config for fast testing."""
        cls.config = GPTConfig(
            vocab_size=1000,  # Small vocab for testing
            context_length=32,
            n_embd=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,  # No dropout for deterministic tests
        )
        cls.device = torch.device("cpu")

    def test_model_creation(self) -> None:
        model = GPT(self.config)
        self.assertIsInstance(model, GPT)
        self.assertGreater(model.num_parameters(), 0)

    def test_layernorm_weight_trainable_without_bias(self) -> None:
        """LayerNorm keeps a learnable scale even when bias=False."""
        self.assertFalse(self.config.bias)
        model = GPT(self.config)

        norms = [model.transformer["ln_f"]]
        for block in model.transformer["h"]:
            norms.extend([block.ln_1, block.ln_2])

        for ln in norms:
            self.assertIsNotNone(ln.weight)
            self.assertTrue(ln.weight.requires_grad)
            self.assertTrue(torch.equal(ln.weight, torch.ones_like(ln.weight)))
            self.assertIsNone(ln.bias)

    def test_param_count_includes_layernorm_weights(self) -> None:
        """Each block contributes 2 LayerNorm scales, plus the final ln_f."""
        model = GPT(self.config)
        expected_ln_params = (2 * self.config.n_layers + 1) * self.config.n_embd
        ln_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if "ln_" in name
        )
        self.assertEqual(ln_params, expected_ln_params)

    def test_forward_pass_inference(self) -> None:
        """Test forward pass without targets (inference mode)."""
        model = GPT(self.config).to(self.device)
        model.eval()

        batch_size, seq_len = 2, 16
        idx = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, loss = model(idx)

        # In inference mode, only last position logits are returned
        self.assertEqual(logits.shape, (batch_size, 1, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_pass_training(self) -> None:
        """Test forward pass with targets (training mode)."""
        model = GPT(self.config).to(self.device)
        model.train()

        batch_size, seq_len = 2, 16
        idx = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        logits, loss = model(idx, targets)

        # In training mode, logits for all positions are returned
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, ())  # Scalar loss

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        model = GPT(self.config).to(self.device)
        model.train()

        idx = torch.randint(0, self.config.vocab_size, (2, 8))
        targets = torch.randint(0, self.config.vocab_size, (2, 8))

        _, loss = model(idx, targets)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad)


class TestCheckpointing(unittest.TestCase):
    """Tests for checkpoint save and load functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = GPTConfig(
            vocab_size=1000,
            context_length=32,
            n_embd=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )

    def test_save_and_load_checkpoint(self) -> None:
        """Test that model can be saved and loaded correctly."""
        model = GPT(self.config)

        # Get original state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            model.save_checkpoint(str(ckpt_path), step=100)

            # Load into new model
            loaded_model = GPT.from_checkpoint(str(ckpt_path))

            # Verify configs match
            self.assertEqual(loaded_model.config.vocab_size, self.config.vocab_size)
            self.assertEqual(loaded_model.config.n_embd, self.config.n_embd)
            self.assertEqual(loaded_model.config.n_layers, self.config.n_layers)

            # Verify weights match
            for key in original_state:
                self.assertTrue(
                    torch.allclose(original_state[key], loaded_model.state_dict()[key]),
                    f"Weight mismatch for {key}",
                )

    def test_checkpoint_contains_step(self) -> None:
        """Test that checkpoint contains the training step."""
        model = GPT(self.config)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            model.save_checkpoint(str(ckpt_path), step=42)

            payload = torch.load(ckpt_path, weights_only=True)
            self.assertEqual(payload["step"], 42)
            self.assertIsInstance(payload["config"], dict)

    def test_checkpoint_with_optimizer_state(self) -> None:
        """Test saving checkpoint with optimizer state."""
        model = GPT(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Do a dummy backward pass to create optimizer state
        idx = torch.randint(0, self.config.vocab_size, (1, 8))
        targets = torch.randint(0, self.config.vocab_size, (1, 8))
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            model.save_checkpoint(str(ckpt_path), step=1, optimizer_state=optimizer.state_dict())

            payload = torch.load(ckpt_path, weights_only=True)
            self.assertIn("optimizer_state_dict", payload)


if __name__ == "__main__":
    unittest.main()
