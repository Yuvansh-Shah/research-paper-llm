"""Tests for training loop components (smoke test)."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model.config import GPTConfig
from model.gpt import GPT
from training.dataset import TokenDataset
from training.lr_schedule import get_lr, CosineScheduler


class TestTokenDataset(unittest.TestCase):
    """Tests for the training dataset."""

    def test_dataset_creation(self) -> None:
        """Test creating dataset from binary file."""
        tmpdir = tempfile.mkdtemp()
        bin_path = Path(tmpdir) / "test.bin"
        try:
            # Create test data: 1000 tokens
            tokens = np.arange(1000, dtype=np.uint16)
            tokens.tofile(bin_path)

            context_length = 32
            dataset = TokenDataset(str(bin_path), context_length)

            # Dataset length should be num_tokens - context_length
            self.assertEqual(len(dataset), 1000 - context_length)
            
            # Explicitly close the memmap
            del dataset.data
            del dataset
            gc.collect()
        finally:
            # On Windows, we may need to retry deletion
            import time
            for _ in range(3):
                try:
                    if bin_path.exists():
                        bin_path.unlink()
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.1)

    def test_dataset_getitem(self) -> None:
        """Test retrieving items from dataset."""
        tmpdir = tempfile.mkdtemp()
        bin_path = Path(tmpdir) / "test.bin"
        try:
            tokens = np.arange(100, dtype=np.uint16)
            tokens.tofile(bin_path)

            context_length = 10
            dataset = TokenDataset(str(bin_path), context_length)

            x, y = dataset[0]

            # x and y should be consecutive sequences
            self.assertEqual(x.shape, (context_length,))
            self.assertEqual(y.shape, (context_length,))
            # y should be x shifted by 1
            self.assertTrue(torch.equal(x[1:], y[:-1]))
            
            # Cleanup
            del dataset.data
            del dataset
            gc.collect()
        finally:
            import time
            for _ in range(3):
                try:
                    if bin_path.exists():
                        bin_path.unlink()
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.1)


class TestLRScheduler(unittest.TestCase):
    """Tests for learning rate scheduling."""

    def test_warmup_phase(self) -> None:
        """Test linear warmup at the start."""
        warmup_steps = 100
        max_lr = 1e-3

        # At step 0, lr should be close to 0 (actually max_lr * 1/warmup)
        lr_start = get_lr(0, warmup_steps=warmup_steps, max_steps=1000, max_lr=max_lr)
        self.assertLess(lr_start, max_lr)

        # At end of warmup, lr should be max_lr
        lr_end_warmup = get_lr(warmup_steps - 1, warmup_steps=warmup_steps, max_steps=1000, max_lr=max_lr)
        self.assertAlmostEqual(lr_end_warmup, max_lr, places=5)

    def test_cosine_decay(self) -> None:
        """Test cosine decay after warmup."""
        warmup_steps = 10
        max_steps = 100
        max_lr = 1e-3
        min_lr = 1e-4

        # LR should decrease after warmup
        lr_mid = get_lr(50, warmup_steps, max_steps, max_lr, min_lr)
        self.assertLess(lr_mid, max_lr)
        self.assertGreater(lr_mid, min_lr)

        # At max_steps, should be at min_lr
        lr_end = get_lr(max_steps, warmup_steps, max_steps, max_lr, min_lr)
        self.assertAlmostEqual(lr_end, min_lr, places=6)

    def test_scheduler_class(self) -> None:
        """Test CosineScheduler wrapper."""
        params = [torch.zeros(1, requires_grad=True)]
        optimizer = torch.optim.SGD(params, lr=0.1)
        scheduler = CosineScheduler(optimizer, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)

        # Step and check LR is updated
        lr1 = scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], lr1)

        lr2 = scheduler.step()
        self.assertNotEqual(lr1, lr2)  # LR should change


class TestTrainingSmokeTest(unittest.TestCase):
    """Smoke test for the training pipeline."""

    def test_mini_training_loop(self) -> None:
        """Test a minimal training loop shows loss decrease."""
        # Create tiny model
        config = GPTConfig(
            vocab_size=100,
            context_length=16,
            n_embd=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        model = GPT(config)
        model.train()

        # Create synthetic dataset (repeating pattern should be learnable)
        tmpdir = tempfile.mkdtemp()
        bin_path = Path(tmpdir) / "train.bin"
        try:
            # Repeating pattern: 0,1,2,3,4,5,6,7,8,9,0,1,2,3...
            pattern = np.tile(np.arange(10, dtype=np.uint16), 100)
            pattern.tofile(bin_path)

            dataset = TokenDataset(str(bin_path), config.context_length)
            loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Track losses
            losses = []
            for epoch in range(3):
                epoch_loss = 0.0
                n_batches = 0
                for x, y in loader:
                    optimizer.zero_grad()
                    _, loss = model(x, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                    if n_batches >= 10:
                        break
                losses.append(epoch_loss / n_batches)

            # Loss should decrease
            self.assertLess(
                losses[-1], losses[0],
                f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
            )
            
            # Cleanup
            del dataset.data
            del dataset
            del loader
            gc.collect()
        finally:
            import time
            for _ in range(3):
                try:
                    if bin_path.exists():
                        bin_path.unlink()
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
