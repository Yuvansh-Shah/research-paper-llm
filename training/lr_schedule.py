"""Cosine learning-rate scheduler with linear warm-up.

This follows the schedule used in GPT-3 and most modern LLM training
runs: linearly ramp from 0 → ``max_lr`` over ``warmup_steps``, then
decay via a cosine curve down to ``min_lr`` over the remaining steps.
"""

from __future__ import annotations

import math


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
) -> float:
    """Compute the learning rate for the given training step.

    Args:
        step: Current optimiser step (0-indexed).
        warmup_steps: Number of linear warm-up steps.
        max_steps: Total number of training steps.
        max_lr: Peak learning rate reached at end of warm-up.
        min_lr: Minimum learning rate at end of cosine decay.
            Defaults to ``max_lr / 10``.

    Returns:
        Learning rate scalar for this step.
    """
    # 1. Linear warm-up phase.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2. After full cosine decay — clamp to min_lr.
    if step >= max_steps:
        return min_lr

    # 3. Cosine decay from max_lr → min_lr.
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_decay * (max_lr - min_lr)


class CosineScheduler:
    """Stateful wrapper around :func:`get_lr` for use with an optimiser.

    Args:
        optimizer: A PyTorch optimiser whose ``lr`` will be updated.
        warmup_steps: Number of linear warm-up steps.
        max_steps: Total number of training steps.
        max_lr: Peak learning rate.
        min_lr: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float = 3e-4,
        min_lr: float = 3e-5,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self._step = 0

    @property
    def last_lr(self) -> float:
        """Most recently set learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def step(self) -> float:
        """Advance the scheduler by one step and update the optimiser LR.

        Returns:
            The learning rate applied at this step.
        """
        lr = get_lr(
            self._step,
            self.warmup_steps,
            self.max_steps,
            self.max_lr,
            self.min_lr,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self._step += 1
        return lr

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["step"]
