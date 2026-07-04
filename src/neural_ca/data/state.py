"""The seed state: a single living cell at the grid centre (see docs/DESIGN.md §4)."""

import torch


def make_seed_state(batch: int, channels: int, size: int) -> torch.Tensor:
    """A black grid with one cell (centre) whose alpha + hidden channels are set to 1."""
    state = torch.zeros(batch, channels, size, size)
    state[:, 3:, size // 2, size // 2] = 1.0
    return state
