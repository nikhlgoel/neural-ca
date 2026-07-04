"""The Neural Cellular Automaton: a tiny per-cell update rule, iterated over a grid.

Written from scratch (see docs/DESIGN.md §3). The only learned parameters are the two 1x1
convolutions of the update MLP; the perception filters are fixed, which is what keeps this a
true cellular automaton rather than a generic conv net.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _perception_weight(channels: int) -> torch.Tensor:
    """Fixed depthwise filters (identity, Sobel-x, Sobel-y) → 3 outputs per channel."""
    ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    sobel_y = sobel_x.t()
    kernels = torch.stack([ident, sobel_x, sobel_y])  # (3, 3, 3)
    return kernels.unsqueeze(1).repeat(channels, 1, 1, 1)  # (3*channels, 1, 3, 3)


class NCA(nn.Module):
    """Growing NCA. ``forward(state, steps)`` iterates the update rule ``steps`` times."""

    def __init__(
        self,
        channels: int = 16,
        hidden: int = 128,
        fire_rate: float = 0.5,
        alive_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate
        self.alive_threshold = alive_threshold

        self.register_buffer("perception", _perception_weight(channels))
        self.w1 = nn.Conv2d(3 * channels, hidden, kernel_size=1)
        self.w2 = nn.Conv2d(hidden, channels, kernel_size=1)
        # Zero-init the output: an untrained cell proposes "no change", so the grid is stable
        # and training departs smoothly from a fixed point.
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def _alive(self, state: torch.Tensor) -> torch.Tensor:
        alpha = state[:, 3:4]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > self.alive_threshold

    def step(self, state: torch.Tensor) -> torch.Tensor:
        """One synchronous-ish update: perceive → think → nudge, with masking."""
        pre_alive = self._alive(state)
        perceived = F.conv2d(state, self.perception, padding=1, groups=self.channels)
        delta = self.w2(F.relu(self.w1(perceived)))
        # Stochastic firing: each cell updates only with probability fire_rate.
        fire = (torch.rand_like(state[:, :1]) <= self.fire_rate).to(state.dtype)
        state = state + delta * fire
        # A cell lives only if it and a neighbour were alive before and after the update.
        alive = (pre_alive & self._alive(state)).to(state.dtype)
        return state * alive

    def forward(
        self, state: torch.Tensor, steps: int, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        if cond is not None:  # Tier-B seam (docs/DESIGN.md §3)
            raise NotImplementedError("class-conditioning is a Tier-B feature, not yet implemented")
        for _ in range(steps):
            state = self.step(state)
        return state
