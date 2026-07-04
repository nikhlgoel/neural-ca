"""Sample pool for persistence training, and circular damage for regeneration.

See docs/DESIGN.md §5. The pool lives on CPU to keep VRAM free; batches are moved to the
training device by the loop.
"""

import torch


class SamplePool:
    """A fixed set of grid states to resume training from (the persistence trick)."""

    def __init__(self, seed_state: torch.Tensor, size: int) -> None:
        # seed_state: (channels, H, W)
        self.states = seed_state.unsqueeze(0).repeat(size, 1, 1, 1).clone()

    def __len__(self) -> int:
        return self.states.shape[0]

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self), (batch,))
        return idx, self.states[idx].clone()

    def commit(self, idx: torch.Tensor, states: torch.Tensor) -> None:
        self.states[idx] = states.detach().to(self.states.device, self.states.dtype)


def damage(states: torch.Tensor, radius: float = 0.3) -> torch.Tensor:
    """Zero a random circular region in each sample, forcing the model to regenerate it."""
    b, _, h, w = states.shape
    ys = torch.linspace(-1.0, 1.0, h, device=states.device).view(1, h, 1)
    xs = torch.linspace(-1.0, 1.0, w, device=states.device).view(1, 1, w)
    cy = torch.empty(b, 1, 1, device=states.device).uniform_(-0.5, 0.5)
    cx = torch.empty(b, 1, 1, device=states.device).uniform_(-0.5, 0.5)
    keep = ((ys - cy) ** 2 + (xs - cx) ** 2) > radius**2  # (b, h, w)
    return states * keep.unsqueeze(1).to(states.dtype)
