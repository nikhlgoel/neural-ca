"""Procedural RGBA targets (no external assets) plus image loading.

A target is a `(4, H, W)` tensor in `[0, 1]`: RGB then alpha. Procedural targets keep the repo
license-clean (see docs/DESIGN.md §4); a file path may also be used.
"""

import torch
import torch.nn.functional as F


def make_target(name: str, size: int = 40) -> torch.Tensor:
    """Render a builtin target (``heart`` | ``circle`` | ``square``) as ``(4, size, size)``."""
    coords = torch.linspace(-1.0, 1.0, size)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    if name == "circle":
        mask = (x**2 + y**2) <= 0.7**2
        rgb = (0.90, 0.20, 0.20)
    elif name == "square":
        mask = (x.abs() <= 0.6) & (y.abs() <= 0.6)
        rgb = (0.20, 0.45, 0.90)
    elif name == "heart":
        # Classic implicit heart curve; the 1.4 factor shrinks it to fit the grid.
        xh, yh = x * 1.4, -y * 1.4
        mask = ((xh**2 + yh**2 - 1.0) ** 3 - xh**2 * yh**3) <= 0.0
        rgb = (0.85, 0.15, 0.30)
    else:
        raise ValueError(f"unknown builtin target: {name!r} (try heart, circle, square)")
    alpha = mask.to(torch.float32)
    color = torch.tensor(rgb).view(3, 1, 1) * alpha
    return torch.cat([color, alpha.unsqueeze(0)], dim=0)


def pad_to_grid(target: torch.Tensor, pad: int) -> torch.Tensor:
    """Zero-pad a target so the pattern has room to grow into a larger grid."""
    return F.pad(target, (pad, pad, pad, pad))


def load_target(path: str, size: int) -> torch.Tensor:
    """Load an RGBA image file, resized to ``(4, size, size)`` in ``[0, 1]``."""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGBA").resize((size, size))
    arr = np.asarray(img, dtype="float32") / 255.0  # (H, W, 4)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
