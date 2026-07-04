"""Render NCA states to images: RGBA stills, growth GIFs, and 3D surfaces (docs/DESIGN.md §6)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("Agg")  # headless: render straight to files, no display needed


def to_rgb(state: torch.Tensor) -> np.ndarray:
    """Composite an NCA state ``(C,H,W)`` or ``(1,C,H,W)`` onto white → ``uint8 (H,W,3)``."""
    if state.dim() == 4:
        state = state[0]
    rgb = state[:3].clamp(0, 1)
    alpha = state[3:4].clamp(0, 1)
    img = rgb * alpha + (1.0 - alpha)  # alpha-composite over a white background
    return (img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")


@torch.no_grad()
def grow_frames(model, seed: torch.Tensor, steps: int, every: int = 4) -> list[np.ndarray]:
    """Run the model from ``seed``, capturing an RGB frame every ``every`` steps."""
    state = seed.clone()
    frames = [to_rgb(state)]
    for i in range(1, steps + 1):
        state = model.step(state)
        if i % every == 0:
            frames.append(to_rgb(state))
    return frames


def save_png(arr: np.ndarray, path: str | Path) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 25) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def save_surface(state: torch.Tensor, path: str | Path) -> None:
    """Render the alpha field as a 3D surface — the model's sense of *where the shape is*."""
    if state.dim() == 4:
        state = state[0]
    z = state[3].detach().clamp(0, 1).cpu().numpy()  # alpha as height
    h, w = z.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(x, y, z, cmap="magma", linewidth=0, antialiased=True)
    ax.set_zlim(0, 1)
    ax.set_title("NCA alpha surface (height = 'aliveness')")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
