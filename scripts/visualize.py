"""Train an NCA briefly and render its growth: a still, a GIF, and a 3D surface.

Usage:
    uv run python scripts/visualize.py --config configs/grow_emoji.yaml --steps 1000
"""

import argparse
from pathlib import Path

import torch

from neural_ca.config import load_config
from neural_ca.data.state import make_seed_state
from neural_ca.eval.render import grow_frames, save_gif, save_png, save_surface
from neural_ca.training.loop import train
from neural_ca.utils import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"steps": args.steps})})
    device = resolve_device(cfg.device)
    print(f"device={device.type}  training {args.steps} steps on '{cfg.data.target}' ...")

    result = train(cfg, device=device)
    model, target = result["model"], result["target"]
    grid = target.shape[-1]
    seed = make_seed_state(1, cfg.model.channels, grid).to(device)

    grow_steps = cfg.train.step_range[1] * 2
    frames = grow_frames(model, seed, steps=grow_steps, every=4)
    save_png(frames[-1], args.out / "heart.png")
    save_gif(frames, args.out / "heart_growth.gif")
    with torch.no_grad():
        final_state = model(seed, steps=grow_steps)
    save_surface(final_state, args.out / "heart_surface.png")

    print(f"final training loss {result['history'][-1][1]:.5f}")
    print(f"wrote {args.out}/heart.png, heart_growth.gif, heart_surface.png")


if __name__ == "__main__":
    main()
