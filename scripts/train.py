"""Train a Growing NCA from a config file.

Usage:
    uv run python scripts/train.py --config configs/grow_emoji.yaml
    uv run python scripts/train.py --config configs/grow_emoji.yaml --steps 300   # short run
"""

import argparse
from pathlib import Path

from neural_ca.config import load_config
from neural_ca.training.loop import train
from neural_ca.utils import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None, help="override train.steps (smoke runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"steps": args.steps})})

    device = resolve_device(cfg.device)
    print(f"device={device.type}  target={cfg.data.target}  steps={cfg.train.steps}")
    result = train(cfg, device=device)
    for it, loss in result["history"]:
        print(f"  step {it:>5}  loss {loss:.5f}")
    print("done")


if __name__ == "__main__":
    main()
