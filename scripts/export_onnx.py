"""Export a trained NCA checkpoint to ONNX and verify it matches PyTorch.

Usage:
    uv run python scripts/export_onnx.py --checkpoint checkpoints/heart.pt --out models/heart.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

from neural_ca.eval.export import DeltaCore, export_onnx
from neural_ca.training.checkpoint import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/heart.onnx"))
    ap.add_argument("--size", type=int, default=72)
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint)
    export_onnx(model, args.out, size=args.size)

    state = torch.randn(1, model.channels, args.size, args.size)
    with torch.no_grad():
        torch_delta = DeltaCore(model).eval()(state).numpy()
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    onnx_delta = sess.run(["delta"], {"state": state.numpy()})[0]

    max_diff = float(np.abs(torch_delta - onnx_delta).max())
    print(f"exported {args.out}  (max |torch - onnx| = {max_diff:.2e})")
    if max_diff >= 1e-4:
        raise SystemExit("ONNX output diverges from PyTorch")
    print("verified OK - ready for onnxruntime-web")


if __name__ == "__main__":
    main()
