"""Export the deterministic core of one NCA step to ONNX for in-browser inference.

We export perception + the update MLP (which produces the per-cell delta). The stochastic fire
mask, the residual add, and alive-masking are cheap element-wise ops done in JavaScript, so the
ONNX graph stays free of the random ops that browser runtimes handle poorly (see ADR-0004).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from neural_ca.model.nca import NCA


class DeltaCore(nn.Module):
    """Wraps an NCA to output just the per-cell update delta from a state (no randomness)."""

    def __init__(self, nca: NCA) -> None:
        super().__init__()
        self.nca = nca

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        perceived = F.conv2d(state, self.nca.perception, padding=1, groups=self.nca.channels)
        return self.nca.w2(F.relu(self.nca.w1(perceived)))


def export_onnx(model: NCA, path: str | Path, size: int = 72) -> None:
    """Export ``DeltaCore(model)`` to ONNX with dynamic batch/height/width axes.

    Uses the torch.export-based (dynamo) exporter — the default since PyTorch 2.9.
    """
    from torch.export import Dim

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = DeltaCore(model).eval()
    dummy = torch.zeros(1, model.channels, size, size)
    torch.onnx.export(
        core,
        (dummy,),
        str(path),
        input_names=["state"],
        output_names=["delta"],
        opset_version=18,
        dynamo=True,
        verbose=False,
        dynamic_shapes={"state": {0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}},
    )
