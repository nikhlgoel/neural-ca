"""Save and load NCA checkpoints (weights + the model config that built them)."""

from pathlib import Path

import torch

from neural_ca.config import Config, ModelConfig
from neural_ca.model.nca import NCA


def save_checkpoint(model: NCA, cfg: Config, path: str | Path) -> None:
    """Write model weights plus the ModelConfig needed to rebuild the architecture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": cfg.model.model_dump()}, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> NCA:
    """Rebuild an NCA from a checkpoint and load its weights."""
    blob = torch.load(path, map_location=device or "cpu", weights_only=True)
    mc = ModelConfig(**blob["model_cfg"])
    model = NCA(mc.channels, mc.hidden, mc.fire_rate, mc.alive_threshold)
    model.load_state_dict(blob["model"])
    return model.to(device) if device is not None else model
