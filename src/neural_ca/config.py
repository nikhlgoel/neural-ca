"""Typed experiment configuration.

A run is fully described by one YAML file, validated against these schemas at load time, so a
typo fails immediately instead of silently during a long run (see docs/DESIGN.md §2 and §8).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channels: int = Field(16, ge=5, description="total state channels; first 4 are RGBA")
    hidden: int = Field(128, ge=1, description="hidden width of the update MLP")
    fire_rate: float = Field(0.5, gt=0, le=1, description="probability a cell updates each step")
    alive_threshold: float = Field(0.1, ge=0, le=1)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(..., description="builtin target name, or a path to an RGBA image")
    size: int = Field(40, ge=8, le=128, description="target height and width in cells")
    pad: int = Field(16, ge=0, description="zero-padding around the target, so it can grow")


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = Field(8000, ge=1)
    batch_size: int = Field(8, ge=1)
    lr: float = Field(2e-3, gt=0)
    step_range: tuple[int, int] = Field((64, 96), description="min/max NCA steps per forward")
    pool_size: int = Field(1024, ge=1)
    damage: bool = True
    precision: str = Field("bf16", pattern="^(fp32|bf16)$")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    seed: int = 42
    device: str = Field("auto", pattern="^(auto|cpu|cuda)$")
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML experiment config into a typed ``Config``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(raw)
