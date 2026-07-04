from pathlib import Path

import pytest
from pydantic import ValidationError

from neural_ca.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_grow_emoji_config_loads() -> None:
    cfg = load_config(CONFIGS / "grow_emoji.yaml")
    assert isinstance(cfg, Config)
    assert cfg.model.channels > 4  # must leave room for hidden channels beyond RGBA
    assert cfg.train.precision in {"fp32", "bf16"}
    lo, hi = cfg.train.step_range
    assert 0 < lo <= hi


def test_unknown_key_is_rejected() -> None:
    # A typo'd key must fail loudly at load time, not run for hours with a wrong setting.
    with pytest.raises(ValidationError):
        Config.model_validate({"data": {"target": "heart"}, "bogus": 1})
