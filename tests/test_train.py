import math

import torch

from neural_ca.config import Config, DataConfig, ModelConfig, TrainConfig
from neural_ca.training.loop import train
from neural_ca.training.pool import SamplePool, damage


def test_pool_sample_and_commit() -> None:
    seed = torch.zeros(16, 12, 12)
    pool = SamplePool(seed, size=8)
    idx, batch = pool.sample(4)
    assert batch.shape == (4, 16, 12, 12)
    pool.commit(idx, torch.ones_like(batch))
    assert pool.states[idx[0]].sum() > 0


def test_damage_zeros_a_region() -> None:
    states = torch.ones(3, 16, 20, 20)
    out = damage(states, radius=0.5)
    assert out.shape == states.shape
    assert (out == 0).any()  # something was zeroed


def test_train_smoke_cpu_runs_and_is_finite() -> None:
    cfg = Config(
        seed=0,
        device="cpu",
        model=ModelConfig(channels=16, hidden=32),
        data=DataConfig(target="circle", size=16, pad=4),
        train=TrainConfig(
            steps=3, batch_size=4, pool_size=16, step_range=(4, 6), damage=True, precision="fp32"
        ),
    )
    result = train(cfg, log_every=1)
    assert result["history"]
    assert all(math.isfinite(loss) for _, loss in result["history"])
