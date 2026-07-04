import pytest
import torch

from neural_ca.model.nca import NCA


def test_parameter_count() -> None:
    # Only the update MLP is learned: (48*128+128) + (128*16+16) = 8336.
    model = NCA(channels=16, hidden=128)
    assert sum(p.numel() for p in model.parameters()) == 8336


def test_forward_shape() -> None:
    model = NCA()
    state = torch.zeros(2, 16, 24, 24)
    state[:, 3:, 12, 12] = 1.0
    out = model(state, steps=8)
    assert out.shape == state.shape
    assert torch.isfinite(out).all()


def test_zero_init_is_stable() -> None:
    # With the output layer zero-initialised the update is 0, so the seed must persist exactly.
    model = NCA()
    state = torch.zeros(1, 16, 24, 24)
    state[:, 3:, 12, 12] = 1.0
    assert torch.allclose(model(state, steps=16), state)


def test_conditioning_is_rejected() -> None:
    model = NCA()
    state = torch.zeros(1, 16, 16, 16)
    state[:, 3:, 8, 8] = 1.0
    with pytest.raises(NotImplementedError):
        model(state, steps=1, cond=torch.zeros(1))
