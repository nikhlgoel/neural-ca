"""Environment smoke tests: fail fast if the toolchain itself is broken."""

import torch

import neural_ca


def test_package_importable() -> None:
    assert neural_ca.__version__


def test_torch_basic_ops() -> None:
    x = torch.randn(4, 8)
    y = x @ x.T  # (4, 4)
    assert y.shape == (4, 4)
    assert torch.isfinite(y).all()


def test_torch_autograd() -> None:
    x = torch.ones(3, requires_grad=True)
    (x**2).sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, 2 * torch.ones(3))
