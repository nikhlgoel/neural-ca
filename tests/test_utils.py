import torch

from neural_ca.utils import resolve_device, set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(0)
    a = torch.rand(5)
    set_seed(0)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_resolve_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}
