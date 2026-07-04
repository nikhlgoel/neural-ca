import torch

from neural_ca.eval.render import to_rgb


def test_to_rgb_shape_and_dtype() -> None:
    state = torch.zeros(16, 12, 12)
    state[3] = 1.0  # fully alive → opaque
    arr = to_rgb(state)
    assert arr.shape == (12, 12, 3)
    assert arr.dtype.name == "uint8"


def test_to_rgb_dead_cells_are_white() -> None:
    state = torch.zeros(16, 8, 8)  # nothing alive
    arr = to_rgb(state)
    assert (arr == 255).all()  # composites onto a white background
