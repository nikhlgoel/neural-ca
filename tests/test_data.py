import pytest

from neural_ca.data.state import make_seed_state
from neural_ca.data.targets import make_target, pad_to_grid


def test_make_target_shape_and_range() -> None:
    target = make_target("heart", size=40)
    assert target.shape == (4, 40, 40)
    assert target.min() >= 0.0 and target.max() <= 1.0
    assert target[3].sum() > 0  # the heart covers some pixels


def test_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="unknown builtin target"):
        make_target("dragon", size=40)


def test_pad_to_grid() -> None:
    target = make_target("circle", size=40)
    assert pad_to_grid(target, pad=16).shape == (4, 72, 72)


def test_seed_state_has_one_live_cell() -> None:
    state = make_seed_state(batch=2, channels=16, size=9)
    assert state.shape == (2, 16, 9, 9)
    assert state[:, 3].sum() == 2.0  # exactly one alive cell per sample
    assert state[0, 3, 4, 4] == 1.0  # at the centre
