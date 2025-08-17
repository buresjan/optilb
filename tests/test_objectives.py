from __future__ import annotations

import numpy as np
import pytest

from optilb import UnknownObjectiveError
from optilb.objectives import (
    get_objective,
    lbm_stub,
    make_checkerboard,
    make_noisy_discontinuous,
    make_spiky_sine,
    make_step_rastrigin,
    plateau_cliff,
    quadratic_bowl,
    rastrigin,
)


def test_quadratic_bowl_minimum() -> None:
    assert quadratic_bowl(np.zeros(3)) == 0.0


def test_rastrigin_minimum() -> None:
    assert rastrigin(np.zeros(4)) == 0.0


def test_noisy_discontinuous_zero_noise() -> None:
    f = make_noisy_discontinuous(sigma=0.0)
    val = f(np.array([0.2, 0.8]))
    assert val == 0.0


def test_plateau_cliff() -> None:
    assert plateau_cliff(np.array([1.5])) == 0.0
    assert plateau_cliff(np.array([-0.5])) == 1.0


def test_lbm_stub_deterministic() -> None:
    val1 = lbm_stub(np.array([0.1, -0.2]))
    val2 = lbm_stub(np.array([0.1, -0.2]))
    assert val1 == val2


def test_spiky_sine_repeatable() -> None:
    f = make_spiky_sine(sigma=0.0)
    assert f(np.array([0.0])) == f(np.array([0.0]))


def test_checkerboard_pattern() -> None:
    f = make_checkerboard(sigma=0.0)
    assert f(np.array([0.0, 0.0])) == 0.0


def test_step_rastrigin_floor() -> None:
    f = make_step_rastrigin(sigma=0.0)
    x = np.array([1.7, -2.3])
    expected = rastrigin(np.floor(x))
    assert f(x) == expected


def test_get_objective_dispatch() -> None:
    f = get_objective("quadratic")
    assert f(np.array([0.0])) == 0.0
    f2 = get_objective("noisy_discontinuous", sigma=0.0)
    assert f2(np.array([0.3])) == 0.0
    f3 = get_objective("lbm_stub")
    assert f3(np.array([0.1, 0.2])) == lbm_stub(np.array([0.1, 0.2]))
    assert callable(get_objective("spiky_sine"))
    assert callable(get_objective("checkerboard"))
    assert callable(get_objective("step_rastrigin"))


def test_get_objective_unknown() -> None:
    with pytest.raises(UnknownObjectiveError):
        get_objective("does-not-exist")
