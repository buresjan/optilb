from __future__ import annotations

import numpy as np

from optilb.objectives import (
    get_objective,
    make_noisy_discontinuous,
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


def test_get_objective_dispatch() -> None:
    f = get_objective("quadratic")
    assert f(np.array([0.0])) == 0.0
    f2 = get_objective("noisy_discontinuous", sigma=0.0)
    assert f2(np.array([0.3])) == 0.0
