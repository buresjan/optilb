from __future__ import annotations

import numpy as np
import pytest

from optilb.core import DesignSpace
from optilb.optimizers.utils import SpaceTransform


def test_round_trip() -> None:
    rng = np.random.default_rng(0)
    lower = rng.uniform(-5.0, 0.0, size=3)
    upper = lower + rng.uniform(1.0, 5.0, size=3)
    space = DesignSpace(lower=lower, upper=upper)
    tf = SpaceTransform(space)
    x = rng.uniform(lower, upper)
    np.testing.assert_allclose(tf.from_unit(tf.to_unit(x)), x, atol=1e-15)


def test_invalid_bounds() -> None:
    space_inf = DesignSpace(lower=np.array([0.0, 0.0]), upper=np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        SpaceTransform(space_inf)
    space_deg = DesignSpace(lower=np.array([0.0, 0.0]), upper=np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        SpaceTransform(space_deg)


def test_identity_unit_space() -> None:
    space = DesignSpace(lower=np.zeros(2), upper=np.ones(2))
    tf = SpaceTransform(space)
    x = np.array([0.3, 0.7])
    np.testing.assert_array_equal(tf.to_unit(x), x)
    np.testing.assert_array_equal(tf.from_unit(x), x)
