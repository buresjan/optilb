from __future__ import annotations

import numpy as np

from optilb import DesignSpace
from optilb.sampling import lhs


def test_lhs_reproducible() -> None:
    ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    pts1 = lhs(4, ds, seed=42)
    pts2 = lhs(4, ds, seed=42)
    arr1 = np.array([p.x for p in pts1])
    arr2 = np.array([p.x for p in pts2])
    np.testing.assert_allclose(arr1, arr2)


def test_lhs_with_integer_dimension() -> None:
    ds = DesignSpace(lower=[0, 0.0], upper=[10, 1.0])
    pts = lhs(6, ds, seed=123)
    arr = np.array([p.x for p in pts])
    assert np.all(arr[:, 0] == np.round(arr[:, 0]))
    assert np.all(arr[:, 0] >= 0) and np.all(arr[:, 0] <= 10)


def test_lhs_centered() -> None:
    ds = DesignSpace(lower=[0.0], upper=[1.5])
    pts = lhs(4, ds, centered=True, seed=0)
    arr = np.array([p.x for p in pts]).flatten()
    expected = np.array([0.1875, 0.5625, 0.9375, 1.3125])
    assert np.all(np.isin(arr, expected))
