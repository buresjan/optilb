from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from optilb import DesignSpace
from optilb.sampling import lhs


def test_lhs_reproducible() -> None:
    ds = DesignSpace(lower=np.array([0.0, 0.0]), upper=np.array([1.0, 1.0]))
    pts1 = lhs(4, ds, seed=42)
    pts2 = lhs(4, ds, seed=42)
    arr1 = np.array([p.x for p in pts1])
    arr2 = np.array([p.x for p in pts2])
    np.testing.assert_allclose(arr1, arr2)


def test_lhs_with_integer_dimension() -> None:
    ds = DesignSpace(lower=np.array([0, 0.0]), upper=np.array([10, 1.0]))
    pts = lhs(6, ds, seed=123)
    arr = np.array([p.x for p in pts])
    assert np.all(arr[:, 0] == np.round(arr[:, 0]))
    assert np.all(arr[:, 0] >= 0) and np.all(arr[:, 0] <= 10)


def test_lhs_centered() -> None:
    ds = DesignSpace(lower=np.array([0.0]), upper=np.array([1.5]))
    pts = lhs(4, ds, centered=True, seed=0)
    arr = np.array([p.x for p in pts]).flatten()
    expected = np.array([0.1875, 0.5625, 0.9375, 1.3125])
    assert np.all(np.isin(arr, expected))


def test_lhs_rounding_matches_previous() -> None:
    ds = DesignSpace(lower=np.array([0, 0.0]), upper=np.array([10, 1.0]))
    sample_count = 6
    rng = np.random.default_rng(123)
    sampler = qmc.LatinHypercube(d=ds.dimension, scramble=True, rng=rng)
    sample = sampler.random(n=sample_count)
    scaled = qmc.scale(sample, ds.lower, ds.upper)
    rounded = scaled.copy()
    for i, (lo, hi) in enumerate(zip(ds.lower, ds.upper)):
        if float(lo).is_integer() and float(hi).is_integer():
            rounded[:, i] = np.rint(rounded[:, i])

    arr_expected = rounded
    pts = lhs(sample_count, ds, seed=123)
    arr = np.array([p.x for p in pts])
    np.testing.assert_allclose(arr, arr_expected)
