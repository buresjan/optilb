from __future__ import annotations

import pickle
from datetime import datetime, timezone

import numpy as np
import pytest

from optilb import Constraint, DesignPoint, DesignSpace, OptResult


def test_designspace_equality_and_pickle() -> None:
    ds1 = DesignSpace(lower=[0, 0], upper=[1, 1], names=["a", "b"])
    ds2 = DesignSpace(
        lower=np.array([0, 0], dtype=float), upper=[1, 1], names=["a", "b"]
    )
    assert ds1 == ds2
    ds3 = pickle.loads(pickle.dumps(ds1))
    assert ds1 == ds3


def test_designspace_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="Lower bounds must not exceed upper bounds"):
        DesignSpace(lower=[1.0, 0.0], upper=[0.0, 1.0])


def test_designspace_names_are_immutable() -> None:
    names = ["x", "y"]
    ds = DesignSpace(lower=[0, 0], upper=[1, 1], names=names)
    names[0] = "z"
    assert ds.names == ("x", "y")


def test_designpoint_equality_and_pickle() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    p1 = DesignPoint(x=[0.5, 0.5], tag="foo", timestamp=ts)
    p2 = DesignPoint(x=np.array([0.5, 0.5]), tag="foo", timestamp=ts)
    assert p1 == p2
    p3 = pickle.loads(pickle.dumps(p1))
    assert p3 == p1


def test_optresult_equality_and_pickle() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pt = DesignPoint(x=[0.1], timestamp=ts)
    r1 = OptResult(best_x=[0.1], best_f=1.0, history=[pt])
    r2 = OptResult(best_x=np.array([0.1]), best_f=1.0, history=[pt])
    assert r1 == r2
    r3 = pickle.loads(pickle.dumps(r1))
    assert r3 == r1


def test_constraint_callable() -> None:
    c = Constraint(func=lambda x: np.sum(x) < 1.0)
    assert bool(c(np.array([0.2, 0.3]))) is True
    assert bool(c(np.array([1.0, 0.0]))) is False
