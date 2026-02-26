from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pytest

import optilb.optimizers.mads as mads_module
from optilb import DesignSpace
from optilb.optimizers import EarlyStopper, MADSOptimizer


class _FakePoint:
    def __init__(self, coords: Sequence[float]) -> None:
        self._coords = np.asarray(coords, dtype=float)
        self._bbo: bytes | None = None

    def get_coord(self, idx: int) -> float:
        return float(self._coords[idx])

    def size(self) -> int:
        return int(self._coords.size)

    def setBBO(self, payload: bytes) -> None:
        self._bbo = payload

    @property
    def bbo(self) -> bytes:
        if self._bbo is None:
            raise AssertionError("BBO payload was not set")
        return self._bbo


def _make_fake_pynomad(points: Sequence[np.ndarray]) -> type:
    class _FakePyNomad:
        @staticmethod
        def setSeed(seed: int) -> None:
            _ = seed

        @staticmethod
        def optimize(
            bb: Callable[[Any], int],
            x0: list[float],
            lower: list[float],
            upper: list[float],
            params: Sequence[str],
        ) -> dict[str, Any]:
            _ = (lower, upper, params)
            best_x = np.asarray(x0, dtype=float).copy()
            best_f = float("inf")
            for coords in points:
                point = _FakePoint(coords)
                bb(point)
                value = float(point.bbo.decode("utf-8").split()[0])
                if value < best_f:
                    best_f = value
                    best_x = np.asarray(coords, dtype=float).copy()
            return {"x_best": best_x.tolist(), "f_best": best_f}

    return _FakePyNomad


def test_mads_early_stopper_patience_with_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_points = [np.array([0.2]), np.array([0.2]), np.array([0.2])]
    monkeypatch.setattr(mads_module, "PyNomad", _make_fake_pynomad(fake_points))

    ds = DesignSpace(lower=np.array([-1.0]), upper=np.array([1.0]))
    stopper = EarlyStopper(eps=0.0, patience=1)
    opt = MADSOptimizer()
    res = opt.optimize(
        lambda _: 1.0,
        np.array([0.2]),
        ds,
        max_iter=20,
        early_stopper=stopper,
    )

    assert stopper.stopped is True
    assert res.nfev == 2
    assert len(res.evaluations) == 2
    best_record = min(res.evaluations, key=lambda record: record.value)
    assert res.best_f == pytest.approx(best_record.value, abs=1e-12)
    np.testing.assert_allclose(res.best_x, best_record.x)


def test_mads_early_stopper_time_limit_with_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_points = [np.array([0.25]), np.array([-0.5])]
    monkeypatch.setattr(mads_module, "PyNomad", _make_fake_pynomad(fake_points))

    ds = DesignSpace(lower=np.array([-1.0]), upper=np.array([1.0]))
    stopper = EarlyStopper(time_limit=0.0)
    opt = MADSOptimizer()
    res = opt.optimize(
        lambda x: float(np.sum(x**2)),
        np.array([0.25]),
        ds,
        max_iter=20,
        early_stopper=stopper,
    )

    assert stopper.stopped is True
    assert res.nfev == 1
    assert len(res.evaluations) == 1
    best_record = min(res.evaluations, key=lambda record: record.value)
    assert res.best_f == pytest.approx(best_record.value, abs=1e-12)
    np.testing.assert_allclose(res.best_x, best_record.x)
