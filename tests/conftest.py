from __future__ import annotations

import os
import sys
import tempfile
import threading

import multiprocessing as _mp

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

try:
    from multiprocessing import heap as _mp_heap
except ImportError:  # pragma: no cover
    _mp_heap = None


def _filter_dir_candidates() -> None:
    if _mp_heap is None or not hasattr(_mp_heap, "Arena"):
        return
    candidates = getattr(_mp_heap.Arena, "_dir_candidates", None)
    if not candidates:
        return
    usable: list[str] = []
    for path in candidates:
        try:
            fd, name = tempfile.mkstemp(dir=path)
        except OSError:
            continue
        else:
            os.close(fd)
            os.unlink(name)
            usable.append(path)
    if usable:
        _mp_heap.Arena._dir_candidates = usable
    else:
        _mp_heap.Arena._dir_candidates = []


_filter_dir_candidates()


_ORIGINAL_VALUE = _mp.Value


class _ThreadValue:
    __slots__ = ("value", "_lock")

    def __init__(self, initial: object, *, lock: bool = True) -> None:
        self.value = initial
        self._lock = threading.RLock() if lock else _NullLock()

    def get_lock(self) -> threading.RLock | "_NullLock":
        return self._lock


class _NullLock:
    def acquire(self) -> bool:  # noqa: D401 - mimic Lock API
        return True

    def release(self) -> None:
        return None

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *_) -> None:
        return None


def _safe_value(typecode_or_type: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
    lock = kwargs.pop("lock", True)
    try:
        return _ORIGINAL_VALUE(typecode_or_type, *args, lock=lock, **kwargs)
    except (PermissionError, OSError):
        os.environ.setdefault("OPTILB_FORCE_THREAD_POOL", "1")
        if args:
            initial = args[0]
        elif callable(typecode_or_type):  # e.g. ctypes.c_double
            initial = typecode_or_type()
        else:
            initial = 0
        return _ThreadValue(initial, lock=bool(lock))


_mp.Value = _safe_value  # type: ignore[assignment]
