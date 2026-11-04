"""
Thread-safe, LRU-cached pool of read-only HDF5 handles.

h5py is not thread-safe (single global HDF5 lock unless built thread-safe), so a
module lock serializes every h5py call. Source files are opened read-only and
never written concurrently — editing always materializes a NEW file — so the
only contention is read parallelism, which we bound with a small threadpool at
the FastAPI layer. Handles are keyed by (path, mtime) so a re-collected /
overwritten file is reopened automatically.
"""
import os
import threading
from collections import OrderedDict

import h5py

from . import config as CFG

_lock = threading.RLock()
_cache: "OrderedDict[str, tuple]" = OrderedDict()   # path -> (mtime, h5py.File)


def _evict_if_needed():
    while len(_cache) > CFG.H5_LRU:
        _, (_, f) = _cache.popitem(last=False)
        try:
            f.close()
        except Exception:
            pass


def get(path):
    """Return a cached read-only h5py.File for `path`, reopening on mtime change.

    Callers MUST hold the pool lock for the duration of any access to the
    returned handle — use the `reading(path)` context manager instead of calling
    this directly.
    """
    mtime = os.path.getmtime(path)
    with _lock:
        ent = _cache.get(path)
        if ent is not None and ent[0] == mtime:
            _cache.move_to_end(path)
            return ent[1]
        if ent is not None:
            try:
                ent[1].close()
            except Exception:
                pass
        f = h5py.File(path, "r")
        _cache[path] = (mtime, f)
        _cache.move_to_end(path)
        _evict_if_needed()
        return f


class reading:
    """Context manager: holds the global HDF5 lock while you touch a handle.

        with reading(path) as f:
            arr = f["obs/joint_pos"][a:b]
    """

    def __init__(self, path):
        self.path = path
        self.f = None

    def __enter__(self):
        _lock.acquire()
        try:
            self.f = get(self.path)
        except Exception:
            _lock.release()
            raise
        return self.f

    def __exit__(self, *exc):
        _lock.release()
        return False


def close_all():
    with _lock:
        for _, (_, f) in _cache.items():
            try:
                f.close()
            except Exception:
                pass
        _cache.clear()
