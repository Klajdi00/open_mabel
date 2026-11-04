"""
Tiny in-memory job registry for long-running background work (materialize,
sample-data generation), with progress the frontend can poll or stream.
"""
import threading
import traceback
import uuid

_jobs = {}
_lock = threading.Lock()


def create(kind):
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = {"id": jid, "kind": kind, "status": "pending",
                      "done": 0, "total": 0, "stage": 0, "result": None, "error": None}
    return jid


def _update(jid, **kw):
    with _lock:
        if jid in _jobs:
            _jobs[jid].update(kw)


def get(jid):
    with _lock:
        return dict(_jobs[jid]) if jid in _jobs else None


def run(jid, fn):
    """Run fn(progress) on a daemon thread; fn gets a progress(done,total,stage)
    callback and returns the job result."""
    def _progress(done, total, stage=0):
        _update(jid, done=int(done), total=int(total), stage=int(stage),
                status="running")

    def _worker():
        _update(jid, status="running")
        try:
            res = fn(_progress)
            _update(jid, status="done", result=res, done=_jobs[jid]["total"]
                    if _jobs[jid]["total"] else _jobs[jid]["done"])
        except Exception as e:
            _update(jid, status="error", error=f"{e}\n{traceback.format_exc()}")

    threading.Thread(target=_worker, daemon=True).start()
    return jid
