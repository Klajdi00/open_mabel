"""
Metrics store — the downstream of the JSONL contract.

A background Tailer per run reads ``metrics.jsonl`` (written by StudioLogger) and
``logs.jsonl`` (stdout captured by trainer.py), ingests them into SQLite, and the
app streams new rows over a WebSocket. This is the W&B replacement's data layer:
runs, scalar metrics (loss/sub-losses/lr/grad-norm/throughput/system.*),
checkpoints and logs — all queryable for plots, tables and cross-run overlay.
"""
import json
import os
import sqlite3
import threading
import time

from . import config as CFG

_lock = threading.Lock()
_tailers = {}


def _conn():
    c = sqlite3.connect(CFG.DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    return c


def init_db():
    with _lock, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS run(
              run_id TEXT PRIMARY KEY, name TEXT, arch TEXT, policy_type TEXT,
              scope TEXT, dataset_id TEXT, dataset_hash TEXT, device TEXT,
              status TEXT, output_dir TEXT, config_json TEXT, meta_json TEXT,
              pid INTEGER, error TEXT, created_at REAL, updated_at REAL);
            CREATE TABLE IF NOT EXISTS metric(
              id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step INTEGER,
              key TEXT, value REAL, mode TEXT, t REAL);
            CREATE TABLE IF NOT EXISTS checkpoint(
              run_id TEXT, step INTEGER, path TEXT, t REAL);
            CREATE TABLE IF NOT EXISTS log(
              id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, t REAL, line TEXT);
            CREATE INDEX IF NOT EXISTS ix_metric_run ON metric(run_id, id);
            CREATE INDEX IF NOT EXISTS ix_log_run ON log(run_id, id);
            CREATE INDEX IF NOT EXISTS ix_ckpt_run ON checkpoint(run_id, step);
            """)


# ── run registry ──────────────────────────────────────────────────────────────
def register_run(run_id, **fields):
    now = time.time()
    cols = {"run_id": run_id, "status": "queued", "created_at": now, "updated_at": now}
    cols.update(fields)
    keys = ",".join(cols)
    qs = ",".join("?" for _ in cols)
    with _lock, _conn() as c:
        c.execute(f"INSERT OR REPLACE INTO run({keys}) VALUES({qs})", list(cols.values()))


def update_run(run_id, **fields):
    fields["updated_at"] = time.time()
    sets = ",".join(f"{k}=?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE run SET {sets} WHERE run_id=?", list(fields.values()) + [run_id])


def get_run(run_id):
    with _conn() as c:
        c.row_factory = sqlite3.Row
        r = c.execute("SELECT * FROM run WHERE run_id=?", (run_id,)).fetchone()
        return dict(r) if r else None


def list_runs(limit=200):
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT run_id,name,arch,policy_type,scope,dataset_id,device,status,"
            "output_dir,pid,error,created_at,updated_at FROM run ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]


# ── queries for plots / tables ────────────────────────────────────────────────
def metric_keys(run_id):
    with _conn() as c:
        return [r[0] for r in c.execute(
            "SELECT DISTINCT key FROM metric WHERE run_id=? ORDER BY key", (run_id,)).fetchall()]


def get_series(run_id, key, mode="train"):
    with _conn() as c:
        rows = c.execute(
            "SELECT step,value,t FROM metric WHERE run_id=? AND key=? AND mode=? ORDER BY step",
            (run_id, key, mode)).fetchall()
        return [{"step": s, "value": v, "t": t} for s, v, t in rows]


def events_since(run_id, last_metric_id=0, last_log_id=0):
    """New metric + log rows for the live WebSocket."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        metrics = [dict(r) for r in c.execute(
            "SELECT id,step,key,value,mode,t FROM metric WHERE run_id=? AND id>? ORDER BY id LIMIT 5000",
            (run_id, last_metric_id)).fetchall()]
        logs = [dict(r) for r in c.execute(
            "SELECT id,t,line FROM log WHERE run_id=? AND id>? ORDER BY id LIMIT 2000",
            (run_id, last_log_id)).fetchall()]
        ckpts = [dict(r) for r in c.execute(
            "SELECT step,path,t FROM checkpoint WHERE run_id=? ORDER BY step", (run_id,)).fetchall()]
    return {"metrics": metrics, "logs": logs, "checkpoints": ckpts,
            "last_metric_id": metrics[-1]["id"] if metrics else last_metric_id,
            "last_log_id": logs[-1]["id"] if logs else last_log_id}


def get_checkpoints(run_id):
    with _conn() as c:
        return [{"step": s, "path": p, "t": t} for s, p, t in c.execute(
            "SELECT step,path,t FROM checkpoint WHERE run_id=? ORDER BY step", (run_id,)).fetchall()]


def append_log(run_id, line):
    with _lock, _conn() as c:
        c.execute("INSERT INTO log(run_id,t,line) VALUES(?,?,?)", (run_id, time.time(), line))


def get_logs(run_id, limit=8000):
    """All persisted stdout/terminal + training log lines for a run — so a stopped
    or failed job's logs can be re-viewed later for debugging."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id,t,line FROM log WHERE run_id=? ORDER BY id LIMIT ?",
            (run_id, limit)).fetchall()
        return [{"id": i, "t": t, "line": ln} for i, t, ln in rows]


# ── the tailer ────────────────────────────────────────────────────────────────
_LOG_NOISE = ("pynvml package is deprecated", "import pynvml", "FutureWarning",
              "nvidia-ml-py instead")


class Tailer(threading.Thread):
    """Follow a run's metrics.jsonl (+ its stdout log file) and ingest into SQLite
    until stopped. Reads from the current file position, so it also works when the
    server restarts and re-attaches to an already-running job."""

    def __init__(self, run_id, output_dir, logfile=None):
        super().__init__(daemon=True)
        self.run_id = run_id
        self.path = os.path.join(output_dir, "metrics.jsonl")
        self.logfile = logfile
        self._stop = threading.Event()
        self._pos = 0
        self._lpos = 0

    def stop(self):
        self._stop.set()

    def _ingest_logs(self, c):
        if not self.logfile or not os.path.exists(self.logfile):
            return
        with open(self.logfile, errors="replace") as f:
            f.seek(self._lpos)
            for line in f:
                if not line.endswith("\n"):
                    break
                self._lpos += len(line.encode("utf-8", "replace"))
                s = line.rstrip("\n")
                st = s.strip()
                if not st or any(n in st for n in _LOG_NOISE) or "s/step]" in st or "it/s]" in st:
                    continue
                c.execute("INSERT INTO log(run_id,t,line) VALUES(?,?,?)", (self.run_id, time.time(), s))

    def _ingest_line(self, rec, c):
        typ = rec.get("_type")
        t = rec.get("t", time.time())
        if typ == "metric":
            step = rec.get("step")
            mode = rec.get("mode", "train")
            for k, v in rec.items():
                if k in ("_type", "step", "mode", "t"):
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                c.execute("INSERT INTO metric(run_id,step,key,value,mode,t) VALUES(?,?,?,?,?,?)",
                          (self.run_id, step, k, float(v), mode, t))
        elif typ == "checkpoint":
            c.execute("INSERT INTO checkpoint(run_id,step,path,t) VALUES(?,?,?,?)",
                      (self.run_id, rec.get("step"), rec.get("path"), t))

    def run(self):
        while not self._stop.is_set():
            try:
                with _lock, _conn() as c:
                    if os.path.exists(self.path):
                        with open(self.path) as f:
                            f.seek(self._pos)
                            for line in f:
                                if not line.endswith("\n"):
                                    break  # partial line; wait for the rest
                                self._pos += len(line.encode("utf-8", "replace"))
                                line = line.strip()
                                if line:
                                    try:
                                        self._ingest_line(json.loads(line), c)
                                    except json.JSONDecodeError:
                                        pass
                    self._ingest_logs(c)
            except Exception:
                pass
            time.sleep(0.5)


def start_tailer(run_id, output_dir, logfile=None, reset=False):
    with _lock:
        if run_id in _tailers:
            return
    if reset:   # re-attaching after a restart: clear old rows, re-ingest from disk
        with _lock, _conn() as c:
            c.execute("DELETE FROM metric WHERE run_id=?", (run_id,))
            c.execute("DELETE FROM log WHERE run_id=?", (run_id,))
            c.execute("DELETE FROM checkpoint WHERE run_id=?", (run_id,))
    t = Tailer(run_id, output_dir, logfile)
    t.start()
    with _lock:
        _tailers[run_id] = t


def stop_tailer(run_id):
    with _lock:
        t = _tailers.pop(run_id, None)
    if t:
        t.stop()


# ── job persistence: PID liveness + startup reconcile ─────────────────────────
def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def run_progress(run_id):
    """Last step / epoch for a run (for the Jobs list)."""
    with _conn() as c:
        row = c.execute("SELECT MAX(step) FROM metric WHERE run_id=? AND key='loss' AND mode='train'",
                        (run_id,)).fetchone()
        step = row[0] if row else None
        er = c.execute("SELECT value FROM metric WHERE run_id=? AND key='epochs' ORDER BY id DESC LIMIT 1",
                       (run_id,)).fetchone()
        return {"step": step, "epoch": er[0] if er else None}


def reconcile(logs_dir):
    """On server startup: for every run that was 'running'/'preparing', check if its
    process is still alive. If so, re-attach a tailer (training kept going while the
    studio was down). If not, mark it 'interrupted' so it can be resumed."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute(
            "SELECT run_id,pid,status,output_dir FROM run WHERE status IN "
            "('running','preparing','queued')").fetchall()]
    for r in rows:
        rid, pid, od = r["run_id"], r["pid"], r["output_dir"]
        if pid and pid_alive(pid) and od:
            logfile = os.path.join(logs_dir, f"{rid}.log")
            start_tailer(rid, od, logfile, reset=True)   # keep ingesting live
            update_run(rid, status="running")
        else:
            update_run(rid, status="interrupted",
                       error="studio restarted; process not found — resume from last checkpoint")
