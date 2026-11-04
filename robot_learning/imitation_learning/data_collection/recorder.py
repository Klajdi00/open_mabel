"""
EpisodeRecorder — accumulates per-frame dicts and writes one HDF5 file per
episode on a background thread, so saving a long episode never stalls teleop.

One file per episode: ``<dataset_dir>/episode_XXXX.h5`` containing every logged
key as a time-major dataset (axis 0 = frame). RGB streams are stored as a
variable-length uint8 array of JPEG blobs (cheap, lossless to re-decode);
everything else is a gzip-compressed numeric array. The schema (names, camera
list, units) is written into the file attrs AND a sibling ``dataset_meta.json``.
"""
import json
import os
import queue
import threading
import time

import h5py
import numpy as np


def _save_worker(save_queue):
    while True:
        item = save_queue.get()
        if item is None:
            save_queue.task_done()
            break
        path, frames, attrs = item
        t0 = time.time()
        n = len(frames.get("time/sim", []))
        print(f"\n[recorder] saving {os.path.basename(path)} ({n} frames)…")
        try:
            with h5py.File(path, "w") as f:
                for k, v in attrs.items():
                    f.attrs[k] = v
                for key, seq in frames.items():
                    if key.endswith("/rgb"):
                        dt = h5py.vlen_dtype(np.uint8)
                        ds = f.create_dataset(key, shape=(len(seq),), dtype=dt)
                        for i, blob in enumerate(seq):
                            ds[i] = np.asarray(blob, dtype=np.uint8)
                    else:
                        arr = np.asarray(seq)
                        comp = "gzip" if arr.nbytes > 4096 else None
                        f.create_dataset(key, data=arr, compression=comp)
            print(f"[recorder] saved {os.path.basename(path)} in {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"[recorder] ERROR saving {path}: {e}")
        save_queue.task_done()


class EpisodeRecorder:
    def __init__(self, dataset_dir, schema_meta):
        self.dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)
        self.schema = schema_meta
        # resume episode numbering from whatever is already on disk
        existing = [f for f in os.listdir(dataset_dir) if f.endswith(".h5")]
        self.ep_count = len(existing)
        self.recording = False
        self.frames = {}
        self.n = 0

        self._q = queue.Queue()
        self._thread = threading.Thread(target=_save_worker, args=(self._q,), daemon=True)
        self._thread.start()

        # persist the human-readable schema once
        with open(os.path.join(dataset_dir, "dataset_meta.json"), "w") as fp:
            json.dump(schema_meta, fp, indent=2)

    # ── episode lifecycle ─────────────────────────────────────────────────────
    def start(self):
        self.frames = {}
        self.n = 0
        self.recording = True

    def append(self, obs, action, sim_time, wall_time):
        if not self.recording:
            return
        row = {"time/sim": np.float64(sim_time), "time/wall": np.float64(wall_time)}
        row.update(obs)
        row.update(action)
        for k, v in row.items():
            self.frames.setdefault(k, []).append(v)
        self.n += 1

    def finish(self, discard=False, task=None):
        """Stop recording; queue the episode for saving (or drop it)."""
        if not self.recording:
            return None
        self.recording = False
        if discard or self.n == 0:
            self.frames = {}
            return None
        path = os.path.join(self.dir, f"episode_{self.ep_count:04d}.h5")
        attrs = {
            "schema_json": json.dumps(self.schema),
            "num_frames": self.n,
            "episode_index": self.ep_count,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if task:
            attrs["task"] = task
        self._q.put((path, self.frames, attrs))
        self.ep_count += 1
        self.frames = {}
        return path

    def close(self):
        if self.recording:
            self.finish(discard=True)
        self._q.put(None)
        self._q.join()
