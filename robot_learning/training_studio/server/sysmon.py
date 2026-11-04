"""
System monitor — CPU + GPU telemetry emitted into the same metric stream as the
training losses (keys ``system.*``), so the UI renders GPU/CPU gauges uniformly
on any host. GPU backend is chosen by compute.py: NVIDIA (pynvml) on the CUDA
machines (4090/5080), Apple/MPS on the Mac, CPU-only otherwise.
"""
import threading
import time

from . import compute, metrics_store


def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def sample():
    """One reading of system metrics as a flat dict of system.* keys."""
    out = {}
    ps = _psutil()
    if ps is not None:
        try:
            out["system.cpu_percent"] = float(ps.cpu_percent(interval=None))
            vm = ps.virtual_memory()
            out["system.ram_percent"] = float(vm.percent)
            out["system.ram_used_gb"] = round(vm.used / 1e9, 3)
        except Exception:
            pass

    backend = compute.gpu_backend()
    if backend == "nvml":
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out["system.gpu_util"] = float(util.gpu)
            out["system.gpu_mem_used_gb"] = round(mem.used / 1e9, 3)
            out["system.gpu_mem_total_gb"] = round(mem.total / 1e9, 3)
            try:
                out["system.gpu_temp"] = float(
                    pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                out["system.gpu_power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                pass
            pynvml.nvmlShutdown()
        except Exception:
            pass
    elif backend == "apple":
        t = compute._torch()
        if t is not None:
            try:
                out["system.gpu_mem_used_gb"] = round(
                    t.mps.current_allocated_memory() / 1e9, 3)
            except Exception:
                pass
    return out


class SysMonitor(threading.Thread):
    """Poll system metrics for a run and write them as system.* metric rows."""

    def __init__(self, run_id, interval=2.0):
        super().__init__(daemon=True)
        self.run_id = run_id
        self.interval = interval
        self._stop = threading.Event()
        self._i = 0

    def stop(self):
        self._stop.set()

    def run(self):
        # prime cpu_percent (first call returns 0.0)
        ps = _psutil()
        if ps:
            try:
                ps.cpu_percent(interval=None)
            except Exception:
                pass
        while not self._stop.wait(self.interval):
            s = sample()
            if not s:
                continue
            t = time.time()
            try:
                with metrics_store._lock, metrics_store._conn() as c:
                    for k, v in s.items():
                        c.execute(
                            "INSERT INTO metric(run_id,step,key,value,mode,t) VALUES(?,?,?,?,?,?)",
                            (self.run_id, self._i, k, float(v), "system", t))
            except Exception:
                pass
            self._i += 1


if __name__ == "__main__":
    import json
    print(json.dumps(sample(), indent=2))
