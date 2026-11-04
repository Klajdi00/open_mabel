"""
Device manager — makes the studio portable across the user's machines.

CUDA is the real training target (Ubuntu 4090 laptop, 5080 desktop); the Mac is
a CPU/MPS dev/fallback box. Everything downstream (config builder, sysmon, eval,
deploy) asks this module for the device and the matching GPU-monitor backend, so
the same package runs unchanged on Mac / 4090 / 5080.
"""
import functools
import platform
import shutil


@functools.lru_cache(maxsize=1)
def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def resolve_device(prefer: str = "auto") -> str:
    """Return 'cuda' | 'mps' | 'cpu'. ``prefer`` may pin a device; 'auto' picks
    the strongest available (cuda > mps > cpu)."""
    prefer = (prefer or "auto").lower()
    t = _torch()
    if prefer in ("cuda", "mps", "cpu"):
        if prefer == "cuda" and (t is None or not t.cuda.is_available()):
            return resolve_device("auto")
        if prefer == "mps" and (t is None or not getattr(t.backends, "mps", None)
                                or not t.backends.mps.is_available()):
            return resolve_device("auto")
        return prefer
    if t is not None:
        if t.cuda.is_available():
            return "cuda"
        if getattr(t.backends, "mps", None) and t.backends.mps.is_available():
            return "mps"
    return "cpu"


def gpu_backend() -> str:
    """Which sysmon backend to poll: 'nvml' (CUDA), 'apple' (Mac Metal), 'none'."""
    dev = resolve_device("auto")
    if dev == "cuda":
        return "nvml"
    if dev == "mps" or platform.system() == "Darwin":
        return "apple"
    return "none"


def describe() -> dict:
    """Human-facing summary for the UI device badge."""
    t = _torch()
    dev = resolve_device("auto")
    info = {"device": dev, "backend": gpu_backend(),
            "torch": getattr(t, "__version__", None) if t else None,
            "platform": platform.platform(), "host": platform.node()}
    if t is not None and dev == "cuda":
        try:
            info["gpu_count"] = t.cuda.device_count()
            info["gpu_names"] = [t.cuda.get_device_name(i) for i in range(t.cuda.device_count())]
        except Exception:
            pass
    elif dev == "mps":
        info["gpu_names"] = ["Apple GPU (Metal / MPS)"]
        info["gpu_count"] = 1
    info["has_nvidia_smi"] = bool(shutil.which("nvidia-smi"))
    return info


if __name__ == "__main__":  # `python -m training_studio.server.compute`
    import json
    print(json.dumps(describe(), indent=2))
