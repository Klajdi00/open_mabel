"""
Snapshot the active Python environment BEFORE installing LeRobot, so it can be
rolled back if pinned versions (torch / mujoco) get disturbed. Writes a stamped
freeze into the studio workspace.

Run:  python -m training_studio.server.tools.snapshot_env
"""
import os
import platform
import subprocess
import sys
import time


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ws = os.path.abspath(os.path.join(here, "..", "..", "workspace"))
    os.makedirs(ws, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    host = platform.node().split(".")[0]
    path = os.path.join(ws, f"env_snapshot_{host}_{stamp}.txt")

    lines = [f"# env snapshot {host} {stamp}", "## key versions"]
    for m in ("torch", "mujoco", "numpy", "gymnasium", "lerobot", "accelerate", "transformers"):
        try:
            mod = __import__(m)
            v = getattr(mod, "__version__", "?")
        except Exception as e:
            v = f"NOT INSTALLED ({type(e).__name__})"
        lines.append(f"{m}=={v}")
    try:
        import torch
        lines.append(f"cuda_available={torch.cuda.is_available()} "
                     f"mps_available={getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()}")
    except Exception:
        pass
    lines.append("\n## pip freeze")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True).stdout
        lines.append(freeze)
    except Exception as e:
        lines.append(f"(pip freeze failed: {e})")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"snapshot -> {path}")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
