#!/usr/bin/env bash
# Host-agnostic LeRobot setup for the Trainer Studio.
# Snapshots the active env first (rollback safety), installs LeRobot + monitors,
# then verifies the train entrypoint, the registered policy types, the DiT plugin,
# and the resolved compute device. Run once per machine (Mac dev / CUDA hosts).
#
#   bash training_studio/server/tools/install_lerobot.sh
set -e
PY="${MABEL_TRAIN_PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LEARNING="$(cd "$HERE/../../.." && pwd)"

echo "== 1/4  snapshot env (rollback safety) =="
"$PY" -m training_studio.server.tools.snapshot_env

echo "== 2/4  install lerobot + psutil + pynvml =="
"$PY" -m pip install "lerobot" psutil pynvml
# LeRobot's transformers may drag an old scikit-learn that breaks on numpy>=2:
"$PY" -m pip install --upgrade "scikit-learn>=1.6" "draccus==0.10.0" >/dev/null 2>&1 || true

echo "== 3/4  verify (torch/mujoco preserved, policies, DiT, device) =="
USE_TF=0 USE_FLAX=0 TRANSFORMERS_VERBOSITY=error PYTHONPATH="$LEARNING" "$PY" - <<'PY'
import importlib
for m in ("torch", "mujoco", "numpy", "lerobot"):
    v = getattr(importlib.import_module(m), "__version__", "?")
    print(f"  {m} == {v}")
import lerobot.policies  # noqa
from lerobot.configs.policies import PreTrainedConfig
print("  registered policy types:", sorted(PreTrainedConfig.get_known_choices().keys()))
from training_studio.server.lerobot_shim import dit_policy  # noqa
from lerobot.policies.factory import get_policy_class
print("  DiT plugin ->", get_policy_class("dit").__name__)
from training_studio.server import compute
print("  resolved device:", compute.resolve_device(), "| gpu backend:", compute.gpu_backend())
import shutil
print("  lerobot-train CLI:", shutil.which("lerobot-train") or "(module: python -m lerobot.scripts.lerobot_train)")
PY

echo "== 4/4  done. Launch the studio with: python -m training_studio.server =="
