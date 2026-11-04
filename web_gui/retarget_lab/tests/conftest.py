import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "controller", ROOT / "web_gui"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# never pre-start NOVA's sim stack (Physics) from a test: it is ~14 s of MuJoCo + a gateway
os.environ.setdefault("MABEL_RTL_PHYSICS_PREWARM", "0")
