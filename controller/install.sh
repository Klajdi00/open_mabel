#!/usr/bin/env bash
# One-command install of the MABEL controller library (the `mabel` package).
#
#   ./install.sh            # editable install (development, the normal case)
#   ./install.sh --qp       # + exact/fast QP backends for self-collision
#   ./install.sh --check    # install, then run the import/self test suite
#
# After install, `import mabel` works from anywhere — no sys.path hacks needed.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
EXTRAS=""
CHECK=0
for arg in "$@"; do
  case "$arg" in
    --qp)    EXTRAS="[qp]" ;;
    --check) CHECK=1 ;;
    *) echo "unknown option: $arg (use --qp / --check)"; exit 2 ;;
  esac
done

echo "==> pip install -e .$EXTRAS  (python: $($PY -V))"
"$PY" -m pip install -e ".$EXTRAS"

echo "==> verifying the install"
"$PY" - <<'EOF'
import sys
import mabel.config as C
assert 'mujoco' not in sys.modules, "mabel.config must stay MuJoCo-free"
C.safe_base_step(0, 0, 0, 0.3, 0, 0, 0.01, 0.2)
import mabel
for s in mabel.__all__:
    getattr(mabel, s)
print(f"mabel {getattr(mabel, '__version__', '0.1.0')}: "
      f"{len(mabel.__all__)} public symbols import cleanly")
EOF

if [ "$CHECK" = 1 ]; then
  echo "==> running the test suite"
  "$PY" -m pytest tests/ -q
fi
echo "==> done. Try:  python3 -c 'from mabel import MabelTeleop'"
