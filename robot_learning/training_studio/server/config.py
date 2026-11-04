"""
MABEL Trainer Studio — server configuration.

Sibling to the Data Curation studio. Reuses the collection pipeline's config as
the single source of truth for the dataset root, camera list and scene XML (the
import also bootstraps the shared ``mabel`` controller onto ``sys.path``), and
adds a studio-private workspace for training runs, derived datasets and the
SQLite metrics store. Runs on its OWN port so it coexists with the curator.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDIO = os.path.dirname(_HERE)                 # training_studio/  (this package)
_LEARNING = os.path.dirname(_STUDIO)             # robot_learning/
LEARNING_DIR = _LEARNING                          # subprocess cwd / PYTHONPATH root
_IL = os.path.join(_LEARNING, "imitation_learning")
if _IL not in sys.path:
    sys.path.insert(0, _IL)
if _LEARNING not in sys.path:
    sys.path.insert(0, _LEARNING)                # so `data_curation` is importable

# Reuse the collection pipeline config (dataset root, cameras, scene XML) and
# bootstrap the `mabel` controller package. Guarded so the studio still imports
# on a machine without the sim deps (stats/registry work regardless).
try:
    from data_collection import config as CFG  # noqa: E402
    DATASET_ROOT = os.environ.get("MABEL_DATASET_ROOT", CFG.DATASET_ROOT)
    SCENE_XML = getattr(CFG, "SCENE_XML", None)
    CAMERAS = list(getattr(CFG, "CAMERAS", []))
    LOG_FPS = float(getattr(CFG, "LOG_FPS", 15.0))
except Exception:  # pragma: no cover - fallback when sim env missing
    CFG = None
    DATASET_ROOT = os.environ.get("MABEL_DATASET_ROOT", "")
    SCENE_XML = None
    CAMERAS = []
    LOG_FPS = 15.0

# Where the curator writes LeRobot v2.1 exports — the trainer's primary source.
CURATOR_EXPORTS = os.environ.get(
    "MABEL_CURATOR_EXPORTS",
    os.path.join(_LEARNING, "data_curation", "workspace", "lerobot_exports"))

# Studio-private workspace (never under DATASET_ROOT). Holds the SQLite index and
# derived/scoped dataset views — the TRANSIENT scratch. Durable, human-browsable
# training + eval artifacts live under robot_learning/{training,eval}/ (see below).
WORKSPACE = os.environ.get(
    "MABEL_TRAINER_WORKSPACE", os.path.join(_STUDIO, "workspace"))
DATASETS_DIR = os.path.join(WORKSPACE, "datasets")  # combined/split/scoped dataset views
DB_PATH = os.path.join(WORKSPACE, "studio.db")      # SQLite: runs, metrics, checkpoints (index)

# ── Durable experiment tree (the professional, browsable layout) ──────────────
# Every training run lands under  training/<arch>/<dataset>/<run_id>/  and every
# evaluation under  eval/<arch>/<dataset>/<run_id>/<eval_id>/ . These are git-
# ignored but are the canonical, hand-inspectable home of checkpoints, configs,
# plots, logs and eval results — organized so it scales across many
# archs × datasets × tasks × runs. Managed by server/runstore.py.
TRAINING_ROOT = os.environ.get("MABEL_TRAINING_ROOT", os.path.join(_LEARNING, "training"))
EVAL_ROOT = os.environ.get("MABEL_EVAL_ROOT", os.path.join(_LEARNING, "eval"))

# Back-compat: some code/tests still refer to RUNS_DIR. It now points at the
# training tree root; per-run paths come from runstore.RunLayout, not RUNS_DIR/<id>.
RUNS_DIR = TRAINING_ROOT

# Where curated LeRobot exports live (the trainer's dataset source). Scans BOTH the
# legacy curator workspace AND the new curated_dataset/{real,simulation}/<name>/<repo>.
CURATED_ROOT = os.environ.get(
    "MABEL_CURATED_ROOT", os.path.abspath(os.path.join(_LEARNING, "..", "curated_dataset")))

WEB_DIR = os.path.join(_STUDIO, "web")
# The website rig assets (GLB + joint manifest) power the small 3D deploy viewer.
WEBSITE_ASSETS = os.path.abspath(
    os.path.join(_LEARNING, "..", "web_gui", "shared_assets"))

PORT = int(os.environ.get("MABEL_TRAINER_PORT", "8095"))   # 8092 stable / 8093 curator / 8095 trainer
HOST = os.environ.get("MABEL_TRAINER_HOST", "127.0.0.1")

# Operator credentials required for EVERY mutating training action (launch, stop,
# pause, resume, retrain, deploy, eval, community approve/reject). A styled
# username+passkey popup establishes a signed session cookie (see auth.py);
# checked SERVER-side so a job cannot be sent or killed without them.
# Credentials come ONLY from the environment — NEVER hard-code them here (e.g.
# set them in an untracked systemd drop-in). Unset => auth.check_credentials denies every login (fail-closed).
TRAIN_USER = os.environ.get("MABEL_TRAIN_USER", "")
TRAIN_KEY = os.environ.get("MABEL_TRAIN_KEY", "")

# The base-conda python that has LeRobot installed (training subprocesses use it).
TRAIN_PYTHON = os.environ.get("MABEL_TRAIN_PYTHON", sys.executable)

# Environment every LeRobot subprocess needs (see trainer-studio memory):
#  - USE_TF/USE_FLAX=0 : stop transformers importing a numpy-1.x TensorFlow that
#    crashes under numpy 2.2 (would break the whole lerobot.policies import).
LEROBOT_ENV = {
    "USE_TF": "0",
    "USE_FLAX": "0",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
# torchcodec's shared libs are missing on this Mac; pyav decodes the export MP4s.
VIDEO_BACKEND = os.environ.get("MABEL_VIDEO_BACKEND", "pyav")

# ── community training queue (optional external job mailbox; off by default) ───
# The website posts job proposals to this service; the studio pulls the pending
# ones with the owner token (kept server-side — never sent to a browser) and, on
# approval, launches them locally and reports status back.
COMMUNITY_API = os.environ.get("MABEL_COMMUNITY_API", "").rstrip("/")
COMMUNITY_OWNER_TOKEN = os.environ.get("MABEL_COMMUNITY_TOKEN", "")

for _d in (WORKSPACE, DATASETS_DIR, TRAINING_ROOT, EVAL_ROOT):
    os.makedirs(_d, exist_ok=True)
