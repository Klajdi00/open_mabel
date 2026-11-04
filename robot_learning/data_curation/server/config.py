"""
Data Studio server configuration.

The dataset root defaults to the SAME folder the collection pipeline writes to
(``data_collection/datasets``), so episodes you collect show up in the Studio
with no extra wiring. The editing *workspace* (EDL projects, transcode cache)
lives OUTSIDE the dataset root so source episodes stay pristine — editing is
strictly non-destructive.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDIO = os.path.dirname(_HERE)                 # data_curation/  (this package)
_LEARNING = os.path.dirname(_STUDIO)             # robot_learning/
# data_collection (dataset root, schema, camera/lidar/depth config, controller
# bootstrap) lives under imitation_learning/, a sibling of this new package.
_IL = os.path.join(_LEARNING, "imitation_learning")
if _IL not in sys.path:
    sys.path.insert(0, _IL)

# Reuse the collection pipeline's config as the single source of truth for the
# dataset location, camera list, lidar/depth scaling, and the scene XML. This
# import also bootstraps the `mabel` controller package onto sys.path.
from data_collection import config as CFG  # noqa: E402

# Where episodes live (one subfolder per dataset, episode_XXXX.h5 inside).
DATASET_ROOT = os.environ.get("MABEL_DATASET_ROOT", CFG.DATASET_ROOT)

# Studio-private workspace (never under DATASET_ROOT).
WORKSPACE = os.environ.get(
    "MABEL_STUDIO_WORKSPACE", os.path.join(_STUDIO, "workspace"))
PROJECTS_DIR = os.path.join(WORKSPACE, "projects")     # <name>.edl.json sidecars
CACHE_DIR = os.path.join(WORKSPACE, "cache")           # on-demand mp4 transcodes

WEB_DIR = os.path.join(_STUDIO, "web")
# The website rig assets (GLB + joint manifest) are copied into web/assets at
# scaffold time; this path is the source of truth if a refresh is needed.
WEBSITE_ASSETS = os.path.abspath(
    os.path.join(_IL, "..", "..", "web_gui", "shared_assets"))

PORT = int(os.environ.get("MABEL_STUDIO_PORT", "8093"))   # 8093 so it runs ALONGSIDE the stable studio (8092)
HOST = os.environ.get("MABEL_STUDIO_HOST", "127.0.0.1")

# Number of open read-only HDF5 handles to keep warm.
H5_LRU = 16
# Reads run in a small threadpool so the event loop never blocks on disk; the
# HDF5 global lock means these are effectively single-flight but non-blocking.
H5_WORKERS = 4

os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
