"""
Prepare a combined multi-episode dataset from the curator's single-episode
exports (convenience CLI used by the test harness and for manual smoke runs).

Run:  python -m training_studio.server.tools.prepare_test_dataset [names...]
"""
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from training_studio.server import dataset_registry as DR  # noqa: E402
from training_studio.server import feature_select as FS    # noqa: E402


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not names:
        names = ["mabel_test", "mabel_curated_test"]
    print(f"combining: {names}")
    rec = DR.combine(names, "prepared_dataset")
    print(f"  -> {rec['n_episodes']} episodes, {rec['total_frames']} frames, "
          f"{rec['action_dim']}-D action, {rec['n_cameras']} cameras")
    sp = DR.split(rec["n_episodes"], (0.8, 0.1, 0.1), 42)
    print(f"  split (80/10/10): {sp['counts']}")
    sel = FS.resolve("whole_body")
    train = DR.subset_dataset(rec["path"], sp["train"] or [0], "prepared_train")
    v30 = DR.ensure_v30(train["path"])
    print(f"  train v3.0 -> {v30}")


if __name__ == "__main__":
    main()
