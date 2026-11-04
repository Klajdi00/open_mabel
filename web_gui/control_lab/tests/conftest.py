"""Suite-wide isolation: tests must never touch the LIVE studio's viewer
memmap (the state file is now BIDIRECTIONAL — a test world writing qpos or
reading a perturbation slot would fight a running studio + mjpython mirror).
Set the env before any test module imports control_lab.server.config."""
import os

os.environ.setdefault("MABEL_CTS_STATE_FILE",
                      "/tmp/mabel_control_lab_TEST_state.f64")
os.environ.setdefault("MABEL_CTS_VP_PORT", "19391")   # keep off the real 9090
