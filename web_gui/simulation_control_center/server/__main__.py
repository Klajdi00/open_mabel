"""
Run the MABEL SIMULATION Control Center (the sim twin of control_center):

    cd ~/open_mabel
    python -m simulation_control_center.server     # http://0.0.0.0:8097

By default it also LAUNCHES the sim teleop stack (server/start_teleop.sh —
bridge :9090 + cameras :8080) unless one is already running; scene switching
from the GUI restarts it. Env overrides: MABEL_SCC_PORT, MABEL_SCC_HOST,
MABEL_SCC_SOURCE=sim|ros|mock|auto, MABEL_SCC_MANAGE_SIM=auto|1|0,
MABEL_SCC_VIEWER=1 (desktop MuJoCo viewer), MABEL_SCENE, MABEL_WS_PORT,
MABEL_MAP_PORT (0 = built-in mirror map plane), MABEL_HTTP_PORT,
MABEL_EPISODES_DIR.
"""
import uvicorn

from . import config as CFG


def main():
    # timeout_graceful_shutdown: on Ctrl-C / SIGTERM, uvicorn otherwise waits
    # FOREVER for lingering connections + background tasks (the telemetry/mirror
    # loops, health-check keep-alives, open teleop websockets) — so the
    # lifespan shutdown that tears down the sim stack never runs and it gets
    # ORPHANED. Bound the wait so shutdown always reaches the
    # teardown hook after a few seconds, with a single Ctrl-C.
    uvicorn.run("simulation_control_center.server.app:app",
                host=CFG.HOST, port=CFG.PORT, log_level="info",
                timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
