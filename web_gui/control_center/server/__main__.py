"""
Run the MABEL Control Center (rviz-style visualizer + teleop + server monitor):

    cd ~/open_mabel
    python -m control_center.server          # http://0.0.0.0:8094

Env overrides: MABEL_CC_PORT, MABEL_CC_HOST, MABEL_CC_SOURCE=mock|ros|auto,
MABEL_CC_ROBOT_HOST, MABEL_WS_PORT, MABEL_MAP_PORT, MABEL_HTTP_PORT.
"""
import uvicorn

from . import config as CFG


def main():
    uvicorn.run("control_center.server.app:app", host=CFG.HOST, port=CFG.PORT,
                log_level="info")


if __name__ == "__main__":
    main()
