"""
Run the MABEL Data Curation server (rosbag → curate → LeRobot):

    cd learning
    python -m data_curation.server          # http://127.0.0.1:8093

Env overrides: MABEL_STUDIO_PORT, MABEL_STUDIO_HOST, MABEL_DATASET_ROOT.
"""
import uvicorn

from . import config as CFG


def main():
    uvicorn.run("data_curation.server.app:app", host=CFG.HOST, port=CFG.PORT,
                log_level="info")


if __name__ == "__main__":
    main()
