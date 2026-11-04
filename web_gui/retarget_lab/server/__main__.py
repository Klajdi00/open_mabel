"""python -m retarget_lab.server  (run from web_gui/)."""
import logging

import uvicorn

from . import config as CFG
from .app import _NoCache, create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    print(f"MABEL Retarget Lab  http://localhost:{CFG.PORT}  ·  Vision Pro → "
          f"ws://{CFG.lan_ip()}:{CFG.VP_PORT}/teleop")
    uvicorn.run(_NoCache(create_app()), host=CFG.HOST, port=CFG.PORT, log_level="warning")
