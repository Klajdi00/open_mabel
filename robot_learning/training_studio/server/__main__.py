"""Launch the MABEL Trainer Studio server: `python -m training_studio.server`."""
import os

import uvicorn

# LeRobot/transformers env must be set before any policy import (numpy-2 / TF).
for k, v in (("USE_TF", "0"), ("USE_FLAX", "0"),
             ("TRANSFORMERS_VERBOSITY", "error"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(k, v)

from . import config as CFG  # noqa: E402


def main():
    uvicorn.run("training_studio.server.app:app", host=CFG.HOST, port=CFG.PORT,
                log_level="info")


if __name__ == "__main__":
    main()
