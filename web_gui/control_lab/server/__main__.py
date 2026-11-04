"""`python -m control_lab.server` — run the Control Lab."""
import uvicorn

from . import config as CFG


def main():
    uvicorn.run("control_lab.server.app:app",
                host=CFG.HOST, port=CFG.PORT,
                timeout_graceful_shutdown=5, log_level="warning")


if __name__ == "__main__":
    main()
