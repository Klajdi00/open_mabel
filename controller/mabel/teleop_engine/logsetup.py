"""Shared coloured logger for the teleop package."""
import sys
import logging


class ColorFormatter(logging.Formatter):
    COLORS = {logging.INFO: "\033[92m", logging.WARNING: "\033[93m",
              logging.ERROR: "\033[91m", logging.DEBUG: "\033[96m"}
    RESET = "\033[0m"
    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}[{record.levelname}] {super().format(record)}{self.RESET}"

logger = logging.getLogger("Retarget")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(ColorFormatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_handler)
