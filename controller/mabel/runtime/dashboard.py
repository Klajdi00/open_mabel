"""Terminal dashboard for the MABEL teleop loops (in-place ANSI redraw)."""
import sys

_C = dict(blue="\033[94m", green="\033[92m", yellow="\033[93m", cyan="\033[96m",
          red="\033[91m", end="\033[0m", bold="\033[1m")


class Dashboard:
    def __init__(self, lines=20):
        self.lines = lines
        print("\n" * lines)

    def render(self, rows):
        """rows: list of preformatted strings."""
        sys.stdout.write(f"\033[{self.lines}A\033[J")
        sys.stdout.write("\n".join(rows[: self.lines]))
        sys.stdout.write("\n")
        sys.stdout.flush()

    @staticmethod
    def header(title):
        bar = _C["blue"] + _C["bold"] + "=" * 52 + _C["end"]
        return [bar, f"{_C['cyan']}{_C['bold']} {title}{_C['end']}", bar]

    @staticmethod
    def line(label, value, color="green"):
        return f" {_C['bold']}{label:<22}{_C['end']} {_C.get(color,'')}{value}{_C['end']}"
