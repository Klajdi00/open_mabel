"""Shared helpers for the feetech CLI scripts.

All scripts here are thin front-ends over the lean :mod:`orcahand` API -- they
never touch the serial bus directly. This module adds the package to ``sys.path``
when run from a source checkout (so ``python scripts/foo.py`` works without
``pip install``), and centralises the ``--model/--port/--mock`` plumbing.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Make the package importable from a bare checkout (feetech/ is the repo root).
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from orcahand import HandConfig, OrcaHand          # noqa: E402
from orcahand.calibrate import Term                 # noqa: E402

__all__ = ["OrcaHand", "HandConfig", "Term", "add_common_args", "make_hand"]


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Add the connection flags every script shares."""
    ap.add_argument("--model", default="left", choices=HandConfig.available_models(),
                    help="which hand config to load (default: left)")
    ap.add_argument("--port", default=None, help="serial device ('' = auto-detect)")
    ap.add_argument("--mock", action="store_true",
                    help="run against an in-memory bus -- no hardware needed")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colour")


def make_hand(args, **kwargs) -> OrcaHand:
    """Build an :class:`OrcaHand` (real or mock) from parsed common args."""
    if args.mock:
        return OrcaHand.mock(args.model, **kwargs)
    return OrcaHand(args.model, port=args.port, **kwargs)
