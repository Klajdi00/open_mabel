"""
damiao_serial.py
================

Backwards-compatible shim.  Serial transport now lives in :mod:`transports`
(``SerialDamiaoTransport``, ported from the firmware driver) and the bus logic in
:class:`damiao.DamiaoBus`, which speaks **serial**, **gs_usb**, and **socketcan**
through one class.

``DamiaoSerialBus`` is kept as a thin subclass that forces the serial backend, so
older code/scripts (e.g. ``hardware/serial_demo.py``) keep working unchanged.
Prefer ``damiao.DamiaoBus(interface=...)`` or ``session.make_bus(...)`` in new code.
"""
from __future__ import annotations

from typing import Optional

from .damiao import DamiaoBus


class DamiaoSerialBus(DamiaoBus):
    """DaMiao bus forced onto the DM-USB2FDCAN serial adapter."""

    def __init__(self, port: Optional[str] = None, baud: int = 921600):
        super().__init__(interface="serial", port=port)
