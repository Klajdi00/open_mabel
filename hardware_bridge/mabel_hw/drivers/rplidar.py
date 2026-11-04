"""rplidar_driver.py — minimal ROS-free Slamtec RPLIDAR serial driver (A2M12).

Speaks the classic RPLIDAR host protocol over the kit's CP2102 USB-UART adapter
(A2M12: 256000 baud 8N1; the adapter also POWERS the lidar from USB 5 V — startup
motor surge ~1.5 A, so keep it on a powered hub port). Implements what the bridge
needs and nothing more:

  * GET_INFO / GET_HEALTH        (identify + preflight, protective-stop recovery)
  * motor control                 DTR DEASSERTED = motor ON (the adapter's MOTOCTL
                                  follows DTR, classic Slamtec wiring — verified
                                  2026-07-09: the rotor would NOT start on the PWM
                                  command alone until DTR was deasserted) + the
                                  A5 F0 PWM command for speed. NOTE an earlier
                                  "never touch DTR, it kills comms" conclusion was
                                  WRONG — those experiments ran while the hands
                                  subsystem was spraying this same port (the real
                                  corruption source, since fenced off).
  * STANDARD scan (A5 20)         5-byte samples: quality, angle_q6, dist_q2.
                                  At 256000 baud that is ~4-5 k samples/s → ~400+
                                  points per 10 Hz revolution — plenty for 2D SLAM.
                                  (DenseBoost/express is a later upgrade if needed.)

`iter_scans()` yields one full revolution at a time as (angles_deg, dists_m,
quality) numpy arrays, cut at the sensor's start-of-sweep flag. Raw angles grow
CLOCKWISE viewed from the top (Slamtec convention); the caller converts to REP-103.

Protocol reference: Slamtec "RPLIDAR Interface Protocol and Application Notes"
(LR001); mirrors what rplidar_sdk / sllidar_ros2 do for A2 on the serial channel.
"""

import struct
import time

import numpy as np
import serial

SYNC = 0xA5
CMD_STOP = 0x25
CMD_RESET = 0x40
CMD_SCAN = 0x20
CMD_GET_INFO = 0x50
CMD_GET_HEALTH = 0x52
CMD_SET_PWM = 0xF0            # payload: u16 little-endian PWM (0..1023)

DESCRIPTOR = b"\xa5\x5a"
DEFAULT_MOTOR_PWM = 660       # Slamtec SDK default duty


class RPLidarError(RuntimeError):
    pass


class RPLidar:
    def __init__(self, port, baud=256000, timeout=1.0):
        # DTR is the adapter's MOTOCTL (deassert = motor ON — start_motor handles
        # it). exclusive=True (TIOCEXCL): any OTHER process/thread trying to open
        # the lidar port fails loudly instead of silently corrupting the scan
        # stream (the hands auto-claim once opened it and sprayed Feetech packets).
        self.ser = serial.Serial(port, baud, timeout=timeout, exclusive=True)

    # ---- low-level ---------------------------------------------------------
    def _cmd(self, cmd, payload=None):
        if payload is None:
            self.ser.write(bytes([SYNC, cmd]))
            return
        req = bytes([SYNC, cmd, len(payload)]) + bytes(payload)
        checksum = 0
        for b in req:
            checksum ^= b
        self.ser.write(req + bytes([checksum]))

    def _read_descriptor(self):
        d = self.ser.read(7)
        if len(d) != 7 or d[:2] != DESCRIPTOR:
            raise RPLidarError(f"bad response descriptor: {d!r}")
        dlen = (d[2] | d[3] << 8 | d[4] << 16 | (d[5] & 0x3F) << 24)
        return dlen, d[6]                            # (data length, data type)

    def _read_exact(self, n):
        d = self.ser.read(n)
        if len(d) != n:
            raise RPLidarError(f"short read ({len(d)}/{n} bytes)")
        return d

    # ---- control -----------------------------------------------------------
    def stop_scan(self):
        self._cmd(CMD_STOP)
        time.sleep(0.05)
        self.ser.reset_input_buffer()

    def reset(self):
        self._cmd(CMD_RESET)
        time.sleep(0.7)                              # boot banner then silence
        self.ser.reset_input_buffer()

    def start_motor(self, pwm=DEFAULT_MOTOR_PWM, kick=False):
        self.ser.dtr = False              # MOTOCTL follows DTR: deassert = motor ON
        time.sleep(0.1)
        if kick:
            # Breakaway kick: full duty for 2 s. A rotor that refuses to start at
            # the default duty (marginal 5 V after a replug, bearing stiction)
            # often breaks free at 100% and then holds fine at the normal speed.
            self._cmd(CMD_SET_PWM, struct.pack("<H", 1023))
            time.sleep(2.0)
        self._cmd(CMD_SET_PWM, struct.pack("<H", pwm))
        time.sleep(1.5)                              # spin-up before sampling

    def stop_motor(self):
        try:
            self._cmd(CMD_SET_PWM, struct.pack("<H", 0))
        except Exception:
            pass
        try:
            self.ser.dtr = True           # MOTOCTL follows DTR: assert = motor OFF
        except Exception:
            pass

    def get_info(self):
        self.ser.reset_input_buffer()
        self._cmd(CMD_GET_INFO)
        dlen, _ = self._read_descriptor()
        d = self._read_exact(dlen)
        return dict(model=d[0], fw=f"{d[2]}.{d[1]:02d}", hw=d[3],
                    serial=d[4:20].hex().upper())

    def get_health(self):
        self.ser.reset_input_buffer()
        self._cmd(CMD_GET_HEALTH)
        dlen, _ = self._read_descriptor()
        d = self._read_exact(dlen)
        return d[0], (d[1] | d[2] << 8)              # status (0/1/2), error code

    # ---- scanning ----------------------------------------------------------
    def start_scan(self):
        self._cmd(CMD_SCAN)
        dlen, dtype = self._read_descriptor()
        if dlen != 5 or dtype != 0x81:
            raise RPLidarError(f"unexpected scan descriptor (len={dlen}, type={dtype:#x})")

    def iter_scans(self, min_points=40):
        """Yield (angles_deg, dists_m, quality) numpy arrays per full revolution.

        Fully VECTORISED: each buffered chunk is parsed as one numpy batch (~30-60
        chunks/s) instead of a per-sample Python loop — the naive loop at the
        stream's 4-5 kHz sample rate stole enough GIL from the camera threads to
        halve the front D435's frame rate (measured 30 -> 12 Hz). Raises
        RPLidarError on protocol desync or a stalled stream."""
        rem = b""
        parts = []                                   # (angle, dist, qual) segments
        count = 0
        bad = 0
        empties = 0
        while True:
            # FIXED-SIZE BLOCKING read (~20 ms of stream), NOT in_waiting-sized:
            # sizing by in_waiting turns this loop into thousands of tiny reads/s
            # (the 26 KB/s stream trickles in), and each wakeup cycles the GIL —
            # measured stealing the front D435 down from 30 to ~10 Hz. 525 B = 105
            # samples fills in ~20 ms; ~50 wakeups/s is noise. A stall still
            # surfaces via the serial timeout (partial/empty read).
            chunk = self.ser.read(525)
            if not chunk:
                empties += 1                          # allow motor spin-up lag
                if empties > 5:
                    raise RPLidarError("scan stream stalled (no bytes)")
                continue
            empties = 0
            data = rem + chunk
            n = len(data) // 5
            rem = data[n * 5:]
            if not n:
                continue
            rows = np.frombuffer(data[:n * 5], dtype=np.uint8).reshape(n, 5)
            b0, b1 = rows[:, 0], rows[:, 1]
            good = (((b0 >> 1) & 1) != (b0 & 1)) & ((b1 & 1) == 1)
            if good.all():
                bad = 0
            else:
                bad += int((~good).sum())             # S == !S or check bit clear
                if bad > 20:
                    raise RPLidarError("scan stream desynchronised")
            angle = ((rows[:, 2].astype(np.uint32) << 7) | (b1 >> 1))[good] / 64.0
            dist = ((rows[:, 4].astype(np.uint32) << 8) | rows[:, 3])[good] / 4000.0
            qual = (b0 >> 2)[good]
            start_idx = np.flatnonzero((b0 & 1).astype(bool)[good])
            prev = 0
            for s in start_idx:                       # revolutions per chunk: ~0-1
                if s > prev:
                    parts.append((angle[prev:s], dist[prev:s], qual[prev:s]))
                    count += int(s) - prev
                if count >= min_points:
                    yield tuple(np.concatenate([p[i] for p in parts])
                                for i in range(3))
                parts, count = [], 0
                prev = int(s)
            if prev < len(angle):
                parts.append((angle[prev:], dist[prev:], qual[prev:]))
                count += len(angle) - prev

    def close(self):
        try:
            self.stop_scan()
            self.stop_motor()
        finally:
            self.ser.close()
