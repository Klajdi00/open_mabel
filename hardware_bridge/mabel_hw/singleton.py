"""singleton.py — the HARD one-owner lock on MABEL's hardware links.

Only ONE process may ever talk to the Teensy/CAN/serial at a time — two owners
fight over setpoints and the base OSCILLATES (the 2026 incident this guards
against). The lock is on the LINK, not on the ROS graph: a second instance in a
different ROS_DOMAIN_ID still opens the same UDP socket to the Teensy, so a
domain check is not enough. flock makes a second start fail loudly (exit 3)
instead of silently double-commanding the hardware.

Moved verbatim from hardware_bridge.py (which re-exports these names — the
selftest and singleton_test.sh surfaces are unchanged). This is scope layer 1 of
the single-writer story; layer 2 is mabel_hw.arbitration (in-process leases),
layer 3 the DDS publisher watchdogs in the ROS adapters.
"""
import atexit
import fcntl
import os
import sys
import time

_LOCK_PATH = os.environ.get("MABEL_HW_LOCK", "/tmp/mabel_hardware_bridge.lock")


def _try_lock():
    """One lock attempt → (fd, "") on success, (None, owner_pid_str) if held."""
    fd = open(_LOCK_PATH, "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            fd.seek(0)
            other = fd.read().strip()
        except Exception:
            other = ""
        fd.close()
        return None, other
    fd.seek(0)
    fd.truncate()
    fd.write(str(os.getpid()))
    fd.flush()
    return fd, ""


def _owner_alive(pid_str):
    try:
        os.kill(int(pid_str), 0)
        return True
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def _acquire_singleton():
    """Exactly-one-owner lock, with two robustness upgrades over plain flock:

    1. STALE-OWNER RECOVERY: if the lock is contended but its recorded owner
       pid is dead (a SIGKILLed bridge's child can keep the inherited flock on
       the old inode alive), unlink the file — the next open makes a fresh
       inode whose flock is free — and retry. No crash-looping behind a ghost.
    2. SYSTEMD STANDBY: contended by a LIVE owner while we are the systemd
       instance (INVOCATION_ID set) → WAIT quietly and poll instead of dying;
       a debugging real_run.sh can take the hardware over, and this instance
       resumes ownership automatically the moment that run exits. A MANUAL
       second start still fails loudly (exit 3) — two live owners is never ok.
    """
    fd, other = _try_lock()
    if fd is None and (not other or not _owner_alive(other)):
        sys.stderr.write(f"[hardware_bridge] stale lock (dead owner "
                         f"{other or 'unknown'}) — recovering.\n")
        try:
            os.unlink(_LOCK_PATH)
        except OSError:
            pass
        fd, other = _try_lock()
    _standby_env = os.environ.get("MABEL_HW_STANDBY")
    _standby = (_standby_env == "1"
                or (_standby_env is None and bool(os.environ.get("INVOCATION_ID"))))
    if fd is None and _standby:
        sys.stderr.write(f"[hardware_bridge] live owner (pid {other or '?'}) "
                         f"holds the hardware — systemd instance STANDING BY "
                         f"(polling every 3 s; resumes when it exits).\n")
        while fd is None:
            time.sleep(3.0)
            fd, other = _try_lock()
            if fd is None and (not other or not _owner_alive(other)):
                try:
                    os.unlink(_LOCK_PATH)
                except OSError:
                    pass
        sys.stderr.write("[hardware_bridge] lock released — systemd instance "
                         "taking ownership back.\n")
    if fd is None:
        sys.stderr.write(
            f"[hardware_bridge] ANOTHER bridge already owns the hardware "
            f"(lock {_LOCK_PATH}{', pid ' + other if other else ''}). "
            f"Refusing to start — a second owner would make the base oscillate.\n")
        sys.exit(3)
    atexit.register(lambda: (fcntl.flock(fd, fcntl.LOCK_UN), fd.close()))
    return fd
