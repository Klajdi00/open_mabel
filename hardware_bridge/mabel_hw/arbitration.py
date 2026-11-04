"""arbitration.py — exactly-one-writer-per-actuator-group leases.

Why this exists: two commanders interleaving on one actuator group is the classic
MABEL failure ("the base won't move" = two publishers on /mabel_cmd, idle zeros
winning). Single-writer is enforced at three scopes:

  1. process  — the flock singleton (mabel_hw.singleton): one device-owning
                process per host;
  2. writer   — THIS module: inside that process, exactly one live writer object
                per actuator group, however many components share the facade;
  3. DDS      — the ROS adapters count publishers on each command topic and log
                ERROR on >1 (detection for the out-of-process analogue).

Rules:
  * Reads are always free — leases gate WRITES only.
  * ``claim()`` is exclusive per group. A busy group raises :class:`GroupBusy`
    unless ``wait=`` (block up to that many seconds for a release) or
    ``steal=True`` (take over NOW; both owners are logged loudly).
  * A lease with ``ttl=`` expires if not touched (every write/renew touches it),
    so a crashed writer thread cannot wedge its group forever.
  * ``estop()``/``stop()`` NEVER require a lease — safety cannot be write-locked.
  * ``release()`` is idempotent; :class:`Lease` is a context manager.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence

from .errors import GroupBusy, NotLeaseHolder

__all__ = ["GROUPS", "LeaseInfo", "Lease", "LeaseBroker",
           "GroupBusy", "NotLeaseHolder"]

# Canonical actuator groups (matches the RobotState fields that accept commands).
GROUPS = ("left_arm", "right_arm", "neck", "torso", "base", "lift",
          "left_hand", "right_hand", "speaker")

_LOG = logging.getLogger("mabel_hw.arbitration")


@dataclass(frozen=True)
class LeaseInfo:
    """Read-only view of a group's current lease (for /claims introspection)."""
    group: str
    owner: str
    token: int
    acquired_at: float
    ttl: Optional[float]
    last_write: float


class Lease:
    """Held by exactly one writer of one group. Mutating subsystem methods hang
    off the Writer built around this, so 'writing without the lease' is
    unrepresentable in the public API. Context manager; release is idempotent."""

    def __init__(self, broker: "LeaseBroker", group: str, owner: str, token: int):
        self._broker = broker
        self.group = group
        self.owner = owner
        self.token = token

    @property
    def active(self) -> bool:
        info = self._broker.holder(self.group)
        return info is not None and info.token == self.token

    def renew(self) -> None:
        """Touch the lease (resets the ttl clock). Raises NotLeaseHolder if lost."""
        self._broker.check(self.group, self.token)

    def release(self) -> None:
        self._broker.release(self.group, self.token)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "active" if self.active else "inactive"
        return f"<Lease {self.group} owner={self.owner!r} token={self.token} {state}>"


class LeaseBroker:
    """Thread-safe registry: group -> at most one live lease."""

    def __init__(self, groups: Sequence[str] = GROUPS, *, logger=None,
                 clock=time.monotonic):
        self._groups = tuple(groups)
        self._log = logger or _LOG
        self._clock = clock
        self._cond = threading.Condition()
        self._leases: Dict[str, LeaseInfo] = {}
        self._tokens = itertools.count(1)

    # ── internals (call with self._cond held) ─────────────────────────────────

    def _live(self, group: str) -> Optional[LeaseInfo]:
        """Current lease if it exists and has not ttl-expired (expiry is lazy)."""
        info = self._leases.get(group)
        if info is None:
            return None
        if info.ttl is not None and (self._clock() - info.last_write) > info.ttl:
            self._log.warning(
                "lease on %r by %r expired (ttl %.2fs, idle %.2fs) — group is free",
                group, info.owner, info.ttl, self._clock() - info.last_write)
            del self._leases[group]
            self._cond.notify_all()
            return None
        return info

    def _check_group(self, group: str) -> None:
        if group not in self._groups:
            raise KeyError(f"unknown actuator group {group!r}; "
                           f"known: {', '.join(self._groups)}")

    # ── public ────────────────────────────────────────────────────────────────

    def claim(self, group: str, owner: str, *, wait: float = 0.0,
              steal: bool = False, ttl: Optional[float] = None) -> Lease:
        """Acquire the exclusive write lease for ``group``.

        wait  — seconds to block for a busy group to free up (0 = fail fast).
        steal — take over a live lease NOW (logged loudly with both owners).
                The previous holder's writes start raising NotLeaseHolder.
        ttl   — seconds of write-inactivity after which the lease auto-expires.
        """
        self._check_group(group)
        deadline = self._clock() + max(0.0, wait)
        with self._cond:
            while True:
                current = self._live(group)
                if current is None or steal:
                    if current is not None:
                        self._log.warning(
                            "lease on %r STOLEN from %r by %r — the previous "
                            "writer is now locked out", group, current.owner, owner)
                    now = self._clock()
                    info = LeaseInfo(group=group, owner=owner,
                                     token=next(self._tokens),
                                     acquired_at=now, ttl=ttl, last_write=now)
                    self._leases[group] = info
                    self._log.info("lease on %r -> %r (token %d%s)", group, owner,
                                   info.token,
                                   f", ttl {ttl:g}s" if ttl is not None else "")
                    return Lease(self, group, owner, info.token)
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise GroupBusy(
                        f"actuator group {group!r} is held by {current.owner!r} "
                        f"(since {self._clock() - current.acquired_at:.1f}s ago); "
                        f"pass wait= to block or steal=True to take over")
                self._cond.wait(timeout=min(remaining, 0.1))

    def check(self, group: str, token: int) -> None:
        """Gate a write: raise NotLeaseHolder unless ``token`` currently holds
        ``group``; on success, touch last_write (feeds the ttl)."""
        with self._cond:
            info = self._live(group)
            if info is None or info.token != token:
                holder = f"held by {info.owner!r}" if info else "unclaimed"
                raise NotLeaseHolder(
                    f"write to {group!r} rejected: lease token {token} is not "
                    f"current ({holder}) — command NOT sent to hardware")
            self._leases[group] = replace(info, last_write=self._clock())

    def release(self, group: str, token: int) -> None:
        """Idempotent: releasing a lease you no longer hold is a no-op."""
        with self._cond:
            info = self._leases.get(group)
            if info is not None and info.token == token:
                del self._leases[group]
                self._log.info("lease on %r released by %r", group, info.owner)
                self._cond.notify_all()

    def holder(self, group: str) -> Optional[LeaseInfo]:
        self._check_group(group)
        with self._cond:
            return self._live(group)

    def snapshot(self) -> Dict[str, Optional[LeaseInfo]]:
        """{group: LeaseInfo|None} for every known group (introspection/GUI)."""
        with self._cond:
            return {g: self._live(g) for g in self._groups}
