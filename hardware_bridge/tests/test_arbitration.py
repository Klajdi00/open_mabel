"""Lease semantics — the in-process single-writer guarantee.

The invariants under test are safety-critical: exactly one writer per actuator
group, a crashed writer cannot wedge its group (ttl), a stolen lease locks the
old writer out atomically, and NOTHING here can ever gate an estop (estop is
lease-free by design — asserted at the facade layer, but the broker must at
least never be needed for it).
"""
import threading
import time

import pytest

from mabel_hw.arbitration import GROUPS, LeaseBroker
from mabel_hw.errors import GroupBusy, NotLeaseHolder


@pytest.fixture()
def broker():
    return LeaseBroker()


def test_claim_release_cycle(broker):
    lease = broker.claim("base", "teleop")
    assert lease.active
    info = broker.holder("base")
    assert info.owner == "teleop" and info.token == lease.token
    lease.release()
    assert not lease.active
    assert broker.holder("base") is None
    lease.release()                       # idempotent


def test_second_claim_rejected(broker):
    broker.claim("left_arm", "dds")
    with pytest.raises(GroupBusy) as e:
        broker.claim("left_arm", "script")
    assert "dds" in str(e.value)


def test_groups_are_independent(broker):
    broker.claim("left_arm", "a")
    broker.claim("right_arm", "b")        # different group: fine
    snap = broker.snapshot()
    assert snap["left_arm"].owner == "a"
    assert snap["right_arm"].owner == "b"
    assert snap["base"] is None
    assert set(snap) == set(GROUPS)


def test_unknown_group_rejected(broker):
    with pytest.raises(KeyError):
        broker.claim("wings", "icarus")


def test_context_manager_releases(broker):
    with broker.claim("lift", "gui") as lease:
        assert lease.active
        with pytest.raises(GroupBusy):
            broker.claim("lift", "other")
    assert broker.holder("lift") is None


def test_write_gate_check(broker):
    lease = broker.claim("torso", "ctl")
    broker.check("torso", lease.token)    # holder writes fine
    with pytest.raises(NotLeaseHolder):
        broker.check("torso", lease.token + 999)
    lease.release()
    with pytest.raises(NotLeaseHolder):   # released token cannot write
        broker.check("torso", lease.token)


def test_steal_locks_out_previous_writer(broker):
    old = broker.claim("base", "teleop")
    new = broker.claim("base", "estop-recovery", steal=True)
    assert not old.active and new.active
    with pytest.raises(NotLeaseHolder):
        broker.check("base", old.token)   # the old writer's next write fails
    broker.check("base", new.token)


def test_ttl_expiry_frees_group():
    t = [0.0]
    broker = LeaseBroker(clock=lambda: t[0])
    lease = broker.claim("neck", "flaky", ttl=1.0)
    t[0] = 0.5
    broker.check("neck", lease.token)     # write touches the lease
    t[0] = 1.4                            # 0.9s idle: still live
    assert broker.holder("neck").owner == "flaky"
    t[0] = 2.6                            # 1.1s idle: expired
    assert broker.holder("neck") is None
    with pytest.raises(NotLeaseHolder):
        broker.check("neck", lease.token)
    assert broker.claim("neck", "next").active   # group is free again


def test_no_ttl_never_expires():
    t = [0.0]
    broker = LeaseBroker(clock=lambda: t[0])
    lease = broker.claim("lift", "steady")       # ttl=None
    t[0] = 1e6
    assert broker.holder("lift").token == lease.token


def test_wait_blocks_until_release(broker):
    first = broker.claim("base", "one")
    got = {}

    def waiter():
        got["lease"] = broker.claim("base", "two", wait=5.0)

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.15)
    assert "lease" not in got             # still blocked
    first.release()
    th.join(timeout=5.0)
    assert not th.is_alive()
    assert got["lease"].active and got["lease"].owner == "two"


def test_wait_times_out(broker):
    broker.claim("base", "one")
    t0 = time.monotonic()
    with pytest.raises(GroupBusy):
        broker.claim("base", "two", wait=0.2)
    assert 0.15 < time.monotonic() - t0 < 2.0


def test_thread_hammering_exactly_one_winner(broker):
    """64 threads race to claim + write; exactly one lease may be live at any
    moment, and every successful check() must belong to the current holder."""
    errors = []
    winners = []

    def contender(i):
        try:
            lease = broker.claim("right_hand", f"t{i}", wait=5.0)
            broker.check("right_hand", lease.token)
            winners.append(i)
            time.sleep(0.001)
            snap = broker.holder("right_hand")
            if snap is None or snap.token != lease.token:
                errors.append(f"t{i}: lease mutated while held")
            lease.release()
        except Exception as e:            # noqa: BLE001 — record, don't die
            errors.append(f"t{i}: {e}")

    threads = [threading.Thread(target=contender, args=(i,)) for i in range(64)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30.0)
    assert not errors, errors[:5]
    assert sorted(winners) == list(range(64))     # everyone eventually got a turn
    assert broker.holder("right_hand") is None
