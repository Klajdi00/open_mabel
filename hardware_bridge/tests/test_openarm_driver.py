"""OpenArmLink / OpenArmGroup against the MockOpenArmTeensy — full loopback of
the openarm-dual wire protocol: Hello->Manifest self-description, left/right
slice merging on ONE link, the t_max clamp, damping-hold semantics, hold-at-
measured on engage, and hot-join. No hardware, no ROS.

These cover the path that ACTUALLY drives the robot's arms.
"""
import time

import pytest

from mabel_hw.drivers.openarm_link import OpenArmLink

from fakes.mock_openarm_teensy import MockOpenArmTeensy

LEFT = [f"l_j{i}" for i in range(1, 8)]
RIGHT = [f"r_j{i}" for i in range(1, 8)]


def _wait(pred, timeout=5.0, dt=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture()
def mock():
    m = MockOpenArmTeensy().start()
    yield m
    m.stop()


@pytest.fixture()
def link(mock):
    lk = OpenArmLink("127.0.0.1", mock.port, stream_hz=200)
    lk.connect()
    assert _wait(lambda: len(lk.specs()) == mock.n), "no manifest"
    yield lk
    lk.close()


# ── self-description ────────────────────────────────────────────────────────

def test_manifest_is_learned_not_hardcoded(link, mock):
    specs = link.specs()
    assert [s.name for s in specs] == mock.names
    # per-joint rated torque comes from the FIRMWARE, not a constant in our code
    by = {s.name: s for s in specs}
    assert by["l_j1"].motor == "DM8009" and by["l_j1"].t_max == pytest.approx(54.0)
    assert by["l_j5"].motor == "DM4310" and by["l_j5"].t_max == pytest.approx(10.0)
    assert by["r_j1"].bus == 1 and by["l_j1"].bus == 2


def test_connect_does_not_energize(link, mock):
    """The whole reason the arms may default enabled: connecting must send NO
    command. An idle link leaves the motors in the firmware's own damping hold."""
    time.sleep(0.2)
    assert mock.cmd_count == 0
    assert mock.hello_count > 0            # ...but the keepalive IS flowing


# ── group resolution + hot-join ─────────────────────────────────────────────

def test_group_resolution_and_unknown_names(link):
    g = link.group(LEFT)
    assert g.resolved
    assert g.names == tuple(LEFT)
    g.set_torque(True)
    with pytest.raises(KeyError):
        g.set_targets({"nope": 0.0})


def test_group_created_before_board_answers_hot_joins():
    """A group may exist before the Teensy is up: unresolved, reads dead, then
    binds itself the moment the manifest arrives — no bridge restart."""
    m = MockOpenArmTeensy()                       # bound, NOT started
    lk = OpenArmLink("127.0.0.1", m.port, stream_hz=200)
    try:
        g = lk.group(LEFT)
        assert not g.resolved
        assert not g.connected()
        assert all(v is False for v in g.read().live)
        lk.connect()
        m.start()                                 # board appears late
        assert _wait(lambda: (g.service() or g.resolved)), "never hot-joined"
        assert _wait(g.connected)
    finally:
        lk.close()
        m.stop()


# ── read side ───────────────────────────────────────────────────────────────

def test_read_reports_raw_manifest_values(link, mock):
    g = link.group(LEFT)
    mock.set_pos(l_j4=1.234, l_j1=-0.5)
    assert _wait(lambda: g.read().position[3] == pytest.approx(1.234, abs=1e-4))
    fb = g.read()
    assert fb.names == tuple(LEFT)
    assert fb.position[0] == pytest.approx(-0.5, abs=1e-4)
    # Per-motor liveness needs a LIVE signal (temp/vel/torque — position alone is
    # excluded, a dead motor streams a frozen pos). The mock's temps ride the
    # Health reply, which the link polls on its own cadence — so liveness comes
    # up a beat after the first State. Wait for it instead of racing it.
    assert _wait(lambda: all(g.read().live), timeout=6.0), \
        "motors never read live (health/temps pending)"


def test_health_reports_temperature_from_health_message(link, mock):
    g = link.group(LEFT)
    assert _wait(lambda: g.read().temperature[0] > 0, timeout=6.0)
    h = g.health()
    assert set(h) == set(LEFT)
    assert h["l_j1"]["tempC"] == pytest.approx(35.0)


# ── write side ──────────────────────────────────────────────────────────────

def test_two_groups_merge_one_command_vector(link, mock):
    """left and right share ONE socket and ONE manifest-indexed command vector:
    each writes only its own slice and must not clobber the other's."""
    gl, gr = link.group(LEFT), link.group(RIGHT)
    gl.set_torque(True)
    gr.set_torque(True)
    gl.set_targets({"l_j1": 0.4})
    gr.set_targets({"r_j1": -0.4})
    assert _wait(lambda: mock.last_cmd is not None and mock.cmd_count > 3)
    cmd = mock.last_cmd
    assert len(cmd["pos"]) == mock.n                 # FULL manifest length
    assert cmd["pos"][mock.names.index("l_j1")] == pytest.approx(0.4)
    assert cmd["pos"][mock.names.index("r_j1")] == pytest.approx(-0.4)
    # both slices are position-controlled (kp>0), nothing was zeroed by the other
    assert cmd["kp"][mock.names.index("l_j1")] > 0
    assert cmd["kp"][mock.names.index("r_j1")] > 0


def test_targets_track_and_settle(link, mock):
    g = link.group(LEFT)
    g.set_torque(True)
    g.set_targets({"l_j2": 0.75})
    assert _wait(lambda: g.read().position[1] == pytest.approx(0.75, abs=1e-3))


def test_torque_off_reverts_to_damping_hold(link, mock):
    g = link.group(LEFT)
    g.set_torque(True)
    g.set_targets({"l_j1": 0.6})
    assert _wait(lambda: mock.cmd_count > 3)
    g.set_torque(False)
    assert not g.is_torque_enabled
    # keeps STREAMING (so off is immediate) but with no position term
    n0 = mock.cmd_count
    assert _wait(lambda: mock.cmd_count > n0 + 3)
    i = mock.names.index("l_j1")
    assert mock.last_cmd["kp"][i] == 0.0
    assert mock.last_cmd["t_ff"][i] == 0.0
    assert mock.last_cmd["kd"][i] > 0                # damping stays


def test_engage_holds_at_measured_pose_not_zero(link, mock):
    """Hold-then-enable: torque-on must latch the MEASURED angle, or the engage
    itself throws the arm to 0. Regression for the 91-deg l_j5 snap."""
    g = link.group(LEFT)
    mock.set_pos(l_j5=1.4)
    assert _wait(lambda: g.read().position[4] == pytest.approx(1.4, abs=1e-4))
    g.set_torque(True)
    assert _wait(lambda: mock.cmd_count > 3)
    i = mock.names.index("l_j5")
    assert mock.last_cmd["pos"][i] == pytest.approx(1.4, abs=1e-3)


def test_link_reset_limps_group_until_recommanded(link, mock):
    """A board that comes back (reboot/replug) must never adopt stale targets."""
    g = link.group(LEFT)
    g.set_torque(True)
    g.set_targets({"l_j1": 0.9})
    assert _wait(lambda: mock.cmd_count > 3)
    g._on_link_reset()
    assert not g.is_torque_enabled


def test_routine_manifest_does_not_limp_the_group(link, mock):
    """REGRESSION: the firmware answers EVERY hello keepalive (~10 Hz) with a
    manifest. Treating each one as a link reset limped the group 10x/second and
    silently dropped every command — torque read 'on' at the API while nothing
    moved. Only a real layout CHANGE is a reset."""
    g = link.group(LEFT)
    g.set_torque(True)
    g.set_targets({"l_j1": 0.5})
    hellos = mock.hello_count
    assert _wait(lambda: mock.hello_count > hellos + 5)   # keepalives flowing...
    assert g.is_torque_enabled, "a routine manifest limped the group"
    assert g.read().position[0] == pytest.approx(0.5, abs=1e-3), "command dropped"


def test_telemetry_gap_resets_groups_to_limp(link, mock):
    """The board stopping and returning IS a reset (it restarts limp and adopts
    whatever we stream). The manifest can't signal that, so the gap does."""
    g = link.group(LEFT)
    g.set_torque(True)
    g.set_targets({"l_j1": 0.5})
    assert _wait(lambda: mock.cmd_count > 3)
    mock.reboot()
    assert _wait(lambda: not g.is_torque_enabled), "stale target survived a gap"


def test_reboot_damps_the_WHOLE_vector_not_just_the_active_flag(link, mock):
    """REGRESSION (severe): after a reboot, re-enabling ONE arm must not drive the
    OTHER.

    The gap reset used to clear only the group's local gains + the link's `_active`
    flag, leaving the reset arm's slice armed (kp=30, stale pos) in the MERGED
    vector. The moment the sibling commanded, `_active` went true again and send()
    shipped every slice — so enabling the RIGHT arm drove the LEFT one, which the
    operator (and the API) believed was limp, back to a pre-reboot target it had
    since fallen away from: kp*error = 30 * 0.8 = 24 N.m into a 'limp' joint."""
    gl, gr = link.group(LEFT), link.group(RIGHT)
    gl.set_torque(True)
    gl.set_targets({"l_j1": 0.8})
    assert _wait(lambda: mock.cmd_count > 5)
    mock.reboot()
    assert _wait(lambda: not gl.is_torque_enabled)
    mock.set_pos(l_j1=0.0)                    # the limp arm fell
    n0 = mock.cmd_count
    gr.set_torque(True)                       # ONLY the right arm is re-enabled
    gr.set_targets({"r_j1": 0.1})
    assert _wait(lambda: mock.cmd_count > n0 + 5)
    i = mock.names.index("l_j1")
    assert mock.last_cmd["kp"][i] == 0.0, "the limp left arm is being DRIVEN"
    assert mock.pos[i] == pytest.approx(0.0, abs=1e-3), "the limp left arm MOVED"


def test_engage_without_telemetry_is_refused_not_faked(link, mock):
    """Engaging means 'hold where the joint IS' — impossible with no measured pose.
    Refuse and stay limp rather than report torque-on while streaming an unlatched
    pos=0 (the adapter republishes torque_state, so the operator sees the refusal)."""
    g = link.group(LEFT)
    with link._lock:                          # the pre-first-STATE window
        link._state = None
    g.set_torque(True)
    assert not g.is_torque_enabled
    with g._lock:
        assert g._kp[0] == 0.0                # nothing armed


# ── the one limit this layer enforces ───────────────────────────────────────

def test_t_ff_is_clamped_to_the_motors_rated_torque(link, mock):
    """The ONLY torque limit in the HAL: t_ff saturates at the manifest's per-joint
    t_max. A wrist (10 N·m) and a shoulder (54 N·m) clamp differently."""
    g = link.group(LEFT)
    g.set_torque(True)
    g.command({"l_j5": 0.0, "l_j1": 0.0},
              effort={"l_j5": 999.0, "l_j1": 999.0})
    assert _wait(lambda: mock.cmd_count > 3)
    assert mock.last_cmd["t_ff"][mock.names.index("l_j5")] == pytest.approx(10.0)
    assert mock.last_cmd["t_ff"][mock.names.index("l_j1")] == pytest.approx(54.0)
    g.command({"l_j5": 0.0}, effort={"l_j5": -999.0})
    assert _wait(lambda: mock.last_cmd["t_ff"][mock.names.index("l_j5")]
                 == pytest.approx(-10.0))


def test_command_passes_gains_and_feedforward_through(link, mock):
    """The control law is the COMMANDER's: kp/kd/t_ff arrive from outside and are
    sent as-is (within t_max). The HAL never invents or overrides a gain."""
    g = link.group(LEFT)
    g.set_torque(True)
    g.command({"l_j3": 0.2}, velocity={"l_j3": 0.1}, kp={"l_j3": 12.0},
              kd={"l_j3": 0.8}, effort={"l_j3": 3.5})
    assert _wait(lambda: mock.cmd_count > 3)
    i = mock.names.index("l_j3")
    c = mock.last_cmd
    assert c["pos"][i] == pytest.approx(0.2)
    assert c["vel"][i] == pytest.approx(0.1)
    assert c["kp"][i] == pytest.approx(12.0)
    assert c["kd"][i] == pytest.approx(0.8)
    assert c["t_ff"][i] == pytest.approx(3.5)      # 3.5 < t_max(DM4340)=27


# ── the wire encoder ────────────────────────────────────────────────────────

def test_command_encoding_is_stable():
    """The Command datagram is the authoritative upstream `robot.proto` schema:
    f1 seq varint, f2..f6 PACKED repeated float. Pinned byte-for-byte."""
    body = OpenArmLink.encode_command(1, [0.0], [0.0], [0.0], [0.0], [0.0])
    assert body == (b"\x08\x01"                     # f1 seq = 1
                    + b"\x12\x04" + b"\x00" * 4     # f2 pos  (len-delim, 1 f32)
                    + b"\x1a\x04" + b"\x00" * 4     # f3 vel
                    + b"\x22\x04" + b"\x00" * 4     # f4 kp
                    + b"\x2a\x04" + b"\x00" * 4     # f5 kd
                    + b"\x32\x04" + b"\x00" * 4)    # f6 t_ff


def test_command_length_mismatch_is_rejected(link):
    with pytest.raises(ValueError):
        link.send_command([0.0, 0.0], [0.0], [0.0], [0.0], [0.0])
