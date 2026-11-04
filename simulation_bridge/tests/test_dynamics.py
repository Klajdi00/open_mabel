"""simulation_bridge.dynamics: SceneDynamics against real dynamic scenes.

Verifies the pieces that make the extended scene library's actors move
naturally — path following, corner rounding, gait coupling, speed envelopes
(traffic-light stop cycles + robot yielding), belt carrying + part recycling —
plus the invariant that drives everything: it never NaNs the world.
"""
import math
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENES = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes")
CITY = os.path.join(SCENES, "mabel_city_sidewalk.xml")
ASSEMBLY = os.path.join(SCENES, "mabel_assembly_line.xml")
INTERSECTION = os.path.join(SCENES, "mabel_intersection.xml")

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(not os.path.isfile(CITY),
                                reason="dynamic scenes not generated")

from simulation_bridge import dynamics as D          # noqa: E402


def _load(path):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    dyn = D.SceneDynamics.load(path, m)
    return m, d, dyn


def _drive(m, d, dyn, seconds):
    n = int(seconds / float(m.opt.timestep))
    for _ in range(n):
        dyn.step(m, d)
        mujoco.mj_step(m, d)


# ── pure path math ────────────────────────────────────────────────────────────
def test_path_probe_endpoints():
    p = D._Path([[0, 0], [10, 0], [10, 5]], closed=False)
    assert p.total == pytest.approx(15.0)
    x, y, yaw = p.probe(0.0)
    assert (x, y) == pytest.approx((0.0, 0.0))
    assert yaw == pytest.approx(0.0)
    x, y, yaw = p.probe(12.5)
    assert (x, y) == pytest.approx((10.0, 2.5))
    assert yaw == pytest.approx(math.pi / 2)


def test_corner_rounding_shortens_and_smooths():
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    sharp = D._Path(pts, closed=True)
    filleted = D._Path(D._round_corners(pts, True, 2.5), closed=True)
    # fillets cut the corners, so the rounded loop is strictly shorter
    assert filleted.total < sharp.total
    # and it has more vertices (arc samples inserted)
    assert len(filleted.pts) > len(sharp.pts)


# ── actor motion ──────────────────────────────────────────────────────────────
def test_walker_advances_and_swings_gait():
    m, d, dyn = _load(CITY)
    ped = next(a for a in dyn.actors if a.kind == "walker")
    a = ped.qadr
    p0 = d.qpos[a:a + 2].copy()
    hip_amp = 0.0
    n = int(6.0 / float(m.opt.timestep))
    for _ in range(n):
        dyn.step(m, d)
        mujoco.mj_step(m, d)
        if ped.gait:
            hip_amp = max(hip_amp, abs(float(d.qpos[ped.gait[0][0]])))
    moved = float(np.hypot(*(d.qpos[a:a + 2] - p0)))
    assert moved > 2.0, "walker should cover ground"
    assert hip_amp > 0.1, "hips should swing while walking"


def test_vehicle_wheels_spin_with_travel():
    m, d, dyn = _load(CITY)
    car = next(a for a in dyn.actors if a.kind == "vehicle")
    a = car.qadr
    p0 = d.qpos[a:a + 2].copy()
    _drive(m, d, dyn, 5.0)
    moved = float(np.hypot(*(d.qpos[a:a + 2] - p0)))
    assert moved > 5.0, "car should drive"
    # wheel angle ~ distance / radius (kinematic rolling, within a turn or two)
    wq = float(d.qpos[car.wheels[0][0]])
    assert abs(wq) > moved / car.wheel_r * 0.5


def test_actors_never_nan():
    for scene in (CITY, INTERSECTION):
        m, d, dyn = _load(scene)
        _drive(m, d, dyn, 12.0)
        assert not np.isnan(d.qpos).any(), f"{scene} NaN'd qpos"
        assert not np.isnan(d.qacc).any(), f"{scene} NaN'd qacc"


# ── speed envelopes ───────────────────────────────────────────────────────────
def test_traffic_light_stop_cycle_halts_then_releases():
    """A red stop-line ahead brings v to ~0; when it clears, v recovers."""
    m = mujoco.MjModel.from_xml_path(INTERSECTION)
    d = mujoco.MjData(m)
    dyn = D.SceneDynamics.load(INTERSECTION, m)
    car = next(a for a in dyn.actors
               if a.kind == "vehicle" and a.stops)
    saw_stop = saw_go = False
    n = int(40.0 / float(m.opt.timestep))          # > one 28 s signal cycle
    for _ in range(n):
        dyn.step(m, d)
        mujoco.mj_step(m, d)
        if car.v < 0.3:
            saw_stop = True
        if car.v > 3.0:
            saw_go = True
    assert saw_stop, "vehicle should stop at the red"
    assert saw_go, "vehicle should proceed when green"


def _empty_field():
    f = D._ObstacleField.__new__(D._ObstacleField)
    f.x = f.y = f.r = np.zeros(0)
    return f


def test_robot_in_path_makes_walker_yield():
    m, d, dyn = _load(CITY)
    ped = next(a for a in dyn.actors if a.kind == "walker" and a.yield_dist > 0)
    field = _empty_field()
    x, y, yaw = ped.path.probe(ped.s)
    hd = (math.cos(yaw), math.sin(yaw))
    base = ped._v_target(0.0, None, x, y, hd, field, [])     # un-obstructed
    # a virtual robot ~0.6 m directly ahead along the heading → slow down
    robot = (x + 0.6 * hd[0], y + 0.6 * hd[1])
    assert ped._v_target(0.0, robot, x, y, hd, field, []) < base
    # a robot off to the side does not
    side = (x - hd[1] * 3.0, y + hd[0] * 3.0)
    assert ped._v_target(0.0, side, x, y, hd, field, []) == pytest.approx(base)


def test_robot_ref_dodge_only_when_stationary():
    """People avoid the robot ONLY when it's stationary; a moving robot can hit
    them. Cars never avoid the robot at all."""
    m, d, dyn = _load(CITY)
    ped = next(a for a in dyn.actors if a.kind == "walker")
    car = next(a for a in dyn.actors if a.kind == "vehicle")
    R = (1.0, 2.0)
    assert ped._robot_ref(R, robot_moving=False) == R    # stationary → dodge
    assert ped._robot_ref(R, robot_moving=True) is None   # moving → collide
    assert car._robot_ref(R, robot_moving=False) is None  # car never dodges
    assert car._robot_ref(R, robot_moving=True) is None
    assert car.yield_dist == 0.0                          # car robot-brake off


def test_walker_dodges_stationary_robot():
    m, d, dyn = _load(CITY)
    ped = next(a for a in dyn.actors if a.kind == "walker")
    j = m.joint("base_root")
    ba, bv = int(j.qposadr[0]), int(j.dofadr[0])
    ch = m.body("mabel_chassis").id
    x, y, yaw = ped.path.probe(ped.s)
    hd = (math.cos(yaw), math.sin(yaw))
    rx, ry = x + hd[0] * 4.0, y + hd[1] * 4.0        # plant it ahead, stationary
    worst = 1e9
    for _ in range(int(6.0 / float(m.opt.timestep))):
        d.qpos[ba:ba + 2] = (rx, ry); d.qvel[bv:bv + 2] = 0.0
        mujoco.mj_forward(m, d)
        dyn.step(m, d); mujoco.mj_step(m, d)
        rc = d.xpos[ch]
        worst = min(worst, math.hypot(d.qpos[ped.qadr] - rc[0],
                                      d.qpos[ped.qadr + 1] - rc[1]))
    assert worst > 0.4, f"walker should keep clear of a parked robot ({worst:.2f} m)"


def test_car_hits_and_pushes_robot():
    """A car that finds the robot on the road shoves it through REAL MuJoCo
    contacts: the scene declares an explicit <contact><pair> between the car
    body and the robot's `chassis_bumper` proxy, so the kinematically-driven
    car imparts a genuine contact force. Assert real contacts form, the robot
    is displaced, and the world never NaNs."""
    m, d, dyn = _load(CITY)
    car = next(a for a in dyn.actors if a.kind == "vehicle")
    j = m.joint("base_root")
    ba = int(j.qposadr[0])
    ch = m.body("mabel_chassis").id
    bumper = next(g for g in range(m.ngeom)
                  if (m.geom(g).name or "") == "chassis_bumper")
    cx, cy, _ = car.path.probe((car.s + car.path.total * 0.25) % car.path.total)
    d.qpos[ba], d.qpos[ba + 1] = cx, cy          # park the robot on the car's lane
    mujoco.mj_forward(m, d)
    start = d.xpos[ch][:2].copy()
    pushed = 0.0
    contacts = 0
    for _ in range(int(8.0 / float(m.opt.timestep))):
        dyn.step(m, d)
        mujoco.mj_step(m, d)
        pushed = max(pushed, float(np.hypot(d.xpos[ch][0] - start[0],
                                            d.xpos[ch][1] - start[1])))
        contacts += sum(1 for c in range(d.ncon)
                        if bumper in (d.contact[c].geom1, d.contact[c].geom2))
    assert contacts > 0, "no real contact ever formed between a car and the robot"
    assert pushed > 0.12, f"the car did not push the robot ({pushed:.2f} m)"
    assert not np.isnan(d.qpos).any()


def test_walker_patience_eventually_pushes_through():
    m, d, dyn = _load(CITY)
    ped = next(a for a in dyn.actors
               if a.kind == "walker" and a.yield_dist > 0 and a.patience)
    field = _empty_field()
    x, y, yaw = ped.path.probe(ped.s)
    hd = (math.cos(yaw), math.sin(yaw))
    base = ped._v_target(0.0, None, x, y, hd, field, [])
    robot = (x + 0.6 * hd[0], y + 0.6 * hd[1])
    ped.blocked = ped.patience + 1.0            # waited past patience → push on
    assert ped._v_target(0.0, robot, x, y, hd, field, []) == pytest.approx(base)


# ── avoidance + collision wiring ──────────────────────────────────────────────
def test_walkers_keep_separation():
    """Moving pedestrians steer/space so they never overlap each other."""
    m, d, dyn = _load(CITY)
    walkers = [a for a in dyn.actors if a.kind == "walker"]
    worst = 1e9
    for _ in range(int(12.0 / float(m.opt.timestep))):
        dyn.step(m, d)
        mujoco.mj_step(m, d)
    pts = np.array([[d.qpos[a.qadr], d.qpos[a.qadr + 1]] for a in walkers])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            worst = min(worst, float(np.hypot(*(pts[i] - pts[j]))))
    assert worst > 0.5, f"pedestrians overlapped (min sep {worst:.2f} m)"


def test_walkers_clear_compact_obstacles():
    m, d, dyn = _load(CITY)
    walkers = [a for a in dyn.actors if a.kind == "walker"]
    worst = 1e9
    for i in range(int(12.0 / float(m.opt.timestep))):
        dyn.step(m, d)
        mujoco.mj_step(m, d)
        if i % 50 == 0 and len(dyn.field.x):
            for a in walkers:
                cl = float((np.hypot(d.qpos[a.qadr] - dyn.field.x,
                                     d.qpos[a.qadr + 1] - dyn.field.y)
                            - dyn.field.r).min())
                worst = min(worst, cl)
    assert worst > -0.15, f"walker entered a compact obstacle ({worst:.2f} m)"


def test_robot_collision_wiring():
    """Actors collide with the robot ONLY through explicit <contact><pair>
    entries against the base proxy (MuJoCo's mask-based broadphase drops the
    actor↔robot pairs in these scenes). The proxy is mask-inert, and actor geoms
    hit nothing via masks — so actors never fight the scene or each other."""
    m, d, dyn = _load(CITY)
    bumper = next(g for g in range(m.ngeom)
                  if (m.geom(g).name or "") == "chassis_bumper")
    # proxy collides with NOTHING via masks — participates only via explicit pairs
    assert int(m.geom_contype[bumper]) == 0 and int(m.geom_conaffinity[bumper]) == 0
    # every actor is explicitly paired with the proxy, and every pair involves it
    assert m.npair > 0, "no explicit robot↔actor contact pairs were generated"
    paired = set()
    for p in range(m.npair):
        g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        assert bumper in (g1, g2), "a contact pair does not involve the robot proxy"
        paired.add(g1 if g2 == bumper else g2)
    assert any((m.geom(g).name or "").endswith("_body") for g in paired), \
        "no vehicle is paired with the robot"
    assert any((m.geom(g).name or "").endswith("_hit") for g in paired), \
        "no pedestrian is paired with the robot"
    # actor hit-geoms hit nothing via masks (contype 2 / conaffinity 0)
    hit = next(g for g in range(m.ngeom) if (m.geom(g).name or "").endswith("_hit"))
    assert int(m.geom_contype[hit]) == 2 and int(m.geom_conaffinity[hit]) == 0
    # two actor hit-capsules must NOT be allowed to collide with each other
    hits = [g for g in range(m.ngeom) if (m.geom(g).name or "").endswith("_hit")]
    a, b = hits[0], hits[1]
    allowed = (int(m.geom_contype[a]) & int(m.geom_conaffinity[b])) or \
              (int(m.geom_contype[b]) & int(m.geom_conaffinity[a]))
    assert not allowed, "actors should not physically collide with each other"


# ── conveyor ─────────────────────────────────────────────────────────────────
def test_belt_carries_parts_and_recycles():
    m, d, dyn = _load(ASSEMBLY)
    assert dyn.belts, "assembly line should have a conveyor"
    belt = dyn.belts[0]
    part_adr = belt.parts[0][0]
    x0 = float(d.qpos[part_adr])
    _drive(m, d, dyn, 8.0)
    x1 = float(d.qpos[part_adr])
    # 8 s @ 0.22 m/s ≈ 1.76 m of carry (unless it wrapped, still != start)
    assert abs(x1 - x0) > 0.5, "belt should carry the part downstream"
    # belt slide stays within its wrap band (never runs away)
    assert abs(float(d.qpos[belt.qadr])) <= belt.wrap


def test_belt_recycle_keeps_parts_on_belt():
    m, d, dyn = _load(ASSEMBLY)
    belt = dyn.belts[0]
    _drive(m, d, dyn, 30.0)                      # long enough to recycle
    for adr, _, _ in belt.parts:
        assert d.qpos[adr] <= belt.limit + 0.2, "recycled part left the belt"


# ── reset ────────────────────────────────────────────────────────────────────
def test_reset_returns_actors_to_start():
    m, d, dyn = _load(CITY)
    starts = [a.s0 % a.path.total if a.path.closed else min(a.s0, a.path.total)
              for a in dyn.actors]
    _drive(m, d, dyn, 5.0)
    dyn.reset()
    for a, s0 in zip(dyn.actors, starts):
        assert a.s == pytest.approx(s0)
        assert a.v == 0.0
