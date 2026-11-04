"""
dynamics.py — SceneDynamics: scripted world actors for the simulation bridge.

Scenes may ship a sidecar next to their MJCF (written by
simulation/mabel_mujoco/scripts/tools/scenegen):

    scenes/mabel_<name>.xml
    scenes/mabel_<name>.dynamics.json      <- this module's input

The sidecar declares kinematic ACTORS (pedestrians walking polyline paths
with a limb gait, vehicles driving lanes with wheel spin and traffic-light
stop cycles) and CONVEYORS (slide-loop belts that carry free parts by real
friction, plus end-of-belt part recycling).

Design (why it works with zero extra plumbing):
  * Actors are free-jointed, gravity-compensated bodies in the scene XML —
    their pose lives in qpos, so the memmap mirror, cameras, web viewer and
    ROS all see the motion for free.
  * The physics loop calls step() every substep, but the controller only
    recomputes targets at KEY_DT (50 Hz); between keyframes mj_step
    integrates the written qvel, so motion stays smooth at 500 Hz for the
    price of a 50 Hz python loop.
  * Writing qpos+qvel makes actors kinematic: a collision shoves the robot,
    never derails the actor (a car has effectively infinite inertia).

Naturalness rules, all deterministic (no RNG):
  * speed envelopes, not teleports: v_target is limited by
    sqrt(2·a·(distance - hold)) toward red stop-lines (period/offset/dur
    cycles = traffic lights, bus dwells, window-shopping) and toward the
    ROBOT when it stands in the path corridor — cars brake for MABEL,
    pedestrians wait, then (patience) squeeze past;
  * corner rounding: polyline corners get arc fillets at load (0.35 m for
    walkers, 2.5 m for vehicles) so nothing pivots on the spot;
  * gait coupling: hip/shoulder swing and walk bob scale with v/cruise, so
    actors glide to a stop mid-stride instead of moonwalking; wheel spin
    integrates v/r.

Path modes: closed loop | open ping-pong (walkers turn around) |
open teleport (through-traffic re-enters at the start, ends fog-masked).
"""
from __future__ import annotations

import json
import logging
import math
import os

import numpy as np

log = logging.getLogger("simbridge.dynamics")

KEY_DT = 0.02                    # actor keyframe period (s of SIM time)
FILLET = {"walker": 0.35, "vehicle": 2.5}


def _round_corners(pts, closed, r):
    """Insert quadratic-bezier fillets of radius ~r at polyline corners."""
    if r <= 0 or len(pts) < 3:
        return [tuple(p) for p in pts]
    n = len(pts)
    out = []
    rng = range(n) if closed else range(1, n - 1)
    corner = set(rng)
    for i in range(n):
        p = np.asarray(pts[i], float)
        if i not in corner:
            out.append(tuple(p))
            continue
        a = np.asarray(pts[(i - 1) % n], float)
        b = np.asarray(pts[(i + 1) % n], float)
        la, lb = np.linalg.norm(p - a), np.linalg.norm(b - p)
        rr = min(r, 0.42 * la, 0.42 * lb)
        if rr < 0.05:
            out.append(tuple(p))
            continue
        pa = p + (a - p) / la * rr
        pb = p + (b - p) / lb * rr
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            q = (1 - t) ** 2 * pa + 2 * (1 - t) * t * p + t ** 2 * pb
            out.append(tuple(q))
    return out


class _Path:
    def __init__(self, pts, closed):
        self.closed = closed
        p = np.asarray(pts, float)
        if closed:
            p = np.vstack([p, p[:1]])
        seg = np.diff(p, axis=0)
        self.pts = p
        self.lens = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.lens)])
        self.total = float(self.cum[-1])

    def probe(self, s):
        """(x, y, yaw) at arclength s (caller keeps s in range)."""
        s = min(max(s, 0.0), self.total)
        i = int(np.searchsorted(self.cum, s, side="right")) - 1
        i = min(max(i, 0), len(self.lens) - 1)
        t = (s - self.cum[i]) / self.lens[i] if self.lens[i] > 0 else 0.0
        x = self.pts[i, 0] + t * (self.pts[i + 1, 0] - self.pts[i, 0])
        y = self.pts[i, 1] + t * (self.pts[i + 1, 1] - self.pts[i, 1])
        yaw = math.atan2(self.pts[i + 1, 1] - self.pts[i, 1],
                         self.pts[i + 1, 0] - self.pts[i, 0])
        return x, y, yaw


def _subtree_bodies(model, joint_name):
    """Body ids in the subtree rooted at `joint_name`'s body (empty if absent)."""
    if not joint_name:
        return set()
    try:
        root = int(model.jnt_bodyid[model.joint(joint_name).id])
    except Exception:
        return set()
    sub = {root}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in sub and int(model.body_parentid[b]) in sub:
                sub.add(b); changed = True
    return sub


def _geom_xy_footprint(model, data, g):
    """(cx, cy, r, z0, z1): the geom's world-frame XY footprint radius and
    z-extent. Uses the ACTUAL footprint per type — NOT geom_rbound, whose 3D
    bounding sphere turns a tall thin pole/trunk into a huge disc."""
    import mujoco
    t = model.geom_type[g]
    p = data.geom_xpos[g]
    R = data.geom_xmat[g].reshape(3, 3)
    s = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(s[0]); return p[0], p[1], r, p[2] - r, p[2] + r
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        hx = abs(R[0, 0]) * s[0] + abs(R[0, 1]) * s[1] + abs(R[0, 2]) * s[2]
        hy = abs(R[1, 0]) * s[0] + abs(R[1, 1]) * s[1] + abs(R[1, 2]) * s[2]
        hz = abs(R[2, 0]) * s[0] + abs(R[2, 1]) * s[1] + abs(R[2, 2]) * s[2]
        return p[0], p[1], math.hypot(hx, hy), p[2] - hz, p[2] + hz
    if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        # radius s[0], half-length s[1] along local z; project into world XY
        hx = abs(R[0, 2]) * s[1] + s[0]
        hy = abs(R[1, 2]) * s[1] + s[0]
        hz = abs(R[2, 2]) * s[1] + s[0]
        return p[0], p[1], math.hypot(hx, hy), p[2] - hz, p[2] + hz
    r = min(float(model.geom_rbound[g]), 1.0)          # ellipsoid/mesh/other
    return p[0], p[1], r, p[2] - r, p[2] + r


class _ObstacleField:
    """Static scene obstacles the actors steer around (furniture, trees, poles,
    tables — NOT walls/buildings, which the authored paths already avoid, and
    NOT the floor/robot/actors). Each is a footprint circle in the actors'
    standing z-band; `repulse` returns a unit push away from any that a query
    point is inside its clearance margin, plus the nearest edge distance."""

    Z_BAND = (0.15, 1.7)
    MAX_R = 1.6          # compact obstacles only (poles, trees, small furniture);
                         # long benches/counters/walls are poorly modelled as a
                         # circle, and the authored paths already route past them

    def __init__(self, model, data, robot_bodies, actor_bodies):
        import mujoco
        skip = robot_bodies | actor_bodies
        cx, cy, cr = [], [], []
        for g in range(model.ngeom):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if int(model.geom_bodyid[g]) in skip:
                continue
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue                                   # visual-only
            _, _, _, z0, z1 = _geom_xy_footprint(model, data, g)
            if z1 < self.Z_BAND[0] or z0 > self.Z_BAND[1]:
                continue                                   # under/over the actor
            for (x, y, r) in self._circles(model, data, g):
                if 0.03 < r <= self.MAX_R:
                    cx.append(x); cy.append(y); cr.append(r)
        self.x = np.array(cx); self.y = np.array(cy); self.r = np.array(cr)

    @staticmethod
    def _circles(model, data, g):
        """One or more footprint circles for a geom. A long box (bench, counter,
        low wall segment) is sampled into several small circles along its long
        axis so it is avoided as a thin barrier — not one giant disc that blocks
        a whole lane."""
        import mujoco
        p = data.geom_xpos[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            R = data.geom_xmat[g].reshape(3, 3)
            s = model.geom_size[g]
            # world XY extent of each local axis, pick the two dominant in XY
            axes = [(math.hypot(R[0, k] * s[k], R[1, k] * s[k]), k) for k in range(3)]
            axes.sort(reverse=True)
            (la, lk), (sa, sk) = axes[0], axes[1]
            # only thin, furniture-scale bars (benches, counters, low rails) —
            # NOT building-scale boxes (those exceed MAX_R and are dropped)
            if la > sa * 1.6 and 0.5 < la < 3.2 and sa < 0.6:
                norm = math.hypot(R[0, lk], R[1, lk]) or 1.0
                ux, uy = R[0, lk] / norm, R[1, lk] / norm  # unit XY direction
                n = min(6, max(2, int(la / max(sa, 0.2))))
                out = []
                for i in range(n + 1):
                    t = -la + 2 * la * i / n               # la = XY half-length
                    out.append((float(p[0] + ux * t), float(p[1] + uy * t),
                                float(max(sa, 0.12))))
                return out
        x, y, r, _, _ = _geom_xy_footprint(model, data, g)
        return [(float(x), float(y), float(r))]

    def repulse(self, x, y, margin):
        """(push_x, push_y, min_clear) at (x, y). push is the summed unit
        vector away from obstacles within `margin` of their edge."""
        if len(self.x) == 0:
            return 0.0, 0.0, 1e9
        dx = x - self.x; dy = y - self.y
        d = np.hypot(dx, dy)
        clear = d - self.r
        near = clear < margin
        mn = float(clear.min())
        if not near.any():
            return 0.0, 0.0, mn
        w = np.clip((margin - clear[near]) / max(margin, 1e-6), 0.0, 1.0)
        dn = np.maximum(d[near], 1e-6)
        return float((dx[near] / dn * w).sum()), float((dy[near] / dn * w).sum()), mn


class _Actor:
    def __init__(self, spec, model, index=0):
        self.kind = spec.get("kind", "walker")
        self.cruise = float(spec.get("speed", 1.0))
        self.accel = float(spec.get("accel", 1.5))
        self.z = float(spec.get("z", 0.0))
        self.bob = float(spec.get("bob", 0.0))
        self.stride = float(spec.get("stride", 0.66))
        self.s0 = float(spec.get("s0", 0.0))
        self.stops = spec.get("stops") or []
        self.yield_dist = float(spec.get("yield_dist", 0.0))
        self.yield_width = float(spec.get("yield_width", 0.0))
        self.patience = spec.get("patience")           # None -> wait forever
        self.teleport = bool(spec.get("teleport", False))
        self.index = index
        self.phase0 = (index * 1.61803) % 1.0          # deterministic per-actor phase
        # avoidance / body model — vehicles are wider and stay on their lane
        # (queue, don't sidestep); walkers sidestep and turn freely
        veh = self.kind == "vehicle"
        self.body_r = float(spec.get("body_r", 1.7 if veh else 0.30))
        self.avoid_r = float(spec.get("avoid_r", 3.0 if veh else 0.55))  # spacing
        # NB: robot collision is now handled by the scene's explicit <contact>
        # <pair> entries (actor collision geom ↔ chassis_bumper), not a force
        # hack, so the old `hit_box`/`hit_force` push parameters are gone. The
        # spec still carries `hit_box` for backward-compat; it is unused here.
        self.max_lat = 0.0 if veh else 0.95           # max sidestep offset (m)
        self.turn_rate = 1.2 if veh else 3.2          # rad/s heading slew
        self.look_ahead = 6.0 if veh else 1.4         # obstacle/neighbour scan (m)
        closed = bool(spec.get("closed", True))
        pts = _round_corners(spec["path"], closed, FILLET.get(self.kind, 0.3))
        self.path = _Path(pts, closed)

        j = model.joint(spec["root"])                  # raises if missing
        self.qadr = int(j.qposadr[0])
        self.vadr = int(j.dofadr[0])
        # gait joints: q = gain * (bias + amp*sin(phase + poff)). `sign` (legacy)
        # folds into amp; `poff`/`bias` (optional) let knees flex one-way and
        # elbows carry a bent bias for a natural articulated walk.
        self.gait = []
        g = spec.get("gait") or {}
        js = g.get("joints", [])
        amps = g.get("amp", [])
        signs = g.get("sign", [1.0] * len(js))
        poffs = g.get("poff", [0.0] * len(js))
        biass = g.get("bias", [0.0] * len(js))
        for i, name in enumerate(js):
            try:
                gj = model.joint(name)
            except Exception:
                continue
            self.gait.append((int(gj.qposadr[0]), int(gj.dofadr[0]),
                              float(amps[i]) * float(signs[i]),
                              float(poffs[i]), float(biass[i])))
        self.wheels = []
        w = spec.get("wheels") or {}
        self.wheel_r = float(w.get("radius", 0.3))
        for name in w.get("joints", []):
            wj = model.joint(name)
            self.wheels.append((int(wj.qposadr[0]), int(wj.dofadr[0])))
        self.reset()

    def reset(self):
        self.s = self.s0 % self.path.total if self.path.closed \
            else min(self.s0, self.path.total)
        self.dir = 1.0
        self.v = 0.0
        self.yaw = self.path.probe(self.s)[2]
        self.blocked = 0.0
        self.wheel_a = 0.0
        self.offset = 0.0                              # current lateral sidestep
        self.px, self.py = self.path.probe(self.s)[:2]
        self.tx = self.px; self.ty = self.py; self.tyaw = self.yaw
        self.fx = self.px; self.fy = self.py
        self.wz = 0.0
        self._idle_t = 0.0

    # ── speed planning ───────────────────────────────────────────────────────
    def _dist_along(self, target_s):
        """Signed travel distance from s to target_s in the current direction."""
        d = (target_s - self.s) * self.dir
        if self.path.closed:
            d %= self.path.total
        return d

    def _v_target(self, t, robot_xy, x, y, hd, field, actors):
        v = self.cruise
        # gentle per-actor cruise variation so they never move in lockstep
        v *= 1.0 + 0.10 * math.sin(0.5 * t + self.phase0 * 6.283)
        for st in self.stops:                          # red stop-lines ahead
            period = float(st.get("period", 0.0))
            if period <= 0:
                continue
            red = ((t - float(st.get("offset", 0.0))) % period) < float(st.get("dur", 0.0))
            if not red:
                continue
            d = self._dist_along(float(st["s"]))
            if d >= 0:
                v = min(v, math.sqrt(2 * self.accel * max(0.0, d - 0.4)))
        # slow for the ROBOT in the path corridor (unless out of patience)
        if self.yield_dist > 0 and robot_xy is not None:
            if self.patience is None or self.blocked < float(self.patience):
                v = min(v, self._brake_for(robot_xy, x, y, hd,
                                           self.yield_dist, self.yield_width, 0.4))
        # slow for a NEIGHBOUR ahead in the corridor (queueing, no rear-ends)
        for o in actors:
            if o is self:
                continue
            v = min(v, self._brake_for((o.px, o.py), x, y, hd,
                                       self.look_ahead,
                                       self.body_r + o.body_r,
                                       self.body_r + o.body_r))
        # static obstacles are handled by STEERING (resolve), not by stopping —
        # a bench/tree never moves, so halting for it would freeze the actor
        # forever; we only ease off a touch when one is right in the path so the
        # sidestep has time to bend around it.
        ax, ay = x + hd[0] * self.body_r * 2, y + hd[1] * self.body_r * 2
        _, _, clear = field.repulse(ax, ay, self.body_r + 0.2)
        if clear < self.body_r:
            v = min(v, 0.45 * self.cruise)
        return max(0.0, v)

    def _brake_for(self, pxy, x, y, hd, dist, width, hold):
        """Speed cap to stop `hold` metres short of a point ahead in-corridor."""
        rel = (pxy[0] - x, pxy[1] - y)
        ahead = rel[0] * hd[0] + rel[1] * hd[1]
        lat = abs(rel[0] * hd[1] - rel[1] * hd[0])
        if 0.0 < ahead < dist and lat < width:
            return math.sqrt(2 * self.accel * max(0.0, ahead - hold))
        return self.cruise * 2.0                        # no constraint

    def _robot_ref(self, robot_xy, robot_moving):
        """The robot position this actor should AVOID, or None if it should
        collide with it. Vehicles never avoid the robot (they hit it). People
        avoid the robot ONLY when it is stationary — a moving robot can hit
        them (and be hit by them)."""
        if self.kind == "vehicle" or robot_moving:
            return None
        return robot_xy

    # ── phase 1: advance along the path at the planned speed ──────────────────
    def plan(self, t, dt, robot_xy, robot_moving, field, actors):
        rob = self._robot_ref(robot_xy, robot_moving)
        x, y, pyaw = self.path.probe(self.s)
        heading = pyaw if self.dir > 0 else pyaw + math.pi
        hd = (math.cos(heading), math.sin(heading))
        vt = self._v_target(t, rob, x, y, hd, field, actors)
        self.blocked = self.blocked + dt if vt < 0.15 * self.cruise else 0.0
        dv = max(-self.accel * dt, min(self.accel * dt, vt - self.v))
        self.v = max(0.0, self.v + dv)

        s_new = self.s + self.v * dt * self.dir
        if self.path.closed:
            s_new %= self.path.total
        elif self.teleport:
            if s_new >= self.path.total:
                s_new -= self.path.total
        else:                                          # ping-pong: turn around
            if s_new >= self.path.total:
                s_new = 2 * self.path.total - s_new
                self.dir = -1.0
            elif s_new < 0.0:
                s_new = -s_new
                self.dir = 1.0
        self.s = s_new
        tx, ty, pyaw = self.path.probe(self.s)
        self.tx, self.ty = tx, ty
        self.tyaw = pyaw if self.dir > 0 else pyaw + math.pi

    # ── phase 2: sidestep obstacles/neighbours, get shoved by the robot, turn ──
    def resolve(self, dt, robot_xy, robot_moving, field, actors):
        rob = self._robot_ref(robot_xy, robot_moving)  # None → collide, don't dodge
        hd = (math.cos(self.tyaw), math.sin(self.tyaw))
        perp = (-hd[1], hd[0])
        nx = ny = 0.0                                  # neighbour spread (any dir)
        if self.max_lat > 0:                           # walkers sidestep + spread
            # steer around obstacles at the body AND a step ahead, so the bend
            # starts early instead of at the last moment
            ox, oy, _ = field.repulse(self.tx, self.ty, self.body_r + 0.55)
            ax = self.tx + hd[0] * 0.6; ay = self.ty + hd[1] * 0.6
            ox2, oy2, _ = field.repulse(ax, ay, self.body_r + 0.55)
            lat = (ox + ox2) * perp[0] + (oy + oy2) * perp[1]
            lat = max(-self.max_lat, min(self.max_lat, lat * 1.6))
            self.offset += (lat - self.offset) * 0.25          # smooth sidestep
            # inter-actor separation as an omnidirectional nudge, so actors
            # stacked at a signal / doorway spread apart instead of overlapping
            for o in actors:
                if o is self:
                    continue
                dx = self.px - o.px; dy = self.py - o.py
                d = math.hypot(dx, dy)
                sep = self.body_r + o.body_r + 0.2
                if d < sep:
                    if d < 1e-3:                       # exactly coincident: break
                        ang = (self.index - o.index) * 2.399  # golden-angle, per-pair
                        dx, dy, d = math.cos(ang), math.sin(ang), 1.0
                    w = (sep - d) / sep
                    nx += dx / d * w; ny += dy / d * w
            mag = math.hypot(nx, ny)
            if mag > 0.5:                              # clamp the spread step
                nx *= 0.5 / mag; ny *= 0.5 / mag
        fx = self.tx + perp[0] * self.offset + nx
        fy = self.ty + perp[1] * self.offset + ny
        # robot shove: push the actor out of a STATIONARY robot's radius so a
        # person walks around it. When the robot is moving (rob is None) there
        # is no shove — the actor holds its line and the robot can hit it.
        if rob is not None:
            dx = fx - rob[0]; dy = fy - rob[1]
            d = math.hypot(dx, dy)
            keep = self.body_r + (0.6 if self.kind == "vehicle" else 0.45)
            if 0.0 < d < keep:
                k = keep - d
                fx += dx / d * k; fy += dy / d * k
        self.fx, self.fy = fx, fy

        # heading from ACTUAL displacement → natural turning through sidesteps;
        # when idle, a slow deterministic "look around" sway
        ddx = fx - self.px; ddy = fy - self.py
        moving = (ddx * ddx + ddy * ddy) > 1e-5 and self.v > 0.06
        if moving:
            want = math.atan2(ddy, ddx)
        elif self.blocked > 0.4 and self.kind != "vehicle":
            want = self.tyaw + 0.35 * math.sin(0.7 * self._idle_t + self.phase0 * 6.283)
        else:
            want = self.tyaw
        dyaw = math.atan2(math.sin(want - self.yaw), math.cos(want - self.yaw))
        step = self.turn_rate * dt
        applied = max(-step, min(step, dyaw))          # turn-rate limited
        self.yaw += applied
        self.wz = applied / dt if dt > 0 else 0.0      # honest angular velocity
        self._idle_t = getattr(self, "_idle_t", 0.0) + dt

    # ── phase 3: write pose + gait + wheels to the world state ────────────────
    def write(self, data, dt):
        gain = self.v / self.cruise if self.cruise > 0 else 0.0
        phase = 2 * math.pi * self.s / self.stride
        z = self.z + self.bob * gain * (0.5 - 0.5 * math.cos(2 * phase))
        # published velocity is clamped: a big avoidance/shove position step must
        # not write a pathological qvel (it would spike a contact impulse before
        # the next keyframe overwrites the pose)
        vcap = max(2.0, 2.5 * self.cruise)
        vx = max(-vcap, min(vcap, (self.fx - self.px) / dt)) if dt > 0 else 0.0
        vy = max(-vcap, min(vcap, (self.fy - self.py) / dt)) if dt > 0 else 0.0
        q, dq = data.qpos, data.qvel
        a, va = self.qadr, self.vadr
        q[a:a + 3] = (self.fx, self.fy, z)
        q[a + 3:a + 7] = (math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2))
        dq[va:va + 3] = (vx, vy, 0.0)
        dq[va + 3:va + 6] = (0.0, 0.0, self.wz)
        w = 2 * math.pi * self.v / self.stride
        for (ga, gv, amp, poff, bias) in self.gait:
            q[ga] = gain * (bias + amp * math.sin(phase + poff))
            dq[gv] = gain * amp * math.cos(phase + poff) * w
        if self.wheels:
            self.wheel_a += (self.v / self.wheel_r) * dt
            for (wa, wv) in self.wheels:
                q[wa] = self.wheel_a
                dq[wv] = self.v / self.wheel_r
        self.px, self.py = self.fx, self.fy


class _Belt:
    def __init__(self, spec, model):
        j = model.joint(spec["joint"])
        self.qadr = int(j.qposadr[0])
        self.vadr = int(j.dofadr[0])
        self.speed = float(spec.get("speed", 0.2))
        self.wrap = float(spec.get("wrap", 0.7))
        rec = spec.get("recycle") or {}
        self.axis = int(rec.get("axis", 0))
        self.limit = float(rec.get("max", math.inf))
        self.respawn = float(rec.get("respawn", 0.0))
        self.parts = []
        for name in rec.get("joints", []):
            pj = model.joint(name)
            adr = int(pj.qposadr[0])
            self.parts.append((adr, int(pj.dofadr[0]),
                               np.array(model.qpos0[adr:adr + 7])))

    def step(self, t, data):
        data.qpos[self.qadr] = ((self.speed * t + self.wrap / 2) % self.wrap
                                - self.wrap / 2)
        data.qvel[self.vadr] = self.speed
        for adr, vadr, home in self.parts:
            if data.qpos[adr + self.axis] > self.limit:
                data.qpos[adr:adr + 7] = home
                data.qpos[adr + self.axis] = self.respawn
                data.qvel[vadr:vadr + 6] = 0.0


class SceneDynamics:
    """Loaded per scene by the engine; step() is called every physics substep
    and internally decimates to KEY_DT keyframes."""

    def __init__(self, spec, model):
        self.actors = []
        self.belts = []
        for i, a in enumerate(spec.get("actors", [])):
            try:
                self.actors.append(_Actor(a, model, index=i))
            except Exception as e:
                log.warning("dynamics actor %s disabled: %s", a.get("root"), e)
        for c in spec.get("conveyors", []):
            try:
                self.belts.append(_Belt(c, model))
            except Exception as e:
                log.warning("dynamics conveyor %s disabled: %s", c.get("joint"), e)
        try:
            self._chassis = model.body("mabel_chassis").id
        except Exception:
            self._chassis = None

        # collision wiring: actor collision geoms are contype=2/conaffinity=0 —
        # they hit nothing via masks (never fighting the scene or each other in
        # physics; the controller handles obstacle + inter-actor spacing). The
        # ROBOT is made collidable with the actors by EXPLICIT <contact><pair>
        # entries the scene generator writes into each dynamic scene XML (each
        # actor's collision geom paired with the robot's `chassis_bumper` proxy).
        # Explicit pairs are required because MuJoCo's mask-based broadphase
        # silently drops the actor↔robot box pairs in these large scenes even
        # when the masks say collide (verified: mj_geomDistance shows overlap,
        # an explicit pair yields real contacts, auto-collision yields none).
        robot_bodies = _subtree_bodies(model, "base_root")

        # static-obstacle field for actor steering (excludes robot + actors)
        actor_bodies = set()
        for a in spec.get("actors", []):
            actor_bodies |= _subtree_bodies(model, a.get("root"))
        try:
            import mujoco
            data0 = mujoco.MjData(model)
            mujoco.mj_forward(model, data0)
            self.field = _ObstacleField(model, data0, robot_bodies, actor_bodies)
        except Exception as e:
            log.warning("obstacle field unavailable: %s", e)
            self.field = _ObstacleField.__new__(_ObstacleField)
            self.field.x = self.field.y = self.field.r = np.zeros(0)
        self._next_t = 0.0
        self._last_t = 0.0
        self._prev_robot = None

    @classmethod
    def sidecar(cls, scene_path):
        return os.path.splitext(scene_path)[0] + ".dynamics.json"

    @classmethod
    def load(cls, scene_path, model):
        path = cls.sidecar(scene_path)
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as fp:
                spec = json.load(fp)
            dyn = cls(spec, model)
        except Exception as e:
            log.warning("bad dynamics sidecar %s: %s", path, e)
            return None
        if not dyn.actors and not dyn.belts:
            return None
        log.info("scene dynamics up — %d actors, %d conveyors (%s)",
                 len(dyn.actors), len(dyn.belts), os.path.basename(path))
        return dyn

    def describe(self):
        return {"actors": len(self.actors), "conveyors": len(self.belts)}

    def reset(self):
        for a in self.actors:
            a.reset()
        self._next_t = 0.0
        self._last_t = 0.0
        self._prev_robot = None

    def step(self, model, data):
        """Kinematic keyframe write; no-op between keyframes."""
        t = float(data.time)
        if t < self._last_t - 1e-9:                    # world was reset
            self.reset()
        self._last_t = t
        # A car/person that runs into the robot shoves it through REAL MuJoCo
        # contacts: the scene's explicit <contact><pair> entries collide each
        # actor's collision geom against the robot's `chassis_bumper` proxy, so
        # the kinematically-driven actor (whose pose is written below) imparts a
        # genuine contact force to the robot chassis. No external force hack.
        if t + 1e-12 < self._next_t:
            return
        dt = KEY_DT
        self._next_t = t + KEY_DT
        robot_xy = None
        robot_moving = False
        if self._chassis is not None:
            p = data.xpos[self._chassis]
            robot_xy = (float(p[0]), float(p[1]))
            # is the robot driving? (translational speed over the last keyframe)
            if self._prev_robot is not None:
                spd = math.hypot(robot_xy[0] - self._prev_robot[0],
                                 robot_xy[1] - self._prev_robot[1]) / max(dt, 1e-6)
                robot_moving = spd > 0.12          # >0.12 m/s = actively moving
            self._prev_robot = robot_xy
        # three phases so avoidance sees a consistent snapshot of the others:
        # plan each actor's path advance, resolve steering/spacing/shove against
        # everyone's current positions, then commit poses.
        for a in self.actors:
            a.plan(t, dt, robot_xy, robot_moving, self.field, self.actors)
        for a in self.actors:
            a.resolve(dt, robot_xy, robot_moving, self.field, self.actors)
        for a in self.actors:
            a.write(data, dt)
        for b in self.belts:
            b.step(t, data)
