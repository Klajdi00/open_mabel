"""actors.py -- animated scene actors: pedestrians, vehicles, and their
dynamics-spec entries.

Animated actors are free-jointed bodies with gravcomp="1" (they float in
place when nothing drives them, so plain viewers and verify_scenes see a
static tableau) plus hinge "gait" joints (hips/shoulders or wheel spin) that
simulation_bridge/dynamics.py writes kinematically at keyframe rate. Only the
torso / chassis geom collides -- swinging limbs and wheels are visual-only so
the contact broadphase stays lean and a brushing limb can't fling anything.

Static variants (standing/seated people, parked cars) are plain jointless
bodies: zero DOF, pure scenery.

Path convention (shared with the runtime): a polyline of [x, y] waypoints,
`closed` loops it. `s0` is the actor's start arclength so several actors can
share one loop, spread out. All of it is deterministic -- no randomness.
"""
import math

# clothing palettes: (shirt, pants, skin)
PALETTES = [
    ("0.75 0.28 0.22 1", "0.20 0.22 0.30 1", "0.87 0.70 0.55 1"),
    ("0.24 0.42 0.65 1", "0.15 0.15 0.18 1", "0.62 0.45 0.32 1"),
    ("0.30 0.55 0.38 1", "0.32 0.28 0.24 1", "0.80 0.62 0.48 1"),
    ("0.85 0.78 0.60 1", "0.25 0.30 0.42 1", "0.55 0.38 0.28 1"),
    ("0.55 0.30 0.58 1", "0.18 0.18 0.20 1", "0.90 0.74 0.60 1"),
    ("0.90 0.55 0.15 1", "0.22 0.26 0.30 1", "0.70 0.52 0.38 1"),
    ("0.20 0.20 0.24 1", "0.45 0.42 0.38 1", "0.84 0.66 0.52 1"),
    ("0.92 0.92 0.94 1", "0.12 0.20 0.35 1", "0.60 0.42 0.30 1"),
]
CAR_COLORS = ["0.75 0.16 0.14 1", "0.16 0.30 0.55 1", "0.86 0.84 0.80 1",
              "0.15 0.16 0.18 1", "0.55 0.58 0.60 1", "0.80 0.62 0.12 1",
              "0.25 0.45 0.30 1"]


# ── path math (mirrored in simulation_bridge/dynamics.py) ────────────────────
def _seglens(path, closed):
    pts = list(path) + ([path[0]] if closed else [])
    return [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)], pts


def path_length(path, closed=True):
    lens, _ = _seglens(path, closed)
    return sum(lens)


def point_at(path, s, closed=True):
    """(x, y, yaw) at arclength s along the polyline."""
    lens, pts = _seglens(path, closed)
    total = sum(lens)
    s = s % total if closed else min(max(s, 0.0), total)
    for (l, i) in zip(lens, range(len(lens))):
        if s <= l or i == len(lens) - 1:
            t = s / l if l > 0 else 0.0
            x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
            y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
            yaw = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
            return x, y, yaw
        s -= l
    return pts[-1][0], pts[-1][1], 0.0


def rect_loop(x0, y0, x1, y1):
    """Counter-clockwise rectangular loop."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ── people ───────────────────────────────────────────────────────────────────
# body archetypes: proportions + palette bias so a crowd reads as men, women,
# kids and the elderly rather than identical mannequins
HAIR = ["0.12 0.09 0.07 1", "0.35 0.22 0.10 1", "0.62 0.52 0.32 1",
        "0.72 0.72 0.74 1", "0.05 0.05 0.06 1", "0.55 0.30 0.14 1"]
PERSON_TYPES = {
    #        scale  torso_r  legf   armf  stoop  hairlen  cadence
    "man":    (1.00, 0.140, 1.00, 1.00, 0.00, 0.05, 1.00),
    "woman":  (0.94, 0.115, 1.02, 0.95, 0.00, 0.16, 1.06),
    "boy":    (0.66, 0.100, 0.92, 0.92, 0.00, 0.05, 1.30),
    "girl":   (0.63, 0.095, 0.94, 0.90, 0.00, 0.20, 1.34),
    "elder":  (0.92, 0.130, 0.92, 0.90, 0.22, 0.06, 0.82),
}


def _pdims(scale, ptype):
    sc, tr, legf, armf, stoop, hairlen, cad = PERSON_TYPES.get(ptype, PERSON_TYPES["man"])
    sc *= scale
    return {"sc": sc, "z": 0.94 * sc, "torso_r": tr * sc, "head_r": 0.093 * sc,
            "thigh": 0.30 * legf * sc, "shin": 0.32 * legf * sc,
            "upper": 0.27 * armf * sc, "fore": 0.26 * armf * sc,
            "hip_y": 0.085 * sc, "sh_y": 0.185 * sc, "stoop": stoop,
            "hairlen": hairlen * sc, "cadence": cad}


def _person_body(tag, d, shirt, pants, skin, hair, jointed, hat=None):
    """Torso + head(+hair) + full-height hit capsule + two 2-segment legs and
    arms (hip/knee, shoulder/elbow). jointed=True emits hinge joints for the
    gait; jointed=False bakes a relaxed standing pose as fixed geoms."""
    sc = d["sc"]
    hz = d["z"]
    stoop = d["stoop"]
    hair_g = (f'      <geom name="{tag}_hair" type="sphere" pos="{0.01 * sc:.3f} 0 {0.50 * sc + d["hairlen"] * 0.3:.3f}" '
              f'size="{d["head_r"] * 1.05:.3f}" rgba="{hair}" contype="0" conaffinity="0"/>\n') if d["hairlen"] > 0.02 else ""
    hat_g = (f'      <geom name="{tag}_hat" type="cylinder" pos="0 0 {0.58 * sc:.3f}" size="{d["head_r"] * 1.15:.3f} 0.02" '
             f'rgba="{hat}" contype="0" conaffinity="0"/>\n') if hat else ""
    # torso, slightly pitched forward for the elderly (stoop)
    torso = (f'      <geom name="{tag}_torso" type="capsule" fromto="{stoop * 0.12:.3f} 0 {-0.22 * sc:.3f} '
             f'{-stoop * 0.10:.3f} 0 {0.30 * sc:.3f}" size="{d["torso_r"]:.3f}" rgba="{shirt}" contype="0" conaffinity="0"/>\n')
    head = (f'      <geom name="{tag}_head" type="sphere" pos="{-stoop * 0.12 + 0.01 * sc:.3f} 0 {0.50 * sc:.3f}" '
            f'size="{d["head_r"]:.3f}" rgba="{skin}" contype="0" conaffinity="0"/>\n')
    hit = (f'      <geom name="{tag}_hit" type="capsule" fromto="0 0 {-0.86 * sc:.3f} 0 0 {0.42 * sc:.3f}" '
           f'size="{0.17 * sc:.3f}" group="3" rgba="0 0 0 0" condim="3" friction="0.8 0.02 0.001" '
           f'contype="2" conaffinity="0" solref="0.02 1" solimp="0.8 0.9 0.01"/>\n')

    def leg(side, sgn):
        hy = sgn * d["hip_y"]
        if jointed:
            return (f'      <body name="{tag}_thigh{side}" pos="0 {hy:.3f} {-0.26 * sc:.3f}" gravcomp="1">\n'
                    f'        <joint name="{tag}_hip{side}" type="hinge" axis="0 1 0" damping="0.6"/>\n'
                    f'        <geom name="{tag}_thigh{side}_g" type="capsule" fromto="0 0 0 0 0 {-d["thigh"]:.3f}" size="{0.062 * sc:.3f}" rgba="{pants}" contype="0" conaffinity="0"/>\n'
                    f'        <body name="{tag}_shin{side}" pos="0 0 {-d["thigh"]:.3f}" gravcomp="1">\n'
                    f'          <joint name="{tag}_knee{side}" type="hinge" axis="0 1 0" damping="0.4"/>\n'
                    f'          <geom name="{tag}_shin{side}_g" type="capsule" fromto="0 0 0 0 0 {-d["shin"]:.3f}" size="{0.055 * sc:.3f}" rgba="{pants}" contype="0" conaffinity="0"/>\n'
                    f'          <geom name="{tag}_foot{side}" type="box" pos="{0.06 * sc:.3f} 0 {-d["shin"]:.3f}" size="{0.08 * sc:.3f} {0.045 * sc:.3f} {0.028 * sc:.3f}" rgba="0.14 0.12 0.11 1" contype="0" conaffinity="0"/>\n'
                    f'        </body>\n'
                    f'      </body>\n')
        return (f'      <geom name="{tag}_leg{side}_g" type="capsule" fromto="0 {hy:.3f} {-0.26 * sc:.3f} 0 {hy:.3f} {-0.26 * sc - d["thigh"] - d["shin"]:.3f}" size="{0.06 * sc:.3f}" rgba="{pants}" contype="0" conaffinity="0"/>\n'
                f'      <geom name="{tag}_foot{side}" type="box" pos="{0.06 * sc:.3f} {hy:.3f} {-0.26 * sc - d["thigh"] - d["shin"]:.3f}" size="{0.08 * sc:.3f} {0.045 * sc:.3f} {0.028 * sc:.3f}" rgba="0.14 0.12 0.11 1" contype="0" conaffinity="0"/>\n')

    def arm(side, sgn):
        sy = sgn * d["sh_y"]
        if jointed:
            return (f'      <body name="{tag}_upper{side}" pos="{-stoop * 0.06:.3f} {sy:.3f} {0.24 * sc:.3f}" gravcomp="1">\n'
                    f'        <joint name="{tag}_sh{side}" type="hinge" axis="0 1 0" damping="0.4"/>\n'
                    f'        <geom name="{tag}_upper{side}_g" type="capsule" fromto="0 0 0 0 0 {-d["upper"]:.3f}" size="{0.043 * sc:.3f}" rgba="{shirt}" contype="0" conaffinity="0"/>\n'
                    f'        <body name="{tag}_fore{side}" pos="0 0 {-d["upper"]:.3f}" gravcomp="1">\n'
                    f'          <joint name="{tag}_el{side}" type="hinge" axis="0 1 0" damping="0.3"/>\n'
                    f'          <geom name="{tag}_fore{side}_g" type="capsule" fromto="0 0 0 0 0 {-d["fore"]:.3f}" size="{0.037 * sc:.3f}" rgba="{skin}" contype="0" conaffinity="0"/>\n'
                    f'        </body>\n'
                    f'      </body>\n')
        return (f'      <geom name="{tag}_arm{side}_g" type="capsule" fromto="{-stoop * 0.06:.3f} {sy:.3f} {0.24 * sc:.3f} {0.10 * sc:.3f} {sy:.3f} {0.24 * sc - d["upper"] - d["fore"]:.3f}" size="{0.041 * sc:.3f}" rgba="{shirt}" contype="0" conaffinity="0"/>\n')

    body = torso + head + hair_g + hat_g + hit
    body += leg("L", 1) + leg("R", -1) + arm("L", 1) + arm("R", -1)
    return body


def walker(s, tag, path, speed=1.25, s0=0.0, closed=True, palette=0,
           scale=1.0, ptype=None, stops=None, bob=0.022, hat=None, comment=None):
    """An articulated walking pedestrian (hip+knee, shoulder+elbow), driven by
    the bridge. `ptype` ∈ man/woman/boy/girl/elder picks proportions + gait."""
    # default: vary the crowd deterministically by palette (men, women, a few
    # kids and elders) unless the caller pins a type
    ptype = ptype or ("man", "woman", "man", "woman", "elder", "man",
                      "woman", "man")[palette % 8]
    shirt, pants, skin = PALETTES[palette % len(PALETTES)]
    hair = HAIR[(palette + (2 if ptype in ("woman", "girl") else 0)) % len(HAIR)]
    d = _pdims(scale, ptype)
    x, y, yaw = point_at(path, s0, closed)
    if comment:
        s.raw(f"    <!-- {comment} ({ptype}) -->")
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {d['z']:.3f}" euler="0 0 {yaw:.4f}" gravcomp="1">
      <freejoint name="{tag}_root"/>
{_person_body(tag, d, shirt, pants, skin, hair, jointed=True, hat=hat)}    </body>""")
    cad = d["cadence"]
    # gait (person faces +local-x, all hinges about +y):
    #   hip  θ>0 swings the leg forward; L/R opposite.
    #   knee θ>0 flexes the shin BACKWARD (heel up) — the natural direction, so
    #        the knee bias/amp are POSITIVE and the joint never hyperextends.
    #        It flexes through swing (phased ~a quarter-cycle after the hip).
    #   sh   arms counter-swing the legs (opposite sign to the hips).
    #   el   θ<0 bends the forearm FORWARD — natural — so the elbow bias is
    #        NEGATIVE (a lightly-carried bent arm) with a small swing.
    joints = [f"{tag}_hipL", f"{tag}_hipR", f"{tag}_kneeL", f"{tag}_kneeR",
              f"{tag}_shL", f"{tag}_shR", f"{tag}_elL", f"{tag}_elR"]
    amp = [0.46, -0.46, 0.40, 0.40, -0.34, 0.34, 0.14, 0.14]
    poff = [0.0, 0.0, 1.6, 1.6 + math.pi, 0.0, 0.0, 0.0, math.pi]
    bias = [0.0, 0.0, 0.40, 0.40, 0.0, 0.0, -0.55, -0.55]
    s.actor({
        "kind": "walker", "root": f"{tag}_root", "z": round(d["z"], 3),
        "path": [[round(px, 3), round(py, 3)] for px, py in path],
        "closed": closed, "speed": speed, "accel": 1.6, "s0": round(s0, 3),
        "bob": round(bob * cad, 4), "stride": round(0.66 * d["sc"] / cad, 3),
        "gait": {"joints": joints, "amp": amp, "poff": [round(p, 3) for p in poff],
                 "bias": bias},
        **({"stops": stops} if stops else {}),
        "yield_dist": 1.1, "yield_width": 0.9, "patience": 8.0,
        "hit_box": [round(0.32 * d["sc"], 2), round(0.32 * d["sc"], 2)],
    })


def person_standing(s, tag, x, y, yaw=0.0, palette=0, scale=1.0, ptype="man", hat=None):
    """Static bystander (baked relaxed pose, zero DOF)."""
    shirt, pants, skin = PALETTES[palette % len(PALETTES)]
    hair = HAIR[(palette + (2 if ptype in ("woman", "girl") else 0)) % len(HAIR)]
    d = _pdims(scale, ptype)
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {d['z']:.3f}" euler="0 0 {yaw:.4f}">
{_person_body(tag, d, shirt, pants, skin, hair, jointed=False, hat=hat)}    </body>""")


def person_seated(s, tag, x, y, yaw=0.0, palette=0, seat_h=0.46):
    """Static seated diner/customer FACING +local-x, hips over (x, y)."""
    shirt, pants, skin = PALETTES[palette % len(PALETTES)]
    z = seat_h + 0.08
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 {yaw:.4f}">
      <geom name="{tag}_torso" type="capsule" fromto="0 0 0 0 0 0.50" size="0.125" rgba="{shirt}" condim="3"/>
      <geom name="{tag}_head" type="sphere" pos="0.02 0 0.68" size="0.093" rgba="{skin}" contype="0" conaffinity="0"/>
      <geom name="{tag}_thighL" type="capsule" fromto="0 0.085 0.02 0.36 0.085 0.02" size="0.062" rgba="{pants}" contype="0" conaffinity="0"/>
      <geom name="{tag}_thighR" type="capsule" fromto="0 -0.085 0.02 0.36 -0.085 0.02" size="0.062" rgba="{pants}" contype="0" conaffinity="0"/>
      <geom name="{tag}_shinL" type="capsule" fromto="0.36 0.085 0.02 0.38 0.085 {-z + 0.06:.3f}" size="0.052" rgba="{pants}" contype="0" conaffinity="0"/>
      <geom name="{tag}_shinR" type="capsule" fromto="0.36 -0.085 0.02 0.38 -0.085 {-z + 0.06:.3f}" size="0.052" rgba="{pants}" contype="0" conaffinity="0"/>
      <geom name="{tag}_armL" type="capsule" fromto="0 0.18 0.42 0.30 0.16 0.16" size="0.040" rgba="{shirt}" contype="0" conaffinity="0"/>
      <geom name="{tag}_armR" type="capsule" fromto="0 -0.18 0.42 0.30 -0.16 0.16" size="0.040" rgba="{shirt}" contype="0" conaffinity="0"/>
    </body>""")


# ── animals (dogs & cats) ─────────────────────────────────────────────────────
# kind -> (body_len, body_r, leg_len, z, tail_len, ear, palette)
PETS = {
    "dog":       (0.34, 0.11, 0.26, 0.40, 0.22, "up",   "0.55 0.40 0.24 1"),
    "dog_black": (0.36, 0.12, 0.28, 0.42, 0.24, "up",   "0.14 0.13 0.13 1"),
    "dog_small": (0.24, 0.085, 0.17, 0.28, 0.16, "flop", "0.82 0.76 0.60 1"),
    "cat":       (0.24, 0.075, 0.16, 0.26, 0.26, "up",   "0.30 0.28 0.30 1"),
    "cat_ginger": (0.24, 0.078, 0.16, 0.26, 0.26, "up",  "0.80 0.50 0.22 1"),
}


def _pet_body(tag, k, jointed):
    bl, br, ll, z, tl, ear, col = k
    def leg(nm, x, y, sgn):
        if jointed:
            return (f'      <body name="{tag}_{nm}" pos="{x:.3f} {y:.3f} {-br * 0.2:.3f}" gravcomp="1">\n'
                    f'        <joint name="{tag}_{nm}_j" type="hinge" axis="0 1 0" damping="0.3"/>\n'
                    f'        <geom name="{tag}_{nm}_g" type="capsule" fromto="0 0 0 0 0 {-ll:.3f}" size="{br * 0.32:.3f}" rgba="{col}" contype="0" conaffinity="0"/>\n'
                    f'      </body>\n')
        return (f'      <geom name="{tag}_{nm}_g" type="capsule" fromto="{x:.3f} {y:.3f} {-br * 0.2:.3f} {x:.3f} {y:.3f} {-br * 0.2 - ll:.3f}" size="{br * 0.32:.3f}" rgba="{col}" contype="0" conaffinity="0"/>\n')
    ear_g = ""
    if ear == "up":
        ear_g = (f'      <geom name="{tag}_earL" type="box" pos="{bl:.3f} {br * 0.5:.3f} {br * 1.6:.3f}" size="0.015 0.02 0.04" rgba="{col}" contype="0" conaffinity="0"/>\n'
                 f'      <geom name="{tag}_earR" type="box" pos="{bl:.3f} {-br * 0.5:.3f} {br * 1.6:.3f}" size="0.015 0.02 0.04" rgba="{col}" contype="0" conaffinity="0"/>\n')
    body = (f'      <geom name="{tag}_body" type="capsule" fromto="{-bl:.3f} 0 0 {bl:.3f} 0 0" size="{br:.3f}" rgba="{col}" contype="0" conaffinity="0"/>\n'
            f'      <geom name="{tag}_neck" type="capsule" fromto="{bl:.3f} 0 0 {bl + 0.10:.3f} 0 {br * 0.9:.3f}" size="{br * 0.7:.3f}" rgba="{col}" contype="0" conaffinity="0"/>\n'
            f'      <geom name="{tag}_head" type="sphere" pos="{bl + 0.13:.3f} 0 {br * 1.2:.3f}" size="{br * 0.95:.3f}" rgba="{col}" contype="0" conaffinity="0"/>\n'
            f'      <geom name="{tag}_tail" type="capsule" fromto="{-bl:.3f} 0 {br * 0.4:.3f} {-bl - tl * 0.7:.3f} 0 {br * 0.4 + tl * 0.6:.3f}" size="{br * 0.28:.3f}" rgba="{col}" contype="0" conaffinity="0"/>\n'
            + ear_g +
            f'      <geom name="{tag}_hit" type="box" pos="0 0 {br * 0.3:.3f}" size="{bl + 0.1:.3f} {br + 0.03:.3f} {(z) * 0.5 + 0.1:.3f}" group="3" rgba="0 0 0 0" contype="2" conaffinity="0" condim="3" solref="0.02 1" solimp="0.8 0.9 0.01"/>\n')
    body += leg("flt", bl * 0.7, br * 0.7, 1) + leg("frt", bl * 0.7, -br * 0.7, -1)
    body += leg("blt", -bl * 0.7, br * 0.7, 1) + leg("brt", -bl * 0.7, -br * 0.7, -1)
    return body


def quadruped(s, tag, path, kind="dog", speed=1.6, s0=0.0, closed=True, comment=None):
    """A trotting dog/cat that follows a path (diagonal-pair trot gait)."""
    k = PETS.get(kind, PETS["dog"])
    z = k[3]
    x, y, yaw = point_at(path, s0, closed)
    if comment:
        s.raw(f"    <!-- {comment} ({kind}) -->")
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 {yaw:.4f}" gravcomp="1">
      <freejoint name="{tag}_root"/>
{_pet_body(tag, k, jointed=True)}    </body>""")
    # trot: diagonal legs in phase (FL+BR vs FR+BL), opposite pairs
    joints = [f"{tag}_flt_j", f"{tag}_frt_j", f"{tag}_blt_j", f"{tag}_brt_j"]
    s.actor({
        "kind": "walker", "root": f"{tag}_root", "z": round(z, 3),
        "path": [[round(px, 3), round(py, 3)] for px, py in path],
        "closed": closed, "speed": speed, "accel": 2.2, "s0": round(s0, 3),
        "bob": 0.012, "stride": round(0.34, 3),
        "gait": {"joints": joints, "amp": [0.5, -0.5, -0.5, 0.5],
                 "poff": [0.0, 0.0, 0.0, 0.0], "bias": [0.0, 0.0, 0.0, 0.0]},
        "yield_dist": 0.7, "yield_width": 0.5, "patience": 4.0,
        "body_r": 0.28, "avoid_r": 0.5, "hit_box": [round(k[0] + 0.1, 2), round(k[1], 2)],
    })


def pet_sitting(s, tag, x, y, yaw=0.0, kind="dog"):
    """Static sitting/standing pet (zero DOF)."""
    k = PETS.get(kind, PETS["dog"])
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {k[3]:.3f}" euler="0 0 {yaw:.4f}">
{_pet_body(tag, k, jointed=False)}    </body>""")


# ── vehicles ─────────────────────────────────────────────────────────────────
# type -> geometry: L,W half-extents · wheel r · body half-h · cabin(h, len-frac,
# x-offset) · glass-h · flags
CAR_TYPES = {
    "sedan":  dict(L=2.05, W=0.86, wr=0.32, bh=0.20, ch=0.24, cf=0.52, coff=-0.15, gl=0.16),
    "suv":    dict(L=2.20, W=0.94, wr=0.38, bh=0.24, ch=0.32, cf=0.62, coff=-0.05, gl=0.20),
    "van":    dict(L=2.35, W=0.96, wr=0.36, bh=0.26, ch=0.44, cf=0.80, coff=0.00, gl=0.26),
    "pickup": dict(L=2.35, W=0.92, wr=0.38, bh=0.22, ch=0.26, cf=0.34, coff=0.42, gl=0.16, bed=True),
    "sports": dict(L=2.05, W=0.92, wr=0.30, bh=0.15, ch=0.15, cf=0.46, coff=-0.10, gl=0.11),
    "taxi":   dict(L=2.05, W=0.86, wr=0.32, bh=0.20, ch=0.24, cf=0.52, coff=-0.15, gl=0.16, sign=True),
    "bus":    dict(L=4.70, W=1.30, wr=0.42, bh=0.00, ch=0.55, cf=1.00, coff=0.00, gl=0.28, bus=True),
}
CAR_ALL = ["sedan", "suv", "van", "pickup", "sports", "sedan", "suv"]


def _car_geoms(tag, color, t, kinematic=False):
    """Car/truck/bus geom block from a CAR_TYPES spec `t` (root at ground)."""
    L, W, bh = t["L"], t["W"], t["bh"]
    coll = 'contype="2" conaffinity="0" ' if kinematic else ''
    if t.get("bus"):
        body = (f'      <geom name="{tag}_body" type="box" pos="0 0 {t["ch"] + 0.1:.2f}" '
                f'size="{L:.2f} {W:.2f} {t["ch"]:.2f}" rgba="{color}" condim="3" friction="0.9 0.02 0.001" {coll}/>\n'
                f'      <geom name="{tag}_glass" type="box" pos="0 0 {t["ch"] + 0.28:.2f}" '
                f'size="{L - 0.02:.2f} {W - 0.02:.2f} {t["gl"]:.2f}" rgba="0.15 0.20 0.26 1" contype="0" conaffinity="0"/>\n')
        return body
    body = (f'      <geom name="{tag}_body" type="box" pos="0 0 {bh - 0.02:.2f}" size="{L:.2f} {W:.2f} {bh:.2f}" '
            f'rgba="{color}" condim="3" friction="0.9 0.02 0.001" {coll}/>\n')
    cz = bh + t["ch"] + 0.02
    cab = (f'      <geom name="{tag}_cabin" type="box" pos="{t["coff"] * L:.2f} 0 {cz:.2f}" '
           f'size="{L * t["cf"]:.2f} {W - 0.06:.2f} {t["ch"]:.2f}" rgba="{color}" contype="0" conaffinity="0"/>\n'
           f'      <geom name="{tag}_glass" type="box" pos="{t["coff"] * L:.2f} 0 {cz + 0.02:.2f}" '
           f'size="{L * (t["cf"] - 0.02):.2f} {W - 0.045:.2f} {t["gl"]:.2f}" rgba="0.15 0.20 0.26 1" contype="0" conaffinity="0"/>\n')
    if t.get("bed"):
        cab += (f'      <geom name="{tag}_bed" type="box" pos="{-0.35 * L:.2f} 0 {bh + 0.10:.2f}" '
                f'size="{L * 0.42:.2f} {W - 0.05:.2f} 0.10" rgba="{color}" contype="0" conaffinity="0"/>\n')
    if t.get("sign"):
        cab += (f'      <geom name="{tag}_sign" type="box" pos="0 0 {cz + t["ch"] + 0.06:.2f}" '
                f'size="0.14 0.05 0.05" material="m_lamp" contype="0" conaffinity="0"/>\n')
    lights = (f'      <geom name="{tag}_hlL" type="sphere" pos="{L:.2f} {W - 0.25:.2f} {bh:.2f}" size="0.055" material="m_lamp" contype="0" conaffinity="0"/>\n'
              f'      <geom name="{tag}_hlR" type="sphere" pos="{L:.2f} {-W + 0.25:.2f} {bh:.2f}" size="0.055" material="m_lamp" contype="0" conaffinity="0"/>\n'
              f'      <geom name="{tag}_tlL" type="sphere" pos="{-L:.2f} {W - 0.25:.2f} {bh:.2f}" size="0.05" material="m_sig_red" contype="0" conaffinity="0"/>\n'
              f'      <geom name="{tag}_tlR" type="sphere" pos="{-L:.2f} {-W + 0.25:.2f} {bh:.2f}" size="0.05" material="m_sig_red" contype="0" conaffinity="0"/>\n')
    return body + cab + lights


def _wheel_block(tag, i, dx, dy, wheel_r, jointed):
    wname = f"{tag}_w{i}"
    if jointed:
        return (f'      <body name="{wname}" pos="{dx:.2f} {dy:.2f} {wheel_r - 0.04:.2f}" gravcomp="1">\n'
                f'        <joint name="{wname}_spin" type="hinge" axis="0 1 0" damping="0.1"/>\n'
                f'        <geom name="{wname}_g" type="cylinder" size="{wheel_r:.2f} 0.10" euler="1.570795 0 0" '
                f'rgba="0.10 0.10 0.11 1" contype="0" conaffinity="0"/>\n'
                f'      </body>\n')
    return (f'      <geom name="{wname}_g" type="cylinder" pos="{dx:.2f} {dy:.2f} {wheel_r - 0.04:.2f}" '
            f'size="{wheel_r:.2f} 0.10" euler="1.570795 0 0" rgba="0.10 0.10 0.11 1" '
            f'contype="0" conaffinity="0"/>\n')


def _wheel_positions(L, W):
    return ((L * 0.62, W - 0.02), (L * 0.62, -W + 0.02),
            (-L * 0.62, W - 0.02), (-L * 0.62, -W + 0.02))


def vehicle(s, tag, path, speed=6.0, s0=0.0, closed=True, color=None,
            body="sedan", bus=False, stops=None, teleport=False, comment=None):
    """A driving vehicle (sedan/suv/van/pickup/sports/taxi/bus): free body +
    spinning wheels, bridge-driven."""
    if bus:
        body = "bus"
    t = CAR_TYPES.get(body, CAR_TYPES["sedan"])
    color = ("0.92 0.75 0.10 1" if body == "taxi" else
             color or CAR_COLORS[sum(map(ord, tag)) % len(CAR_COLORS)])
    L, W, wheel_r = t["L"], t["W"], t["wr"]
    z = 0.04 + 0.005
    x, y, yaw = point_at(path, s0, closed)
    wheels, wjoints = "", []
    for i, (dx, dy) in enumerate(_wheel_positions(L, W)):
        wheels += _wheel_block(tag, i, dx, dy, wheel_r, jointed=True)
        wjoints.append(f"{tag}_w{i}_spin")
    if comment:
        s.raw(f"    <!-- {comment} ({body}) -->")
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 {yaw:.4f}" gravcomp="1">
      <freejoint name="{tag}_root"/>
{_car_geoms(tag, color, t, kinematic=True)}{wheels}    </body>""", "m_lamp", "m_sig_red")
    is_bus = bool(t.get("bus"))
    s.actor({
        "kind": "vehicle", "root": f"{tag}_root", "z": round(z, 3),
        "path": [[round(px, 3), round(py, 3)] for px, py in path],
        "closed": closed, "speed": speed, "accel": 1.4 if is_bus else 2.2,
        "s0": round(s0, 3), **({"teleport": True} if teleport else {}),
        "wheels": {"joints": wjoints, "radius": wheel_r},
        **({"stops": stops} if stops else {}),
        # cars NEVER yield to the robot — drive their lane; if the robot is in
        # the way they hit it (yield_dist 0 disables the robot brake/shove).
        # hit_box is the chassis footprint the runtime uses to shove the robot.
        "yield_dist": 0.0, "yield_width": 0.0,
        "body_r": L + 0.4, "avoid_r": L + 1.5,
        "hit_box": [round(L, 2), round(W, 2)],
    })


def parked_car(s, tag, x, y, yaw=0.0, color=None, body="sedan"):
    """Static parked car (no joints)."""
    t = CAR_TYPES.get(body, CAR_TYPES["sedan"])
    color = ("0.92 0.75 0.10 1" if body == "taxi" else
             color or CAR_COLORS[sum(map(ord, tag)) % len(CAR_COLORS)])
    z = 0.04
    wheels = ""
    for i, (dx, dy) in enumerate(_wheel_positions(t["L"], t["W"])):
        wheels += _wheel_block(tag, i, dx, dy, t["wr"], jointed=False)
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 {yaw:.4f}">
{_car_geoms(tag, color, t, kinematic=False)}{wheels}    </body>""", "m_lamp", "m_sig_red")
