"""scenes_la.py -- the Los Angeles suite (Outdoor & Leisure): a hillside suburb
with the Hollywood sign, and a Venice-style beach with palms and the ocean."""
import math

from .core import Scene
from . import prefabs as P
from . import actors as A


def _sun(s, warm=True):
    s.light("sun", (8, -6, 14), (-0.3, 0.35, -1),
            "0.78 0.72 0.60" if warm else "0.7 0.7 0.66", castshadow=True,
            specular="0.3 0.3 0.3")
    s.light("sky", (-8, 4, 14), (0.3, -0.3, -1), "0.30 0.30 0.32")


def la_suburb():
    s = Scene("la_suburb", "LA Hills Suburb", "Outdoor & Leisure",
              "A Los Angeles hillside neighbourhood: single-family houses with "
              "pitched roofs, driveways and lawns along a palm- and plum-lined "
              "street, brown hills rising behind with the HOLLYWOOD sign, and a "
              "dog trotting the block.",
              headlight=("0.54 0.52 0.48", "0.32 0.30 0.28"),
              skybox=("0.45 0.62 0.86", "0.90 0.86 0.74"),
              fog=("0.86 0.82 0.72 1", 30, 95))
    s.geom("floor", "plane", (0, 0, 0), (34, 26, 0.1), mat="m_grass",
           extra='contype="1" conaffinity="1" friction="1.5 0.05 0.001"')
    _sun(s)
    # street down the middle (E-W)
    s.geom("street", "box", (0, -2, 0.006), (30, 3.0, 0.006), mat="m_asphalt", collide=False)
    s.geom("sidewalk_n", "box", (0, 1.3, 0.007), (30, 1.2, 0.006), mat="m_sidewalk", collide=False)
    s.geom("sidewalk_s", "box", (0, -5.3, 0.007), (30, 1.2, 0.006), mat="m_sidewalk", collide=False)
    # houses on both sides with lawns + driveways
    colors = ["m_cream", "m_white", "m_cream", "m_brownstone", "m_white"]
    for i, hx in enumerate(range(-24, 28, 12)):
        # north row: face south (-y)
        P.house(s, f"hN{i}", hx, 7.5, yaw=-math.pi / 2, w=8, d=7,
                wall=colors[i % len(colors)], roof="m_dark")
        s.geom(f"driveN{i}", "box", (hx - 2, 3.5, 0.007), (1.4, 2.0, 0.006), mat="m_concrete", collide=False)
        A.parked_car(s, f"carN{i}", hx - 2, 4.0, yaw=1.570795, body=["sedan", "suv", "pickup"][i % 3])
        # south row: face north (+y)
        P.house(s, f"hS{i}", hx + 6, -10.5, yaw=math.pi / 2, w=8, d=7,
                wall=colors[(i + 2) % len(colors)], roof="m_dark")
        s.geom(f"driveS{i}", "box", (hx + 6 - 2, -7.5, 0.007), (1.4, 2.0, 0.006), mat="m_concrete", collide=False)
    # palms + plum trees lining the street
    for i, tx in enumerate(range(-26, 30, 8)):
        P.palm_tree(s, f"palm{i}", tx, 2.6, h=5.5 + (i % 3), lean=0.08 * (i % 3 - 1))
        P.plum_tree(s, f"plum{i}", tx + 4, -6.6, h=2.3)
    # hills behind (north) with the HOLLYWOOD sign
    for i, (hx, hy, hr) in enumerate(((-14, 16, 7), (0, 20, 9), (14, 17, 8), (-4, 22, 6))):
        s.geom(f"hill{i}", "ellipsoid", (hx, hy, 1.0), (hr, hr * 0.7, 4.0 + i), rgba="0.62 0.52 0.34 1")
    letters = "HOLLYWOOD"
    for i, ch in enumerate(letters):
        lx = -8 + i * 2.0
        s.geom(f"sign{i}", "box", (lx, 19, 6.5), (0.12, 0.05, 0.9), rgba="0.94 0.94 0.92 1", collide=False)
    # people + a dog
    A.walker(s, "jog", [[-26, 2.4], [26, 2.4]], closed=False, speed=2.3, s0=2.0, palette=2, ptype="woman")
    A.walker(s, "walk", [[26, -5.2], [-26, -5.2]], closed=False, speed=1.1, s0=10.0, palette=5, ptype="man")
    A.quadruped(s, "dog", [[-20, -5.0], [20, -5.0]], kind="dog", closed=False, speed=1.3, s0=4.0)
    A.person_standing(s, "neighbor", -18, 5.5, yaw=-1.2, palette=3)
    A.vehicle(s, "car1", [[-30, -2.5], [30, -2.5]], closed=False, teleport=True, speed=6.0, s0=5.0, body="suv")
    A.vehicle(s, "car2", [[30, -0.5], [-30, -0.5]], closed=False, teleport=True, speed=5.5, s0=22.0, body="sports")
    P.free_sphere(s, "ball", 1.0, 0.5, 0.11, 0.11, "0.85 0.30 0.20 1", mass=0.25)
    P.free_cylinder(s, "can", -2, -4.5, 0.13, 0.09, 0.12, "0.30 0.55 0.38 1", mass=0.4)
    return s


def venice_beach():
    s = Scene("venice_beach", "Venice Beach", "Outdoor & Leisure",
              "A California beach: soft sand down to a glittering ocean, a "
              "boardwalk lined with palms, a lifeguard tower and volleyball net, "
              "cars cruising the road behind. Sun, palms, surf.",
              headlight=("0.56 0.54 0.50", "0.34 0.32 0.30"),
              skybox=("0.35 0.60 0.90", "0.88 0.90 0.82"),
              fog=("0.82 0.86 0.90 1", 34, 110))
    # sand beach; ocean is a big water plane to the south (-y)
    s.geom("sand", "plane", (0, 6, 0), (36, 24, 0.1), mat="m_sand",
           extra='contype="1" conaffinity="1" friction="1.4 0.05 0.001"')
    s.geom("ocean", "box", (0, -22, -0.1), (46, 16, 0.1), mat="m_water", collide=False)
    s.geom("surf", "box", (0, -6, 0.02), (44, 1.2, 0.02), rgba="0.90 0.94 0.96 0.8", collide=False)
    _sun(s)
    # boardwalk (north) + palms
    s.geom("boardwalk", "box", (0, 16, 0.05), (34, 3.0, 0.05), mat="m_wood_floor", collide=False)
    for i, tx in enumerate(range(-30, 34, 7)):
        P.palm_tree(s, f"palm{i}", tx, 20 + (i % 2), h=5.5 + (i % 3), lean=0.06 * (i % 3 - 1))
    # shops behind the boardwalk
    for i, tx in enumerate(range(-28, 32, 12)):
        P.building(s, f"shop{i}", tx, 24, w=10, d=5, h=6 + i % 3 * 2, mat="m_storefront")
        P.awning(s, f"shop{i}_awn", tx, 21.4, yaw=0.0, L=5.0,
                 mat="m_awning_red" if i % 2 else "m_awning_green", z=3.0)
    # lifeguard tower
    s.geom("lg_legs", "box", (10, 4, 0.9), (0.9, 0.9, 0.9), mat="m_wood")
    s.geom("lg_hut", "box", (10, 4, 2.1), (1.1, 1.1, 0.6), rgba="0.85 0.20 0.16 1")
    s.geom("lg_roof", "box", (10, 4, 2.75), (1.3, 1.3, 0.06), mat="m_dark", collide=False)
    s.geom("lg_ramp", "box", (10, 6.0, 0.5), (0.6, 1.6, 0.04), euler=(0.5, 0, 0), mat="m_wood", collide=False)
    # volleyball net
    for sgn in (-1, 1):
        s.geom(f"vpost{sgn}", "cylinder", (-8, 6 + sgn * 3, 0.9), (0.05, 0.9), mat="m_dark")
    s.geom("vnet", "box", (-8, 6, 1.3), (0.02, 3.0, 0.4), rgba="0.9 0.9 0.9 0.4", collide=False)
    # beach umbrellas + towels
    for i, (ux, uy, c) in enumerate(((-3, 8, "m_awning_red"), (4, 10, "m_awning_green"), (-12, 11, "m_awning_red"))):
        P.parasol(s, f"umb{i}", ux, uy, mat=c)
        P.rug(s, f"towel{i}", ux, uy + 1.0, 1.2, 0.6, mat="m_rug")
    # people: strollers on the boardwalk, sunbathers, a jogger at the surf line, a dog
    for k in range(4):
        A.walker(s, f"stroll{k}", [[-30, 16 + (k % 2) * 0.8], [30, 16 + (k % 2) * 0.8]],
                 closed=False, speed=1.1 + 0.1 * k, s0=k * 12.0, palette=k + 1)
    A.walker(s, "jogger", [[28, -4.5], [-28, -4.5]], closed=False, speed=2.4, s0=3.0, palette=2, ptype="man")
    A.walker(s, "skater", [[-28, 17.5], [28, 17.5]], closed=False, speed=2.8, s0=8.0, palette=6, ptype="man")
    A.person_standing(s, "sun1", -3, 9.2, yaw=1.0, palette=4, ptype="woman")
    A.person_standing(s, "sun2", 4, 11.2, yaw=2.4, palette=0)
    A.quadruped(s, "dog", [[-20, 3], [20, 3]], kind="dog_small", closed=False, speed=1.6, s0=5.0)
    # cars cruising the road behind the shops
    A.vehicle(s, "car1", [[-32, 28], [32, 28]], closed=False, teleport=True, speed=6.0, s0=4.0, body="sports")
    A.vehicle(s, "van", [[32, 30], [-32, 30]], closed=False, teleport=True, speed=5.0, s0=18.0, body="van")
    P.free_sphere(s, "beachball", 0.0, 7.5, 0.16, 0.16, "0.95 0.55 0.15 1", mass=0.15)
    P.free_cylinder(s, "soda", 2.0, 8.0, 0.06, 0.033, 0.06, "0.75 0.16 0.14 1", mass=0.25)
    return s


def build_all():
    return [la_suburb(), venice_beach()]
