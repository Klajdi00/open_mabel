"""scenes_city.py -- City & Streets: the dynamic suite. Cars drive their
lanes, pedestrians work the sidewalks, traffic lights gate the intersection.
All actor motion is deterministic (period/offset stop cycles, fixed start
arclengths); vehicles brake for the robot, walkers wait then squeeze past."""
import math

from .core import Scene
from . import prefabs as P
from . import actors as A


def _lane_dashes(s, tag, x0, x1, y, pitch=4.0, along_x=True):
    n = int(abs(x1 - x0) / pitch)
    for i in range(n):
        t = x0 + (i + 0.5) * pitch * (1 if x1 > x0 else -1)
        pos = (t, y, 0.013) if along_x else (y, t, 0.013)
        size = (0.55, 0.06, 0.002) if along_x else (0.06, 0.55, 0.002)
        s.geom(f"{tag}_d{i}", "box", pos, size, rgba="0.85 0.85 0.80 1", collide=False)


def city_sidewalk():
    s = Scene("city_sidewalk", "City Sidewalk", "City & Streets",
              "A downtown block: the robot works a wide sidewalk between "
              "storefronts and a two-way street; taxis and cars circle the "
              "block, pedestrians stream both ways, lights and awnings "
              "overhead.",
              headlight=("0.46 0.47 0.50", "0.28 0.28 0.30"),
              skybox=("0.40 0.55 0.75", "0.80 0.85 0.92"),
              fog=("0.72 0.76 0.84 1", 22, 60))
    s.comment("ground: sidewalk plane, asphalt streets (robot spawns on the south sidewalk)")
    s.geom("floor", "plane", (0, 0, 0), (34, 34, 0.1), mat="m_sidewalk",
           extra='contype="1" conaffinity="1" friction="1.5 0.05 0.001"')
    s.geom("street_s", "box", (0, 5.5, 0.006), (30, 3.5, 0.006), mat="m_asphalt")
    s.geom("street_n", "box", (0, 22.5, 0.006), (30, 3.5, 0.006), mat="m_asphalt")
    s.geom("xstreet_e", "box", (28.0, 14.0, 0.006), (2.5, 12.0, 0.006), mat="m_asphalt")
    s.geom("xstreet_w", "box", (-28.0, 14.0, 0.006), (2.5, 12.0, 0.006), mat="m_asphalt")
    _lane_dashes(s, "lane_s", -26, 26, 5.5)
    _lane_dashes(s, "lane_n", -26, 26, 22.5)
    s.comment("crosswalks over the south street")
    s.geom("xwalk1", "box", (10.0, 5.5, 0.010), (1.5, 3.5, 0.004), mat="m_crosswalk", euler=(0, 0, 1.570795))
    s.geom("xwalk2", "box", (-14.0, 5.5, 0.010), (1.5, 3.5, 0.004), mat="m_crosswalk", euler=(0, 0, 1.570795))
    s.comment("south building row (storefronts facing the robot's sidewalk)")
    for i, (x, w, h) in enumerate(((-24, 9, 17), (-14.5, 8, 22), (-5.5, 8, 14),
                                   (3.5, 8, 25), (12.5, 8, 16), (22, 10, 20))):
        P.building(s, f"bldgS{i + 1}", x, -8.2, w=w, d=8.0, h=h)
    s.comment("central block between the streets")
    for i, (x, w, h) in enumerate(((-22, 11, 19), (-9.5, 12, 26), (3.5, 12, 15), (16.5, 12, 23))):
        P.building(s, f"bldgC{i + 1}", x, 14.0, w=w, d=6.0, h=h)
    s.comment("north building row (backdrop)")
    for i, (x, w, h) in enumerate(((-18, 14, 21), (-2, 14, 17), (14, 14, 24))):
        P.building(s, f"bldgN{i + 1}", x, 31.5, w=w, d=8.0, h=h, shopfront=False)
    s.comment("street furniture on the robot's sidewalk")
    P.awning(s, "awn1", -14.5, -4.05, yaw=0.0, L=5.0, mat="m_awning_red", z=2.7)
    P.awning(s, "awn2", 3.5, -4.05, yaw=0.0, L=5.0, mat="m_awning_green", z=2.7)
    for i, x in enumerate((-20.0, -8.0, 4.0, 16.0)):
        P.streetlight(s, f"sl{i + 1}", x, 1.7, yaw=1.570795)
    for i, x in enumerate((-16.0, 8.0)):
        P.tree(s, f"stree{i + 1}", x, -2.8, h=1.3, r=0.8)
        s.geom(f"stree{i + 1}_pit", "box", (x, -2.8, 0.012), (0.8, 0.8, 0.004), rgba="0.36 0.25 0.18 1")
    P.hydrant(s, "hyd1", -6.5, 1.5)
    P.trash_can(s, "tc1", -12.0, 1.5)
    P.trash_can(s, "tc2", 12.5, 1.6)
    P.bench(s, "bench1", -2.0, -3.4, yaw=math.pi, L=1.8)
    s.geom("newsbox1", "box", (6.3, 1.5, 0.45), (0.22, 0.18, 0.45), rgba="0.16 0.30 0.55 1")
    s.geom("newsbox2", "box", (6.9, 1.5, 0.45), (0.22, 0.18, 0.45), rgba="0.75 0.16 0.14 1")
    s.geom("trashbags", "sphere", (18.5, 1.3, 0.3), (0.35,), rgba="0.14 0.14 0.15 1")
    s.geom("trashbags2", "sphere", (19.2, 1.4, 0.26), (0.30,), rgba="0.18 0.18 0.20 1")
    for i, (x, y) in enumerate(((-25.8, 2.4), (25.8, 2.4))):
        P.traffic_light(s, f"tl{i + 1}", x, y, yaw=0.0 if x < 0 else math.pi)
    s.comment("parked cars along the south curb")
    A.parked_car(s, "park1", -20.5, 2.9, yaw=0.0, color="0.86 0.84 0.80 1")
    A.parked_car(s, "park2", -3.0, 2.9, yaw=0.0, color="0.15 0.16 0.18 1")
    A.parked_car(s, "park3", 16.5, 2.9, yaw=0.0, color="0.80 0.62 0.12 1")
    s.light("sun", (6, -6, 10), (-0.3, 0.45, -1), "0.62 0.60 0.55", castshadow=True,
            specular="0.3 0.3 0.3")
    s.light("sky", (-6, 8, 10), (0.3, -0.4, -1), "0.26 0.28 0.33")
    s.comment("graspables at the newsstand bench")
    P.free_box(s, "paper", -2.0, -3.1, 0.51, (0.14, 0.10, 0.012), "0.86 0.84 0.80 1", mass=0.3)
    P.free_cylinder(s, "togo_cup", 6.55, 1.2, 0.055, 0.04, 0.055, "0.92 0.92 0.94 1", mass=0.2)
    s.comment("traffic: two counter-rotating loops around the central block")
    ccw = [[-26.0, 4.3], [26.0, 4.3], [26.0, 20.8], [-26.0, 20.8]]      # eastbound south st
    cw = [[26.0, 7.3], [-26.0, 7.3], [-26.0, 24.3], [26.0, 24.3]]       # westbound south st
    A.vehicle(s, "taxi1", ccw, speed=6.5, s0=8.0, color="0.92 0.75 0.10 1",
              stops=[{"s": 36.0, "period": 24.0, "dur": 7.0, "offset": 5.0}],
              comment="yellow cab, eastbound, stops at the xwalk1 crosswalk")
    A.vehicle(s, "car1", ccw, speed=6.0, s0=55.0, color="0.16 0.30 0.55 1")
    A.vehicle(s, "car2", ccw, speed=6.8, s0=110.0, color="0.55 0.58 0.60 1")
    A.vehicle(s, "car3", cw, speed=6.2, s0=20.0, color="0.75 0.16 0.14 1",
              stops=[{"s": 40.0, "period": 27.0, "dur": 6.0, "offset": 13.0}],
              comment="westbound, pauses at the xwalk2 crosswalk")
    A.vehicle(s, "car4", cw, speed=7.2, s0=90.0, color="0.15 0.16 0.18 1")
    s.comment("pedestrians: sidewalk streams both ways + block loop")
    A.walker(s, "ped1", [[-27.0, -1.6], [27.0, -1.6]], closed=False, speed=1.35,
             s0=10.0, palette=0, comment="south sidewalk, west<->east stream")
    A.walker(s, "ped2", [[-27.0, 1.3], [27.0, 1.3]], closed=False, speed=1.2,
             s0=40.0, palette=2, scale=0.95)
    A.walker(s, "ped3", [[-27.0, -3.2], [27.0, -3.2]], closed=False, speed=1.5,
             s0=26.0, palette=5, scale=1.04)
    A.walker(s, "ped4", [[27.0, -0.4], [-27.0, -0.4]], closed=False, speed=1.1,
             s0=18.0, palette=7,
             stops=[{"s": 30.0, "period": 30.0, "dur": 6.0, "offset": 9.0}],
             comment="window shopper pausing under the awnings")
    A.walker(s, "ped5", A.rect_loop(-24.0, 10.2, 24.0, 17.8), speed=1.3, s0=20.0,
             palette=1, comment="central block loop (both streets' north sidewalk)")
    A.walker(s, "ped6", A.rect_loop(-24.0, 10.2, 24.0, 17.8), speed=1.25, s0=75.0, palette=4)
    A.walker(s, "ped7", [[10.0, -2.5], [10.0, 9.4]], closed=False, speed=1.15,
             s0=0.0, palette=3,
             stops=[{"s": 1.8, "period": 24.0, "dur": 8.0, "offset": 0.0},
                    {"s": 9.2, "period": 24.0, "dur": 8.0, "offset": 12.0}],
             comment="crosses the south street at xwalk1, waits for the cab gap")
    A.person_standing(s, "idle1", -13.5, -3.3, yaw=1.2, palette=6)
    A.person_standing(s, "idle2", 19.0, -0.8, yaw=-2.0, palette=4, scale=0.97)
    return s


def intersection():
    s = Scene("intersection", "Busy Intersection", "City & Streets",
              "A signalized four-way crossing: alternating N-S / E-W traffic "
              "with cars braking at the stop lines, pedestrians timed to the "
              "gaps, corner shops on every quadrant. The robot owns the SW "
              "corner.",
              headlight=("0.46 0.47 0.50", "0.28 0.28 0.30"),
              skybox=("0.42 0.58 0.78", "0.82 0.87 0.93"),
              fog=("0.74 0.78 0.85 1", 16, 42))
    C = 6.5                     # intersection center (x = y = C)
    s.comment("ground + crossing roads (center at %.1f, %.1f)" % (C, C))
    s.geom("floor", "plane", (0, 0, 0), (26, 26, 0.1), mat="m_sidewalk",
           extra='contype="1" conaffinity="1" friction="1.5 0.05 0.001"')
    s.geom("road_ew", "box", (0, C, 0.006), (25, 4.5, 0.006), mat="m_asphalt")
    s.geom("road_ns", "box", (C, 0, 0.006), (4.5, 25, 0.006), mat="m_asphalt")
    _lane_dashes(s, "dash_ew_w", -24, 0.5, C)
    _lane_dashes(s, "dash_ew_e", 12.5, 24, C)
    _lane_dashes(s, "dash_ns_s", -24, 0.5, C, along_x=False)
    _lane_dashes(s, "dash_ns_n", 12.5, 24, C, along_x=False)
    s.comment("crosswalks on all four approaches")
    s.geom("xw_w", "box", (0.9, C, 0.010), (1.1, 4.5, 0.004), mat="m_crosswalk", euler=(0, 0, 1.570795))
    s.geom("xw_e", "box", (12.1, C, 0.010), (1.1, 4.5, 0.004), mat="m_crosswalk", euler=(0, 0, 1.570795))
    s.geom("xw_s", "box", (C, 0.9, 0.010), (4.5, 1.1, 0.004), mat="m_crosswalk")
    s.geom("xw_n", "box", (C, 12.1, 0.010), (4.5, 1.1, 0.004), mat="m_crosswalk")
    s.comment("corner buildings, robot's SW corner kept open as a plaza")
    P.building(s, "bldg_sw", -9.0, -9.0, w=13, d=13, h=15)
    P.building(s, "bldg_se", 19.5, -8.0, w=12, d=14, h=21)
    P.building(s, "bldg_ne", 19.5, 20.0, w=12, d=12, h=17)
    P.building(s, "bldg_nw", -7.5, 20.0, w=15, d=12, h=24)
    P.awning(s, "awn_sw", -9.0, -2.4, yaw=0.0, L=6.0, mat="m_awning_green", z=2.6)
    P.awning(s, "awn_se", 19.5, -0.9, yaw=0.0, L=5.0, mat="m_awning_red", z=2.6)
    s.comment("signals + corner furniture (SW plaza is the spawn)")
    for i, (x, y, yaw) in enumerate(((1.4, 1.4, 0.7854), (11.6, 1.4, 2.3562),
                                     (11.6, 11.6, 3.927), (1.4, 11.6, -0.7854))):
        P.traffic_light(s, f"sig{i + 1}", x, y, yaw=yaw)
    P.streetlight(s, "sl1", -3.0, 1.2, yaw=0.0)
    P.streetlight(s, "sl2", 15.5, 11.8, yaw=math.pi)
    P.bench(s, "cbench", -2.5, -2.5, yaw=0.7854, L=1.6)
    P.plant(s, "cplanter", -4.5, -1.0, h=0.9)
    P.hydrant(s, "hyd", 13.3, -1.5)
    P.trash_can(s, "tc", -1.5, 13.2)
    s.light("sun", (5, -5, 9), (-0.3, 0.4, -1), "0.60 0.58 0.54", castshadow=True,
            specular="0.3 0.3 0.3")
    s.light("sky", (-5, 6, 9), (0.3, -0.35, -1), "0.27 0.29 0.34")
    s.comment("signal plan: 28 s cycle — EW green [0,10), NS green [14,24)")
    EW_RED = {"period": 28.0, "dur": 18.0, "offset": 10.0}
    NS_RED = {"period": 28.0, "dur": 18.0, "offset": 24.0}
    s.comment("through-traffic (teleport loops, ends fog-masked)")
    A.vehicle(s, "car_e1", [[-25.0, 4.25], [25.0, 4.25]], closed=False, teleport=True,
              speed=6.5, s0=4.0, color="0.75 0.16 0.14 1",
              stops=[{"s": 23.5, **EW_RED}], comment="eastbound through")
    A.vehicle(s, "car_e2", [[-25.0, 4.25], [25.0, 4.25]], closed=False, teleport=True,
              speed=6.5, s0=27.0, color="0.86 0.84 0.80 1", stops=[{"s": 23.5, **EW_RED}])
    A.vehicle(s, "car_w1", [[25.0, 8.75], [-25.0, 8.75]], closed=False, teleport=True,
              speed=6.0, s0=14.0, color="0.16 0.30 0.55 1",
              stops=[{"s": 11.5, **EW_RED}], comment="westbound through")
    A.vehicle(s, "car_n1", [[8.75, -25.0], [8.75, 25.0]], closed=False, teleport=True,
              speed=6.2, s0=9.0, color="0.92 0.75 0.10 1",
              stops=[{"s": 23.5, **NS_RED}], comment="northbound cab")
    A.vehicle(s, "car_n2", [[8.75, -25.0], [8.75, 25.0]], closed=False, teleport=True,
              speed=6.2, s0=33.0, color="0.55 0.58 0.60 1", stops=[{"s": 23.5, **NS_RED}])
    A.vehicle(s, "car_s1", [[4.25, 25.0], [4.25, -25.0]], closed=False, teleport=True,
              speed=5.8, s0=20.0, color="0.25 0.45 0.30 1",
              stops=[{"s": 11.5, **NS_RED}], comment="southbound through")
    s.comment("pedestrians: corner-to-corner ring timed against the signals")
    ring = [[0.9, 0.2], [12.1, 0.2], [13.0, 0.9], [13.0, 12.1], [12.1, 13.0],
            [0.9, 13.0], [0.2, 12.1], [0.2, 0.9]]
    # NS-road crossings happen on the s/n crosswalk legs -> red for peds while
    # NS traffic flows; EW-road crossings gated by EW flow, with walk margins.
    PED_NS = {"period": 28.0, "dur": 12.0, "offset": 13.0}   # blocks during NS green
    PED_EW = {"period": 28.0, "dur": 12.0, "offset": 27.0}   # blocks during EW green
    A.walker(s, "xped1", ring, speed=1.3, s0=1.0, palette=0,
             stops=[{"s": 0.5, **PED_NS}, {"s": 24.0, **PED_EW}],
             comment="clockwise corner ring, waits for the walk phase")
    A.walker(s, "xped2", ring, speed=1.15, s0=14.0, palette=3, scale=0.96,
             stops=[{"s": 12.6, **PED_EW}, {"s": 36.5, **PED_NS}])
    A.walker(s, "xped3", list(reversed(ring)), speed=1.4, s0=6.0, palette=5,
             stops=[{"s": 3.6, **PED_EW}, {"s": 27.5, **PED_NS}],
             comment="counter-clockwise ring")
    A.walker(s, "wped1", [[-24.0, -1.6], [-1.5, -1.6], [-1.0, -4.0], [-24.0, -4.2]],
             speed=1.2, s0=8.0, palette=2,
             comment="SW plaza promenade past the robot spawn")
    A.walker(s, "wped2", [[12.3, -24.0], [12.3, -2.0], [13.2, -1.5], [13.2, -24.0]],
             speed=1.25, s0=16.0, palette=6, comment="SE sidewalk out-and-back loop")
    A.person_standing(s, "waiting_ped", 1.6, -0.6, yaw=1.570795, palette=4)
    s.comment("curbside graspables")
    P.free_cylinder(s, "cone1", 2.4, 3.1, 0.18, 0.12, 0.18, "0.95 0.45 0.10 1", mass=0.8)
    P.free_box(s, "parcel", -3.2, 0.6, 0.12, (0.16, 0.12, 0.12), "0.72 0.58 0.42 1", mass=1.2)
    return s


def boulevard():
    s = Scene("boulevard", "Grand Boulevard", "City & Streets",
              "A 70 m four-lane boulevard with a planted median: a bus works "
              "its stop, cars overtake in both directions, pedestrians and a "
              "median promenade — the robot starts curbside by the bus "
              "shelter.",
              headlight=("0.47 0.48 0.51", "0.29 0.29 0.31"),
              skybox=("0.44 0.62 0.84", "0.84 0.89 0.95"),
              fog=("0.76 0.80 0.87 1", 24, 62))
    s.comment("ground: sidewalks + 4-lane asphalt with median (robot on south sidewalk)")
    s.geom("floor", "plane", (0, 6, 0), (36, 18, 0.1), mat="m_sidewalk",
           extra='contype="1" conaffinity="1" friction="1.5 0.05 0.001"')
    s.geom("road", "box", (0, 6.9, 0.006), (34, 5.9, 0.006), mat="m_asphalt")
    s.geom("median", "box", (0, 6.9, 0.025), (30, 0.8, 0.025), mat="m_concrete")
    s.geom("median_lawn", "box", (0, 6.9, 0.055), (29.4, 0.62, 0.008), rgba="0.40 0.60 0.34 1", collide=False)
    _lane_dashes(s, "dashS", -32, 32, 4.15)
    _lane_dashes(s, "dashN", -32, 32, 9.65)
    s.geom("xwalk", "box", (14.0, 6.9, 0.012), (1.4, 5.9, 0.004), mat="m_crosswalk", euler=(0, 0, 1.570795))
    s.comment("median trees + benches (promenade)")
    for i, x in enumerate((-24, -12, 0, 12, 24)):
        P.tree(s, f"mtree{i + 1}", x, 6.9, h=1.4, r=0.85)
    P.bench(s, "mbench1", -6.0, 6.9, yaw=0.0, L=1.6)
    P.bench(s, "mbench2", 6.0, 6.9, yaw=math.pi, L=1.6)
    s.comment("bus stop on the south sidewalk (robot spawn beside it)")
    s.geom("shelter_roof", "box", (5.5, -0.5, 2.45), (2.2, 0.8, 0.05), mat="m_steel")
    for i, (dx, dy) in enumerate(((-2.0, -0.6), (2.0, -0.6), (-2.0, 0.6), (2.0, 0.6))):
        s.geom(f"shelter_p{i + 1}", "box", (5.5 + dx, -0.5 + dy, 1.2), (0.05, 0.05, 1.2), mat="m_steel")
    s.geom("shelter_glass", "box", (5.5, -1.05, 1.35), (2.1, 0.02, 1.0), mat="m_glass")
    P.bench(s, "sbench", 5.5, -0.6, yaw=0.0, L=2.6, back=False)
    s.geom("busstop_sign", "cylinder", (3.2, 0.9, 1.4), (0.03, 1.4), mat="m_steel")
    s.geom("busstop_top", "box", (3.2, 0.9, 2.9), (0.25, 0.02, 0.15), mat="m_sig_amber", collide=False)
    s.comment("frontages")
    for i, (x, w, h) in enumerate(((-26, 12, 18), (-13, 12, 14), (0.5, 13, 22),
                                   (14, 12, 16), (27, 12, 20))):
        P.building(s, f"fS{i + 1}", x, -7.5, w=w, d=7.0, h=h)
    for i, (x, w, h) in enumerate(((-22, 16, 15), (-4, 17, 19), (14, 16, 13), (28, 10, 17))):
        P.building(s, f"fN{i + 1}", x, 20.5, w=w, d=7.0, h=h)
    P.awning(s, "awnS1", -13, -3.9, yaw=0.0, L=6.0, mat="m_awning_red", z=2.7)
    P.awning(s, "awnS2", 0.5, -3.9, yaw=0.0, L=6.5, mat="m_awning_green", z=2.7)
    for i, x in enumerate((-28, -14, 0, 14, 28)):
        P.streetlight(s, f"slS{i + 1}", x, 0.6, yaw=1.570795)
        P.streetlight(s, f"slN{i + 1}", x + 7, 13.2, yaw=-1.570795)
    P.hydrant(s, "hyd", -9.0, 0.6)
    P.trash_can(s, "tc1", 8.4, 0.7)
    s.light("sun", (8, -6, 11), (-0.3, 0.4, -1), "0.64 0.62 0.56", castshadow=True,
            specular="0.3 0.3 0.3")
    s.light("sky", (-8, 8, 11), (0.3, -0.35, -1), "0.27 0.29 0.34")
    s.comment("transit + traffic (u-turn loops beyond the fog line)")
    east_outer = [[-33.0, 2.6], [33.0, 2.6], [33.5, 11.2], [-33.5, 11.2]]
    east_inner = [[-33.0, 5.4], [33.0, 5.4], [33.5, 8.4], [-33.5, 8.4]]
    A.vehicle(s, "bus", east_outer, speed=7.0, s0=50.0, bus=True,
              color="0.20 0.45 0.75 1",
              stops=[{"s": 38.5, "period": 40.0, "dur": 12.0, "offset": 0.0}],
              comment="city bus: 12 s dwell at the shelter every 40 s")
    A.vehicle(s, "blvd_car1", east_inner, speed=8.5, s0=10.0, color="0.75 0.16 0.14 1")
    A.vehicle(s, "blvd_car2", east_inner, speed=7.8, s0=70.0, color="0.86 0.84 0.80 1",
              stops=[{"s": 47.0, "period": 26.0, "dur": 6.0, "offset": 8.0}],
              comment="inner-lane car pausing at the median crosswalk")
    A.vehicle(s, "blvd_car3", east_inner, speed=9.2, s0=110.0, color="0.15 0.16 0.18 1")
    A.vehicle(s, "blvd_car4", east_outer, speed=8.0, s0=140.0, color="0.80 0.62 0.12 1")
    s.comment("pedestrians")
    A.walker(s, "bped1", [[-30.0, -1.8], [30.0, -1.8]], closed=False, speed=1.3,
             s0=20.0, palette=1, comment="south sidewalk stream")
    A.walker(s, "bped2", [[30.0, -3.0], [-30.0, -3.0]], closed=False, speed=1.45,
             s0=36.0, palette=4, scale=1.03)
    A.walker(s, "bped3", [[-28.0, 13.6], [28.0, 13.6]], closed=False, speed=1.2,
             s0=12.0, palette=6, comment="north sidewalk")
    A.walker(s, "bped4", [[-27.0, 6.9], [27.0, 6.9]], closed=False, speed=1.0,
             s0=30.0, palette=2, scale=0.95,
             stops=[{"s": 21.0, "period": 30.0, "dur": 7.0, "offset": 6.0}],
             comment="median promenade, rests at the benches")
    A.walker(s, "bped5", [[14.0, -2.2], [14.0, 15.0]], closed=False, speed=1.15,
             s0=3.0, palette=7,
             stops=[{"s": 2.6, "period": 26.0, "dur": 7.0, "offset": 12.0},
                    {"s": 14.5, "period": 26.0, "dur": 7.0, "offset": 0.0}],
             comment="crosses all four lanes at the median crosswalk")
    A.person_seated(s, "waiting_bus", 5.5, -0.6, yaw=1.570795, palette=3, seat_h=0.47)
    A.person_standing(s, "waiting_bus2", 4.2, -0.2, yaw=1.2, palette=5)
    s.comment("curbside graspables")
    P.free_box(s, "delivery", 2.2, -1.4, 0.14, (0.18, 0.14, 0.14), "0.72 0.58 0.42 1", mass=1.5)
    P.free_cylinder(s, "umbrella_drop", 7.1, -1.1, 0.05, 0.03, 0.35, "0.16 0.30 0.55 1",
                    mass=0.4, yaw=0.6)
    return s


def plaza():
    s = Scene("plaza", "City Plaza", "City & Streets",
              "A pedestrian plaza around a fountain: crowds criss-crossing "
              "with pauses at the water, a coffee kiosk, benches, planters "
              "and pigeons — no cars, all people.",
              headlight=("0.50 0.50 0.52", "0.30 0.30 0.30"),
              skybox=("0.46 0.66 0.88", "0.86 0.91 0.97"),
              fog=("0.80 0.84 0.90 1", 20, 50))
    s.comment("paved plaza 30 x 30 m ringed by low arcade facades")
    s.geom("floor", "plane", (0, 0, 0), (16, 16, 0.1), mat="m_sidewalk",
           extra='contype="1" conaffinity="1" friction="1.5 0.05 0.001"')
    for i, (x, y, w, d, yaw) in enumerate(((0, 15.2, 26, 3.5, 0.0), (0, -15.2, 26, 3.5, 0.0),
                                           (15.2, 0, 3.5, 26, 0.0), (-15.2, 0, 3.5, 26, 0.0))):
        P.building(s, f"arc{i + 1}", x, y, w=w, d=d, h=9, shopfront=True)
    s.comment("fountain (offset so the robot spawn stays clear)")
    FX, FY = 5.0, 3.5
    s.geom("ftn_base", "cylinder", (FX, FY, 0.22), (2.6, 0.22), mat="m_concrete")
    s.geom("ftn_water", "cylinder", (FX, FY, 0.40), (2.3, 0.04), mat="m_water", collide=False)
    s.geom("ftn_column", "cylinder", (FX, FY, 0.85), (0.35, 0.45), mat="m_concrete")
    s.geom("ftn_bowl", "cylinder", (FX, FY, 1.35), (0.95, 0.06), mat="m_concrete", collide=False)
    s.geom("ftn_jet", "cylinder", (FX, FY, 1.75), (0.08, 0.35), rgba="0.75 0.88 0.95 0.6", collide=False)
    s.comment("kiosk, benches, planters, lamps")
    s.geom("kiosk_base", "box", (-7.5, 6.0, 0.75), (1.3, 1.0, 0.75), mat="m_wood")
    P.awning(s, "kiosk_awn", -7.5, 6.8, yaw=0.0, L=2.9, mat="m_awning_red", z=2.3)
    s.geom("kiosk_menu", "box", (-7.5, 5.0, 1.55), (0.8, 0.02, 0.3), mat="m_menu", collide=False)
    for i, (x, y, yaw) in enumerate(((1.0, 8.5, 0.0), (10.5, 3.5, 1.570795),
                                     (5.0, -3.0, math.pi), (-3.0, -6.5, -0.6),
                                     (-8.0, -1.0, 1.0))):
        P.bench(s, f"pbench{i + 1}", x, y, yaw=yaw, L=1.8)
    for i, (x, y) in enumerate(((-11.5, 10.5), (11.5, 10.5), (11.5, -10.5), (-11.5, -10.5))):
        s.geom(f"pl{i + 1}_pot", "box", (x, y, 0.32), (0.8, 0.8, 0.32), mat="m_concrete")
        P.tree(s, f"pl{i + 1}_tree", x, y, h=1.1, r=0.75)
    for i, (x, y) in enumerate(((-6.0, 10.0), (9.0, 8.5), (12.0, -4.0), (-10.0, -8.0), (0.0, -11.0))):
        s.geom(f"plamp{i + 1}_pole", "cylinder", (x, y, 1.8), (0.05, 1.8), mat="m_dark")
        s.geom(f"plamp{i + 1}_glob", "sphere", (x, y, 3.75), (0.16,), mat="m_lamp", collide=False)
    s.comment("pigeons")
    for i, (x, y) in enumerate(((3.0, 6.8), (3.6, 7.2), (7.8, 0.4), (-2.5, 4.0))):
        s.geom(f"pigeon{i + 1}", "sphere", (x, y, 0.055), (0.055,), rgba="0.52 0.54 0.58 1", collide=False)
        s.geom(f"pigeon{i + 1}_head", "sphere", (x + 0.06, y + 0.02, 0.12), (0.03,),
               rgba="0.35 0.40 0.48 1", collide=False)
    s.light("sun", (5, -4, 9), (-0.3, 0.35, -1), "0.66 0.64 0.58", castshadow=True,
            specular="0.3 0.3 0.3")
    s.light("sky", (-5, 5, 9), (0.3, -0.3, -1), "0.28 0.30 0.35")
    s.comment("the crowd — loops, orbits and diagonal crossers, all deterministic")
    A.walker(s, "pped1", A.rect_loop(-12.5, -12.0, 12.5, 12.0), speed=1.3, s0=8.0,
             palette=0, comment="perimeter promenade CCW")
    A.walker(s, "pped2", A.rect_loop(-12.5, -12.0, 12.5, 12.0), speed=1.2, s0=45.0, palette=1)
    A.walker(s, "pped3", list(reversed(A.rect_loop(-11.5, -11.0, 11.5, 11.0))),
             speed=1.4, s0=20.0, palette=2, comment="perimeter CW, brisk")
    A.walker(s, "pped4", [[FX + 4.2, FY], [FX, FY + 4.2], [FX - 4.2, FY], [FX, FY - 4.2]],
             speed=0.95, s0=3.0, palette=3, scale=0.95,
             stops=[{"s": 4.0, "period": 24.0, "dur": 7.0, "offset": 5.0}],
             comment="fountain orbit with a long look at the water")
    A.walker(s, "pped5", [[-13.0, -9.0], [-2.0, -2.5], [9.0, 9.5], [12.5, 12.0]],
             closed=False, speed=1.35, s0=10.0, palette=4,
             comment="diagonal crosser, passes near the spawn")
    A.walker(s, "pped6", [[13.0, -10.0], [4.0, -4.0], [-6.0, 3.0], [-12.5, 9.0]],
             closed=False, speed=1.25, s0=22.0, palette=5, scale=1.05)
    A.walker(s, "pped7", [[-7.5, 4.6], [-4.0, 1.5], [1.0, 7.3], [-4.5, 8.6]],
             speed=0.9, s0=6.0, palette=6, scale=0.92,
             stops=[{"s": 0.5, "period": 28.0, "dur": 9.0, "offset": 14.0}],
             comment="kiosk regular: queue, coffee, lap around")
    A.walker(s, "pped8", [[10.5, -8.0], [6.5, -9.5], [1.5, -8.5], [6.0, -6.0]],
             speed=1.05, s0=0.0, palette=7)
    A.person_seated(s, "resting1", 1.0, 8.28, yaw=-1.570795, palette=6, seat_h=0.47)
    A.person_seated(s, "resting2", 5.0, -2.78, yaw=1.570795, palette=1, seat_h=0.47)
    A.person_standing(s, "tourist", 8.2, 5.6, yaw=-2.4, palette=2)
    s.comment("graspables: gelato cup on a bench, lost scarf")
    P.free_cylinder(s, "gelato", 1.35, 8.42, 0.53, 0.04, 0.05, "0.95 0.80 0.70 1", mass=0.15)
    P.free_box(s, "scarf", -2.0, -3.2, 0.03, (0.14, 0.09, 0.02), "0.75 0.28 0.22 1", mass=0.1)
    return s


def build_all():
    return [city_sidewalk(), intersection(), boulevard(), plaza()]
