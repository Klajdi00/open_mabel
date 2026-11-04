"""scenes_outdoor.py -- Outdoor & Leisure + the airport terminal: golf course
(BIG), front yard, rooftop lounge at dusk, airport concourse (BIG, dynamic)."""
import math

from .core import Scene
from . import prefabs as P
from . import actors as A


def golf():
    s = Scene("golf", "Golf Course", "Outdoor & Leisure",
              "A 40 m par-3 hole: striped fairway to a raised green with pin "
              "and real golf balls, two sand bunkers, cart path with parked "
              "cart, tee boxes and tree lines.",
              headlight=("0.52 0.54 0.56", "0.30 0.30 0.30"),
              skybox=("0.45 0.68 0.93", "0.85 0.92 1.0"),
              fog=("0.78 0.86 0.95 1", 18, 55))
    s.comment("fairway 44 x 30 m — robot spawns at the tee boxes")
    P.floor(s, 44, 30, "m_fairway")
    s.light("sun", (5, 3, 8), (-0.35, -0.25, -1), "0.72 0.70 0.62", castshadow=True,
            specular="0.3 0.3 0.3")
    s.light("sky", (-5, 0, 8), (0.3, 0.0, -1), "0.28 0.31 0.38")
    s.comment("green + pin (18 m east)")
    s.geom("green", "cylinder", (15.0, 2.0, 0.012), (4.2, 0.012), rgba="0.36 0.62 0.30 1")
    s.geom("fringe", "cylinder", (15.0, 2.0, 0.006), (4.9, 0.006), rgba="0.42 0.66 0.34 1")
    s.geom("cup", "cylinder", (15.8, 2.4, 0.028), (0.054, 0.004), rgba="0.10 0.10 0.10 1", collide=False)
    s.geom("pin", "cylinder", (15.8, 2.4, 1.15), (0.014, 1.12), mat="m_white")
    s.geom("pin_flag", "box", (15.97, 2.4, 2.05), (0.16, 0.005, 0.11), rgba="0.85 0.15 0.12 1", collide=False)
    s.comment("bunkers")
    s.geom("bunker1", "cylinder", (11.0, -1.6, 0.008), (2.4, 0.008), mat="m_sand")
    s.geom("bunker2", "cylinder", (16.5, 5.8, 0.008), (1.8, 0.008), mat="m_sand")
    s.geom("bunker1_lip", "cylinder", (11.0, -1.6, 0.004), (2.7, 0.004), rgba="0.52 0.70 0.40 1", collide=False)
    s.comment("tee boxes (spawn) + markers")
    s.geom("teebox", "box", (-1.8, 0.0, 0.008), (1.6, 1.2, 0.008), rgba="0.40 0.64 0.32 1")
    for i, y in enumerate((-0.8, 0.8)):
        s.geom(f"teemark{i + 1}", "sphere", (-1.2, y, 0.06), (0.05,), rgba="0.85 0.15 0.12 1")
    s.comment("cart path + golf cart")
    for i, (x, y, L, yaw) in enumerate(((-2.0, -5.0, 6.0, 0.0), (5.0, -6.2, 8.5, -0.12),
                                        (13.0, -5.2, 8.0, 0.25))):
        s.geom(f"cartpath{i + 1}", "box", (x, y, 0.006), (L / 2, 0.9, 0.006), euler=(0, 0, yaw), mat="m_gravel")
    s.geom("cart_body", "box", (4.0, -5.6, 0.55), (1.05, 0.62, 0.16), mat="m_white")
    s.geom("cart_roof", "box", (4.0, -5.6, 1.55), (0.95, 0.60, 0.03), mat="m_white")
    for i, (dx, dy) in enumerate(((0.75, 0.55), (0.75, -0.55), (-0.75, 0.55), (-0.75, -0.55))):
        s.geom(f"cart_p{i + 1}", "box", (4.0 + dx * 0.9, -5.6 + dy, 1.0), (0.03, 0.03, 0.52), mat="m_dark")
        s.geom(f"cart_w{i + 1}", "cylinder", (4.0 + dx, -5.6 + dy, 0.22), (0.22, 0.09),
               euler=(1.570795, 0, 0), mat="m_dark")
    s.geom("cart_seat", "box", (4.35, -5.6, 0.78), (0.35, 0.55, 0.08), mat="m_fabric")
    s.comment("tree lines + rough")
    for i, (x, y, h, r) in enumerate(((-6, 8, 1.8, 1.1), (0, 10, 1.5, 0.95), (7, 9.5, 2.0, 1.2),
                                      (14, 10, 1.6, 1.0), (20, 7, 1.8, 1.1), (-8, -8, 1.6, 1.0),
                                      (0, -10.5, 1.9, 1.15), (9, -10, 1.5, 0.9), (19, -9, 1.7, 1.05),
                                      (-12, 2, 2.0, 1.2), (21, -1, 1.9, 1.15))):
        P.tree(s, f"tree{i + 1}", x, y, h=h, r=r)
    for i, (x, y) in enumerate(((-6, 4.5), (3, 7.0), (8, -6.8))):
        P.hedge(s, f"rough{i + 1}", x, y, yaw=0.4 * i, L=2.4, h=0.28)
    s.geom("bench_marker", "box", (-3.5, 2.6, 0.25), (0.5, 0.2, 0.04), mat="m_wood")
    s.geom("bench_marker_l1", "box", (-3.9, 2.6, 0.11), (0.04, 0.18, 0.11), mat="m_dark")
    s.geom("bench_marker_l2", "box", (-3.1, 2.6, 0.11), (0.04, 0.18, 0.11), mat="m_dark")
    s.comment("golf balls (graspable)")
    P.free_sphere(s, "ball_tee", -1.2, 0.0, 0.021, 0.021, "0.95 0.95 0.92 1", mass=0.046)
    P.free_sphere(s, "ball_green1", 14.2, 1.4, 0.021, 0.021, "0.95 0.95 0.92 1", mass=0.046)
    P.free_sphere(s, "ball_bunker", 10.6, -1.2, 0.021, 0.021, "0.95 0.95 0.92 1", mass=0.046)
    return s


def front_yard():
    s = Scene("front_yard", "Front Yard", "Outdoor & Leisure",
              "Suburban front yard: house facade with porch, lawn, driveway "
              "with a parked car, picket fence, flower beds, mailbox — and a "
              "kid's ball to fetch.",
              headlight=("0.52 0.53 0.55", "0.30 0.30 0.30"),
              skybox=("0.45 0.68 0.93", "0.85 0.92 1.0"),
              fog=("0.80 0.87 0.95 1", 12, 40))
    s.comment("lawn 18 x 13 m; house along the north edge")
    P.floor(s, 18, 13, "m_grass")
    s.light("sun", (4, -3, 7), (-0.3, 0.3, -1), "0.72 0.70 0.62", castshadow=True,
            specular="0.3 0.3 0.3")
    s.light("sky", (-4, 0, 7), (0.3, 0.0, -1), "0.28 0.31 0.38")
    s.comment("house facade (brick, porch, door, windows)")
    s.geom("house_face", "box", (0.0, 5.9, 2.6), (7.5, 0.35, 2.6), mat="m_brick")
    s.geom("house_roofline", "box", (0.0, 5.85, 5.36), (7.9, 0.55, 0.16), mat="m_dark", collide=False)
    s.geom("house_door", "box", (-1.6, 5.52, 1.05), (0.55, 0.04, 1.05), mat="m_wood")
    s.geom("house_knob", "sphere", (-1.25, 5.47, 1.0), (0.03,), mat="m_chrome", collide=False)
    for i, x in enumerate((-4.8, 1.6, 4.6)):
        s.geom(f"house_win{i + 1}", "box", (x, 5.52, 1.9), (0.9, 0.03, 0.75), mat="m_storefront", collide=False)
        s.geom(f"house_sill{i + 1}", "box", (x, 5.48, 1.1), (1.0, 0.05, 0.05), mat="m_white", collide=False)
    s.comment("porch + steps")
    s.geom("porch", "box", (-1.6, 4.9, 0.22), (1.5, 0.7, 0.22), mat="m_concrete")
    s.geom("porch_step1", "box", (-1.6, 4.06, 0.075), (1.0, 0.16, 0.075), mat="m_concrete")
    s.geom("porch_step2", "box", (-1.6, 4.38, 0.15), (1.0, 0.16, 0.15), mat="m_concrete")
    for i, x in enumerate((-2.9, -0.3)):
        s.geom(f"porch_col{i + 1}", "box", (x, 4.35, 1.55), (0.09, 0.09, 1.1), mat="m_white")
    s.geom("porch_roof", "box", (-1.6, 4.9, 2.72), (1.7, 0.85, 0.07), mat="m_dark")
    s.comment("driveway + parked car (east)")
    s.geom("driveway", "box", (5.4, 0.6, 0.007), (1.9, 5.2, 0.007), mat="m_concrete")
    A.parked_car(s, "familycar", 5.4, 2.2, yaw=1.570795, color="0.16 0.30 0.55 1")
    s.comment("front walk + flower beds + trees")
    s.geom("frontwalk", "box", (-1.6, 0.6, 0.006), (0.7, 3.4, 0.006), mat="m_sidewalk")
    for i, (x, y) in enumerate(((-4.6, 4.6), (2.2, 4.6))):
        s.geom(f"bed{i + 1}", "box", (x, y, 0.05), (1.6, 0.5, 0.05), rgba="0.36 0.25 0.18 1")
        for k in range(4):
            s.geom(f"bed{i + 1}_fl{k + 1}", "sphere", (x - 1.2 + k * 0.8, y, 0.18), (0.09,),
                   rgba=("0.85 0.30 0.40 1" if (i + k) % 2 else "0.95 0.80 0.25 1"), collide=False)
    P.tree(s, "yardtree1", -6.5, 1.5, h=1.9, r=1.15)
    P.tree(s, "yardtree2", 7.6, -4.6, h=1.5, r=0.95)
    P.hedge(s, "hedge1", -4.2, 3.6, yaw=0.0, L=3.2, h=0.55)
    s.comment("picket fence along the street (south) with a gate gap")
    P.fence_picket(s, "fenceW", -5.6, -5.9, yaw=0.0, L=6.0)
    P.fence_picket(s, "fenceE", 2.2, -5.9, yaw=0.0, L=5.2)
    P.fence_picket(s, "fenceL", -8.7, -1.0, yaw=1.570795, L=9.6)
    s.comment("mailbox + hose reel + trash bins")
    s.geom("mail_post", "box", (-1.0, -5.6, 0.5), (0.04, 0.04, 0.5), mat="m_wood")
    s.geom("mail_box", "box", (-1.0, -5.6, 1.12), (0.22, 0.12, 0.12), rgba="0.16 0.30 0.55 1")
    s.geom("mail_flag", "box", (-0.85, -5.47, 1.22), (0.02, 0.015, 0.07), rgba="0.85 0.15 0.12 1", collide=False)
    P.trash_can(s, "bin1", 7.9, 4.4)
    P.trash_can(s, "bin2", 7.3, 4.6, mat="m_frame")
    s.comment("toys on the lawn (graspable)")
    P.free_sphere(s, "kidball", -3.4, -1.8, 0.11, 0.11, "0.85 0.30 0.20 1", mass=0.25)
    P.free_box(s, "toytruck", -2.6, -2.6, 0.06, (0.09, 0.05, 0.045), "0.90 0.55 0.15 1", mass=0.3)
    P.free_cylinder(s, "wateringcan", 3.6, 3.3, 0.13, 0.09, 0.12, "0.30 0.55 0.38 1", mass=0.5)
    return s


def rooftop_lounge():
    s = Scene("rooftop_lounge", "Rooftop Lounge", "Outdoor & Leisure",
              "Dusk rooftop bar: glowing pool, loungers and parasols, bar "
              "counter with stools, planters, string lights and a skyline of "
              "lit facades beyond the parapet.",
              headlight=("0.34 0.33 0.38", "0.22 0.20 0.24"),
              skybox=("0.30 0.28 0.45", "0.86 0.52 0.28"),
              fog=("0.42 0.34 0.42 1", 16, 48))
    s.comment("deck 20 x 14 m with parapet walls")
    P.floor(s, 20, 14, "m_concrete")
    for nm, p, sz in (("wall_par_n", (0, 7.05, 0.6), (10.1, 0.08, 0.6)),
                      ("wall_par_s", (0, -7.05, 0.6), (10.1, 0.08, 0.6)),
                      ("wall_par_e", (10.05, 0, 0.6), (0.08, 7.0, 0.6)),
                      ("wall_par_w", (-10.05, 0, 0.6), (0.08, 7.0, 0.6))):
        s.geom(nm, "box", p, sz, mat="m_concrete")
    s.light("dusk", (4, -4, 6), (-0.3, 0.35, -1), "0.40 0.30 0.24", castshadow=True)
    s.light("ambient", (-4, 3, 6), (0.3, -0.2, -1), "0.16 0.15 0.20")
    s.comment("pool (east) with deck rim")
    s.geom("pool_rim", "box", (5.5, 1.0, 0.09), (3.4, 2.4, 0.09), mat="m_white")
    s.geom("pool_water", "box", (5.5, 1.0, 0.14), (3.1, 2.1, 0.045), mat="m_water", collide=False)
    s.geom("pool_ladder", "box", (2.35, 1.0, 0.45), (0.03, 0.25, 0.30), mat="m_chrome", collide=False)
    s.comment("loungers + parasols along the pool")
    for i, y in enumerate((-2.6, -3.9)):
        s.geom(f"lng{i + 1}_seat", "box", (5.0, y, 0.30), (0.95, 0.32, 0.045), mat="m_fabric", euler=(0, -0.12, 0))
        s.geom(f"lng{i + 1}_legs", "box", (5.0, y, 0.14), (0.7, 0.26, 0.10), mat="m_dark")
    P.parasol(s, "para1", 6.4, -3.2, mat="m_awning_red")
    P.parasol(s, "para2", 2.2, 4.6, mat="m_awning_red")
    s.comment("bar (west) + stools + pergola of string lights")
    P.counter(s, "bar", -6.0, 1.2, yaw=1.570795, L=3.6, top="m_marble", body="m_wood")
    for i in range(4):
        P.stool(s, f"bst{i + 1}", -5.15, -0.3 + i * 1.0)
    s.geom("backbar", "box", (-6.85, 1.2, 1.3), (0.18, 1.7, 0.55), mat="m_wood")
    s.geom("bottles", "box", (-6.8, 1.2, 1.98), (0.12, 1.5, 0.12), mat="m_book", collide=False)
    for i in range(6):
        x = -6.0 + i * 2.4
        s.geom(f"string{i + 1}", "sphere", (x, 3.2, 2.6), (0.05,), mat="m_lamp", collide=False)
        s.geom(f"string_b{i + 1}", "sphere", (x + 1.2, 3.0, 2.45), (0.05,), mat="m_lamp", collide=False)
    for i, (x, y) in enumerate(((-8.6, 5.6), (-8.6, -5.6), (8.6, 5.8), (0.4, 5.9))):
        s.geom(f"planter{i + 1}", "box", (x, y, 0.28), (0.55, 0.4, 0.28), mat="m_dark")
        P.hedge(s, f"planter{i + 1}_shrub", x, y, L=1.0, h=0.5)
    s.comment("fire table + seating (south-west social corner)")
    s.geom("fire_table", "cylinder", (-3.2, -3.6, 0.32), (0.55, 0.32), mat="m_concrete")
    s.geom("fire_flame", "sphere", (-3.2, -3.6, 0.72), (0.13,), mat="m_sig_amber", collide=False)
    P.sofa(s, "fire_sofa", -4.6, -3.6, yaw=0.0, L=2.0, mat="m_fabric")
    P.bench(s, "fire_bench", -3.2, -2.2, yaw=math.pi, L=1.8, seat="m_wood")
    s.comment("skyline beyond the parapet")
    for i, (x, y, w, h) in enumerate(((-16, 12, 5, 18), (-6, 14, 6, 24), (4, 13, 5, 15),
                                      (13, 14, 7, 21), (18, 4, 5, 26), (17, -8, 5, 13))):
        P.building(s, f"sky{i + 1}", x, y, w=w, d=4.5, h=h, shopfront=False)
    s.comment("guests + graspables")
    A.person_standing(s, "guest1", -4.9, 0.3, yaw=0.6, palette=4)
    A.person_seated(s, "guest2", -4.05, -3.6, yaw=0.0, palette=5, seat_h=0.48)
    P.free_cylinder(s, "cocktail", -5.65, 1.0, 0.99, 0.032, 0.06, "0.90 0.55 0.15 0.8", mass=0.2)
    P.free_cylinder(s, "tumbler", -3.2, -3.5, 0.70, 0.035, 0.045, "0.62 0.74 0.80 0.6", mass=0.2)
    return s


def airport():
    s = Scene("airport", "Airport Terminal", "Work & Public",
              "A 36 m departure concourse: check-in island, security lane, "
              "gate seating, duty-free kiosk, departure boards — travelers "
              "with roller bags criss-crossing, a jet at the glass.",
              headlight=("0.50 0.50 0.54", "0.30 0.30 0.32"))
    s.comment("concourse 36 x 18 m")
    P.floor(s, 36, 18, "m_tile")
    for nm, p, sz in (("wall_back", (0, -9.05, 3.0), (18.1, 0.08, 3.0)),
                      ("wall_e", (18.05, 0, 3.0), (0.08, 9.0, 3.0)),
                      ("wall_w", (-18.05, 0, 3.0), (0.08, 9.0, 3.0))):
        s.geom(nm, "box", p, sz, mat="m_wall")
    s.comment("north curtain wall onto the apron (glass) + jet outside")
    for i in range(9):
        x = -16 + i * 4.0
        s.geom(f"mullion{i + 1}", "box", (x, 9.0, 3.0), (0.09, 0.09, 3.0), mat="m_dark")
    s.geom("curtain_glass", "box", (0, 9.06, 3.0), (18.0, 0.02, 3.0), mat="m_glass")
    s.geom("apron", "box", (0, 16.0, 0.004), (24.0, 7.0, 0.004), mat="m_asphalt")
    s.geom("jet_fuse", "cylinder", (2.0, 15.5, 2.6), (1.35, 9.0), euler=(0, 1.570795, 0), mat="m_white")
    s.geom("jet_nose", "sphere", (11.0, 15.5, 2.6), (1.33,), mat="m_white", collide=False)
    s.geom("jet_tail", "box", (-8.6, 15.5, 4.6), (1.5, 0.12, 1.9), mat="m_frame", collide=False)
    s.geom("jet_wing", "box", (1.0, 13.4, 1.9), (1.1, 3.4, 0.09), mat="m_steel", euler=(0, 0, -0.35), collide=False)
    s.geom("jet_engine", "cylinder", (2.4, 12.6, 1.35), (0.55, 1.1), euler=(0, 1.570795, 0), mat="m_steel", collide=False)
    s.geom("jetbridge", "box", (6.5, 11.2, 2.4), (0.9, 2.4, 0.75), mat="m_steel")
    s.comment("check-in island (west) + queue posts")
    for i in range(3):
        y = -2.0 + i * 2.4
        P.counter(s, f"checkin{i + 1}", -14.5, y, yaw=1.570795, L=1.8, top="m_white", body="m_dark")
        s.geom(f"checkin{i + 1}_scr", "box", (-14.5, y + 0.55, 1.55), (0.30, 0.02, 0.22), mat="m_screen", collide=False)
    for i in range(5):
        s.geom(f"qpost{i + 1}", "cylinder", (-12.4, -3.0 + i * 1.5, 0.5), (0.04, 0.5), mat="m_chrome")
        s.geom(f"qbelt{i + 1}", "box", (-12.4, -2.25 + i * 1.5, 0.88), (0.03, 0.72, 0.03),
               rgba="0.75 0.16 0.14 1", collide=False)
    s.comment("departure board + totems")
    s.geom("depboard", "box", (-8.0, -8.9, 4.0), (3.2, 0.06, 1.0), mat="m_menu", collide=False)
    s.geom("depboard_posts", "box", (-8.0, -8.85, 1.5), (2.8, 0.04, 1.5), mat="m_dark")
    for i, x in enumerate((-4.0, 6.0)):
        s.geom(f"totem{i + 1}", "box", (x, -5.0, 1.1), (0.35, 0.10, 1.1), mat="m_dark")
        s.geom(f"totem{i + 1}_scr", "box", (x, -4.88, 1.35), (0.28, 0.015, 0.6), mat="m_screen", collide=False)
    s.comment("gate seating rows (east)")
    for i in range(3):
        y = 1.0 + i * 2.6
        for j in range(2):
            x = 9.0 + j * 5.2
            P.bench(s, f"seat{i}{j}", x, y, yaw=0.0, L=3.6, seat="m_fabric", back=True)
    s.geom("gate_desk", "box", (16.2, 6.8, 0.55), (1.1, 0.45, 0.55), mat="m_white")
    s.geom("gate_sign", "box", (16.2, 6.8, 2.5), (0.7, 0.06, 0.35), mat="m_menu", collide=False)
    s.comment("duty-free kiosk (center-south)")
    s.geom("kiosk_base", "box", (1.0, -6.8, 0.55), (2.2, 1.1, 0.55), mat="m_wood")
    s.geom("kiosk_shelf", "box", (1.0, -6.8, 1.6), (2.0, 0.9, 0.45), mat="m_products")
    P.awning(s, "kiosk_awn", 1.0, -5.9, yaw=0.0, L=4.2, mat="m_awning_red", z=2.5)
    P.plant(s, "palm1", -5.5, 6.5, h=1.4)
    P.plant(s, "palm2", 5.0, -3.2, h=1.4)
    for i, x in enumerate((-12.0, -4.0, 4.0, 12.0)):
        P.ceiling_lamp(s, f"bay{i + 1}", x, 0.0, z=5.4, r=0.4)
    s.light("key", (4.0, 4.0, 6.5), (-0.25, -0.3, -1), "0.5 0.5 0.52", castshadow=True)
    s.light("fill", (-6.0, -4.0, 6.5), (0.3, 0.3, -1), "0.26 0.26 0.28")
    s.comment("luggage props + graspables")
    s.geom("cart_frame", "box", (-10.0, 5.6, 0.5), (0.35, 0.25, 0.04), mat="m_chrome")
    s.geom("cart_handle", "box", (-10.35, 5.6, 0.95), (0.03, 0.25, 0.45), mat="m_chrome")
    P.free_box(s, "suitcase_red", -10.0, 5.6, 0.75, (0.28, 0.18, 0.22), "0.75 0.16 0.14 1", mass=6.0)
    P.free_box(s, "duffel", 9.6, 0.35, 0.62, (0.22, 0.13, 0.13), "0.30 0.55 0.38 1", mass=3.0)
    P.free_cylinder(s, "coffee_cup", 9.2, 1.35, 0.53, 0.035, 0.055, "0.86 0.84 0.80 1", mass=0.2)
    s.comment("travelers criss-crossing the concourse")
    A.walker(s, "trav1", [[-15.0, -5.5], [-6.0, -3.5], [4.0, -3.5], [14.0, -4.5],
                          [15.5, 2.0], [8.0, 5.8], [-2.0, 5.5], [-11.0, 4.5]],
             speed=1.25, s0=4.0, palette=1, comment="traveler: check-in to gates loop")
    A.walker(s, "trav2", [[-15.0, -5.5], [-6.0, -3.5], [4.0, -3.5], [14.0, -4.5],
                          [15.5, 2.0], [8.0, 5.8], [-2.0, 5.5], [-11.0, 4.5]],
             speed=1.45, s0=38.0, palette=3, scale=1.03,
             comment="traveler: same loop, opposite phase, hurried")
    A.walker(s, "trav3", [[13.0, 7.6], [6.0, 6.9], [-1.0, 7.2], [-8.0, 6.8], [-14.5, 7.0]],
             closed=False, speed=1.0, s0=6.0, palette=5,
             stops=[{"s": 20.0, "period": 34.0, "dur": 8.0, "offset": 10.0}],
             comment="window shopper pacing the glass, pausing at the jet")
    A.walker(s, "trav4", [[-2.5, -7.8], [-2.0, -5.6], [3.0, -5.3], [6.5, -6.6],
                          [3.5, -7.9]],
             speed=0.85, s0=2.0, palette=6, scale=0.92,
             stops=[{"s": 9.0, "period": 27.0, "dur": 6.0, "offset": 5.0}],
             comment="duty-free browser orbiting the kiosk")
    A.walker(s, "trav5", [[-11.5, 1.6], [-6.0, 1.3], [0.0, 1.2], [6.0, 1.3], [11.5, 1.6],
                          [6.0, 2.8], [0.0, 2.6], [-6.0, 2.9]],
             speed=1.1, s0=15.0, palette=0,
             comment="mid-concourse wanderer (crosses the robot's spawn aisle)")
    A.person_seated(s, "waiting1", 9.75, 3.6, yaw=-1.570795, palette=2, seat_h=0.48)
    A.person_seated(s, "waiting2", 13.6, 0.98, yaw=1.570795, palette=4, seat_h=0.48)
    A.person_standing(s, "gate_agent", 16.2, 5.9, yaw=-1.570795, palette=7)
    return s


def build_all():
    return [golf(), front_yard(), rooftop_lounge(), airport()]
