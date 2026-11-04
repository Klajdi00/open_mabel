"""scenes_food.py -- Food & Dining + retail: supermarket (dynamic shoppers),
two cafes, casual restaurant (waiter), fast food, fine dining (waiter)."""
import math

from .core import Scene
from . import prefabs as P
from . import actors as A


def supermarket():
    s = Scene("supermarket", "Supermarket", "Work & Public",
              "Grocery store: three product aisles, produce tables with real "
              "fruit to pick, a glass cooler wall and two checkout lanes — "
              "with shoppers wandering the aisles.",
              headlight=("0.52 0.52 0.54", "0.30 0.30 0.30"))
    s.comment("hall 16 x 11 m")
    P.room_shell(s, 16, 11, 3.4, "m_tile", "m_wall")
    s.comment("gondola aisles (aisle pitch 2.5 m, robot spawns in the middle aisle)")
    for i, y in enumerate((-3.75, -1.25, 1.25, 3.75)):
        if i == 1:
            continue                              # keep the spawn aisle open
        P.gondola(s, f"gnd{i}", -2.0, y, yaw=0.0, L=7.0)
    P.gondola(s, "gnd1w", -4.5, -1.25, yaw=0.0, L=2.0)   # short shelf, spawn stays clear
    s.comment("produce corner (east)")
    for j, (x, y) in enumerate(((5.2, 2.6), (5.2, 0.8), (6.6, 1.7))):
        s.geom(f"produce{j}_tbl", "box", (x, y, 0.42), (0.6, 0.5, 0.06), mat="m_wood")
        s.geom(f"produce{j}_leg", "box", (x, y, 0.18), (0.45, 0.35, 0.18), mat="m_dark")
        s.geom(f"produce{j}_bin", "box", (x, y, 0.53), (0.55, 0.45, 0.05), rgba="0.30 0.55 0.38 1")
    s.comment("cooler wall (north) + bakery endcap")
    P.fridge_wall(s, "cooler", -1.0, 5.05, yaw=0.0, L=9.0)
    s.geom("bakery_case", "box", (7.0, -1.5, 0.6), (0.5, 1.2, 0.6), mat="m_wood")
    s.geom("bakery_glass", "box", (6.62, -1.5, 0.95), (0.02, 1.1, 0.25), mat="m_glass", collide=False)
    s.comment("checkout lanes (south-west)")
    P.checkout(s, "co1", -4.5, -4.2, yaw=0.0)
    P.checkout(s, "co2", -1.5, -4.2, yaw=0.0)
    s.geom("basket_stack", "box", (0.6, -4.6, 0.25), (0.25, 0.25, 0.25), rgba="0.75 0.28 0.22 1")
    for i, x in enumerate((-4.0, 0.0, 4.0)):
        P.ceiling_lamp(s, f"bay{i + 1}", x, 0.0, z=3.15, r=0.26)
    s.light("key", (3.0, -2.0, 3.6), (-0.25, 0.2, -1), "0.52 0.52 0.52", castshadow=True)
    s.light("fill", (-4.0, 3.0, 3.6), (0.35, -0.3, -1), "0.26 0.26 0.28")
    s.comment("graspable groceries")
    for j, (x, y, c) in enumerate(((5.0, 2.5, "0.80 0.15 0.12 1"), (5.35, 2.75, "0.80 0.15 0.12 1"),
                                   (5.1, 0.9, "0.95 0.60 0.10 1"), (5.4, 0.65, "0.95 0.60 0.10 1"),
                                   (6.5, 1.6, "0.55 0.75 0.25 1"))):
        P.free_sphere(s, f"fruit{j + 1}", x, y, 0.63, 0.042, c, mass=0.15)
    P.free_cylinder(s, "can1", -2.9, -0.87, 0.63, 0.035, 0.055, "0.75 0.22 0.18 1", mass=0.4)
    P.free_cylinder(s, "can2", -2.7, -0.87, 0.63, 0.035, 0.055, "0.24 0.42 0.65 1", mass=0.4)
    P.free_box(s, "cereal_sm", -1.4, -0.86, 0.66, (0.09, 0.03, 0.14), "0.85 0.65 0.20 1", mass=0.35)
    s.comment("shoppers strolling the aisles")
    A.walker(s, "shopper1", [[-6.0, -2.5], [3.0, -2.5], [3.0, 0.0], [-6.0, 0.0]],
             speed=0.9, s0=2.0, palette=1,
             stops=[{"s": 5.0, "period": 26.0, "dur": 6.0, "offset": 4.0}],
             comment="shopper: middle aisle loop, pauses at the shelves")
    A.walker(s, "shopper2", [[-6.2, 2.5], [4.0, 2.5], [4.6, 3.9], [-5.0, 3.9]],
             speed=0.8, s0=9.0, palette=4, scale=0.95,
             stops=[{"s": 14.0, "period": 31.0, "dur": 7.0, "offset": 11.0}],
             comment="shopper: north aisle + cooler wall browse")
    return s


def _cafe_base(name, title, desc, floor_mat, wall_mat, awn):
    s = Scene(name, title, "Food & Dining", desc,
              headlight=("0.46 0.45 0.44", "0.27 0.27 0.27"))
    P.room_shell(s, 9, 7, 3.0, floor_mat, wall_mat)
    s.comment("service counter + espresso bar (north)")
    P.counter(s, "bar", -1.0, 2.6, yaw=0.0, L=3.6, top="m_marble", body="m_wood")
    s.geom("espresso", "box", (-2.0, 2.75, 1.12), (0.28, 0.20, 0.18), mat="m_steel")
    s.geom("grinder", "cylinder", (-1.3, 2.8, 1.10), (0.07, 0.16), mat="m_dark")
    s.geom("pastry_case", "box", (0.4, 2.7, 1.13), (0.55, 0.25, 0.21), mat="m_glass")
    s.geom("pastry_shelf", "box", (0.4, 2.7, 1.03), (0.5, 0.22, 0.02), mat="m_wood", collide=False)
    s.geom("menu", "box", (-1.0, 3.42, 2.35), (1.5, 0.02, 0.45), mat="m_menu", collide=False)
    P.awning(s, "awn", -1.0, 2.2, yaw=0.0, L=3.9, mat=awn, z=2.75)
    return s


def cafe():
    s = _cafe_base("cafe", "Corner Cafe",
                   "Cozy corner cafe: espresso bar with pastry case under a red "
                   "awning, round tables with seated customers, window bench; "
                   "a mug to bus and a pastry to deliver.",
                   "m_wood_floor", "m_brick", "m_awning_red")
    s.comment("round tables + chairs")
    for i, (x, y) in enumerate(((-2.6, -0.2), (-2.6, -2.2), (0.0, -2.3), (2.4, -1.4), (2.5, 0.9))):
        P.round_table(s, f"tbl{i + 1}", x, y, r=0.36)
        for k in range(2):
            ang = k * math.pi + (0.5 if i % 2 else 0.0)
            cx, cy = x + 0.72 * math.cos(ang), y + 0.72 * math.sin(ang)
            P.chair(s, f"tbl{i + 1}_ch{k + 1}", cx, cy, yaw=ang + math.pi)
    s.comment("window bench (south) + plants")
    P.bench(s, "winbench", 0.6, -3.15, yaw=0.0, L=4.2, back=False)
    P.plant(s, "plant1", 4.1, 2.8)
    P.plant(s, "plant2", -4.1, -3.0)
    P.window_band(s, "front_glass", 0.0, -3.44, 0.0, 7.5, z0=0.8, z1=2.6)
    P.bookshelf(s, "zines", -4.3, 1.0, yaw=0.0, w=0.9, h=1.5)
    for i, (x, y) in enumerate(((-2.6, -1.2), (0.6, -0.8), (2.4, -0.2))):
        P.ceiling_lamp(s, f"pend{i + 1}", x, y, z=2.5, r=0.14)
    s.light("key", (1.5, -2.0, 3.0), (-0.25, 0.3, -1), "0.5 0.47 0.42", castshadow=True)
    s.light("fill", (-2.0, 2.0, 3.0), (0.3, -0.3, -1), "0.24 0.23 0.22")
    s.comment("customers + graspables")
    A.person_seated(s, "cust1", -3.32, -0.2, yaw=0.0, palette=2)
    A.person_seated(s, "cust2", 3.12, 0.9, yaw=math.pi, palette=5)
    A.person_standing(s, "cust3", 1.6, 1.9, yaw=2.2, palette=0)
    P.free_cylinder(s, "mug_c", -2.55, -0.1, 0.79, 0.045, 0.05, "0.86 0.84 0.80 1", mass=0.3)
    P.free_box(s, "pastry", 0.45, 2.68, 1.07, (0.05, 0.035, 0.025), "0.80 0.58 0.28 1", mass=0.12)
    return s


def cafe_industrial():
    s = _cafe_base("cafe_industrial", "Roastery Cafe",
                   "Industrial roastery cafe: concrete floor, long communal "
                   "tables with stools, exposed ducting, green awning and a "
                   "roaster drum in the corner.",
                   "m_concrete", "m_wall", "m_awning_green")
    s.comment("communal tables + stools (kept clear of the spawn circle)")
    for i, y in enumerate((-1.3, -2.7)):
        P.rect_table(s, f"comm{i + 1}", 0.8, y, yaw=0.0, w=3.6, d=0.8, h=0.76, top="m_wood")
        for k in range(4):
            x = -0.6 + k * 0.95
            if i == 0:
                P.stool(s, f"comm{i + 1}_st{k + 1}", x, y - 0.62, h=0.52)
            P.stool(s, f"comm{i + 1}_stb{k + 1}", x, y + 0.62, h=0.52)
    s.comment("roaster + sacks (west)")
    s.geom("roaster_drum", "cylinder", (-3.7, -0.5, 0.9), (0.45, 0.55), mat="m_metal")
    s.geom("roaster_hopper", "box", (-3.7, -0.5, 1.75), (0.3, 0.3, 0.25), mat="m_dark")
    s.geom("sack1", "box", (-3.9, -1.8, 0.3), (0.25, 0.2, 0.3), mat="m_cardboard", euler=(0, 0, 0.3))
    s.geom("sack2", "box", (-3.5, -2.3, 0.3), (0.25, 0.2, 0.3), mat="m_cardboard", euler=(0, 0, -0.2))
    s.comment("exposed ducting")
    s.geom("duct_main", "cylinder", (0, 0.5, 2.88), (0.18, 4.2), euler=(0, 1.570795, 0), mat="m_steel", collide=False)
    s.geom("duct_drop", "cylinder", (-3.7, -0.4, 2.3), (0.14, 0.55), mat="m_steel", collide=False)
    P.window_band(s, "front_glass", 0.0, -3.44, 0.0, 7.5, z0=0.8, z1=2.6)
    for i, x in enumerate((-1.4, 0.4, 2.2)):
        P.ceiling_lamp(s, f"pend{i + 1}", x, -1.5, z=2.4, r=0.12)
    s.light("key", (1.5, -2.0, 3.0), (-0.25, 0.3, -1), "0.48 0.47 0.45", castshadow=True)
    s.light("fill", (-2.0, 2.0, 3.0), (0.3, -0.3, -1), "0.24 0.24 0.25")
    A.person_seated(s, "cust1", 0.35, -0.68, yaw=-1.570795, palette=3, seat_h=0.54)
    A.person_seated(s, "cust2", 1.3, -2.08, yaw=-1.570795, palette=6, seat_h=0.54)
    P.free_cylinder(s, "cup1", 0.35, -1.25, 0.84, 0.04, 0.045, "0.92 0.92 0.94 1", mass=0.25)
    P.free_box(s, "beans", -3.65, -1.75, 0.66, (0.07, 0.05, 0.06), "0.45 0.30 0.18 1", mass=0.5)
    return s


def restaurant():
    s = Scene("restaurant", "Restaurant", "Food & Dining",
              "Casual bistro: wood four-tops with seated diners, wall "
              "banquette, service bar, pendant lights — and a waiter doing "
              "rounds between the tables.",
              headlight=("0.44 0.43 0.42", "0.26 0.26 0.26"))
    P.room_shell(s, 12, 8, 3.0, "m_wood_floor", "m_wall", wainscot="m_wood")
    s.comment("dining tables (robot spawn kept clear at the origin)")
    tables = [(-3.2, 1.6), (-3.2, -1.8), (0.0, 2.2), (0.0, -2.2), (3.2, 1.6), (3.2, -1.8)]
    for i, (x, y) in enumerate(tables):
        P.rect_table(s, f"tbl{i + 1}", x, y, yaw=0.0, w=0.9, d=0.9, top="m_wood")
        for k, (dx, dy, yy) in enumerate(((0.75, 0, math.pi), (-0.75, 0, 0.0))):
            P.chair(s, f"tbl{i + 1}_ch{k + 1}", x + dx, y + dy, yaw=yy)
    s.comment("banquette along the north wall")
    P.bench(s, "banq", 0.0, 3.6, yaw=0.0, L=9.0, seat="m_fabric")
    s.comment("service bar (east)")
    P.counter(s, "svc", 5.4, 0.0, yaw=1.570795, L=3.0, top="m_marble", body="m_wood")
    s.geom("backbar", "box", (5.93, 0.0, 1.5), (0.02, 1.6, 0.5), mat="m_book", collide=False)
    s.geom("glassrack", "box", (5.5, 0.0, 2.2), (0.25, 1.4, 0.04), mat="m_wood", collide=False)
    P.plant(s, "plant1", -5.5, 3.4)
    P.plant(s, "plant2", -5.5, -3.4)
    for i, (x, y) in enumerate([(-3.2, 0.0), (0.0, 0.0), (3.2, 0.0)]):
        P.ceiling_lamp(s, f"pend{i + 1}", x, y + 2.0, z=2.3, r=0.15)
        P.ceiling_lamp(s, f"pend{i + 4}", x, y - 2.0, z=2.3, r=0.15)
    s.light("key", (2.0, -1.5, 3.0), (-0.3, 0.25, -1), "0.44 0.42 0.38", castshadow=True)
    s.light("fill", (-2.5, 2.0, 3.0), (0.35, -0.3, -1), "0.22 0.21 0.20")
    s.comment("diners")
    A.person_seated(s, "diner1", -2.45, 1.6, yaw=math.pi, palette=0)
    A.person_seated(s, "diner2", -3.95, 1.6, yaw=0.0, palette=3)
    A.person_seated(s, "diner3", 0.75, -2.2, yaw=math.pi, palette=5)
    A.person_seated(s, "diner4", 3.95, -1.8, yaw=0.0, palette=7)
    s.comment("table settings (graspable)")
    P.free_cylinder(s, "plate1", -3.2, 1.35, 0.78, 0.11, 0.012, "0.92 0.92 0.94 1", mass=0.4)
    P.free_cylinder(s, "cup_r1", -3.0, 1.85, 0.80, 0.038, 0.05, "0.86 0.84 0.80 1", mass=0.25)
    P.free_cylinder(s, "plate2", 3.2, -1.55, 0.78, 0.11, 0.012, "0.92 0.92 0.94 1", mass=0.4)
    s.comment("waiter loop: bar -> tables -> banquette -> bar")
    A.walker(s, "waiter", [[4.6, -0.5], [1.6, -1.25], [-1.7, -1.25], [-1.7, 1.25],
                           [1.6, 1.25], [4.6, 0.5]],
             speed=0.95, s0=0.0, palette=7,
             stops=[{"s": 6.5, "period": 22.0, "dur": 4.0, "offset": 6.0},
                    {"s": 14.0, "period": 22.0, "dur": 4.0, "offset": 14.0}],
             comment="waiter doing rounds, pausing tableside")
    return s


def fastfood():
    s = Scene("fastfood", "Fast Food", "Food & Dining",
              "Burger joint: checkerboard floor, glowing menu boards over the "
              "order counter, soda fountain, red booths — trays and a burger "
              "box to clear.",
              headlight=("0.54 0.53 0.52", "0.32 0.32 0.32"))
    P.room_shell(s, 11, 8, 3.0, "m_tile_checker", "m_wall")
    s.comment("order counter + kitchen pass (north)")
    P.counter(s, "order", 0.0, 2.7, yaw=0.0, L=4.5, top="m_white", body="m_dark")
    for i in range(3):
        s.geom(f"till{i + 1}", "box", (-1.4 + i * 1.4, 2.75, 1.06), (0.14, 0.10, 0.12), mat="m_dark")
    s.geom("menu_l", "box", (-1.5, 3.42, 2.45), (1.35, 0.02, 0.4), mat="m_menu", collide=False)
    s.geom("menu_r", "box", (1.5, 3.42, 2.45), (1.35, 0.02, 0.4), mat="m_menu", collide=False)
    s.geom("heatlamp", "box", (0.0, 3.2, 1.9), (1.2, 0.15, 0.04), mat="m_sig_amber", collide=False)
    s.geom("soda", "box", (3.6, 2.9, 1.1), (0.5, 0.35, 0.5), mat="m_steel")
    s.geom("soda_face", "box", (3.6, 2.53, 1.2), (0.42, 0.02, 0.3), mat="m_screen", collide=False)
    s.comment("booths along the south window")
    for i, x in enumerate((-3.6, -1.2, 1.2, 3.6)):
        P.booth(s, f"booth{i + 1}", x, -2.9, yaw=1.570795)
    P.window_band(s, "front_glass", 0.0, -3.94, 0.0, 10.0, z0=0.9, z1=2.5)
    s.comment("center tables")
    for i, (x, y) in enumerate(((-2.4, 0.2), (2.4, 0.2))):
        P.round_table(s, f"ctbl{i + 1}", x, y, r=0.35, h=0.72, top="m_white", base="m_chrome")
        for k in range(2):
            ang = k * math.pi + 0.7
            P.stool(s, f"ctbl{i + 1}_st{k + 1}", x + 0.65 * math.cos(ang), y + 0.65 * math.sin(ang), h=0.50)
    s.geom("trashst", "box", (5.0, -0.5, 0.55), (0.3, 0.3, 0.55), rgba="0.55 0.20 0.16 1")
    s.geom("trash_flap", "box", (4.68, -0.5, 0.85), (0.02, 0.2, 0.12), mat="m_dark", collide=False)
    for i, x in enumerate((-3.0, 0.0, 3.0)):
        P.ceiling_lamp(s, f"strip{i + 1}", x, 0.5, z=2.8, r=0.2)
    s.light("key", (2.0, -1.0, 3.2), (-0.3, 0.2, -1), "0.55 0.53 0.50", castshadow=True)
    s.light("fill", (-3.0, 2.0, 3.2), (0.4, -0.3, -1), "0.28 0.28 0.28")
    s.comment("customers + trays")
    A.person_seated(s, "cust1", -3.6, -2.15, yaw=-1.570795, palette=1)
    A.person_seated(s, "cust2", 1.2, -3.65, yaw=1.570795, palette=4)
    A.person_standing(s, "cust3", -0.7, 1.6, yaw=1.4, palette=6)
    P.free_box(s, "tray1", -3.6, -2.9, 0.79, (0.17, 0.12, 0.012), "0.60 0.25 0.20 1", mass=0.3)
    P.free_box(s, "burgerbox", -3.52, -2.88, 0.84, (0.055, 0.055, 0.03), "0.85 0.65 0.20 1", mass=0.15)
    P.free_cylinder(s, "cola", -3.75, -2.95, 0.86, 0.035, 0.06, "0.75 0.16 0.14 1", mass=0.3)
    return s


def fine_dining():
    s = Scene("fine_dining", "Fine Dining", "Food & Dining",
              "White-tablecloth dining room: clothed rounds with candles and "
              "place settings, high-back chairs, drapes, a grand piano — and "
              "a waiter circulating silently.",
              headlight=("0.38 0.37 0.36", "0.22 0.22 0.22"))
    P.room_shell(s, 12, 9, 3.4, "m_wood_floor", "m_wall", wainscot="m_wood")
    s.comment("clothed tables (spawn kept clear)")
    tables = [(-3.4, 2.0), (-3.4, -2.0), (0.0, 2.6), (0.0, -2.6), (3.4, 2.0), (3.4, -2.0)]
    for i, (x, y) in enumerate(tables):
        P.round_table(s, f"tbl{i + 1}", x, y, r=0.55, h=0.76, top="m_tablecloth",
                      base="m_dark", cloth="m_tablecloth")
        s.geom(f"tbl{i + 1}_candle", "cylinder", (x, y, 0.83), (0.012, 0.05), mat="m_cream", collide=False)
        s.geom(f"tbl{i + 1}_flame", "sphere", (x, y, 0.90), (0.012,), mat="m_lamp", collide=False)
        for k in range(3):
            ang = k * 2.0944 + (0.4 if i % 2 else 1.0)
            cx, cy = x + 0.95 * math.cos(ang), y + 0.95 * math.sin(ang)
            P.chair(s, f"tbl{i + 1}_ch{k + 1}", cx, cy, yaw=ang + math.pi, seat="m_fabric")
    s.comment("grand piano (north-east)")
    s.geom("piano_body", "box", (4.6, 3.2, 0.85), (0.75, 0.65, 0.12), mat="m_dark", euler=(0, 0, -0.5))
    s.geom("piano_lid", "box", (4.75, 3.35, 1.12), (0.7, 0.6, 0.015), mat="m_dark", euler=(-0.35, 0, -0.5))
    for i, (dx, dy) in enumerate(((-0.5, -0.45), (0.55, -0.4), (0.1, 0.55))):
        s.geom(f"piano_leg{i + 1}", "cylinder", (4.6 + dx, 3.2 + dy, 0.36), (0.045, 0.36), mat="m_dark")
    s.geom("piano_keys", "box", (4.05, 2.55, 0.78), (0.4, 0.08, 0.02), mat="m_white", euler=(0, 0, -0.5), collide=False)
    s.comment("drapes + sideboard + wine station")
    for i, x in enumerate((-4.0, 0.0, 4.0)):
        s.geom(f"drape{i + 1}", "box", (x, 4.42, 1.85), (0.7, 0.03, 1.5), mat="m_fabric", collide=False)
    s.geom("sideboard", "box", (-5.6, 0.0, 0.5), (0.3, 1.4, 0.5), mat="m_wood")
    s.geom("wine_rack", "box", (-5.6, 0.0, 1.55), (0.22, 1.1, 0.45), mat="m_book", collide=False)
    for i, (x, y) in enumerate([(-1.7, 0.0), (1.7, 0.0)]):
        P.ceiling_lamp(s, f"chand{i + 1}", x, y, z=2.9, r=0.34)
    s.light("key", (1.5, -1.5, 3.2), (-0.25, 0.25, -1), "0.36 0.33 0.28", castshadow=True)
    s.light("warm", (-2.0, 2.0, 3.0), (0.3, -0.3, -1), "0.20 0.17 0.14")
    s.comment("diners + settings")
    A.person_seated(s, "diner1", -2.45, 2.0, yaw=math.pi, palette=7)
    A.person_seated(s, "diner2", -4.35, 2.0, yaw=0.0, palette=6)
    A.person_seated(s, "diner3", 0.95, -2.6, yaw=math.pi, palette=3)
    A.person_seated(s, "diner4", 3.4, -1.05, yaw=-1.570795, palette=0)
    P.free_cylinder(s, "fd_plate", -3.4, 1.72, 0.80, 0.12, 0.012, "0.92 0.92 0.94 1", mass=0.4)
    P.free_cylinder(s, "fd_glass", -3.15, 2.2, 0.83, 0.03, 0.07, "0.62 0.74 0.80 0.6", mass=0.15)
    s.comment("waiter circulating between the rounds")
    A.walker(s, "waiter", [[-5.0, 0.0], [-1.7, 1.15], [1.7, 1.15], [4.6, 0.0],
                           [1.7, -1.15], [-1.7, -1.15]],
             speed=0.8, s0=3.0, palette=7,
             stops=[{"s": 3.4, "period": 26.0, "dur": 5.0, "offset": 8.0},
                    {"s": 13.0, "period": 26.0, "dur": 5.0, "offset": 21.0}],
             comment="waiter: slow silent rounds with tableside pauses")
    return s


def build_all():
    return [supermarket(), cafe(), cafe_industrial(), restaurant(), fastfood(),
            fine_dining()]
