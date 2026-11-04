"""scenes_home.py -- Home & Living: washroom, two more living rooms, the
one-floor full house (BIG), and the gym (Work & Public)."""
import math

from .core import Scene
from . import prefabs as P
from . import actors as A


def washroom():
    s = Scene("washroom", "Washroom", "Home & Living",
              "Family bathroom in small white tile: double vanity with mirror, "
              "bathtub, glass shower stall and toilet; towels and a laundry "
              "basket within reach.",
              headlight=("0.52 0.52 0.54", "0.30 0.30 0.30"))
    s.comment("room shell 6 x 5 m")
    P.room_shell(s, 6, 5, 2.5, "m_tile_small", "m_tile_small")
    s.comment("fixtures")
    P.vanity(s, "van", -2.55, 0.6, yaw=0.0, L=1.6)            # mirror against W wall
    P.toilet(s, "wc", -2.5, -1.7, yaw=0.0)
    P.bathtub(s, "tub", 1.4, 2.0, yaw=0.0)
    P.shower(s, "shw", 2.35, -1.8, yaw=1.570795)
    s.comment("towels, mat, laundry")
    s.geom("rail", "cylinder", (0.0, 2.44, 1.15), (0.015, 0.5), euler=(0, 1.570795, 0), mat="m_chrome", collide=False)
    s.geom("towel1", "box", (-0.2, 2.42, 0.92), (0.16, 0.02, 0.24), rgba="0.35 0.55 0.75 1", collide=False)
    s.geom("towel2", "box", (0.22, 2.42, 0.92), (0.16, 0.02, 0.24), rgba="0.85 0.83 0.78 1", collide=False)
    P.rug(s, "bathmat", 0.4, 1.1, 0.9, 0.5, mat="m_rug")
    s.geom("basket", "cylinder", (2.5, 0.4, 0.28), (0.24, 0.28), mat="m_wood")
    s.geom("window", "box", (0.0, -2.53, 1.7), (0.8, 0.02, 0.5), mat="m_storefront", collide=False)
    P.ceiling_lamp(s, "lamp1", 0, 0, z=2.42)
    s.light("key", (1.5, 1.0, 2.4), (-0.3, -0.2, -1), "0.5 0.5 0.52", castshadow=True)
    s.light("fill", (-2.0, -1.5, 2.3), (0.4, 0.3, -1), "0.22 0.22 0.24")
    # graspables: soap bottle, rubber duck on the tub rim
    P.free_cylinder(s, "soap", -2.55, 1.25, 0.94, 0.035, 0.09, "0.90 0.60 0.20 1", mass=0.2)
    P.free_sphere(s, "duck", 1.0, 1.75, 0.66, 0.05, "0.98 0.85 0.15 1", mass=0.05)
    return s


def living_room_loft():
    s = Scene("living_room_loft", "Loft Living Room", "Home & Living",
              "Industrial loft: brick walls, wood floor, L-sofa around a slab "
              "coffee table, big TV, kitchenette bar with stools and a wall of "
              "glass; speaker and mug to fetch.",
              headlight=("0.44 0.44 0.46", "0.26 0.26 0.26"))
    s.comment("shell 10 x 7 m, brick loft")
    P.room_shell(s, 10, 7, 3.2, "m_wood_floor", "m_brick")
    P.window_band(s, "glazing", 0, 3.44, 0, 8.5, z0=0.4, z1=2.9)
    s.comment("L-sofa + slab table + TV")
    P.sofa(s, "sofaA", -1.2, 1.6, yaw=-math.pi / 2, L=2.6, mat="m_fabric")
    P.sofa(s, "sofaB", -2.6, 0.2, yaw=0.0, L=1.8, mat="m_fabric")
    s.geom("slab_top", "box", (-1.6, 0.2, 0.35), (0.65, 0.40, 0.04), mat="m_concrete")
    s.geom("slab_leg", "box", (-1.6, 0.2, 0.16), (0.5, 0.28, 0.16), mat="m_dark")
    P.rug(s, "rug1", -1.2, 0.4, 3.4, 2.6)
    P.tv_wall(s, "tv", -4.75, 0.4, yaw=0.0, w=1.9)
    s.comment("kitchenette bar along the east wall")
    P.counter(s, "bar", 3.6, -1.2, yaw=1.570795, L=3.0, top="m_marble", body="m_dark")
    for i in range(3):
        P.stool(s, f"barstool{i + 1}", 2.9, -2.1 + i * 0.9)
    s.geom("shelf_hi", "box", (4.42, -1.2, 1.9), (0.14, 1.4, 0.02), mat="m_wood")
    s.geom("bottles", "box", (4.42, -1.2, 2.02), (0.10, 1.2, 0.10), mat="m_book", collide=False)
    s.comment("greenery + lighting")
    P.plant(s, "plant1", 4.4, 2.9)
    P.plant(s, "plant2", -4.4, 2.9)
    P.bookshelf(s, "shelf", -4.6, -2.4, yaw=0.0, w=1.4)
    for i, x in enumerate((-2.5, 0.0, 2.5)):
        P.ceiling_lamp(s, f"pend{i + 1}", x, 0.8, z=2.6, r=0.20)
    s.light("key", (2.0, 2.0, 3.0), (-0.3, -0.35, -1), "0.5 0.5 0.48", castshadow=True)
    s.light("fill", (-3.0, -2.0, 3.0), (0.4, 0.3, -1), "0.24 0.24 0.26")
    P.free_cylinder(s, "mug", -1.75, 0.05, 0.45, 0.045, 0.05, "0.20 0.35 0.68 1", mass=0.3)
    P.free_box(s, "speaker", 3.5, -1.15, 0.99, (0.06, 0.06, 0.09), "0.16 0.17 0.19 1", mass=0.6)
    return s


def living_room_family():
    s = Scene("living_room_family", "Family Living Room", "Home & Living",
              "Warm carpeted family room: two facing sofas, toy-strewn play "
              "rug, TV and bookshelf; foam blocks and a ball for pick-up "
              "practice.",
              headlight=("0.50 0.48 0.46", "0.28 0.28 0.28"))
    s.comment("shell 9 x 7 m, warm carpet")
    P.room_shell(s, 9, 7, 2.6, "m_carpet_beige", "m_wall")
    s.comment("facing sofas + coffee table")
    P.sofa(s, "sofaN", 1.2, 1.9, yaw=-math.pi / 2, L=2.2)
    P.sofa(s, "sofaS", 1.2, -1.9, yaw=math.pi / 2, L=2.2, mat="m_rug")
    P.round_table(s, "coffee", 1.2, 0.0, r=0.45, h=0.42)
    P.tv_wall(s, "tv", 4.25, 0.0, yaw=math.pi, w=1.6)
    P.bookshelf(s, "shelf1", -4.3, 1.8, yaw=0.0, w=1.2)
    P.bookshelf(s, "shelf2", -4.3, -1.8, yaw=0.0, w=1.2)
    s.comment("play corner")
    P.rug(s, "playrug", -2.2, 0.0, 2.4, 2.0, mat="m_rug")
    s.geom("toybin", "box", (-3.2, 1.0, 0.18), (0.3, 0.22, 0.18), rgba="0.90 0.55 0.15 1")
    P.plant(s, "plant1", 4.2, 2.9)
    s.geom("lamp_post", "cylinder", (4.2, -2.9, 0.75), (0.03, 0.75), mat="m_dark")
    s.geom("lamp_shade", "cylinder", (4.2, -2.9, 1.62), (0.18, 0.12), mat="m_lamp", collide=False)
    P.ceiling_lamp(s, "ceil", 1.2, 0, z=2.52)
    s.light("key", (2.0, 1.5, 2.4), (-0.3, -0.25, -1), "0.5 0.48 0.45", castshadow=True)
    s.light("fill", (-2.5, -1.5, 2.4), (0.4, 0.25, -1), "0.24 0.23 0.22")
    s.comment("graspable toys")
    for i, (x, y, c) in enumerate(((-2.0, 0.3, "0.75 0.28 0.22 1"), (-2.45, -0.25, "0.24 0.42 0.65 1"),
                                   (-1.7, -0.45, "0.30 0.55 0.38 1"))):
        P.free_box(s, f"block{i + 1}", x, y, 0.05, (0.045, 0.045, 0.045), c, mass=0.12)
    P.free_sphere(s, "ball", -2.7, 0.7, 0.09, 0.09, "0.85 0.30 0.20 1", mass=0.2)
    return s


def full_house():
    s = Scene("full_house", "Full House (One Floor)", "Home & Living",
              "A whole 17 x 10 m single-floor home: living room, open kitchen "
              "with island, dining, bedroom, washroom and hallway — one "
              "continuous navigable space for long-horizon mobile tasks.",
              headlight=("0.48 0.48 0.50", "0.28 0.28 0.28"))
    s.comment("shell 17 x 10 m — robot spawns in the living area")
    P.room_shell(s, 17, 10, 2.7, "m_wood_floor", "m_wall")
    s.comment("interior partitions (living | kitchen/dining | bedroom+bath wing)")
    # bedroom wing: east 5 m, split bedroom (N) / washroom (S) by a spine wall;
    # two door gaps in the wing wall (washroom at y=-3.6..-2.4, bedroom at 1.4..5)
    P.partition(s, "part_bedwingS", 3.5, -5.0, 3.5, -3.6, h=2.7)
    P.partition(s, "part_bedwingN", 3.5, -2.4, 3.5, 1.4, h=2.7)
    P.partition(s, "part_bedbath", 3.5, -0.9, 8.5, -0.9, h=2.7)      # bedroom / washroom
    # kitchen split from living by a half wall
    P.partition(s, "part_kitchen", -8.5, 0.8, -4.6, 0.8, h=1.1, mat="m_wall")
    s.comment("living area (robot spawn)")
    P.sofa(s, "sofa", -1.5, -2.4, yaw=math.pi / 2, L=2.4)
    P.round_table(s, "coffee", -1.5, -0.9, r=0.42, h=0.40)
    P.tv_wall(s, "tv", -1.5, 4.62, yaw=-math.pi / 2, w=1.7)
    P.rug(s, "rug_liv", -1.5, -1.2, 3.2, 2.4)
    P.bookshelf(s, "shelf_liv", 2.9, 3.9, yaw=math.pi / 2, w=1.2)
    P.plant(s, "plant_liv", 0.6, 4.3)
    s.comment("kitchen north-west + island + dining")
    P.counter(s, "kcount1", -8.05, 3.0, yaw=1.570795, L=3.4, top="m_marble", body="m_wood")
    P.counter(s, "kcount2", -6.8, 4.55, yaw=0.0, L=2.6, top="m_marble", body="m_wood")
    s.geom("fridge", "box", (-4.9, 4.4, 0.95), (0.42, 0.42, 0.95), mat="m_steel")
    P.counter(s, "island", -6.2, 2.2, yaw=0.0, L=2.0, top="m_marble", body="m_wood")
    for i in range(2):
        P.stool(s, f"kstool{i + 1}", -6.6 + i * 0.9, 1.45)
    P.rect_table(s, "dining", -6.0, -2.6, yaw=0.0, w=1.8, d=1.0, top="m_wood")
    for i, (dx, dy, yy) in enumerate(((-0.6, 0.85, -math.pi / 2), (0.6, 0.85, -math.pi / 2),
                                      (-0.6, -0.85, math.pi / 2), (0.6, -0.85, math.pi / 2))):
        P.chair(s, f"dchair{i + 1}", -6.0 + dx, -2.6 + dy, yaw=yy)
    P.ceiling_lamp(s, "pend_dining", -6.0, -2.6, z=2.2, r=0.22)
    s.comment("bedroom (NE)")
    P.bed(s, "bed", 6.6, 2.6, yaw=math.pi / 2, w=1.6, L=2.0)
    s.geom("nightstand1", "box", (5.5, 4.5, 0.28), (0.24, 0.22, 0.28), mat="m_wood")
    s.geom("nightstand2", "box", (7.7, 4.5, 0.28), (0.24, 0.22, 0.28), mat="m_wood")
    P.wardrobe(s, "wardrobe", 8.15, 0.4, yaw=math.pi, w=1.4)
    P.rug(s, "rug_bed", 6.2, 1.2, 2.2, 1.6, mat="m_carpet_beige")
    s.comment("washroom (SE)")
    P.vanity(s, "van", 8.1, -2.0, yaw=math.pi, L=1.2)
    P.toilet(s, "wc", 8.0, -4.3, yaw=math.pi)
    P.bathtub(s, "tub", 5.2, -4.3, yaw=0.0)
    s.comment("hallway props")
    P.plant(s, "plant_hall", 2.6, -4.4)
    s.geom("console_hall", "box", (0.2, -4.6, 0.4), (0.7, 0.19, 0.4), mat="m_wood")
    for i, (x, y) in enumerate(((-1.5, -1.2), (-6.2, 2.2), (1.0, -4.0), (6.6, 2.0), (6.8, -2.6))):
        P.ceiling_lamp(s, f"halo{i + 1}", x, y, z=2.6)
    s.light("key", (3.0, 2.0, 3.2), (-0.3, -0.25, -1), "0.46 0.46 0.44", castshadow=True)
    s.light("fill1", (-5.0, -2.0, 3.0), (0.35, 0.2, -1), "0.26 0.26 0.28")
    s.light("fill2", (6.0, -3.0, 3.0), (-0.2, 0.3, -1), "0.20 0.20 0.22")
    s.comment("graspables around the house")
    P.free_cylinder(s, "mug", -6.2, 2.1, 0.99, 0.045, 0.05, "0.75 0.22 0.18 1", mass=0.3)
    P.free_box(s, "cereal", -6.9, 4.5, 1.12, (0.09, 0.03, 0.14), "0.85 0.65 0.20 1", mass=0.35)
    P.free_sphere(s, "apple_fh", -6.55, 2.28, 0.99, 0.04, "0.80 0.15 0.12 1", mass=0.15)
    P.free_box(s, "book", -1.55, -0.85, 0.47, (0.11, 0.075, 0.02), "0.30 0.40 0.60 1", mass=0.4)
    P.free_cylinder(s, "alarm", 5.5, 4.45, 0.61, 0.05, 0.035, "0.85 0.83 0.80 1", mass=0.2)
    return s


def gym():
    s = Scene("gym", "Gym", "Work & Public",
              "Strength & cardio floor on speckled rubber: squat rack, benches, "
              "dumbbell rack, treadmills, plyo boxes and a mirror wall; "
              "kettlebell and dumbbells are real free bodies.",
              headlight=("0.46 0.46 0.48", "0.28 0.28 0.28"))
    s.comment("hall 12 x 9 m")
    P.room_shell(s, 12, 9, 3.2, "m_rubber", "m_concrete")
    s.geom("mirror_wall", "box", (0, 4.42, 1.5), (5.4, 0.015, 1.3), mat="m_mirror", collide=False)
    s.comment("squat rack + bench zone (west)")
    for i, dx in enumerate((-0.65, 0.65)):
        s.geom(f"rack_up{i + 1}", "box", (-4.4 + dx, 2.6, 1.1), (0.05, 0.05, 1.1), mat="m_frame")
        s.geom(f"rack_arm{i + 1}", "box", (-4.4 + dx, 2.35, 1.45), (0.04, 0.18, 0.03), mat="m_dark")
    s.geom("rack_bar", "cylinder", (-4.4, 2.35, 1.52), (0.025, 1.1), euler=(0, 1.570795, 0), mat="m_chrome")
    for i, dx in enumerate((-0.95, 0.95)):
        s.geom(f"rack_plate{i + 1}", "cylinder", (-4.4 + dx, 2.35, 1.52), (0.20, 0.02),
               euler=(0, 1.570795, 0), mat="m_dark")
    s.geom("bench_pad", "box", (-4.2, 0.6, 0.44), (0.55, 0.16, 0.05), mat="m_fabric")
    s.geom("bench_leg1", "box", (-4.6, 0.6, 0.2), (0.05, 0.12, 0.2), mat="m_dark")
    s.geom("bench_leg2", "box", (-3.8, 0.6, 0.2), (0.05, 0.12, 0.2), mat="m_dark")
    s.comment("dumbbell rack along the mirror")
    P.shelf_unit(s, "dbrack", -1.2, 4.0, yaw=0.0, L=2.2, h=0.9, mat="m_metal")
    for i in range(5):
        x = -2.1 + i * 0.45
        s.geom(f"db_fixed{i}", "cylinder", (x, 4.0, 0.72), (0.055, 0.10),
               euler=(1.570795, 0, 0), mat="m_dark", collide=False)
    s.comment("cardio row (east): two treadmills")
    for i, y in enumerate((2.2, 0.6)):
        s.geom(f"tm{i + 1}_deck", "box", (4.3, y, 0.12), (0.85, 0.35, 0.06), mat="m_dark")
        s.geom(f"tm{i + 1}_belt", "box", (4.35, y, 0.185), (0.65, 0.26, 0.008), mat="m_belt")
        s.geom(f"tm{i + 1}_mast", "box", (3.55, y, 0.75), (0.06, 0.30, 0.6), mat="m_steel")
        s.geom(f"tm{i + 1}_console", "box", (3.5, y, 1.42), (0.04, 0.28, 0.10), mat="m_screen", collide=False)
    s.comment("functional zone: plyo boxes, mats, ball")
    s.geom("plyo1", "box", (2.8, -3.0, 0.25), (0.35, 0.30, 0.25), mat="m_wood")
    s.geom("plyo2", "box", (3.8, -3.1, 0.35), (0.35, 0.30, 0.35), mat="m_wood")
    P.rug(s, "mat1", 0.5, -3.0, 1.8, 0.7, mat="m_rug")
    P.rug(s, "mat2", 0.5, -1.9, 1.8, 0.7, mat="m_rug")
    s.geom("wall_clock", "cylinder", (0, -4.44, 2.4), (0.22, 0.01), euler=(1.570795, 0, 0),
           mat="m_white", collide=False)
    s.geom("cooler", "box", (-5.6, -3.9, 0.55), (0.25, 0.25, 0.55), mat="m_steel")
    for i, x in enumerate((-3.0, 0.0, 3.0)):
        P.ceiling_lamp(s, f"bay{i + 1}", x, 0.5, z=3.0, r=0.24)
    s.light("key", (2.0, -2.0, 3.4), (-0.25, 0.3, -1), "0.5 0.5 0.5", castshadow=True)
    s.light("fill", (-3.0, 2.0, 3.4), (0.35, -0.25, -1), "0.24 0.24 0.26")
    s.comment("free weights (graspable)")
    P.free_sphere(s, "kettlebell", 1.1, -2.9, 0.12, 0.12, "0.16 0.17 0.19 1", mass=4.0)
    for i, x in enumerate((-0.4, 0.1)):
        P.free_cylinder(s, f"dumbbell{i + 1}", x, -3.05, 0.06, 0.05, 0.09,
                        "0.30 0.32 0.36 1", mass=2.0, yaw=1.570795)
    P.free_sphere(s, "medball", 4.0, -1.8, 0.14, 0.14, "0.55 0.30 0.58 1", mass=3.0)
    return s


def build_all():
    return [washroom(), living_room_loft(), living_room_family(), full_house(),
            gym()]
