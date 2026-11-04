"""core.py -- Scene builder (MJCF emit + dynamics sidecar + registry writer)."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
SCENES_DIR = os.path.join(ROOT, "scenes")
REGISTRY = os.path.join(SCENES_DIR, "scenes.json")

# ── shared texture/material library ──────────────────────────────────────────
# material name -> (texture file under assets/textures or None, extra attrs)
MAT = {
    # architecture
    "m_wall":      ("wall.png",           'texuniform="true" texrepeat="2 2" reflectance="0.0"'),
    "m_concrete":  ("concrete.png",       'texuniform="true" texrepeat="0.6 0.6" reflectance="0.08" shininess="0.2"'),
    "m_wood_floor": ("wood_floor.png",    'texuniform="true" texrepeat="0.8 0.8" reflectance="0.10" shininess="0.3"'),
    "m_tile":      ("tile.png",           'texuniform="true" texrepeat="0.7 0.7" reflectance="0.15" shininess="0.4"'),
    "m_tile_small": ("tile_small.png",    'texuniform="true" texrepeat="1.4 1.4" reflectance="0.12" shininess="0.4"'),
    "m_tile_checker": ("tile_checker.png", 'texuniform="true" texrepeat="0.9 0.9" reflectance="0.10" shininess="0.3"'),
    "m_carpet_gray": ("carpet_gray.png",  'texuniform="true" texrepeat="1.2 1.2" reflectance="0.0"'),
    "m_carpet_beige": ("carpet_beige.png", 'texuniform="true" texrepeat="1.2 1.2" reflectance="0.0"'),
    "m_rubber":    ("rubber_floor.png",   'texuniform="true" texrepeat="0.9 0.9" reflectance="0.04" shininess="0.15"'),
    "m_brick":     ("brick.png",          'texuniform="true" texrepeat="1.5 1.5" reflectance="0.0"'),
    "m_marble":    ("marble_counter.png", 'texuniform="true" texrepeat="1 1" reflectance="0.25" shininess="0.6"'),
    "m_metal":     ("metal_brushed.png",  'texuniform="true" texrepeat="2 2" reflectance="0.35" shininess="0.7" specular="0.5"'),
    "m_hazard":    ("hazard.png",         'texuniform="true" texrepeat="2 2" reflectance="0.0" shininess="0.1"'),
    "m_cardboard": ("cardboard.png",      'texuniform="true" texrepeat="1 1" reflectance="0.0"'),
    "m_facade":    ("facade.png",         'texuniform="true" texrepeat="0.11 0.11" reflectance="0.05" shininess="0.2"'),
    "m_storefront": ("storefront.png",    'texuniform="true" texrepeat="0.14 0.14" reflectance="0.3" shininess="0.6" emission="0.25"'),
    "m_curtain":   ("curtainwall.png",    'texuniform="true" texrepeat="0.10 0.10" reflectance="0.5" shininess="0.85" specular="0.7" emission="0.1"'),
    "m_brownstone": ("brownstone.png",    'texuniform="true" texrepeat="0.14 0.14" reflectance="0.04" shininess="0.15"'),
    "m_artdeco":   ("artdeco.png",        'texuniform="true" texrepeat="0.09 0.09" reflectance="0.12" shininess="0.35" emission="0.08"'),
    # soft / fabric
    "m_wood":      ("wood_furniture.png", 'texuniform="true" texrepeat="2 2" reflectance="0.05" shininess="0.2"'),
    "m_fabric":    ("fabric_blue.png",    'texuniform="true" texrepeat="2 2" reflectance="0.0"'),
    "m_bedding":   ("bedding.png",        'texuniform="true" texrepeat="1.2 1.2" reflectance="0.0"'),
    "m_rug":       ("rug.png",            'texrepeat="1 1" reflectance="0.0"'),
    "m_tablecloth": ("tablecloth.png",    'texuniform="true" texrepeat="1.5 1.5" reflectance="0.0"'),
    "m_book":      ("book_spines.png",    'texrepeat="1 1" reflectance="0.0"'),
    # outdoor
    "m_grass":     ("grass.png",          'texuniform="true" texrepeat="1.5 1.5" reflectance="0.0" shininess="0.0"'),
    "m_fairway":   ("fairway.png",        'texuniform="true" texrepeat="0.8 0.8" reflectance="0.0" shininess="0.0"'),
    "m_sand":      ("sand.png",           'texuniform="true" texrepeat="1.2 1.2" reflectance="0.0"'),
    "m_gravel":    ("gravel.png",         'texuniform="true" texrepeat="3 3" reflectance="0.0" shininess="0.05"'),
    "m_foliage":   ("foliage.png",        'texuniform="true" texrepeat="3 3" reflectance="0.0" shininess="0.0"'),
    "m_bark":      ("bark.png",           'texuniform="true" texrepeat="2 2" reflectance="0.0" shininess="0.05"'),
    "m_water":     ("water.png",          'texuniform="true" texrepeat="0.8 0.8" reflectance="0.45" shininess="0.8" specular="0.6"'),
    # street
    "m_asphalt":   ("asphalt.png",        'texuniform="true" texrepeat="0.5 0.5" reflectance="0.06" shininess="0.15"'),
    "m_sidewalk":  ("sidewalk.png",       'texuniform="true" texrepeat="0.6 0.6" reflectance="0.05" shininess="0.1"'),
    "m_crosswalk": ("crosswalk.png",      'texrepeat="1 1" reflectance="0.05" shininess="0.1"'),
    # retail / food
    "m_products":  ("products.png",       'texrepeat="1 1" reflectance="0.0" emission="0.12"'),
    "m_menu":      ("menu_board.png",     'texrepeat="1 1" reflectance="0.2" shininess="0.6" emission="0.6"'),
    "m_awning_red": ("awning_red.png",    'texrepeat="2 1" reflectance="0.0"'),
    "m_awning_green": ("awning_green.png", 'texrepeat="2 1" reflectance="0.0"'),
    "m_screen":    ("monitor.png",        'texrepeat="1 1" reflectance="0.3" shininess="0.8" emission="0.55"'),
    "m_whiteboard": ("whiteboard.png",    'texrepeat="1 1" reflectance="0.1"'),
    "m_belt":      ("conveyor.png",       'texrepeat="14 1" reflectance="0.05" shininess="0.2"'),
    # plain colors
    "m_frame":     (None, 'rgba="0.30 0.42 0.55 1" reflectance="0.25" shininess="0.5"'),
    "m_dark":      (None, 'rgba="0.16 0.17 0.19 1" reflectance="0.2" shininess="0.4"'),
    "m_white":     (None, 'rgba="0.92 0.92 0.90 1" reflectance="0.1" shininess="0.3"'),
    "m_cream":     (None, 'rgba="0.90 0.87 0.78 1" reflectance="0.05" shininess="0.2"'),
    "m_steel":     (None, 'rgba="0.62 0.65 0.68 1" reflectance="0.4" shininess="0.7" specular="0.5"'),
    "m_chrome":    (None, 'rgba="0.75 0.77 0.80 1" reflectance="0.6" shininess="0.9" specular="0.8"'),
    "m_glass":     (None, 'rgba="0.62 0.74 0.80 0.35" reflectance="0.4" shininess="0.9"'),
    "m_mirror":    (None, 'rgba="0.80 0.84 0.88 1" reflectance="0.85" shininess="1.0" specular="1.0"'),
    "m_lamp":      (None, 'rgba="1.0 0.96 0.85 1" emission="1.0"'),
    "m_sig_red":   (None, 'rgba="0.95 0.15 0.10 1" emission="0.9"'),
    "m_sig_green": (None, 'rgba="0.15 0.9 0.30 1" emission="0.9"'),
    "m_sig_amber": (None, 'rgba="1.0 0.65 0.10 1" emission="0.9"'),
}


def f(v):
    """Compact float -> str (matches the hand-authored scenes' style)."""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(f(x) for x in v)
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def rot(dx, dy, yaw):
    """Rotate a local XY offset by yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    return dx * c - dy * s, dx * s + dy * c


class Scene:
    def __init__(self, name, title, category, description, tags="",
                 headlight=("0.46 0.46 0.48", "0.26 0.26 0.26"),
                 skybox=("0.52 0.58 0.66", "0.86 0.89 0.93"),
                 fog=None, dynamic=False, viewer_hint=None):
        self.name = name
        self.title = title
        self.category = category
        self.description = description
        self.headlight = headlight
        self.skybox = skybox
        self.fog = fog                    # (rgba, fogstart, fogend) or None
        self.dynamic = dynamic
        self.viewer_hint = viewer_hint
        self._mats = set()
        self._lights = []
        self._geoms = []                  # xml lines inside worldbody
        self._bodies = []                 # xml blocks (actors, jointed props)
        self._counters = {}
        self.dyn = {"version": 1, "actors": [], "conveyors": []}

    # ── low-level emit ───────────────────────────────────────────────────────
    def nid(self, prefix):
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}{n}"

    def light(self, name, pos, direction, diffuse, castshadow=False,
              directional=True, specular=None):
        a = (f'<light name="{name}" directional="{"true" if directional else "false"}" '
             f'pos="{f(pos)}" dir="{f(direction)}" diffuse="{f(diffuse)}"')
        if specular:
            a += f' specular="{f(specular)}"'
        a += f' castshadow="{"true" if castshadow else "false"}"/>'
        self._geoms.append(a)

    def geom(self, name, gtype, pos, size, mat=None, rgba=None, euler=None,
             fromto=None, collide=True, extra=""):
        a = f'<geom name="{name}" type="{gtype}"'
        if fromto is not None:
            a += f' fromto="{f(fromto)}"'
        else:
            a += f' pos="{f(pos)}"'
        a += f' size="{f(size)}"'
        if euler is not None:
            a += f' euler="{f(euler)}"'
        if mat:
            self._mats.add(mat)
            a += f' material="{mat}"'
        if rgba:
            a += f' rgba="{f(rgba)}"'
        if not collide:
            a += ' contype="0" conaffinity="0"'
        if extra:
            a += " " + extra
        self._geoms.append(a + "/>")

    def raw(self, xml, *mats):
        """Append a raw worldbody block (bodies with joints, etc.)."""
        self._bodies.append(xml)
        for m in mats:
            self._mats.add(m)

    def comment(self, text):
        self._geoms.append(f"<!-- ===== {text} ===== -->")

    # ── dynamics sidecar ─────────────────────────────────────────────────────
    def actor(self, spec):
        self.dyn["actors"].append(spec)
        self.dynamic = True

    def conveyor(self, spec):
        self.dyn["conveyors"].append(spec)
        self.dynamic = True

    # ── files ────────────────────────────────────────────────────────────────
    @property
    def xml_name(self):
        return f"mabel_{self.name}.xml"

    def write(self, scenes_dir=SCENES_DIR):
        assets = []
        for m in sorted(self._mats):
            tex, attrs = MAT[m]
            if tex:
                tname = "t_" + m[2:]
                assets.append(f'    <texture type="2d" name="{tname}" '
                              f'file="../assets/textures/{tex}"/>')
                assets.append(f'    <material name="{m}" texture="{tname}" {attrs}/>')
            else:
                assets.append(f'    <material name="{m}" {attrs}/>')
        fog = ""
        if self.fog:
            rgba, f0, f1 = self.fog
            fog = (f'    <rgba fog="{rgba}"/>\n'
                   f'    <map fogstart="{f(f0)}" fogend="{f(f1)}"/>\n')
        world = "\n".join("    " + ln for ln in self._geoms)
        bodies = "\n\n".join(self._bodies)

        # Explicit robot↔actor contact pairs. MuJoCo's mask-based broadphase
        # silently drops the actor-vs-robot box/capsule pairs in these large
        # scenes (verified: mj_geomDistance shows overlap but auto-collision
        # yields no contact), so a car/person would pass straight through the
        # robot. An explicit <pair> bypasses broadphase + masks and always runs
        # narrowphase, giving a real contact that shoves the robot. Pair every
        # kinematic actor's collision geom against the base proxy `chassis_bumper`
        # (walkers/pets → <tag>_hit box/capsule, vehicles → <tag>_body box).
        _COLL = {"walker": "_hit", "vehicle": "_body"}
        pair_lines = []
        for a in self.dyn["actors"]:
            root = a.get("root", "")
            suf = _COLL.get(a.get("kind", ""))
            if suf and root.endswith("_root"):
                tag = root[:-5]
                pair_lines.append(
                    f'    <pair geom1="{tag}{suf}" geom2="chassis_bumper" '
                    f'solref="0.02 1" solimp="0.8 0.9 0.01"/>')
        contact_block = (
            "\n  <contact>\n" + "\n".join(pair_lines) + "\n  </contact>\n"
        ) if pair_lines else ""
        xml = f"""<mujoco model="mabel_{self.name}_environment">
  <!-- {self.description}
       AUTO-GENERATED by scripts/tools/scenegen (gen_extra_scenes.py) -- edit the
       builder, not this file. Textures: gen_home_textures.py / gen_env_textures.py /
       gen_env_textures2.py.{chr(10) + '       Dynamic: scripted actors in mabel_' + self.name + '.dynamics.json, driven by simulation_bridge/dynamics.py.' if self.dyn['actors'] or self.dyn['conveyors'] else ''}
       Run from repo root:  python scripts/visualize_sensors.py --scene scenes/{self.xml_name} -->

  <compiler angle="radian" eulerseq="XYZ"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"
          cone="elliptic" noslip_iterations="3"/>

  <include file="../models/mabel_full.xml"/>

  <visual>
    <headlight ambient="{self.headlight[0]}" diffuse="{self.headlight[1]}" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
{fog}  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="{self.skybox[0]}" rgb2="{self.skybox[1]}" width="256" height="256"/>
{chr(10).join(assets)}
  </asset>

  <worldbody>
{world}

{bodies}

    <!-- ===== teleop mocap targets (kept compatible with the teleop / WBC scripts) =====
         The bodies are load-bearing (scripts/avp_teleop.py drives IK by looking them
         up by name), so they stay. Their marker spheres sit in geom group 3, which
         MuJoCo hides by default - same "debug decoration, off unless you ask" rule
         mabel_full.xml uses for the camera/IMU markers. Toggle group 3 in the viewer
         to bring them back when debugging IK. -->
    <body name="target_l" mocap="true" pos="0.45 0.3 1.0">
      <geom group="3" type="sphere" size="0.03" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </body>
    <body name="target_r" mocap="true" pos="0.45 -0.3 1.0">
      <geom group="3" type="sphere" size="0.03" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </body>
    <body name="gaze_point" mocap="true" pos="0.85 0 1.0">
      <geom group="3" type="sphere" size="0.04" rgba="1 0.5 0 0.4" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
{contact_block}
</mujoco>
"""
        path = os.path.join(scenes_dir, self.xml_name)
        with open(path, "w") as fp:
            fp.write(xml)
        print("wrote", path)

        dyn_path = os.path.join(scenes_dir, f"mabel_{self.name}.dynamics.json")
        if self.dyn["actors"] or self.dyn["conveyors"]:
            with open(dyn_path, "w") as fp:
                json.dump(self.dyn, fp, indent=1)
                fp.write("\n")
            print("wrote", dyn_path)
        elif os.path.exists(dyn_path):
            os.remove(dyn_path)

    def registry_entry(self):
        return {"name": self.name, "title": self.title, "xml": self.xml_name,
                "thumb": f"thumbs/{self.name}.png",
                "description": self.description,
                "category": self.category,
                **({"dynamic": True} if self.dynamic else {})}


# ── registry ──────────────────────────────────────────────────────────────────
# The hand-authored scenes keep their entries verbatim, but gain a category
# (and the assembly line its dynamic flag: gen_assembly_dynamics.py sidecar).
HAND_AUTHORED = [
    ("home",          "Home & Living"),
    ("living_room",   "Home & Living"),
    ("bedroom",       "Home & Living"),
    ("kitchen_fk",    "Food & Dining"),
    ("office",        "Work & Public"),
    ("factory",       "Industrial"),
    ("assembly_line", "Industrial"),
    ("warehouse",     "Industrial"),
    ("outdoor",       "Outdoor & Leisure"),
]
CATEGORY_ORDER = ["Home & Living", "Food & Dining", "Work & Public",
                  "Industrial", "Outdoor & Leisure", "City & Streets"]


def generate_registry(scenes):
    with open(REGISTRY) as fp:
        old = {e["name"]: e for e in json.load(fp)}
    entries = []
    for name, cat in HAND_AUTHORED:
        e = dict(old.get(name) or {})
        if not e:
            continue
        e["category"] = cat
        if name == "assembly_line":
            e["dynamic"] = os.path.exists(
                os.path.join(SCENES_DIR, "mabel_assembly_line.dynamics.json"))
        entries.append(e)
    entries += [s.registry_entry() for s in scenes]
    entries.sort(key=lambda e: (CATEGORY_ORDER.index(e.get("category", "Work & Public")),))
    with open(REGISTRY, "w") as fp:
        json.dump(entries, fp, indent=2)
        fp.write("\n")
    print(f"wrote {REGISTRY} ({len(entries)} scenes)")
