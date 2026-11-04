# OpenArm (Enactic) — official link masses, COM and inertia

Retrieved from the official Enactic GitHub org (`github.com/enactic`) and
`openarm.dev` / `docs.openarm.dev`. **Every number below is quoted verbatim from the source file**
(no rounding, no derivation), except sums, which are marked as computed.

---

## 0. Provenance / where the numbers live

| Artifact | Repo | Path | Raw URL |
|---|---|---|---|
| **Source of truth for inertials (v2.0 arm)** | `enactic/openarm_description` | `assets/robot/openarm_v2.0/config/arm/inertials/nominal.yaml` | https://raw.githubusercontent.com/enactic/openarm_description/main/assets/robot/openarm_v2.0/config/arm/inertials/nominal.yaml |
| Source of truth for inertials (v1.0 arm) | `enactic/openarm_description` | `assets/robot/openarm_v1.0/config/arm/inertials.yaml` | https://raw.githubusercontent.com/enactic/openarm_description/main/assets/robot/openarm_v1.0/config/arm/inertials.yaml |
| Body / pedestal (both versions) | `enactic/openarm_description` | `assets/robot/openarm_v2.0/config/body/inertials/nominal.yaml` | https://raw.githubusercontent.com/enactic/openarm_description/main/assets/robot/openarm_v2.0/config/body/inertials/nominal.yaml |
| Pinch gripper (v2.0 EE) | `enactic/openarm_description` | `assets/end_effector/pinch_gripper/config/inertials/nominal.yaml` | https://raw.githubusercontent.com/enactic/openarm_description/main/assets/end_effector/pinch_gripper/config/inertials/nominal.yaml |
| Parallel-link gripper (v1.0 EE) | `enactic/openarm_description` | `assets/end_effector/parallel_link/config/inertials.yaml` | https://raw.githubusercontent.com/enactic/openarm_description/main/assets/end_effector/parallel_link/config/inertials.yaml |
| Generated example URDF (v2.0, bimanual + pinch grippers) | `enactic/openarm_description` | `assets/robot/openarm_v2.0/urdf/example/v2.urdf` | https://raw.githubusercontent.com/enactic/openarm_description/main/assets/robot/openarm_v2.0/urdf/example/v2.urdf |
| Generated example URDF (v1.0) | `enactic/openarm_description` | `assets/robot/openarm_v1.0/urdf/example/v1.urdf` | https://raw.githubusercontent.com/enactic/openarm_description/main/assets/robot/openarm_v1.0/urdf/example/v1.urdf |
| MJCF (v2.0 bimanual) | `enactic/openarm_mujoco` (branch **master**) | `v2/openarm_bimanual.xml` | https://raw.githubusercontent.com/enactic/openarm_mujoco/master/v2/openarm_bimanual.xml |
| MJCF (v1) | `enactic/openarm_mujoco` | `v1/openarm.xml`, `v1/openarm_bimanual.xml` | https://raw.githubusercontent.com/enactic/openarm_mujoco/master/v1/openarm.xml |
| Motor spec table (official) | `enactic/openarm` (docs site) | `website/docs/hardware/openarm-2.0/motor.mdx` | https://raw.githubusercontent.com/enactic/openarm/main/website/docs/hardware/openarm-2.0/motor.mdx |
| Payload / DOF spec | `enactic/openarm` | `website/docs/hardware/openarm-2.0/general.mdx`, `website/docs/overview/index.mdx` | https://docs.openarm.dev/hardware/openarm-2.0/general |
| Arm weight / reach / BOM cost | openarm.dev landing page (Nuxt bundle `/_nuxt/BEQSa1vG.js`, animated counters) | — | https://openarm.dev/ |
| DaMiao datasheets (EN) | `enactic/damiao` | `website/i18n/en/docusaurus-plugin-content-docs/current/products/hardware/*.mdx` | https://damiao.enactic.ai/en/products/hardware/… |

**mujoco_menagerie: NOT FOUND.** `google-deepmind/mujoco_menagerie` has **no** openarm/enactic entry
(full top-level listing checked — closest arms are `trs_so_arm100`, `ufactory_xarm7`, `arx_l5`).
The only official MJCF is `enactic/openarm_mujoco`.

**Consistency check performed:** the v2.0 masses/COMs in `inertials/nominal.yaml`, the generated
`v2.urdf`, and the MJCF `v2/openarm_bimanual.xml` are **identical** (MJCF stores the same COM with a
diagonalized inertia + quaternion; masses agree to the printed precision, e.g. `mass="1.14323"`).

---

## 1. OpenArm **v2.0** — per-link mass + COM + inertia (THE CURRENT ARM)

Source (verbatim): `openarm_description/assets/robot/openarm_v2.0/config/arm/inertials/nominal.yaml`
Confirmed identical in `assets/robot/openarm_v2.0/urdf/example/v2.urdf` (as `openarm_right_*`) and in
`openarm_mujoco/v2/openarm_bimanual.xml`.

Frame convention: COM `xyz` is in the **link frame**, all `rpy = 0`. The YAML values are the
**right-arm** convention; the **left arm mirrors `y` (and the signs of `ixy`, `iyz`)** — see `v2.urdf`.
Inertia is about the COM (URDF `<inertial><origin>` + `<inertia>`).

| Link | mass (kg) | COM x (m) | COM y (m) | COM z (m) | ixx | iyy | izz | ixy | ixz | iyz |
|---|---|---|---|---|---|---|---|---|---|---|
| `base_link` | 1.1432284943239561 | 0.00253123748061566 | -0.0258185540330169 | 2.70658291858903e-05 | 0.00113607806226198 | 0.00159935208714031 | 0.000750541980863571 | 3.81487993180929e-07 | 1.09161247709715e-08 | 2.7521421737084e-08 |
| `link1` | 1.1416684646202298 | 0.000900350355624244 | -0.0444659716892874 | 2.88441021165683e-05 | 0.0010943212038551 | 0.000759199728161993 | 0.000670659201019629 | 9.41628673674543e-06 | 4.53422182483102e-08 | 6.66755698859637e-08 |
| `link2` | 0.2775092746011571 | -0.0247184924346635 | 7.60578399428002e-08 | -0.0312643102292087 | 0.000153588833784759 | 0.000113062180225725 | 8.22999351863579e-05 | -2.94422592188677e-10 | 5.21056076779068e-06 | 3.75816911469744e-10 |
| `link3` | 1.073863338202347 | -0.00300861804638783 | -0.000514813312550808 | -0.109368241429434 | 0.00128939636591746 | 0.00114018992179734 | 0.000559765478676303 | -2.29951046666492e-07 | -6.76161926984358e-05 | -1.06260802872572e-05 |
| `link4` | 0.6348534566833373 | -0.00310326243752606 | -0.00136538979048502 | -0.0574601379354351 | 0.0004690036047182 | 0.000433926270522842 | 0.000245211429759716 | 5.63626910265645e-06 | -1.77877991597693e-06 | 3.41757797773822e-06 |
| `link5` | 0.6156588026168502 | -0.000737188580130584 | -0.000412406544351845 | -0.047093829077211 | 0.000279653368752989 | 0.000334599623120095 | 0.000176662464388138 | 3.31745289972445e-06 | 7.68097311403736e-07 | -1.49934409279756e-07 |
| `link6` | 0.475202773187987 | 2.27438423593244e-05 | 0.00068649602098908 | 5.44000494765995e-05 | 0.000179275941446964 | 0.000136247512758587 | 0.000135340695937386 | 6.58439346712228e-08 | -3.37101167120887e-08 | 3.32253600295544e-08 |
| `link7` | **no inertial — massless** | — | — | — | — | — | — | — | — | — |

> **`link7` note (important).** The v2.0 topology (`config/arm/struct/topology.yaml`) lists 8 links
> (`base_link, link1…link7`) and 7 revolute joints, but `link7` maps to `empty_link_for_J7`
> (`config/arm/struct/name_mapping.yaml`) and **has no entry in `inertials/nominal.yaml`** — it is a
> zero-mass frame. In the generated `v2.urdf`, `joint7` goes **`link6 → ee_base_link`** directly, i.e.
> the J7 (wrist-roll) distal inertia is entirely the end-effector's. So a 7-DOF arm has only
> **7 inertial bodies** (`base_link` + `link1…link6`) plus the EE.

**Per-arm total, v2.0, arm only (no EE) — COMPUTED sum of the 7 masses above:**
**`5.3619846042358645 kg`**

### v2.0 joint origins (for building the chain) — `config/arm/joint/joint_origins.yaml`
| joint | x | y | z | (URDF axis, right arm, from `v2.urdf`) |
|---|---|---|---|---|
| joint1 | 0 | -0.0625 | 0 | `0 -1 0` |
| joint2 | 0 | -0.0600000000000001 | 0 | `-1 0 0` |
| joint3 | 0 | 0 | -0.06625000000000079 | `0 0 -1` |
| joint4 | 0 | 0 | -0.15375 | `0 -1 0` |
| joint5 | 0 | 0 | -0.0955000000000005 | `0 0 -1` |
| joint6 | 0 | 0 | -0.120499999999998 | `0 1 0` |
| joint7 | 0 | 0 | 0 | `1 0 0` |

(All `rpy = 0`. Arm mount: `openarm_body_link0 → openarm_right_base_link` at `xyz = 0.0 -0.031 0.698`.)

---

## 2. OpenArm **v2.0 pinch gripper** (the current end-effector)

Source: `assets/end_effector/pinch_gripper/config/inertials/nominal.yaml` (COM sign convention as
published; the mirrored/right-arm signs in `v2.urdf` are given in parentheses where they differ).

| Link (URDF name in `v2.urdf`) | mass (kg) | COM x | COM y | COM z | ixx | iyy | izz | ixy | ixz | iyz |
|---|---|---|---|---|---|---|---|---|---|---|
| `base_link` (`*_ee_base_link`) | 0.52832 | -0.0067098 *(URDF: 0.0137902)* | 0.008407299999999999 *(right arm: 0.0084073)* | -0.05938 | 0.00021911 | 0.00015627 | 0.0001659 | 4.1683e-06 | 1.4043e-06 | -1.5124e-06 |
| `link1` (`*_ee_link1` / MJCF `ee_inner_finger`) | 0.125979047366671 | -0.0124990911725494 | 0.00176592789915655 | -0.0241663193888537 | 3.61105363396623e-05 | 3.62137424319396e-05 | 1.74958961632907e-05 | 1.35880208293387e-06 | 7.13916341634141e-06 | -3.60343788693423e-06 |
| `link2` (`*_ee_link2` / MJCF `ee_outer_finger`) | 0.11372 | -0.012796 | -0.0020183 | -0.026792 | 3.4565e-05 | 3.5306e-05 | 1.6571e-05 | -1.3392e-06 | 7.1576e-06 | 3.7172e-06 |

**Pinch-gripper total — COMPUTED:** `0.768019047366671 kg`
**v2.0 arm + pinch gripper — COMPUTED:** `6.130003651602536 kg`

---

## 3. OpenArm **v1.0** — per-link mass + COM + inertia (previous generation)

Source: `assets/robot/openarm_v1.0/config/arm/inertials.yaml` (identical to `urdf/example/v1.urdf`).
**Masses are identical to v2.0 for link0…link6** — only the link frames (hence COM xyz) and the
(coarser, 6-dp-rounded) inertia tensors differ. v1.0 **does** have a massive `link7`.

| Link | mass (kg) | COM x (m) | COM y (m) | COM z (m) | ixx | iyy | izz | ixy | ixz | iyz |
|---|---|---|---|---|---|---|---|---|---|---|
| `link0` | 1.1432284943239561 | -0.0009483362816297526 | 0.0001580207020448382 | 0.03076860287587199 | 0.001128 | 0.000962 | 0.00147 | -4.0e-06 | -3.3e-05 | -7.0e-06 |
| `link1` | 1.1416684646202298 | 0.0011467657911800769 | 3.319987657026362e-05 | 0.05395284380736254 | 0.001567 | 0.001273 | 0.001016 | -1.0e-06 | -2.9e-05 | 1e-06 |
| `link2` | 0.2775092746011571 | 0.00839629182351943 | -2.0145102027597523e-08 | 0.03256649300522363 | 0.000359 | 0.000376 | 0.000232 | 1e-06 | -0.000109 | 1e-06 |
| `link3` | 1.073863338202347 | -0.002104752099628911 | 0.0005549085042607548 | 0.09047470545721961 | 0.004372 | 0.004319 | 0.000661 | 1e-06 | 1.1e-05 | -3.6e-05 |
| `link4` | 0.6348534566833373 | -0.0029006831074562967 | -0.03030575826634669 | 0.06339637422196209 | 0.000623 | 0.000511 | 0.000334 | -1.0e-06 | -1.9e-05 | 3.8e-05 |
| `link5` | 0.6156588026168502 | -0.003049665024221911 | 0.0008866902457326625 | 0.043079803024980934 | 0.000423 | 0.000445 | 0.000324 | -8.0e-06 | 6.0e-06 | -6.0e-06 |
| `link6` | 0.475202773187987 | -0.037136587005447405 | 0.00033230528343419053 | -9.498374522309838e-05 | 0.000143 | 0.000157 | 0.000159 | 1e-06 | 1e-06 | 1e-06 |
| `link7` | 0.4659771327380578 | 6.875510271106056e-05 | 0.01266175250761268 | 0.06951945409987448 | 0.000639 | 0.000497 | 0.000342 | 1e-06 | 1e-06 | 8.9e-05 |

**Per-arm total, v1.0, link0…link7 (no fingers) — COMPUTED:** `5.827961736973922 kg`

### v1.0 parallel-link gripper — `assets/end_effector/parallel_link/config/inertials.yaml`
| Link | mass (kg) | COM x | COM y | COM z | ixx | iyy | izz | ixy | ixz | iyz |
|---|---|---|---|---|---|---|---|---|---|---|
| `hand` | 0.127 | 0.000001 | 0.0 | -0.000549 | 1.65558e-04 | 1.74660e-05 | 1.73524e-04 | 0.0 | 0.0 | 0.0 |
| `left_finger` | 0.03602545343277134 | 0.0064528 | 0.017020 | 0.0219685 | 2.3749999999999997e-06 | 2.3749999999999997e-06 | 7.5e-07 | 1e-06 | 1e-06 | 1e-06 |
| `right_finger` | 0.03602545343277134 | 0.0064528 | -0.017020 | 0.0219685 | 2.3749999999999997e-06 | 2.3749999999999997e-06 | 7.5e-07 | 1e-06 | 1e-06 | 1e-06 |

> In the generated `v1.urdf` the arm carries **only the two fingers** (`0.03602545343277134` each) — the
> `hand` body is folded into `link7` (0.4659771327380578). The standalone `hand` (0.127 kg) appears only
> in the standalone `parallel_link` EE description.
> **v1.0 arm + 2 fingers — COMPUTED:** `5.900012643839465 kg`

---

## 4. Body / pedestal (both versions) — `config/body/inertials/nominal.yaml`

| Link | mass (kg) | COM x | COM y | COM z | ixx | iyy | izz | ixy | ixz | iyz |
|---|---|---|---|---|---|---|---|---|---|---|
| `body_link0` | 13.89 | 0 | 0 | 0 | 1.653 | 1.653 | 0.051 | 0.0 | 0.0 | 0.0 |

This is the MISUMI-frame pedestal both arms mount on (`base_link` at `z = 0.698`), **not** part of the arm.

**Full bimanual v2.urdf total (both arms + both pinch grippers + body) — COMPUTED:** `26.15000730320508 kg`
(= 2 × 6.130003651602536 + 13.89.)

---

## 5. Official published specs (openarm.dev / docs.openarm.dev)

| Spec | Official value | Source |
|---|---|---|
| Version | **OpenArm 2.0** (current; 1.0 docs archived at `docs.openarm.dev/1.0/`) | `website/docs/overview/whats-new-in-2.0.mdx` |
| DOF | **7 per arm** | `website/docs/overview/index.mdx` ("Specifications at a Glance") |
| **Arm Reach** | **633 mm** | openarm.dev landing-page counter (`/_nuxt/BEQSa1vG.js`: `{counter:u, num:633}` → unit `mm`, label "Arm Reach") |
| **Weight per Arm** | **5.5 kg** | same bundle: `{counter:v, num:55}` rendered as `(v/10).toFixed(1)` kg, label "Weight per Arm" |
| Nominal payload | **4.1 kg** (held 1 min in worst posture, arms fully extended; *includes* the end-effector) | `general.mdx`, landing counter `num:41` |
| Peak payload | **6.0 kg** (3 s move + 1 s hold in worst posture; *includes* the end-effector) | `general.mdx`, landing counter `num:60` |
| Control | **1 kHz CAN-FD** | landing counter `num:1` kHz "CAN-FD Control"; `overview/index.mdx` "Control bus: CAN-FD" |
| BOM cost | **$6,500** | landing counter `num:6500` |
| Human scale | proportions for a person ~**160–165 cm** tall | `overview/index.mdx` |
| Mass variants | **Only one arm variant.** The gripper is the variant axis: v2.0 pinch gripper (0.768019047366671 kg) vs v1.0 parallel-link gripper. `openarm_description` presets: `right_arm.yaml`, `left_arm.yaml`, `*_with_pinch_gripper.yaml`, `default_bimanual.yaml` — i.e. **with / without gripper** | `assets/robot/openarm_v2.0/config/robot_presets/` |
| OpenArm KER (motorless leader replica) | **1.7 kg** | `website/docs/hardware/openarm-ker/general.mdx` |
| OpenArm Cell (eval enclosure) | ~100 kg, ~480 W | `website/docs/hardware/openarm-cell/general.mdx` |

**Reconciling "5.5 kg per arm" with the URDF:** the URDF arm links (`base_link`+`link1..link6`) sum to
**5.362 kg**; with the pinch gripper, **6.130 kg**. The published "Weight per Arm = 5.5 kg" sits between
the two and is **not** reproduced exactly by any URDF subset — treat 5.5 kg as a marketing/measured
figure and the URDF sums as the model-consistent numbers.

---

## 6. Motor–joint mapping (confirmed from official effort limits)

`assets/robot/openarm_v2.0/config/arm/joint/joint_limits.yaml` effort values match the datasheet **peak
torques** exactly, which independently confirms the mapping:

| Joint | effort limit (Nm) | velocity limit (rad/s) | position limits (rad) | ⇒ Motor |
|---|---|---|---|---|
| joint1 | 40 | 16.755 | -1.3963 … 3.4907 | **DM-J8009P-2EC** (peak 40 NM) |
| joint2 | 40 | 16.755 | -0.17453 … 3.3161 | **DM-J8009P-2EC** |
| joint3 | 27 | 5.4454 | -1.5708 … 1.5708 | **DM-J4340(P)-2EC** (peak 27 NM) |
| joint4 | 27 | 5.4454 | 0 … 2.4435 | **DM-J4340(P)-2EC** |
| joint5 | 7 | 20.944 | -1.5708 … 1.5708 | **DM-J4310-2EC** (peak 7 NM) |
| joint6 | 7 | 20.944 | -0.7854 … 0.7854 | **DM-J4310-2EC** |
| joint7 | 7 | 20.944 | -1.5708 … 1.5708 | **DM-J4310-2EC** |

Docs note (`motor.mdx`): "OpenArm 2.0 uses the DAMIAO 43 series and DAMIAO 8009P motors… motors
equipped with cross-roller bearings are used in sections supported on one side" (⇒ the "P" variants),
and: "While the linked datasheet is for the DM8009 model, the actual motor used in our arms is the
DM8009P."
`assets/robot/openarm_v2.0/config/arm/control/friction.yaml` exists but is **EMPTY** (no official
friction/damping numbers published).

---

## 7. DaMiao motor datasheets (verbatim, from `damiao.enactic.ai` = repo `enactic/damiao`)

| Parameter | **DM-J4310-2EC V1.1** | **DM-J4340-2EC** (24 V) | **DM-J4340P-2EC** (24 V, cross-roller) | **DM-J8009P-2EC** |
|---|---|---|---|---|
| **Motor weight** | **300 g approx** | **362 g approx** | **375 g Approx** | **Approx. 930 g** *(OpenArm's own motor.mdx table says **896 g** for DM-J8009P-2EC — the two official pages disagree)* |
| Reduction ratio | **10:1** | **40:1** | **40:1** | **9:1** |
| Rated torque | 3 NM | 9 NM | 9 NM | 20 NM |
| **Peak torque** | **7 NM** | **27 NM** | **27 NM** | **40 NM** |
| Rated / peak current | 2.5 A / 7.5 A | 2.5 A / 8 A | 2.5 A / 8 A | 20 A / 50 A |
| Rated voltage | 24 V | 24 V (48 V variant) | 24 V (48 V variant) | 24 V (24–48 V) |
| Rated speed | 120 rpm | 36 rpm | 36 rpm | 100 rpm (24 V) / 200 rpm (48 V) |
| Max no-load speed | 200 rpm | 52 rpm (24 V) / 100 rpm (48 V) | 52 rpm (24 V) / 100 rpm (48 V) | 24 V: 160 rpm / 48 V: 320 rpm |
| Pole pairs | 14 | 14 | 14 | 21 |
| Phase inductance | 340 μH | 317 μH *(motor.mdx)* | 360 μH *(datasheet)* | 61 μH @25 °C *(motor.mdx)* / 80 μH *(datasheet)* |
| Phase resistance | 650 mΩ | 760 mΩ *(motor.mdx)* | 880 mΩ *(datasheet)* | 90 mΩ @25 °C *(motor.mdx)* / 0.145 Ω *(datasheet)* |
| Outer diameter × height | 56 × 46 mm | 57 × 53.3 mm | 57 × 56.5 mm | 98 × 61.7 mm |
| Encoder | 14-bit, ×2, magnetic single-turn | 14-bit, ×2 | 14-bit, ×2 | 14-bit, ×2 |
| **Rotor inertia** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** | **NOT FOUND** |

> **Rotor / reflected inertia is NOT PUBLISHED by DaMiao.** The only "inertia" in the datasheets is a
> **read-only CAN register** (`0x0C` `Inertia` — "Motor rotational inertia", `float`, RO) in the
> DM-J4340P register list, i.e. it can be *read off the motor* but no datasheet number is given.
> The same holds for the `Damp` register (`0x0B`, "Motor damping factor", RO).
> ⇒ For MABEL's `armature` / `damping` / `frictionloss`, the identified values from
> `calibration/motor_calibration/` remain the only source.
>
> **DM-J8009-2EC vs DM-J8009P-2EC:** only the *P* (cross-roller) manual is on damiao.enactic.ai; there is
> no separate DM-J8009-2EC page (OpenArm's motor.mdx links the same 8009P page and warns the DM8009
> datasheet is used as a stand-in). Difference is the output bearing only:
> "*The output shaft of DM-J8009P-2EC uses a crossed roller bearing; the output shaft of DM-J8009-2EC
> uses a thin-walled deep groove ball bearing.*"
> Note also: OpenArm uses **DM-J4340 series** for J3/J4 — motor.mdx lumps DM-J4340-2EC and
> DM-J4340P-2EC as "DM4340 series"; both are 40:1, 9/27 NM.

Sanity check on motor mass vs link mass: 2 × 930 g (J1,J2) + 2 × 375 g (J3,J4) + 3 × 300 g (J5–J7)
= **3.51 kg of motors** inside a **5.36 kg** URDF arm — consistent.

---

## 8. Explicit NOT-FOUNDs

- **mujoco_menagerie entry for OpenArm** — NOT FOUND (no `enactic_openarm` / `openarm` directory).
- **Rotor inertia / gearbox inertia** for DM-J4310, DM-J4340(P), DM-J8009(P) — NOT FOUND in any official
  datasheet (only a runtime CAN register).
- **Official friction/damping numbers** — NOT FOUND (`config/arm/control/friction.yaml` is empty).
- **Official arm mass stated in the docs site** — NOT FOUND in `docs.openarm.dev`; the only official arm
  mass anywhere is the **openarm.dev landing counter: 5.5 kg per arm**.
- **v2.0 `link7` inertial** — genuinely absent (massless `empty_link_for_J7`), not a fetch failure.
- **Reach in the docs site** — NOT FOUND as text (only the landing counter, 633 mm, and a dimensioned
  image `v2_mech_stop_conf.png`).
