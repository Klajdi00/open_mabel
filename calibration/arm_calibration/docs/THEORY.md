# Theory: Lagrangian System Identification of Link Inertial Parameters

This note derives the math behind the identification code. It is written so you
can follow it without a robotics-dynamics background, then connect every line to
a function in `src/mabel_sysid/`.

---

## 1. The equation of motion (Euler–Lagrange)

For a rigid open-chain manipulator with joint coordinates `q ∈ ℝⁿ`, the Lagrangian
`L = T − P` (kinetic minus potential energy) yields the equations of motion

```
M(q) q̈  +  C(q, q̇) q̇  +  g(q)  =  τ
└──────┘   └─────────┘    └──┘     └─┘
 inertia    Coriolis/    gravity   joint
 torque     centrifugal            torque
```

- `M(q)` — symmetric positive-definite inertia matrix,
- `C(q,q̇) q̇` — velocity-dependent (Coriolis/centrifugal) torque,
- `g(q) = ∂P/∂q` — gravity torque,
- `τ` — the generalized force the actuators apply.

MuJoCo evaluates exactly this model. `mj_inverse` returns `τ` given `(q, q̇, q̈)`;
`qfrc_bias` returns `C(q,q̇)q̇ + g(q)`, which at standstill (`q̇ = 0`) is just `g(q)`.

---

## 2. Each link has 10 inertial parameters

The dynamics depend on the mass distribution of every link `i` through 10 numbers:

| symbol | meaning | count |
|--------|---------|-------|
| `mᵢ` | mass | 1 |
| `φᵢ = mᵢ cᵢ` | **first moment of mass** (mass × centre of mass) | 3 |
| `Iᵢ` | rotational inertia (`Ixx,Ixy,Ixz,Iyy,Iyz,Izz`) | 6 |

Collect them into `πᵢ = [mᵢ, φᵢ, Iᵢ] ∈ ℝ¹⁰`, and stack all links into
`π ∈ ℝ¹⁰ⁿ`.

**Centre of mass is recovered as** `cᵢ = φᵢ / mᵢ`.
This project identifies `mᵢ` and `cᵢ` (equivalently `mᵢ` and `φᵢ`); the rotational
inertia `Iᵢ` is taken from CAD and held fixed (see §6 for the extension).

---

## 3. The crucial property: dynamics are **linear in the parameters**

Although `M`, `C`, `g` are nonlinear in `q`, they are **linear in `π`**. Hence

```
τ  =  Y(q, q̇, q̈) · π
```

where `Y` is the **regressor** (a known function of the measured motion only).
This is what turns identification into linear regression. Stacking `K` measured
samples:

```
⎡ τ⁽¹⁾ ⎤     ⎡ Y(q⁽¹⁾, q̇⁽¹⁾, q̈⁽¹⁾) ⎤
⎢  ⋮   ⎥  =  ⎢          ⋮           ⎥ · π        →     τ_stack = Y_stack · π
⎣ τ⁽ᴷ⁾ ⎦     ⎣ Y(q⁽ᴷ⁾, q̇⁽ᴷ⁾, q̈⁽ᴷ⁾) ⎦
```

and the least-squares estimate is `π̂ = argminₚ ‖Y_stack π − τ_stack‖²`.

`scripts/00_check_model.py` verifies this linearity numerically: the analytic
gravity regressor reproduces MuJoCo's torque to ~1e-15.

---

## 4. The gravity sub-problem (what gravity compensation needs)

Gravity compensation only needs `g(q)`. The potential energy is

```
P(q) = − Σᵢ mᵢ gᵥ · p_cᵢ(q)
```

with `gᵥ` the gravity acceleration vector (`[0,0,−9.81]`) and `p_cᵢ` the **world**
position of link `i`'s COM. Differentiating,

```
g(q) = ∂P/∂q = − Σᵢ J_cᵢ(q)ᵀ (mᵢ gᵥ)
```

where `J_cᵢ = ∂p_cᵢ/∂q` is the 3×n translational Jacobian of the COM. Writing the
COM as the link-frame origin plus the rotated local offset, `p_cᵢ = pᵢ + Rᵢ cᵢ`,
and splitting the Jacobian gives the entry for joint `j`:

```
g_j(q) = − Σᵢ [  mᵢ · (gᵥ · ∂pᵢ/∂q_j)  +  (gᵥ × ω_j) · (Rᵢ φᵢ)  ]
                └──── multiplies mᵢ ────┘  └──── multiplies φᵢ ────┘
```

- `∂pᵢ/∂q_j` — translational Jacobian of the **link-frame origin** (`mj_jacBody`),
- `ω_j` — world axis of ancestor joint `j` (rotational Jacobian),
- `Rᵢ` — link orientation (`xmat`).

So **`g(q)` is linear in the 4 numbers `(mᵢ, φᵢ)` per link**. Assembling these
coefficients into `Y_g(q)` gives `g(q) = Y_g(q) · θ`, with
`θ = [m₁, φ₁, …, mₙ, φₙ]`. This is implemented in
`regressor.gravity_regressor_single`.

---

## 5. Identifiability — why the torque fit can be perfect yet the masses wrong

Not every parameter influences the torque, and some only appear in fixed *linear
combinations*. The identifiable combinations are called **base parameters**
(Gautier & Khalil). Two consequences you will see in the experiment:

1. **Gravity (static) data identifies first moments `φᵢ`, not masses.**
   A link's mass `mᵢ` multiplies `∂pᵢ/∂q_j` — the translation of its own frame
   origin. For the most proximal link that origin does not translate when its own
   joint rotates, so `mᵢ` drops out of the gravity torque. Only `φᵢ = mᵢ cᵢ`
   survives. **This is exactly enough for gravity compensation**, which is why
   the static method is the right tool for that job.

2. **Separating `mᵢ` from `cᵢ` requires acceleration.**
   Mass enters the *inertial* torque `M(q) q̈`. Only when the trajectory has
   significant `q̈` does `mᵢ` become observable on its own. Slow motions →
   small `q̈` → `mᵢ` weakly constrained → the optimizer leaves it near its
   initial value while still fitting the torque (via `φ`). The fix is **rich,
   fast excitation** (high-frequency Fourier trajectories) and/or a **CAD prior**
   on the directions the data cannot see.

Diagnosing identifiability:
- The **singular value spectrum** of `Y_stack` (or `Y_g`) reveals it: tiny/zero
  singular values ⇒ unobservable parameter directions. A large **condition
  number** warns that the solution is sensitive to noise.
- We handle weak directions with **Tikhonov (ridge) regularization** toward the
  CAD prior: `minₓ ‖Y x − τ‖² + λ‖x − x_prior‖²`. Observable directions follow
  the data; unobservable ones fall back to CAD.

---

## 6. The two estimators in this repo

**Method B — analytic linear least squares** (`identify_gravity_linear`)
Builds `Y_g` in closed form, solves the (ridge) normal equations. Fast, exact,
and the singular values quantify identifiability. Best for **gravity
compensation** and for *understanding* the problem.

**Method A — MuJoCo-in-the-loop nonlinear least squares** (`identify_nonlinear`)
Writes candidate `(mᵢ, cᵢ)` into a MuJoCo model and minimizes the full
inverse-dynamics residual `τ_pred(x) − τ_meas` with `scipy.least_squares`,
under box bounds (`mᵢ > 0`, `|cᵢ| < bound`) and a CAD prior. It uses
velocity/acceleration data, so it can recover mass and COM **separately** when
the excitation is rich enough. Best for **full inverse dynamics**.

---

## 7. Extension to the full 10 parameters

To also identify the rotational inertia `Iᵢ`, augment the parameter vector to the
full `πᵢ ∈ ℝ¹⁰` and impose **physical consistency** (positive mass, COM inside
the convex hull, positive-definite inertia satisfying the triangle inequalities).
A standard way is the **LMI / log-Cholesky parameterization** of the
pseudo-inertia matrix (Wensing et al., 2018), which keeps every iterate physically
realizable. The regressor and optimization scaffolding here extend directly; only
the parameter packing and the consistency constraints change.

---

## References

- W. Khalil, E. Dombre, *Modeling, Identification and Control of Robots*, 2004.
- M. Gautier, W. Khalil, "Direct calculation of minimum set of inertial
  parameters of serial robots," *IEEE T-RA*, 1990.
- J. Hollerbach, W. Khalil, M. Gautier, "Model Identification," *Springer
  Handbook of Robotics*, 2016.
- P. Wensing, S. Kim, J.-J. Slotine, "Linear Matrix Inequalities for Physically
  Consistent Inertial Parameter Identification," *IEEE RA-L*, 2018.
