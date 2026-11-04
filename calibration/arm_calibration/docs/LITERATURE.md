# Literature brief — inertial-parameter identification for gravity compensation and compliance on a 7-DOF QDD arm

**Scope.** Lagrangian / linear-in-parameters identification of link inertial parameters, specialised to
MABEL's right arm: 7 revolute DoF, DaMiao quasi-direct-drive actuators, **joint torque inferred from
motor current** (no joint torque sensors), **significant Coulomb friction**, and an operating constraint of
**slow, safe, one-joint-at-a-time motion**. Primary goal: gravity compensation. Secondary (later):
compliance / impedance control, which needs the same `θ` plus a friction model.

Companion documents: `docs/THEORY.md` (derivation as implemented), `docs/ARM_CALIBRATION.pdf` (results of
record), `../motor_calibration/docs/ACTUATOR_CALIBRATION.pdf` (friction / damping / armature per joint).

---

## 1. The canonical linear model

For an open chain with `q ∈ ℝⁿ`, the Euler–Lagrange equations

    M(q) q̈ + C(q,q̇) q̇ + g(q) = τ

are **nonlinear in `q` but exactly linear in the inertial parameters**. Each link `i` carries 10 numbers

    θ_i = [ m_i , h_i = m_i c_i ∈ ℝ³ , I_i = (Ixx,Ixy,Ixz,Iyy,Iyz,Izz) ] ∈ ℝ¹⁰ ,   θ = [θ_1 … θ_n] ∈ ℝ^{10n}

(`h_i` is the *first moment of mass*; the inertia is taken about the **link frame origin**, not the CoM —
that is what makes the map linear). Hence

    τ = Y(q, q̇, q̈) · θ                                                             (1)

with `Y` the **regressor**, computable from measured motion alone. Stacking `N` samples gives an ordinary
least-squares problem. This formulation and its first experimental use (PUMA 600, MIT Serial Link Direct
Drive Arm) are **Atkeson, An & Hollerbach (1986)** [1]; the textbook treatment — Newton–Euler and energy
regressors, DH-frame parameter tables — is **Khalil & Dombre (2004)** [2] and **Hollerbach, Khalil & Gautier
(2016)** [3]. A modern survey placing this against learned models is **Lee, Kwon, Wensing & Park (2024)** [4].

In practice `τ` in (1) must include the actuator terms the rigid-body model omits:

    τ_meas = Y(q,q̇,q̈) θ + I_a q̈ + F_v q̇ + F_c sign(q̇) + τ_off                    (2)

with `I_a` the reflected rotor inertia (MuJoCo `armature`), `F_v` viscous, `F_c` Coulomb (`frictionloss`),
`τ_off` a constant per-joint offset (current-sensor bias, cable routing). These columns are appended to
`Y`; they are linear too. MABEL's `motor_calibration/` package identifies exactly `(F_c, F_v, I_a)` per motor
type — the arm package deliberately zeroes them so its regression stays linear.

## 2. Base (minimal) parameters — and what survives in the gravity-only case

Not all 10n parameters affect `τ`; only linear combinations, the **base parameters**, do. Closed-form
regrouping rules per DH joint: **Gautier & Khalil (1990)** [5]. Numerical determination (the right way in
code): **Gautier (1991)** [6] — sample `W = [Y(q₁);…;Y(q_N)]` at many random configurations, **column-scale**
it, take a rank-revealing **QR with column pivoting** `W Π = Q [R₁ R₂; 0 0]`; the rank `r` (from the SVD
spectrum with a tolerance) is the number of base parameters, the first `r` pivoted columns are an
independent set, and

    β = R₁⁻¹ R₂                                                                     (3)

gives the exact regrouping coefficients `θ_B = θ₁ + β θ₂`. Everything in `θ₂` is structurally invisible.

### 2.1 Gravity-only (static) identifiability — the result that matters here

With `q̇ = q̈ = 0` only the potential matters:

    U(q) = − gᵥᵀ Σ_i [ m_i p_i(q) + R_i(q) h_i ] ,      g(q) = ∂U/∂q                (4)

so the static torque is linear in `(m_i, h_i)` — **4 numbers per link, the inertia tensor `I_i` never
appears.** Now the structure. For a revolute joint `i` with unit axis `a_i` and origin `o_i`, both *constant
in the parent frame*, the pose of link `i` is `p_i = p_p + R_p ℓ_i` (ℓ_i constant!) and
`R_i = R_p R_i⁰ Rot(e_z, q_i)` with `a_i = R_i⁰ e_z`. Split the first moment into axial + perpendicular parts
in the link frame, `h_i = h_i^z e_z + h_i^⊥`. Then

    m_i p_i + R_i h_i  =  m_i p_p + R_p [ m_i ℓ_i + h_i^z a_i ]  +  R_p R_i⁰ Rot(e_z,q_i) h_i^⊥
                          └──────── q_i-INDEPENDENT ────────┘      └── the only q_i-dependent part ──┘

**Therefore, from gravity data alone:**

* **`m_i` is NOT identifiable, and neither is `h_i^z = a_iᵀ h_i` (the first moment along the joint's own
  axis).** Mass placed *on* a revolute joint's axis does not move when that joint turns, so it is
  dynamically indistinguishable from mass belonging to the **parent** link. They regroup upward.
* **What IS identifiable is `β_i = (I − a_i a_iᵀ) h̄_i` — 2 numbers per revolute joint.** A fixed-base
  `n`-R chain therefore has **exactly `2n` gravity base parameters** (generic geometry), *not* `4n`.

**Regrouping recursion** (frame-free, leaves → root; expresses each child's invisible content in its parent's
frame). Let `ℓ_c` = origin of joint `c` in its parent's frame, `a_c` = axis of joint `c`:

    m̄_i = m_i + Σ_{c ∈ children(i)} m̄_c                                             (5a)
    h̄_i = h_i + Σ_{c ∈ children(i)} [ m̄_c ℓ_c + (a_cᵀ h̄_c) a_c ]                    (5b)
    identifiable:  β_i = (I − a_i a_iᵀ) h̄_i    ∈ ℝ²                                  (5c)

For a **prismatic** joint (MABEL's torso lift) the situation inverts: `R_i` does not rotate with `q_i`, so the
whole `h_i` regroups into the parent, while the joint force `f_i = −(gᵥᵀ R_p a_i) · m̄_i` **does** reveal the
**subtree mass** `m̄_i` — 1 identifiable parameter. Degenerate case worth checking on hardware: if a joint's
axis is *permanently parallel to gravity* (a base yaw on a level base), rotating it changes no height, so
`g_i(q) ≡ 0` **and** that link's `β_i` is unobservable — 2 more parameters lost and joint 1 needs no gravity
compensation at all. MABEL's shoulder is torso-mounted and tilted, so this should not bite, but assert it
numerically (rank of `W`), don't assume it.

**Consequence:** gravity compensation needs only `{β_i}` (2n numbers) — and *those are exactly the ones static
data gives you*. The static experiment is not a compromise for gravity comp; it is the matched tool.

## 3. Excitation design

**Dynamic (classical).** **Swevers et al. (1997)** [7] parameterise the excitation as periodic finite Fourier
series (which allows time-domain averaging over periods to kill noise — extremely valuable) and optimise the
coefficients under joint/Cartesian constraints; **Gautier & Khalil (1992)** [8] minimise `cond(W)` of the
scaled regressor; **Park (2006)** [9] uses Fourier+polynomial with a **D-optimality** objective;
**Calafiore, Indri & Bona (2001)** [10] compare `cond(W)` vs `−log det(WᵀW)` directly. The alphabet:

    D-optimal:  max log det(Wᵀ Σ⁻¹ W)      (minimises volume of the parameter confidence ellipsoid)
    A-optimal:  min trace((Wᵀ W)⁻¹)        (minimises mean parameter variance)
    E-optimal:  max σ_min(W)               (minimises the worst-direction variance)
    conditioning: min cond(W) = σ_max/σ_min (a *ratio*; scale-free but ignores absolute magnitude)

**Sun & Hollerbach (2008)** [11] show these "observability indices" bound one another and become equivalent
under perfect column scaling; **D-optimality is the right choice when the goal is parameter variance**.
**Column-scale `W` first** (Gautier [6]) — otherwise the criterion is dominated by units (kg vs kg·m).

**What changes when motion must be slow, safe, and one-joint-at-a-time.** Velocity and acceleration columns
collapse: `M(q)q̈` and `C q̇` contribute nothing, so **the excitation problem degenerates from trajectory
design to *pose* design**. The right formulation is classical **optimal design of experiments over a discrete
candidate set** — pick `K` static poses `P ⊂ Q_feasible` maximising

    Φ_D(P) = log det ( Σ_{q ∈ P} Y_g(q)ᵀ W_r Y_g(q) )      (Y_g column-scaled, W_r = row weights, §7)   (6)

Solve it with a **greedy / Fedorov–Wynn exchange (DETMAX)** over a pool of ~10⁴ random collision-free,
limit-respecting poses — the same machinery Sun & Hollerbach [11] use for kinematic-calibration pose
selection, and the tradition of **Fedorov (1972)** [12] / **Pukelsheim (1993)** [13]. *There is no
dedicated paper on D-optimal static-pose selection for gravity-only robot identification* — it is the obvious
composition of [6,11,12] and you should just implement (6). It is cheap: `Y_g(q)` is one `mj_forward` +
Jacobian call per pose (already in `regressor.gravity_regressor_single`).

Crucially, **one-joint-at-a-time is not a restriction on the pose set** — only on *how you travel between
poses*. You may still realise an arbitrary D-optimal pose set by sequencing single-joint moves. If you truly
must restrict data to single-joint sweeps, use **anchors**: pick `K_a ≈ 5–8` diverse anchor configurations and
sweep each joint through its range from each anchor; a single-joint sweep from one home pose traces a 1-D
manifold and leaves `W` badly rank-deficient across links.

## 4. Physical consistency (needed only when you go beyond gravity)

Unconstrained LS can return negative masses or non-realisable inertias — fatal for simulation and for
model-based impedance control. Three levels:

* **Traversaro, Brossette, Escande & Nori (2016)** [14]: *full physical consistency* = positive mass +
  positive-definite inertia + **triangle inequalities** on the principal moments; enforced by a
  parameterisation that makes the constraint manifold explicit (Riemannian / manifold optimisation).
* **Wensing, Kim & Slotine (2018)** [15]: the same conditions are **an LMI** on the 4×4 **pseudo-inertia**

      J(θ) = [ Σ_C     h ]  ⪰ 0 ,     Σ_C = ½ tr(I) 1₃ − I  (density-weighted covariance)          (7)
             [ hᵀ      m ]

  `J ≻ 0` implies positive mass, CoM well-defined, `I ≻ 0`, *and* the triangle inequalities — one clean SDP
  constraint per link. This is the formulation to implement.
* **Sousa & Cortesão (2014)** [16] (IJRR): consistency must be imposed on the **standard** parameters while
  only the **base** parameters are observable. Feasible standard parameters form a convex set; parameterise
  `θ = θ_LS + Z λ` with `Z` a basis of `null(W)` (from the QR of §2) and solve

      min_{λ, (θ)}  ‖ W θ − τ ‖²_{Σ⁻¹}   s.t.  J(θ_i) ⪰ 0  ∀i   [+ box bounds on m_i, c_i]        (8)

  as an SDP. Their WAM-7 code/data is public [16b]. **Gaz et al. (2019)** [17] do the Panda equivalent with a
  penalty-based nonlinear program plus a CAD prior.
* **Solvers:** `cvxpy` with **SCS** (free, fine for 7 links) or **MOSEK** (academic licence, faster/more
  accurate); `J ⪰ 0` is a `cp.PSD` constraint; `−log det` is `cp.log_det`. Fallback: LDL/log-Cholesky
  parameterisation + `scipy.optimize.least_squares` (unconstrained by construction), which is what
  `identify_nonlinear` already resembles.

**For gravity comp only, none of this is required** — you identify `β_i` (2n numbers), which have no
consistency constraint of their own beyond box bounds. Consistency becomes essential the moment you want
`M(q)` for impedance control or an updated MJCF `<inertial>`.

## 5. CAD priors and regularisation — the right tool for weak data

With slow, safe data the regressor is *deliberately* rank-deficient. You must decide what happens in the
null space. Options, in increasing sophistication:

1. **Tikhonov / ridge toward CAD**: `min ‖Wθ − τ‖² + λ‖Λ(θ − θ_CAD)‖²`. Closed-form, and the null-space
   directions fall back exactly to CAD. `Λ` must be a *scaling* matrix (mass in kg vs first moment in kg·m)
   or `λ` is meaningless. Choose `λ` by held-out cross-validation (§7), not by taste. **This is what the repo
   already does** and it is the correct default.
2. **Bayesian / MAP**: identical algebra with `λΛᵀΛ = σ² Σ_prior⁻¹`, but now `Σ_prior` encodes *how much you
   trust CAD per parameter* (mass ±5 %, CoM ±10 mm ⇒ very different weights). **Ting, Mistry, Peters, Schaal
   & Nakanishi (2006)** [18] give the Bayesian treatment with noise in both `W` and `τ`.
3. **Geometric / entropic regularisation**: **Lee, Wensing & Park (2020)** [19] observe that inertial
   parameters live on a curved (Riemannian) space, so a Euclidean ridge is the wrong metric; they use the
   **log-det Bregman (entropic) divergence** between pseudo-inertias

       D(J ‖ J₀) = tr(J₀⁻¹ J) − log det(J₀⁻¹ J) − 4                                      (9)

   which is **convex in `θ`**, is `+∞` outside the physically-consistent cone (so it *implies* consistency),
   and is invariant to the choice of link frame. Their whole identification is then a single convex program.
   **With limited, poorly-exciting data this is the best-behaved regulariser** — it pulls unobserved
   directions toward CAD *along the geometry of the parameter manifold* instead of along an arbitrary
   Euclidean straight line, and never leaves the feasible cone. Recommendation: **ridge-to-CAD now (gravity),
   entropic (9) when you extend to the full 10-parameter model.**

## 6. Friction — the thing that will actually corrupt your gravity identification

QDD + current-sensed torque means `τ̂ = k_τ · i · N · η`, and the rigid-body torque you want is buried under

    τ_f(q̇) = F_c sign(q̇) + F_v q̇ + (F_s − F_c) exp(−|q̇/v_s|^δ) sign(q̇)              (10)

(Coulomb + viscous + Stribeck; survey: **Armstrong-Hélouvry, Dupont & Canudas de Wit (1994)** [20];
robot-joint specifics incl. load/temperature dependence: **Bittencourt & Gunnarsson (2012)** [21];
identification recipes: **Kermani, Patel & Moallem (2007)** [22]). At a *static* pose the sign of `F_c` is
**undetermined** — the joint sits anywhere inside a **stiction band** of width `2F_s`, and the measured
current tells you `g(q) + τ_stick` with `τ_stick ∈ [−F_s, +F_s]`. Naively holding poses therefore injects a
**bias, not noise**, into every row of the regression.

**The bidirectional-averaging trick (do this).** Approach each pose `q*` twice — once moving in `+` on the
joint(s) of interest, once in `−` — creeping in at a small constant velocity (or settling under a PD servo
after a `±` approach). Then

    τ̄(q*) = ½ ( τ⁺ + τ⁻ ) = g(q*) + τ_off + O(F_v v)      ← Coulomb/stiction cancels    (11)
    Δ(q*)  = ½ ( τ⁺ − τ⁻ ) = F_c + (F_s − F_c)·(Stribeck)  ← free per-pose friction estimate

Equation (11) is the standard hysteresis-cancelling measurement: it is exactly why Coulomb friction is
sign-antisymmetric and gravity is not. Use `τ̄` in the regression and keep `Δ` as a **pose-dependent friction
map** (it is load- and temperature-dependent [21]) — that map is precisely what compliance control will need
later. **Liu & Quach (2001)** [23] do gravity + static-friction estimation from steady-state equations at
static poses under PD set-point control; the recent dVRK-Si gravity-compensation work of **Zhou et al.
(2025/2026)** [24,25] is the most directly analogous modern pipeline (a torque-sensorless, cable/gear-driven
arm, gravity comp validated by static-hold drift), and combines static-pose collection with an
optimally-conditioned excitation trajectory and convex, physically-feasible identification.

Additional corruptions, and what to do:

* **Dither.** Superimposed low-amplitude high-frequency torque (e.g. 20–50 Hz, ~0.3·F_c) linearises the
  friction through the origin and removes the dead zone [20, §compensation]. Useful *at deployment* (to make
  the arm actually float), and as an alternative to (11) *during collection*. It does inject motion — for a
  "slow and safe" constraint, prefer (11) for identification and reserve dither for the demo.
* **Gear efficiency asymmetry** (driving vs back-driving `η`). Multiplicative and sign-of-power-flow
  dependent; at strict statics power flow ≈ 0 so it is small, and the `±` averaging cancels the
  first-order term. DaMiao's low-ratio planetary (η ≈ 0.9) makes this second-order — but it is one more
  reason **not** to identify from one-directional creep.
* **Torque-constant error.** A **common** multiplicative error `α` on all joints is *exactly degenerate*:
  `τ = α·Y θ` ⇒ you identify `α·θ`. Harmless for gravity comp (you command in the same units you measured),
  fatal if you paste `θ̂` into the MJCF. **Per-joint** gain errors are not degenerate (parameters are shared
  across rows) and can be co-estimated by alternating LS — but this is only worth doing with a hanging-mass
  torque calibration to break the global scale.
* **Temperature drift** [21]: warm the joints up, **randomise pose order**, and log motor temperature; a
  drifting `F_c` is a slow drift in the *residual*, easy to spot in a residual-vs-time plot.

## 7. Validation methodology

1. **Weighted LS per joint.** The rows of `W` for different joints have wildly different noise. Estimate
   `σ_j²` from a first unweighted pass, then re-solve with row weights `1/σ_j` (Gautier's weighted LS [6,3]).
2. **Parameter covariance and the %σ criterion.**

       σ̂² = ‖τ − W θ̂‖² / (N_rows − r) ,   Cov(θ̂) = σ̂² (Wᵀ W)⁻¹ ,   %σ_i = 100 · √Cov_ii / |θ̂_i|   (12)

   **Pham & Gautier (1991)** [26] call the well-determined subset the **essential parameters**: drop `θ_i`
   whose relative standard deviation is large — practical rule: `%σ_i > 10 × min_j %σ_j`, or an absolute
   `%σ_i > 20–30 %`. Refit on the reduced set. Note (12) assumes white noise — with (11) the dominant error
   is *not* white, so treat `%σ` as a **ranking** tool, not an absolute confidence.
3. **Held-out cross-validation.** Split the pose set (e.g. 80/20, or k-fold) and report **torque-prediction
   RMS on unseen poses**, per joint and normalised: `RMS_j / max_q |g_j(q)|`. This is the number that matters
   and the only honest way to pick `λ` in §5. A second, independent *validation trajectory* (slow sweep) is
   the standard practice in [7,8,17].
4. **Physical plausibility check**: recovered `c_i = h_i/m_i` inside the link's convex hull; `m̄_i` (subtree
   masses) within a few % of CAD; pseudo-inertia (7) PSD.
5. **The end-to-end test that actually counts**: hold the gravity-compensated arm at held-out poses and
   measure **drift / steady-state error over 30 s** and the **force needed to move it by hand** ([24,25] use
   exactly this).

---

## Recommended pipeline for MABEL (slow, safe, one-joint-at-a-time; gravity first)

1. **Model the identifiable set, not the parameters.** Build the gravity regressor `Y_g(q)` (already in
   `regressor.py`), then compute the **gravity base set numerically** (QR + column pivoting, §2) — expect
   `rank ≈ 2·7 = 14` for the arm, plus 7 offset columns. Assert the rank; it is a bug if it is not what (5c)
   predicts.
2. **Design ~40–80 static poses by D-optimality** (6) via greedy exchange over a pool of random
   joint-limit-respecting, self-collision-free poses (check collisions in MuJoCo). If you must sweep single
   joints, use 5–8 anchors × 7 joints × ~5 angles.
3. **Collect with bidirectional averaging** (11): reach each pose from `+` and from `−` (creep at
   ~0.02–0.05 rad/s, then stop and settle ≥ 1 s), record ~2 s of current at each, low-pass filter (zero-phase
   `filtfilt`), average. Save `τ⁺, τ⁻, τ̄, Δ`, pose, joint temperature, timestamp. Randomise pose order.
   Repeat the whole set twice for a repeatability estimate.
4. **Identify** with weighted ridge-to-CAD least squares on the base set, `λ` chosen by held-out CV.
   Report `%σ` per base parameter; drop the non-essential ones and refit.
5. **Validate** per §7 (held-out RMS, drift test, plausibility).
6. **Deploy**: `τ_cmd = ĝ(q) + τ̂_off` (+ optional `F̂_c·sign(q̇_filtered)` friction feedforward, or dither, to
   make the arm truly floatable).
7. **Later, for compliance**: extend to the full 10-parameter model with slow-but-not-static excitation,
   solve the **SDP** (8) with pseudo-inertia LMIs (7) and the **entropic regulariser** (9) toward CAD, and
   fold in the friction/armature numbers from `motor_calibration/`. Paste `<inertial>` +
   `frictionloss/damping/armature` into `simulation/mabel_mujoco/models/mabel_full.xml` (the canonical model) and regenerate the
   derived artifacts as described in the top-level README.

### What is IMPOSSIBLE under these constraints — state it plainly

* **No inertia tensors.** `I_i` never enters `g(q)` (4). Zero rows of information. Static data cannot produce
  a `<inertial>` block's `diaginertia`; keep CAD.
* **No individual link masses.** From gravity alone, `m_i` is only ever observed inside the **subtree mass**
  `m̄_i` of a *prismatic* joint above it (5a). For a pure revolute chain with a fixed base, **no link mass is
  separately identifiable**. Any "identified mass" you print from static data is the prior, not a measurement.
  (MABEL's torso lift *is* prismatic, so the arm's **total mass** is identifiable through the lift.)
* **No first moment along a joint's own axis** (`h_i^z`): it is dynamically identical to mass on the parent.
* **No armature / reflected inertia, no viscous friction** worth the name: both need acceleration/velocity.
  Get them from `motor_calibration/`, not here.
* **Global torque-constant scale `α`** is unidentifiable from any amount of self-measured data. Gravity comp
  is unaffected (self-consistent units); the MJCF is affected. Break it with a hanging-mass test if you care.
* **Coulomb friction sign at rest** is unidentifiable per-measurement — only the *band* is. Hence (11).

### Achievable gravity-compensation accuracy (expectation, to be confirmed by `experiments/validate.py`)

The floor is set by friction, not by the fit. Once `|ĝ(q) − g(q)|` drops below the joint's **breakaway
torque**, the joint cannot move in response to the residual and further parameter accuracy is invisible at
zero velocity. With this repo's measured DaMiao numbers (`DM-J4340P`: `frictionloss ≈ 0.43 N·m`;
`DM-J4310`: `≈ 0.083 N·m`), the useful target is:

* **held-out static torque-prediction RMS ≈ 2–5 % of each joint's peak gravity torque** — for a shoulder
  joint with ~15–25 N·m peak that is ≈ **0.3–1.0 N·m**; for wrist joints, ≈ 0.05–0.1 N·m;
* **static-hold drift** ≈ 0 by construction while the residual stays inside the stiction band;
* **true "zero-G float"** (arm moves under a light hand push, from any pose) is **not** achievable from
  gravity compensation alone on these actuators — it additionally requires friction feedforward
  (`F̂_c sign(q̇)`) and/or dither. Document it rather than claiming a floating arm.

Bidirectional averaging is the single highest-leverage step: without it the Coulomb band enters as a *bias*
of order `F_c` (0.4 N·m on the big joints), which is the same size as the entire error budget above.

---

## References (BibTeX-ready)

```bibtex
@article{atkeson1986estimation,
  author  = {Atkeson, Christopher G. and An, Chae H. and Hollerbach, John M.},
  title   = {Estimation of Inertial Parameters of Manipulator Loads and Links},
  journal = {The International Journal of Robotics Research},
  volume  = {5}, number = {3}, pages = {101--119}, year = {1986},
  doi     = {10.1177/027836498600500306},
  url     = {https://journals.sagepub.com/doi/10.1177/027836498600500306}
}                                                                                   % [1]

@book{khalil2004modeling,
  author    = {Khalil, Wisama and Dombre, Etienne},
  title     = {Modeling, Identification and Control of Robots},
  publisher = {Butterworth-Heinemann / Kogan Page Science},
  year      = {2004}, isbn = {978-1-903996-66-9}
}                                                                                   % [2]

@incollection{hollerbach2016model,
  author    = {Hollerbach, John and Khalil, Wisama and Gautier, Maxime},
  title     = {Model Identification},
  booktitle = {Springer Handbook of Robotics},
  editor    = {Siciliano, Bruno and Khatib, Oussama},
  edition   = {2nd}, publisher = {Springer}, year = {2016}, pages = {113--138},
  doi       = {10.1007/978-3-319-32552-1_6}
}                                                                                   % [3]

@article{lee2024robot,
  author  = {Lee, Taeyoon and Kwon, Jaewoon and Wensing, Patrick M. and Park, Frank C.},
  title   = {Robot Model Identification and Learning: A Modern Perspective},
  journal = {Annual Review of Control, Robotics, and Autonomous Systems},
  volume  = {7}, pages = {311--334}, year = {2024},
  doi     = {10.1146/annurev-control-061523-102310},
  url     = {https://www.annualreviews.org/content/journals/10.1146/annurev-control-061523-102310}
}                                                                                   % [4]

@article{gautier1990direct,
  author  = {Gautier, Maxime and Khalil, Wisama},
  title   = {Direct Calculation of Minimum Set of Inertial Parameters of Serial Robots},
  journal = {IEEE Transactions on Robotics and Automation},
  volume  = {6}, number = {3}, pages = {368--373}, year = {1990},
  doi     = {10.1109/70.56655}
}                                                                                   % [5]

@article{gautier1991numerical,
  author  = {Gautier, Maxime},
  title   = {Numerical Calculation of the Base Inertial Parameters of Robots},
  journal = {Journal of Robotic Systems},
  volume  = {8}, number = {4}, pages = {485--506}, year = {1991},
  doi     = {10.1002/rob.4620080405},
  note    = {Also in Proc. IEEE ICRA 1990, pp. 1020--1025, doi:10.1109/ROBOT.1990.126126}
}                                                                                   % [6]

@article{swevers1997optimal,
  author  = {Swevers, Jan and Ganseman, Chris and Tukel, D. Bilgin and De Schutter, Joris and Van Brussel, Hendrik},
  title   = {Optimal Robot Excitation and Identification},
  journal = {IEEE Transactions on Robotics and Automation},
  volume  = {13}, number = {5}, pages = {730--740}, year = {1997},
  doi     = {10.1109/70.631234}
}                                                                                   % [7]

@article{gautier1992exciting,
  author  = {Gautier, Maxime and Khalil, Wisama},
  title   = {Exciting Trajectories for the Identification of Base Inertial Parameters of Robots},
  journal = {The International Journal of Robotics Research},
  volume  = {11}, number = {4}, pages = {362--375}, year = {1992},
  doi     = {10.1177/027836499201100408}
}                                                                                   % [8]

@article{park2006fourier,
  author  = {Park, Kyung-Jin},
  title   = {Fourier-Based Optimal Excitation Trajectories for the Dynamic Identification of Robots},
  journal = {Robotica},
  volume  = {24}, number = {5}, pages = {625--633}, year = {2006},
  doi     = {10.1017/S0263574706002712}
}                                                                                   % [9]

@article{calafiore2001robot,
  author  = {Calafiore, Giuseppe and Indri, Marina and Bona, Basilio},
  title   = {Robot Dynamic Calibration: Optimal Excitation Trajectories and Experimental Parameter Estimation},
  journal = {Journal of Robotic Systems},
  volume  = {18}, number = {2}, pages = {55--68}, year = {2001},
  doi     = {10.1002/1097-4563(200102)18:2<55::AID-ROB1005>3.0.CO;2-O}
}                                                                                   % [10]

@inproceedings{sun2008observability,
  author    = {Sun, Yu and Hollerbach, John M.},
  title     = {Observability Index Selection for Robot Calibration},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {831--836}, year = {2008},
  doi       = {10.1109/ROBOT.2008.4543308},
  url       = {https://cse.usf.edu/~yusun/lab_pub/icra0801.pdf}
}                                                                                   % [11]

@book{fedorov1972theory,
  author    = {Fedorov, Valerii V.},
  title     = {Theory of Optimal Experiments},
  publisher = {Academic Press}, year = {1972}
}                                                                                   % [12]

@book{pukelsheim1993optimal,
  author    = {Pukelsheim, Friedrich},
  title     = {Optimal Design of Experiments},
  publisher = {Wiley}, year = {1993}, isbn = {978-0-471-61971-0}
}                                                                                   % [13]

@inproceedings{traversaro2016identification,
  author    = {Traversaro, Silvio and Brossette, Stanislas and Escande, Adrien and Nori, Francesco},
  title     = {Identification of Fully Physical Consistent Inertial Parameters Using Optimization on Manifolds},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {5446--5451}, year = {2016},
  doi       = {10.1109/IROS.2016.7759801},
  url       = {https://arxiv.org/abs/1610.08703}
}                                                                                   % [14]

@article{wensing2018linear,
  author  = {Wensing, Patrick M. and Kim, Sangbae and Slotine, Jean-Jacques E.},
  title   = {Linear Matrix Inequalities for Physically Consistent Inertial Parameter Identification:
             A Statistical Perspective on the Mass Distribution},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {3}, number = {1}, pages = {60--67}, year = {2018},
  doi     = {10.1109/LRA.2017.2729659},
  url     = {https://arxiv.org/abs/1701.04395}
}                                                                                   % [15]

@article{sousa2014physical,
  author  = {Sousa, Crist{\'o}v{\~a}o D. and Cortes{\~a}o, Rui},
  title   = {Physical Feasibility of Robot Base Inertial Parameter Identification:
             A Linear Matrix Inequality Approach},
  journal = {The International Journal of Robotics Research},
  volume  = {33}, number = {6}, pages = {931--944}, year = {2014},
  doi     = {10.1177/0278364913514870},
  note    = {Code + WAM-7 data: https://github.com/cdsousa/wam7_dyn_ident}
}                                                                                   % [16, 16b]

@article{gaz2019dynamic,
  author  = {Gaz, Claudio and Cognetti, Marco and Oliva, Alexander and Robuffo Giordano, Paolo and De Luca, Alessandro},
  title   = {Dynamic Identification of the Franka Emika Panda Robot with Retrieval of Feasible Parameters
             Using Penalty-Based Optimization},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {4}, number = {4}, pages = {4147--4154}, year = {2019},
  doi     = {10.1109/LRA.2019.2931248},
  url     = {https://hal.science/hal-02265294}
}                                                                                   % [17]

@inproceedings{ting2006bayesian,
  author    = {Ting, Jo-Anne and Mistry, Michael and Peters, Jan and Schaal, Stefan and Nakanishi, Jun},
  title     = {A {B}ayesian Approach to Nonlinear Parameter Identification for Rigid Body Dynamics},
  booktitle = {Robotics: Science and Systems (RSS)}, year = {2006},
  doi       = {10.15607/RSS.2006.II.032},
  url       = {https://www.roboticsproceedings.org/rss02/p32.pdf}
}                                                                                   % [18]

@article{lee2020geometric,
  author  = {Lee, Taeyoon and Wensing, Patrick M. and Park, Frank C.},
  title   = {Geometric Robot Dynamic Identification: A Convex Programming Approach},
  journal = {IEEE Transactions on Robotics},
  volume  = {36}, number = {2}, pages = {348--365}, year = {2020},
  doi     = {10.1109/TRO.2019.2926491},
  note    = {Code: https://github.com/alex07143/Geometric-Robot-DynID}
}                                                                                   % [19]

@article{armstrong1994survey,
  author  = {Armstrong-H{\'e}louvry, Brian and Dupont, Pierre and Canudas de Wit, Carlos},
  title   = {A Survey of Models, Analysis Tools and Compensation Methods for the Control of Machines with Friction},
  journal = {Automatica},
  volume  = {30}, number = {7}, pages = {1083--1138}, year = {1994},
  doi     = {10.1016/0005-1098(94)90209-7}
}                                                                                   % [20]

@article{bittencourt2012static,
  author  = {Bittencourt, Andr{\'e} C. and Gunnarsson, Svante},
  title   = {Static Friction in a Robot Joint---Modeling and Identification of Load and Temperature Effects},
  journal = {ASME Journal of Dynamic Systems, Measurement, and Control},
  volume  = {134}, number = {5}, pages = {051013}, year = {2012},
  doi     = {10.1115/1.4006589}
}                                                                                   % [21]

@article{kermani2007friction,
  author  = {Kermani, Mehrdad R. and Patel, Rajni V. and Moallem, Mehrdad},
  title   = {Friction Identification and Compensation in Robotic Manipulators},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  volume  = {56}, number = {6}, pages = {2346--2353}, year = {2007},
  doi     = {10.1109/TIM.2007.907957}
}                                                                                   % [22]

@article{liu2001estimation,
  author  = {Liu, Ming and Quach, Nghe H.},
  title   = {Estimation and Compensation of Gravity and Friction Forces for Robot Arms: Theory and Experiments},
  journal = {Journal of Intelligent and Robotic Systems},
  volume  = {31}, number = {4}, pages = {339--354}, year = {2001},
  doi     = {10.1023/A:1012089607929}
}                                                                                   % [23]

@inproceedings{zhou2025gravity,
  author    = {Zhou, Haoying and Yang, Hao and Deguet, Anton and Fichera, Loris and Wu, Jie Ying and Kazanzides, Peter},
  title     = {Gravity Compensation of the {dVRK-Si} Patient Side Manipulator Based on Dynamic Model Identification},
  booktitle = {Hamlyn Symposium on Medical Robotics (HSMR)}, year = {2025},
  doi       = {10.31256/HSMR25.46},
  url       = {https://arxiv.org/abs/2501.19058}
}                                                                                   % [24]

@article{zhou2026towards,
  author  = {Zhou, Haoying and Yang, Hao and Burkhart, Brendan and Deguet, Anton and Fichera, Loris
             and Fischer, Gregory S. and Wu, Jie Ying and Kazanzides, Peter},
  title   = {Towards Dynamic Model Identification and Gravity Compensation for the {dVRK-Si} Patient Side Manipulator},
  journal = {arXiv preprint arXiv:2603.12099}, year = {2026},
  url     = {https://arxiv.org/abs/2603.12099}
}                                                                                   % [25]

@inproceedings{pham1991essential,
  author    = {Pham, Minh Tu and Gautier, Maxime},
  title     = {Essential Parameters of Robots},
  booktitle = {Proc. 30th IEEE Conference on Decision and Control (CDC)},
  pages     = {2769--2774}, address = {Brighton, UK}, year = {1991},
  doi       = {10.1109/CDC.1991.261851}
}                                                                                   % [26]

@article{wensing2017proprioceptive,
  author  = {Wensing, Patrick M. and Wang, Albert and Seok, Sangok and Otten, David and Lang, Jeffrey and Kim, Sangbae},
  title   = {Proprioceptive Actuator Design in the {MIT} Cheetah: Impact Mitigation and High-Bandwidth
             Physical Interaction for Dynamic Legged Robots},
  journal = {IEEE Transactions on Robotics},
  volume  = {33}, number = {3}, pages = {509--522}, year = {2017},
  doi     = {10.1109/TRO.2016.2640183}
}                                                                                   % [27] QDD current-based torque estimation
```
