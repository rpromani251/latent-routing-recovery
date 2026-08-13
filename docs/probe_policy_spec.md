# Probe Policy: Geometry-Driven Scale Selection and Tangent-Frame Probing

**Status:** design specification, partially untested. Sections marked `[VALIDATED]` rest on E1–E4 results; `[PROPOSED]` items have a stated falsification test in §9 and have not been run.

**Scope:** this document specifies Stage 0 (local geometry) and Stage 1 (probe generation) of the routing audit, and the interfaces they expose to Stages 2–3. It replaces the hand-set three-scale ladder with a per-anchor, data-driven policy, and it introduces geometric abstention.

---

## 1. Where this fits

The audit at a single anchor `x` currently runs:

```
  probe (3 fixed scales)  ->  trimmed residualization  ->  dip test  ->  {flag, abstain}
```

This spec inserts a query-free geometry stage upstream and replaces the fixed ladder:

```
  STAGE 0   local geometry              [no queries to f]
            - GRIDE d-hat curve across neighbor ranks
            - plateau detection -> radius window [r_lo, r_hi]
            - local PCA -> tangent frame U (d-hat columns)
            - geometric abstention gate
                |
                v
  STAGE 1   probe generation            [m queries per rung]
            - ladder placed inside plateau
            - isotropic in tangent frame, fixed per-coordinate scale
            - on-manifold density filter
                |
                v
  STAGE 2   trimmed local-linear residualization    [VALIDATED: E1, E4]
            - LTS fit in tangent coordinates z, h = 0.75m
                |
                v
  STAGE 3   structure test
            - dip (nonparametric) + equal-variance mixture LRT (parametric)
            - bootstrap calibration through the full pipeline
            - spatial separability guard
                |
                v
  STAGE 4   report: {flag / abstain}, pi-hat, Delta-hat, boundary normal estimate
```

Two things to hold onto about the placement:

1. **Stage 0 costs zero model queries.** It reads the anchor covariate set only. This is what makes the adaptivity statistically free (§7).
2. **Stage 0 is an enhancement over a working baseline, not a prerequisite.** If it abstains or fails, the fallback is ambient isotropic probing at fixed per-coordinate scale plus trimmed residualization — which E1 established at power ≈ 0.75, flat in dimension. Nothing here is load-bearing for correctness; it is load-bearing for power, for reach, and for knowing when to shut up.

---

## 2. Notation and inputs

| Symbol | Meaning |
|---|---|
| `f` | black-box model under audit, queried as `y = f(x)` |
| `D` | ambient input dimension |
| `A` | anchor set: real covariate vectors, fixed **before** any query to `f` |
| `x` | current anchor, `x` in `A` |
| `d̂` | estimated intrinsic dimension at `x` |
| `U` | tangent frame, `D × d̂`, orthonormal columns |
| `z_i` | probe displacement in tangent coordinates, `z_i` in `R^d̂` |
| `δ_i` | ambient displacement, `δ_i = U z_i` |
| `σ_s` | per-coordinate probe scale at rung `s` |
| `m` | queries per rung (default 1000) |
| `π` | mixing fraction: share of probes landing across the gate |
| `Δ` | gate penalty magnitude |
| `τ` | observation noise sd (default 0.02) |

**Hard precondition.** `A` must be fixed independently of `f`. If anchors were selected using prior responses, the independence argument in §7 fails and the ladder selection must be Bonferroni-corrected as if hand-set.

---

## 3. Stage 0 — local geometry

### 3.1 Why an intrinsic-dimension estimator at all

The probe must satisfy three things at once, and they trade against each other:

- **Reach.** The ball must be large enough to actually cross the gate boundary. Under a fixed *total* budget the crossing rate is `Φ(−d√D/σ)`, which collapses in `D` — E1(a), verified to three decimals. So the per-coordinate scale must be held fixed, meaning the ball radius grows like `σ√d̂`, and the effective dimension in that expression should be `d̂`, not `D`. **Reducing the working dimension improves detectability itself, not merely power.**
- **Validity.** The ball must stay small enough that the honest surface is approximately linear over it (A11). Otherwise the plane fitted in Stage 2 is the wrong shape, residuals carry systematic curvature, and curvature can present as bimodality.
- **Relevance.** Probes should land where real inputs land, or the audit reports gates that exist in input space but never fire in deployment.

An intrinsic-dimension curve addresses all three from one computation, and does it without querying `f`.

### 3.2 Estimator choice

GRIDE (Denti, Doimo, Laio, Mira), the generalized-ratio successor to TWO-NN (Facco et al.). The construction: for a locally uniform sample on a `d`-manifold, the ratio `μ = r_{n₂} / r_{n₁}` of neighbor distances has a distribution depending only on `d`. **The unknown local density cancels in the ratio.** That is the whole trick, and it is why this family survives the clumpy covariate distributions that broke E4.

Reasons for GRIDE specifically over TWO-NN or the alternatives:

1. **It is natively scale-resolved.** TWO-NN returns one number at the smallest available scale. GRIDE returns `d̂(n₁)` — a curve. We need the curve, not the number.
2. **It decouples noise-averaging from scale.** With TWO-NN, reducing variance means using more points means larger radii means more curvature bias; the two are chained. GRIDE lets you raise `n₁, n₂` together to average over more neighbors at roughly fixed scale. Anchor neighborhoods here are small and noisy, so buying variance reduction without automatically buying bias matters.
3. **Density-robustness by construction**, as above.
4. **Fractional output is informative.** `d̂ = 4.7` is not a failed integer estimate; it typically means the sampled scale spans structure of more than one dimension. We round for the frame, but the fractional part calibrates confidence in the rounding.
5. **Query-free**, needs only pairwise distances on `A`.

Rejected alternatives: MLE (Levina–Bickel) underestimates at high `d`; correlation dimension requires eye-identifying a scaling region; PCA eigenvalue cutoffs reintroduce exactly the arbitrary threshold we are trying to remove. Note that we still run local PCA for the *frame* — GRIDE decides where to truncate it, replacing a scree-plot judgment with an estimate that has a stated generative model.

### 3.3 The three regimes of the d̂ curve

| Scale | Behavior | Cause |
|---|---|---|
| small | `d̂` inflated toward `D` | observation noise is full-dimensional; it kicks points off-manifold in all `D` directions |
| middle | `d̂` flat — **the plateau** | past noise, not yet feeling curvature; `d̂` ≈ true intrinsic dimension |
| large | `d̂` drifts (usually up) | neighborhood is bending, or has swallowed a second cluster; local uniformity fails |

**The central claim of this spec:** the plateau is defined by "the neighborhood looks like a flat, evenly-sampled patch," and *that is A11*. The radius range where `d̂` is stable is the radius range where local-linear fitting is honest. This is not a heuristic correlation — it is the same geometric condition read two different ways.

`[PROPOSED]` — this equivalence is the load-bearing assumption and V4 in §9 is its test.

### 3.4 Procedure

```
PROCEDURE Stage0(x, A, params):

  1. NEIGHBORS
     N <- k_max nearest neighbors of x in A          # k_max default 200
     if |N| < k_min or median-kNN-radius(x) > R_iso:
         return ABSTAIN(A0_isolated)

  2. d-HAT CURVE
     for n1 in ladder of neighbor ranks (n2 = 2*n1):
         d_hat[n1], CI[n1] <- GRIDE(N, n1, n2)       # profile-likelihood CI
         r[n1]             <- median distance to n1-th neighbor
                                                      # r[n1] is the characteristic
                                                      # radius at that rung

  3. PLATEAU DETECTION
     P <- longest contiguous run of ranks whose CIs have nonempty
          common intersection
     W <- log10( r[max P] / r[min P] )                # plateau width, decades
     if W < W_min:                                    # default W_min = 0.35
         return ABSTAIN(A1_no_plateau)
     d_hat  <- round( midpoint of the common CI intersection )
     r_lo   <- r[min P] ;  r_hi <- r[max P]

  4. DIMENSION SANITY
     if d_hat >= gamma * D:                           # default gamma = 0.75
         return FALLBACK(ambient)                     # no reduction available;
                                                      # use E1 baseline
  5. TANGENT FRAME
     U <- top d_hat principal directions of the neighbors within radius r_hi,
          centered at x                               # local PCA, weighted
     record lambda_ratio = lambda_{d_hat+1} / lambda_{d_hat}   # leakage diagnostic

  return {d_hat, U, r_lo, r_hi, W, lambda_ratio}
```

Notes:

- Plateau detection via **overlapping confidence intervals** is the formalization that turns "read the flat bit by eye" into a decidable rule. It also makes plateau *width* a number, which §5 uses for abstention.
- `lambda_ratio` near 1 means the `d̂+1`-th direction is nearly as strong as the `d̂`-th — the truncation is unstable. Log it; it is a leading indicator of frame error.

---

## 4. Stage 1 — probe generation

### 4.1 Placing the ladder

Two facts pull in opposite directions and both are validated:

- **Bigger is better for detection.** Larger radius raises the crossing rate and hence `π`, and `π` must be non-trivial for any test to have power. The classical objection — that a wide ball sweeps in more honest variation which masks the gap — is *exactly what trimmed residualization removes*. E4-uniform: power `0.14 → 0.90` at large scale purely by stripping the masking trend. E1: power flat in `D` after trimming.
- **Counting and estimating want different widths.** E2: `K`-selection wants a probe spanning multiple boundaries (wide); gap estimation wants a probe resolving one boundary (narrow). Median gap recovery was `0.228 / 0.477` at boundary-local scales versus poorly-attributed at wide scale.

Resolution: **keep a ladder, but bound it by the plateau rather than by hand.**

```
  r_top    = r_hi                       # detection + K-counting rung
  r_bottom = max(r_lo, r_hi / 10)       # gap-estimation rung
  ladder   = geometric sequence, S rungs, r_bottom -> r_top      # default S = 2
```

**Radius-to-scale conversion.** For `z ~ N(0, σ² I_d̂)`, the displacement norm concentrates at `σ√d̂`. So to place a rung at characteristic radius `r`:

```
  sigma_s = r_s / sqrt(d_hat)
```

This is the step where per-coordinate scale and ball radius are reconciled. Getting it wrong is the difference between the fixed-total-budget failure mode and the fixed-per-coordinate one.

### 4.2 Generating probes

```
PROCEDURE Stage1(x, U, d_hat, sigma_s, m, A):

  for i in 1..m:
      z_i     <- draw from N(0, sigma_s^2 * I_{d_hat})
      delta_i <- U z_i
      keep_i  <- DensityFilter(x + delta_i, A)
  retain <- {i : keep_i}
  rho    <- |retain| / m                     # retention rate

  if rho < rho_min:                          # default 0.5
      return ABSTAIN(A3_off_manifold)

  for i in retain:
      y_i <- f(x + delta_i)                  # the only queries spent

  return {z_i, y_i}_{i in retain}, rho
```

**Isotropic in the tangent frame** means: round within the `d̂` estimated directions, zero off them. In ambient `R^D` this is a flat pancake — highly anisotropic — but isotropic in the coordinates that matter. The rationale for roundness is that the gate boundary's orientation is unknown; a round cloud has no blind spot, so the crossing rate is orientation-invariant. Stretching the cloud buys sensitivity along the long axis at the cost of going partly blind to boundaries perpendicular to it.

**Critical distinction from E4.** E4's clumpy probes inherited the data's multimodality and drove honest-model FP to 0.49. That happened because those probes *resampled the empirical distribution* — interpolation- or mixup-style construction. A smooth isotropic cloud in the tangent frame is on-manifold in the geometric sense while remaining **unimodal by construction**. The two senses of "on-manifold" must not be conflated:

- *follows the manifold's geometry* — what we want;
- *reproduces the data's density along it* — what caused the A12 violation.

### 4.3 Density filter

```
PROCEDURE DensityFilter(p, A):
      d_k(p)  <- distance from p to its k-th nearest neighbor in A
      s(x)    <- median k-th-NN distance among neighbors of x     # local spacing
      return  d_k(p) <= c * s(x)                                  # default c = 1.5
```

Cost: one k-d tree over data already held. No queries.

**Validity note.** The filter is a function of `δ` and `A` only — it never sees `y`. So retention is independent of the responses given the design, and dropping probes cannot manufacture response-side structure. It does distort the design distribution (the retained `z` are no longer exactly Gaussian), which the Stage-3 bootstrap absorbs because the bootstrap reruns the entire pipeline including the filter.

**Reporting requirement.** Report `π̂` restricted to retained probes, and report `ρ` alongside. This is what separates *"a gate exists in input space"* from *"a gate fires on inputs the model actually sees."* Only the second is an audit claim. Trimming cannot make this distinction; only the filter can.

---

## 5. Abstention taxonomy

Abstention is not a null result. "I cannot see here" and "there is no gate here" are different claims, and for a system whose job is detecting hidden behavior the difference is the entire product. Current protocol has one abstention trigger (minimum signal, Stage 3). This spec adds four earlier ones, all query-free.

| Code | Trigger | Meaning | Action |
|---|---|---|---|
| `A0_isolated` | too few neighbors, or `k`-NN radius exceeds isolation bound | anchor is not in a sampled region | abstain; do not spend queries |
| `A1_no_plateau` | plateau width `W < W_min` | noise regime runs straight into curvature regime; no radius exists at which assumptions hold | abstain; do not spend queries |
| `A2_no_reduction` | `d̂ ≥ γD` | manifold is effectively full-dimensional; frame buys nothing | **fall back** to ambient E1 baseline, do not abstain |
| `A3_off_manifold` | retention `ρ < ρ_min` | cannot place a ball that stays in-distribution | abstain after filter, before queries |
| `A4_min_signal` | response spread below noise floor (existing) | probe did not perturb the output | abstain |
| `A5_not_separable` | Stage-3 separability check fails | structure present but not gate-attributable | report as *structure, unattributed* — a third outcome, not a flag |

`A0`, `A1`, `A3` all fire **before any query is spent**. This is a budget argument as well as a validity argument: queries currently spent on anchors that cannot support a conclusion get redirected to anchors that can.

---

## 6. Curvature monitoring

`d̂` drift *within the selected ladder* is a live A11 alarm, available at no extra cost since the curve is already computed:

```
  if d_hat(r_top) - d_hat(r_bottom) > CI_width:
      flag A11_MARGIN
      action: (a) shrink r_top to the last rank with CI overlap, or
              (b) escalate to local-quadratic trimmed fit   [UNTESTED]
```

Option (b) is the natural repair — fit a quadratic surface, trim, residualize — and is precisely the item the consolidated note lists as open ("behaviour under curved honest surfaces near the A11 margin"). It is named here for completeness but should not be run without its own validation.

---

## 7. Statistical validity of adaptive scale selection

Choosing analysis settings by looking at the data is normally a selection effect: try several radii, keep the one where the signal looks strongest, and the reported p-value is meaningless because you fished. That failure mode does not arise here, and the reason is structural rather than a matter of care:

> **Stage 0 reads only `A`. It never queries `f`. Therefore the selected radius is statistically independent of the responses subsequently tested.**

Consequences:

1. **No selection penalty for adaptivity.** Per-anchor radii, chosen from geometry, tested without correction for the choice.
2. **Ladder shrinks from 3 hand-set rungs to `S` geometry-placed rungs** (default 2). Bonferroni over rungs actually used — so a modest but free power gain from fewer tests at the same threshold.
3. **The precondition in §2 is what carries this.** If `A` were chosen using prior responses to `f`, independence breaks and both consequences vanish.

Independence of the *selection* does not, however, license skipping bootstrap calibration in Stage 3. Trimming is a nonlinear, data-dependent operator; anything calibrated against untrimmed theory is miscalibrated. Bootstrap **through the full pipeline** — resample inlier residuals, regenerate responses, rerun filter, trim, refit, residualize, test. `B = 300`. This costs zero model queries, which is the binding constraint; it costs only CPU.

---

## 8. Interface to Stages 2–3

Stage 0/1 hand downstream:

```
  z_i in R^d_hat        design matrix in tangent coordinates
                        (Stage 2 regresses in the frame, NOT in R^D —
                         this is where the dimension reduction is realized)
  y_i                   responses, retained probes only
  d_hat, U              for lifting recovered boundary normals back to R^D
  rho                   retention rate, for the claim scope
  W, lambda_ratio       geometry confidence, for the report
  A11_MARGIN flag       modulates trust in the Stage-2 linear fit
```

Downstream behavior unchanged except:

- Stage 2 LTS operates on `d̂`-dimensional `z`, not `D`-dimensional `δ`. Fewer parameters, better-conditioned fit.
- Stage 3's separability guard fits its linear classifier on `z`. **Its decision boundary, lifted through `U`, is an estimate of the gate boundary's normal in ambient space** — the object Stage-B certification needs. Test and recovery are the same computation.

---

## 9. Validation plan

All four are cheap; V1 and V2 reuse existing anchors.

**V1 — retro-diction on E4, zero new queries.** `[PROPOSED]`
E4's false positives occurred at probe scales comparable to inter-cluster spacing. That is also the radius at which a neighborhood starts straddling two clusters, which inflates `d̂` and ends the plateau.

> **Prediction:** on the E4 clumpy manifold, `r_hi` falls below the inter-cluster spacing, and the plateau rule excludes the radii that produced FP = 0.49.

Falsifiable, and the anchors already exist. If it holds, the plateau rule *prevents* the A12 violation before the test rather than repairing it after. If it fails, the two scales are unrelated and §3.3's central claim needs rework — worth knowing early either way.

*Caveat to state in any writeup:* plateau collapse and FP inflation share a common cause (the clustering scale) rather than one independently diagnosing the other. For a decision *rule* this is fine — the boundary only has to land in the right place — but it is not evidence of an independent mechanism.

**V2 — policy comparison on E4 and E1 cells.**
Fixed 3-scale ladder vs. plateau-placed ladder. Report FP on honest controls, power on detectable anchors, query spend, and abstention rate by code. The interesting cell is whether abstention concentrates on anchors where the fixed ladder was silently wrong.

**V3 — mixing-fraction sweep.**
Fix separation at 1.5 component-sd (the E1 post-OLS regime, below the dip's resolution floor). Sweep `π` from 0.1 to 0.5. Compare dip / equal-variance mixture LRT / separability guard. Two questions: does the LRT have power where the dip provably has none, and does the LRT degrade more gracefully than trimming as `π → 0.5`, where LTS cannot identify a majority branch?

**V4 — curvature sweep.** The test of §3.3.
Impose known curvature on the honest surface, sweep its magnitude. Does `r_hi` track the radius at which local-linear fitting measurably degrades? If plateau top and A11 margin move together, the equivalence claim is supported; if they decouple, the plateau is a dimension diagnostic only and the A11 story must be dropped.

---

## 10. Open items

1. **`π ≈ 0.5`.** Trimming at `h = 0.75m` has 25% breakdown by construction; covering mixing up to a half requires `h → 0.5m`, which is exactly where "majority branch" ceases to be defined. Unresolved by anything in this spec. The mixture LRT does not need a majority and may degrade better (V3).
2. **Curved honest surfaces.** Local-quadratic trimmed fitting is the obvious repair and is untested.
3. **Heterogeneous intrinsic dimension.** If different clusters have genuinely different local `d` (rather than one manifold with curvature), that is a different diagnosis with a different fix. The Hidalgo line of work is the relevant tool.
4. **Noise-induced plateau shift.** `τ` is known here. If noise displaces the plateau's lower edge predictably, correct rather than avoid.
5. **Estimated non-isotropic whitening** for vector outputs (carried over from E3).
6. **E3's identifiability limit is untouched by any of this.** A penalty parallel to strong within-branch variation is invisible at every radius in every dimension. Not a probing problem; do not expect the probe policy to help.

---

## 11. Parameters

| Param | Default | Source |
|---|---|---|
| `k_max` | 200 | neighborhood size for GRIDE / PCA |
| `k_min` | 30 | below this, `d̂` variance unacceptable |
| `W_min` | 0.35 decades | plateau width abstention threshold — **tune in V2** |
| `γ` | 0.75 | `d̂/D` above which reduction is not worth it |
| `c` | 1.5 | density filter tolerance, multiples of local spacing |
| `ρ_min` | 0.5 | retention abstention threshold |
| `S` | 2 | ladder rungs inside plateau |
| `h` | `0.75m` | LTS trim fraction (E1-validated) |
| `m` | 1000 | queries per rung |
| `B` | 300 | bootstrap replicates |
| `α` | 0.05 | with Bonferroni over rungs actually used |

Unsourced defaults (`W_min`, `c`, `ρ_min`) are placeholders pending V2 and should be registered before the confirmatory run, not tuned on it.

---

## 12. Summary of the argument

- Detectability, not just power, is governed by effective dimension. Working in `d̂` rather than `D` improves the crossing rate directly. `[VALIDATED: E1(a)]`
- Trimmed residualization removes the classical penalty for probing wide. `[VALIDATED: E1, E4]`
- With that penalty gone, the binding constraint on radius is A11 validity, not variance.
- The GRIDE plateau is hypothesized to be exactly the A11-valid window, making "probe at the top of the plateau" a derived rule rather than a tuned one. `[PROPOSED — V4]`
- Absence of a plateau is a principled reason to abstain before spending queries, which converts silent failures into declared ones.
- Because the geometry stage never queries `f`, all of this adaptivity is free of selection penalty.
- The density filter, not the trimming, is what licenses the deployment-relevant form of the audit claim.
