# Routing Audit: Probe Geometry, Scale Selection, and Structure Testing

**Consolidated working note**
Covers: the four future-work experiments (E1–E4), the trimmed-residualization operator, probe geometry in high dimension, tangent-frame on-manifold probing, GRIDE-based scale selection, the choice of structure test, the resulting end-to-end pipeline, a validation plan, and the scope limits of the method.

---

## 0. How to read this

Claims carry provenance tags throughout:

| Tag | Meaning |
|---|---|
| `[VALIDATED]` | established by the E1–E4 simulations with known ground truth |
| `[DERIVED]` | follows analytically from a validated result |
| `[PROPOSED]` | design proposal with a stated falsification test in §9; **not run** |
| `[OPEN]` | acknowledged gap, no proposed fix or an untested one |
| `[LIMIT]` | proven or structural impossibility; not expected to yield to engineering |

The distinction matters more than usual here. Two of the four original propositions were falsified, and the repairs that came out of those falsifications are now the backbone of the method. Keeping the tags visible is what prevents the proposed layer from being read as the tested layer.

A glossary of statistical terms is in Appendix A.

---

# PART I — WHAT THE EXPERIMENTS ESTABLISHED

## 1. The audit in one paragraph

A black-box model `f` may contain a hidden behavioral switch: one rule on some inputs, a different rule on others, with no visible indication. Access is query-only. The audit works locally: pick an anchor `x`, send a cloud of nearby queries `f(x + δ_i)`, and examine the response distribution. If the responses separate into two clumps rather than one smooth spread, the probe cloud plausibly straddled a boundary. The mixing fraction `π` is the share of probes landing on the far side; the penalty `Δ` is the size of the jump.

Common protocol across all four experiments: three probe scales, `m = 1000` queries per test, `α = 0.05` with Bonferroni over tests actually made, gate penalty `Δ = 0.30`, observation noise `τ = 0.02`, per-scale minimum-signal abstention, and — critically — **a matched no-gate control for every gated model**, so false positives are measured on identical geometry. Power is reported only on anchors whose probes actually reach a boundary (`π_true ≥ 0.05`).

## 2. E1 — probing the full input vector

**Question.** The original probes perturbed location only. What happens when the probe perturbs every input coordinate?

**Result on budget geometry.** `[VALIDATED]`

Two probe budgets behave completely differently:

- **Fixed total budget** (per-coordinate sd `σ/√D`): the ball's radius stays ~`σ` as `D` grows, so it shrinks relative to the space. Crossing rate is `Φ(−d√D/σ)` and collapses. This destroys *detectability*, not merely power — detectable anchors fell 100 → 78 → 56 at `D = 8, 16`. Empirical median `π_true` matched the formula to three decimals (`D = 16`: 0.063 observed vs. 0.067 predicted).
- **Fixed per-coordinate** (sd `σ` on each coordinate): the ball's radius grows like `σ√D`, so crossing is `D`-independent. But within-branch spread from the other coordinates grows as `σβ√(D−1)`, masking the gap. Power fell 1.00 → 0.55 by `D = 16`.

**The falsification.** The predicted repair — OLS-residualize `y` on the displacement — failed badly (power 0.03–0.05).

**Why, in closed form.** `[DERIVED]` The step is strongly correlated with `δ₁`, so the fitted plane absorbs most of it. With `z = δ₁/σ` and a boundary through the anchor, residual between-branch separation is

```
  Δ · ( 1 − φ(0) · ( E[z | z>0] − E[z | z<0] ) )  =  Δ · ( 1 − 2φ(0)√(2/π) )  ≈  0.36 Δ
```

— plain OLS absorbs roughly **64%** of the gap at balanced mixing — while the tilted fit adds within-branch spread ≈ `0.24Δ`. Net separation ≈ 1.5 component standard deviations, and the dip goes blind.

**The repair: trimmed residualization.** `[VALIDATED]` Fit the plane, drop the worst-fitting 25% by `|residual|`, refit on the remainder, then residualize *all* points against the refit plane. Because the discarded points are predominantly the minority branch, the plane describes the majority branch only, and the jump survives at full size.

| power | D=2 | D=4 | D=8 | D=16 | queries |
|---|---|---|---|---|---|
| isotropic, fixed total | 1.00 | 0.95 | 0.83* | 0.86* | 3000 |
| isotropic, per-coord | 1.00 | 0.98 | 0.80 | 0.55 | 3000 |
| + OLS residualization | 0.03 | 0.05 | 0.01 | 0.02 | 3000 |
| **+ trimmed residualization** | **0.76** | **0.75** | **0.74** | **0.74** | 3000 |
| coordinate scan | 1.00 | 1.00 | 1.00 | 1.00 | 3000·D |

\*conditional on detectability, which itself collapses. FP = 0.00 throughout.

**Headline:** trimmed residualization makes power **independent of ambient dimension**, and at `D = 16` beats raw probing at equal query cost. The coordinate scan still wins outright (1.00) but costs `D×` queries; trimming is the fixed-budget choice.

## 3. E2 — more than two regimes

Three regimes at levels `0 / −0.25 / −0.75`.

- **Counting.** `[VALIDATED, with repair]` Raw BIC *over*-partitions (accuracy 0.64, dominant error `K=3 → K̂=4`), because trend smear inside each branch is non-Gaussian and BIC buys extra components to absorb it. Reusing the audit's **already-registered effect-size floor (0.15) as a component-merge rule** lifts accuracy to 0.75 and flips the error direction to mild under-counting — the conservative direction for an audit claim.
- **Gap estimation.** `[VALIDATED]` Accurate at boundary-local scales (medians 0.228 and 0.477 against true 0.25 and 0.50). At wide scale the total span is recovered (0.61 vs. 0.75) but attributed poorly across the two gaps. **Design lesson: count wide, estimate narrow.**
- **Label matching.** `[VALIDATED, with repair]` Per-anchor slope estimates absorb the step near boundaries — the E1 mechanism again — so trend adjustment must use the **global median** slope (`β̂ = 0.134` vs. true 0.15). With that fix, clustering trend-adjusted component levels labels components at **0.997** accuracy (n = 330), despite biased cluster centers (`0 / −0.19 / −0.64`). Matching is far easier than estimation, which is why the scalar case needs no assignment machinery.

## 4. E3 — vector-valued outputs

Responses in `R^V`, within-branch variation along `u`, penalty along `v`, `θ = ∠(u,v)`.

| power (V=3) | θ=0° | 30° | 60° | 90° |
|---|---|---|---|---|
| PC1 only | 0.00 | 0.00 | 0.62 | 1.00 |
| top-2 PCs (Bonf.) | 0.00 | 1.00 | 0.67 | 1.00 |
| output axes (Bonf.) | 0.00 | 1.00 | 1.00 | 1.00 |
| 8 random projections | 0.00 | 0.97 | 1.00 | 1.00 |

**The governing geometry is the decomposition of the penalty into components parallel and orthogonal to the within-branch variation.** The parallel component hides inside the trend; the orthogonal component is nearly noise-free. Three consequences:

- At `θ = 0`, **nothing detects, at any radius, in any dimension, under any projection.** `[LIMIT]` This is a genuine identifiability limit, not a projection failure.
- At intermediate angles PC1 fails because it sees only the masked parallel part; any projection with weight on the orthogonal component succeeds.
- At `θ = 90°` PC1 *succeeds*, contrary to the prediction, because the scale ladder immunizes it: within-branch variance shrinks like `(βσ)²` while the gap stays `Δ`, so at the smallest scale PC1 rotates onto `v`.

**Penalty-vector recovery.** `[VALIDATED, with repair]` Raw group-mean differences are contaminated by trend × support offset (`‖Δ̂‖ = 0.57`, cosine 0.52). Redefining the estimand as the difference of **group-conditional local-linear predictions at the anchor** recovers `‖Δ̂‖ = 0.297–0.301` with cosine **1.000** to `v`, at both `V = 3` and `V = 8`.

## 5. E4 — clumpy (non-geographic) covariate manifolds

Six-cluster covariate mixture, strong smooth trend, A11 satisfied.

| manifold | method | FP (honest) | power |
|---|---|---|---|
| clumpy | raw | **0.49** | 0.86 |
| clumpy | OLS residualization | 0.00 | 0.16 |
| clumpy | **trimmed residualization** | 0.00 | **0.57** |
| uniform | raw | 0.00 | 0.14 |
| uniform | **trimmed residualization** | 0.00 | **0.90** |

- The A12 violation is reproduced in vitro: same model, same probes, FP 0.49 vs. 0.00, **differing only in the data distribution**. `[VALIDATED]`
- OLS restores validity but pays the E1 absorption cost; trimming keeps validity *and* power.
- The unexpected row is uniform + trimmed: power **0.14 → 0.90**, because trimming strips a trend that had been masking the gap at large scales. This converts a "trend width vs. gap" power limit into a "residual width vs. gap" limit — strictly better wherever the honest surface is locally linear (A11).

## 6. Cross-cutting conclusions from E1–E4

1. **Trimmed local-linear residualization should be a standard pipeline stage.** It appears independently as the repair in E1 (dimension-independence), E4-clumpy (A12 validity), and E4-uniform (trend unmasking), and its failure mode is understood in closed form.
2. **The registered effect-size floor does double duty** as a merge rule for `K` selection.
3. **Scale ladders have two distinct jobs** — count wide, estimate narrow — and the small end is what immunizes PC1 in E3.
4. **Two genuine identifiability limits surfaced:** the parallel penalty (E3, θ=0) and fixed-total-budget probing in high dimension (E1a).
5. **Recovery estimands must be group-conditional predictions at the anchor**, never raw component-mean differences.

---

# PART II — THE TRIMMING OPERATOR

## 7. What it is, precisely

The operator is **least trimmed squares** with `h = 0.75m`. Naming it that is worth doing, because it imports existing theory: Rousseeuw's breakdown results, FAST-LTS as a solver, and standard robust scale estimates.

It also makes the `π ≈ 0.5` problem legible rather than mysterious. `h = 0.75m` has **25% breakdown by construction**. Covering mixing fractions up to a half requires `h → 0.5m`, which is precisely the point at which "majority branch" ceases to be a defined concept. `[LIMIT-adjacent]`

## 8. Why it works, and where the intuition should stop

The mechanism is not variance reduction. It is *keeping the fitted plane on one branch*. Plain OLS finds a compromise plane tilted between the two branches, which is why it eats the gap; trimming removes the minority points from the fit so the plane commits to the majority, and the minority's jump then appears at full size in the residuals.

Two boundaries on this intuition:

- **Trimming fixes masking, not reaching.** `[DERIVED]` No post-hoc residualization creates mixing that isn't there. Whether the ball touches the boundary is pure geometry, governed by `Φ(−d√D/σ)`. This is why the per-coordinate budget is mandatory, and why reducing the working dimension (Part IV) helps in a way that trimming cannot.
- **It assumes a minority branch exists to trim.** At `π ≈ 0.5` the assignment is ambiguous and the plane may land between branches.

**Practical consequence for probe design:** the classical reason to keep probe radius small — that a wide ball sweeps in honest variation that masks the gap — is exactly what trimming removes. This is what makes the aggressive radius policy in Part VI defensible.

---

# PART III — PROBE GEOMETRY

## 9. Isotropy, and what it does and doesn't fix

**Isotropic = same in every direction** (*iso*, equal + *tropos*, turn). An isotropic probe cloud has no preferred orientation; it is a round ball.

- Isotropic: `δ ~ N(0, σ²I)`.
- Anisotropic: `δ ~ N(0, Σ)`, `Σ ∝̸ I` — a stretched ellipsoid.
- Coordinate-targeted (E1's scan): one nonzero entry — maximally anisotropic.

**Why round is the default:** the gate boundary's orientation is unknown. A round cloud has no blind spot, so the crossing rate is orientation-invariant. Stretching buys sensitivity along the long axis at the cost of going partly blind to boundaries perpendicular to it.

**Isotropy fixes shape, not size.** Both E1 budget rows are isotropic; they differ only in scale. This is the vocabulary trap worth avoiding in any writeup: "isotropic probe" underdetermines the design, and the budget choice is what actually decides whether detection is possible.

## 10. The delta-ball verdict

> **Fixed per-coordinate isotropic probe + trimmed residualization = one probe, dimension-independent, no coordinate scan needed.** `[VALIDATED]`

With three caveats:

- The coordinate scan still reaches 1.00 vs. 0.75. If `D×` queries are affordable, power is being left on the table. The delta-ball is the fixed-budget choice, not the strictly better one.
- Fixed per-coordinate means the ball radius grows like `σ√D`. It is a fixed-*marginal* ball, not a fixed-*radius* one. Holding radius fixed for budget reasons reinstates the failure.
- It rests on A11. Curved honest surfaces near the margin are untested; a curved trend leaves structured residuals that can present as bimodal. `[OPEN]`

---

# PART IV — ON-MANIFOLD PROBING

## 11. Two different things called "on-manifold"

E4's clumpy probes were on-manifold and that was the **liability**: FP = 0.49 on an honest model. This needs care, because the phrase covers two different operations:

| Sense | Construction | Effect |
|---|---|---|
| follows the manifold's **geometry** | smooth cloud in the local tangent space | what we want — stays in-distribution, remains unimodal by construction |
| reproduces the data's **density** along it | interpolation / mixup-style resampling of neighbors | inherits the data's multimodality → the A12 violation |

E4 conflated them. The fix in Part V uses the first sense only.

## 12. Why bother with on-manifold probing at all

The reason is **not** statistical — it is about what the claim means. Querying `f` at points that never occur in deployment yields "a switch exists somewhere in input space." What an audit needs is "a switch fires on inputs the model actually sees." Off-manifold probes cannot support the second claim, and no amount of trimming converts one into the other.

## 13. The black-box fix: estimate the tangent space from the anchor sample

No generative model of the manifold is needed, and none is available. What is needed is local *directions*, and the anchors are already a sample from the data distribution:

1. Take the `k` nearest real anchors to `x`.
2. Estimate the local intrinsic dimension `d̂` (Part V).
3. Local PCA on those neighbors; keep the leading `d̂` directions → tangent frame `U`.
4. Probe isotropically **in that frame**, at fixed per-coordinate scale.

Three payoffs:

- Effective dimension drops from `D` to `d̂`, so the crossing rate improves as `Φ(−d√d̂/σ)`. **Detectability recovers, not just power.** `[DERIVED from E1a]`
- On-manifold in the geometric sense, without ever writing the manifold down.
- Clumpiness is **not** inherited, because a smooth Gaussian cloud in the tangent frame is unimodal by construction.

"Isotropic in the tangent frame" means round within the `d̂` estimated directions and zero off them. In ambient `R^D` that is a flat pancake — highly anisotropic — but isotropic in the coordinates that matter. Which is the point: no preferred direction among the directions the data actually varies in, and no probing at all in the ones it doesn't.

## 14. Plausibility filter

For each probe point, compute distance to its `k`-th nearest real anchor and compare against local spacing; drop or flag probes in low-density regions. Costs one k-d tree over data already held; **zero queries**.

Two properties worth stating explicitly:

- **Validity:** the filter is a function of `δ` and the anchor set only. It never sees `y`. So retention is independent of responses given the design, and filtering cannot manufacture response-side structure.
- **It is what licenses the deployment-relevant claim.** Reporting `π̂` restricted to retained probes separates *gate exists* from *gate fires*. Trimming cannot make this distinction; only the filter can.

**Trimming stays as the backstop.** Tangent estimates will be wrong near curvature and near cluster boundaries; leaked off-tangent variation appears as extra within-branch spread, which is exactly the failure trimming already handles. Layering them means geometry improves reach and claim scope, and the robust fit covers geometry's errors.

---

# PART V — INTRINSIC DIMENSION AND THE PLATEAU

## 15. GRIDE and why this estimator

GRIDE (Denti, Doimo, Laio, Mira) is the generalized-ratio successor to TWO-NN (Facco et al.). The construction: for a locally uniform sample on a `d`-manifold, the ratio `μ = r_{n₂}/r_{n₁}` of neighbor distances has a distribution depending only on `d`. **The unknown local density cancels in the ratio.** TWO-NN uses `μ = r₂/r₁`, whose CDF is `1 − μ^(−d)`; GRIDE generalizes to arbitrary ranks.

Reasons for this choice:

1. **Natively scale-resolved.** TWO-NN gives one number at the smallest scale. GRIDE gives a curve `d̂(n₁)`, and we need the curve, not the number. The audit already runs a scale ladder, so `d̂` is evaluated at the same radii as the probes.
2. **Decouples noise-averaging from scale.** With TWO-NN, variance reduction requires more points requires larger radii requires more curvature bias — the three are chained. GRIDE breaks the chain: raise `n₁, n₂` together to average over more neighbors at roughly fixed scale. With small, noisy anchor neighborhoods this matters.
3. **Density-robust by construction** — necessary, since E4's failure mode was a six-cluster covariate mixture.
4. **Fractional output is informative.** `d̂ = 4.7` usually means the sampled scale spans structure of more than one dimension; we round for the frame, but the fractional part calibrates confidence in the rounding.
5. **Query-free**; needs only pairwise distances.

Rejected: MLE (Levina–Bickel) underestimates at high `d`; correlation dimension needs an eye-identified scaling region; PCA eigenvalue cutoffs reintroduce the arbitrary threshold being eliminated. Local PCA is still used for the *frame* — GRIDE decides where to truncate it, replacing a scree-plot judgment with an estimate that has a stated generative model.

**Caveat worth taking seriously.** `[OPEN]` GRIDE assumes local uniformity within the neighborhood used; density cancels *because* it is treated as constant there. On a six-cluster mixture at radii comparable to inter-cluster spacing, that assumption is exactly what fails, and `d̂` inflates because the neighborhood sees two clusters rather than one patch. "Robust to density variation" means robust to density varying *across* the dataset, not to a neighborhood straddling a density discontinuity. In practice this is arguably useful signal — but it means `d̂` inflation and FP inflation share a common cause rather than one independently diagnosing the other.

## 16. The three regimes of the d̂ curve

| Scale | Behavior | Cause |
|---|---|---|
| small | `d̂` inflated toward `D` | measurement noise is full-dimensional, kicking points off-manifold in all `D` directions |
| middle | `d̂` flat — **the plateau** | past the noise, not yet feeling curvature; `d̂` ≈ true intrinsic dimension |
| large | `d̂` drifts | neighborhood bending, or has swallowed a second cluster; local uniformity fails |

---

# PART VI — THE PLATEAU RULE

## 17. The central hypothesis

> **The plateau is defined by "the neighborhood looks like a flat, evenly-sampled patch." That is A11.** The radius range where `d̂` is stable is the radius range where local-linear fitting is honest. `[PROPOSED]`

This is not claimed as a heuristic correlation but as the same geometric condition read two ways. It is the load-bearing assumption of Parts VI–VIII, and §22 (V4) is its test. If it fails, the plateau remains a dimension diagnostic and the A11 story must be dropped.

**What the plateau does not tell you:**

- **Whether the gate is reachable.** The plateau is a property of the data's shape; the gate is a property of `f`. A gate outside the locally-flat window is simply not found — and the rule makes this *more* likely, since it bounds radius from above. `[LIMIT]`
- **Which directions to keep** — only how many.

## 18. The radius rule

Within the plateau, larger radius is strictly better for detection: it raises the crossing rate and hence `π`, and `π` must be non-trivial for any test to have power. The classical counterargument — wide balls sweep in masking variation — is exactly what trimming removes (§8, E4-uniform 0.14 → 0.90).

The two results compose:

> **Trimming removes the reason to be timid about radius. The plateau says how far you can push before the plane-fitting itself becomes invalid. Probe at the largest radius still inside the plateau.**

E2 still wants two rungs (count wide, estimate narrow), so the ladder survives — but it is now *bounded by geometry* rather than hand-set.

**Radius-to-scale conversion.** For `z ~ N(0, σ²I_d̂)` the displacement norm concentrates at `σ√d̂`, so placing a rung at characteristic radius `r` requires

```
  σ_s = r_s / √d̂
```

Conflating radius with per-coordinate scale here reproduces the fixed-total-budget failure mode.

## 19. Plateau width as an abstention criterion

- **Wide plateau** → a comfortable band of radii where the manifold is flat and evenly sampled. Trust the frame, trust the plane, run the test.
- **Narrow or absent plateau** → the noise regime runs straight into the curvature regime with nothing clean between. **There is no radius at which the assumptions hold.** Decline to test rather than test badly.

This matters because **abstaining is not the same as finding nothing.** An audit that abstains says "I cannot see here." An audit that tests on a bad neighborhood and returns clean says "there is no gate here," which may be false. For a system whose purpose is detecting hidden behavior, that distinction is the entire product.

## 20. Why the adaptivity is statistically free

Choosing analysis settings by looking at the data is normally a **selection effect**: try several radii, keep the one where the signal looks strongest, and the p-value is meaningless. That does not arise here, for a structural reason:

> **`d̂` is computed from the anchor covariates alone. It never queries `f`. Therefore the selected radius is statistically independent of the responses subsequently tested.** `[DERIVED]`

Consequences:

1. Per-anchor adaptive radii with **no multiple-comparison penalty for the selection**.
2. The ladder shrinks from three hand-set rungs to `S` geometry-placed rungs (default 2), so **Bonferroni divides by fewer tests — free power at the same threshold**.
3. This rests entirely on the precondition that the anchor set is fixed *before* any query to `f`. If anchors were chosen using prior responses, both consequences vanish.

Independence of the *selection* does not license skipping bootstrap calibration (§21.4).

---

# PART VII — THE STRUCTURE TEST

## 21. Dip versus mixture LRT

### 21.1 What the dip buys and costs

Hartigan's dip tests `H₀: the density is unimodal`. It assumes almost nothing — no Gaussianity, no `K`, no equal variances. Two costs:

- **A hard resolution floor.** A two-component Gaussian mixture is *literally unimodal* until separation exceeds roughly 2 component sds. Below that, the density has one hump, so the dip is not losing power — **the alternative does not exist under its null.**
- **Conservative calibration.** The tabulated critical values use the uniform as the least-favorable unimodal null. Residuals here are roughly Gaussian, which is far more compact, so standard tables give away power to protect against a null shape we do not have.

**This reframes E1.** The diagnosis there was separation ≈ 1.5 component-sds after OLS absorption, "and the dip goes blind." That is a mixture sitting *just below the dip's floor*, not a signal that vanished. A likelihood-based test would plausibly have had real power at 1.5 sds with `m = 1000`. So part of what was attributed to OLS destroying the signal was **the test** being unable to see a signal that was still present. Trimming worked by pushing separation back above the floor; a better test would have made trimming an improvement rather than a rescue. `[PROPOSED — V3 tests this]`

### 21.2 Making the GLRT admissible

The standard regularity failures, each with a specific fix:

| Problem | Fix |
|---|---|
| null on the boundary (`π = 0` is an endpoint) | bootstrap calibration |
| non-identifiability under the null (`μ₂` free when `π = 0`) | bootstrap calibration |
| unbounded likelihood (one component's `σ → 0` on a point) | **model choice**: equal-variance location mixture |

For unrestricted normal mixtures the LRT statistic diverges (Hartigan, 1985), so this is not a technicality.

**Model choice does most of the work.** The alternative here is not "any two-component mixture." A gate is an *additive offset*: both branches share the honest surface and the same observation noise. So fit

```
  H0:  r ~ N(mu, sigma^2)
  H1:  r ~ pi * N(mu1, sigma^2) + (1-pi) * N(mu2, sigma^2)      # shared sigma
```

One shared `σ` eliminates the degenerate spike, and the narrower alternative buys power.

### 21.3 What the fit returns beyond a p-value

`π̂` and `Δ̂ = |μ̂₁ − μ̂₂|` fall out directly. Both are needed anyway — `π̂` is the mixing fraction, `Δ̂` is what the registered effect-size floor screens.

### 21.4 Bootstrap calibration, through the whole pipeline

Fit the robust plane, resample residuals from the inlier set, generate `y*ᵢ = â + b̂′zᵢ + e*ᵢ`, and rerun **the entire pipeline** — filter, trim, refit, residualize, fit both mixtures, compute `2(ℓ₁ − ℓ₀)`. `B = 300`. The p-value is the exceedance fraction.

"Entire pipeline" is not optional. Trimming is a nonlinear, data-dependent operator; testing post-trim residuals against any null derived from untrimmed theory is miscalibrated. This applies to the dip as well — bootstrapping its null through the pipeline recovers the power lost to uniform calibration.

**The cost structure is favorable:** bootstrap costs **zero model queries**, only CPU, and queries are the binding constraint.

For asymptotics rather than a bootstrap, the penalized LRT (Chen–Chen–Kalbfleisch) and the EM-test (Li–Chen–Marriott) both restore tractable limits — mixtures of chi-squares — by penalizing `π` away from the boundary. For an audit, bootstrap and cite these as backing.

### 21.5 The tradeoff, stated plainly

| | `H₀` | Fails when |
|---|---|---|
| Dip | residual density is unimodal | separation < ~2 sd (**no power**) |
| Mixture LRT | residual density is one Gaussian | A11 violated (**false positives**) |

The LRT's null is much narrower. Under curvature, residuals become skewed or heavy-tailed but remain unimodal — the dip stays valid, while the LRT reports two components because a mixture fits skewness better than one Gaussian does. **A power floor is traded for a robustness hole.**

The iid residual bootstrap only partly patches this: it inherits the *marginal* shape of the misspecification, so calibration improves, but resampling destroys the **spatial coupling** between residual and `z`. Curvature is spatially coherent; the bootstrapped null is not. The observed statistic still looks anomalous.

### 21.6 The separability guard

The structure neither test uses: a real gate is not merely a mixture in response space, it is a mixture whose components are **linearly separable in `z`**, because the boundary is locally a hyperplane. Curvature artifacts produce spatially interleaved components.

So after fitting the mixture: take the responsibilities, fit a linear classifier on `z`, evaluate held-out balanced accuracy.

- Real gate → near-perfect separation.
- Skew/curvature artifact → near chance.

Costs nothing, and discriminates exactly the failure mode the LRT introduces. **And the classifier's decision boundary, lifted through `U`, is an estimate of the gate boundary's normal in ambient space** — the object Stage-B certification needs. Test and recovery become the same computation.

### 21.7 Decision rule

Run dip and LRT in parallel at each rung:

| Outcome | Interpretation |
|---|---|
| both fire | strong evidence; report `Δ̂`, `π̂` from the mixture fit |
| LRT only | sub-bimodality regime — **the new operating range**; gate the claim on the separability check and the effect-size floor |
| dip only | suspicious; usually `K > 2` or badly non-Gaussian components — escalate to the E2 path |
| neither | abstain per minimum-signal |

Bonferroni over rungs actually used.

---

# PART VIII — THE PIPELINE

## 22. Where everything sits

```
  STAGE 0   local geometry                        [ZERO queries]
            GRIDE d-hat curve -> plateau -> [r_lo, r_hi]
            local PCA -> tangent frame U (d-hat columns)
            geometric abstention gate
                |
  STAGE 1   probe generation                      [m queries per rung]
            ladder placed inside plateau, sigma_s = r_s / sqrt(d_hat)
            isotropic in tangent frame, fixed per-coordinate
            density filter -> retention rho
                |
  STAGE 2   trimmed local-linear residualization  [VALIDATED: E1, E4]
            LTS in tangent coordinates z, h = 0.75m
                |
  STAGE 3   structure test
            dip + equal-variance mixture LRT
            bootstrap calibration through full pipeline
            spatial separability guard
                |
  STAGE 4   report: {flag / abstain / structure-unattributed},
                    pi-hat, Delta-hat, boundary normal, scope (rho)
```

**Stage 0 is an enhancement over a working baseline, not a prerequisite.** If it abstains or fails, fall back to ambient isotropic probing at fixed per-coordinate scale plus trimming — E1's validated 0.75, flat in dimension. Nothing in Stage 0 is load-bearing for *correctness*; it is load-bearing for power, reach, and knowing when to stop.

## 23. Stage 0 procedure

```
PROCEDURE Stage0(x, A, params):

  1. NEIGHBORS
     N <- k_max nearest neighbors of x in A
     if |N| < k_min or median-kNN-radius(x) > R_iso:
         return ABSTAIN(A0_isolated)

  2. d-HAT CURVE
     for n1 in ladder of neighbor ranks (n2 = 2*n1):
         d_hat[n1], CI[n1] <- GRIDE(N, n1, n2)     # profile-likelihood CI
         r[n1]             <- median distance to n1-th neighbor

  3. PLATEAU DETECTION
     P <- longest contiguous run of ranks whose CIs have
          nonempty common intersection
     W <- log10( r[max P] / r[min P] )              # width, in decades
     if W < W_min: return ABSTAIN(A1_no_plateau)
     d_hat <- round(midpoint of common CI intersection)
     r_lo, r_hi <- r[min P], r[max P]

  4. DIMENSION SANITY
     if d_hat >= gamma * D: return FALLBACK(ambient)   # E1 baseline

  5. TANGENT FRAME
     U <- top d_hat principal directions of neighbors within r_hi, centered at x
     lambda_ratio <- lambda_{d_hat+1} / lambda_{d_hat}   # leakage diagnostic

  return {d_hat, U, r_lo, r_hi, W, lambda_ratio}
```

Plateau detection by **overlapping confidence intervals** is what turns "read the flat bit by eye" into a decidable rule, and makes width a number that §19 can threshold. `lambda_ratio` near 1 means the truncation is unstable — a leading indicator of frame error.

## 24. Stage 1 procedure

```
  r_top    = r_hi                        # detection + K-counting rung
  r_bottom = max(r_lo, r_hi / 10)        # gap-estimation rung
  ladder   = geometric sequence, S rungs                 # default S = 2

PROCEDURE Stage1(x, U, d_hat, sigma_s, m, A):
  for i in 1..m:
      z_i     <- N(0, sigma_s^2 * I_{d_hat})
      delta_i <- U z_i
      keep_i  <- DensityFilter(x + delta_i, A)
  rho <- |retain| / m
  if rho < rho_min: return ABSTAIN(A3_off_manifold)
  for i in retain: y_i <- f(x + delta_i)          # the only queries spent
  return {z_i, y_i}, rho

PROCEDURE DensityFilter(p, A):
  d_k(p) <- distance from p to its k-th nearest neighbor in A
  s(x)   <- median k-th-NN distance among neighbors of x
  return d_k(p) <= c * s(x)
```

## 25. Abstention taxonomy

| Code | Trigger | Meaning | Action |
|---|---|---|---|
| `A0_isolated` | too few neighbors / `k`-NN radius too large | anchor not in a sampled region | abstain, **no queries spent** |
| `A1_no_plateau` | `W < W_min` | no radius exists at which assumptions hold | abstain, **no queries spent** |
| `A2_no_reduction` | `d̂ ≥ γD` | manifold effectively full-dimensional | **fall back** to E1 baseline, do not abstain |
| `A3_off_manifold` | `ρ < ρ_min` | cannot place an in-distribution ball | abstain, **before queries** |
| `A4_min_signal` | response spread below noise floor | probe did not move the output | abstain |
| `A5_not_separable` | separability guard fails | structure present, not gate-attributable | report as third outcome |

Three of six fire before any query is spent — a budget argument as well as a validity one.

## 26. Curvature monitoring

```
  if d_hat(r_top) - d_hat(r_bottom) > CI_width:
      flag A11_MARGIN
      action: (a) shrink r_top to last rank with CI overlap, or
              (b) escalate to local-quadratic trimmed fit    [UNTESTED]
```

Option (b) is the natural repair for the consolidated note's open item ("behaviour under curved honest surfaces near the A11 margin") but must not ship without its own validation.

## 27. Interface to downstream stages

```
  z_i in R^d_hat     design matrix in tangent coordinates
                     (Stage 2 regresses in the frame, NOT in R^D --
                      this is where the dimension reduction is realized)
  y_i                responses, retained probes only
  d_hat, U           for lifting recovered boundary normals back to R^D
  rho                retention rate -> claim scope
  W, lambda_ratio    geometry confidence
  A11_MARGIN         modulates trust in the Stage-2 linear fit
```

---

# PART IX — VALIDATION PLAN

## 28. Four experiments

**V1 — retro-diction on E4, zero new queries.** `[PROPOSED]`
E4's false positives occurred at scales comparable to inter-cluster spacing. That is also the radius at which a neighborhood starts straddling two clusters, inflating `d̂` and ending the plateau.

> **Prediction: on the E4 clumpy manifold, `r_hi` falls below the inter-cluster spacing, and the plateau rule automatically excludes the radii that produced FP = 0.49.**

Falsifiable and cheap; the anchors exist. If it holds, the plateau rule *prevents* the A12 violation before the test rather than repairing it after. If it fails, the two scales are unrelated and §17 needs rework.

*Caveat to state in any writeup:* plateau collapse and FP inflation share a common cause (the clustering scale) rather than one independently diagnosing the other. For a decision *rule* this is acceptable — the boundary only has to land in the right place — but it is not evidence of an independent mechanism.

**V2 — policy comparison.** Fixed three-scale ladder vs. plateau-placed ladder on E4 and E1 cells. Report FP on honest controls, power on detectable anchors, query spend, abstention rate by code. Key question: does abstention concentrate on anchors where the fixed ladder was silently wrong?

**V3 — mixing-fraction sweep.** Fix separation at 1.5 component-sd (the E1 post-OLS regime, below the dip's floor); sweep `π` from 0.1 to 0.5. Compare dip / equal-variance LRT / separability guard. Two questions: does the LRT have power where the dip provably has none, and does it degrade more gracefully than trimming as `π → 0.5`?

**V4 — curvature sweep.** The test of §17. Impose known curvature, sweep magnitude, ask whether `r_hi` tracks the radius at which local-linear fitting measurably degrades. If plateau top and A11 margin move together, the equivalence holds; if they decouple, the plateau is a dimension diagnostic only.

---

# PART X — SCOPE AND LIMITS

## 29. How black-box is this?

**With respect to the model: fully.** Query access only — no weights, activations, gradients, or architecture.

**With respect to the data: not at all.** The probe policy requires an anchor set `A` drawn from the deployment distribution, and:

- `A` must be fixed **before** querying `f`, or §20's independence argument collapses.
- `A` must be locally dense enough for k-NN geometry to be meaningful; sparse regions abstain via `A0`.
- `A` must actually be the deployment distribution. If it is a proxy, the density filter certifies relevance to the proxy, not to deployment.

Accurate description: **query-only with respect to the model, sample-dependent with respect to the input space.**

## 30. What "if V1–V4 pass" actually buys

> Per-anchor, query-only detection of additive behavioral switches, with power roughly independent of ambient dimension, valid on non-uniform covariate distributions, at a data-driven radius selected without selection penalty — and with a principled account of where the method declines to answer.

The last clause is the one to lead with. An audit that abstains explicitly and says why is qualitatively different from one that returns clean because it could not see.

## 31. What it does not buy

**Structural limits** — not expected to yield to further engineering:

- **Parallel penalty (E3, θ=0).** `[LIMIT]` A penalty aligned with strong within-branch variation is invisible at every radius, in every dimension, under every projection. Proven, not conjectured.
- **Reach is not validity.** `[LIMIT]` The plateau says where assumptions hold, not where the gate lives. A gate outside the locally-flat window is not found — and the plateau rule makes this *more* likely, since it bounds radius from above. The spec buys validity partly by shrinking the search region.
- **Only gates near probed anchors are findable.** Gates keyed on rare-but-real inputs need those inputs in `A`.
- **`π ≈ 0.5`.** LTS breakdown at `h = 0.75m` is structural, and "majority branch" ceases to be defined.

**Open items** — no fix, or an untested one:

- Curvature at the A11 margin; local-quadratic trimming is the candidate. `[OPEN]`
- Heterogeneous intrinsic dimension — different clusters with genuinely different local `d` is a different diagnosis with a different fix (the Hidalgo line of work). `[OPEN]`
- Noise-induced plateau shift: `τ` is known, so if noise displaces the plateau's lower edge predictably, correct rather than avoid. `[OPEN]`
- Estimated non-isotropic whitening for vector outputs (from E3). `[OPEN]`
- **Stage-B certification.** Even a perfect detector returns *boundaries*; the poster's item 1 wants the *partition*. E2 narrows it — global labels come from level clustering at 0.997 accuracy, so certification reduces to marking **territory** (where between-anchor edges are probe-supported), not to label agreement. Still the largest open item. `[OPEN]`

## 32. Framing advice

Resist "complete detection," in writing and internally. A reviewer will find E3's `θ = 0` case in minutes. Naming it yourself as a proven limit reads as rigor; implying completeness reads as not having looked. **Completeness was never the achievable target; characterized incompleteness is** — and the abstention taxonomy is what delivers it.

---

# APPENDICES

## A. Glossary

| Term | Plain meaning |
|---|---|
| **ambient dimension `D`** | how many coordinates the input has |
| **intrinsic dimension `d`** | how many directions the data actually varies in; the rest are along for the ride |
| **manifold** | the lower-dimensional surface the real data lies on inside the bigger space |
| **tangent space** | the flat approximation to that surface at a point; the directions that locally matter |
| **isotropic** | same in every direction; a round cloud with no preferred orientation |
| **anisotropic** | stretched; more spread in some directions than others |
| **mixing fraction `π`** | the share of probe points landing on the far side of the hidden switch |
| **residuals** | what's left over after subtracting a fitted trend; ideally formless noise |
| **residualization** | fitting a trend and subtracting it, to expose what the trend was hiding |
| **trimming / LTS** | fitting while ignoring the worst-fitting points, so the fit commits to the majority |
| **breakdown point** | the fraction of contamination a robust method tolerates before it fails |
| **unimodal** | a density with one hump |
| **bimodal** | two humps — the signature of a hidden switch |
| **dip test** | a test whose null hypothesis is "one hump," assuming little else |
| **LRT** | likelihood ratio test: compare how well two models explain the data |
| **GLRT regularity failure** | the standard chi-squared calibration for an LRT breaks when the null sits at a parameter boundary |
| **bootstrap** | simulate the null many times by resampling, to calibrate a test empirically |
| **selection effect** | trying many analyses and reporting the best, which invalidates p-values |
| **Bonferroni** | divide the significance threshold by the number of tests run |
| **power** | probability of detecting a real effect |
| **false positive (FP)** | flagging a switch in a model that has none |
| **abstention** | declining to answer — "I can't see here," not "there's nothing here" |
| **A11** | assumption: the honest surface is approximately linear over the probe region |
| **A12** | assumption whose violation produced E4's clumpy-manifold false positives |

## B. Parameters

| Param | Default | Source |
|---|---|---|
| `k_max` | 200 | neighborhood size for GRIDE / PCA |
| `k_min` | 30 | below this, `d̂` variance unacceptable |
| `W_min` | 0.35 decades | plateau abstention threshold — **tune in V2** |
| `γ` | 0.75 | `d̂/D` above which reduction isn't worth it |
| `c` | 1.5 | density filter tolerance, multiples of local spacing |
| `ρ_min` | 0.5 | retention abstention threshold |
| `S` | 2 | ladder rungs inside the plateau |
| `h` | `0.75m` | LTS trim fraction (E1-validated) |
| `m` | 1000 | queries per rung |
| `B` | 300 | bootstrap replicates |
| `α` | 0.05 | Bonferroni over rungs actually used |
| `Δ` | 0.30 | gate penalty (simulation) |
| `τ` | 0.02 | observation noise (simulation) |

Unsourced defaults (`W_min`, `c`, `ρ_min`) are placeholders pending V2 and should be **registered before** the confirmatory run, not tuned on it.

## C. Questions for Laio

1. **Is there a principled plateau-detection criterion, or is it read by eye?** The overlapping-CI formalization in §23 is the obvious candidate; has it been done, or is there a reason it doesn't work?
2. **Does the plateau shift systematically under known noise?** `τ` is known here, so a predictable shift in the lower edge could be corrected rather than avoided.
3. **Heterogeneous intrinsic dimension.** Is the E4 clumpy case one manifold with a curvature problem, or genuinely different local `d` per cluster (Hidalgo)? Different diagnosis, different fix.
4. **The framing question.** Has anyone used the plateau as a **scale-selection device for a downstream estimator**, rather than as a route to reporting a single `d̂`? If the ID literature treats the curve as a means to one number and discards the rest, then "the plateau is the operating window for local methods" is a reusable idea well beyond this audit — and the strongest novelty claim in this whole line of work.

## D. Provenance summary

| Component | Status | Evidence |
|---|---|---|
| Fixed per-coordinate budget mandatory | `[VALIDATED]` | E1a, crossing formula to 3 decimals |
| Trimmed residualization, dimension-independent power | `[VALIDATED]` | E1, 0.74–0.76 flat in `D` |
| OLS absorbs ~64% of gap | `[DERIVED]` | closed form, matches E1 |
| Trimming restores validity on clumpy manifolds | `[VALIDATED]` | E4, FP 0.49 → 0.00 |
| Trimming unmasks trend-hidden gaps | `[VALIDATED]` | E4-uniform, 0.14 → 0.90 |
| Effect-size floor as merge rule | `[VALIDATED]` | E2, 0.64 → 0.75 |
| Count wide, estimate narrow | `[VALIDATED]` | E2 |
| Group-conditional estimand at anchor | `[VALIDATED]` | E3d, cosine 0.52 → 1.000 |
| Parallel-penalty invisibility | `[LIMIT]` | E3, θ=0 |
| Tangent-frame probing improves reach | `[DERIVED]` | from E1a, `d̂` for `D` |
| Density filter licenses deployment claim | `[PROPOSED]` | argument only |
| Plateau = A11 window | `[PROPOSED]` | **V4** |
| Plateau rule prevents A12 violation | `[PROPOSED]` | **V1** |
| Mixture LRT beats dip below the floor | `[PROPOSED]` | **V3** |
| Separability guard discriminates curvature | `[PROPOSED]` | V2/V4 |
| Adaptive radius is selection-penalty-free | `[DERIVED]` | independence argument, §20 |
