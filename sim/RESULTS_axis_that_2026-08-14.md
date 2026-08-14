# Experiment T — axis dominance, t̂, and the interval

*Run 14 August 2026. Step 4 of the `boundary_recovery_v5` critical path, plus the
interval-coverage deliverable of §14 (*The obligation
the reframe creates*), which arrived early because it had to.*

Pooling returns a hyperplane. The claim an audit needs is *"it routes on feature i above
t̂ = 4.98 ± 0.07."* This measures the conversion.

**Headline: the conversion does not work as specified, in two independent ways, and both
are repairable.** The dominance test claims a single-coordinate rule on gates that have
none, and the interval on t̂ covers at **0.00**, not 0.95.

---

## Summary against pre-stated predictions

| # | Prediction | Outcome |
|---|---|---|
| T-1 | The permutation calibration is structurally broken and near-powerless | **MECHANISM CONFIRMED, consequence milder.** The null's q95 is a near-constant 0.99 exactly as derived, so the rule reduces to "orientation error below ~5–8°". No power at Δ/τ = 1.0 (0.05–0.09), marginal at 1.5 (0.31 → 0.98 in N), full at ≥ 2.5. |
| T-2 | Sharp usability boundary between Δ/τ = 1.5 and 1.0 | **CONFIRMED, sharply.** The certifiable off-axis angle never falls below **56°** at Δ/τ = 1.0 even at N = 200; it reaches 13.5° at 1.5 and 3.4° at 2.5. |
| T-3 | t̂ inherits both error sources, so its error exceeds the offset error alone | **CONFIRMED, and worse than predicted.** The bias is *multiplicative*: −1.10 at Δ/τ = 1.5, which is 22% of the true threshold. |
| T-4 | Dominance fires on oblique gates | **CONFIRMED.** False-claim rate 0.20–0.79 for the specified rule, and ~1.00 for the repair I proposed. |

---

## 1. Three rules, and the first two are unusable

At Δ/τ = 2.5, N = 100. θ is the true gate's tilt off a coordinate axis: at θ = 0 a
single-coordinate rule exists, and **at every θ > 0 there is no such truth, so any fire
is a wrong claim, not a weak one.**

| θ | permutation *(Stage 7 as written)* | bootstrap stability *(my proposed repair)* | equivalence, 10° |
|---|---|---|---|
| **0°** | **1.000** | **1.000** | **1.000** |
| 10° | 0.925 | 1.000 | 0.000 |
| 20° | 0.789 | 1.000 | 0.000 |
| 30° | 0.376 | 1.000 | 0.000 |
| 45° | 0.401 | 1.000 | 0.000 |
| 90° | 0.000 | 0.953 | 0.000 |

**The permutation rule is over-permissive.** At 20° off-axis — where a third of the
routing direction lies in other features — it names a single feature **79%** of the time.

**The bootstrap-stability rule is far worse, and it was my idea.** It fires at ~1.000 at
every tilt including a fully oblique gate. The reason is instructive: it asks *"would I
name the same feature again on a re-draw?"* and the answer is yes — the argmax of a
well-determined oblique normal is **stably** the largest of several comparable
components. **Stability is not validity. A stably wrong answer is still wrong**, and this
rule cannot tell the two apart even in principle.

### The reframing that fixes it

Axis-dominance is an **equivalence** claim, not a significance claim. Stage 7 asks
whether `max|n̂ᵢ|` beats a no-signal null — but a normal can beat that null by being
merely *well determined* while pointing anywhere. What the audit needs is that the
off-axis part is **provably small**, which puts the burden the other way round.

So: bootstrap the anchors, form a one-sided **upper confidence bound on the angle between
n̂ and the candidate axis**, and claim a single-coordinate rule only if that bound is
below a stated tolerance. The tolerance becomes an explicit, reportable scientific choice
— *"the routing direction is within 10° of feature i"* — instead of a hidden one.

**The bound is an honest estimate of the tilt.** At Δ/τ = 5.0, N = 100:

| true tilt | 0° | 10° | 20° | 30° | 45° | 90°\* |
|---|---|---|---|---|---|---|
| **certified upper bound** | **0.87°** | **10.14°** | **20.13°** | **30.09°** | **45.12°** | 60.97° |

\* a generic direction in ℝ²⁰ is at most ~61° from its nearest axis, so 61° is the
saturation value, not a miss.

That is not just a decision rule — it is a **better deliverable than the binary claim**.
Report the bound itself.

## 2. The interval on t̂ covers at 0.00

Coverage of the nominal 95% bootstrap interval, planted threshold T = 5.0, θ = 0:

| Δ/τ | N | **as specified** — bias / coverage / width | **repaired** — bias / coverage / width |
|---|---|---|---|
| 1.5 | 25 | −0.970 / 0.232 / 2.054 | −0.002 / **0.970** / 1.817 |
| 1.5 | 100 | −1.129 / **0.000** / 0.939 | +0.010 / **0.988** / 0.230 |
| 1.5 | 200 | −1.145 / **0.000** / 0.691 | +0.008 / **0.998** / 0.127 |
| 2.5 | 100 | −0.135 / 0.065 / 0.164 | −0.001 / **0.945** / 0.051 |
| 2.5 | 200 | −0.135 / **0.000** / 0.118 | −0.001 / **0.970** / 0.034 |
| 5.0 | 200 | −0.007 / 0.932 / 0.042 | −0.000 / **0.995** / 0.005 |

**Coverage gets worse as N grows.** That is the signature of bias plus a shrinking
interval: the interval tightens around a displaced centre. The specified estimator never
reaches nominal coverage at *any* signal-to-noise — even at Δ/τ = 5.0 it plateaus at 0.93.
(The dominance grid's independent estimate of the same quantity ran lower still, 0.89 →
0.64 over the same N range, on different frame draws; the two bracket it, and neither
reaches 0.95.)

The repair is **not a trade-off**: it is simultaneously unbiased and about **5× narrower**
(0.127 against 0.691 at Δ/τ = 1.5, N = 200).

## 3. Where the bias comes from — two independent defects

**(1) Origin amplification — a multiplicative bias, and the dominant one.**
The per-anchor offset `c_a = ν_a · t_a + t0_a` uses *that anchor's own* normal. Writing
`t_a = (T ± d)·ν_true + along·tangent` and letting φ_a be the anchor's orientation error:

```
ν_a · t_a  =  (T ± d)·cos(φ_a)  +  along·sin(φ_a)
```

The second term averages away. The first does not — **E[cos φ] < 1, so the pooled offset
is attenuated toward the coordinate origin by a factor E[cos φ]**, and the absolute bias
grows with T. Measured bias/T against the predicted E[cos φ] − 1:

| Δ/τ | E[cos φ] − 1 | T = 1 | T = 5 | T = 20 |
|---|---|---|---|---|
| 1.5 | −0.259 | −0.302 | −0.287 | −0.196 |
| 2.5 | −0.056 | −0.012 | −0.028 | −0.024 |
| 5.0 | −0.002 | +0.006 | −0.004 | −0.003 |

**Experiment P measured the offset at `c_true = 0`, where a multiplicative bias is
invisible by construction** — zero times anything is zero. That is the design blind spot
that let this through, and it is worth generalising: *an estimator validated only at the
origin has not been validated for a threshold.*

**(2) The LDA midpoint — an additive bias.**
`t0` is the midpoint between the two projected class means. For a half-space cut of a
Gaussian at distance d those means are −σφ/Φ and +σφ/(1−Φ), and their midpoint is **not**
d unless the split is balanced. At π = 0.10 it sits at **0.780 σ** against a true
**1.282 σ** — an error of −0.50 σ, toward the anchor. It largely cancels between anchors
on opposite sides of the gate, so it costs variance rather than bias under two-sided
placement, but it is the residual the second repair removes.

### The repair, in two parts

- **Project each anchor's estimated boundary *point* onto the *pooled* normal**, rather
  than averaging per-anchor offsets. Per-anchor rotation error is then no longer
  multiplied by the anchor's distance from the origin.
- **Take the anchor-to-boundary distance from the validated crossing law**
  `d̂ = −σ Φ⁻¹(π̂)` rather than the LDA midpoint — direction from separability (validated
  unbiased), distance from the crossing law (validated r = 0.996).

**v5 §11 — Stage 5(c) — already contains both routes to the offset**, but describes them as agreeing
"by construction under a correctly specified model" and proposes their comparison as a
*specification diagnostic*. They do not agree: they differ systematically whenever
π ≠ 0.5, and the classifier route is the biased one. **The offset-agreement check is
detecting a real defect in one of its own arms.**

## 4. Where the method can and cannot name a feature

Certifiable upper bound on the off-axis angle, θ = 0 (a perfectly axis-aligned gate):

| Δ/τ | N = 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| 1.0 | 69.5° | 68.9° | 63.6° | **56.0°** |
| 1.5 | 43.8° | 30.4° | 20.9° | **13.5°** |
| 2.5 | 6.7° | 4.9° | 3.4° | **2.4°** |
| 5.0 | 1.7° | 1.2° | 0.9° | **0.6°** |

Read operationally:

- **Δ/τ = 1.0** — no feature can ever be named. The bound is barely below the 61°
  saturation value at N = 200. Report the hyperplane, or nothing.
- **Δ/τ = 1.5** — *the real scaffold's regime.* 200 anchors certify only ~13°. A
  single-coordinate claim at 10° tolerance fires at **0.217**. The honest output here is
  the bound, not a feature name.
- **Δ/τ ≥ 2.5** — 25 anchors certify better than 7°, and the target sentence is available.

## Reproduction

| what | command | output |
|---|---|---|
| the dominance grid (6 tilts × 4 SNR × 5 frames) | `python3 exp_axis_dominance.py` | `axis_dominance_rows.csv` (480) |
| t̂ coverage, specified vs repaired | `python3 exp_that_coverage.py` | `that_coverage_rows.csv` |
| the offset bias, with the T sweep | `python3 diag_offset_bias.py` | `offset_bias_rows.csv` |
| figure | `python3 fig_axis_that.py` | `fig_axis_that.png` |

Seeded and resumable (`_parts_axis2/`, `_parts_tcov/`). Ambient D = 20, intrinsic d = 2,
frame supplied, by-design placement in the π = 0.10 shell. ~2 h on 2 cores.

## What this did not establish

- **Anchor placement is by design throughout.** Experiment P showed geometric pooling is
  dead under random placement (0.96 against 0.00), so every number here is conditional on
  a boundary-seeking placement rule that does not exist.
- **The frame is supplied and exact**, so the lift contributes no error. A frame estimated
  by local PCA will not contain the gate normal exactly, and the off-frame component is
  unrecoverable — which will inflate the certified bound by an amount not measured here.
- **The tolerance is a free parameter.** 10° is a placeholder; what a defensible audit
  needs is a tolerance argued from the deployment, not from this simulation.
- **Coverage is measured for t̂ only**, at θ = 0, under Gaussian noise, with a single
  planted boundary. Coverage for n̂ and ĉ as objects in their own right is still unmeasured.
- **Estimator C assumes π̂ is a genuine crossing fraction**, which is conditional on
  detection — the caveat the distance estimator already carries.
