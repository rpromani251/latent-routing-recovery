# Experiment L — the ladder cannot be re-anchored, and Experiment A's π_top = 0.35 result is retracted

*Run 15 August 2026. This was meant to be an hour of bookkeeping before the screening
experiment. It falsified the proposal it was testing, and retracted a positive result from
Experiment A.*

## What was proposed, and why it was wrong

I argued that the ladder points the wrong way: with the top rung anchored at the
orientation-useful shell (d/σ = 1.28), the deeper rungs land at d/σ = 2.56 and 5.13 where
the gate is unreachable, so the r⁰-versus-r² exponent cannot be fitted. The proposed fix
was to widen the ladder — parameterised by κ = σ_top / d, moving from the current 0.78 up
to ~2.6 so the *deepest* rung sits on the shell.

**Every axis gets worse.** Mean estimable rungs per anchor, by the calibrated test:

| κ | d/σ top | d/σ deep | gated 1.95 | gated 2.5 | resonant 1.0 L | honest |
|---|---|---|---|---|---|---|
| **0.78** *(current)* | 1.28 | 5.13 | 0.30 | **0.73** | 0.89 | 0.10 |
| 1.3 | 0.77 | 3.08 | 0.36 | 0.86 | 1.28 | 0.13 |
| 2.0 | 0.50 | 2.00 | 0.28 | 0.78 | 2.02 | 0.14 |
| 2.6 | 0.38 | 1.54 | 0.40 | 1.03 | 2.21 | 0.14 |
| 3.5 | 0.29 | 1.14 | 0.29 | 0.86 | **3.00** | 0.14 |
| 5.0 | 0.20 | 0.80 | 0.26 | 0.53 | **3.00** | 0.14 |

The gate never gets more than ~1 estimable rung at any κ. The **confounder** gets all
three. And detection power at the top rung falls monotonically — gated Δ/τ = 2.5 goes
0.636 → 0.148 — while every resonant surface sits at **1.000** for κ ≥ 1.3.

### The geometric reason, which should have been derived first

The estimable window in d/σ is bounded on both sides:

- **below** by LTS breakdown — π ≳ 0.25 leaves no majority branch to trim toward, so
  d/σ ≳ 0.67;
- **above** by minimum detectable minority mass — π ≳ 0.05, so d/σ ≲ 1.64.

That window spans a factor of **2.4**. A three-rung geometric ladder of ratio 2 spans a
factor of **4**. **The ladder does not fit inside the window at any position.** At most two
rungs can be simultaneously estimable, and only at a ratio below ~1.55.

This is a structural constraint on the multi-scale design, not a tuning problem. It says
the 3-rung ladder was never going to deliver a three-point regression at a gate anchor.

## The exponent discriminator does not survive widening either

Median fitted α, with the fraction of anchors where it is fittable at all:

| κ | gated 2.5 | resonant 0.5 L | resonant 1.0 L | resonant 1.5 L |
|---|---|---|---|---|
| 0.78 | +0.19 (7%) | +0.13 (98%) | +0.56 (5%) | +0.65 (74%) |
| 2.0 | +0.02 (16%) | 0.00 (100%) | +0.01 (100%) | +0.09 (100%) |
| 2.6 | −0.07 (24%) | 0.00 (100%) | 0.00 (100%) | +0.01 (100%) |

Two things go wrong at once.

**The exponent is available where it is not needed and missing where it is.** At every κ it
is fittable on ~100% of curvature anchors and 7–24% of gate anchors. A discriminator that
cannot be computed at the anchors you want to keep is not a filter.

**And it stops discriminating.** Theory says r⁰ for a step against r² for curvature, but the
resonant surfaces' α collapses to ~0.00 once κ ≥ 2.0. The r² law only holds while the probe
ball is *small relative to the curvature wavelength*; a wide ball spanning many periods sees
the sine's fixed full amplitude, which is scale-free — indistinguishable from a jump.
Widening the ladder destroys the very signature it was widened to measure.

## The retraction

Commit `125555a` reports that at π_top = 0.35 all three rungs are estimable and the r⁰
signature is "recovered exactly". Reproducing that cell (κ = 2.6) under both estimability
rules, 120 anchors each:

| surface | rule | rungs | α fittable | median α |
|---|---|---|---|---|
| gated Δ/τ = 2.5 | minimum-mass | 2.89 | 100% | **−0.160** |
| | calibrated | 0.93 | 17% | −0.106 |
| **honest — no gate anywhere** | minimum-mass | **3.00** | **100%** | **−0.052** |
| | calibrated | 0.15 | 0% | — |
| resonant 1.0 L | minimum-mass | 2.75 | 100% | +0.295 |
| | calibrated | 2.25 | 100% | +0.011 |

**An honest surface with no gate reproduces the result exactly**: three "estimable" rungs
and α ≈ 0, the r⁰ signature of a step. It is *closer* to zero than the real gate's −0.160.

The mechanism is the one identified on 14 August. The minimum-mass rule passes ~99% of pure
noise, so at rungs the gate never reaches, EM splits noise and returns a Δ̂ that scales with
the residual sd τ — which is **independent of the probe radius**, hence α = 0. The "exact
recovery of r⁰" was measuring noise's scale-freedom, not a step's.

**Experiment A's π_top = 0.35 conclusion should be withdrawn.** It is not evidence that the
3-rung rule works there; it is a second symptom of the estimability gate that yesterday's
run showed is not a gate.

## What this changes for the screening design

The screen I proposed was: calibrated LRT fires **and** π̂ in the shell window **and** the
across-rung exponent is consistent with r⁰. **The third term has to be dropped** — it is
unfittable at 76–83% of gate anchors and fittable at 100% of curvature anchors, so
including it would reject the signal and keep the confounder.

That leaves the screen with no multi-scale defence against curvature, and the enrichment
problem stands unchanged: the calibrated LRT fires at 1.000 on resonant curvature against
0.203 on the real target, so screening on it selects confounders 5× more readily than gates.

**So the separability guard moves onto the critical path for the screening experiment, not
after it.** It is the only remaining instrument that distinguishes a boundary from
curvature — a real gate's mixture components are linearly separable *in probe space*,
curvature's are spatially interleaved — and it has never been tested. It has to be inside
the selection rule.

## What to keep

- **The current ladder orientation (κ = 0.78) is the best of those tested** for the gate.
  Do not re-anchor it. Its deeper rungs are not wasted so much as structurally unavailable;
  the window is too narrow for three rungs at ratio 2.
- If a second estimable rung is wanted, the ladder ratio must drop to ≲1.55, not the
  position change. Untested.
- The honest surface's fire rate stays at 0.03–0.05 across every κ, so the calibrated
  threshold from 14 August holds up under this whole sweep.

## Reproduction

```
python3 exp_ladder_anchor.py     # ~11 min on 2 cores, 36 cells x 250 anchors x 3 rungs
```
Outputs `ladder_anchor_rows.csv` (9,000 rows). Resumable via `_parts_ladder/`.
Settings match Experiments A and B: τ = 0.02, m = 800, calibrated LRT threshold 5.459,
resonant wavelengths held fixed in absolute units while the ladder moves.
