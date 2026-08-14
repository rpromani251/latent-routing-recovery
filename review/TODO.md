# TODO — the path to AISTATS, reordered

**Date:** 14 August 2026. Companion to `STATUS.md`.

**Ordering principle.** The ladder result (Experiment L) removed the scaling exponent
from any selection rule, and the calibrated LRT fires at 1.000 on resonant curvature
against 0.203 on the real target — the screen currently enriches its own confounder 5×.
The separability guard is the only remaining instrument that distinguishes a boundary
from curvature, it has never been run, and it is therefore **upstream** of
screen→select→recover. Everything below is sequenced off that fact.

**Clock.** Three hard dates now govern: MITEI poster print **Wed Sep 9, 11:59 PM**;
sponsor presentation with the paper in hand **Tue Sep 22**; AISTATS 2027 deadline TBD
(2026-cycle pattern: abstract ~24 Sep, paper ~1 Oct AoE). Scope freeze **Fri Sep 4**
(= poster content freeze). The date-by-date operating schedule is `PLAN_SEP22.md`;
this file stays the science ordering.

---

## 0. Now — Experiment G: does the separability guard separate a gate from curvature?

Cheap (the LRT/EM harness, calibrated threshold, and surface constructions all exist),
decisive either way, and both outcomes are useful before anything is built on top.

**Design essentials**
- Through the actual pipeline (probe → density filter → LTS → EM → calibrated LRT
  5.459), κ = 0.78 ladder, m = 800, τ = 0.02 — the Experiment A/L settings.
- Surfaces, core contrast first: honest-smooth · honest-resonant at wavelength ∈
  {0.5, 1.0, 1.5}·ball · gated at Δ/τ ∈ {1.5, 1.95, 2.5} · **gated + resonant
  coexistence** (the masking cell, 0.68 → 0.085) · **the P-2 long-wavelength surface**
  (see below). Second wave, if the guard survives the core contrast: quadratic
  curvature, GP ℓ ≈ σ, the kink control, heteroskedastic honest — cheap, the
  constructions exist, and together they become the paper's confound table.
- Guard: fit a linear classifier on probe coordinates z with EM responsibilities as
  labels, scored strictly **out of sample** (cross-fitted). Score three ways: held-out
  balanced accuracy, held-out AUROC, and held-out log-loss improvement over the
  no-structure model. Log the fitted normal and its angle to truth on gate cells — the
  guard's boundary lifted through the frame is the normal estimate, so the test doubles
  as free recovery validation.
- **Calibrate with a within-anchor permutation null**: permute the responsibility/
  residual assignment relative to z within each anchor and rerun the guard, ≥500
  permutations (this is Experiment P's null design reused). That yields an actual
  p-value for "spatial halfspace structure is present," instead of an arbitrary
  accuracy threshold.
- n ≥ 200 anchor-trials per cell (48–60 will not resolve conditional metrics).

**Metrics**
1. The separation itself: guard-statistic distributions, gate vs resonant, at matched
   LRT-fire rates. This is the headline.
2. Screen operating characteristic with the guard added — **pre-declare the operating
   question before looking at the sweep**: at 80–90% retention of actual gates, what
   fraction of the resonant confound passes? (Target: ≤0.05 at retention ≥0.9.)
3. Coexistence cell: does LRT + guard recover any of the 0.68 → 0.085 masking loss?
4. **Pooled level (P-2):** re-run the pooled estimator on the long-wavelength smooth
   surface with guard-weighted anchors — does the 100%-confident false boundary
   (coherence R ≈ 0.879) die?
5. Guard-normal angle vs truth on gates, by Δ/τ.

**Pre-register, including the failure conditions**
- Gate: near-perfect held-out separation at Δ/τ ≥ 1.95 (residuals were observed
  near-linearly-separable in z when a gate is present).
- Resonant ≈ 1 oscillation/ball: near chance (interleaved components).
- **The sharpest risk is the long-wavelength cell.** A slowly-curving surface is locally
  plane-like; if its spurious components are also spatially separable, the guard passes
  the exact confounder that beat pooling in P-2. State this before running.
- Known trap (31 July doc): a more sensitive detector should false-alarm harder — the
  guard is being asked to fix precisely the failure the LRT's sensitivity created.

**Decision rule**
- **Guard separates** (incl. the pooled cell) → it goes inside the selection rule;
  proceed to item 1 with the screen designed around LRT + shell-window π̂ + guard.
- **Guard fails per-anchor** → no confounder defence exists at any level; the paper
  reframes to estimator + identifiability boundary (bounded-curvature H₀ as *scope*,
  stated as an identifiability result, not an FP nuisance), and item 1's screen reduces
  to placement + LRT with declared curvature scope.
- **Guard works per-anchor but fails on long-wavelength** → defence covers sub-ball-scale
  curvature only; declare the exposure window explicitly (the resonance-band framing
  already exists) and proceed with that scope.

Deliverable: `RESULTS_guard_<date>.md` + go/no-go for the screen design.

**Already done — cite, don't redo.** The second audit's proposed steps 2–4 largely ran
on 13–14 Aug: normal-estimator validation *including* the true-labels vs
estimated-responsibilities ablation (unbiased; misassignment costs variance —
`b779879`); Experiment P including the smooth-curvature pooling cell (the P-2
falsification); t̂ interval coverage (0.00 as specified → 0.945–0.998 repaired). The
live remainders are: n̂/ĉ coverage for whatever the paper reports (item 4), the guard
at the pooled level (item 0, metric 4), and estimated-frame sensitivity (item 5).

---

## 1. screen→select→recover, end to end (the binding item)

With the guard verdict in hand: run the full chain — screen anchors, select survivors,
pool, recover (n̂, ĉ, t̂) with the repaired estimator (project boundary points onto the
pooled normal; distance from the crossing law, not the LDA midpoint) and the equivalence
bound on the angle. Report the whole operating characteristic: what fraction of
flagged anchors are confounders, what survives selection, what the recovered rule and
its interval are, against ground truth. This is the paper's central table.

## 2. Boundary-seeking anchor placement (prerequisite, not a nicety)

P: 0.96 with by-design placement vs 0.00 random. Even a simple two-pass rule (cheap
wide detection pass → concentrate anchors near hits) needs to exist and be validated,
with the budget split and the selection effect on downstream inference accounted for
(the Stage-0-reads-only-A independence argument does **not** cover placement informed by
responses — this needs its own correction or a split-budget design).

## 3. Housing scaffold end-to-end (rung 2)

First full-pipeline run against a model with a genuine internal router the audit cannot
see. Every current power figure is against planted gates; this is the milestone that
converts "estimator works in a clean room" into "audit ran against a model." At
Δ/τ = 1.6–1.95, expect the honest output to be the hyperplane + angle bound, not a
feature name — that is the scope sentence, demonstrated.

## 4. Paper track (parallel, start now)

- **Done in draft (14 Aug): `boundary_recovery_v6`** applies the change-list in full —
  calibration, dip floor, min-mass deletion, Stage 5 retirements, pooled-offset repair,
  equivalence-bound Stage 7, measured coverage, the ladder result — with retired claims
  kept visible under `[RETIRED]` tags. **Review it before drafting paper text from it**;
  v5 stays in `docs/` as the paper trail.
- AISTATS skeleton (8 pp): contribution phrasing verbatim from
  `related_work_positioning.md` (latent membership · calibrated uncertainty · bounded
  curvature); the limits register is essentially a finished limitations section; the
  scope sentence (13° at Δ/τ = 1.5, N = 200) goes in the abstract-adjacent claims, not
  buried.
- Report pre-registered predictions as hits and misses, including the retractions —
  this is a strength, present it as one.
- Consolidate the confound/null suite into **one paper table, one row per assumption or
  falsifier**: hard gate, quadratic, resonant, GP ℓ ≈ σ, kink, heteroskedastic, clumpy
  geometry, π ≈ 0.5, parallel penalty, multi-boundary ball. Baselines answer specific
  questions (what a calibrated existence test adds; what geometry adds over Fisher
  pooling) — no leaderboard.
- **Coverage must back every interval the paper prints.** The t̂ interval and the angle
  bound have measured coverage; if the paper reports intervals on n̂ or ĉ directly,
  measure their coverage first or scope the reported objects to what is measured.
- Watch for the AISTATS 2027 CFP (aistats.org); confirm page limits/dates the day it
  posts.
- Decide at scope freeze: Seattle/routing-audit material as motivation + appendix, or
  held for a separate paper.

## 5. Scope freeze (~1 Sep) — the explicit defer list

Deferred, stated plainly in the paper rather than half-done: Stage 0 implementation
(results conditional on supplied geometry — say so); discrete-response Stage 3 and the
Slack rung-3 test (the genuine held-out target; post-submission); estimated-frame
sensitivity (if cheap, measure the certified-bound inflation on one cell before
freezing); π ≈ 0.5 recovery; general-K. (n̂/ĉ coverage is *not* deferrable if those
intervals are reported — see item 4.) On Stage 0, the honest line for the paper:
deferring it is defensible; pretending it is validated is not — state every result as
conditional on a valid local probe geometry.

## 6. Hygiene (half a day, this week)

- Fix hardcoded `/tmp` in `exp_a_invariance.py`, `exp_p_pooling.py` (use the existing
  `paths.py` convention); verify fresh-checkout reproduction of one experiment.
- **Do not re-anchor the ladder** — the `plain_status.md` item is superseded by
  Experiment L (keep κ = 0.78; ratio ≲1.55 for a second estimable rung is untested and
  optional).
- README top: replace the July 28 dip-primary headline with the v5 estimand framing and
  a current status table; move `COMMIT_MSG*.txt` out of `sim/`.
- Resolve the never-made `geospatial-xai-attacks` staging (commit as archive or delete —
  DRI owns the thread since `fd09b19`).
- Freeze the `3 Stage Audit` vault folder with a `POINTER.md` to the repo; stop editing
  vault copies.
- Repo restructure per `REPO_REVIEW.md` — hygiene now, structure **after** the guard and
  screen experiments settle what the library API is.

---

## Timeline

The operating schedule with the poster gates is **`PLAN_SEP22.md`** — the single source
of truth for dates. Condensed:

| Window | Work |
|---|---|
| Aug 14–21 | Experiment G (guard) · hygiene · v6 review — **guard verdict Aug 21** |
| Aug 24–28 | screen→select→recover (guard verdict inside the rule) · placement rule · paper skeleton · poster story lock Aug 28 |
| Aug 31 – Sep 4 | Housing end-to-end attempt · poster draft to Peng · **content + scope freeze Sep 4** |
| Sep 7–9 | Poster finalize · **submit Sep 8** · print deadline Sep 9, 11:59 PM |
| Sep 9–18 | Paper sprint · last result in Sep 15 · red-team pass Sep 16–17 · **paper freeze Sep 18** |
| Sep 21–22 | Copies + QR link + pitch rehearsal · **sponsor day Sep 22, paper in hand** |
| Sep 23 – Oct 1 | AISTATS-ify the frozen paper: anonymize, format, checklist (deadline est.) |

If the guard fails and the paper reframes, the same timeline holds — items 1–3 shrink
(screen reduces to placement + LRT with declared scope) and the writing window grows,
which the identifiability-boundary framing will need.
