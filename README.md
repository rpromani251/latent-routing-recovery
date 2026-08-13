# detect-recover-interpret

Black-box detection and recovery of a hidden decision boundary inside a deployed model —
via on-manifold probing and the *modality* of the response distribution, not its spread.

Robert Romani, advised by Peng Luo, MIT 2026.

## The problem

A deployed model routes some inputs onto a penalized branch based on a protected
attribute (`f(x) = h(x) - penalty * 1[b(x) >= threshold]`), without disclosing the rule.
Can an auditor querying the model as a black box — no labels, no internals, no protected
attribute except for scoring — detect that routing exists, localize where it fires, and
recover the penalty's size?

## Headline finding

**Dispersion does not discriminate. Modality does.**

A hard routing boundary viewed through a probe of scale σ is observationally a smooth
transition of width ≈ σ — a stationary Gaussian process with lengthscale ℓ ≈ σ reproduces
the same *dispersion* curve. No statistic computed from the dispersion curve alone can
separate a routing boundary from ordinary smooth curvature; this is an identifiability
fact, not a bad choice of statistic. What differs is the *distribution* of probe
responses: a boundary splits them into two clumps separated by the penalty, a smooth
branch does not. **Hartigan's dip test on the response distribution is the
discriminator.**

On Seattle building-energy data (n = 3,581 buildings, a planted gate on tract
demographics): dip flag rate 0.0% on the honest model vs 22.8% on the gated model, power
0.74 on detectable anchors, penalty recovered to within 0.002 of the planted 0.30. See
[`docs/results_2026-07-28.md`](docs/results_2026-07-28.md) for the full evidence,
including three roles tested for the dispersion statistic (scale selection, anchor
screening, corroboration — only corroboration survives, and even there the recovered
effect size dominates it).

## Relationship to `geospatial-xai-attacks`

This project grew out of Phase 3 (SBM / behavioral-fingerprint regime recovery) of the
sibling [`geospatial-xai-attacks`](../geospatial-xai-attacks) repo. The findings above
substantially revise that phase's algorithm (`routing_audit_v2`, see
[`docs/routing_audit_v2.pdf`](docs/routing_audit_v2.pdf)) — dropping the dispersion
apparatus and the hierarchical null generator it required, promoting the dip test from
corroboration to primary, and replacing per-anchor multiplicity control (BH/BY) with a
spatial-randomization alignment test. This repo is the clean, current-method-only
implementation; the SBM/pEx thread and the original Phase 1–2 fooling-XAI work stay in
`geospatial-xai-attacks`.

## Entry points

- **Method note (start here):**
  [`docs/consolidated_draft_v3.pdf`](docs/consolidated_draft_v3.pdf) — the current method
  in one 8-page document: regime definition and identifiability ceiling, inputs/outputs,
  the full assumption set, algorithmic steps, what changed from v2 and why, the
  controlled simulation, and the baseline comparison.
- **Results log:** [`docs/results_2026-07-28.md`](docs/results_2026-07-28.md) — every
  empirical claim behind the design choices below, with numbers.
- **Current status and open problems:**
  [`docs/status_2026-07-29.md`](docs/status_2026-07-29.md).
- **Simulation + baselines:** [`sim/README_SIM.md`](sim/README_SIM.md).
- **Registered settings (incl. the A11 constraint):**
  [`configs/registered_settings.yaml`](configs/registered_settings.yaml).
- **Method derivation:** [`docs/derivation_guide.md`](docs/derivation_guide.md) (the why)
  and [`docs/routing_audit_v2.pdf`](docs/routing_audit_v2.pdf) (the algorithm note this
  repo's findings revise).
- **Data setup:** [`docs/data_dependencies.md`](docs/data_dependencies.md).
- **What moved to `legacy/` and why:**
  [`src/detect_recover_interpret/legacy/README.md`](src/detect_recover_interpret/legacy/README.md).

### Current method note (13 August)

- **[`docs/boundary_recovery_v5.pdf`](docs/boundary_recovery_v5.pdf)** — *Recovering a
  Hidden Routing Boundary from Query-Only Access.* The current note, organised around the
  object being estimated rather than around a detection statistic: the mixture's *weight*
  gives distance to the boundary, its *separating direction* gives the normal, and pooling
  local estimates across anchors yields one hyperplane with error bars. Supersedes
  `pipeline_v4` and the Stage 0/1 description in `probe_policy_spec`. Source:
  `boundary_recovery_v5.tex` (needs `fig_distance_estimator.png` alongside it).

- **[`docs/experiment_designs.pdf`](docs/experiment_designs.pdf)** — *pre-registration* for
  Experiment S (scaffold characterisation) and Experiment P (pooled recovery): designs,
  metrics, falsification conditions, and predicted outcomes stated **before** either is run.
  Extracts §8–§9 of the 31 July working document and updates the axes for what has been
  measured since.

### Superseded and supporting documents (29 July onward)

`consolidated_draft_v3` describes the method as of 29 July. Three later documents revise
it; v5 above consolidates them:

- **Probe geometry and scale selection:**
  [`docs/routing_audit_probe_geometry_consolidated.md`](docs/routing_audit_probe_geometry_consolidated.md)
  — the E1–E4 findings, the trimming operator in closed form, tangent-frame on-manifold
  probing, GRIDE-based scale selection, and the dip-vs-mixture-LRT choice. Every claim
  carries a provenance tag (`[VALIDATED]` / `[DERIVED]` / `[PROPOSED]` / `[LIMIT]`).
- **Stage 0/1 specification:** [`docs/probe_policy_spec.md`](docs/probe_policy_spec.md) —
  the geometry-driven probe policy the above implies, with its abstention taxonomy.
- **Detection-phase review (most recent, 31 July):**
  [`docs/dri_detection_phase_working_doc.md`](docs/dri_detection_phase_working_doc.md) —
  falsifies the plateau-equals-A11 equivalence, identifies the resonance band (honest FP
  0.136–0.169 at ~1 oscillation per probe ball), rejects the quadratic lack-of-fit repair
  as uncalibratable, adopts Δ-invariance over a 3-rung ladder, and specifies Experiments
  A, P and S.
- **Future-work experiments (E1–E4):**
  [`docs/future_work_experiment.pdf`](docs/future_work_experiment.pdf); code in
  `sim/ext*.py`, driver `sim/run_ext.py`.

> **Implementation gap.** Nothing in `src/` implements Stage 0 (GRIDE, plateau detection,
> tangent frame, density filter) or the equal-variance mixture LRT. The numbers in the
> detection-phase review come from a separate reimplementation that was not preserved.
> Per `boundary_recovery_v5` §15, Stage 0 is deliberately deferred below the critical path:
> geometry recovery is established first under known-valid local probes, and Stage 0 is the
> extension that removes that assumption. The density filter is kept regardless, because it
> scopes the claim rather than improving it.

## A11 — the binding precondition

The method assumes the honest model is smooth at **every probed scale**. The controlled
simulation showed this is sharp rather than cosmetic: dipping at scales above the honest
model's curvature lengthscale drives the false-positive rate to **0.33–0.81**, while
capping the ladder below it restores nominal error (**0.00–0.04**). Seattle satisfied the
condition with a factor of ~2 to spare (ℓ ≈ 2367 m vs a 1200 m ladder top), which is why
the published rates calibrate.

**Operational rule: cap the probe ladder below the honest model's lengthscale, and
re-measure ℓ before applying the method to a new model or dataset.** Note that an auditor
with black-box access alone cannot run this check directly — see the caveat in
`docs/results_2026-07-28.md` §4b.

Related specificity result from the same sweep: a *continuous* kink (one mechanism, sharp
derivative change, no routing) is **not** flagged — 0.050, exactly nominal. The dip is
specific to jumps, a narrower observational-equivalence class than v2 assumed.

## The current method, in one pass

1. **On-manifold probe** (kernel-weighted resample of real building locations) at 3–6
   log-spaced scales — chosen because the probe distribution itself must be locally
   smooth in the probe metric, or the data's own clustering reads as model multimodality
   (see the attribute-graph negative result in `legacy/attribute_graph.py`).
2. **Dip test at every scale**, Bonferroni over the scales actually dipped. No dispersion
   curve, no σ\* scale selection — naive scanning dominates at every query budget tested.
3. **K=2 Gaussian mixture** at every scale gives the recovered penalty Δ̂ and soft
   composition π̂ for free.
4. **Effect-size filter** on Δ̂ for a high-precision operating point.
5. **Spatial randomization** (permute the protected attribute across tracts, holding
   geometry fixed) for the alignment claim — valid under arbitrary spatial dependence,
   no independence assumption.

See `src/detect_recover_interpret/audit.py` for the implementation and `configs/main_audit.yaml`
for the exact settings behind the headline numbers.

## Reproduce

```bash
pip install -r requirements.txt
# populate data/raw/ and data/processed/ -- see docs/data_dependencies.md
bash scripts/run_all.sh
```

Outputs land in `results/` (per-building audit CSVs) and `results/figures/` (the current
figure set: `fig_poster_1_mechanism.png`, `fig_poster_2_map.png`,
`fig_poster_3_results.png`, `fig_coverage.png`, `fig_lengthscale.png`,
`fig_dip_discriminator.png`). Supporting experiments (query-budget reallocation, the
σ\*-value ablation, the spatial-randomization radius sweep) live in
`scripts/experiments/` and aren't run by `run_all.sh` by default — see the docstring in
each for usage.

## Controlled evaluation with known regimes

`sim/` builds 1-D and 2-D routing models whose regimes are known by construction, and
runs the same audit against them. Self-contained (no Seattle data, falls back to a
Monte-Carlo dip calibration if `diptest` is absent), seeded, and resumable.

| axis | finding |
|---|---|
| noise (τ_obs 0.005→0.2, Δ/τ 60×→1.5×) | power 0.98 → 0.89 → 0.57 → ~0; FP ≤ 0.004 throughout; the method **abstains** (0.79) rather than false-flags at extreme noise; Δ̂ holds 0.25–0.29 vs true 0.30 wherever flags exist |
| distance to boundary | power is effectively binary in the true mixing fraction: 1.00 above π≈0.10, 0.55–0.71 in 0.05–0.10, 0 below — minor-mode mass m·π is the binding constraint, reproducing the Seattle result in a controlled setting |
| smoothness (A11) | see above — the precondition is sharp and checkable |

Baseline comparison on the same 2-D construction (`sim/baseline_comparison.csv`), with
the graph-based approach run using the **actual SBM code** from `geospatial-xai-attacks`:

| method | existence test | partition acc (all / detectable) | Δ̂ err | queries/anchor |
|---|---|---|---|---|
| dip-scan (ours) | yes, calibrated | 0.80 / **0.93** | 0.03 | 6001 |
| global cluster + BIC | yes | 0.54 / 0.54 | 0.30 | 25 |
| residual outlier | yes | 0.51 / 0.52 | — | 25 |
| SPA profile-mean | none | 0.52 / 0.58 | 0.21 | 50 |
| plain SBM K=2 (their code) | none | 0.61 / — | — | 33 |

Global clustering does answer existence here (BIC picks K=2 gated, K=1 honest) but
partitions at chance. The SBM aligns only weakly (ARI 0.048) because fingerprint
similarity cancels level effects except across boundaries — intrinsic to the encoder,
not a tuning failure.

```bash
cd sim && python3 run_cells.py        # 1-D suite (parallel, resumable)
python3 sim2d_known_regimes.py        # 2-D
python3 sim_baselines.py              # simple baselines
python3 sim_sbm_baseline.py           # SBM (needs the sibling repo; DRI_GEOXAI_REPO)
python3 fig_sim_results.py            # figures
```

## Layout

- `sim/` — controlled known-regimes simulation, baselines, and the SBM comparison.
- `src/detect_recover_interpret/` — library code: honest model + gate (`model.py`), probe families
  (`probes.py`), the production audit protocol (`audit.py`), dispersion-curve utilities
  kept for validation/corroboration (`dispersion.py`), spatial-randomization utilities
  (`spatial_randomization.py`), conditional-coverage conformal check (`conformal.py`).
  `legacy/` holds superseded protocols and one documented negative result — nothing there
  is imported by `run_all.sh`.
- `scripts/` — runnable entry points, one per pipeline stage, plus `scripts/experiments/`
  for the supporting ablations cited in `docs/results_2026-07-28.md`.
- `configs/` — YAML settings for the model fit and each audit protocol (scales, probe
  counts, τ_obs, penalty, threshold, seed).
- `data/` — raw + processed (git-ignored, see `docs/data_dependencies.md`).
- `results/` — audit CSVs and figures (git-ignored).
- `docs/` — method notes (current and superseded), results log, probe-policy spec,
  detection-phase review, data setup.
