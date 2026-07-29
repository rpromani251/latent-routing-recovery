# routing-audit

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

- **Results (start here):** [`docs/results_2026-07-28.md`](docs/results_2026-07-28.md) —
  every empirical claim behind the design choices below, with numbers.
- **Method derivation:** [`docs/derivation_guide.md`](docs/derivation_guide.md) (the why)
  and [`docs/routing_audit_v2.pdf`](docs/routing_audit_v2.pdf) (the algorithm note this
  repo's findings revise).
- **Data setup:** [`docs/data_dependencies.md`](docs/data_dependencies.md).
- **What moved to `legacy/` and why:**
  [`src/routing_audit/legacy/README.md`](src/routing_audit/legacy/README.md).

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

See `src/routing_audit/audit.py` for the implementation and `configs/main_audit.yaml`
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

## Layout

- `src/routing_audit/` — library code: honest model + gate (`model.py`), probe families
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
- `docs/` — results, derivation, data setup.
