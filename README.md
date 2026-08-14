# latent-routing-recovery

Recovering a hidden routing boundary — its **distance**, **orientation**, **offset** and
**threshold**, with calibrated uncertainty — from query-only access to a deployed model,
where route membership is never observed.

Robert Romani, advised by Peng Luo, MIT 2026.

## Start here

| | |
|---|---|
| **The method** | [`docs/boundary_recovery_v6.pdf`](docs/boundary_recovery_v6.pdf) — current method note |
| **What changed, and why** | [`docs/method_revisions_2026-08-15.md`](docs/method_revisions_2026-08-15.md) — every v5 claim the 14–15 August runs moved |
| **Plain-language status** | [`docs/plain_status.md`](docs/plain_status.md) — what is confirmed, what is theory, what is impossible |
| **Positioning** | [`docs/related_work_positioning.md`](docs/related_work_positioning.md) — GeoDA, active halfspace learning, and the distinction |

**Results notes**, newest first:

- [`sim/RESULTS_ladder_2026-08-15.md`](sim/RESULTS_ladder_2026-08-15.md) — the ladder cannot be re-anchored; an Experiment A result retracted
- [`sim/RESULTS_axis_that_2026-08-14.md`](sim/RESULTS_axis_that_2026-08-14.md) — axis dominance, t̂, and the interval that covers at 0.00
- [`sim/RESULTS_bootstrap_2026-08-14.md`](sim/RESULTS_bootstrap_2026-08-14.md) — the full-pipeline bootstrap is both wrong and unnecessary
- [`docs/experimental_report.pdf`](docs/experimental_report.pdf) — the distance, orientation and scaffold experiments (13 August)
- [`docs/experiment_designs.pdf`](docs/experiment_designs.pdf) — pre-registration and outcomes for Experiments S and P

## The problem

A deployed model routes some inputs onto a penalised branch —
`f(x) = h(x) − Δ·1[wᵀx > b]` — without disclosing the rule. An auditor has query access
only: no weights, no gradients, no routing metadata, and **no observation of which branch
any query took.**

The claim we want to end up making is not *"something looks suspicious here."* It is:

> *The system routes on coordinate i at threshold t̂ = 4.98 ± 0.07, recovered from N
> anchors under query-only access, p < 0.05 against a permutation null.*

That is an inverse problem, not a test.

## The idea

Probe around a real input (an **anchor**). If the anchor sits near the hidden boundary,
some probes land on the far side and come back shifted by the penalty, so the response
distribution splits into two clumps rather than one spread. Then:

- **the mixture's weight identifies the distance** — d̂ = −σΦ⁻¹(π̂), the crossing law read
  backwards;
- **the mixture's spatial responsibilities identify the orientation** — the separating
  direction of a discriminant fitted on (probe coordinates, responsibilities);
- **pooling across anchors** turns noisy local estimates into one hyperplane with an
  interval.

Membership is latent throughout — that is what separates this from boundary estimation
with observable labels (see the positioning note).

## What is validated

| | |
|---|---|
| Distance estimator d̂ = −σΦ⁻¹(π̂) | r = 0.996 against truth, median relative error −0.9%, flat across a 10× noise sweep |
| Orientation estimator | no detectable bias in any of 20 cells — error is pure variance, so it pools away |
| Pooling | geometric > thresholded > statistic-only, with the largest gap at Δ/τ = 1.5 |
| Trimmed residualisation | honest false positives 0.49 → 0.00 on clumpy data |
| Stage-3 threshold | one calibrated number (5.46 at m = 800); honest size 0.028–0.052 across the whole ladder sweep |
| Interval on t̂ | 0.945–0.998 coverage **after** the offset repair — 0.00 before it |
| Dispersion cannot discriminate a gate from curvature | identifiability, not a bad statistic |

## What is not

- **Stage 0 does not exist** — automatic radius and frame selection is fully specified with
  zero implementation. Every result is conditional on a supplied probe geometry.
- **The separability guard has never been tested**, and it is now load-bearing: curvature
  fires the calibrated test at up to 1.000.
- **Anchor placement is by design throughout.** Geometric pooling is dead under random
  placement (0.96 → 0.00), so `screen → select → recover` is the binding missing algorithm.
- **Nothing has been run end-to-end** against any scaffold.

See [`docs/plain_status.md`](docs/plain_status.md) for the full map, including the
four-rung realism ladder for target selection.

## Layout

- `sim/` — the research code: experiments (`exp_*.py`), diagnostics (`diag_*.py`),
  figures (`fig_*.py`), result rows (`*_rows.csv`), results notes (`RESULTS_*.md`).
- `src/detect_recover_interpret/` — **the July protocol**, superseded. It implements
  multi-scale dip + plain 2-GMM, not the current estimator. Factoring the current
  primitives into the package is an open cleanup item.
- `docs/` — method notes (current and superseded), positioning, status.
- `data/`, `results/` — git-ignored; see `docs/data_dependencies.md`.

## Reproduce

Each experiment is seeded and resumable via its `_parts_*/` cache.

```bash
pip install -r requirements.txt
cd sim
python3 exp_bootstrap_calibration.py nulls    # then: anchors, boot
python3 exp_axis_dominance.py
python3 exp_that_coverage.py
python3 exp_ladder_anchor.py
python3 verify_bootstrap.py                   # 10 checks, exits 0
python3 fig_bootstrap.py ; python3 fig_axis_that.py
```

## A note on the supplement

This is the working repository and it carries author names, PDF metadata, git history and
commit messages. **It is not the submission supplement.** Any double-blind submission needs
a separate clean export — no `.git`, no names, no metadata, no session links.
