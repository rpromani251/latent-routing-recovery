# Legacy

Superseded protocols and one negative result, kept for provenance rather than deleted.
Nothing in this package is imported by `scripts/run_all.sh` or the current pipeline in
`src/routing_audit/`. Each module below is explained in more depth in
[`docs/results_2026-07-28.md`](../../../docs/results_2026-07-28.md).

| module | what it was | why it's here, not in the current pipeline |
|---|---|---|
| `stage_a_full.py` | The original per-anchor Stage A driver: dispersion curve → σ\* selection → single dip at σ\*. | Replaced by the naive multi-scale scan in `audit.py` (RESULTS S6c: σ\* is a *worse* scale-selection rule than picking a scale at random). |
| `null_generator.py`, `null_grid.py` | The hierarchical GP null generator and its precomputed-grid speedup, used to calibrate the dispersion statistic (`T_scale`). | Dropped along with the dispersion apparatus it calibrated (RESULTS S7, "Drop the null generator") — worst-case-null vs. power tradeoffs never cleared a usable operating point. |
| `toy_calibrated_statistic.py` | Toy suite comparing `R_log` and the null-generator-calibrated `T_scale` against literal draws from the registered null class. | Depends on the null generator above; superseded by `stage_a.dispersion_curve` + the dip toy (`scripts/run_dip_discriminator_toy.py`), which is simpler and matches production. |
| `seattle_audit_v1.py` | The first Seattle run: σ\*-based single-scale dip, scored against distance-to-boundary. | Superseded by `audit.py`'s naive multi-scale scan (RESULTS S6d), which also corrects the ground truth from "distance to boundary" to "does the probe neighbourhood actually contain a crossing." |
| `probe_family_comparison.py` | Ambient-vs-on-manifold probe comparison under the σ\*-based v1 protocol. | The *finding* (on-manifold should be primary) still holds and is cited in `docs/results_2026-07-28.md` S6b/S7 — but the protocol it was measured under is superseded, and the current pipeline hasn't been re-run with an ambient variant (`audit.py` is on-manifold only). |
| `control_curvature.py` | Negative control: can `R_log` / `T_iso` tell a routing boundary from smooth curvature? | Subsumed by the modality-vs-dispersion finding — dispersion statistics are demoted entirely, not just on this control. |
| `diag_model_class.py` | Checks whether a gradient-boosted honest model sits inside the smooth null class. | Runs against the *original* Phase 1/2 housing dataset in the sibling `geospatial-xai-attacks` repo (`data/processed/seattle_housing_with_demographics.csv`), not this repo's Seattle EUI data — kept for reference but not reproducible from this repo alone. |
| `attribute_graph.py` | Negative-result test of the generalization claim: non-geographic anchor graph (kNN over building attributes), non-spatial gate (vintage/`YearBuilt`). | Documented failure with a real diagnosis (RESULTS S7b): on-manifold probing needs the *probe distribution itself* to be locally smooth in the probe metric, or the data's own clustering reads as model multimodality. Worth keeping as a cautionary result, not as something to build on. |
| `fig_seattle_map.py`, `fig_three_panel_map.py`, `fig_recovery_results.py`, `fig_spatial_randomization.py` | Figure scripts built on the σ\*-based v1/onmanifold-comparison protocols. | Explicitly superseded per `docs/results_2026-07-28.md` S7c; current figures come from `scripts/make_figures.py`. |
