"""Split conformal on the gated model: marginal vs branch-conditional coverage.

Marginal coverage is fine by construction; the audit-relevant question is coverage
CONDITIONAL on which branch the input was routed to. RESULTS_2026-07-28.md S3: at the
planted penalty (0.30, well below the model's own residual scale) marginal coverage
lands on nominal and an auditor checking only the aggregate sees nothing wrong, while
the penalized branch is under-covered by +0.06. The routing partition this package
recovers is exactly the grouping a Mondrian/group-conditional conformal predictor would
need to fix it.
"""
import numpy as np


def conformal_report(deployed_pred, gate_mask, y, coverage=0.90, seed=0):
    """deployed_pred: the DEPLOYED (gated) model's predictions f(x) = h(x) - penalty*1[gate].
    gate_mask: bool, True where a building actually sits on the penalized branch."""
    f = np.asarray(deployed_pred, float)
    y = np.asarray(y, float)
    gate_mask = np.asarray(gate_mask, bool)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cal, test = idx[: len(y) // 2], idx[len(y) // 2:]
    q = np.quantile(np.abs(y[cal] - f[cal]), coverage)  # split-conformal radius

    cover = np.abs(y[test] - f[test]) <= q
    gate_t = gate_mask[test]
    return {
        "q": float(q),
        "marginal": float(cover.mean()),
        "penalized": float(cover[gate_t].mean()),
        "unpenalized": float(cover[~gate_t].mean()),
        "n_pen": int(gate_t.sum()),
        "n_unp": int((~gate_t).sum()),
    }
