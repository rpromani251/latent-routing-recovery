"""Synthetic smooth-function generation, used by the toy validation scripts."""
import numpy as np


def rff_gp_path(d, amp, ell, n_feat, rng):
    """Random-Fourier-feature draw from a stationary SE-GP of amplitude `amp`,
    lengthscale `ell`. Exact SE covariance in the n_feat -> inf limit and O(m*n_feat)
    to evaluate, which is what makes many replicates affordable."""
    W = rng.normal(0.0, 1.0 / ell, size=(n_feat, d))
    b = rng.uniform(0.0, 2 * np.pi, size=n_feat)
    w = rng.normal(0.0, 1.0, size=n_feat) * np.sqrt(2.0 * amp**2 / n_feat)
    return lambda Z: np.cos(Z @ W.T + b) @ w
