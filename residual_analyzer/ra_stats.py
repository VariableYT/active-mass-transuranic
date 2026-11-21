"""ra_stats.py - observable-agnostic statistics for residual searches.

A residual vector y (experiment minus model) on points with coordinates (Z, N) is modelled as

    y = X0 b + amplitude * template + e,     e ~ N(0, K),
    K = a^2 exp(-d^2 / 2 l^2) + diag(s_w^2 + sigma_exp^2)

The squared-exponential term is the model's own correlated error (mass-model residuals are smooth in (Z, N)); its
amplitude a, length l and the white term s_w are fitted to the residuals by restricted maximum likelihood (REML).
Every amplitude, z-score and limit below is generalised least squares under that K, with the nuisance columns X0
profiled out. This is the noise model of the paper's Sec. II D (data_test_robust.py), made reusable.
"""
import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


def pairwise_distance(Z, N):
    return np.sqrt((Z[:, None] - Z[None, :]) ** 2 + (N[:, None] - N[None, :]) ** 2)


def _reml(K, X, y):
    L = np.linalg.cholesky(K)
    Xw = solve_triangular(L, X, lower=True); yw = solve_triangular(L, y, lower=True)
    A = Xw.T @ Xw
    b = np.linalg.solve(A, Xw.T @ yw)
    rw = yw - Xw @ b
    return -0.5 * (2 * np.sum(np.log(np.diag(L))) + np.linalg.slogdet(A)[1] + rw @ rw)


def fit_gp(D, se, y, X0, starts=((300, 3, 150), (200, 8, 250), (350, 1.5, 80))):
    """REML fit of (amplitude, length, white). Returns dict with the fitted values and K."""
    def K_of(p):
        a, l, sw = np.exp(p)
        return a * a * np.exp(-D ** 2 / (2 * l * l)) + np.diag(sw * sw + se ** 2)

    best = None
    for s in starts:
        r = minimize(lambda p: -_reml(K_of(p), X0, y), np.log(s), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    a, l, sw = (float(v) for v in np.exp(best.x))
    w = 1 / (np.var(y) + se ** 2)
    b0 = np.linalg.solve(X0.T @ (X0 * w[:, None]), X0.T @ (w * y))
    r0 = y - X0 @ b0
    return dict(amp=a, length=l, white=sw, reml=float(-best.fun),
                reml_independent=float(_reml(np.diag(np.var(r0) + se ** 2), X0, y)), K=K_of(best.x))


class Projected:
    """GLS machinery for a fixed covariance K with nuisance columns X0 profiled out.

    P = K^-1 - K^-1 X0 (X0' K^-1 X0)^-1 X0' K^-1. For a template t: amplitude = t'Py / t'Pt, sd = (t'Pt)^-1/2,
    z = t'Py / sqrt(t'Pt), identical to regressing y on [X0, t] by GLS."""

    def __init__(self, K, X0):
        self.L = np.linalg.cholesky(K)
        Ki = np.linalg.inv(K)
        KiX = Ki @ X0
        self.P = Ki - KiX @ np.linalg.solve(X0.T @ KiX, KiX.T)
        self.n = K.shape[0]

    def amplitude(self, t, y):
        info = float(t @ self.P @ t)
        return float(t @ self.P @ y) / info, 1 / math.sqrt(info)

    def z(self, T, y):
        """z-score of each column of T (n x m). Columns the nuisance terms absorb get z = 0."""
        PT = self.P @ T
        info = np.einsum("ij,ij->j", T, PT)
        ok = info > 1e-12 * np.max(info)
        z = np.zeros(T.shape[1])
        z[ok] = (PT[:, ok].T @ y) / np.sqrt(info[ok])
        return z, ok

    def null_max(self, T, groups, nsim, rng, two_sided=True):
        """Distribution of max |z| (or max z if not two_sided) over the template bank under the fitted noise
        model (look-elsewhere). groups: dict name -> boolean mask over templates. Returns dict name -> sorted
        array of nsim maxima. The noise hyperparameters are held at their fitted values (not refitted per
        simulation)."""
        PT = self.P @ T
        info = np.einsum("ij,ij->j", T, PT)
        ok = info > 1e-12 * np.max(info)
        M = (PT[:, ok] / np.sqrt(info[ok])).T @ self.L            # z* = M eps, eps ~ N(0, I)
        zs = M @ rng.standard_normal((self.n, nsim))
        if two_sided:
            zs = np.abs(zs)
        out = {}
        for name, mask in groups.items():
            m = mask[ok]
            out[name] = np.sort(zs[m].max(axis=0)) if m.any() else np.zeros(nsim)
        return out


def p_global(null_sorted, observed):
    """Fraction of simulated maxima at or above the observed one (with the +1 correction)."""
    k = len(null_sorted) - np.searchsorted(null_sorted, observed, side="left")
    return float((k + 1) / (len(null_sorted) + 1))


def upper_limit_nonneg(b, sb, cl_z=1.645):
    """One-sided 95% limit on an amplitude that is physically >= 0 (same convention as the paper)."""
    return max(b, 0.0) + cl_z * sb
