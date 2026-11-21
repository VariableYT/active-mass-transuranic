#!/usr/bin/env python3
"""
data_test_robust.py - the Q_alpha self-energy fit of data_test.py, redone without assuming independent residuals.

data_test.py fits r_i = Q_exp - Q_model = c + c_SHE + beta * t_i(s) with independent noise of rms 370 keV. Mass-model
errors are smooth in (Z, N), so the residuals of neighbouring nuclei (and of members of one alpha-decay chain, which
share N - Z) are correlated, and the independent-noise limit can be too tight. This script:

  0. reproduces data_test.py exactly (same data, same selection, same template) as a check;
  1. measures the correlation directly: residual-pair correlation vs separation in (Z, N), and along chains;
  2. fits a Gaussian-process noise model to the residuals themselves (REML, null hypothesis):
       K = a^2 exp(-d^2 / 2 l^2) + s_w^2 I + diag(sigma_exp^2),  d = distance in the (Z, N) plane,
     and redoes the limit by generalised least squares with that covariance;
  3. an alternative noise model: a random offset per decay chain (same N - Z) plus white noise (REML);
  4. cluster-robust (sandwich, CR1) errors for the original weighted fit, clustered by N - Z;
  5. how far the original fit understates its own error if the residuals follow the fitted GP (exact, h K h^T);
  6. the same GP fit against the WS model with its data-driven (RBF) correction;
  7. the full width scan (2.5-12 nucleons) with the GP noise model, for the paper's figures.
Same data as data_test.py (AME2020 rct1.mas20, WS model pure 'bth' column); seeded.

Run from this folder:  python data_test_robust.py   ->  data_test_robust.json
"""
import json
import math
import os
import sys

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Same definitions as channels.py (copied, not imported: importing channels.py reruns its checks and rewrites
# channels.json).
G_N = 6.67430e-11; U_AMU = 1.66053906660e-27; R0_NUC = 1.2e-15


def E_self_unit(A):  # J, for q_nuc = 1
    M = A * U_AMU; Rn = R0_NUC * A ** (1 / 3)
    return -0.6 * G_N * M * M / Rn


def _f_yuk(r, lam):
    return (1 + r / lam) * math.exp(-r / lam)


q0_B = math.sqrt(6.2e-9 / 8.44e-4)
q0_M = math.sqrt((abs(1.2e-4) + 1.96 * 7e-4) / abs(_f_yuk(1.05, 0.5) - _f_yuk(0.05, 0.5)))

DATA = os.path.join(HERE, "..", "data")
AME = os.path.join(DATA, "rct1.mas20")
WS = os.path.join(DATA, "WS_mass_table.txt")
B_ALPHA = 28295.6616
SEED = 20261114
WIDTHS = [4.5, 5.0, 7.0, 10.0]


def load_qa():
    out = {}
    lines = open(AME).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("1 A  elt"))
    for l in lines[start + 1:]:
        if len(l) < 78:
            continue
        try:
            A = int(l[1:4]); Z = int(l[8:11])
        except ValueError:
            continue
        v, s = l[56:68], l[68:78]
        if "*" in v or "#" in v or not v.strip():
            continue
        out[(Z, A - Z)] = (float(v), float(s))
    return out


def load_ws():
    out = {}
    for l in open(WS):
        f = l.split()
        if len(f) < 17:
            continue
        Z, N = int(f[0]), int(f[1])
        out[(Z, N)] = dict(bcal=float(f[3]) * 1e3, bth=float(f[4]) * 1e3)
    return out


qa, ws = load_qa(), load_ws()


def q_model(Z, N, col="bth"):
    return ws[(Z - 2, N - 2)][col] + B_ALPHA - ws[(Z, N)][col]


rows = []
for (Z, N), (Q, s) in qa.items():
    if Z < 84 or (Z, N) not in ws or (Z - 2, N - 2) not in ws:
        continue
    rows.append((Z, N, Q, s, Q - q_model(Z, N), Q - q_model(Z, N, "bcal")))
rows = np.array(rows)
Zr, Nr, Qe, se, res_th, res_cal = rows.T
she = Zr >= 104
sig_model = float(np.sqrt(np.mean(res_th[she] ** 2)))
y = res_th
n = len(y)
chain = (Nr - Zr).astype(int)


def template(Z, N, s, Zc=114, Nc=184):
    def E(Z, N):
        g = math.exp(-((Z - Zc) ** 2 + (N - Nc) ** 2) / (2 * s * s))
        return E_self_unit(Z + N) / 1.602176634e-16 * g * g
    return E(Z, N) - E(Z - 2, N - 2)


T = {s: np.array([template(int(z), int(nn), s) for z, nn in zip(Zr, Nr)]) for s in WIDTHS}
X0 = np.vstack([np.ones(n), she.astype(float)]).T


def gls(K, t):
    """GLS of y on [1, SHE, t] with covariance K; returns beta_t, sd(beta_t). The template column is scaled to
    unit maximum for the solve (it is ~1e-31 keV per unit q_c^2) and the result scaled back."""
    sc = float(np.max(np.abs(t)))
    X = np.column_stack([X0, t / sc])
    L = np.linalg.cholesky(K)
    Xw = solve_triangular(L, lower=True, b=X); yw = solve_triangular(L, lower=True, b=y)
    cov = np.linalg.inv(Xw.T @ Xw)
    b = cov @ Xw.T @ yw
    return float(b[2]) / sc, float(math.sqrt(cov[2, 2])) / sc


def limits(b, sb, s):
    b95 = max(b, 0.0) + 1.645 * sb
    qc = math.sqrt(b95)
    return dict(beta_hat=b, beta_sd=sb, z=b / sb, beta95=b95, qc95=qc, Pc95_baryon=qc * q0_B, Pc95_mass=qc * q0_M)


# ---------------------------------------------------------------- 0. reproduce data_test.py
K_ind = np.diag(sig_model ** 2 + se ** 2)
base = {s: limits(*gls(K_ind, T[s]), s) for s in WIDTHS}
ref = {r["sigma"]: r for r in json.load(open("data_test.json"))["scan"]}
repro = {s: base[s]["qc95"] / ref[s]["qc95"] for s in WIDTHS}
assert all(abs(v - 1) < 1e-6 for v in repro.values()), repro

# ---------------------------------------------------------------- 1. measured correlations
dZ = Zr[:, None] - Zr[None, :]; dN = Nr[:, None] - Nr[None, :]
D = np.sqrt(dZ ** 2 + dN ** 2)
# remove the two fitted offsets first (null model, independent weights)
w = 1 / np.diag(K_ind)
b0 = np.linalg.solve(X0.T @ (X0 * w[:, None]), X0.T @ (w * y))
r0 = y - X0 @ b0
iu = np.triu_indices(n, 1)
corr_by_d = []
for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 3.2), (3.2, 4.5), (4.5, 6.5), (6.5, 10), (10, 20), (20, 60)]:
    m = (D[iu] > lo) & (D[iu] <= hi)
    a, bb = r0[iu[0][m]], r0[iu[1][m]]
    corr_by_d.append(dict(d_lo=lo, d_hi=hi, pairs=int(m.sum()), corr=float(np.corrcoef(a, bb)[0, 1])))
same_chain_adj = (dZ[iu] == 2) & (dN[iu] == 2) | (dZ[iu] == -2) & (dN[iu] == -2)
a, bb = r0[iu[0][same_chain_adj]], r0[iu[1][same_chain_adj]]
chain_adjacent = dict(pairs=int(same_chain_adj.sum()), corr=float(np.corrcoef(a, bb)[0, 1]))
same_chain = (chain[:, None] == chain[None, :])[iu]
a, bb = r0[iu[0][same_chain]], r0[iu[1][same_chain]]
chain_any = dict(pairs=int(same_chain.sum()), corr=float(np.corrcoef(a, bb)[0, 1]))


# ---------------------------------------------------------------- 2. GP noise model by REML
def reml(K, X=X0):
    L = np.linalg.cholesky(K)
    Xw = solve_triangular(L, lower=True, b=X); yw = solve_triangular(L, lower=True, b=y)
    A = Xw.T @ Xw
    b = np.linalg.solve(A, Xw.T @ yw)
    rw = yw - Xw @ b
    return -0.5 * (2 * np.sum(np.log(np.diag(L))) + np.linalg.slogdet(A)[1] + rw @ rw)


def K_gp(p):
    a, l, sw = np.exp(p)
    return a * a * np.exp(-D ** 2 / (2 * l * l)) + np.diag(sw * sw + se ** 2)


best = None
for p0 in ([math.log(300), math.log(3), math.log(150)], [math.log(200), math.log(8), math.log(250)],
           [math.log(350), math.log(1.5), math.log(80)]):
    r = minimize(lambda p: -reml(K_gp(p)), p0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
    if best is None or r.fun < best.fun:
        best = r
p_gp = best.x
gp = dict(amp_keV=float(np.exp(p_gp[0])), length_nucleons=float(np.exp(p_gp[1])), white_keV=float(np.exp(p_gp[2])),
          reml=float(-best.fun), reml_independent=float(reml(np.diag(np.var(r0) + se ** 2))))
K_hat = K_gp(p_gp)
gp_lim = {s: limits(*gls(K_hat, T[s]), s) for s in WIDTHS}


# ---------------------------------------------------------------- 3. chain random effect by REML
C = (chain[:, None] == chain[None, :]).astype(float)


def K_ch(p):
    tau, sw = np.exp(p)
    return tau * tau * C + np.diag(sw * sw + se ** 2)


rc = minimize(lambda p: -reml(K_ch(p)), [math.log(150), math.log(300)], method="Nelder-Mead",
              options=dict(xatol=1e-5, fatol=1e-5, maxiter=4000))
ch = dict(tau_keV=float(np.exp(rc.x[0])), white_keV=float(np.exp(rc.x[1])), reml=float(-rc.fun))
ch_lim = {s: limits(*gls(K_ch(rc.x), T[s]), s) for s in WIDTHS}


# ---------------------------------------------------------------- 4. cluster-robust (CR1) on the original fit
def cr1(t):
    X = np.column_stack([X0, t]); W = 1 / np.diag(K_ind)
    A = np.linalg.inv(X.T @ (X * W[:, None]))
    b = A @ X.T @ (W * y)
    u = W * (y - X @ b)
    meat = np.zeros((3, 3))
    groups = np.unique(chain)
    for g in groups:
        m = chain == g
        sg = X[m].T @ u[m]
        meat += np.outer(sg, sg)
    G = len(groups); k = 3
    V = A @ meat @ A * G / (G - 1) * (n - 1) / (n - k)
    return float(b[2]), float(math.sqrt(V[2, 2])), G


cr = {}
for s in WIDTHS:
    b, sb, G = cr1(T[s])
    cr[s] = limits(b, sb, s); cr[s]["clusters"] = G

# ---------------------------------------------------------------- 5. how much the original fit understates its error
# Exact sampling sd of the original (independent-weights) estimator if the residuals follow the fitted GP:
# sd^2 = h K h^T with h the estimator's row, compared with the sd that fit reports.
calib = {}
for s in WIDTHS:
    sc = float(np.max(np.abs(T[s])))
    X = np.column_stack([X0, T[s] / sc]); W = 1 / np.diag(K_ind)
    A = np.linalg.inv(X.T @ (X * W[:, None]))
    h = (A @ (X.T * W[None, :]))[2]
    calib[s] = dict(true_sd_of_beta_over_naive_sd=float(math.sqrt(h @ K_hat @ h) / math.sqrt(A[2, 2])))

# ---------------------------------------------------------------- 6. second-column check (WS with RBF correction)
y_cal = res_cal


def gls_y(K, t, yy):
    sc = float(np.max(np.abs(t)))
    X = np.column_stack([X0, t / sc]); L = np.linalg.cholesky(K)
    Xw = solve_triangular(L, lower=True, b=X); yw = solve_triangular(L, lower=True, b=yy)
    cov = np.linalg.inv(Xw.T @ Xw); b = cov @ Xw.T @ yw
    return float(b[2]) / sc, float(math.sqrt(cov[2, 2])) / sc


cal_lim = {s: limits(*gls_y(K_hat, T[s], y_cal), s) for s in WIDTHS}

# ---------------------------------------------------------------- 7. full width scan with the GP noise model
# same grid and record format as data_test.json "scan", for the paper's figures
scan_gp = []
for s in np.arange(2.5, 12.01, 0.25):
    t = np.array([template(int(z), int(nn), s) for z, nn in zip(Zr, Nr)])
    r = limits(*gls(K_hat, t), s)
    r.update(sigma=float(s), template_max_keV=float(np.max(np.abs(t))))
    scan_gp.append(r)

out = dict(
    n=int(n), n_she=int(she.sum()), sig_model_keV=sig_model, reproduces_data_test=repro,
    correlation=dict(by_distance=corr_by_d, chain_adjacent=chain_adjacent, same_chain_any=chain_any),
    gp=gp, chain_effect=ch,
    limits=dict(independent=base, gp=gp_lim, chain=ch_lim, cluster_robust=cr, gp_with_rbf_corrected_model=cal_lim),
    independent_fit_sd_understated_by=calib,
    scan_gp=scan_gp, max_z_gp=float(max(r["z"] for r in scan_gp)),
    ratio_qc95_to_independent={k: {s: v[s]["qc95"] / base[s]["qc95"] for s in WIDTHS}
                               for k, v in dict(gp=gp_lim, chain=ch_lim, cluster_robust=cr, gp_rbf=cal_lim).items()},
)
json.dump(out, open("data_test_robust.json", "w"), indent=1, default=float)
print("reproduces data_test.py:", {s: round(v, 9) for s, v in repro.items()})
print("residual correlation vs distance in (Z,N):")
for c in corr_by_d:
    print("   d in (%4.1f, %4.1f]  pairs %6d  corr %+.3f" % (c["d_lo"], c["d_hi"], c["pairs"], c["corr"]))
print("   chain-adjacent pairs %d corr %+.3f;  same chain any %d corr %+.3f" % (
    chain_adjacent["pairs"], chain_adjacent["corr"], chain_any["pairs"], chain_any["corr"]))
print("GP noise:", {k: round(v, 2) for k, v in gp.items()})
print("chain effect:", {k: round(v, 2) for k, v in ch.items()})
for s in WIDTHS:
    print("width %4.1f  qc95: indep %.3g | GP %.3g | chain %.3g | CR1 %.3g   (independent fit's sd understated "
          "x%.2f)  Pc95_B: indep %.3g GP %.3g" % (
              s, base[s]["qc95"], gp_lim[s]["qc95"], ch_lim[s]["qc95"], cr[s]["qc95"],
              calib[s]["true_sd_of_beta_over_naive_sd"], base[s]["Pc95_baryon"], gp_lim[s]["Pc95_baryon"]))
print("GP z-scores:", {s: round(gp_lim[s]["z"], 2) for s in WIDTHS}, " RBF-model GP qc95:",
      {s: "%.3g" % cal_lim[s]["qc95"] for s in WIDTHS})
