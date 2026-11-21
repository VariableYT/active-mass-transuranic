#!/usr/bin/env python3
"""
data_test.py — search the measured alpha-decay energies of the heaviest nuclei for the
self-energy signature of a nuclear scalar charge, and bound it.

Data:   AME2020 rct1.mas20 (verbatim), experimental Q_alpha only (no '#').
Theory: improved Weizsacker-Skyrme mass model (Chin. Phys. C 43, 044105; fitted to AME2016,
        github.com/lukeronger/NuclearData-LZU). Pure model binding energies (column 'bth',
        without the data-driven RBF correction). AME2016 had no experimental masses for the
        unconnected superheavy chains, so those Q_alpha were never used to fit the model.

Signal: a Gaussian nuclear charge q_nuc = q_c exp(-d^2/2 s^2) shifts every mass by
        E_self = -(3/5) G (q_nuc A u)^2 / (1.2 fm A^1/3), so Q_alpha(Z,N) shifts by
        dQ = E_self(Z,N) - E_self(Z-2,N-2)   (alpha's own charge ~ q0, negligible).
Fit:    r_i = Q_exp - Q_model = c + beta * t_i(s) + noise, beta = q_c^2 >= 0.
        Noise = model scatter (from residual rms) (+) experimental uncertainty.
"""
import math, json, os, sys
import numpy as np
from scipy.stats import chi2, norm
sys.path.insert(0, ".")
from channels import E_self_unit, q0_B, q0_M

AME = os.path.join("..", "data", "rct1.mas20")          # AME2020 reaction-energy table 1
WS = os.path.join("..", "data", "WS_mass_table.txt")    # fetched by data/fetch_data.py
B_ALPHA = 28295.6616  # keV, AME2020 4He binding energy

def load_qa():
    out = {}
    lines = open(AME).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("1 A  elt"))
    for l in lines[start + 1:]:
        if len(l) < 78: continue
        try:
            A = int(l[1:4]); Z = int(l[8:11])
        except ValueError:
            continue
        v, s = l[56:68], l[68:78]
        if "*" in v or "#" in v or not v.strip(): continue
        out[(Z, A - Z)] = (float(v), float(s))
    return out

def load_ws():
    out = {}
    for l in open(WS):
        f = l.split()
        if len(f) < 17: continue
        Z, N, A = int(f[0]), int(f[1]), int(f[2])
        out[(Z, N)] = dict(bcal=float(f[3]) * 1e3, bth=float(f[4]) * 1e3, qa=float(f[11]) * 1e3)
    return out

qa = load_qa(); ws = load_ws()
# --- integrity checks on the data
checks = {}
checks["n_exp_qalpha"] = len(qa)
checks["U238_Qa_keV"] = qa.get((92, 146))
checks["Po212_Qa_keV"] = qa.get((84, 128))      # well-known 8954 keV
checks["Og294_Qa_keV"] = qa.get((118, 176))
# model column consistency: q_alpha column should equal B(d)+B(a)-B(p) from bcal
d = [ws[(Z, N)]["qa"] - (ws[(Z - 2, N - 2)]["bcal"] + B_ALPHA - ws[(Z, N)]["bcal"])
     for (Z, N) in ws if Z > 90 and (Z - 2, N - 2) in ws and ws[(Z, N)]["qa"] < 8e8]
checks["ws_qalpha_column_vs_bcal_max_abs_keV"] = float(np.max(np.abs(d)))

def q_model(Z, N, col="bth"):
    return ws[(Z - 2, N - 2)][col] + B_ALPHA - ws[(Z, N)][col]

rows = []
for (Z, N), (Q, s) in qa.items():
    if Z < 84 or (Z, N) not in ws or (Z - 2, N - 2) not in ws: continue
    rows.append((Z, N, Q, s, Q - q_model(Z, N), Q - q_model(Z, N, "bcal")))
rows = np.array(rows)
Zr, Nr, Qe, se, res_th, res_cal = rows.T
she = Zr >= 104; ctl = (Zr >= 84) & (Zr <= 100)
stats = dict(
    n_she=int(she.sum()), n_ctl=int(ctl.sum()),
    rms_th_she=float(np.sqrt(np.mean(res_th[she] ** 2))), mean_th_she=float(res_th[she].mean()),
    rms_th_ctl=float(np.sqrt(np.mean(res_th[ctl] ** 2))), mean_th_ctl=float(res_th[ctl].mean()),
    rms_cal_she=float(np.sqrt(np.mean(res_cal[she] ** 2))))

def template(Z, N, s, Zc=114, Nc=184):
    def E(Z, N):
        g = math.exp(-((Z - Zc) ** 2 + (N - Nc) ** 2) / (2 * s * s))
        return E_self_unit(Z + N) / 1.602176634e-16 * g * g      # keV per unit q_c^2
    return E(Z, N) - E(Z - 2, N - 2)

sel = Zr >= 84
sig_model = stats["rms_th_she"]
results = []
for s in np.arange(2.5, 12.01, 0.25):
    t = np.array([template(int(z), int(n), s) for z, n in zip(Zr[sel], Nr[sel])])
    w = 1 / (sig_model ** 2 + se[sel] ** 2)
    y = res_th[sel]
    # fit y = c + beta t with weights; also a region offset for SHE to absorb a model bias
    X = np.vstack([np.ones_like(t), (Zr[sel] >= 104).astype(float), t]).T
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ y
    sb = math.sqrt(cov[2, 2])
    chi_null = None
    b_hat = beta[2]
    # one-sided 95% upper limit on beta >= 0 (Feldman-Cousins-free conservative choice)
    b95 = max(b_hat, 0) + 1.645 * sb
    signif = b_hat / sb
    results.append(dict(sigma=float(s), beta_hat=float(b_hat), beta_sd=float(sb), beta95=float(b95),
                        z=float(signif), qc95=float(math.sqrt(b95)),
                        Pc95_baryon=float(q0_B * math.sqrt(b95)), Pc95_mass=float(q0_M * math.sqrt(b95)),
                        template_max_keV=float(np.max(np.abs(t)))))
zs = np.array([r["z"] for r in results])
# look-elsewhere: trials factor ~ number of independent widths scanned (correlation length ~1)
out = dict(checks=checks, stats=stats, scan=results,
           max_z=float(zs.max()), sigma_at_max_z=results[int(zs.argmax())]["sigma"],
           p_local=float(norm.sf(zs.max())), trials=len(results) / 4,
           p_global=float(min(1, norm.sf(zs.max()) * len(results) / 4)))
json.dump(out, open("data_test.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("checks", "stats", "max_z", "sigma_at_max_z", "p_local", "p_global")}, indent=1))
for r in results[::4]:
    print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in r.items()})

# residual table for the superheavy region
np.savetxt("residuals_she.csv", np.column_stack([Zr[she], Nr[she], Qe[she], se[she], res_th[she]]),
           delimiter=",", header="Z,N,Qalpha_exp_keV,unc_keV,Qexp_minus_WSmodel_keV", fmt="%.0f,%.0f,%.3f,%.3f,%.1f")
