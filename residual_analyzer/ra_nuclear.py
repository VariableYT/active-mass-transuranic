"""ra_nuclear.py - nuclear sector: binding-energy and Q_alpha residuals against published mass models.

Observables (keV), for each model:
    B    residual = B_exp - B_model = ME_model - ME_exp          (positive: more bound than the model says)
    Qa   residual = Q_alpha_exp - Q_alpha_model, Q_alpha_model = ME_model(Z,N) - ME_model(Z-2,N-2) - ME(4He)
Only measured values enter (AME2020 '#' estimates are excluded and reported as the data cut-off).

Templates:
    generic bank      single nuclides, Gaussian bumps, steps and kinks in N and in Z (two-sided search)
    physical          the self-energy of a scalar charge switched on around (Zc, Nc): the paper's Sec. II D
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G_N, U_AMU, R0_NUC, KEV = 6.67430e-11, 1.66053906660e-27, 1.2e-15, 1.602176634e-16
# ordinary matter's universal charge, from the paper: baryon-number reading (uranium-attractor bound) and mass-like
Q0_B = math.sqrt(6.2e-9 / 8.44e-4)


def _f_yuk(r, lam):
    return (1 + r / lam) * math.exp(-r / lam)


Q0_M = math.sqrt((abs(1.2e-4) + 1.96 * 7e-4) / abs(_f_yuk(1.05, 0.5) - _f_yuk(0.05, 0.5)))


def load_cache():
    exp = json.load(open(HERE / "cache" / "experiment.json"))
    models = json.load(open(HERE / "cache" / "models.json"))
    nuc = {tuple(int(x) for x in k.split(",")): v for k, v in exp["nuclides"].items()}
    mods = {m: {tuple(int(x) for x in k.split(",")): v for k, v in t.items()} for m, t in models.items()}
    return nuc, mods, exp["ME_alpha"]


def dataset(nuc, model, me_alpha, observable, zmin=82):
    """Arrays for one (model, observable): Z, N, residual y, experimental sigma, all in keV."""
    rows = []
    for (Z, N), e in nuc.items():
        if Z < zmin or (Z, N) not in model:
            continue
        if observable == "B":
            if e["ME_est"]:
                continue
            rows.append((Z, N, model[(Z, N)] - e["ME"], e["dME"]))
        elif observable == "Qa":
            if "Qa" not in e or e["Qa_est"] or (Z - 2, N - 2) not in model:
                continue
            rows.append((Z, N, e["Qa"] - (model[(Z, N)] - model[(Z - 2, N - 2)] - me_alpha), e["dQa"]))
        else:
            raise ValueError(observable)
    a = np.array(sorted(rows), dtype=float)
    return dict(Z=a[:, 0], N=a[:, 1], y=a[:, 2], se=a[:, 3])


def nuisance(Z):
    """Constant plus an offset for the superheavy block (Z >= 104), as in the paper's fit."""
    return np.vstack([np.ones(len(Z)), (Z >= 104).astype(float)]).T


# ------------------------------------------------------------------------------------- physical template
def e_self_unit_keV(A):
    """Self-energy (keV, negative) of a uniformly charged sphere of mass A u, radius 1.2 fm A^(1/3), for q = 1."""
    M = A * U_AMU
    return -0.6 * G_N * M * M / (R0_NUC * A ** (1 / 3)) / KEV


def self_energy_template(Z, N, observable, width, Zc=114, Nc=184):
    """Residual per unit q_c^2 for a Gaussian charge q = q_c exp(-d^2 / 2 width^2) centred on (Zc, Nc)."""
    def E(z, n):
        g = np.exp(-((z - Zc) ** 2 + (n - Nc) ** 2) / (2 * width * width))
        return e_self_unit_keV(z + n) * g * g
    if observable == "B":
        return -E(Z, N)                       # extra binding
    return E(Z, N) - E(Z - 2, N - 2)          # Q_alpha: parent minus daughter


# ------------------------------------------------------------------------------------- generic bank
def template_bank(Z, N, bump_widths=(1.0, 2.0, 3.5), min_side=5):
    """Generic templates on the data's own support. Returns (T, keys, family) with T of shape n x m.
    key = (family, parameters), identical across models so that features can be compared between them."""
    cols, keys = [], []
    for z, n in zip(Z, N):                                   # single nuclides (leave-one-out outliers)
        cols.append(((Z == z) & (N == n)).astype(float)); keys.append(("point", int(z), int(n)))
    zs = np.arange(int(Z.min()) - 1, int(Z.max()) + 2); ns = np.arange(int(N.min()) - 1, int(N.max()) + 2)
    for s in bump_widths:                                    # localized bumps
        for zc in zs:
            for nc in ns:
                d2 = (Z - zc) ** 2 + (N - nc) ** 2
                if np.sum(d2 <= (1.5 * s) ** 2) < 3:
                    continue
                cols.append(np.exp(-d2 / (2 * s * s))); keys.append(("bump", int(zc), int(nc), float(s)))
    for name, v in (("N", N), ("Z", Z)):                     # change points along N and along Z
        for v0 in np.unique(v)[1:]:
            hi = v >= v0
            if hi.sum() < min_side or (~hi).sum() < min_side:
                continue
            cols.append(hi.astype(float)); keys.append(("step" + name, int(v0)))
            cols.append(np.where(hi, v - v0, 0.0)); keys.append(("kink" + name, int(v0)))
    return np.array(cols).T, keys


def describe(key):
    fam = key[0]
    if fam == "point":
        return f"single nuclide Z={key[1]} N={key[2]} (A={key[1] + key[2]})"
    if fam == "bump":
        return f"bump at Z={key[1]} N={key[2]}, width {key[3]:g}"
    axis = fam[-1]
    return f"{'step' if fam.startswith('step') else 'change of slope'} at {axis}={key[1]}"


# ------------------------------------------------------------------------------------- data cut-off
def coverage(nuc, zmin=82):
    """Per element: where measured masses and Q_alpha stop, and where their uncertainties are large."""
    out = {}
    for Z in sorted({z for z, _ in nuc if z >= zmin}):
        iso = sorted((n, e) for (z, n), e in nuc.items() if z == Z)
        meas = [n for n, e in iso if not e["ME_est"]]
        qa = [n for n, e in iso if "Qa" in e and not e["Qa_est"]]
        out[Z] = dict(
            el=iso[0][1]["el"], n_listed=len(iso), n_mass_measured=len(meas), n_mass_estimated=len(iso) - len(meas),
            N_mass_measured=[min(meas), max(meas)] if meas else None,
            n_qa_measured=len(qa), N_qa_measured=[min(qa), max(qa)] if qa else None,
            mass_sigma_gt_100keV=[n for n, e in iso if not e["ME_est"] and e["dME"] > 100],
            qa_sigma_gt_100keV=[n for n, e in iso if "Qa" in e and not e["Qa_est"] and e["dQa"] > 100])
    return out
