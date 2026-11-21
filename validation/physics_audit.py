"""Adversarial audit of the active-mass model in paper/main.tex (rev 5k).

The model under test (paper Sec. II-III):
    V = -(G m_a m_b / r) [1 + q_a q_b exp(-r/lam)],   q = q0 + q_nuc(Z, N),
    q_nuc(Z, N) = q_c exp(-d^2 / 2 sigma^2),  d^2 = (Z-114)^2 + (N-184)^2,
scalar exchange, lam in [0.02, 0.51] m.  The aim is to break it, not to confirm it.

Four tests:
  1. low-Z EP tail leakage: Be-Ti and Cu-Pb (Eot-Wash), Ti-Pt (MICROSCOPE), plus the Smith et al. uranium attractor;
     the widest switch that the tail allows, and the window in which a uranium-pair signal is still detectable.
  2. self-screening: Yukawa form factor of 1-10 kg DU / Th bodies (volume vs surface scaling) and the size of any
     nonlinear (field-energy) correction.
  3. 3D quadrature of the Yukawa and Newtonian torques (DU source, DU test body, 2-10 cm gaps) against the thermal
     floor over 30 days and the Newtonian residual from +-10 um machining / placement.
  4. the weak-field Lagrangian: units, sign, ghosts/tachyons, energy-momentum exchange, Lorentz invariance.

Run:  python validation/physics_audit.py      (from the top folder)
Writes validation/out/audit_results.json and validation/out/AUDIT_SUMMARY.md.  Deterministic (seed 20261118).
"""
import json
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261118)

# ------------------------------------------------------------------ constants (CODATA 2018)
G = 6.67430e-11; C = 2.99792458e8; HBAR = 1.054571817e-34; KB = 1.380649e-23
U_KG = 1.66053906660e-27; EV = 1.602176634e-19; G0 = 9.80665
HBARC_EVM = HBAR * C / EV                                  # 1.973e-7 eV m

# ------------------------------------------------------------------ inputs taken from the paper's own outputs
NUM = json.loads((ROOT / "paper" / "numbers.json").read_text())
HAND = ROOT
QA = json.loads((HAND / "2_indirect_search" / "data_test_robust.json").read_text())
SIM = json.loads((HAND / "1_experiment_sim" / "results.json").read_text())
PEND = SIM["pendulum"]
PASSIVE_95 = NUM["passive_95"]                              # Smith et al. 2000: 5.7e-13 / 9.2e-5
DBMU = NUM["dBmu"]                                          # B/mu (Cu) - B/mu (Pb)
S_B = {0.02: 1.53e-6, 0.05: 5.45e-5, 0.10: 2.47e-4, 0.20: 5.61e-4, 0.50: 8.44e-4, 1.0: 9.19e-4, 10.0: 9.48e-4}
Q0_B = NUM["qalpha"]["q0_B"]                                # |q0| bound, baryon reading, 0.5 m
U_REACH_PAPER = NUM["nuclides"]["U-238"]["S"]               # paper's 30-day uranium reach (Cu test body)
_QS = sorted((r["sigma"], r["qc95"]) for r in QA["scan_gp"])
QS_SIG = np.array([s for s, _ in _QS]); QS_QC = np.array([q for _, q in _QS])
LAMS = [0.02, 0.05, 0.10, 0.20, 0.51]
Q0_TEST = 1.0e-3                                            # nonzero-q0 case: below |q0| <= 2.7e-3, leaves room for the tail


def lerp_log(x, xs, ys):
    return float(10 ** np.interp(np.log10(x), np.log10(xs), np.log10(ys)))


def I_A(lam):
    """Smith et al. attractor factor at range lam (paper: I_A = Sigma(lam) / (dBmu * 238/238.05))."""
    return lerp_log(lam, list(S_B), list(S_B.values())) / (DBMU * (238 / 238.05079))


def B_U(lam):
    """|P_U| bound at range lam (paper Eq. 4)."""
    return PASSIVE_95 / lerp_log(lam, list(S_B), list(S_B.values()))


def qc_decay(sig):
    """Decay-energy ceiling on q_c (paper Sec. III, GP noise model). Held flat outside the scanned widths."""
    return float(np.interp(sig, QS_SIG, QS_QC))


# natural (or depleted / single-isotope) compositions, mass fractions: (Z, N, fraction)
MAT = {
    "Be": [(4, 5, 1.0)],
    "Ti": [(22, 24, .0825), (22, 25, .0744), (22, 26, .7372), (22, 27, .0541), (22, 28, .0518)],
    "Cu": [(29, 34, .6915), (29, 36, .3085)],
    "Pt": [(78, 114, .0078), (78, 116, .3286), (78, 117, .3378), (78, 118, .2521), (78, 120, .0736)],
    "W": [(74, 108, .2650), (74, 109, .1431), (74, 110, .3064), (74, 112, .2843)],
    "Pb": [(82, 122, .014), (82, 124, .241), (82, 125, .221), (82, 126, .524)],
    "Th": [(90, 142, 1.0)],
    "DU": [(92, 146, .998), (92, 143, .002)],
    "Pu244": [(94, 150, 1.0)],
    "rock": [(8, 8, .47), (14, 14, .28), (13, 14, .08), (26, 30, .05), (20, 20, .04), (11, 12, .03), (19, 20, .03),
             (12, 12, .02)],
}
RHO = {"DU": 19050., "Th": 11720., "W": 19300., "Cu": 8960.}


def d2(Z, N):
    return (Z - 114) ** 2 + (N - 184) ** 2


def qnuc(mat, qc, sig):
    return sum(w * qc * np.exp(-d2(Z, N) / (2 * sig * sig)) for Z, N, w in MAT[mat])


def log_qnuc_unit(mat, sig):
    """log of q_nuc / q_c, safe for very narrow switches."""
    terms = [np.log(w) - d2(Z, N) / (2 * sig * sig) for Z, N, w in MAT[mat]]
    m = max(terms)
    return m + np.log(sum(np.exp(t - m) for t in terms))


# =================================================================== TEST 1: EP tail leakage
# Bounds used (95 % CL):
#   Eot-Wash, Be-Ti toward the Earth: eta = (0.3 +- 1.8) e-13 against g_horizontal = 1.68e-2 m/s^2
#     (Schlamminger et al., PRL 100, 041101 (2008)) -> |Delta a| < 6.0e-15 m/s^2.  The same acceleration bound is
#     applied to Cu-Pb toward the Earth (stand-in; the Eot-Wash Cu-Pb result is the Smith attractor test below).
#   MICROSCOPE, Ti-Pt: eta = [-1.5 +- 2.3 (stat) +- 1.5 (syst)] e-15 (Touboul et al., PRL 129, 121102 (2022))
#     -> |eta| < 6.9e-15, at 710 km altitude.
#   Smith et al., PRD 61, 022001 (2000): Cu-Pb toward a 3 t uranium attractor (the paper's Sec. II C bound).
# A Yukawa field of range lam from the ground is at most 2 pi G rho lam (half-space, h = 0); counting all of it as
# horizontal is the most generous (most constraining) assumption possible for a lab balance.
EW_DA95 = 2 * 1.8e-13 * 1.68e-2 * (1 + 0.3 / 1.8 * 0)    # 6.0e-15 m/s^2
MS_ETA95 = 1.5 + 1.96 * np.hypot(2.3, 1.5); MS_ETA95 *= 1e-15
RHO_ROCK = 2670.0; H_MICROSCOPE = 7.1e5


def ground_yukawa(lam, h=0.0):
    return 2 * np.pi * G * RHO_ROCK * lam * np.exp(-h / lam)


def eta_pair_earth(A, B, qc, sig, q0, lam):
    """Eotvos parameter from the tail, toward the ground, maximal (all-horizontal) geometry, relative to g."""
    qS = q0 + qnuc("rock", qc, sig)
    return abs(qS * (qnuc(A, qc, sig) - qnuc(B, qc, sig))) * ground_yukawa(lam) / G0


def smith_lhs(qc, sig, q0, lam):
    """|q_U| (|q0 dBmu| + |q_Cu - q_Pb|) I_A: q0's own baryon-like part and the tail added without any cancellation
    between them (a cancellation would be a fine-tuning of the sign of q0 against lead's charge). Must be <= PASSIVE_95."""
    qU = q0 + qnuc("DU", qc, sig)
    return abs(qU) * (abs(q0 * DBMU) + abs(qnuc("Cu", qc, sig) - qnuc("Pb", qc, sig))) * I_A(lam)


def qc_max(sig, q0, lam):
    """Largest q_c allowed by every existing test at this width, q0 and range (bisection in log q_c)."""
    hi = qc_decay(sig)

    def ok(qc):
        return (smith_lhs(qc, sig, q0, lam) <= PASSIVE_95 * (1 + 1e-12)
                and eta_pair_earth("Be", "Ti", qc, sig, q0, lam) * G0 <= EW_DA95 * (1 + 1e-12))
    if ok(hi):
        return hi, "decay energies"
    lo = 0.0 if not ok(1e-30) else 1e-30
    a, b = np.log(max(lo, 1e-30)), np.log(hi)
    for _ in range(90):
        m = 0.5 * (a + b)
        a, b = (m, b) if ok(np.exp(m)) else (a, m)
    which = "Smith (Cu-Pb toward U)" if smith_lhs(np.exp(b), sig, q0, lam) > PASSIVE_95 else "Eot-Wash Be-Ti"
    return float(np.exp(a)), which


def test1(reach_by_lam):
    sigs = np.round(np.arange(3.0, 40.01, 0.25), 2)
    out = {"bounds": dict(eotwash_da95=EW_DA95, microscope_eta95=MS_ETA95, smith_passive95=PASSIVE_95,
                          q0_B=Q0_B, rho_rock=RHO_ROCK),
           "microscope_log10_suppression": {f"{l:g}": -H_MICROSCOPE / l / np.log(10) for l in LAMS},
           "scan": {}, "window": {}}
    for q0 in (0.0, Q0_TEST):
        for lam in LAMS:
            rows = []
            for s in sigs:
                qc, binder = qc_max(s, q0, lam)
                qU = qnuc("DU", qc, s); qW = qnuc("W", qc, s)
                # DU source vs a W reference of equal mass, DU test body: anomaly = (q_U - q_W)(q0 + q_U)
                sig_U = abs((qU - qW) * (q0 + qU))
                rows.append(dict(
                    sigma=float(s), qc_max=qc, binder=binder, q_U=qU, signal_U_pair=sig_U,
                    eta_BeTi=eta_pair_earth("Be", "Ti", qc, s, q0, lam),
                    eta_CuPb=eta_pair_earth("Cu", "Pb", qc, s, q0, lam),
                    eta_TiPt_microscope=0.0,      # exp(-7.1e5 m / lam) underflows: no sensitivity at lam <= 0.51 m
                    eta_TiPt_if_ground_level=eta_pair_earth("Ti", "Pt", qc, s, q0, lam),
                    smith_ratio=smith_lhs(qc, s, q0, lam) / PASSIVE_95))
            reach = reach_by_lam[lam]
            det = [r["sigma"] for r in rows if r["signal_U_pair"] >= reach["T5"]]
            exc = [r["sigma"] for r in rows if r["signal_U_pair"] >= reach["S"]]
            key = f"q0={'0' if q0 == 0 else '1e-3'},lam={lam:g}"
            out["scan"][key] = rows
            out["window"][key] = dict(
                detect_5sigma=[min(det), max(det)] if det else None,
                exclude_95=[min(exc), max(exc)] if exc else None,
                reach_S=reach["S"], T5=reach["T5"])
    return out


# =================================================================== TEST 2: self-screening / form factor
def phi_sphere(x):
    """Yukawa form factor of a uniform sphere, x = R/lam: exterior field = point field x phi."""
    x = np.asarray(x, float)
    small = x < 1e-3
    xs = np.where(small, 1.0, x)
    # 3 (x cosh x - sinh x)/x^3, written with exp(-x) factored out to stay finite
    v = 3 * (xs * (1 + np.exp(-2 * xs)) - (1 - np.exp(-2 * xs))) / (2 * xs ** 3) * np.exp(xs)
    return np.where(small, 1 + x * x / 10, v)


def test2():
    res = {"sphere": {}, "scaling_exponent": {}, "nonlinear": {}}
    for mat in ("DU", "Th"):
        for M in (1.0, 3.0, 7.0, 10.0):
            R = (3 * M / (4 * np.pi * RHO[mat])) ** (1 / 3)
            for lam in LAMS:
                x = R / lam
                # effective active mass seen at the surface (contact): point-Yukawa at the centre x phi(x)
                ff = float(phi_sphere(x))
                at_surface = ff * np.exp(-x)                      # Yukawa field at r = R relative to lam -> inf
                res["sphere"][f"{mat},{M:g}kg,{lam:g}"] = dict(R_m=R, x=x, form_factor=ff,
                                                              active_fraction_at_surface=float(at_surface * (1 + x) / 1.0))
    # volume (R^3) or surface (R^2) scaling: d ln(Yukawa field at the surface) / d ln R, DU, 1-10 kg
    for lam in LAMS:
        Rs = (3 * np.array([1.0, 10.0]) / (4 * np.pi * RHO["DU"])) ** (1 / 3)
        def surf_field(R):
            # field per unit alpha at r = R (+ 2 cm gap), in units of G rho (so the mass scaling is explicit)
            r = R + 0.02
            M = 4 / 3 * np.pi * R ** 3
            return M * phi_sphere(R / lam) * (1 + r / lam) * np.exp(-r / lam) / r ** 2
        newton = lambda R: 4 / 3 * np.pi * R ** 3 / (R + 0.02) ** 2
        p_y = np.log(surf_field(Rs[1]) / surf_field(Rs[0])) / np.log(Rs[1] / Rs[0])
        p_n = np.log(newton(Rs[1]) / newton(Rs[0])) / np.log(Rs[1] / Rs[0])
        res["scaling_exponent"][f"{lam:g}"] = dict(yukawa=float(p_y), newton=float(p_n),
                                                  note="exponent of R at a fixed 2 cm gap; 3 = volume, 2 = surface")
    # nonlinearity: the scalar's own potential energy per unit mass inside a bulk body, in units of c^2.
    # Deep inside (R >> lam) the field solves (lap - 1/lam^2) phi = -4 pi G rho q -> phi = 4 pi G rho q lam^2, and the
    # fractional mass shift is eps = q * phi / c^2 = 4 pi G rho q^2 lam^2 / c^2.  Linear theory needs eps << 1.
    for lam in LAMS:
        eps_u = 4 * np.pi * G * RHO["DU"] * Q0_B ** 2 * lam ** 2 / C ** 2
        qc_lin = np.sqrt(C ** 2 / (4 * np.pi * G * 2.0e4 * lam ** 2))   # q at which bulk island matter has eps = 1
        res["nonlinear"][f"{lam:g}"] = dict(eps_DU_at_q0_bound=eps_u, q_where_bulk_island_matter_goes_nonlinear=qc_lin)
    # single nucleus at the island centre (A = 298, r ~ 1.2 A^1/3 fm), lam >> nucleus: eps = q^2 G m / (r c^2)
    m298 = 298 * U_KG; r298 = 1.2e-15 * 298 ** (1 / 3)
    res["nonlinear"]["nucleus_eps_per_q2"] = G * m298 / (r298 * C ** 2)
    res["nonlinear"]["nucleus_q_where_eps_1"] = float(np.sqrt(r298 * C ** 2 / (G * m298)))
    res["nonlinear"]["decay_energy_qc_ceiling_w5"] = qc_decay(5.0)
    return res


# =================================================================== TEST 3: 3D quadrature torque
def gl(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def cylinder_points(a, h, rho, centre, n=(14, 28, 14)):
    r, wr = gl(n[0], 0, a); z, wz = gl(n[2], -h / 2, h / 2)
    th = np.linspace(0, 2 * np.pi, n[1], endpoint=False); wt = np.full(n[1], 2 * np.pi / n[1])
    R, T, Z = np.meshgrid(r, th, z, indexing="ij")
    W = (wr[:, None, None] * r[:, None, None]) * wt[None, :, None] * wz[None, None, :] * rho
    P = np.stack([R * np.cos(T), R * np.sin(T), Z], -1).reshape(-1, 3) + np.asarray(centre)
    return P, W.reshape(-1)


def sphere_points(a, rho, centre, n=(10, 12, 20)):
    r, wr = gl(n[0], 0, a); ct, wc = gl(n[1], -1, 1)
    ph = np.linspace(0, 2 * np.pi, n[2], endpoint=False); wp = np.full(n[2], 2 * np.pi / n[2])
    R, CT, PH = np.meshgrid(r, ct, ph, indexing="ij")
    ST = np.sqrt(1 - CT ** 2)
    W = (wr[:, None, None] * r[:, None, None] ** 2) * wc[None, :, None] * wp[None, None, :] * rho
    P = np.stack([R * ST * np.cos(PH), R * ST * np.sin(PH), R * CT], -1).reshape(-1, 3) + np.asarray(centre)
    return P, W.reshape(-1)


def torque_z(src, tst, lam=None):
    """tau_z on the test body about the fibre (z axis through the origin). lam=None: Newton; else the Yukawa factor."""
    Ps, Ws = src; Pt, Wt = tst
    tau = 0.0
    for i in range(0, len(Pt), 256):
        d = Ps[None, :, :] - Pt[i:i + 256, None, :]
        r = np.linalg.norm(d, axis=-1)
        k = np.ones_like(r) if lam is None else (1 + r / lam) * np.exp(-r / lam)
        F = G * (Wt[i:i + 256, None, None] * Ws[None, :, None]) * k[..., None] * d / r[..., None] ** 3
        F = F.sum(1)
        tau += np.sum(Pt[i:i + 256, 0] * F[:, 1] - Pt[i:i + 256, 1] * F[:, 0])
    return float(tau)


def source_geom(M, rho):
    a = (M / (2 * np.pi * rho)) ** (1 / 3)              # cylinder with h = 2a
    return a, 2 * a


def thermal_sigma_tau(T_int, kappa, Q, f_sig):
    asd = np.sqrt(4 * KB * PEND["T"] * kappa / (Q * 2 * np.pi * f_sig))
    return float(asd * np.sqrt(2.0 / T_int)), float(asd)       # square-wave exchange: sqrt(2/T)


def test3():
    m_t = PEND["m_t"]; R_arm = PEND["R_arm"]
    a_t = (3 * m_t / (4 * np.pi * RHO["DU"])) ** (1 / 3)        # 3 g DU test body, same as the paper's 3 g Cu
    tst_c = (R_arm, 0.0, 0.0)
    T_int = 30 * 86400.0; f_sig = 1 / (2 * PEND["T_half"])
    sig_rep, asd_rep = thermal_sigma_tau(T_int, PEND["kappa"], PEND["Q"], f_sig)
    sig_ew, asd_ew = thermal_sigma_tau(T_int, 2.34e-9, 5000.0, f_sig)
    sig_day_check = thermal_sigma_tau(86400.0, PEND["kappa"], PEND["Q"], f_sig)[0]
    out = {"test_body": dict(material="DU", m=m_t, radius=a_t, arm=R_arm),
           "thermal": dict(asd_rep=asd_rep, sigma_tau_30d_rep=sig_rep, asd_eotwash=asd_ew, sigma_tau_30d_eotwash=sig_ew,
                           sigma_tau_1d_rep=sig_day_check, paper_time_domain_1d=4.578e-18),
           "configs": {}, "convergence": {}}
    tst = sphere_points(a_t, RHO["DU"], tst_c)
    for mat, masses in (("DU", (1.0, 3.0, 7.0, 10.0)), ("Th", (7.0,))):
        for M in masses:
            a, h = source_geom(M, RHO[mat])
            for gap in (0.02, 0.03, 0.05, 0.07, 0.10):
                yc = a + gap + a_t
                src = cylinder_points(a, h, RHO[mat], (R_arm, yc, 0.0))
                tN = torque_z(src, tst)
                pointN = G * M * m_t / yc ** 2 * R_arm
                row = dict(a=a, h=h, centre_dist=yc, tau_newton=tN, point_newton=pointN, newton_point_ratio=tN / pointN,
                           yukawa={})
                for lam in LAMS:
                    tY = torque_z(src, tst, lam)
                    pointY = pointN * (1 + yc / lam) * np.exp(-yc / lam)
                    row["yukawa"][f"{lam:g}"] = dict(
                        tau_per_unit_P=tY, point_ratio=tY / pointY, eff_over_newton=tY / tN,
                        P_floor_thermal_95=2 * sig_rep / tY, P_floor_thermal_95_eotwash=2 * sig_ew / tY)
                out["configs"][f"{mat},{M:g}kg,gap{gap * 100:g}cm"] = row
    # convergence: double every quadrature order at one hard case (7 kg DU, 2 cm gap, lam = 2 cm)
    a, h = source_geom(7.0, RHO["DU"]); yc = a + 0.02 + a_t
    lo = torque_z(cylinder_points(a, h, RHO["DU"], (R_arm, yc, 0)), tst, 0.02)
    hi = torque_z(cylinder_points(a, h, RHO["DU"], (R_arm, yc, 0), n=(28, 56, 28)),
                  sphere_points(a_t, RHO["DU"], tst_c, n=(20, 24, 40)), 0.02)
    out["convergence"]["7kg_gap2cm_lam2cm_rel_change"] = abs(hi / lo - 1)
    # sphere self-check: a sphere source must reproduce point-Yukawa x phi(R/lam) x phi(a_t/lam) exactly
    Rs = (3 * 7.0 / (4 * np.pi * RHO["DU"])) ** (1 / 3); ycs = Rs + 0.02 + a_t
    num = torque_z(sphere_points(Rs, RHO["DU"], (R_arm, ycs, 0), n=(24, 24, 40)), tst, 0.05)
    # torque about the fibre from a sphere-sphere pair: the force is along the line of centres (y), arm = R_arm
    ana = (G * 7.0 * m_t / ycs ** 2 * (1 + ycs / 0.05) * np.exp(-ycs / 0.05) * phi_sphere(Rs / 0.05)
           * phi_sphere(a_t / 0.05) * R_arm)
    out["convergence"]["sphere_vs_analytic_rel_err"] = abs(num / float(ana) - 1)

    # ---- Newtonian residual from +-10 um machining and placement (source vs equal-mass reference exchange)
    M = 3.0; gap = 0.03
    a, h = source_geom(M, RHO["DU"]); yc = a + gap + a_t
    base = dict(a=a, h=h, x=R_arm, y=yc, z=0.0)

    def tau_of(p, fixed_mass):
        rho = RHO["DU"] if not fixed_mass else M / (np.pi * p["a"] ** 2 * p["h"])
        return torque_z(cylinder_points(p["a"], p["h"], rho, (p["x"], p["y"], p["z"])), tst)
    t0 = tau_of(base, False)
    step = 2e-6
    grads = {}
    for fixed in (False, True):
        g = {}
        for k in base:
            pp = dict(base); pm = dict(base); pp[k] += step; pm[k] -= step
            g[k] = (tau_of(pp, fixed) - tau_of(pm, fixed)) / (2 * step) / t0     # d ln tau / d p   (1/m)
        grads["mass_matched" if fixed else "as_machined"] = g
    tol = 10e-6
    res = {}
    for case, g in grads.items():
        # each of source and reference independently off by U(-tol, tol) in every parameter
        n = 200000
        e = RNG.uniform(-tol, tol, (n, 2, 5))
        vec = np.array([g[k] for k in ("a", "h", "x", "y", "z")])
        d = (e[:, 0, :] - e[:, 1, :]) @ vec
        res[case] = dict(p95=float(np.percentile(np.abs(d), 95)), rms=float(np.std(d)),
                         dlntau_dp=g, placement_only_p95=float(np.percentile(np.abs((e[:, 0, 2:] - e[:, 1, 2:]) @ vec[2:]), 95)))
    # tolerance needed for the residual to reach the 30-day thermal 95 % floor at lam = 0.51 m
    floor = out["configs"][f"DU,3kg,gap3cm"]["yukawa"]["0.51"]["P_floor_thermal_95"]
    res["tolerance_for_thermal_floor_um"] = tol * 1e6 * floor / res["mass_matched"]["p95"]
    res["config"] = dict(M=M, gap=gap, tau_newton=t0, thermal_floor_P95_lam051=floor)
    res["paper_flip_p95_oxide"] = SIM["sources"]["238U"]["newton_mc"]["flip_p95_oxide"]
    out["newtonian_residual"] = res
    return out


# =================================================================== TEST 4: weak-field action
def test4():
    t, x, y, z, r, m, g, gb, s0 = sp.symbols("t x y z r m g g_b s0", real=True)
    res = {}
    # (a) static Green's function of (lap - m^2) phi = -g delta: phi = g e^{-mr}/(4 pi r)
    phi = g * sp.exp(-m * r) / (4 * sp.pi * r)
    lap = sp.diff(r ** 2 * sp.diff(phi, r), r) / r ** 2
    res["green_homogeneous"] = sp.simplify(lap - m ** 2 * phi) == 0
    res["green_flux_at_origin"] = str(sp.limit(-4 * sp.pi * r ** 2 * sp.diff(phi, r), r, 0))      # must be g
    # (b) sign: H_int = -L_int = -g_b phi; V(r) = -g g_b e^{-mr}/(4 pi r); dV/dr > 0 -> attractive for like signs
    V = -gb * phi
    dV = sp.simplify(sp.diff(V, r))
    res["dV_dr"] = str(dV)
    res["attractive_like_charges"] = bool(sp.simplify(dV.subs({g: 1, gb: 1, m: 1, r: 1})) > 0)
    Vvec = +gb * phi                                                       # vector (spin-1) exchange, same charges
    res["vector_would_repel"] = bool(sp.diff(Vvec, r).subs({g: 1, gb: 1, m: 1, r: 1}) < 0)
    # (c) energy-momentum: L = 1/2 (phi_t^2 - grad^2) - 1/2 m^2 phi^2 + g s phi  (mostly-minus metric)
    f = sp.Function("phi")(t, x, y, z); S = sp.Function("s")(t, x, y, z)
    X = (t, x, y, z); eta = sp.diag(1, -1, -1, -1)
    dphi = [sp.diff(f, v) for v in X]
    dphi_up = [eta[i, i] * dphi[i] for i in range(4)]
    Lfree = sp.Rational(1, 2) * sum(dphi[i] * dphi_up[i] for i in range(4)) - m ** 2 * f ** 2 / 2
    T = [[dphi_up[mu] * dphi_up[nu] - eta[mu, nu] * Lfree for nu in range(4)] for mu in range(4)]
    divT = [sp.expand(sum(sp.diff(T[mu][nu], X[mu]) for mu in range(4))) for nu in range(4)]
    box = sum(eta[i, i] * sp.diff(f, X[i], 2) for i in range(4))
    eom_rhs = sp.expand(g * S - m ** 2 * f)                               # EOM: box phi = -m^2 phi + g s
    checks = []
    for nu in range(4):
        on_shell = divT[nu].subs(sp.Derivative(f, (t, 2)), sp.solve(sp.Eq(box, eom_rhs), sp.Derivative(f, (t, 2)))[0])
        checks.append(sp.simplify(on_shell - g * S * dphi_up[nu]) == 0)
    res["dT_field = g s d^nu phi on shell (all nu)"] = all(checks)
    # matter: S_m = -int (m_X - g phi) dtau  ->  d/dtau[(m - g phi) u^nu] = g d^nu phi  -> divT_m = -g s d^nu phi
    res["total_conserved"] = "dT_field + dT_matter = g s d phi - g s d phi = 0 (diffeomorphism identity)"
    # positivity of the field Hamiltonian, no tachyon
    pi_, gr, ph = sp.symbols("pi grad phi_", real=True)
    H = pi_ ** 2 / 2 + gr ** 2 / 2 + m ** 2 * ph ** 2 / 2
    res["kinetic_sign_positive"] = sp.diff(H, pi_, 2) > 0
    res["no_tachyon_if_m2_positive"] = True
    # (d) units: g dimensionless; q (gravity units) <-> g:  q^2 = g^2 hbar c / (4 pi G m_u^2)
    gq = float(np.sqrt(4 * np.pi * G * U_KG ** 2 / (HBAR * C)))
    res["g_per_unit_q"] = gq
    res["m_eff_eV_at_lam"] = {f"{l:g}": HBARC_EVM / l for l in LAMS}
    res["perturbative_q_limit_g_lt_1"] = 1 / gq
    res["qc_decay_ceiling_as_g"] = {f"{s:g}": qc_decay(s) * gq for s in (3.5, 5.0, 7.0, 10.0)}
    # (e) Lorentz invariance: coupling to J^0 = n_B (what "phi q rho_B" literally says) vs the scalar density n_B/gamma
    v_cmb = 369.82e3; v_orb = 29.78e3
    res["LLI_if_coupled_to_J0"] = dict(
        frac_shift_cmb=(1 / np.sqrt(1 - (v_cmb / C) ** 2) - 1),
        annual_modulation=2 * v_cmb * v_orb / C ** 2,
        note="coupling to n_B (time component of a 4-vector) gives a frame-dependent charge; use m_X(phi) or phi psi-bar psi")
    return res


# =================================================================== main
def _json_default(o):
    if isinstance(o, (np.bool_, bool)) or isinstance(o, sp.logic.boolalg.BooleanAtom) or isinstance(o, sp.Basic) and o.is_Boolean:
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


def main():
    t3 = test3()
    # reach of a DU-source / DU-test-body exchange per range: the larger of the paper's systematic floor and our
    # 30-day thermal floor (3 kg source, 3 cm gap)
    sys_floor = NUM["nuclides"]["U-238"]["S"]
    reach = {}
    for lam in LAMS:
        th = t3["configs"]["DU,3kg,gap3cm"]["yukawa"][f"{lam:g}"]["P_floor_thermal_95"]
        S = max(th, sys_floor)
        reach[lam] = dict(S=S, T5=2.5 * S, thermal=th, sys=sys_floor)
    t1 = test1(reach)
    t2 = test2()
    t4 = test4()
    res = dict(test1=t1, test2=t2, test3=t3, test4=t4, reach_used=reach)
    (OUT / "audit_results.json").write_text(json.dumps(res, indent=1, default=_json_default))
    report(res)


def fmt(x, d=2):
    if x is None:
        return "-"
    if x == 0:
        return "0"
    return f"{x:.{d}g}" if 1e-3 <= abs(x) < 1e4 else f"{x:.{d - 1}e}"


def report(R):
    t1, t2, t3, t4 = R["test1"], R["test2"], R["test3"], R["test4"]
    L = ["# Physics audit, active-mass model (paper rev 5k)\n"]
    L.append("## Test 1: EP tail leakage and the detectable window\n")
    L.append(f"MICROSCOPE suppression at 710 km: exp(-h/lam) = 10^({t1['microscope_log10_suppression']['0.51']:.3g}) at "
             f"lam = 0.51 m. Its Ti-Pt bound says nothing about this model.\n")
    L.append("| q0 | lam (m) | 5-sigma window in sigma (U pair) | 95 % exclusion window | what caps q_c at the wide edge |")
    L.append("|---|---|---|---|---|")
    for k, w in t1["window"].items():
        rows = t1["scan"][k]
        edge = w["exclude_95"][1] if w["exclude_95"] else None
        binder = next((r["binder"] for r in rows if edge is not None and r["sigma"] > edge), "-")
        dw = f"{w['detect_5sigma'][0]:g}-{w['detect_5sigma'][1]:g}" if w["detect_5sigma"] else "none"
        ew = f"{w['exclude_95'][0]:g}-{w['exclude_95'][1]:g}" if w["exclude_95"] else "none"
        q0s, lam = k.split(",")
        L.append(f"| {q0s.split('=')[1]} | {lam.split('=')[1]} | {dw} | {ew} | {binder} |")
    L.append("\nMax tail eta at the largest allowed q_c, over sigma 3-40 (all-horizontal ground field, q0 = 1e-3):\n")
    L.append("| lam | eta Be-Ti | eta Cu-Pb | eta Ti-Pt (if at ground level) | eta Ti-Pt (MICROSCOPE, 710 km) |")
    L.append("|---|---|---|---|---|")
    for lam in LAMS:
        rows = t1["scan"][f"q0=1e-3,lam={lam:g}"]
        L.append(f"| {lam:g} | {fmt(max(r['eta_BeTi'] for r in rows))} | {fmt(max(r['eta_CuPb'] for r in rows))} | "
                 f"{fmt(max(r['eta_TiPt_if_ground_level'] for r in rows))} | 0 |")
    L.append("\n## Test 2: self-screening\n")
    L.append("| body | lam | R (cm) | R/lam | form factor phi | ")
    L.append("|---|---|---|---|---|")
    for k, v in t2["sphere"].items():
        mat, M, lam = k.split(",")
        if M in ("1kg", "10kg"):
            L.append(f"| {mat} {M} | {lam} | {v['R_m'] * 100:.2f} | {v['x']:.3g} | {v['form_factor']:.4g} |")
    L.append("\nExponent of R in the Yukawa field at a 2 cm gap (3 = volume, 2 = surface; Newton for reference):\n")
    for lam, v in t2["scaling_exponent"].items():
        L.append(f"- lam {lam} m: Yukawa {v['yukawa']:.2f}, Newton {v['newton']:.2f}")
    nl = t2["nonlinear"]
    L.append(f"\nNonlinearity eps = 4 pi G rho q^2 lam^2 / c^2 for DU at the q0 bound: {fmt(nl['0.51']['eps_DU_at_q0_bound'])} "
             f"(lam 0.51 m). Bulk island matter goes nonlinear at q = {fmt(nl['0.51']['q_where_bulk_island_matter_goes_nonlinear'])}; "
             f"a single A = 298 nucleus at q = {fmt(nl['nucleus_q_where_eps_1'])}; decay-energy ceiling at width 5 is "
             f"{fmt(nl['decay_energy_qc_ceiling_w5'])}.\n")
    L.append("## Test 3: 3D torque, thermal floor, machining\n")
    th = t3["thermal"]
    L.append(f"Thermal ASD {fmt(th['asd_rep'])} N m/rtHz (paper pendulum), 1-day sigma {fmt(th['sigma_tau_1d_rep'])} vs the paper's "
             f"time-domain {fmt(th['paper_time_domain_1d'])}; 30-day sigma {fmt(th['sigma_tau_30d_rep'])} N m. Quadrature "
             f"convergence {fmt(t3['convergence']['7kg_gap2cm_lam2cm_rel_change'])}; sphere vs analytic "
             f"{fmt(t3['convergence']['sphere_vs_analytic_rel_err'])}.\n")
    L.append("| source | gap | tau_N (N m) | Newton 3D/point | P floor 95 % (thermal, 30 d) at lam = 0.02 / 0.05 / 0.1 / 0.51 m |")
    L.append("|---|---|---|---|---|")
    for k, v in t3["configs"].items():
        y = v["yukawa"]
        L.append(f"| {k.split(',gap')[0]} | {k.split('gap')[1]} | {fmt(v['tau_newton'])} | {v['newton_point_ratio']:.3f} | "
                 + " / ".join(fmt(y[l]['P_floor_thermal_95']) for l in ("0.02", "0.05", "0.1", "0.51")) + " |")
    nr = t3["newtonian_residual"]
    L.append(f"\nNewtonian residual, +-10 um on a, h, x, y, z of source and reference (3 kg DU, 3 cm gap), 95 %: "
             f"as machined {fmt(nr['as_machined']['p95'])}, mass-matched {fmt(nr['mass_matched']['p95'])} "
             f"(placement alone {fmt(nr['mass_matched']['placement_only_p95'])}). Thermal floor there: "
             f"{fmt(nr['config']['thermal_floor_P95_lam051'])}. Tolerance needed to reach it: "
             f"{fmt(nr['tolerance_for_thermal_floor_um'])} um. Paper's flip residual: {fmt(nr['paper_flip_p95_oxide'])}.\n")
    L.append("## Test 4: action\n")
    for k, v in t4.items():
        L.append(f"- {k}: {v}")
    (OUT / "AUDIT_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
