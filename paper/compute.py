"""Every number, table and figure in main.tex.

Run:  python paper/compute.py      (from the top folder of code_and_data)
Writes paper/numbers.json, paper/tab_predictions.tex, paper/tab_practical.tex,
paper/figs/fig{1,2,3}_*.pdf (+ .png copies for the preview).

Model (Sec. II-III of the paper)
--------------------------------
Two bodies a, b interact through gravity plus a Yukawa term from SCALAR exchange,
    V = -(G m_a m_b / r) [1 + q_a q_b * exp(-r/lam)],
q = charge per unit mass in gravitational units, with a sign (like signs attract, opposite signs repel).
q = q0 + q_nuc(Z, N): a universal piece plus a nuclear-structure piece. q_nuc is not a conserved charge, so it
cannot source a vector field (longitudinal-mode emission); hence scalar exchange.
P_X = q_nuc(X) * q0 is the fractional anomaly in the force that a source of nuclide X exerts on ordinary matter at
r << lam: the quantity an active-mass comparison measures. Magnitudes |P_X| are tabulated.

Benchmark switches (both negligible for Z <= 92 by construction), with the strength at the center P_c FREE:
  Gaussian    P_X = P_c exp(-d_X^2 / 2 sigma^2), d^2 = (Z-114)^2 + (N-184)^2
  power law   P_X = B_U (A_X / 238)^k, k fixed so P(A = 298) = P_c
The widest Gaussian allowed by a uranium bound P_U <= B_U gives the closed form
  P_X = P_c^(1 - w_X) * B_U^(w_X),  w_X = d_X^2 / d_U^2.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).parent
FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)

G = 6.67430e-11
YEAR = 3.15576e7
NA = 6.02214076e23
MEV = 1.602176634e-13
ZC, NC = 114, 184
AC = ZC + NC
f = lambda r, lam: (1 + r / lam) * np.exp(-r / lam)

# strength at the island centre: a free parameter; tables at two round values, Fig. 1 at the larger
PC_LO, PC_HI = 1e4, 1e10

# --------------------------------------------------------------- nuclides
# half-life (y) and spontaneous-fission branch: NUBASE2020; alpha Q (MeV): AME2020; SF neutron multiplicity: ENSDF /
# literature. dheat: beta and conversion-electron energy (MeV) per decay of the short-lived daughters in equilibrium
# (234Th + 234mPa for 238U, 240U + 240mNp for 244Pu), from the ENSDF decay radiations (IAEA LiveChart, 2026-11-18)
NUC = {
    "Pb-208": dict(Z=82, N=126, thalf=None, avail="metric tons"),
    "Th-232": dict(Z=90, N=142, thalf=1.40e10, avail="metric tons"),
    "U-238": dict(Z=92, N=146, thalf=4.463e9, sf=5.44e-7, nu=2.0, Qa=4.270, avail="metric tons", rho=19.05, dheat=0.872),
    # rev 5m: the most plentiful plutonium isotope, for the decay-rate benchmark (Sec. III C); nu-bar from ENSDF
    "Pu-239": dict(Z=94, N=145, thalf=2.411e4, sf=3.1e-12, nu=2.16, Qa=5.244, avail="metric tons", rho=19.8),
    "Pu-242": dict(Z=94, N=148, thalf=3.75e5, sf=5.51e-6, nu=2.15, Qa=4.984, avail="grams", rho=19.8),
    "Pu-244": dict(Z=94, N=150, thalf=8.13e7, sf=1.23e-3, nu=2.29, Qa=4.666, avail="grams (scarce)", rho=19.8,
                   dheat=0.689),
    "Cm-248": dict(Z=96, N=152, thalf=3.48e5, sf=8.39e-2, nu=3.13, Qa=5.162, avail="milligrams", rho=13.5),
    "Cf-252": dict(Z=98, N=154, thalf=2.645, sf=3.092e-2, nu=3.77, Qa=6.217, avail="milligrams", rho=15.1),
}
SHE = {"Fl-289": (114, 175), "Mc-290": (115, 175), "Og-294": (118, 176)}   # observed atom-by-atom only
d2 = lambda Z, N: (Z - ZC) ** 2 + (N - NC) ** 2
D2U = d2(92, 146)
for k, v in NUC.items():
    v["A"] = v["Z"] + v["N"]; v["d2"] = d2(v["Z"], v["N"]); v["w"] = v["d2"] / D2U

# --------------------------------------------------------------- the present uranium bound (Sec. II C)
# Smith et al., PRD 61, 022001 (2000), arXiv:2405.10982: Cu and Pb toward a 3-tonne 238U attractor,
# |a_Cu - a_Pb| < 5.7e-13 cm/s^2 at 95% CL, against the attractor's g' = 9.2e-5 cm/s^2. Their Table X gives the
# intrinsic sensitivity S(lambda) = d(q/mu) <(q/mu)_A I_A> for q = N + Z (baryon number); the paper calls it
# Sigma(lambda), since S there is the reach. With a universal part that varies like baryon number, Eq. (4) gives
# P_U * Sigma(lambda) <= 5.7e-13 / 9.2e-5.
SMITH = dict(da95_cm=5.7e-13, g_att_cm=9.2e-5)
S_B_TABLE = {0.02: 1.53e-6, 0.05: 5.45e-5, 0.10: 2.47e-4, 0.20: 5.61e-4, 0.50: 8.44e-4, 1.0: 9.19e-4, 10.0: 9.48e-4}
PASSIVE_95 = SMITH["da95_cm"] / SMITH["g_att_cm"]
B_OF_LAM = {lam: PASSIVE_95 / S for lam, S in S_B_TABLE.items()}
LAM = 0.50                                   # representative range used below
B_U = B_OF_LAM[LAM]


def p_gauss_widest(X, Pc, BU=B_U):
    w = NUC[X]["w"]
    return float(np.exp((1 - w) * np.log(Pc) + w * np.log(BU)))


def sigma_widest(Pc, BU=B_U):
    return float(np.sqrt(D2U / (2 * np.log(Pc / BU))))


def k_power(Pc, BU=B_U):
    return float(np.log(Pc / BU) / np.log(AC / 238))


def p_power(X, Pc, BU=B_U):
    return float(BU * (NUC[X]["A"] / 238) ** k_power(Pc, BU))


# --------------------------------------------------------------- decay data and example configurations
for k, v in NUC.items():
    if v.get("thalf") is None or "sf" not in v:
        continue
    lam_d = np.log(2) / (v["thalf"] * YEAR)
    act = lam_d * NA / v["A"]                                  # Bq per gram
    v["act"] = act
    v["neutrons"] = act * v["sf"] * v["nu"]                    # n / s / g
    v["heat"] = act * ((1 - v["sf"]) * v["Qa"] + v["sf"] * 180.0 + v.get("dheat", 0.0)) * MEV   # W / g (fission
    # fragments deposit locally; daughters' betas counted, their gamma rays mostly escape)

# impurities that set the heat of real material, per 0.1% by mass (NUBASE2020 half-lives and branches, AME2020 Q_alpha)
IMPURITY = {"Am-241 in Pu-244": ("Pu-244", 432.6, 241, 5.638, 3.6e-12), "Pu-240 in Pu-244": ("Pu-244", 6561.0, 240, 5.256, 5.80e-8),
            "Cm-244 in Cm-248": ("Cm-248", 18.11, 244, 5.902, 1.37e-6)}
IMPURITY_HEAT = {}
for _k, (_host, _t, _A, _Q, _sf) in IMPURITY.items():
    _act = np.log(2) / (_t * YEAR) * NA / _A
    IMPURITY_HEAT[_k] = 1e-3 * _act * ((1 - _sf) * _Q + _sf * 180.0) * MEV / NUC[_host]["heat"]

CONF = {  # nuclide: (source mass kg, centre distance m)
    "U-238": (0.100, 0.03), "Pu-242": (1e-3, 0.02), "Pu-244": (1e-3, 0.02), "Cm-248": (1e-5, 0.01), "Cf-252": (1e-6, 0.01),
    "Pu-239": (1e-3, 0.02),
}
for X, (M, r) in CONF.items():
    NUC[X].update(M=M, r=r, aN=G * M / r**2)

# --------------------------------------------------------------- simulated campaigns (Sec. IV B)
# Time-domain simulation of the measurement (1_experiment_sim: active_mass_sim.py and
# detection_map.py, seed 20261114; both reproduce on rerun). Representative room-temperature pendulum:
# I = 1e-5 kg m^2, T0 = 600 s, Q = 4000, 3 g Cu test body on a 2 cm arm, exchange half-period 600 s.
# sigma_stat is calibrated against 520 independent 90-day records (calibration_check.py); sigma_sys assumes Sec. IV C.
# Reach S = 2 sigma_tot (95% exclusion of a null result); detection threshold T5 = 5 sigma_tot.
HAND = HERE.parent
SIM = json.load(open(HAND / "1_experiment_sim" / "results.json"))
DET = json.load(open(HAND / "1_experiment_sim" / "detection_results.json"))
QA = json.load(open(HAND / "2_indirect_search" / "data_test.json"))
# rev 5: the same fit with a Gaussian-process noise model fitted to the residuals (data_test_robust.py), and the
# uranium stage with the ferromagnetic term recomputed for its own geometry (1_experiment_sim/u_stage_refine.py)
QAR = json.load(open(HAND / "2_indirect_search" / "data_test_robust.json"))
# rev 5b: the same fit repeated against three other published mass models (residual_analyzer/residual_analyzer.py)
MM = json.load(open(HERE.parent / "residual_analyzer" / "out" / "results.json"))
UREF = json.load(open(HAND / "1_experiment_sim" / "u_stage_refine.json"))
SIMKEY = {"U-238": "238U", "Pu-242": "242Pu", "Pu-244": "244Pu", "Cm-248": "248Cm", "Cf-252": "252Cf",
          "Pu-239": "242Pu"}   # rev 5m: no Pu-239 run; same mass, distance and capsule as the 242Pu run (kept last)
# rev 5: statistical calibration from 520 independent 90-day null/residual records (calibration_check.py); the
# first calibration's factor 0.85 (detection_map.py, one set of 60 runs) reproduces exactly but was a low draw.
CC = json.load(open(HAND / "1_experiment_sim" / "calibration_check.json"))
INST = json.load(open(HAND / "1_experiment_sim" / "instrument_calibration.json"))
CAL_V4 = DET["calibration"]["factor_vs_120d_estimate"]
CAL = CC["pooled_factor"]
PEND = SIM["pendulum"]
for X in CONF:
    NUC[X]["sigma_day"] = SIM["sources"][SIMKEY[X]]["sigmaP_1day"] * CAL
CAMP_V4 = {}
for label, c in DET["campaigns"].items():
    X = next(k for k, s in SIMKEY.items() if label.startswith(s))
    days = int(label.split(",")[-1].split()[0])
    CAMP_V4[X] = dict(S=2 * c["sigma_tot"], T5=5 * c["sigma_tot"])
    s_stat = NUC[X]["sigma_day"] / np.sqrt(days)
    s_tot = float(np.hypot(s_stat, c["sigma_sys"]))
    NUC[X].update(days=days, s_stat=s_stat, s_sys=c["sigma_sys"], s_tot=s_tot, S=2 * s_tot, T5=5 * s_tot)
# rev 5m: one gram of Pu-239 in the plutonium geometry. The simulation has no Pu-239 run: its statistical error is the
# plutonium one (same mass, distance and capsule), and its systematic error is that of Pu-242 with the decay-heat term
# for Pu-239's larger heat added in quadrature (dummy heater matched to 1%, the simulation's thermal model). The heat
# ratio uses Table II's heats; the simulation's Pu-242 heat is 3% higher, so the scaled terms err high.
_h = NUC["Pu-239"]["heat"] / NUC["Pu-242"]["heat"]
PU239 = dict(heat_ratio=_h, heat_push_1pct=SIM["budget"]["242Pu"]["Decay-heat radiation (heater 1%)"] * _h,
             heat_push_unmatched=SIM["budget"]["242Pu"]["Decay-heat radiation (unmatched)"] * _h,
             capsule_dT_K=SIM["sources"]["242Pu"]["capsule_dT_K"] * _h)
_v = NUC["Pu-239"]
_v.update(days=90, s_stat=_v["sigma_day"] / np.sqrt(90),
          s_sys=float(np.hypot(NUC["Pu-242"]["s_sys"], PU239["heat_push_1pct"])))
_v["s_tot"] = float(np.hypot(_v["s_stat"], _v["s_sys"]))
_v.update(S=2 * _v["s_tot"], T5=5 * _v["s_tot"])
PU239["sys_increase"] = _v["s_sys"] / NUC["Pu-242"]["s_sys"] - 1
# weapons-grade material, about 6% Pu-240 (NUBASE2020 6561 y, SF 5.80e-8, AME2020 Q_alpha 5.256 MeV, nu-bar 2.15):
# its extra heat, and the neutron push of its fission neutrons (the push per neutron of the Pu-242 source)
_a240 = np.log(2) / (6561.0 * YEAR) * NA / 240
PU239["wg_pu240_frac"] = 0.06
# rev 5r: per gram of material, the 6% Pu-240 replaces Pu-239 (was added on top of a full gram, which gave 22%)
PU239["wg_heat_increase"] = 0.06 * (_a240 * 5.256 * MEV / NUC["Pu-239"]["heat"] - 1)
PU239["wg_neutrons_per_s_g"] = 0.06 * _a240 * 5.80e-8 * 2.15          # push added below, after the neutron model
# uranium: detection_map.py carries plutonium's ferromagnetic term (1e-5) for every source. For 100 g at 3 cm the
# same spec, scaled with source mass (1e-9 A m^2 per gram), gives 4e-6; flip-only Newtonian term unchanged.
_u = UREF["flip_only_corrected_ferro"]["per_gram_spec"]
U_STAGE_V4 = dict(s_sys=DET["campaigns"]["238U 100 g, 30 d"]["sigma_sys"], S=CAMP_V4["U-238"]["S"])
_ut = float(np.hypot(NUC["U-238"]["s_stat"], _u["sigma_sys"]))
NUC["U-238"].update(s_sys=_u["sigma_sys"], s_tot=_ut, S=2 * _ut, T5=5 * _ut)
U_MULTISEP = UREF["flip_plus_multisep"]["good"]           # flip + five separations: removes 'dim', costs statistics
# named instruments (instrument_calibration.py): Eot-Wash EP fibre (kappa 2.34e-9, Q 5000) has 1.31x the thermal
# torque noise of the representative pendulum; the Vienna mg balance's measured acceleration noise
EW_RATIO = INST["analytic_thermal"]["eotwash_ep"]["ratio_to_representative"]
EOTWASH = {}
for X in ("Pu-244", "Cm-248", "U-238"):
    st = NUC[X]["sigma_day"] * EW_RATIO / np.sqrt(NUC[X]["days"])
    EOTWASH[X] = dict(sigma_day=NUC[X]["sigma_day"] * EW_RATIO, S=2 * float(np.hypot(st, NUC[X]["s_sys"])))
VIENNA = INST["vienna"]
# rev 5s: torque noise further above thermal (noise_scaling.py, common random numbers). The representative pendulum
# is 1.14x thermal in amplitude at the exchange frequency; real balances reach about 1.6-2x under good conditions.
NOISE = json.load(open(HAND / "1_experiment_sim" / "noise_scaling.json"))
# independent check of the statistical error: thermal torque noise of the pendulum, point-mass signal torque
KB = 1.380649e-23
f_sig = 1 / (2 * PEND["T_half"])
asd_th = np.sqrt(4 * KB * PEND["T"] * PEND["kappa"] / (PEND["Q"] * 2 * np.pi * f_sig))
tau_unit = G * 1e-3 * PEND["m_t"] / 0.02**2 * PEND["R_arm"]
SIG_DAY_THERMAL = float(asd_th / np.sqrt(86400.0) / tau_unit)       # per unit P, 1 g at 2 cm, one day
# residual Newtonian coupling (95th percentile, oxide pellet with a 1 %/mm density gradient), 244Pu
_nm = SIM["sources"]["244Pu"]["newton_mc"]
RESID = dict(raw=_nm["raw_p95_oxide"], flip=_nm["flip_p95_oxide"], multisep=SIM["multisep"]["fit_p95"],
             cm_mass_term=SIM["sources"]["248Cm"]["newton_mc"]["components_rms_oxide"]["M"],
             u_flip=SIM["sources"]["238U"]["newton_mc"]["flip_p95_oxide"])
# magnetic: dipole-dipole force between remanent moments mu on source and test body, coaxial worst case
MU_SPEC = 1e-9                                    # A m^2
P_FERRO_SPEC = float(6 * 1e-7 * MU_SPEC**2 / 0.02**4 / (G * 1e-3 * PEND["m_t"] / 0.02**2))
P_FERRO_WORST = SIM["budget"]["244Pu"]["Ferromagnetic contamination"]
HEAT_242 = (SIM["budget"]["242Pu"]["Decay-heat radiation (unmatched)"], SIM["budget"]["242Pu"]["Decay-heat radiation (heater 1%)"])
LAMFIT = SIM["lambda_fit"]

# --------------------------------------------------------------- neutron momentum push on the test body
# Spontaneous-fission neutrons (mean ~2 MeV) carry momentum away from the source; the 3 g copper test sphere
# intercepts its geometric cross-section of the flux and takes up EFF of the momentum (interaction probability
# over its mean chord), a steady push that mimics a repulsive anomaly. The transport-level simulation gives
# the same to within 5% (SIM budget).
M_N, E_N = 1.67493e-27, 2.0 * MEV
P_NEUT = np.sqrt(2 * M_N * E_N)                   # kg m/s per neutron
M_TEST, EFF = PEND["m_t"], 0.1
A_INT = float(np.pi * PEND["a_t"] ** 2)           # 0.58 cm^2
for X, (M, r) in CONF.items():
    v = NUC[X]
    rate = v["neutrons"] * M * 1e3                 # n/s from the example source
    F_n = EFF * rate / (4 * np.pi * r**2) * A_INT * P_NEUT
    F_N = G * M * M_TEST / r**2
    v["P_neutron"] = F_n / F_N

# --------------------------------------------------------------- gamma-ray momentum push (rev 5g)
# Gamma rays push the test body as the fission neutrons do. gamma_eff(E): fraction of the momentum of a parallel beam
# of energy E that the 3 g copper sphere takes up, from a seeded Monte Carlo: Compton scattering by Klein-Nishina (free
# electrons); the rest of the NIST total attenuation absorbs the photon (photoabsorption, plus a few per cent of
# coherent scattering counted as absorption); escaping photons carry their momentum away, electrons and fluorescence
# stay in the sphere. The push relative to the source's own gravity is, like the neutron push, independent of source
# mass and distance: P = sum_lines(eff I E/c) A / (4 pi G m_t), A = activity per kg. A tungsten capsule wall is counted
# by its uncollided transmission exp(-mu t), ignoring buildup (an underestimate of the push).
# Mass attenuation coefficients (MeV, cm^2/g, total with coherent): Hubbell & Seltzer, NIST SRD 126 (Cu Z=29, W Z=74).
CU_MU = np.array([[0.01, 215.9], [0.015, 74.05], [0.02, 33.79], [0.03, 10.92], [0.04, 4.862], [0.05, 2.613],
                  [0.06, 1.593], [0.08, 0.7630], [0.1, 0.4584], [0.15, 0.2217], [0.2, 0.1559], [0.3, 0.1119],
                  [0.4, 0.09413], [0.5, 0.08362], [0.6, 0.07625], [0.8, 0.06605], [1.0, 0.05901], [1.25, 0.05261],
                  [1.5, 0.04803], [2.0, 0.04205], [3.0, 0.03599]])
W_MU = np.array([[0.015, 138.9], [0.02, 65.73], [0.03, 22.73], [0.04, 10.67], [0.05, 5.949], [0.06, 3.713],
                 [0.069525 - 1e-9, 2.552], [0.069525, 11.23], [0.08, 7.810], [0.1, 4.438], [0.15, 1.581],
                 [0.2, 0.7844], [0.3, 0.3238], [0.4, 0.1925], [0.5, 0.1378], [0.6, 0.1093], [0.8, 0.08066],
                 [1.0, 0.06618], [1.25, 0.05561], [1.5, 0.05000], [2.0, 0.04433], [3.0, 0.04075]])   # K edge 69.525 keV
RHO_CU_G, RHO_W_G, ME_MEV, C_LIGHT = PEND["rho_t"] / 1e3, 19.3, 0.51099895, 299792458.0
E_PER_G_CU = NA * 29 / 63.546
_loglog = lambda tab, E: np.exp(np.interp(np.log(E), np.log(tab[:, 0]), np.log(tab[:, 1])))


def _sigma_kn(E):
    """Klein-Nishina cross-section per free electron (cm^2) at photon energy E (MeV)."""
    k = E / ME_MEV
    l = np.log1p(2 * k)
    return 2 * np.pi * 7.9407877e-26 * ((1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - l / k) + l / (2 * k)
                                        - (1 + 3 * k) / (1 + 2 * k) ** 2)


def _sample_kn(k, rng):
    out, todo = np.empty(k.size), np.arange(k.size)
    while todo.size:                              # rejection sampling of cos(theta); the density is at most 2
        mu = 2 * rng.random(todo.size) - 1
        eps = 1 / (1 + k[todo] * (1 - mu))
        acc = 2 * rng.random(todo.size) < eps * eps * (eps + 1 / eps - (1 - mu * mu))
        out[todo[acc]] = mu[acc]
        todo = todo[~acc]
    return out


def _rotate(d, cost, phi):
    sint = np.sqrt(np.maximum(1 - cost ** 2, 0))
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    s = np.sqrt(np.maximum(1 - dz ** 2, 1e-30))
    cp, sp = np.cos(phi), np.sin(phi)
    near = s < 1e-6
    out = np.column_stack([np.where(near, sint * cp, sint * (dx * dz * cp - dy * sp) / s + dx * cost),
                           np.where(near, sint * sp, sint * (dy * dz * cp + dx * sp) / s + dy * cost),
                           np.where(near, np.sign(dz) * cost, -sint * cp * s + dz * cost)])
    return out / np.linalg.norm(out, axis=1)[:, None]


def gamma_eff(E0, n=60000):
    """(momentum fraction taken up, first-interaction probability) for a parallel beam of E0 MeV on the sphere."""
    a = PEND["a_t"] * 100                          # cm
    rng = np.random.default_rng([20261117, int(round(E0 * 1e6))])
    r, ph = a * np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)
    pos = np.column_stack([r * np.cos(ph), r * np.sin(ph), -np.sqrt(np.maximum(a * a - r * r, 0))])
    d, E, pz_out, active, first = np.tile([0.0, 0.0, 1.0], (n, 1)), np.full(n, E0), np.zeros(n), np.arange(n), None
    while active.size:
        mu_t = _loglog(CU_MU, E[active]) * RHO_CU_G
        mu_c = np.minimum(_sigma_kn(E[active]) * E_PER_G_CU * RHO_CU_G, mu_t)
        p, dd = pos[active], d[active]
        b, c = np.einsum("ij,ij->i", p, dd), np.einsum("ij,ij->i", p, p) - a * a
        s = -np.log(rng.random(active.size)) / mu_t
        esc = s >= -b + np.sqrt(np.maximum(b * b - c, 0))
        first = 1 - esc.mean() if first is None else first
        pz_out[active[esc]] = E[active[esc]] * dd[esc, 2]
        hit = active[~esc]
        pos[hit] = p[~esc] + s[~esc, None] * dd[~esc]
        ci = hit[rng.random(hit.size) < mu_c[~esc] / mu_t[~esc]]   # the rest are absorbed: pz_out stays 0
        k = E[ci] / ME_MEV
        cost = _sample_kn(k, rng)
        E[ci] = E[ci] / (1 + k * (1 - cost))
        d[ci] = _rotate(d[ci], cost, 2 * np.pi * rng.random(ci.size))
        active = ci
    return float(np.mean(E0 - pz_out) / E0), float(first)


# Photon lines carrying more than 98% of the photon energy per decay (keV, photons per 100 decays), from ENSDF through
# the IAEA LiveChart of Nuclides (retrieved 2026-11-17; evaluations by C. D. Nesaraja, A = 249 (2023) and A = 251
# (2014), and M. S. Basunia, A = 241 (2006)); X-ray lines included
GAMMA_LINES = {
    "Cf-249": dict(thalf=351.0, A=249, lines=[(388.150, 66.0), (333.370, 14.8), (252.820, 2.55), (109.286, 3.878),
                                              (104.606, 2.477), (18.579, 13.75), (124.855, 1.964), (266.630, 0.707),
                                              (123.280, 1.453)]),
    "Cf-251": dict(thalf=898.0, A=251, lines=[(177.520, 17.3), (109.286, 19.95), (227.380, 6.8), (104.606, 12.74),
                                              (124.855, 10.10), (18.579, 53.29), (123.280, 7.472), (126.917, 2.630),
                                              (285.410, 1.13), (265.860, 0.43), (291.200, 0.30)]),
    "Am-241": dict(thalf=432.6, A=241, lines=[(59.541, 35.9), (17.136, 36.64), (26.345, 2.27)]),
    # the daughters of 244Pu in equilibrium (240U, 240mNp, 0.12% via 240Np), per 100 decays of 244Pu; ENSDF A = 240
    # (C. Morse, 2024)
    "Pu-244 daughters": dict(thalf=8.13e7, A=244, lines=[
        (554.6, 20.874), (597.4, 11.676), (1496.9, 1.3264), (1539.62, 0.838), (938.02, 1.2085), (817.89, 1.2764),
        (915.98, 1.0177), (758.61, 1.1775), (1445.3, 0.3795), (17.604, 27.085), (17.136, 26.902), (857.48, 0.4884),
        (606.1, 0.6713), (507.2, 0.6991), (302.98, 0.9988), (263.37, 1.1376), (1488.2, 0.1998), (1633.33, 0.1538),
        (251.47, 0.8579), (1590.5, 0.0969), (813.41, 0.1798), (900.37, 0.1598), (789.59, 0.1798), (928.55, 0.1498),
        (910.1, 0.1398), (841.11, 0.1498), (961.62, 0.1298), (860.0, 0.1397), (1046.62, 0.0969), (942.39, 0.0959),
        (1607.6, 0.0549), (989.2, 0.0809), (1131.0, 0.0619), (1604.8, 0.037), (895.3, 0.0609), (44.1, 1.0487),
        (189.7, 0.2397), (1321.1, 0.034), (1483.0, 0.027), (1428.3, 0.028), (1088.3, 0.032), (103.741, 0.3212),
        (1417.2, 0.023), (1667.6, 0.019), (1061.6, 0.029), (973.9, 0.031), (1305.8, 0.023), (1584.1, 0.017),
        (1911.4, 0.014), (1180.1, 0.02), (1515.9, 0.015), (1874.9, 0.012), (1223.0, 0.018), (1094.2, 0.02),
        (340.7, 0.0599)]),
    # rev 5m: 239Pu, ENSDF A = 239 (E. Browne and J. K. Tuli, 2014): the lines that carry 98% of the push bare (the
    # L X-rays, 51.6 keV) and 98% of it behind 1 mm of tungsten (the weak 130-770 keV gamma rays)
    "Pu-239": dict(thalf=2.411e4, A=239, lines=[
        (16.678, 4.29), (12.975, 0.0341), (51.624, 0.02722), (38.661, 0.01044), (98.439, 0.0057), (94.657, 0.0036),
        (111.238, 0.00207), (112.645, 0.0028), (129.296, 0.00631), (413.713, 0.001466), (375.054, 0.001554),
        (345.013, 0.000556), (332.845, 0.000494), (393.14, 0.000348), (380.191, 0.000305), (382.75, 0.000259),
        (451.481, 0.000189), (392.53, 0.000205), (203.55, 0.000569), (422.598, 0.000122), (336.113, 0.000112),
        (367.073, 8.9e-05), (368.554, 8.8e-05), (341.506, 6.6e-05), (345.014, 5e-05), (323.84, 5.4e-05),
        (320.862, 5.4e-05), (255.384, 8e-05), (297.46, 5e-05), (195.679, 0.000107), (426.68, 2.3e-05),
        (645.94, 1.5e-05), (311.78, 2.6e-05), (189.36, 8.3e-05), (263.95, 2.7e-05), (658.86, 1e-05),
        (171.393, 0.00011), (639.99, 9e-06), (243.38, 2.5e-05), (769.37, 7e-06), (161.45, 0.000123),
        (144.201, 0.000283), (361.89, 1.2e-05), (179.22, 6.6e-05), (316.41, 1.3e-05), (652.05, 7e-06),
        (445.72, 9e-06), (769.15, 5e-06), (411.2, 7e-06), (237.77, 1.4e-05), (703.68, 4e-06), (399.53, 6e-06),
        (225.42, 1.5e-05), (481.66, 5e-06)]),
}
GAMMA_EFF = {Ek: gamma_eff(Ek / 1e3) for Ek in sorted({Ek for g in GAMMA_LINES.values() for Ek, _ in g["lines"]})}
for Ek, (eff, p1) in GAMMA_EFF.items():         # the first-interaction probability against the chord formula
    _c = 2 * _loglog(CU_MU, Ek / 1e3) * RHO_CU_G * PEND["a_t"] * 100
    assert abs(p1 - (1 - 2 * (1 / _c ** 2 - np.exp(-_c) * (1 / _c ** 2 + 1 / _c)))) < 0.01, Ek


def gamma_push(X, wall_W_mm=0.0):
    g = GAMMA_LINES[X]
    act = np.log(2) / (g["thalf"] * YEAR) * NA / g["A"] * 1e3               # Bq per kg
    flux = sum(I / 100 * Ek * 1e-3 * MEV / C_LIGHT * GAMMA_EFF[Ek][0]
               * float(np.exp(-_loglog(W_MU, Ek / 1e3) * RHO_W_G * wall_W_mm / 10)) for Ek, I in g["lines"])
    return float(act * flux * A_INT / (4 * np.pi * G * M_TEST))


W_TRANS = {f"{Ek:g}": {f"{t:g}": float(np.exp(-_loglog(W_MU, Ek / 1e3) * RHO_W_G * t / 10)) for t in (0.05, 1, 3)}
           for Ek in (59.541, 177.52, 388.15)}
GAMMA = dict(eff={f"{Ek:g}": v[0] for Ek, v in GAMMA_EFF.items()}, p_first={f"{Ek:g}": v[1] for Ek, v in GAMMA_EFF.items()},
             W_transmission=W_TRANS,
             push={X: {f"{t:g} mm W": gamma_push(X, t) for t in (0.0, 0.05, 0.5, 1.0, 3.0)} for X in GAMMA_LINES})
# americium-241 in plutonium: per 0.1% by mass, against the plutonium stages' reach (Pu-244, 90 days)
GAMMA["Am241_per_0.1pct"] = {k: 1e-3 * v for k, v in GAMMA["push"]["Am-241"].items()}
GAMMA["Am241_per_0.1pct_vs_Pu_reach"] = GAMMA["Am241_per_0.1pct"]["0 mm W"] / NUC["Pu-244"]["S"]
# 244Pu's own daughters: always present, so every plutonium-244 source carries this push
GAMMA["Pu244_daughters_vs"] = {k: dict(reach=v / NUC["Pu-244"]["S"], neutron=v / NUC["Pu-244"]["P_neutron"])
                               for k, v in GAMMA["push"]["Pu-244 daughters"].items()}
# prompt fission gamma rays: 6.84 MeV in 7.8 photons per fission for 252Cf (Verbinski et al., PRC 7, 1173 (1973)),
# used for every source; fission products add at most as much again once they build up
FISSION_GAMMA = dict(E_tot=6.84, E_mean=0.88)
_feff = gamma_eff(FISSION_GAMMA["E_mean"])[0]


def fission_gamma_push(X, wall_W_mm=0.0):
    v = NUC[X]
    rate = v["act"] * 1e3 * v["sf"]                                          # fissions per s per kg
    t = float(np.exp(-_loglog(W_MU, FISSION_GAMMA["E_mean"]) * RHO_W_G * wall_W_mm / 10))
    return float(rate * FISSION_GAMMA["E_tot"] * MEV / C_LIGHT * _feff * t * A_INT / (4 * np.pi * G * M_TEST))


GAMMA["fission"] = {X: {f"{t:g} mm W": fission_gamma_push(X, t) for t in (0.0, 1.0, 3.0)}
                    for X in ("Pu-242", "Pu-244", "Cm-248")}
GAMMA["fission"]["eff"] = _feff
GAMMA["fission_Cm248_vs_neutron"] = GAMMA["fission"]["Cm-248"]["0 mm W"] / NUC["Cm-248"]["P_neutron"]
GAMMA["fission_Cm248_vs_reach"] = GAMMA["fission"]["Cm-248"]["0 mm W"] / NUC["Cm-248"]["S"]
GAMMA["impurity_heat"] = IMPURITY_HEAT

# --------------------------------------------------------------- the other long-lived californium isotopes (rev 5g)
# NUBASE2020: 249Cf 351 y, SF 5.0e-7 %; 250Cf 13.08 y, SF 0.077 %; 251Cf 898 y, SF not observed ('SF ?').
# Q_alpha (ENSDF): 6.2933 and 6.177 MeV. Charge relative to 248Cm at the widest Gaussians of Table I.
CF_OTHER = {"Cf-249": dict(Z=98, N=151, thalf=351.0, sf=5.0e-9, Qa=6.2933),
            "Cf-250": dict(Z=98, N=152, thalf=13.08, sf=7.7e-4),
            "Cf-251": dict(Z=98, N=153, thalf=898.0, sf=None, Qa=6.177)}
FISS_CM = NUC["Cm-248"]["act"] * NUC["Cm-248"]["sf"]                       # fissions / s / g
FISS_CF252 = NUC["Cf-252"]["act"] * NUC["Cf-252"]["sf"]
CALI = {}
for _X, _c in CF_OTHER.items():
    _act = np.log(2) / (_c["thalf"] * YEAR) * NA / (_c["Z"] + _c["N"])     # Bq per gram
    _row = dict(d2=d2(_c["Z"], _c["N"]), thalf=_c["thalf"], act=_act,
                fissions=None if _c["sf"] is None else _act * _c["sf"],
                charge_vs_Cm248={f"{Pc:.0e}": float(np.exp((NUC["Cm-248"]["d2"] - d2(_c["Z"], _c["N"])) / D2U
                                                           * np.log(Pc / B_U))) for Pc in (PC_LO, PC_HI)})
    if "Qa" in _c:
        _row["heat"] = _act * _c["Qa"] * MEV
        _row["heat_vs_Cm248"] = _row["heat"] / NUC["Cm-248"]["heat"]
    CALI[_X] = _row
CALI["Cf-249"]["fission_ratio_Cm248"] = FISS_CM / CALI["Cf-249"]["fissions"]
CALI["Cf-249"]["gamma_push_vs_Cm248_neutron"] = {k: v / NUC["Cm-248"]["P_neutron"]
                                                  for k, v in GAMMA["push"]["Cf-249"].items()}
CALI["Cf-251"]["gamma_push_vs_Cm248_neutron"] = {k: v / NUC["Cm-248"]["P_neutron"]
                                                  for k, v in GAMMA["push"]["Cf-251"].items()}
# contamination of a 251Cf source that fissions as often per gram as 248Cm itself, and the push of 1 ppm of 252Cf
CALI["Cf-251"].update(cf250_frac_matching_Cm=FISS_CM / CALI["Cf-250"]["fissions"],
                      cf252_frac_matching_Cm=FISS_CM / FISS_CF252,
                      cf252_1ppm_push=1e-6 * NUC["Cf-252"]["P_neutron"])

# --------------------------------------------------------------- headline numbers
# --------------------------------------------------------------- rev 5m: a charge that follows the decay rate (Sec. III C)
# |P_X| = B_U (lambda_alpha,X / lambda_alpha,U)^k, partial alpha-decay rates from NUBASE2020 (half-lives in NUC, alpha
# branches below). Measured masses bound k: validation/decay_rate_bound.py puts the self-energy of Eq. (self), with q0
# at its bound, against every AME2020 measured mass and Q_alpha that has a NUBASE2020 half-life.
ALPHA_BR = {"Pb-208": 0.0, "Th-232": 1.0, "U-238": 1.0, "Pu-239": 1.0, "Pu-242": 1.0, "Pu-244": 0.99877,
            "Cm-248": 0.9161, "Cf-252": 0.96908}
DRB = json.load(open(HERE.parent / "validation" / "out" / "decay_rate_bound.json"))
assert abs(DRB["B_U"] / B_U - 1) < 1e-9
_drb = DRB["results"]
K_DEC = round(_drb["all measured"]["1 MeV"]["k_max"], 2)          # 0.60, set by 8Be
K_DEC_LO = 0.3                                                     # the lower value quoted in the text


def rate_ratio(X):
    v = NUC[X]
    if v.get("thalf") is None or ALPHA_BR[X] == 0:
        return 0.0
    return NUC["U-238"]["thalf"] / v["thalf"] * ALPHA_BR[X] / ALPHA_BR["U-238"]


def p_decay(X, k=K_DEC):
    r = rate_ratio(X)
    return 0.0 if r == 0 else float(B_U * r ** k)


def k_decay(X, level="S"):
    """the smallest k at which the decay-rate benchmark puts X at its reach S (or its 5-sigma threshold T5)"""
    return float(np.log(NUC[X][level] / B_U) / np.log(rate_ratio(X)))


# the smooth properties, for comparison: how much each grows from 238U to the island centre, and the power a charge
# following it needs to put 244Pu at its reach (as in the power law, Eq. power)
_SMOOTH = {"Z": lambda Z, N: Z, "N": lambda Z, N: N, "A": lambda Z, N: Z + N, "Z^2/A": lambda Z, N: Z * Z / (Z + N),
           "Z^2/A^(1/3)": lambda Z, N: Z * Z / (Z + N) ** (1 / 3)}
SMOOTH = {}
for _k, _fn in _SMOOTH.items():
    _u, _p, _c = _fn(92, 146), _fn(94, 150), _fn(ZC, NC)
    SMOOTH[_k] = dict(growth_to_centre=_c / _u - 1, k_for_Pu244_S=float(np.log(NUC["Pu-244"]["S"] / B_U) / np.log(_p / _u)))
PU239["wg_neutron_push"] = PU239["wg_neutrons_per_s_g"] * NUC["Pu-242"]["P_neutron"] / NUC["Pu-242"]["neutrons"]
PU239["gamma_push"] = GAMMA["push"]["Pu-239"]
DECAY = dict(k_max=_drb["all measured"]["1 MeV"]["k_max"],
             k_max_band=[_drb["all measured"]["0.1 MeV"]["k_max"], _drb["all measured"]["10 MeV"]["k_max"]],
             k_max_slow=_drb["living at least 1 ns"]["1 MeV"]["k_max"],
             k_max_slow_band=[_drb["living at least 1 ns"]["0.1 MeV"]["k_max"],
                              _drb["living at least 1 ns"]["10 MeV"]["k_max"]],
             witness=_drb["all measured"]["1 MeV"], witness_slow=_drb["living at least 1 ns"]["1 MeV"],
             Be8=DRB["Be8"], superheavy_decades=DRB["superheavy_alpha_rate_decades"], k_table=K_DEC,
             rate_ratio={X: rate_ratio(X) for X in ALPHA_BR},
             pred={X: dict(lo=p_decay(X, K_DEC_LO), hi=p_decay(X)) for X in ALPHA_BR},
             k_needed={X: dict(S=k_decay(X), T5=k_decay(X, "T5")) for X in ("Pu-239", "Pu-242", "Pu-244", "Cm-248")},
             ratio_242_244={str(k): p_decay("Pu-242", k) / p_decay("Pu-244", k) for k in (K_DEC_LO, K_DEC)},
             x_reach={X: p_decay(X) / NUC[X]["S"] for X in ("Pu-239", "Pu-242", "Pu-244", "Cm-248")},
             smooth=SMOOTH, pu239=PU239)

# --------------------------------------------------------------- rev 5o: a charge that follows the fission rate (Sec. III D)
# |P_X| = B_U (lambda_SF,X / lambda_SF,U)^k, spontaneous-fission branches of NUC (NUBASE2020); thorium's branch is only
# a limit and lead is stable, so both carry none. validation/decay_rate_bound.py bounds k with the measured masses.
_fis = DRB["fission"]
K_FIS = round(_fis["results"]["all measured"]["1 MeV"]["k_max"], 2)       # 0.71, set by the Q_alpha of 286Fl
K_FIS_LO = 0.3


def sf_ratio(X):
    v = NUC[X]
    if v.get("thalf") is None or not v.get("sf"):
        return 0.0
    return NUC["U-238"]["thalf"] / v["thalf"] * v["sf"] / NUC["U-238"]["sf"]


def p_fission(X, k=K_FIS):
    r = sf_ratio(X)
    return 0.0 if r == 0 else float(B_U * r ** k)


def k_fission(X, level="S"):
    return float(np.log(NUC[X][level] / B_U) / np.log(sf_ratio(X)))


FISSION = dict(k_max=_fis["results"]["all measured"]["1 MeV"]["k_max"],
               k_max_band=[_fis["results"]["all measured"]["0.1 MeV"]["k_max"],
                           _fis["results"]["all measured"]["10 MeV"]["k_max"]],
               witness=_fis["results"]["all measured"]["1 MeV"], superheavy_decades=_fis["superheavy_fission_rate_decades"],
               k_table=K_FIS, ratio={X: sf_ratio(X) for X in ALPHA_BR},
               pred={X: dict(lo=p_fission(X, K_FIS_LO), hi=p_fission(X)) for X in ALPHA_BR},
               k_needed={X: dict(S=k_fission(X), T5=k_fission(X, "T5")) for X in ("Pu-242", "Pu-244", "Cm-248")},
               x_reach={X: p_fission(X) / NUC[X]["S"] for X in ("Pu-239", "Pu-242", "Pu-244", "Cm-248")},
               pu242_over_pu244=sf_ratio("Pu-242") / sf_ratio("Pu-244"))

# rev 5n: charges that follow shell closure (Sec. III D), from validation/shell_closure_scan.py: lead, a test body in
# Smith et al., is held to dBmu x B_U; the summary carries the largest signal / reach per range and source
SHELL = json.load(open(HERE.parent / "validation" / "out" / "shell_closure_scan.json"))["summary"]
assert abs(SHELL["by_range"]["0.5"]["B_U"] / B_U - 1) < 1e-9 and abs(SHELL["by_range"]["0.5"]["B_Pb"] / (B_U * 1.0e-3) - 1) < 1e-9

PRED_X = ("Pb-208", "Th-232", "U-238", "Pu-239", "Pu-242", "Pu-244", "Cm-248", "Cf-252")
pred = {X: dict(gauss_lo=p_gauss_widest(X, PC_LO), gauss_hi=p_gauss_widest(X, PC_HI),
                power_lo=p_power(X, PC_LO), power_hi=p_power(X, PC_HI),
                decay_lo=p_decay(X, K_DEC_LO), decay_hi=p_decay(X),
                fission_lo=p_fission(X, K_FIS_LO), fission_hi=p_fission(X)) for X in PRED_X}


def pc_gauss(n, X="Pu-244", level="S", BU=B_U, supp=1.0):
    """P_c above which the widest Gaussian puts X at n times its reach S (or its 5-sigma threshold T5).
    supp = f(r, lambda): Yukawa suppression of the signal at the measuring distance."""
    w, S = NUC[X]["w"], NUC[X][level] / supp
    return float(np.exp((np.log(n * S) - w * np.log(BU)) / (1 - w)))


def pc_power(n, X="Pu-244", level="S"):
    S = NUC[X][level]
    return float(B_U * (AC / 238) ** (np.log(n * S / B_U) / np.log(NUC[X]["A"] / 238)))


# range dependence: the uranium bound weakens at short range, so the 5-sigma threshold in P_c falls
RANGE = {}
for lam, BU in B_OF_LAM.items():
    if lam < 0.05 or lam > 0.5:
        continue
    supp = float(f(NUC["Pu-244"]["r"], lam))
    RANGE[lam] = dict(B_U=BU, supp=supp, Pc_T5=pc_gauss(1, "Pu-244", "T5", BU, supp),
                      Pc_S=pc_gauss(1, "Pu-244", "S", BU, supp),
                      U_stage_T5=NUC["U-238"]["T5"] / float(f(NUC["U-238"]["r"], lam)))
K_T5 = float(np.log(NUC["Pu-244"]["T5"] / B_U) / np.log(244 / 238))       # power-law steepness needed for 5 sigma
# how much the depleted-uranium stage (reach S) improves on the present uranium bound at short range
U_IMPROVE = {lam: B_OF_LAM[lam] * float(f(NUC["U-238"]["r"], lam)) / NUC["U-238"]["S"] for lam in (0.02, 0.05, 0.1, 0.5)}

# --------------------------------------------------------------- superheavy Q_alpha bound (Sec. II D)
# Self-energy -(3/5) G (q_nuc M)^2 / R shifts alpha-decay energies; fitted to 68 measured Q_alpha (Z = 104-118,
# AME2020) against the WS-type mass model of Ma et al. (2_indirect_search/data_test.py,
# reproduces on rerun). 95% upper limit on the central charge q_c, then P_c = q_c * q0 with q0 at its own bound.
Q0_B = float(np.sqrt(B_U))                                     # baryon-like: the uranium test bounds q0^2 too
Q0_M = QA["scan"][0]["Pc95_mass"] / QA["scan"][0]["qc95"]      # mass-like: inverse-square-law tests only
QSCAN_IND = {r["sigma"]: r for r in QA["scan"]}               # rev 4: independent residuals
QSCAN = {r["sigma"]: r for r in QAR["scan_gp"]}                # rev 5: GP noise model (used from here on)
_band = [s for s in QSCAN if 4.5 <= s <= 10.0]
QALPHA = dict(n=QA["stats"]["n_she"], n_fit=QAR["n"], rms_keV=QA["stats"]["rms_th_she"], max_z=QA["max_z"],
              max_z_gp=QAR["max_z_gp"], q0_B=Q0_B, q0_M=Q0_M,
              qc95={s: QSCAN[s]["qc95"] for s in (4.5, 5.0, 7.0, 10.0)},
              Pc95_B={s: QSCAN[s]["qc95"] * Q0_B for s in (4.5, 5.0, 7.0, 10.0)},
              Pc95_M={s: QSCAN[s]["qc95"] * Q0_M for s in (4.5, 5.0, 7.0, 10.0)},
              qc95_band=(min(QSCAN[s]["qc95"] for s in _band), max(QSCAN[s]["qc95"] for s in _band)),
              Pc95_B_band=(min(QSCAN[s]["qc95"] for s in _band) * Q0_B, max(QSCAN[s]["qc95"] for s in _band) * Q0_B),
              shift_keV_at_limit=QSCAN[5.0]["beta95"] * QSCAN[5.0]["template_max_keV"],
              independent_Pc95_B={s: QSCAN_IND[s]["qc95"] * Q0_B for s in (4.5, 5.0, 7.0, 10.0)},
              ratio_gp_to_independent=QAR["ratio_qc95_to_independent"]["gp"],
              gp=QAR["gp"], corr=QAR["correlation"],
              sd_understated=QAR["independent_fit_sd_understated_by"])
assert abs(QSCAN[5.0]["Pc95_baryon"] / (QSCAN[5.0]["qc95"] * Q0_B) - 1) < 1e-3
# Mass-model dependence of that limit: the paper's own selection (484 Q_alpha, Z >= 84) and noise treatment, with
# the mass model swapped. Only models tabulated up to Z = 118 can test the island.
_mm = {m: {w["width"]: w for w in r["window"]} for m, r in MM["window_paper_selection_Z_ge_84"].items()
       if r["Zmax"] >= 118}
assert MM["validation"]["passed"] and abs(_mm["WS-Ma2019"][5.0]["qc95"] / QSCAN[5.0]["qc95"] - 1) < 1e-6
assert all(r["n"] == QAR["n"] for m, r in MM["window_paper_selection_Z_ge_84"].items() if m in _mm)
_others = [m for m in _mm if m != "WS-Ma2019"]
QALPHA["mass_models"] = dict(
    models=list(_mm), others=_others,
    Pc95_B_w5={m: _mm[m][5.0]["Pc95_baryon"] for m in _mm},
    Pc95_B_w5_others=(min(_mm[m][5.0]["Pc95_baryon"] for m in _others), max(_mm[m][5.0]["Pc95_baryon"] for m in _others)),
    Pc95_B_all=(min(w["Pc95_baryon"] for m in _mm for w in _mm[m].values()),
                max(w["Pc95_baryon"] for m in _mm for w in _mm[m].values())),
    rel_to_paper_w5=(min(_mm[m][5.0]["qc95"] for m in _others) / QSCAN[5.0]["qc95"] - 1,
                     max(_mm[m][5.0]["qc95"] for m in _others) / QSCAN[5.0]["qc95"] - 1),
    z_range=(min(w["z"] for m in _mm for w in _mm[m].values()), max(w["z"] for m in _mm for w in _mm[m].values())),
    z_max_others=max(w["z"] for m in _others for w in _mm[m].values()))

# passive-test suppression: baryon number per atomic mass unit, Cu vs Pb
B_per_u = {"Cu": 0.6915 * 63 / 62.92960 + 0.3085 * 65 / 64.92779, "Pb": 208 / 207.97665}
dBmu = B_per_u["Cu"] - B_per_u["Pb"]

# --------------------------------------------------------------- heavy source on a heavy test body (Sec. II E)
# A test body of the source's own nuclide X sees q_nuc(X) [q0 + q_nuc(X)] = P_X + q_nuc(X)^2 (Eq. heavy), which
# survives q0 = 0, where every ordinary test body is blind. Gaussian benchmark: q_nuc(X) = q_c exp(-d_X^2 / 2 s^2).
Q0B2 = Q0_B ** 2                                   # q0^2 <= 7.3e-6 in the baryon reading (Sec. II C)
# gain over an ordinary test body is |1 + q_nuc/q0| >= |P_X|/q0^2 - 1 (opposite signs, the worse case)
HEAVY_GAIN = {X: dict(at_S=NUC[X]["S"] / Q0B2 - 1, at_T5=NUC[X]["T5"] / Q0B2 - 1) for X in ("Pu-242", "Pu-244")}
# and at other ranges: q0^2 <= B_U(lambda) (the same test), the reach scaled by the Yukawa factor at the source
# distance as in RANGE; a value at or below zero means an opposite-sign test-body charge can cancel the signal
HEAVY_GAIN_LAM = {lam: NUC["Pu-244"]["S"] / float(f(NUC["Pu-244"]["r"], lam)) / B_OF_LAM[lam] - 1
                  for lam in sorted(B_OF_LAM)}


def qnuc_max(X, s):
    """Existing, q0 = 0: the decay energies cap q_c, and through the switch every nuclide's charge."""
    return QSCAN[s]["qc95"] * float(np.exp(-NUC[X]["d2"] / (2 * s * s)))


# Existing, q0 = 0: the uranium-attractor test through lead's own charge, |q_U q_Pb| I_A <= 5.7e-13 / 9.2e-5, with the
# attractor factor I_A taken from Table X's sensitivity (Sigma(lambda) in the paper) = d(B/mu) (B/mu)_U I_A at 0.5 m.
I_A = S_B_TABLE[LAM] / (dBmu * (238 / 238.05079))
D2PB = d2(82, 126)
PB_NAT = [(122, 0.014), (124, 0.241), (125, 0.221), (126, 0.524)]   # natural lead: N and mass fraction


def qc_via_lead(s):
    """Lead's charge averaged over its natural isotopes, relative to uranium's, for a switch of width s."""
    r = sum(wt * float(np.exp(-(d2(82, n) - D2U) / (2 * s * s))) for n, wt in PB_NAT)
    return float(np.sqrt(PASSIVE_95 / (I_A * r))) * float(np.exp(D2U / (2 * s * s)))


# Existing but site-dependent: uranium dropped beside copper (Niebauer et al., PRL 59, 609 (1987)), |da|/g < 5e-10.
# Uranium in the ground (mass fraction FF_CG in a half-space of density 2400 kg/m^3 a height FF_H below) acts on the
# uranium body through q_U^2: |da|/g = FF_CG q_U^2 2 pi G rho lambda exp(-FF_H / lambda) / g. Not in Table IV.
FF_CG, FF_H = 1e-6, 0.5
FF_QU2 = 5e-10 / (FF_CG * 2 * np.pi * G * 2400.0 * LAM * np.exp(-FF_H / LAM) / 9.80)


def qc_free_fall(s):
    return float(np.sqrt(FF_QU2)) * float(np.exp(D2U / (2 * s * s)))


def qc_from_pair(X, s):
    """Proposed, q0 = 0: a null result at a campaign's reach S with a matching test body gives q_nuc(X)^2 <= S."""
    return float(np.sqrt(NUC[X]["S"])) * float(np.exp(NUC[X]["d2"] / (2 * s * s)))


# with q0 at its bound and of opposite sign, |q (q0 + q)| <= S allows q up to (q0 + sqrt(q0^2 + 4 S)) / 2
HEAVY_Q0_WEAKEN = max(float((Q0_B + np.sqrt(Q0B2 + 4 * NUC[X]["S"])) / 2 / np.sqrt(NUC[X]["S"]))
                      for X in ("U-238", "Pu-242", "Pu-244"))
HEAVY_W = [4.5, 5.0, 6.0, 7.0, 10.0]
HEAVY_TAB = {}
for _s in HEAVY_W:
    _r = dict(qalpha=QSCAN[_s]["qc95"], via_lead=qc_via_lead(_s), U=qc_from_pair("U-238", _s),
              Pu242=qc_from_pair("Pu-242", _s), Pu244=qc_from_pair("Pu-244", _s))
    _best = min(_r["qalpha"], _r["via_lead"])
    _r.update(best_existing=_best, gain_U=_best / _r["U"], gain_Pu242=_best / _r["Pu242"], gain_Pu244=_best / _r["Pu244"])
    HEAVY_TAB[_s] = _r
# lead's limit overtakes the decay energies from this width on (0.25 grid of the decay-energy scan)
LEAD_CROSS = min(s for s in sorted(QSCAN) if qc_via_lead(s) < QSCAN[s]["qc95"])
# how much the free fall would tighten the existing limits, at most. rev 5i: on a fine grid of widths (the decay-energy
# limit interpolated in log between its 0.25 grid), because the peak sits where lead's limit crosses the decay energies,
# near 5.8, between grid points; the 0.25 grid alone gave 33 at 5.75
_SGF = np.linspace(3.0, 12.0, 361)
_QSA = np.array(sorted(QSCAN))
_FFG = (np.minimum(10 ** np.interp(_SGF, _QSA, [np.log10(QSCAN[x]["qc95"]) for x in _QSA]),
                   [qc_via_lead(s) for s in _SGF]) / np.array([qc_free_fall(s) for s in _SGF]))
FF_GAIN_MAX, FF_BEST_W = float(_FFG.max()), float(_SGF[np.argmax(_FFG)])
# Stage 1 claim: the uranium pair beats the best existing limit at widths 6 and 7 by more than 1e3, free fall or not
U_STAGE_MIN_GAIN = min(min(QSCAN[s]["qc95"], qc_via_lead(s), qc_free_fall(s)) / qc_from_pair("U-238", s)
                       for s in (6.0, 7.0))
assert U_STAGE_MIN_GAIN > 1e3, U_STAGE_MIN_GAIN
# the trace-uranium difference between the attractor test's Cu and Pb test bodies above which that test would itself
# bound q_U^2 below the decay-energy value at width 5 (the "28%" statement)
TRACE_DC = PASSIVE_95 / (I_A * qnuc_max("U-238", 5.0) ** 2)

# --------------------------------------------------------------- Stage 1 with a uranium test body (Sec. IV C, IV D)
# Time-domain checks in ../heavy_pair (heavy_pair_sim.py, u238_pair_campaign.py; common random numbers): a test body
# of the source's own nuclide leaves every campaign's reach unchanged, and the uranium reach holds from 5 cm to 10 m.
HP = json.load(open(HERE.parent / "heavy_pair" / "out" / "heavy_pair_sim.json"))
UP = json.load(open(HERE.parent / "heavy_pair" / "out" / "u238_pair_campaign.json"))
MATCH_REACH_CHANGE = max(abs(v["S_ratio"] - 1) for v in HP["heavy_over_copper"].values())
_u30 = [c["S"] for k, c in UP["campaigns"].items() if k.startswith("U 30 d")]
U_PAIR_REACH_RANGE = (min(_u30), max(_u30))
assert abs(UP["campaigns"]["U 30 d 0.5 m"]["S"] / NUC["U-238"]["S"] - 1) < 0.01   # same reach as Table III
# Independent of any switch: with q0 = 0 a null result limits q_U^2 to the reach. Existing switch-independent limits:
# the free fall above (site-dependent) and the normal density of large uranium ingots (radius 10 cm, density known to
# 0.1%, bulk modulus 100 GPa). The pair beats the route through P_U (Eq. BU) when |q0| < B_U / sqrt(S).
BULK_QU2 = 1e-3 * 1e11 / (2 * np.pi / 3 * G * 19050.0 ** 2 * 0.10 ** 2)
U_PAIR = dict(qU2_limit=NUC["U-238"]["S"], reach_5cm_10m=U_PAIR_REACH_RANGE, free_fall_qU2=FF_QU2,
              bulk_qU2=BULK_QU2, q0_crossover=B_U / float(np.sqrt(NUC["U-238"]["S"])), q0_bound=Q0_B,
              match_reach_change=MATCH_REACH_CHANGE)
# Iron in a 3 g test body of any material, as 1%-remanent precipitates (the paper's worst case), against the 1e-9 A m^2
# requirement; and what 10 ppm would fake with the uranium source at its own (per-gram) requirement
FE_MOMENT_PER_PPM = 1e-6 * PEND["m_t"] / 7874.0 * 1.71e6 * 0.01
FE_PPM_AT_SPEC = 1e-9 / FE_MOMENT_PER_PPM
FE_10PPM_FAKE_U = UREF["ferro"]["238U"]["per_gram_spec"] * 10 * FE_MOMENT_PER_PPM / 1e-9
# Beta momentum from the 100 g uranium source: 234mPa (in equilibrium with 238U, mean kinetic energy 0.82 MeV),
# upper bound with no capsule attenuation, point-source estimate P = N p a_t^2 / (4 G M m_t) (the full simulation in
# ../heavy_pair gives the same, 1.9e-7 with copper)
_pc = float(np.sqrt(0.82 ** 2 + 2 * 0.82 * 0.51099895)) * MEV / 299792458.0
_nb = NUC["U-238"]["act"] * NUC["U-238"]["M"] * 1e3
_at_u = (3 * PEND["m_t"] / (4 * np.pi * 19050.0)) ** (1 / 3)
BETA_PUSH = {t: _nb * _pc * a * a / (4 * G * NUC["U-238"]["M"] * PEND["m_t"])
             for t, a in (("Cu", PEND["a_t"]), ("U", _at_u))}
assert abs(BETA_PUSH["Cu"] / UP["budget"]["Cu"]["extra"]["beta_push_source_upper"] - 1) < 0.05
U_PAIR.update(fe_ppm_at_spec=FE_PPM_AT_SPEC, fe_10ppm_fake=FE_10PPM_FAKE_U, beta_rate=_nb, beta_push=BETA_PUSH)
# 244Pu: 240mNp betas (0.996 per decay, endpoint 2.18 MeV), summed momentum 1.026 MeV/c per 244Pu decay (ENSDF), same
# point-source upper bound; tungsten that stops them: Katz-Penfold range of a 2.18 MeV electron
_p_pu = 1.026 * MEV / 299792458.0
BETA_PU244 = dict(push_Cu=NUC["Pu-244"]["act"] * 1e3 * _p_pu * PEND["a_t"] ** 2 / (4 * G * PEND["m_t"]),
                  stop_W_mm=0.412 * 2.18 ** (1.265 - 0.0954 * np.log(2.18)) / 19.3 * 10,
                  test_body_per_s=NUC["Pu-244"]["act"] * 0.996 * PEND["m_t"] * 1e3)
BETA_PU244["test_body_A"] = BETA_PU244["test_body_per_s"] * 1.602176634e-19
BETA_PU244["push_vs_reach"] = BETA_PU244["push_Cu"] / NUC["Pu-244"]["S"]
# test-body systematics: the paramagnetic term scales with the test body's |chi| x volume (active_mass_sim values:
# Cu -9.63e-6 at 8960 kg/m^3, U 4.1e-4 at 19050, Pu 4.4e-4 at 15920); the remanent-moment requirement is unchanged
_chiV = lambda chi, rho: abs(chi) * PEND["m_t"] / rho
PARA_FACTOR = {"U-238": _chiV(4.1e-4, 19050.) / _chiV(PEND["chi_t"], PEND["rho_t"]),
               "Pu-244": _chiV(4.4e-4, 15920.) / _chiV(PEND["chi_t"], PEND["rho_t"])}
PARA_HEAVY = {"U-238": PARA_FACTOR["U-238"] * SIM["budget"]["238U"]["Paramagnetic (100 nT)"],
              "Pu-244": PARA_FACTOR["Pu-244"] * SIM["budget"]["244Pu"]["Paramagnetic (100 nT)"]}
# betas of 234mPa (in equilibrium with 238U) from a 3 g depleted-uranium test body; all escaping is the worst case
BETA_DU = NUC["U-238"]["act"] * PEND["m_t"] * 1e3
# Paramagnetic term of a uranium test body with no magnetic shielding (50 uT instead of 100 nT; it goes as B0^2), and
# the shielding factor that keeps it below 1% of the uranium stage's systematic error
U_PAIR.update(para_unshielded=PARA_HEAVY["U-238"] * (50e-6 / 1e-7) ** 2,
              shield_for_1pct=50e-6 / (1e-7 * float(np.sqrt(0.01 * NUC["U-238"]["s_sys"] / PARA_HEAVY["U-238"]))))
HEAVY = dict(gain=HEAVY_GAIN, q0_weaken=HEAVY_Q0_WEAKEN, I_A=I_A, d2_Pb=D2PB, lead_crossover_width=LEAD_CROSS,
             table={str(s): r for s, r in HEAVY_TAB.items()},
             qnuc_max_w5={X: qnuc_max(X, 5.0) for X in ("U-238", "Pu-244")},
             allowed_same_material_w5={X: qnuc_max(X, 5.0) ** 2 for X in ("U-238", "Pu-244")},
             para_factor=PARA_FACTOR, para_heavy=PARA_HEAVY, beta_DU_per_s=BETA_DU,
             beta_DU_current_A=BETA_DU * 1.602176634e-19, beta_Pu244=BETA_PU244,
             gain_vs_range={str(k): v for k, v in HEAVY_GAIN_LAM.items()},
             free_fall=dict(cg=FF_CG, h=FF_H, qU2=FF_QU2, max_gain=FF_GAIN_MAX, at_width=FF_BEST_W),
             U_stage_min_gain_w6_7=U_STAGE_MIN_GAIN, trace_dc_w5=TRACE_DC, uranium_pair=U_PAIR)


# how far a null result at the reach narrows the allowed width of the Gaussian switch
narrow = {}
for Pc in (1e2, PC_LO, 1e6, 1e8, PC_HI, 1e12):
    smax = sigma_widest(Pc)
    row = dict(sigma_max=smax)
    for X in ("Pu-242", "Pu-244", "Cm-248"):
        smin = float(np.sqrt(NUC[X]["d2"] / (2 * np.log(Pc / NUC[X]["S"]))))
        row[X] = dict(sigma_min=smin, frac=1 - smin / smax)
    narrow[f"{Pc:.0e}"] = row
iso = {f"{Pc:.0e}": dict(gauss=p_gauss_widest("Pu-244", Pc) / p_gauss_widest("Pu-242", Pc),
                         power=p_power("Pu-244", Pc) / p_power("Pu-242", Pc)) for Pc in (PC_LO, PC_HI)}
nums = dict(PC=[PC_LO, PC_HI], D2U=D2U, B_U=B_U, B_of_lambda=B_OF_LAM, smith=SMITH, passive_95=PASSIVE_95,
            sigma_widest={f"{Pc:.0e}": sigma_widest(Pc) for Pc in (PC_LO, PC_HI)},
            k_power={f"{Pc:.0e}": k_power(Pc) for Pc in (PC_LO, PC_HI)}, pred=pred, narrow=narrow, iso=iso,
            Pc_gauss_1S=pc_gauss(1), Pc_gauss_5S=pc_gauss(5), Pc_power_1S=pc_power(1), Pc_power_5S=pc_power(5),
            Pc_gauss_1S_Pu242=pc_gauss(1, "Pu-242"), Pc_gauss_T5=pc_gauss(1, level="T5"),
            Pc_gauss_T5_Pu242=pc_gauss(1, "Pu-242", "T5"), Pc_power_T5=pc_power(1, level="T5"), k_T5=K_T5,
            range={str(k): v for k, v in RANGE.items()}, U_improve={str(k): v for k, v in U_IMPROVE.items()},
            qalpha=QALPHA, dBmu=dBmu, heavy=HEAVY,
            sim=dict(calibration=CAL, sigma_day_thermal_check=SIG_DAY_THERMAL, resid=RESID, mu_spec=MU_SPEC,
                     P_ferro_spec=P_FERRO_SPEC, P_ferro_worst=P_FERRO_WORST, heat_242=HEAT_242,
                     P_ferro_by_source=UREF["ferro"], U_stage_v4=U_STAGE_V4, U_multisep=U_MULTISEP,
                     cal_v4=CAL_V4, cal_check=CC, campaigns_v4=CAMP_V4, eotwash_ratio=EW_RATIO, eotwash=EOTWASH,
                     eotwash_asd=INST["analytic_thermal"]["eotwash_ep"]["asd_Nm_rtHz"],
                     rep_asd=INST["analytic_thermal"]["representative"]["asd_Nm_rtHz"], vienna=VIENNA,
                     noise_scaling=NOISE,
                     U_multisep_seps_m=UREF["separations_m"],
                     lam_fit=[LAMFIT["lam_true"], LAMFIT["lam_best"]] + LAMFIT["lam_1sig"]),
            nuclides={k: {kk: vv for kk, vv in v.items()} for k, v in NUC.items()},
            f_2cm_lam05=float(f(0.02, LAM)),
            neutron_assumptions=dict(E_MeV=2.0, A_int_cm2=A_INT * 1e4, eff=EFF, m_test_g=M_TEST * 1e3),
            P_neutron={X: NUC[X]["P_neutron"] for X in CONF}, gamma=GAMMA, californium=CALI, decay=DECAY,
            shell=SHELL, fission=FISSION)
json.dump(nums, open(HERE / "numbers.json", "w"), indent=1)


# --------------------------------------------------------------- LaTeX number formatting
def tex(x, sig=2):
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    if -2 <= e <= 2:
        s = f"{float(f'{x:.{sig}g}'):.{max(0, sig - 1 - e)}f}"
        return f"${s}$"
    m = x / 10**e
    m = float(f"{m:.{sig - 1}f}")
    if m >= 10:
        m /= 10; e += 1
    ms = f"{m:.{sig - 1}f}"
    return f"${ms}\\times10^{{{e}}}$" if ms not in ("1", "1.0") else f"$10^{{{e}}}$"


def name(X):
    el, a = X.split("-")
    return f"$^{{{a}}}$" + el


# --------------------------------------------------------------- Table I: predictions
rows = []
for X in PRED_X:
    v = NUC[X]; p = pred[X]
    if X == "U-238":
        c1 = c2 = c3 = c4 = c5 = tex(B_U) + "\\,$^\\dagger$"
    else:
        c1, c2, c3, c4, c5 = (tex(p["gauss_lo"]), tex(p["gauss_hi"]), tex(p["power_hi"]), tex(p["decay_hi"]),
                              tex(p["fission_hi"]))
    rows.append(f"{name(X)} & {v['Z']} & {v['N']} & {v['avail']} & {v['w']:.2f} & {c1} & {c2} & {c3} & {c4} & {c5} \\\\")
tab1 = r"""\begin{table}[!htbp]
\caption{\label{tab:pred}Predicted magnitude of the active-mass anomaly, $|P_X|$, the fractional change in the force
a source of nuclide $X$ exerts on ordinary matter at $r\ll\lambda$, for the broadest switches of
Sec.~\ref{sec:bench} allowed by the present uranium bound $B_U=%s$ ($\lambda=0.5$~m), at two values of the
strength $P_c$ at the island center. $w_X=d_X^2/d_U^2$ sets the Gaussian prediction through Eq.~(\ref{eq:closed}).
The last two columns are the decay-rate and fission-rate benchmarks, Eqs.~(\ref{eq:decay}) and (\ref{eq:fission}),
at the largest powers the measured masses allow; they do not depend on $P_c$, and they give a stable nuclide no
charge. $^\dagger$By construction: these entries equal the uranium bound.}
\begin{ruledtabular}
\begin{tabular}{lcclcccccc}
Nuclide & $Z$ & $N$ & Available as & $w_X$ & Widest Gaussian & Widest Gaussian & Power law & Decay rate & Fission rate\\
 & & & & & $P_c=%s$ & $P_c=%s$ & $P_c=%s$ & $k=%.1f$ & $k=%.2f$\\
\hline
%s
\end{tabular}
\end{ruledtabular}
\end{table}
""" % (tex(B_U).strip("$"), tex(PC_LO).strip("$"), tex(PC_HI).strip("$"), tex(PC_HI).strip("$"), K_DEC, K_FIS,
       "\n".join(rows))
(HERE / "tab_predictions.tex").write_text(tab1, encoding="utf-8")

# --------------------------------------------------------------- Table II: practical
rows = []
for X in ("U-238", "Pu-239", "Pu-242", "Pu-244", "Cm-248", "Cf-252"):
    v = NUC[X]
    M, r = v["M"], v["r"]
    msrc = f"{M*1e3:g}\\,g" if M >= 1e-3 else f"{M*1e6:g}\\,mg"
    rows.append(f"{name(X)} & {tex(v['thalf'])} & {tex(v['act'])} & {tex(v['neutrons'])} & {tex(v['heat'])} & "
                f"{msrc} at {r*100:g}\\,cm & {tex(v['aN'])} & {tex(v['P_neutron'])} \\\\")
tab2 = r"""\begin{table}[!htbp]
\caption{\label{tab:practical}Practical properties of candidate sources. Specific activity, spontaneous-fission
neutron emission and decay heat are per gram of pure isotope. $a_N$ is the Newtonian acceleration the example
source produces at the test body. The last column is the apparent anomaly faked by fission-neutron momentum
(Sec.~\ref{sec:sys}). The heat includes the betas of the short-lived daughters (\nuc{234m}{Pa} with \nuc{238}{U},
\nuc{240m}{Np} with \Pu{244}). Half-lives and decay branches from Ref.~\cite{Kondev2021}.}
\begin{ruledtabular}
\begin{tabular}{lccccccc}
Nuclide & $t_{1/2}$ & Activity & Neutrons & Heat & Example source & $a_N$ & Neutron\\
 & (y) & (Bq\,g$^{-1}$) & (s$^{-1}$\,g$^{-1}$) & (W\,g$^{-1}$) & & (m\,s$^{-2}$) & fake\\
\hline
%s
\end{tabular}
\end{ruledtabular}
\end{table}
""" % "\n".join(rows)
(HERE / "tab_practical.tex").write_text(tab2, encoding="utf-8")

# --------------------------------------------------------------- Table III: simulated campaigns
rows = []
for X in ("U-238", "Pu-239", "Pu-242", "Pu-244", "Cm-248"):
    v = NUC[X]
    M, r = v["M"], v["r"]
    msrc = f"{M*1e3:g}\\,g" if M >= 1e-3 else f"{M*1e6:g}\\,mg"
    rows.append(f"{name(X)}, {msrc} at {r*100:g}\\,cm & {tex(v['sigma_day'])} & {v['days']} & {tex(v['s_stat'])} & "
                f"{tex(v['s_sys'])} & {tex(v['S'])} & {tex(v['T5'])} \\\\")
tab3 = r"""\begin{table}[!htbp]
\caption{\label{tab:campaign}Simulated campaigns with the pendulum of Sec.~\ref{sec:signal}. $\sigma_{\rm day}$ is
the statistical error on $P_X$ after one day; $\sigma_{\rm stat}$ and $\sigma_{\rm sys}$ are the statistical and
systematic errors for the stated integration time, the latter with the mitigations of Sec.~\ref{sec:sys}. The reach
$S=2\sigma_{\rm tot}$ is the 95\%%-confidence limit a null result would set; $5\sigma_{\rm tot}$ is the detection
threshold.}
\begin{ruledtabular}
\begin{tabular}{lcccccc}
Source & $\sigma_{\rm day}$ & Days & $\sigma_{\rm stat}$ & $\sigma_{\rm sys}$ & Reach $S$ & $5\sigma_{\rm tot}$\\
\hline
%s
\end{tabular}
\end{ruledtabular}
\end{table}
""" % "\n".join(rows)
(HERE / "tab_campaign.tex").write_text(tab3, encoding="utf-8")

# --------------------------------------------------------------- rev 5o: fingerprints (Sec. V A)
FP_SRC = ("Pu-239", "Pu-242", "Pu-244", "Cm-248")
FP_ROWS = [(f"Widest Gaussian, $P_c={tex(PC_LO).strip('$')}$", lambda X: p_gauss_widest(X, PC_LO)),
           (f"Widest Gaussian, $P_c={tex(PC_HI).strip('$')}$", lambda X: p_gauss_widest(X, PC_HI)),
           (f"Power law, $P_c={tex(PC_HI).strip('$')}$", lambda X: p_power(X, PC_HI)),
           (f"Decay rate, $k={K_DEC_LO:.1f}$", lambda X: p_decay(X, K_DEC_LO)),
           (f"Decay rate, $k={K_DEC:.1f}$", lambda X: p_decay(X)),
           (f"Fission rate, $k={K_FIS_LO:.1f}$", lambda X: p_fission(X, K_FIS_LO)),
           (f"Fission rate, $k={K_FIS:.2f}$", lambda X: p_fission(X))]          # rev 5p: 0.71, as computed
FINGERPRINT = {lab: {X: fn(X) / NUC[X]["S"] for X in FP_SRC} for lab, fn in FP_ROWS}
rows = [f"{lab} & " + " & ".join(tex(FINGERPRINT[lab][X]) for X in FP_SRC) + " \\\\" for lab, fn in FP_ROWS]
tab5 = r"""\begin{table}[!htbp]
\caption{\label{tab:fingerprint}Signal over reach, $|P_X|/S$, for the campaigns of Table~\ref{tab:campaign} under each
benchmark of Sec.~\ref{sec:bench} at $\lambda=0.5$~m. A ratio above 1 would be seen at 95\%% confidence and above 2.5
at five standard deviations. The pattern across the four sources tells the benchmarks apart.}
\begin{ruledtabular}
\begin{tabular}{lcccc}
Benchmark & %s\\
\hline
%s
\end{tabular}
\end{ruledtabular}
\end{table}
""" % (" & ".join(name(X) for X in FP_SRC), "\n".join(rows))
(HERE / "tab_fingerprint.tex").write_text(tab5, encoding="utf-8")
nums["fingerprint"] = FINGERPRINT
json.dump(nums, open(HERE / "numbers.json", "w"), indent=1)

# --------------------------------------------------------------- Table IV: heavy source on a matching test body
rows = []
for s in HEAVY_W:
    r = HEAVY_TAB[s]
    lead = tex(r["via_lead"]) if r["via_lead"] < 1e20 else "---"
    rows.append(f"{s:g} & {tex(r['qalpha'])} & {lead} & {tex(r['U'])} & {tex(r['Pu242'])} & {tex(r['Pu244'])} \\\\")
tab4 = r"""\begin{table}[!htbp]
\caption{\label{tab:heavy}Upper limits (95\%%) on the charge at the island center, $\qt_c$, for the Gaussian switch of
width $\sigma$ if ordinary matter carries no charge ($\qt_0=0$), when only a heavy test body can see the switch.
Existing: the superheavy decay energies (Sec.~\ref{sec:qalpha}) and the uranium-attractor test through the charge
the switch gives its lead test bodies, natural isotopic mix (Sec.~\ref{sec:heavy}; a dash marks limits above
$10^{20}$); the site-dependent free-fall limit of Sec.~\ref{sec:heavy} is not included. Proposed: null
results at the reaches of Table~\ref{tab:campaign}, each source acting on a test body of its own nuclide. With
$\qt_0$ at its baryon-number bound and of opposite sign, the proposed limits weaken by at most %d\%%.}
\begin{ruledtabular}
\begin{tabular}{cccccc}
 & \multicolumn{2}{c}{Existing} & \multicolumn{3}{c}{Proposed: source on a test body of its own nuclide}\\
$\sigma$ & Decay energies & Uranium attractor, via lead & $^{238}$U on $^{238}$U & $^{242}$Pu on $^{242}$Pu &
$^{244}$Pu on $^{244}$Pu\\
\hline
%s
\end{tabular}
\end{ruledtabular}
\end{table}
""" % (round(100 * (HEAVY_Q0_WEAKEN - 1)), "\n".join(rows))
(HERE / "tab_heavy.tex").write_text(tab4, encoding="utf-8")

# --------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 220,
                     "pdf.fonttype": 42})
C_U, C_PU, C_PU2, C_CM, C_CF = "#52514e", "#d95926", "#e8a07c", "#2a78d6", "#1baf7a"


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{stem}.png", bbox_inches="tight"); plt.close(fig)


# Fig 1: the widest Gaussian switch at the present uranium bound, P_c = PC_HI, on the chart of nuclides
Zs = np.arange(80, 123); Ns = np.arange(120, 191)
ZZ, NN = np.meshgrid(Zs, Ns, indexing="ij")
sw = sigma_widest(PC_HI)
logP = np.log10(PC_HI) - d2(ZZ, NN) / (2 * sw**2) / np.log(10)
fig, ax = plt.subplots(figsize=(3.4, 2.75))
im = ax.pcolormesh(NN, ZZ, np.clip(logP, -12, 12), cmap="magma", norm=Normalize(-12, 12), shading="nearest", rasterized=True)
cs = ax.contour(NN, ZZ, logP, levels=[-4, 0, 4, 8], colors="white", linewidths=[1.1, 0.6, 0.6, 0.6], linestyles=["--", "-", "-", "-"])
ax.clabel(cs, fmt={-4: "$10^{-4}$", 0: "1", 4: "$10^{4}$", 8: "$10^{8}$"}, fontsize=6.5, inline_spacing=2,
          manual=[(184 - np.sqrt(2 * sw**2 * np.log(PC_HI / 10.0**L) - 4), 112) for L in (-4, 0, 4, 8)])
A_ = np.linspace(200, 310, 200); Zst = A_ / (1.98 + 0.0155 * A_ ** (2 / 3))
ax.plot(A_ - Zst, Zst, color="#bdbdbd", lw=0.7, ls=":")
for X, off in (("U-238", (-8.5, -4.5)), ("Pu-242", (-14.0, 0.2)), ("Pu-244", (4.0, -5.5)), ("Cm-248", (-21.0, 3.5)), ("Cf-252", (3.5, 3.0))):
    v = NUC[X]
    ax.plot(v["N"], v["Z"], "o", ms=3.6, mfc="white", mec="black", mew=0.6, zorder=5)
    ax.annotate(name(X), (v["N"], v["Z"]), xytext=(v["N"] + off[0], v["Z"] + off[1]), fontsize=6.8, color="white",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="white"), zorder=6)
for X, (Z, N) in SHE.items():
    ax.plot(N, Z, "s", ms=2.6, mfc="none", mec="white", mew=0.6)
ax.text(171.5, 120.3, "observed atoms only", fontsize=6.3, color="white", ha="center")
ax.plot(NC, ZC, "x", color="white", ms=5, mew=1.0)
ax.text(NC, ZC - 2.8, "center", fontsize=6.5, color="white", ha="center")
ax.set_xlabel("neutron number $N$"); ax.set_ylabel("proton number $Z$")
ax.set_xlim(120, 190); ax.set_ylim(80, 122)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05); cb.set_label(r"$\log_{10} |P|$", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
save(fig, "fig1_chart")

# Fig 2: predictions per nuclide (P_c from 1e2 to 1e12) against the reach and the neutron fake
Xs = ["U-238", "Pu-239", "Pu-242", "Pu-244", "Cm-248", "Cf-252"]
C_DEC = "#7a5ea8"                                  # rev 5m: the decay-rate benchmark
C_FIS = "#0e7c86"                                  # rev 5o: the fission-rate benchmark (squares)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
x = np.arange(len(Xs))
for i, X in enumerate(Xs):
    if X == "U-238":
        ax.plot(i, B_U, "o", ms=4.5, color=C_U)
    else:
        ax.plot([i - 0.16] * 2, [p_gauss_widest(X, 1e2), p_gauss_widest(X, 1e12)], color=C_PU, lw=3.2, solid_capstyle="butt")
        ax.plot(i - 0.16, p_gauss_widest(X, PC_HI), "o", ms=3.4, mfc="white", mec=C_PU, mew=1.0)
        ax.plot([i + 0.16] * 2, [p_power(X, 1e2), p_power(X, 1e12)], color=C_CM, lw=1.2)
        ax.plot(i + 0.16, p_power(X, PC_HI), "D", ms=3.6, color=C_CM)
        ax.plot([i - 0.05] * 2, [B_U, p_decay(X)], color=C_DEC, lw=1.6, solid_capstyle="butt")
        ax.plot(i - 0.05, p_decay(X), "^", ms=3.8, color=C_DEC)
        if p_fission(X) > 0:
            ax.plot([i + 0.05] * 2, [B_U, p_fission(X)], color=C_FIS, lw=1.6, solid_capstyle="butt")
            ax.plot(i + 0.05, p_fission(X), "s", ms=3.2, color=C_FIS)
    if "S" in NUC[X]:
        ax.plot([i - 0.32, i + 0.32], [NUC[X]["S"]] * 2, color="black", lw=1.1)
        ax.plot([i - 0.32, i + 0.32], [NUC[X]["T5"]] * 2, color="black", lw=0.7, ls=(0, (2, 1.5)))
    if NUC[X]["P_neutron"] > 1e-6:
        ax.plot(i + 0.3, NUC[X]["P_neutron"], "x", ms=5, mew=1.2, color=C_CF)
ax.set_yscale("log"); ax.set_ylim(1e-6, 1e6)
ax.set_xticks(x); ax.set_xticklabels([name(X) for X in Xs])
ax.set_ylabel(r"active-mass anomaly $|P_X|$")
ax.plot([], [], color=C_PU, lw=3.2, label=r"widest Gaussian, $P_c=10^{2}\ldots10^{12}$")
ax.plot([], [], color=C_CM, lw=1.2, label=r"power law, $P_c=10^{2}\ldots10^{12}$")
ax.plot([], [], "D", ms=3.6, color=C_CM, label=r"markers: $P_c=10^{10}$")
ax.plot([], [], color=C_DEC, lw=1.6, marker="^", ms=3.8, label=r"decay rate, $k=0\ldots%.1f$" % K_DEC)
ax.plot([], [], color=C_FIS, lw=1.6, marker="s", ms=3.2, label=r"fission rate, $k=0\ldots%.2f$" % K_FIS)
ax.plot([], [], color="black", lw=1.1, label="reach $S$ (95% limit)")
ax.plot([], [], color="black", lw=0.7, ls=(0, (2, 1.5)), label=r"$5\sigma$ detection threshold")
ax.plot([], [], "x", ms=5, mew=1.2, color=C_CF, label="fission-neutron fake")
ax.plot([], [], "o", ms=4.5, color=C_U, label="present uranium bound")
ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=1, labelspacing=0.3)
ax.grid(axis="y", lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
save(fig, "fig2_predictions")

# Fig 3: reach in the (sigma, P_c) plane of the Gaussian switch
sg = np.linspace(3.0, 10.0, 600)
curve = lambda d2X, S: np.log10(S) + d2X / (2 * sg**2 * np.log(10))
fig, ax = plt.subplots(figsize=(3.4, 2.6))
U4, U0 = curve(D2U, B_U), curve(D2U, 1.0)
top = 30
ax.fill_between(sg, U4, top, color="#d9d9d9", lw=0)
ax.plot(sg, U4, color=C_U, lw=1.1)
ax.plot(sg, U0, color=C_U, lw=0.8, ls="--")
for X, col in (("Cm-248", C_CM), ("Pu-244", C_PU), ("Pu-242", C_PU2)):
    cX = curve(NUC[X]["d2"], NUC[X]["S"])
    ax.fill_between(sg, cX, U4, where=cX < U4, color=col, alpha=0.28, lw=0)
    ax.plot(sg, cX, color=col, lw=1.0)
qs = np.array(sorted(QSCAN)); qs = qs[qs >= 3.0]
qa_line = np.interp(sg, qs, [np.log10(QSCAN[x]["qc95"] * Q0_B) for x in qs])
C_QA = "#7a5ea8"
ax.fill_between(sg, qa_line, np.minimum(U4, top), where=qa_line < U4, color=C_QA, alpha=0.30, lw=0)
ax.plot(sg, np.where(qa_line < U4, qa_line, np.nan), color=C_QA, lw=1.0)
ax.annotate("excluded by\nsuperheavy $Q_\\alpha$", xy=(3.85, 17.5), xytext=(5.15, 21.3), fontsize=6.0, color="#3d2a63",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#3d2a63"))
for Pc in (PC_LO, PC_HI):
    ax.axhline(np.log10(Pc), color="black", lw=0.6, ls=(0, (1, 1.5)))
ax.text(9.9, np.log10(PC_HI) + 0.5, "Table I", fontsize=6.3, ha="right")
ax.text(8.2, np.log10(PC_LO) - 1.3, "Table I", fontsize=6.3, ha="right")
ax.text(7.0, 12.3, "excluded by $^{238}$U\n($B_U=7.3\\times10^{-6}$)", fontsize=6.5, color="#333333")
ax.text(9.9, -1.9, r"$^{238}$U with $B_U=1$ (dashed)", fontsize=6.0, ha="right", color="#333333")
ax.set_xlim(3.5, 10); ax.set_ylim(-4, 25)
ax.set_xlabel(r"switch width $\sigma$ (nucleons)"); ax.set_ylabel(r"strength at center, $\log_{10} P_c$")
ax.plot([], [], color=C_PU2, lw=4, alpha=0.5, label=r"$^{242}$Pu, 1 g"); ax.plot([], [], color=C_PU, lw=4, alpha=0.5, label=r"$^{244}$Pu, 1 g")
ax.plot([], [], color=C_CM, lw=4, alpha=0.5, label=r"$^{248}$Cm, 10 mg")
ax.legend(fontsize=6.3, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
save(fig, "fig3_reach")

# ---- heavy test bodies (Sec. II E, IV D, V): shared inputs at the paper's centre, range 0.5 m
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402
sgf = np.linspace(3.0, 12.0, 361)
_qs = np.array(sorted(QSCAN))
qa_f = 10 ** np.interp(sgf, _qs, [np.log10(QSCAN[x]["qc95"]) for x in _qs])
lead_f = np.array([qc_via_lead(s) for s in sgf])
ff_f = np.array([qc_free_fall(s) for s in sgf])
shp = lambda X: np.exp(-NUC[X]["d2"] / (2 * sgf ** 2))

# Fig: what a copper and a matching Pu-244 test body can see, versus width and ordinary matter's charge
q0s = np.logspace(-12, np.log10(Q0_B), 240)
allowed0 = np.minimum(qa_f, lead_f)
sPu, sU = shp("Pu-244"), shp("U-238")
T5 = NUC["Pu-244"]["T5"]


def windows(q0):
    allowed = np.minimum(allowed0, B_U / (q0 * sU)) if q0 > 0 else allowed0
    cu = np.log10(allowed / (T5 / (q0 * sPu))) if q0 > 0 else np.full_like(sgf, -np.inf)
    hv = np.log10(allowed / ((q0 + np.sqrt(q0 * q0 + 4 * T5)) / 2 / sPu))
    return cu, hv


W = [np.array([windows(q)[i] for q in q0s]) for i in (0, 1)]
W0 = windows(0.0)
blues = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], N=8)
blues.set_under("#e9e8e4")
bnorm = BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7, 8], blues.N)
fig = plt.figure(figsize=(6.6, 2.65))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045], height_ratios=[9, 1], hspace=0.08, wspace=0.08)
for col, title in ((0, "(a) copper test body"), (1, r"(b) $^{244}$Pu test body")):
    ax = fig.add_subplot(gs[0, col]); ax0 = fig.add_subplot(gs[1, col], sharex=ax)
    Wm = np.where(np.isfinite(W[col]), W[col], -1)
    im = ax.pcolormesh(sgf, q0s, Wm, cmap=blues, norm=bnorm, shading="nearest", rasterized=True)
    ax0.pcolormesh(sgf, [0, 1], np.vstack([np.where(np.isfinite(W0[col]), W0[col], -1)] * 2), cmap=blues, norm=bnorm,
                   shading="nearest", rasterized=True)
    ax.set_yscale("log"); ax.set_xlim(3, 12)
    ax.text(0.03, 0.95, title, transform=ax.transAxes, fontsize=7.2, va="top")
    plt.setp(ax.get_xticklabels(), visible=False); ax0.set_yticks([])
    ax0.set_xlabel(r"switch width $\sigma$ (nucleons)")
    if col == 0:
        ax.set_ylabel(r"$|\tilde q_0|$"); ax0.set_ylabel(r"$\tilde q_0\!=\!0$", rotation=0, ha="right", va="center", fontsize=7)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(im, cax=cax, extend="min", ticks=[0, 2, 4, 6, 8])
cb.set_label(r"decades of $\tilde q_c$ excluded", fontsize=7); cb.ax.tick_params(labelsize=6.5)
save(fig, "fig_blind_spot")

# Fig: limits on the central charge if ordinary matter carries no charge (Table IV, continuous in width)
fig, ax = plt.subplots(figsize=(3.3, 2.65))
ax.plot(sgf, qa_f, color="#7a5ea8", lw=1.1, label="decay energies")
ax.plot(sgf, lead_f, color="black", lw=0.9, label="U attractor, via lead")
ax.plot(sgf, ff_f, color="black", lw=0.8, ls=(0, (3, 2)), label="U free fall (one site)")
for X, col, lw in (("U-238", C_U, 1.6), ("Pu-242", C_PU2, 1.6), ("Pu-244", C_PU, 1.6)):
    ax.plot(sgf, np.sqrt(NUC[X]["S"]) / shp(X), color=col, lw=lw, label=f"{name(X)} pair, proposed")
ax.set_yscale("log"); ax.set_xlim(3, 12); ax.set_ylim(1, 1e20)
ax.set_xlabel(r"switch width $\sigma$ (nucleons)"); ax.set_ylabel(r"95% limit on $\tilde q_c$ ($\tilde q_0=0$)")
ax.grid(axis="y", lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
# legend in the empty corner between the decay-energy line and the free-fall line (rev 5i: in the lower left it
# crossed the proposed curves); short labels, the caption gives the full names
ax.legend(fontsize=5.8, frameon=False, loc="upper right", bbox_to_anchor=(1.01, 0.80), labelspacing=0.22,
          borderpad=0.2, handlelength=1.6, handletextpad=0.5)
save(fig, "fig_heavy_limits")

# Fig: Stage 1 by range, from the time-domain campaigns (../heavy_pair/u238_pair_campaign.py)
lam_u = sorted({c["lam"] for c in UP["campaigns"].values()})
SU = lambda tb: np.array([UP["campaigns"][f"{tb} 30 d {l:g} m"]["S"] for l in lam_u])
qU2_ff = lambda lam: 5e-10 / (FF_CG * 2 * np.pi * G * 2400.0 * lam * np.exp(-FF_H / lam) / 9.80)
qU2_bulk = lambda lam: 1e-3 * 1e11 / (2 * np.pi / 3 * G * 19050.0 ** 2 * min(0.10, lam) ** 2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
a1.plot(lam_u, [B_OF_LAM[l] for l in lam_u], color="black", lw=0.9, label="present limit (uranium attractor)")
a1.plot(lam_u, SU("Cu"), color=C_U, lw=1.6, marker="o", ms=3, label="Stage 1 reach, 30 days")
a1.set_ylabel(r"$|P_U|$, 95%"); a1.set_ylim(5e-6, 4e-4)
a2.plot(lam_u, [qU2_bulk(l) for l in lam_u], color="black", lw=0.9, label="density of uranium ingots")
a2.plot(lam_u, [qU2_ff(l) for l in lam_u], color="black", lw=0.8, ls=(0, (3, 2)), label="uranium free fall (site-dependent)")
a2.plot(lam_u, SU("U"), color=C_U, lw=1.6, marker="o", ms=3, label="uranium test body, 30 days")
a2.set_ylabel(r"$\tilde q_U^{\,2}$, 95% ($\tilde q_0=0$)"); a2.set_ylim(1e-7, 1e23)
for ax, lab in ((a1, "(a) uranium on ordinary matter"), (a2, "(b) uranium on uranium")):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"range $\lambda$ (m)")
    ax.grid(axis="y", lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
    ax.text(0.03, 0.95, lab, transform=ax.transAxes, fontsize=7.2, va="top")
    ax.legend(fontsize=5.6, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86), labelspacing=0.3)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_uranium_pair")

# --------------------------------------------------------------- what Figs. 2, 5 and 6 show, as quoted in the text (rev 5i)
# Fig. 2 (Stage 1 by range): where the copper run overtakes the present bound, and the uranium-pair gap
_Bl = np.array([B_OF_LAM[l] for l in lam_u])
_Sc, _Su = SU("Cu"), SU("U")
_lr, _ll = np.log(_Bl / _Sc), np.log(lam_u)
_k = int(np.where(np.diff(np.sign(_lr)))[0][0])
_ex = np.array([min(qU2_ff(l), qU2_bulk(l)) for l in lam_u])
FIG_UPAIR = dict(crossover_m=float(np.exp(_ll[_k] - _lr[_k] * (_ll[_k + 1] - _ll[_k]) / (_lr[_k + 1] - _lr[_k]))),
                 gain_at_5cm=float(_Bl[0] / _Sc[0]),
                 shortfall_from_0_5m=[float((_Sc / _Bl)[3:].min()), float((_Sc / _Bl)[3:].max())],
                 present_from_0_5m=[float(_Bl[3:].min()), float(_Bl[3:].max())],
                 free_fall={f"{l:g}": float(qU2_ff(l)) for l in lam_u}, ingots=float(qU2_bulk(0.5)),
                 gap_decades=[float(np.log10(_ex / _Su).min()), float(np.log10(_ex / _Su).max())])
assert lam_u[0] == 0.05 and lam_u[3] == 0.5
# Fig. 5 (blind spot): decades of q_c excluded with copper and with a Pu-244 test body, against width and q0
_cu = np.where(np.isfinite(W[0]), W[0], -9.0)
_cumax, _pumax, _w0, _jb = _cu.max(axis=1), W[1].max(axis=1), W0[1], len(q0s) - 1    # q0s[-1] is the bound
FIG_BLIND = dict(copper_at_bound=float(_cumax[_jb]), copper_width_at_bound=float(sgf[np.argmax(_cu[_jb])]),
                 copper_sees_nothing_below_q0=float(q0s[_cumax > 0].min()),
                 pu_at_bound=float(_pumax[_jb]), pu_minus_copper_at_bound=float(np.max(W[1][_jb] - _cu[_jb])),
                 pu_q0_zero=float(_w0.max()), pu_q0_zero_width=float(sgf[np.argmax(_w0)]),
                 pu_q0_zero_widths_over_4=[float(sgf[_w0 > 4].min()), float(sgf[_w0 > 4].max())],
                 pu_q0_zero_widths_over_1=[float(sgf[_w0 > 1].min()), float(sgf[_w0 > 1].max())])
# Fig. 6 (limits with ordinary matter uncharged): where each pair overtakes the existing limits, and by how much
_best = np.minimum(qa_f, lead_f)
FIG_LIMITS = dict(lead_beats_decay_energies_from=float(sgf[np.argmax(lead_f < qa_f)]), pairs={})
for _X in ("U-238", "Pu-242", "Pu-244"):
    _L = np.sqrt(NUC[_X]["S"]) / shp(_X)
    _g = _best / _L
    _k = int(np.argmax(_g))
    FIG_LIMITS["pairs"][_X] = dict(beats_existing_from=float(sgf[_L < _best].min()), max_gain=float(_g[_k]),
                                   at_width=float(sgf[_k]), existing_there=float(_best[_k]), limit_there=float(_L[_k]),
                                   max_gain_with_free_fall=float((np.minimum(_best, ff_f) / _L).max()))
assert abs(FIG_LIMITS["lead_beats_decay_energies_from"] - FF_BEST_W) < 0.1

nums["figs"] = dict(uranium_pair=FIG_UPAIR, blind_spot=FIG_BLIND, heavy_limits=FIG_LIMITS)
json.dump(nums, open(HERE / "numbers.json", "w"), indent=1)

# --------------------------------------------------------------- report
print(f"uranium bound: passive {PASSIVE_95:.2e}; B_U(lambda):", {k: f"{v:.2g}" for k, v in B_OF_LAM.items()})
print(f"B_U = {B_U:.3e} at lambda = {LAM} m;  d_U^2 = {D2U}")
print("widest sigma:", {k: round(v, 2) for k, v in nums["sigma_widest"].items()}, " power k:", {k: round(v, 1) for k, v in nums["k_power"].items()})
for X, p in pred.items():
    v = NUC[X]
    print(f"{X:7s} w={v['w']:.3f}  gauss {p['gauss_lo']:.2e} (1e4) {p['gauss_hi']:.2e} (1e10)   power {p['power_lo']:.2e} (1e4) {p['power_hi']:.2e} (1e10)")
for X in CONF:
    v = NUC[X]
    simn = "" if X == "Pu-239" else f" (sim {SIM['budget'][SIMKEY[X]]['Fission-neutron momentum']:.2e})"
    print(f"{X:7s} act {v['act']:.3e} Bq/g  n {v['neutrons']:.3e} /s/g  heat {v['heat']:.2e} W/g  aN {v['aN']:.2e}  neutron fake {v['P_neutron']:.2e}{simn}")
print("decay-rate benchmark:", {k: DECAY[k] for k in ("k_max", "k_max_band", "k_max_slow", "k_max_slow_band")},
      "Be8", {k: f"{v:.3g}" for k, v in DECAY["Be8"].items()}, "superheavy decades", DECAY["superheavy_decades"])
print("  pred k=0.3/0.6:", {X: f"{d['lo']:.2g}/{d['hi']:.2g}" for X, d in DECAY["pred"].items()})
print("  k needed:", {X: {kk: round(vv, 3) for kk, vv in d.items()} for X, d in DECAY["k_needed"].items()},
      " x reach at k=0.6:", {X: round(v, 2) for X, v in DECAY["x_reach"].items()}, " 242/244:", DECAY["ratio_242_244"])
print("  smooth:", {k: {kk: round(vv, 3) for kk, vv in d.items()} for k, d in SMOOTH.items()})
print("  Pu-239:", {k: f"{v:.3g}" for k, v in PU239.items() if not isinstance(v, dict)}, " gamma push", {k: f"{v:.2g}" for k, v in GAMMA["push"]["Pu-239"].items()})
for X in ("U-238", "Pu-239", "Pu-242", "Pu-244", "Cm-248"):
    v = NUC[X]
    print(f"campaign {X}: {v['days']} d, sigma_day {v['sigma_day']:.2e}, stat {v['s_stat']:.2e}, sys {v['s_sys']:.2e}, S {v['S']:.2e}, T5 {v['T5']:.2e}")
print(f"thermal-limit check of sigma_day (1 g at 2 cm): {SIG_DAY_THERMAL:.2e} vs simulated {NUC['Pu-244']['sigma_day']:.2e}")
print(f"days for 1e-4 at 3 sigma (stat only): {(3 * NUC['Pu-244']['sigma_day'] / 1e-4) ** 2:.0f}")
print("torque noise above thermal (beta at the exchange frequency; representative %.2f):" % NOISE["beta_representative"])
for _k, _c in NOISE["cases"].items():
    print("   %-15s beta %.2f: sigma_day Pu %.3g, days for 1e-4/3 %.0f, S U %.2g Pu %.2g Cm %.2g" % (
        _k, _c["beta"], _c["sigmaP_day"]["244Pu"], _c["days_to_target_Pu244"], _c["campaigns"]["238U"]["S"],
        _c["campaigns"]["244Pu"]["S"], _c["campaigns"]["248Cm"]["S"]))
print("residual:", {k: f"{v:.2g}" for k, v in RESID.items()}, " ferro spec:", f"{P_FERRO_SPEC:.2g}", "worst", f"{P_FERRO_WORST:.2g}", " heat 242:", HEAT_242)
print(f"5 sigma: Gaussian Pu-244 P_c >= {pc_gauss(1, level='T5'):.3g}; Pu-242 >= {pc_gauss(1, 'Pu-242', 'T5'):.3g}; power law P_c >= {pc_power(1, level='T5'):.3g} (k >= {K_T5:.0f})")
for lam, r in RANGE.items():
    print(f"  lambda {lam} m: B_U {r['B_U']:.2g}, f {r['supp']:.3f}, Pu-244 P_c for 5 sigma {r['Pc_T5']:.3g}, for S {r['Pc_S']:.3g}; U stage T5 {r['U_stage_T5']:.2g}")
print("Q_alpha:", {k: (v if not isinstance(v, dict) else {a: (f"{b:.3g}" if isinstance(b, float) else "...") for a, b in v.items()})
                   for k, v in QALPHA.items() if k not in ("corr", "sd_understated")})
print("calibration: v4 %.3f -> rev 5 %.3f; sigma_day Pu %.3g; Eot-Wash ratio %.3f:" % (CAL_V4, CAL, NUC["Pu-244"]["sigma_day"], EW_RATIO),
      {k: {kk: f"{vv:.3g}" for kk, vv in v.items()} for k, v in EOTWASH.items()}, " Vienna sigma_day Pu %.3g" % VIENNA["sigmaP_day"]["244Pu"])
print("U stage: rev 4", {k: f"{v:.3g}" for k, v in U_STAGE_V4.items()}, "-> rev 5 S", f"{NUC['U-238']['S']:.3g}",
      "; flip + five separations S", f"{U_MULTISEP['S']:.3g}", "; ferro by source", UREF["ferro"])
print("lambda fit:", nums["sim"]["lam_fit"])
print("uranium-stage improvement over the present bound:", {k: round(v, 1) for k, v in U_IMPROVE.items()})
print("sigmas in 90 d, Pu-244:", {k: round(pred['Pu-244'][k] / NUC['Pu-244']['s_tot'], 1) for k in ('gauss_lo', 'gauss_hi', 'power_hi')},
      " x reach:", {k: round(pred['Pu-244'][k] / NUC['Pu-244']['S'], 1) for k in ('gauss_lo', 'gauss_hi', 'power_hi')},
      " Cm x reach:", {k: round(pred['Cm-248'][k] / NUC['Cm-248']['S'], 1) for k in ('gauss_lo', 'gauss_hi', 'power_hi')})
print("days to 5 sigma (stat), Pu-244:", {k: round((5 * NUC['Pu-244']['sigma_day'] / pred['Pu-244'][k]) ** 2, 2) for k in ('gauss_lo', 'gauss_hi', 'power_hi')})
print(f"Gaussian Pu-244: >= reach for P_c >= {pc_gauss(1):.2g}, 5x for {pc_gauss(5):.2g};  Pu-242 >= reach for {pc_gauss(1, 'Pu-242'):.2g}")
print(f"power law Pu-244: >= reach for P_c >= {pc_power(1):.2g}, 5x for {pc_power(5):.2g}")
for k, r in narrow.items():
    print(f"P_c {k}: sigma_max {r['sigma_max']:.2f};", "; ".join(f"{X} -> {r[X]['sigma_min']:.2f} ({r[X]['frac']*100:.1f}%)" for X in ("Pu-242", "Pu-244", "Cm-248")))
print("isotope ratios:", {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in iso.items()})
print(f"Cm-248 power law at 1e10: {pred['Cm-248']['power_hi']:.2e} vs neutron fake {NUC['Cm-248']['P_neutron']:.2e}; Gaussian {pred['Cm-248']['gauss_hi']:.2e}")
print("dB/mu Cu-Pb:", f"{dBmu:.2e}", " f(2 cm, 0.5 m):", round(float(f(0.02, LAM)), 4))
