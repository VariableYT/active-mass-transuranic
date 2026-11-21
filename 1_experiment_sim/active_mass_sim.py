#!/usr/bin/env python3
"""
active_mass_sim.py
End-to-end simulation of the transuranic active-mass torsion-balance comparison
proposed in "Active-mass tests with transuranic sources" (Grady, 2026).

Deterministic: every random draw comes from numpy Generators seeded from SEED.

Modules
  1. Geometry and materials (pendulum, test bodies, source/dummy capsules, turntable)
  2. Force model: Newton + Yukawa, Gauss-Legendre volume integration over cylindrical
     sources, exact Yukawa form factor for spherical test bodies, torque about the fibre
  3. Newtonian-residual Monte Carlo (mass, COM, dimension, density-gradient tolerances),
     capsule-flip and multi-separation mitigation
  4. Systematics: fission-neutron momentum (Maxwellian spectrum, Cu transport cross
     section), decay-heat radiation pressure, gamma radiation pressure (241Am),
     paramagnetic and ferromagnetic couplings, mass-energy loss
  5. Pendulum dynamics: damped oscillator (ZOH-discretised), structural-damping thermal
     torque noise (fluctuation-dissipation), excess red torque noise, readout noise
  6. Analysis: torque reconstruction, half-cycle demodulation with linear-drift
     cancellation, daily blocking for the error
  7. Injection-recovery, sensitivity vs integration time, reach vs range lambda,
     lambda reconstruction from multiple separations
"""
import json, math, os, sys
import numpy as np
if not hasattr(np, "trapezoid"):  # numpy < 2.0
    np.trapezoid = np.trapz
from numpy.polynomial.legendre import leggauss
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261114
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- constants
G = 6.67430e-11; kB = 1.380649e-23; c = 299792458.0
SIGMA_SB = 5.670374419e-8; MU0 = 4e-7 * math.pi
MEV = 1.602176634e-13; M_N = 1.67492750e-27; U_AMU = 1.66053907e-27
DAY = 86400.0

# ----------------------------------------------------------------------------- pendulum
PEND = dict(
    m_t=3.0e-3,        # kg, Cu test sphere
    rho_t=8960.0,      # kg/m^3
    chi_t=-9.63e-6,    # SI volume susceptibility, Cu
    R_arm=0.020,       # m, test body radius from fibre axis
    I=1.0e-5,          # kg m^2, total moment of inertia
    T0=600.0,          # s, free period
    Q=4000.0,          # mechanical quality factor (structural damping)
    T=293.15,          # K
    readout_asd=1.0e-9,    # rad/sqrt(Hz), autocollimator white noise
    red_corner=2.5e-4,     # Hz, excess red torque noise equals thermal at this frequency
    T_half=600.0,          # s, half-period of the source/dummy exchange
    t_skip=60.0,           # s, data excluded after each exchange (turntable transit)
    turntable_R=0.05,      # m, turntable radius; parked capsule is 2R further away
)
PEND["a_t"] = (3 * PEND["m_t"] / (4 * math.pi * PEND["rho_t"])) ** (1 / 3)
PEND["w0"] = 2 * math.pi / PEND["T0"]
PEND["kappa"] = PEND["I"] * PEND["w0"] ** 2

# ----------------------------------------------------------------------------- sources
# heat: W/g; nps: spontaneous-fission neutrons s^-1 g^-1; chi: SI volume susceptibility
# (Curie/measured estimates, see report); T_n: Maxwellian temperature of fission spectrum, MeV
SOURCES = {
    "238U":  dict(M=100e-3, rho=19050., r=0.030, heat=8.5e-9, nps=0.014,  chi=4.1e-4, T_n=1.30, A=238),
    "242Pu": dict(M=1e-3,   rho=15920., r=0.020, heat=1.2e-4, nps=1.7e3,  chi=4.4e-4, T_n=1.33, A=242),
    "244Pu": dict(M=1e-3,   rho=15920., r=0.020, heat=5.3e-7, nps=1.9e3,  chi=4.4e-4, T_n=1.33, A=244),
    "248Cm": dict(M=10e-6,  rho=13510., r=0.010, heat=4.9e-4, nps=4.0e7,  chi=1.8e-2, T_n=1.40, A=248),
    "252Cf": dict(M=1e-6,   rho=15100., r=0.010, heat=37.0,   nps=2.3e12, chi=1.0e-2, T_n=1.42, A=252),
}
DUMMY_CHI = 7.8e-5   # tungsten-based density-matched dummy
for s in SOURCES.values():
    V = s["M"] / s["rho"]
    s["a"] = (V / (2 * math.pi)) ** (1 / 3)   # cylinder with h = 2a
    s["h"] = 2 * s["a"]

# Benchmark predictions at lambda = 0.5 m, present uranium bound (paper Table I)
BENCH = {
    "238U":  dict(G4=7.3e-6, G10=7.3e-6, PL10=7.3e-6),
    "242Pu": dict(G4=9.2e-5, G10=4.9e-4, PL10=9.7e-5),
    "244Pu": dict(G4=4.2e-4, G10=6.1e-3, PL10=3.5e-4),
    "248Cm": dict(G4=4.1e-3, G10=0.26,   PL10=4.3e-3),
    "252Cf": dict(G4=0.033,  G10=8.4,    PL10=0.052),
}

# ----------------------------------------------------------------------------- force model
def cyl_nodes(a, h, nr=10, nphi=24, nz=10):
    """Gauss-Legendre (rho, z) x periodic trapezoid (phi) nodes for a solid cylinder.
    Returns local positions (n,3) and volume weights (n,) summing to pi a^2 h."""
    xr, wr = leggauss(nr); xz, wz = leggauss(nz)
    rr = (xr + 1) * a / 2; wr = wr * a / 2 * rr
    zz = xz * h / 2; wz = wz * h / 2
    ph = np.arange(nphi) * 2 * math.pi / nphi; wp = np.full(nphi, 2 * math.pi / nphi)
    R, P, Z = np.meshgrid(rr, ph, zz, indexing="ij")
    W = wr[:, None, None] * wp[None, :, None] * wz[None, None, :]
    pts = np.stack([R * np.cos(P), R * np.sin(P), Z], -1).reshape(-1, 3)
    return pts, W.reshape(-1)

def sphere_ff(a, lam):
    """Yukawa form factor of a uniform sphere: exterior Yukawa field equals that of a
    point mass scaled by 3(x cosh x - sinh x)/x^3, x = a/lam."""
    x = a / lam
    if x < 1e-4:
        return 1.0 + x * x / 10
    return 3 * (x * math.cosh(x) - math.sinh(x)) / x ** 3

def pendulum_bodies():
    """Point-equivalent bodies on the pendulum: test sphere and counterweight sphere."""
    R = PEND["R_arm"]
    return [(np.array([R, 0.0, 0.0]), PEND["m_t"], PEND["a_t"]),
            (np.array([-R, 0.0, 0.0]), PEND["m_t"], PEND["a_t"])]

def source_mass_elements(src, center, dM=0.0, dcom=(0, 0, 0), da=0.0, dh=0.0, grad=(0, 0)):
    """Mass elements of a cylindrical source/dummy: tolerances dM (kg), COM offset (m),
    radius/height errors (m), linear density gradient (1/m) in x and y."""
    a, h = src["a"] + da, src["h"] + dh
    pts, w = cyl_nodes(a, h)
    dens = 1.0 + grad[0] * pts[:, 0] + grad[1] * pts[:, 1]
    m = w * dens
    m *= (src["M"] + dM) / m.sum()
    return pts + np.asarray(center) + np.asarray(dcom), m

def torque(pos, m, lam=None):
    """z-torque on the pendulum from source mass elements. lam=None: Newtonian;
    otherwise Yukawa part per unit coupling alpha (attractive sign convention)."""
    tz = 0.0
    for X, mb, ab in pendulum_bodies():
        d = X[None, :] - pos
        s = np.linalg.norm(d, axis=1)
        if lam is None:
            k = 1.0 / s ** 3
        else:
            k = np.exp(-s / lam) * (1 + s / lam) / s ** 3 * sphere_ff(ab, lam)
        F = -G * mb * (m[:, None] * k[:, None] * d).sum(0)
        tz += X[0] * F[1] - X[1] * F[0]
    return tz

def positions(src):
    """Measuring and parked capsule centres (turntable swaps them)."""
    R, r, Rt = PEND["R_arm"], src["r"], PEND["turntable_R"]
    meas = np.array([R, r, 0.0]); park = np.array([R, r + 2 * Rt, 0.0])
    return meas, park

def unit_torque(src, lam):
    """Modulated torque per unit anomaly P: (source at meas - source at park)."""
    meas, park = positions(src)
    p1, m1 = source_mass_elements(src, meas); p2, m2 = source_mass_elements(src, park)
    return torque(p1, m1, lam) - torque(p2, m2, lam)

def newton_mod_torque(src, tol_s, tol_d):
    """Modulated Newtonian torque from source-vs-dummy differences.
    State A: source at meas, dummy parked. State B: dummy at meas, source parked."""
    meas, park = positions(src)
    def T(tol, ctr):
        p, m = source_mass_elements(src, ctr, **tol); return torque(p, m)
    A = T(tol_s, meas) + T(tol_d, park)
    B = T(tol_d, meas) + T(tol_s, park)
    return A - B

# ----------------------------------------------------------------------------- Newtonian residual MC
TOL = dict(sig_M=1e-9, sig_com=1e-6, sig_dim=1e-6, sig_grad=1.0)   # 1 ug, 1 um, 1 um, 0.1 %/mm

def draw_tol(rng, sig_grad):
    return dict(dM=rng.normal(0, TOL["sig_M"]),
                dcom=tuple(rng.normal(0, TOL["sig_com"], 3)),
                da=rng.normal(0, TOL["sig_dim"]), dh=rng.normal(0, TOL["sig_dim"]),
                grad=tuple(rng.normal(0, sig_grad, 2)))

def flipped(t):
    """Capsule rotated 180 deg about its own vertical axis: in-plane offsets and gradients reverse."""
    f = dict(t); f["dcom"] = (-t["dcom"][0], -t["dcom"][1], t["dcom"][2])
    f["grad"] = (-t["grad"][0], -t["grad"][1]); return f

def newton_mc(name, n, sig_grad, seed):
    src = SOURCES[name]; rng = np.random.default_rng(seed)
    tau_ref = unit_torque(src, 0.5)
    raw, flip = np.empty(n), np.empty(n); parts = {k: np.empty(n) for k in ("M", "com", "dim", "grad")}
    zero = dict(dM=0.0, dcom=(0, 0, 0), da=0.0, dh=0.0, grad=(0, 0))
    for i in range(n):
        ts, td = draw_tol(rng, sig_grad), draw_tol(rng, sig_grad)
        raw[i] = newton_mod_torque(src, ts, td) / tau_ref
        flip[i] = 0.5 * (raw[i] + newton_mod_torque(src, flipped(ts), flipped(td)) / tau_ref)
        if i < 400:
            for k, keys in (("M", ["dM"]), ("com", ["dcom"]), ("dim", ["da", "dh"]), ("grad", ["grad"])):
                a = dict(zero); b = dict(zero)
                for kk in keys: a[kk] = ts[kk]; b[kk] = td[kk]
                parts[k][i] = newton_mod_torque(src, a, b) / tau_ref
    parts = {k: float(np.std(v[:min(n, 400)])) for k, v in parts.items()}
    return raw, flip, parts

# ----------------------------------------------------------------------------- other systematics (P-equivalent, lambda = 0.5 m)
def maxwell_mean_p(T):
    E = np.linspace(1e-4, 25 * T, 200001); f = np.sqrt(E) * np.exp(-E / T)
    p = np.sqrt(2 * M_N * E * MEV)
    return float(np.trapezoid(f * p, E) / np.trapezoid(f, E))

SIG_TR_CU = 2.2e-28   # m^2: Cu transport cross section near 2 MeV, sigma_el(1-mu) + sigma_nonel
N_CU = 8960.0 / (63.546 * U_AMU)

def radial_push(src, F_of_d):
    """Torque from a radial repulsive force F(d) acting on each pendulum body, modulated
    between meas and park positions (dummy emits nothing)."""
    meas, park = positions(src); tz = 0.0
    for ctr, sgn in ((meas, 1), (park, -1)):
        for X, mb, ab in pendulum_bodies():
            d = X - ctr; s = np.linalg.norm(d)
            F = F_of_d(s, ab) * d / s
            tz += sgn * (X[0] * F[1] - X[1] * F[0])
    return tz

def sys_neutron(src):
    N = src["nps"] * src["M"] * 1e3; p = maxwell_mean_p(src["T_n"])
    def F(s, ab):
        frac = 1 - math.exp(-N_CU * SIG_TR_CU * 4 * ab / 3)
        return N / (4 * math.pi * s * s) * math.pi * ab * ab * frac * p
    return radial_push(src, F)

def sys_thermal_rad(src, R_th=100.0, eps=0.05, heater_match=0.01, membrane=0.5):
    Pw = src["heat"] * src["M"] * 1e3
    A = 2 * math.pi * src["a"] ** 2 + 2 * math.pi * src["a"] * src["h"]
    g = eps * 4 * SIGMA_SB * PEND["T"] ** 3 * A
    dT = Pw / (1 / R_th + g); Prad = g * dT
    def F(s, ab): return membrane * Prad / (4 * math.pi * s * s) * math.pi * ab * ab / c
    full = radial_push(src, F)
    return full * heater_match, full, dT

def sys_gamma_am(src, am_frac=1e-4):
    if src is not SOURCES["242Pu"]: return 0.0
    act = am_frac * src["M"] * 1e3 * 1.27e11; Pg = act * 0.359 * 59.54e-3 * MEV
    def F(s, ab): return Pg / (4 * math.pi * s * s) * math.pi * ab * ab / c * 0.5   # ~50% absorbed in Cu test body path
    return radial_push(src, F)

def sys_paramagnetic(src, B0=1e-7):
    Vs = src["M"] / src["rho"]; Vt = PEND["m_t"] / PEND["rho_t"]
    ms = (src["chi"] - DUMMY_CHI) * Vs * B0 / MU0
    def F(s, ab): return 6 * abs(PEND["chi_t"]) * Vt * B0 * ms / (4 * math.pi * s ** 4)
    return radial_push(src, F)

def sys_ferro(src, ppm_src=10.0, ppm_t=1.0, remanence=0.01):
    """Worst case: Fe impurity present as ferromagnetic precipitates with 1% remanence."""
    Ms = 1.71e6; rho_fe = 7874.0
    m_fe_src = ppm_src * 1e-6 * src["M"]; m_fe_t = ppm_t * 1e-6 * PEND["m_t"]
    mu_s = m_fe_src / rho_fe * Ms * remanence; mu_t = m_fe_t / rho_fe * Ms * remanence
    def F(s, ab): return 6 * MU0 * mu_s * mu_t / (4 * math.pi * s ** 4)   # coaxial worst case
    return radial_push(src, F)

def sys_massloss(src, days=30.0):
    Pw = src["heat"] * src["M"] * 1e3
    return (Pw / c ** 2) * days * DAY / src["M"]   # fractional mass change -> P-equivalent

# ----------------------------------------------------------------------------- pendulum dynamics
def make_filter(dt):
    I, w0, Q = PEND["I"], PEND["w0"], PEND["Q"]
    sysd = signal.cont2discrete(([1 / I], [1, w0 / Q, w0 ** 2]), dt, method="zoh")
    return np.squeeze(sysd[0]), np.squeeze(sysd[1])

def colored(rng, N, dt, psd_fn):
    w = rng.standard_normal(N); f = np.fft.rfftfreq(N, dt)
    S = np.zeros_like(f); S[1:] = psd_fn(f[1:])
    return np.fft.irfft(np.fft.rfft(w) * np.sqrt(S / (2 * dt)), n=N)

def S_thermal(f):
    return 4 * kB * PEND["T"] * PEND["kappa"] / (PEND["Q"] * 2 * math.pi * f)

def S_torque(f):
    fc = PEND["red_corner"]
    return S_thermal(f) * (1 + fc / f)

def simulate(dtau, days, dt, seed):
    """Time series of the pendulum angle for a modulated torque dtau (source state)."""
    rng = np.random.default_rng(seed); N = int(days * DAY / dt); t = np.arange(N) * dt
    state = ((t // PEND["T_half"]) % 2 == 0).astype(float)
    tau = dtau * state + colored(rng, N, dt, S_torque)
    b, a = make_filter(dt)
    theta = signal.lfilter(b, a, tau)
    theta += rng.normal(0, PEND["readout_asd"] / math.sqrt(2 * dt), N)
    return t, theta, state

def reconstruct_torque(theta, dt):
    I, w0, Q = PEND["I"], PEND["w0"], PEND["Q"]
    th2 = np.zeros_like(theta); th1 = np.zeros_like(theta)
    th2[1:-1] = (theta[2:] - 2 * theta[1:-1] + theta[:-2]) / dt ** 2
    th1[1:-1] = (theta[2:] - theta[:-2]) / (2 * dt)
    return I * (th2 + w0 / Q * th1 + w0 ** 2 * theta)

def demodulate(t, tau_est, dt):
    """Half-cycle means (skipping transit), then A-B-A drift-cancelling differences,
    blocked by day for the uncertainty. Returns (mean, sem, per-cycle estimates, times)."""
    Th = PEND["T_half"]; nh = int(t[-1] // Th)
    idx = (t // Th).astype(int); keep = (t % Th) >= PEND["t_skip"]
    sums = np.bincount(idx[keep], tau_est[keep], minlength=nh + 1)
    cnt = np.bincount(idx[keep], minlength=nh + 1)
    hm = sums[:nh] / np.maximum(cnt[:nh], 1)
    k = np.arange(1, nh - 1)
    sgn = np.where(k % 2 == 0, 1.0, -1.0)            # even halves: source present
    est = sgn * (hm[k] - 0.5 * (hm[k - 1] + hm[k + 1]))
    tk = (k + 0.5) * Th
    day = (tk // DAY).astype(int); nd = day.max() + 1
    dm = np.array([est[day == d].mean() for d in range(nd) if np.any(day == d)])
    return float(est.mean()), float(dm.std(ddof=1) / math.sqrt(len(dm))), est, tk

# ----------------------------------------------------------------------------- main
def main():
    res = dict(seed=SEED, pendulum={k: v for k, v in PEND.items()}, sources={})
    lam_ref = 0.5

    # 1. unit torques and Newtonian scales
    for n, s in SOURCES.items():
        tau1 = unit_torque(s, lam_ref)
        aN = G * s["M"] / s["r"] ** 2
        res["sources"][n] = dict(a=s["a"], h=s["h"], r=s["r"], aN=aN, unit_torque=tau1)

    # 2. Newtonian residual MC
    rng_seeds = np.random.SeedSequence(SEED).spawn(20)
    mc = {}
    for i, n in enumerate(["238U", "242Pu", "248Cm"]):
        good = newton_mc(n, 1500, 1.0, rng_seeds[i])
        oxide = newton_mc(n, 1500, 10.0, rng_seeds[i + 5])
        mc[n] = dict(good=good, oxide=oxide)
        res["sources"][n]["newton_mc"] = dict(
            raw_p95_good=float(np.percentile(abs(good[0]), 95)),
            flip_p95_good=float(np.percentile(abs(good[1]), 95)),
            raw_p95_oxide=float(np.percentile(abs(oxide[0]), 95)),
            flip_p95_oxide=float(np.percentile(abs(oxide[1]), 95)),
            components_rms_good=good[2], components_rms_oxide=oxide[2])
    mc["244Pu"] = mc["242Pu"]; res["sources"]["244Pu"]["newton_mc"] = res["sources"]["242Pu"]["newton_mc"]

    # 3. multi-separation fit with physical templates (COM x/y, radius, height) + anomaly
    src = SOURCES["244Pu"]; rng = np.random.default_rng(rng_seeds[12])
    rs = np.array([0.015, 0.020, 0.025, 0.030, 0.040])
    zero = dict(dM=0.0, dcom=(0, 0, 0), da=0.0, dh=0.0, grad=(0, 0))
    def sr(r):
        s2 = dict(src); s2["r"] = r; return s2
    uref = np.array([unit_torque(sr(r), lam_ref) for r in rs])
    templ = []
    for pert in (dict(dcom=(1e-6, 0, 0)), dict(dcom=(0, 1e-6, 0)), dict(da=1e-6), dict(dh=1e-6)):
        tp = dict(zero); tp.update(pert)
        templ.append(np.array([newton_mod_torque(sr(r), tp, zero) for r in rs]) / uref)
    X = np.vstack([np.ones_like(rs)] + templ).T
    fits = []
    for i in range(300):
        ts, td = draw_tol(rng, 10.0), draw_tol(rng, 10.0)
        y = np.array([newton_mod_torque(sr(r), ts, td) for r in rs]) / uref
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        fits.append((y[1], coef[0], (ts["dM"] - td["dM"]) / src["M"]))
    fits = np.array(fits)
    res["multisep"] = dict(raw_p95=float(np.percentile(abs(fits[:, 0]), 95)),
                           fit_p95=float(np.percentile(abs(fits[:, 1]), 95)),
                           fit_minus_mass_p95=float(np.percentile(abs(fits[:, 1] - fits[:, 2]), 95)),
                           mass_term_p95=float(np.percentile(abs(fits[:, 2]), 95)))

    # 4. systematics budget
    budget = {}
    for n, s in SOURCES.items():
        tau1 = res["sources"][n]["unit_torque"]
        th_res, th_full, dT = sys_thermal_rad(s)
        nm = res["sources"].get(n, {}).get("newton_mc", res["sources"]["242Pu"]["newton_mc"])
        budget[n] = {
            "Newtonian residual (flip, oxide)": nm["flip_p95_oxide"] if n != "252Cf" else None,
            "Fission-neutron momentum": abs(sys_neutron(s) / tau1),
            "Decay-heat radiation (unmatched)": abs(th_full / tau1),
            "Decay-heat radiation (heater 1%)": abs(th_res / tau1),
            "241Am gamma pressure": abs(sys_gamma_am(s) / tau1),
            "Paramagnetic (100 nT)": abs(sys_paramagnetic(s) / tau1),
            "Ferromagnetic contamination": abs(sys_ferro(s) / tau1),
            "Mass-energy loss (30 d)": sys_massloss(s),
        }
        res["sources"][n]["capsule_dT_K"] = dT
    res["budget"] = budget

    # 5. time-domain: one long null run for noise characterisation
    dt = 10.0
    t, th, st = simulate(0.0, 120, dt, rng_seeds[13])
    tau_est = reconstruct_torque(th, dt)
    m0, e0, est0, tk0 = demodulate(t, tau_est, dt)
    Tlist = np.array([1, 2, 4, 6, 10, 15, 20, 30, 40, 60, 120])
    sig_T, sig_T_meas = [], []
    sig_day = e0 * math.sqrt(120)
    for T in Tlist:
        nch = int(120 // T)
        if nch >= 4:
            ch = np.array([est0[(tk0 >= j * T * DAY) & (tk0 < (j + 1) * T * DAY)].mean() for j in range(nch)])
            sig_T_meas.append(float(ch.std(ddof=1)))
        else:
            sig_T_meas.append(None)
        sig_T.append(sig_day / math.sqrt(T))
    sig_T = np.array(sig_T)
    # analytic: lock-in of square wave, fundamental carries 4/pi; difference estimator
    f_s = 1 / (2 * PEND["T_half"])
    sig_an_day = math.sqrt(S_torque(f_s) / DAY)   # ideal single-frequency estimate, 1 day
    res["noise"] = dict(f_signal=f_s, S_thermal_asd=math.sqrt(S_thermal(f_s)),
                        S_total_asd=math.sqrt(S_torque(f_s)),
                        sigma_tau_120d=e0, sigma_tau_day=sig_day, ideal_day=sig_an_day,
                        estimator_efficiency=sig_an_day / sig_day,
                        Tdays=Tlist.tolist(), sigma_tau=sig_T.tolist(), sigma_tau_chunks=sig_T_meas)
    # per-source statistical sigma_P vs time (scale by unit torque)
    for n in SOURCES:
        tau1 = res["sources"][n]["unit_torque"]
        sP = sig_T / abs(tau1)
        res["sources"][n]["sigmaP_vs_T"] = sP.tolist()
        # days to reach 3 sigma at the Gaussian Pc=1e4 prediction: sigma ~ T^-1/2
        s1 = e0 * math.sqrt(120) / abs(tau1)
        for key in ("G4", "G10", "PL10"):
            P = BENCH[n][key]
            res["sources"][n][f"days_5sigma_{key}"] = (5 * s1 / P) ** 2
        res["sources"][n]["days_stat_1e-4_3sigma"] = (3 * s1 / 1e-4) ** 2
        res["sources"][n]["sigmaP_1day"] = s1

    # 6. injection-recovery, 30-day runs, 10 seeds each
    inj = {}
    nmP = res["sources"]["244Pu"]["newton_mc"]["flip_p95_good"]
    scen = [("244Pu", 0.0, "null"), ("244Pu", BENCH["244Pu"]["G4"], "Gaussian Pc=1e4"),
            ("244Pu", BENCH["244Pu"]["PL10"], "Power law Pc=1e10"),
            ("242Pu", BENCH["242Pu"]["G4"], "Gaussian Pc=1e4"),
            ("248Cm", BENCH["248Cm"]["G4"], "Gaussian Pc=1e4 (+neutron push)")]
    rsub = np.random.SeedSequence(SEED + 1).spawn(len(scen) * 10)
    k = 0
    for n, P, lab in scen:
        tau1 = res["sources"][n]["unit_torque"]
        syst = (-abs(budget[n]["Fission-neutron momentum"]) - budget[n]["Decay-heat radiation (heater 1%)"]) * tau1
        rec = []
        for j in range(10):
            sysN = np.random.default_rng(rsub[k]).normal(0, nmP / 2) * tau1
            t, th, st = simulate(P * tau1 + syst + sysN, 30, dt, rsub[k]); k += 1
            m, e, _, _ = demodulate(t, reconstruct_torque(th, dt), dt)
            rec.append((m / tau1, e / abs(tau1)))
        rec = np.array(rec)
        inj[f"{n} | {lab}"] = dict(P_true=P, P_rec_mean=float(rec[:, 0].mean()),
                                   P_rec_sd=float(rec[:, 0].std(ddof=1)), sem_typ=float(rec[:, 1].mean()),
                                   runs=rec.tolist())
    res["injection"] = inj

    # 7. reach vs lambda (60-day statistics at 2 sigma, plus Newtonian floor rescaled)
    lams = np.logspace(-3, 1, 41)
    reach = {}
    for n in ["238U", "242Pu", "244Pu", "248Cm"]:
        s = SOURCES[n]; tau_ref = res["sources"][n]["unit_torque"]
        sig60 = e0 * math.sqrt(120 / 60)
        floor = res["sources"][n]["newton_mc"]["flip_p95_good"] * abs(tau_ref)
        neut = abs(sys_neutron(s))
        out = []
        for lam in lams:
            tu = abs(unit_torque(s, lam))
            out.append(max(2 * sig60, floor, 0.1 * neut) / tu if tu > 0 else np.inf)
        reach[n] = np.array(out)
    res["reach_lambda"] = dict(lam=lams.tolist(), **{n: v.tolist() for n, v in reach.items()})

    # 8. lambda reconstruction: detection of 244Pu at Pc=1e10 (6.1e-3) with lambda = 2 cm
    lam_true, P_true = 0.02, BENCH["244Pu"]["G10"]
    s = SOURCES["244Pu"]; rs = np.array([0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
    def ut(r, lam):
        s2 = dict(s); s2["r"] = r; return unit_torque(s2, lam)
    tau_obs, tau_err = [], []
    rng = np.random.default_rng(rng_seeds[15])
    for r in rs:
        e = e0 * math.sqrt(120 / 30)                   # 30 days per separation
        tau_obs.append(P_true * ut(r, lam_true) + rng.normal(0, e)); tau_err.append(e)
    tau_obs, tau_err = np.array(tau_obs), np.array(tau_err)
    lam_grid = np.logspace(-2.7, 0.5, 120)
    U = np.array([[ut(r, l) for r in rs] for l in lam_grid])
    chi2 = np.empty(len(lam_grid)); Pbest = np.empty(len(lam_grid))
    for i in range(len(lam_grid)):
        w = 1 / tau_err ** 2; Pb = np.sum(w * U[i] * tau_obs) / np.sum(w * U[i] ** 2)
        Pbest[i] = Pb; chi2[i] = np.sum(w * (tau_obs - Pb * U[i]) ** 2)
    ib = int(np.argmin(chi2)); ok = chi2 <= chi2[ib] + 1
    res["lambda_fit"] = dict(lam_true=lam_true, P_true=P_true, lam_best=float(lam_grid[ib]),
                             lam_1sig=[float(lam_grid[ok].min()), float(lam_grid[ok].max())],
                             P_best=float(Pbest[ib]), r=rs.tolist(),
                             tau_obs=tau_obs.tolist(), tau_err=tau_err.tolist(),
                             grid=lam_grid.tolist(), chi2=chi2.tolist(), Pgrid=Pbest.tolist())

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    cols = {"238U": "#555", "242Pu": "#d9886b", "244Pu": "#c4532c", "248Cm": "#3a78c2", "252Cf": "#2a9d6f"}

    # Fig A: time series excerpt
    t, th, st = simulate(BENCH["244Pu"]["G10"] * res["sources"]["244Pu"]["unit_torque"], 2, dt, rng_seeds[16])
    te = reconstruct_torque(th, dt)
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    h = t / 3600
    ax[0].plot(h, th * 1e9, lw=.6); ax[0].set_ylabel("θ (nrad)")
    ax[1].plot(h, te * 1e18, lw=.3, color="#999"); ax[1].set_ylabel("τ_est (aN·m)")
    kern = np.ones(30) / 30; ax[1].plot(h, np.convolve(te, kern, "same") * 1e18, lw=.8, color="k")
    ax[2].step(h, st, where="post", color=cols["244Pu"]); ax[2].set_ylabel("source in\nposition")
    ax[2].set_xlabel("time (h)")
    ax[0].set_title("Simulated pendulum record, 1 g 244Pu at 2 cm, P = 6.1e-3 (Gaussian, Pc = 1e10)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_timeseries.png"); plt.close(fig)

    # Fig B: noise ASD + sensitivity vs time
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    t, th, _ = simulate(0.0, 20, dt, rng_seeds[17]); te = reconstruct_torque(th, dt)
    f, P = signal.welch(te, 1 / dt, nperseg=2 ** 14)
    ax[0].loglog(f[1:], np.sqrt(P[1:]), lw=.7, label="reconstructed torque (sim)")
    ax[0].loglog(f[1:], np.sqrt(S_thermal(f[1:])), "--", label="thermal (FDT, structural)")
    ax[0].loglog(f[1:], np.sqrt(S_torque(f[1:])), ":", label="thermal + red excess")
    ax[0].axvline(f_s, color="k", lw=.6); ax[0].set_ylim(1e-17, 1e-12)
    ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("torque ASD (N·m/√Hz)"); ax[0].legend(fontsize=7)
    for n in ["238U", "242Pu", "248Cm"]:
        lab = n if n != "242Pu" else "242/244Pu (1 g)"
        ax[1].loglog(Tlist, res["sources"][n]["sigmaP_vs_T"], "-", color=cols[n], label=lab)
        mt = np.array([np.nan if v is None else v for v in sig_T_meas]) / abs(res["sources"][n]["unit_torque"])
        ax[1].loglog(Tlist, mt, "o", color=cols[n], ms=3)
    ax[1].axhline(1e-4, color="k", ls="--", lw=.7); ax[1].text(1.1, 1.2e-4, "10⁻⁴ floor", fontsize=7)
    ax[1].set_xlabel("integration time (days)"); ax[1].set_ylabel("1σ statistical error on P")
    ax[1].set_title("lines: σ_day/√T; points: scatter of independent chunks", fontsize=7)
    ax[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig_noise.png"); plt.close(fig)

    # Fig C: Newtonian MC
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for lab, (raw, flip, _), col in (("homogeneous (0.1%/mm)", mc["242Pu"]["good"], "#c4532c"),
                                     ("pressed oxide (1%/mm)", mc["242Pu"]["oxide"], "#3a78c2")):
        bins = np.logspace(-8, -1.5, 50)
        ax[0].hist(abs(raw), bins, histtype="step", color=col, label=f"{lab}, raw")
        ax[0].hist(abs(flip), bins, histtype="stepfilled", alpha=.35, color=col, label=f"{lab}, capsule flip")
    ax[0].axvline(1e-4, color="k", ls="--", lw=.7); ax[0].set_xscale("log")
    ax[0].set_xlabel("|P_fake| from Newtonian residual (1 g Pu, 2 cm)"); ax[0].legend(fontsize=6)
    ax[1].scatter(abs(fits[:, 0]), abs(fits[:, 1]), s=4, color="#c4532c", label="after 5-separation fit")
    ax[1].plot([1e-9, 1e-2], [1e-9, 1e-2], "k:", lw=.6)
    ax[1].axhline(1e-4, color="k", ls="--", lw=.7)
    ax[1].scatter(abs(fits[:, 0]), abs(fits[:, 2]), s=4, color="#555", label="mass-mismatch term Δm/M")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("|P_fake| at 2 cm, oxide pellet, no flip")
    ax[1].set_ylabel("|P_fake| after template fit"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_newton_mc.png"); plt.close(fig)

    # Fig D: systematics budget
    fig, ax = plt.subplots(figsize=(9, 4))
    names = ["238U", "242Pu", "244Pu", "248Cm", "252Cf"]; keys = list(budget["242Pu"].keys())
    wbar = 0.8 / len(keys); cmap = plt.get_cmap("tab10")
    for j, kk in enumerate(keys):
        v = [budget[n][kk] if budget[n][kk] else np.nan for n in names]
        v = [x if (x is not None and x > 0) else np.nan for x in v]
        ax.bar(np.arange(5) + (j - len(keys) / 2) * wbar + wbar / 2, v, wbar, label=kk, color=cmap(j), log=True)
    for i, n in enumerate(names):
        for key, mk in (("G4", "o"), ("G10", "^"), ("PL10", "D")):
            ax.plot(i, BENCH[n][key], mk, color="k", ms=5, mfc="none")
    ax.plot([], [], "ko", mfc="none", label="pred. Gaussian Pc=1e4"); ax.plot([], [], "k^", mfc="none", label="pred. Gaussian Pc=1e10")
    ax.plot([], [], "kD", mfc="none", label="pred. power law Pc=1e10")
    ax.set_xticks(range(5)); ax.set_xticklabels(names); ax.set_ylim(1e-14, 1e4)
    ax.set_ylabel("|P| equivalent (λ = 0.5 m)"); ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_budget.png"); plt.close(fig)

    # Fig E: reach vs lambda
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n in ["238U", "242Pu", "244Pu", "248Cm"]:
        ax.loglog(lams, reach[n], color=cols[n], label=n + (" (stat ≡ 242Pu)" if n == "244Pu" else ""),
                  ls="--" if n == "244Pu" else "-")
    for n in ["242Pu", "244Pu", "248Cm"]:
        ax.plot(0.5, BENCH[n]["G4"], "o", color=cols[n], mfc="none"); ax.plot(0.5, BENCH[n]["G10"], "^", color=cols[n], mfc="none")
    ax.set_xlabel("range λ (m)"); ax.set_ylabel("|P| reach (60 d, 2σ, with floors)")
    ax.set_ylim(1e-6, 1e3); ax.legend(fontsize=7)
    ax.set_title("Markers: Gaussian predictions at λ = 0.5 m (○ Pc=1e4, △ Pc=1e10)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_reach_lambda.png"); plt.close(fig)

    # Fig F: injection recovery + lambda fit
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    labs = list(inj.keys())
    for i, kk in enumerate(labs):
        runs = np.array(inj[kk]["runs"])
        pull = (runs[:, 0] - inj[kk]["P_true"]) / runs[:, 1]
        ax[0].plot(np.full(len(runs), i) + np.linspace(-.25, .25, len(runs)), pull, "o", ms=3, color="#3a78c2")
        ax[0].hlines(pull.mean(), i - .35, i + .35, color="#c4532c")
    ax[0].axhspan(-1, 1, color="#ddd", zorder=0); ax[0].axhline(0, color="k", lw=.6)
    short = ["244Pu\nnull", "244Pu\nGauss 1e4", "244Pu\nPL 1e10", "242Pu\nGauss 1e4", "248Cm\nGauss 1e4\n+n push"]
    ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels(short, fontsize=7)
    ax[0].set_ylabel("pull (P_rec − P_true)/σ, 30-day runs")
    ax[0].set_title("Injection-recovery (red bar: mean pull)", fontsize=8)
    lf = res["lambda_fit"]
    ax[1].semilogx(lf["grid"], np.array(lf["chi2"]) - min(lf["chi2"]), color="#c4532c")
    ax[1].axvline(lf["lam_true"], color="k", ls=":"); ax[1].axhline(1, color="k", lw=.5)
    ax[1].set_ylim(0, 25); ax[1].set_xlim(5e-3, 0.2); ax[1].set_xlabel("λ (m)"); ax[1].set_ylabel("Δχ²")
    ax[1].set_title("λ reconstruction, 6 separations × 30 d, 244Pu P=6.1e-3, λ=2 cm", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_injection_lambda.png"); plt.close(fig)

    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(clean(res), fh, indent=1)
    print(json.dumps(clean({k: res[k] for k in ("noise", "multisep", "lambda_fit")}), indent=1)[:3000])
    for n in SOURCES:
        s = res["sources"][n]
        print(n, {k: (f"{v:.3g}" if isinstance(v, float) else None) for k, v in s.items() if isinstance(v, float)})
    for n, b in budget.items(): print(n, {k: (f"{v:.2g}" if v else None) for k, v in b.items()})
    for k, v in inj.items(): print(k, {kk: (f"{vv:.3g}" if isinstance(vv, float) else None) for kk, vv in v.items() if kk != "runs"})

if __name__ == "__main__":
    main()
