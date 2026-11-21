#!/usr/bin/env python3
"""
u_stage_refine.py - the depleted-uranium stage (100 g at 3 cm, 30 days) with two corrections the first simulation pass left out.

1. Ferromagnetic term. detection_map.sigma_sys() adds a fixed 1e-5 for every source: that is the plutonium value
   (1 g at 2 cm) of the paper's spec, remanent moment < 1e-9 A m^2 on source and test body. In units of the anomaly
   P the dipole-dipole torque is divided by the source's own gravitational torque, which for 100 g at 3 cm is ~37x
   larger, while the dipole force falls as 1/r^4. Here it is recomputed for each geometry with the same coaxial
   worst case as active_mass_sim.sys_ferro, both at the absolute spec and with the moment scaled up with source
   mass (1e-9 A m^2 per gram, the harder requirement for a 100 g source).
2. Capsule flip combined with a multi-separation fit. The flip (capsule rotated 180 deg about its own axis) cancels
   in-plane centre-of-mass offsets and density gradients but leaves the radius and height tolerances (the 'dim'
   term, 1.4e-5 rms for uranium), which dominate the stage's systematic error. Those change only the source's
   higher multipoles, so their torque falls faster with separation than the anomaly's. The 30 days are split over
   five separations, both capsule orientations at each, and P is fitted together with radius and height templates
   by weighted least squares, with each separation's statistical error from the calibrated pendulum noise.
   Monte Carlo over the same tolerance distributions as active_mass_sim.newton_mc (seeded).

Run from this folder:  python u_stage_refine.py   ->  u_stage_refine.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.argv = [sys.argv[0], "out"]
import active_mass_sim as S  # noqa: E402

R = json.load(open("results.json"))
DET = json.load(open("detection_results.json"))
CALF = DET["calibration"]["factor_vs_120d_estimate"]
LAM = 0.5
DAYS = 30.0
NMC = 1000


def ferro_P(src, mu_s, mu_t=1e-9):
    def F(s, ab):
        return 6 * S.MU0 * mu_s * mu_t / (4 * math.pi * s ** 4)
    return abs(S.radial_push(src, F) / S.unit_torque(src, LAM))


ferro = {}
for name in ("238U", "244Pu", "248Cm"):
    src = S.SOURCES[name]
    ferro[name] = dict(abs_spec=ferro_P(src, 1e-9), per_gram_spec=ferro_P(src, 1e-9 * src["M"] * 1e3))

# ---------------------------------------------------------------- stage as in the paper (flip only)
U = S.SOURCES["238U"]
nm = R["sources"]["238U"]["newton_mc"]
s_stat_30 = R["sources"]["238U"]["sigmaP_1day"] / math.sqrt(DAYS) * CALF
other = [0.1 * R["budget"]["238U"]["Fission-neutron momentum"], R["budget"]["238U"]["Decay-heat radiation (heater 1%)"]]


def tot(s_newton, s_ferro):
    s_sys = math.sqrt(s_newton ** 2 + s_ferro ** 2 + sum(x * x for x in other))
    return dict(sigma_newton=s_newton, sigma_ferro=s_ferro, sigma_sys=s_sys, sigma_stat=s_stat_30,
                sigma_tot=math.hypot(s_sys, s_stat_30), S=2 * math.hypot(s_sys, s_stat_30))


paper = tot(nm["flip_p95_good"] / 2, 1e-5)
assert abs(paper["sigma_tot"] / DET["campaigns"]["238U 100 g, 30 d"]["sigma_tot"] - 1) < 1e-6
flip_fixed_ferro = dict(abs_spec=tot(nm["flip_p95_good"] / 2, ferro["238U"]["abs_spec"]),
                        per_gram_spec=tot(nm["flip_p95_good"] / 2, ferro["238U"]["per_gram_spec"]))

# ---------------------------------------------------------------- flip + five separations
RS = np.array([0.025, 0.030, 0.035, 0.045, 0.060])


def at(r):
    s = dict(U); s["r"] = float(r); return s


uref = np.array([S.unit_torque(at(r), LAM) for r in RS])
u3 = S.unit_torque(at(0.030), LAM)
# statistical error per separation: same torque noise, P sensitivity scales with that separation's unit torque
sig_r = R["sources"]["238U"]["sigmaP_1day"] * CALF * abs(u3 / uref) / math.sqrt(DAYS / len(RS))
zero = dict(dM=0.0, dcom=(0, 0, 0), da=0.0, dh=0.0, grad=(0, 0))
templ = []
for pert in (dict(da=1e-6), dict(dh=1e-6)):
    tp = dict(zero); tp.update(pert)
    templ.append(np.array([S.newton_mod_torque(at(r), tp, zero) for r in RS]) / uref)
X = np.vstack([np.ones_like(RS)] + templ).T
W = 1 / sig_r ** 2
A = np.linalg.inv(X.T @ (X * W[:, None]))
H = A @ (X.T * W[None, :])
stat_fit = math.sqrt(A[0, 0])              # statistical error of P after the fit (30 d total)


def run(sig_grad, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((NMC, 3))
    for i in range(NMC):
        ts, td = S.draw_tol(rng, sig_grad), S.draw_tol(rng, sig_grad)
        y_raw3 = S.newton_mod_torque(at(0.030), ts, td) / u3
        y = np.array([0.5 * (S.newton_mod_torque(at(r), ts, td) + S.newton_mod_torque(at(r), S.flipped(ts), S.flipped(td)))
                      for r in RS]) / uref
        sys_part = float(H[0] @ y)
        out[i] = (y_raw3, y[1], sys_part)
    return out


res_mc = {}
for label, sg, seed in (("good", 1.0, S.SEED + 101), ("oxide", 10.0, S.SEED + 102)):
    o = run(sg, seed)
    sys_fit = float(np.percentile(np.abs(o[:, 2]), 95)) / 2         # same 1-sigma convention as the paper (p95 / 2)
    s_sys = math.sqrt(sys_fit ** 2 + ferro["238U"]["per_gram_spec"] ** 2 + sum(x * x for x in other))
    res_mc[label] = dict(flip_only_p95=float(np.percentile(np.abs(o[:, 1]), 95)),
                         flip_plus_fit_sys_p95=float(np.percentile(np.abs(o[:, 2]), 95)),
                         sigma_newton=sys_fit, sigma_sys=s_sys, sigma_stat=stat_fit,
                         sigma_tot=math.hypot(s_sys, stat_fit), S=2 * math.hypot(s_sys, stat_fit))

out = dict(calibration_factor=CALF, ferro=ferro, paper_stage=paper, flip_only_corrected_ferro=flip_fixed_ferro,
           separations_m=RS.tolist(), stat_per_separation=sig_r.tolist(), stat_after_fit=stat_fit,
           flip_plus_multisep=res_mc,
           note="sigma conventions follow detection_map.py: Newtonian 1-sigma = p95/2; S = 2 sigma_tot")
json.dump(out, open("u_stage_refine.json", "w"), indent=1, default=float)
print("ferro P at spec:", {k: {kk: "%.2g" % vv for kk, vv in v.items()} for k, v in ferro.items()})
print("paper U stage: sigma_sys %.3g sigma_tot %.3g S %.3g" % (paper["sigma_sys"], paper["sigma_tot"], paper["S"]))
for k, v in flip_fixed_ferro.items():
    print("flip only, ferro %-13s: sigma_sys %.3g sigma_tot %.3g S %.3g" % (k, v["sigma_sys"], v["sigma_tot"], v["S"]))
print("five separations", RS, "stat per sep", ["%.2g" % x for x in sig_r], "stat after fit %.3g" % stat_fit)
for k, v in res_mc.items():
    print("flip + fit (%s): flip-only p95 %.3g -> fit p95 %.3g; sigma_sys %.3g sigma_tot %.3g S %.3g" % (
        k, v["flip_only_p95"], v["flip_plus_fit_sys_p95"], v["sigma_sys"], v["sigma_tot"], v["S"]))
