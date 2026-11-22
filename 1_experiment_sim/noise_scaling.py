#!/usr/bin/env python3
"""
noise_scaling.py - the measurement's statistical error when the balance's torque noise sits further above its thermal
limit than the representative pendulum of active_mass_sim.py (rev 5s, 2026-11-22).

Why: active_mass_sim.py's torque noise is thermal x (1 + f_c/f) with f_c = 2.5e-4 Hz, i.e. 1.14 x thermal in amplitude
at the exchange frequency (8.33e-4 Hz). Real balances reach about 1.6-2 x thermal under good conditions (low gas
pressure, magnetic shielding, gold-coated surfaces, well-separated torsion and pendulum modes). This script measures
how sigma_P per day, the integration time and the campaign reach grow with that ratio.

beta = torque-noise amplitude / thermal amplitude at the exchange frequency. Two spectral shapes bracket a real
balance:
  red      thermal x (1 + f_c/f), f_c = (beta^2 - 1) f_sig: the excess rises toward low frequency (the
           representative pendulum's shape, so beta = 1.14 reproduces it);
  uniform  thermal x beta^2 at every frequency (a lossier or warmer fibre).
Readout noise is unchanged. Every case uses the same NRUN noise draws (common random numbers), so the ratios between
cases are much tighter than each sigma. Absolute sigma_P/day = ratio to the representative case x the paper's
calibrated value (calibration_check.py, 520 records).

Run from this folder:  python noise_scaling.py   ->  noise_scaling.json   (about 5 minutes)
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
UREF = json.load(open("u_stage_refine.json"))
CC = json.load(open("calibration_check.json"))
NRUN = 300
DAYS = 90.0
DT = 10.0
LAM = 0.5
F_SIG = 1 / (2 * S.PEND["T_half"])
BETA_REP = math.sqrt(1 + S.PEND["red_corner"] / F_SIG)
S_THERMAL = S.S_thermal
S_TORQUE_ORIG = S.S_torque


def shape(kind, beta):
    if kind == "red":
        fc = (beta ** 2 - 1) * F_SIG
        return lambda f: S_THERMAL(f) * (1 + fc / f)
    return lambda f: S_THERMAL(f) * beta ** 2


CASES = {"thermal": ("uniform", 1.0), "representative": ("red", BETA_REP)}
for b in (1.6, 2.0):
    CASES["red_%.1f" % b] = ("red", b)
    CASES["uniform_%.1f" % b] = ("uniform", b)

seeds = np.random.SeedSequence(S.SEED + 5150).spawn(NRUN)
tau_unit = {src: abs(S.unit_torque(S.SOURCES[src], LAM)) for src in ("238U", "244Pu", "248Cm")}
sd_tau = {}
for name, (kind, beta) in CASES.items():
    S.S_torque = shape(kind, beta)
    means = []
    for i in range(NRUN):
        t, th, _ = S.simulate(0.0, DAYS, DT, seeds[i])
        m, _, _, _ = S.demodulate(t, S.reconstruct_torque(th, DT), DT)
        means.append(m)
    S.S_torque = S_TORQUE_ORIG
    sd_tau[name] = float(np.std(means, ddof=1) * math.sqrt(DAYS))
    print("%-16s beta %.3f  sigma_tau/day %.3g N m" % (name, beta, sd_tau[name]), flush=True)

# paper's calibrated sigma_P per day (representative pendulum) and systematics, as in instrument_calibration.py
sig_rep = {src: R["sources"][src]["sigmaP_1day"] * CC["pooled_factor"] for src in tau_unit}
CAMP = {"238U": 30, "244Pu": 90, "248Cm": 180}
SYS = {"238U": UREF["flip_only_corrected_ferro"]["per_gram_spec"]["sigma_sys"],
       "244Pu": DET["campaigns"]["244Pu 1 g, 90 d"]["sigma_sys"],
       "248Cm": DET["campaigns"]["248Cm 10 mg, 180 d"]["sigma_sys"]}
TARGET = 1e-4 / 3     # the paper's "statistical error of 1e-4/3"

out = dict(nrun=NRUN, days=DAYS, f_sig_Hz=F_SIG, beta_representative=BETA_REP, cases={})
for name, (kind, beta) in CASES.items():
    ratio = sd_tau[name] / sd_tau["representative"]
    sig = {src: v * ratio for src, v in sig_rep.items()}
    camp = {}
    for src, d in CAMP.items():
        st = sig[src] / math.sqrt(d)
        camp[src] = dict(days=d, sigma_stat=st, sigma_sys=SYS[src], S=2 * math.hypot(st, SYS[src]))
    out["cases"][name] = dict(shape=kind, beta=beta, sigma_tau_day_Nm=sd_tau[name], ratio_to_representative=ratio,
                              ratio_to_thermal=sd_tau[name] / sd_tau["thermal"], sigmaP_day=sig,
                              days_to_target_Pu244=(sig["244Pu"] / TARGET) ** 2, campaigns=camp)
json.dump(out, open("noise_scaling.json", "w"), indent=1)

print("\nbeta at the exchange frequency of the representative pendulum: %.3f" % BETA_REP)
print("%-16s %6s %8s %8s %10s %8s %9s %9s %9s" % ("case", "beta", "x rep", "x therm", "sigPu/day", "days",
                                                  "S U 30d", "S Pu 90d", "S Cm 180d"))
for name, c in out["cases"].items():
    print("%-16s %6.2f %8.3f %8.3f %10.3g %8.0f %9.2g %9.2g %9.2g" % (
        name, c["beta"], c["ratio_to_representative"], c["ratio_to_thermal"], c["sigmaP_day"]["244Pu"],
        c["days_to_target_Pu244"], c["campaigns"]["238U"]["S"], c["campaigns"]["244Pu"]["S"],
        c["campaigns"]["248Cm"]["S"]))
