#!/usr/bin/env python3
"""
instrument_calibration.py - the measurement's statistical error with named, published torsion balances in place of
the representative pendulum of active_mass_sim.py.

The experiment's own design (3 g Cu test sphere on a 2 cm arm, sources at the paper's distances, source/dummy
exchange every 600 s) is kept; only the instrument's torque noise changes. Thermal torque noise depends on the fibre
(kappa / Q at the signal frequency) and temperature, not on the moment of inertia, so a published kappa and Q fix it.

Instruments (published numbers only):
  representative  kappa 1.10e-9 N m/rad, Q 4000, T0 600 s, room temperature            (active_mass_sim.PEND)
  eotwash_ep      kappa 2.34e-9 N m/rad, Q 5000, T0 798 s, 20 um W fibre, room temp.
                  Schlamminger et al., PRL 100, 041101 (2008); "operates close to the thermal limit",
                  Wagner et al., CQG 29, 184002 (2012).
  vienna_mg       Westphal et al., Nature 591, 225 (2021): 90 mg Au test mass, f0 3.59 mHz, Q 4.9; statistical
                  uncertainty ~3e-12 m/s^2 from ~150 h of good data. Used as a measured acceleration noise.

Method:
  1. analytic thermal sigma_P per day for each fibre (same formula as paper/compute.py), and the ratio to the
     representative pendulum;
  2. time-domain check with active_mass_sim.simulate/demodulate: NRUN null runs of DAYS days for each fibre
     (common random numbers across fibres), with
     (a) the representative excess low-frequency noise (red corner 2.5e-4 Hz) and readout, (b) thermal noise and
     readout only. sigma_P per day = scatter of the run means x sqrt(DAYS) / unit torque. Seeded.
  3. campaign reach S = 2 sigma_tot for each source with each instrument, systematics as in detection_map.py
     (uranium as revised in u_stage_refine.py).

Run from this folder:  python instrument_calibration.py   ->  instrument_calibration.json
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
CALF = DET["calibration"]["factor_vs_120d_estimate"]
LAM = 0.5
NRUN = 60
DAYS = 90.0
DT = 10.0
kB = 1.380649e-23

BASE = {k: S.PEND[k] for k in ("I", "T0", "Q", "w0", "kappa", "red_corner", "readout_asd")}
FIBRES = {
    "representative": dict(kappa=BASE["kappa"], Q=BASE["Q"], T0=BASE["T0"],
                           ref="active_mass_sim.py representative pendulum"),
    "eotwash_ep": dict(kappa=2.34e-9, Q=5000.0, T0=798.0,
                       ref="Schlamminger et al., PRL 100, 041101 (2008); Wagner et al., CQG 29, 184002 (2012)"),
}
VIENNA = dict(acc_stat=3e-12, hours=150.0, ref="Westphal et al., Nature 591, 225 (2021)")


def set_pend(kappa, Q, T0, red=True):
    S.PEND.update(kappa=kappa, Q=Q, T0=T0, w0=2 * math.pi / T0, I=kappa * (T0 / (2 * math.pi)) ** 2,
                  red_corner=BASE["red_corner"] if red else 1e-12)


def restore():
    S.PEND.update(BASE)


f_sig = 1 / (2 * S.PEND["T_half"])
tau_unit = {src: abs(S.unit_torque(S.SOURCES[src], LAM)) for src in ("238U", "244Pu", "248Cm")}


def asd_thermal(kappa, Q, T=S.PEND["T"]):
    return math.sqrt(4 * kB * T * kappa / (Q * 2 * math.pi * f_sig))


# ---------------------------------------------------------------- 1. analytic thermal limit
analytic = {}
for name, fb in FIBRES.items():
    a = asd_thermal(fb["kappa"], fb["Q"])
    analytic[name] = dict(asd_Nm_rtHz=a, ratio_to_representative=a / asd_thermal(BASE["kappa"], BASE["Q"]))

# ---------------------------------------------------------------- 2. time-domain null runs
# common random numbers: every fibre sees the same noise draws, so the ratios between fibres are exact
SEEDS = {red: np.random.SeedSequence(S.SEED + 4242 + int(red)).spawn(NRUN) for red in (True, False)}
td = {}
for name, fb in FIBRES.items():
    for red in (True, False):
        set_pend(fb["kappa"], fb["Q"], fb["T0"], red=red)
        means = []
        for i in range(NRUN):
            t, th, _ = S.simulate(0.0, DAYS, DT, SEEDS[red][i])
            m, e, _, _ = S.demodulate(t, S.reconstruct_torque(th, DT), DT)
            means.append(m)
        restore()
        sd_tau_day = float(np.std(means, ddof=1) * math.sqrt(DAYS))
        td["%s_%s" % (name, "red" if red else "thermal")] = dict(
            sigma_tau_day_Nm=sd_tau_day, se_frac=1 / math.sqrt(2 * (NRUN - 1)),
            sigmaP_day={src: sd_tau_day / tau_unit[src] for src in tau_unit})

# the paper's sigma_day (calibrated against 60 full runs) for comparison
paper_sigma_day = {src: R["sources"][src]["sigmaP_1day"] * CALF for src in tau_unit}
# rev 5 calibration: pooled 120 x 90-day runs (calibration_check.py) replaces detection_map's 0.85
CC = json.load(open("calibration_check.json"))
CAL5 = CC["pooled_factor"]
rev5_sigma_day = {src: R["sources"][src]["sigmaP_1day"] * CAL5 for src in tau_unit}
rep_check = td["representative_red"]["sigmaP_day"]["244Pu"] / paper_sigma_day["244Pu"]

# ---------------------------------------------------------------- 3. Vienna miniature balance (measured noise)
a_day = VIENNA["acc_stat"] * math.sqrt(VIENNA["hours"] / 24.0)
aN = {"244Pu": S.G * 1e-3 / 0.02 ** 2, "238U": S.G * 0.1 / 0.03 ** 2, "248Cm": S.G * 1e-5 / 0.01 ** 2}
vienna = dict(acc_per_day=a_day, sigmaP_day={src: a_day / aN[src] for src in aN}, ref=VIENNA["ref"])

# ---------------------------------------------------------------- 4. campaigns with each instrument
CAMP = {"238U": 30, "244Pu": 90, "248Cm": 180}
sys_P = {"238U": UREF["flip_only_corrected_ferro"]["per_gram_spec"]["sigma_sys"],
         "244Pu": DET["campaigns"]["244Pu 1 g, 90 d"]["sigma_sys"],
         "248Cm": DET["campaigns"]["248Cm 10 mg, 180 d"]["sigma_sys"]}


def campaign(sig_day):
    out = {}
    for src, days in CAMP.items():
        st = sig_day[src] / math.sqrt(days)
        tot = math.hypot(st, sys_P[src])
        out[src] = dict(days=days, sigma_day=sig_day[src], sigma_stat=st, sigma_sys=sys_P[src], S=2 * tot, T5=5 * tot)
    return out


camps = {"paper rev 4 (factor 0.85)": campaign(paper_sigma_day),
         "representative, pooled calibration": campaign(rev5_sigma_day),
         "eotwash_ep fibre, pooled calibration x thermal ratio":
             campaign({k: v * analytic["eotwash_ep"]["ratio_to_representative"] for k, v in rev5_sigma_day.items()})}
for k, v in td.items():
    camps[k] = campaign(v["sigmaP_day"])
camps["vienna_mg (measured)"] = campaign(vienna["sigmaP_day"])

out = dict(f_signal_Hz=f_sig, unit_torque_Nm=tau_unit, fibres=FIBRES, analytic_thermal=analytic, time_domain=td,
           nrun=NRUN, days=DAYS, paper_sigma_day=paper_sigma_day, representative_td_over_paper=rep_check,
           cal_rev5=CAL5, rev5_sigma_day=rev5_sigma_day,
           td_ratio_eotwash_to_rep={k: td["eotwash_ep_" + k]["sigma_tau_day_Nm"] / td["representative_" + k]["sigma_tau_day_Nm"]
                                    for k in ("red", "thermal")},
           vienna=vienna, campaigns=camps)
json.dump(out, open("instrument_calibration.json", "w"), indent=1, default=float)

print("signal frequency %.3g Hz; unit torques" % f_sig, {k: "%.3g" % v for k, v in tau_unit.items()})
for k, v in analytic.items():
    print("thermal ASD %-15s %.3g N m/rtHz  (x%.2f representative)" % (k, v["asd_Nm_rtHz"], v["ratio_to_representative"]))
print("time-domain (%d x %g d):" % (NRUN, DAYS))
for k, v in td.items():
    print("   %-26s sigma_P/day Pu-244 %.3g  U %.3g  Cm %.3g  (+-%.0f%%)" % (
        k, v["sigmaP_day"]["244Pu"], v["sigmaP_day"]["238U"], v["sigmaP_day"]["248Cm"], 100 * v["se_frac"]))
print("paper sigma_day Pu-244 %.3g; representative time-domain / paper = %.2f; pooled rev 5 %.3g (factor %.3f)" % (
    paper_sigma_day["244Pu"], rep_check, rev5_sigma_day["244Pu"], CAL5))
print("time-domain ratio eotwash/representative:", {k: round(td["eotwash_ep_" + k]["sigma_tau_day_Nm"] /
      td["representative_" + k]["sigma_tau_day_Nm"], 3) for k in ("red", "thermal")})
print("Vienna mg balance: %.2g m/s^2 per day -> sigma_P/day Pu %.3g, U %.3g, Cm %.3g" % (
    a_day, vienna["sigmaP_day"]["244Pu"], vienna["sigmaP_day"]["238U"], vienna["sigmaP_day"]["248Cm"]))
for k, c in camps.items():
    print("%-36s S: U %.2g  Pu %.2g  Cm %.2g" % (k, c["238U"]["S"], c["244Pu"]["S"], c["248Cm"]["S"]))
