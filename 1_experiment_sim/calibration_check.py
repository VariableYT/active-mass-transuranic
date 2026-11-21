#!/usr/bin/env python3
"""
calibration_check.py - does detection_map.py's statistical calibration (factor 0.85 on sigma_P per day) reproduce,
and does it agree with independent null runs?

instrument_calibration.py's null runs (24 x 30 d, fresh seeds) gave sigma_P/day = 6.4e-4 for 1 g Pu-244 at 2 cm,
1.45x the paper's 4.4e-4 (= 5.2e-4 x 0.85). This script
  A. replays detection_map.main()'s validation loop exactly (same seeds, 60 x 90 d, injected P + systematic bias)
     and recomputes its factor;
  B. runs 60 independent 90-day null runs (fresh seeds) and 60 independent 30-day null runs;
  C. reports sigma_P/day from each, with standard errors.
Run from this folder:  python calibration_check.py  ->  calibration_check.json
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
src = "244Pu"; days = 90
tau1 = R["sources"][src]["unit_torque"]
s1 = R["sources"][src]["sigmaP_1day"]
# detection_map.sigma_sys / sigma_tot with CAL = 1 (inlined: detection_map.py reads out/results.json on import)
b = R["budget"][src]
ssys = math.sqrt(sum(x * x for x in (R["sources"][src]["newton_mc"]["flip_p95_good"] / 2, 1e-5,
                                     0.1 * b["Fission-neutron momentum"], b["Decay-heat radiation (heater 1%)"],
                                     b["241Am gamma pressure"] or 0.0)))
s_prov = math.hypot(s1 / math.sqrt(days), ssys)
assert abs(ssys / json.load(open("detection_results.json"))["campaigns"]["244Pu 1 g, 90 d"]["sigma_sys"] - 1) < 1e-9

# A. exact replay
seeds = np.random.SeedSequence(S.SEED + 7).spawn(60)
res = []; j = 0
for mult in (4.0, 5.0, 6.0):
    P = mult * s_prov
    for i in range(20):
        bias = np.random.default_rng(seeds[j]).normal(0, ssys)
        t, th, _ = S.simulate((P + bias) * tau1, days, 10.0, seeds[j]); j += 1
        m, e, _, _ = S.demodulate(t, S.reconstruct_torque(th, 10.0), 10.0)
        res.append((m / tau1 - P, bias, m / tau1 - P - bias))
res = np.array(res)
sd_tot = float(np.std(res[:, 0], ddof=1))
sd_stat_dm = math.sqrt(max(sd_tot ** 2 - ssys ** 2, 1e-30))
factor_dm = sd_stat_dm / (s1 / math.sqrt(days))
# the same runs with the known bias removed: pure statistical scatter
sd_stat_direct = float(np.std(res[:, 2], ddof=1))
factor_direct = sd_stat_direct / (s1 / math.sqrt(days))
# the bias was drawn from the SAME generator seed as the noise: check its realised scatter
bias_sd = float(np.std(res[:, 1], ddof=1))
corr_bias_noise = float(np.corrcoef(res[:, 1], res[:, 2])[0, 1])


def nulls(n, d, seed_off):
    sd = np.random.SeedSequence(S.SEED + seed_off).spawn(n)
    ms = []
    for k in range(n):
        t, th, _ = S.simulate(0.0, d, 10.0, sd[k])
        m, e, _, _ = S.demodulate(t, S.reconstruct_torque(th, 10.0), 10.0)
        ms.append(m / tau1)
    return float(np.std(ms, ddof=1) * math.sqrt(d))


n90 = nulls(60, 90, 9001)
n30 = nulls(60, 30, 9002)


def null_means(n, d, seed_off):
    sd = np.random.SeedSequence(S.SEED + seed_off).spawn(n)
    ms = []
    for k in range(n):
        t, th, _ = S.simulate(0.0, d, 10.0, sd[k])
        m, e, _, _ = S.demodulate(t, S.reconstruct_torque(th, 10.0), 10.0)
        ms.append(m / tau1)
    return np.array(ms)


# D. a large null set: 400 x 90 d, for the calibration itself and to check whether the run means are Gaussian
big = null_means(400, 90, 9100)
big_sd_day = float(np.std(big, ddof=1) * math.sqrt(90))
z = (big - big.mean()) / big.std(ddof=1)
big_kurt = float(np.mean(z ** 4) - 3)
big_frac_3sig = float(np.mean(np.abs(z) > 3))
# robust scale (normal-consistent MAD) as a cross-check on heavy tails
big_mad_day = float(1.4826 * np.median(np.abs(big - np.median(big))) * math.sqrt(90))
# pooled estimate from all 520 independent 90-day records (replay with its known bias removed, 60 fresh, 400 fresh)
var = [(sd_stat_direct ** 2 * days, 60), (n90 ** 2, 60), (big_sd_day ** 2, 400)]
pooled = math.sqrt(sum(v * n for v, n in var) / sum(n for _, n in var))
out = dict(sigmaP_1day_first_report=s1, s_prov=s_prov, sigma_sys=ssys,
           big_400x90d=dict(sigma_day=big_sd_day, se_frac=1 / math.sqrt(2 * 399), excess_kurtosis=big_kurt,
                            frac_beyond_3sigma=big_frac_3sig, mad_sigma_day=big_mad_day),
           pooled_90d_sigma_day=pooled, pooled_factor=pooled / s1, pooled_n=520, pooled_se_frac=1 / math.sqrt(2 * 519),
           replay=dict(sd_tot=sd_tot, sd_stat_detection_map=sd_stat_dm, factor_detection_map=factor_dm,
                       sd_stat_bias_removed=sd_stat_direct, factor_bias_removed=factor_direct,
                       realised_bias_sd=bias_sd, corr_bias_vs_noise=corr_bias_noise),
           null_90d_sigma_day=n90, null_30d_sigma_day=n30, se_frac_60=1 / math.sqrt(2 * 59))
json.dump(out, open("calibration_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("sigma_P/day: detection_map-calibrated %.3g | replay, bias removed %.3g | null 90 d %.3g | null 30 d %.3g (+-9%% each)"
      % (factor_dm * s1, factor_direct * s1, n90, n30))
print("400 x 90 d: sigma_day %.3g (MAD %.3g), excess kurtosis %.2f, beyond 3 sigma %.3f; pooled 520 runs %.3g (factor %.3f +- %.0f%%)" % (
    big_sd_day, big_mad_day, big_kurt, big_frac_3sig, pooled, pooled / s1, 100 / math.sqrt(2 * 519)))
