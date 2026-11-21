"""heavy_pair_sim.py - the paper's campaigns re-simulated with a test body of the source's own nuclide.

    python heavy_pair_sim.py [runs_per_config]        (default 2000; uses every CPU core)

For each source (U-238 100 g at 3 cm for 30 d; Pu-242 and Pu-244 1 g at 2 cm for 90 d) this runs the paper's
time-domain simulation (active_mass_sim.py, unchanged) twice: with the paper's 3 g copper test body and with a 3 g
test body of the source's own element. What changes with the test body:
  - its radius (density), which enters the Yukawa form factor and the areas that intercept neutrons, heat and gammas;
  - its susceptibility (paramagnetic term);
  - the neutron momentum it takes up (copper's transport cross section in the paper; for U and Pu an upper bound,
    8 b, about their total cross section near 2 MeV, as if every collision transferred all the momentum);
  - its gamma absorption (copper 50% in the paper; U and Pu stop 59.5 keV gammas fully, so 100%).
The Newtonian coupling is unchanged: a uniform sphere attracts exactly like a point of the same mass.

Systematic budget as in the paper (detection_map.sigma_sys; the uranium ferromagnetic term per gram as in
u_stage_refine.py), with the paramagnetic term added. Statistical noise from the pooled calibration (factor 1.053).

Per configuration: null runs (statistical sigma), then injection-recovery at the five-sigma threshold with a
systematic offset drawn from N(0, sigma_sys) in every run, so the scatter of recovered values is the total error.
Writes out/heavy_pair_sim.json.
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "trapezoid"):  # numpy < 2.0 (the simulation calls np.trapezoid)
    np.trapezoid = np.trapz

HERE = Path(__file__).resolve().parent
SIMDIR = HERE.parent / "1_experiment_sim"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(SIMDIR))
_argv = sys.argv
sys.argv = [sys.argv[0], str(OUT / "_sim_scratch")]
import active_mass_sim as S  # noqa: E402
sys.argv = _argv

R = json.load(open(SIMDIR / "results.json"))
UREF = json.load(open(SIMDIR / "u_stage_refine.json"))
CC = json.load(open(SIMDIR / "calibration_check.json"))
CAL = CC["pooled_factor"]
SEED = 20261117
DT = 10.0

MATERIALS = {
    "Cu": dict(rho=8960.0, chi=-9.63e-6, n=S.N_CU, sig_tr=S.SIG_TR_CU, gamma_abs=0.5),
    "U": dict(rho=19050.0, chi=4.1e-4, n=19050.0 / (238.05 * S.U_AMU), sig_tr=8.0e-28, gamma_abs=1.0),
    "Pu": dict(rho=15920.0, chi=4.4e-4, n=15920.0 / (243.0 * S.U_AMU), sig_tr=8.0e-28, gamma_abs=1.0),
}
CONFIGS = [("238U", "Cu", 30), ("238U", "U", 30), ("242Pu", "Cu", 90), ("242Pu", "Pu", 90),
           ("244Pu", "Cu", 90), ("244Pu", "Pu", 90)]
LAMBDAS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 10.0]


def set_test_body(mat):
    m = MATERIALS[mat]
    S.PEND.update(rho_t=m["rho"], chi_t=m["chi"])
    S.PEND["a_t"] = (3 * S.PEND["m_t"] / (4 * math.pi * m["rho"])) ** (1 / 3)
    S.N_CU, S.SIG_TR_CU = m["n"], m["sig_tr"]


def budget(src_name, mat):
    set_test_body(mat)
    s = S.SOURCES[src_name]
    tau1 = S.unit_torque(s, 0.5)
    nm = R["sources"][src_name]["newton_mc"]["flip_p95_good"] / 2
    ferro = UREF["ferro"]["238U"]["per_gram_spec"] if src_name == "238U" else 1e-5
    terms = dict(
        newtonian=nm,
        ferromagnetic=ferro,
        neutron_10pct=0.1 * abs(S.sys_neutron(s) / tau1),
        heat_1pct=abs(S.sys_thermal_rad(s)[0] / tau1),
        gamma=abs(S.sys_gamma_am(s) / tau1) * MATERIALS[mat]["gamma_abs"] / 0.5,
        paramagnetic=abs(S.sys_paramagnetic(s) / tau1),
    )
    s_sys = math.sqrt(sum(v * v for v in terms.values()))
    tau_lam = {str(l): S.unit_torque(s, l) for l in LAMBDAS}
    return dict(tau1=tau1, terms=terms, s_sys=s_sys, a_t=S.PEND["a_t"], tau_lambda=tau_lam)


def run_one(args):
    """One time-domain campaign. Returns the recovered anomaly (P units)."""
    src_name, mat, days, p_inj, bias_sd, seed = args
    set_test_body(mat)
    s = S.SOURCES[src_name]
    tau1 = S.unit_torque(s, 0.5)
    ss = np.random.SeedSequence(seed)
    bias = float(np.random.default_rng(ss.spawn(1)[0]).normal(0, bias_sd)) if bias_sd > 0 else 0.0
    t, th, _ = S.simulate((p_inj + bias) * tau1, days, DT, ss)
    m, e, _, _ = S.demodulate(t, S.reconstruct_torque(th, DT), DT)
    return m / tau1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    t0 = time.time()
    res = dict(seed=SEED, runs_per_config=n, calibration=CAL, materials=MATERIALS, configs={})
    # common random numbers: the copper and heavy runs of one source share every noise and offset draw, so their
    # difference isolates the test body (the noise model does not depend on it)
    root = np.random.SeedSequence(SEED)
    src_seeds = {X: root.spawn(1)[0] for X in ("238U", "242Pu", "244Pu")}
    seed_lists = {X: [int(x.generate_state(1)[0]) for x in ss.spawn(2 * n)] for X, ss in src_seeds.items()}
    with mp.Pool(os.cpu_count()) as pool:
        for src_name, mat, days in CONFIGS:
            b = budget(src_name, mat)
            seeds = seed_lists[src_name][:n]
            # null runs: the statistical scatter of the whole time-domain pipeline, used directly as sigma_stat
            null = np.array(pool.map(run_one, [(src_name, mat, days, 0.0, 0.0, sd) for sd in seeds], chunksize=8))
            sd_day_raw = float(np.std(null, ddof=1) * math.sqrt(days))
            s_stat = float(np.std(null, ddof=1))
            s_tot = math.hypot(s_stat, b["s_sys"])
            T5 = 5 * s_tot
            inj_seeds = seed_lists[src_name][n:]
            rec = np.array(pool.map(run_one, [(src_name, mat, days, T5, b["s_sys"], sd) for sd in inj_seeds],
                                    chunksize=8))
            key = f"{src_name} on {mat}"
            res["configs"][key] = dict(
                source=src_name, test_body=mat, days=days, budget=b,
                sigma_day_null=sd_day_raw, sigma_stat=s_stat, sigma_tot=s_tot, S=2 * s_tot, T5=T5,
                injected=T5, recovered_mean=float(rec.mean()), recovered_sd=float(rec.std(ddof=1)),
                bias_in_sigma=float((rec.mean() - T5) / (rec.std(ddof=1) / math.sqrt(len(rec)))),
                detect_fraction_at_T5=float(np.mean(rec > T5)),
                sd_ratio_recovered_to_predicted=float(rec.std(ddof=1) / s_tot))
            c = res["configs"][key]
            print(f"{key:13s} {days:3d} d | sigma_day {sd_day_raw:.3e} | sys {b['s_sys']:.3e} | S {c['S']:.3e} | "
                  f"recovered/injected {c['recovered_mean'] / T5:.4f} (bias {c['bias_in_sigma']:+.1f} se), "
                  f"sd ratio {c['sd_ratio_recovered_to_predicted']:.3f}, detected {c['detect_fraction_at_T5']:.3f}"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
    for src_name in ("238U", "242Pu", "244Pu"):
        cu, hv = res["configs"][f"{src_name} on Cu"], res["configs"][f"{src_name} on {'U' if src_name == '238U' else 'Pu'}"]
        res.setdefault("heavy_over_copper", {})[src_name] = dict(S_ratio=hv["S"] / cu["S"],
                                                                sys_ratio=hv["budget"]["s_sys"] / cu["budget"]["s_sys"],
                                                                tau1_ratio=hv["budget"]["tau1"] / cu["budget"]["tau1"])
    res["paper_table_III"] = {X: dict(sigma_day=v["sigma_day"], s_sys=v["s_sys"], S=v["S"])
                              for X, v in json.load(open(HERE.parent / "paper" / "numbers.json"))["nuclides"].items()
                              if "S" in v}
    json.dump(res, open(OUT / "heavy_pair_sim.json", "w"), indent=1, default=float)
    print(json.dumps(res["heavy_over_copper"], indent=1))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
