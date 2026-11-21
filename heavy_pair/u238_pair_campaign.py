"""u238_pair_campaign.py - Stage 1 with a depleted-uranium test body: time-domain campaigns and error budget.

    python u238_pair_campaign.py [injection_runs_per_cell]      (default 600; null runs are 4x that; all cores)

Source: the paper's 100 g depleted-uranium cylinder at 3 cm against a tungsten dummy. Test body: 3 g sphere of
copper (the paper) or of depleted uranium. Campaigns of 30 days (the paper) and 90 days, ranges 0.05-10 m.

1. Error budget at 0.5 m, the paper's terms plus what a uranium test body or a uranium source adds:
   - paramagnetic coupling versus the residual field B0 (the paper assumes 100 nT, i.e. shielding), and the shielding
     factor that keeps it below 1% of the systematic error;
   - the test body's own remanent moment for 10-300 ppm iron (1% remanent precipitates) against the paper's 1e-9 A m^2;
   - charging: alphas from a bare uranium surface, stopped by a gold coat thicker than their ~10 um range; betas of
     234mPa (in equilibrium with 238U) that a coat does not stop, drained through the tungsten fibre;
   - beta momentum: 234mPa betas from the 100 g source reaching the test body (upper bound: no capsule attenuation;
     a 1 mm tungsten wall stops them). This term is not in the paper's budget and applies to copper too.
2. Time-domain injection-recovery with common random numbers (same noise for both test bodies and every range):
   null runs give the statistical error; injections at the five-sigma threshold carry a systematic offset drawn
   from N(0, sigma_sys) and check bias, scatter and detection probability. All error terms are fixed in torque, so
   at range lambda they scale with the modulated torque per unit anomaly, unit_torque(source, lambda).
3. What the reach means. Copper measures P_U = q_U q0; a uranium test body measures q_U (q0 + q_U). Compared with
   the paper's bound on P_U (Smith et al.) and with the shape-independent bounds on q_U^2 if q0 = 0: uranium's free
   fall (Niebauer et al. 1987, 1 ppm uranium in the ground 0.5 m below, site-dependent) and the normal density of
   large uranium ingots.
Writes out/u238_pair_campaign.json and out/u238_pair_campaign.md.
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heavy_pair_sim as H  # noqa: E402  (imports the paper's simulation as H.S, unchanged)

S = H.S
OUT = H.OUT
NUM = json.load(open(HERE.parent / "paper" / "numbers.json"))
SEED = 20261118
LAMS = [0.05, 0.10, 0.20, 0.50, 1.0, 10.0]
DAYS = [30, 90]
SRC = "238U"
E_CHARGE = 1.602176634e-19
B_OF_LAM = {float(k): v for k, v in NUM["B_of_lambda"].items()}


def b_u(lam):
    ks = sorted(B_OF_LAM)
    return float(np.exp(np.interp(math.log(lam), np.log(ks), np.log([B_OF_LAM[k] for k in ks]))))


def budget_extra(mat):
    """The paper's terms (H.budget) plus the audit's terms, at 0.5 m, in units of the anomaly P."""
    b = H.budget(SRC, mat)
    s = S.SOURCES[SRC]
    tau1 = b["tau1"]
    extra = {}
    # paramagnetic versus residual field: the term goes as B0^2 (both induced moments scale with B0)
    para = {f"{B0:g} T": abs(S.sys_paramagnetic(s, B0) / tau1) for B0 in (50e-6, 5e-6, 5e-7, 1e-7, 1e-8)}
    target = 0.01 * b["s_sys"]
    b0_max = 1e-7 * math.sqrt(target / abs(S.sys_paramagnetic(s, 1e-7) / tau1))
    extra["paramagnetic_vs_field"] = dict(terms=para, B0_max_for_1pct_T=b0_max, shielding_from_50uT=50e-6 / b0_max)
    # the test body's own remanent moment for iron impurities, against the paper's 1e-9 A m^2 requirement
    ms_fe, rho_fe, rem = 1.71e6, 7874.0, 0.01
    mt = S.PEND["m_t"]
    moments = {ppm: ppm * 1e-6 * mt / rho_fe * ms_fe * rem for ppm in (0.1, 1, 10, 100)}
    extra["test_body_moment_A_m2"] = {f"{ppm:g} ppm Fe": v for ppm, v in moments.items()}
    # the paper's term (source at 1e-9 A m^2 per gram, test body at 1e-9 A m^2) scales with the test body's moment
    extra["ferro_term_per_1e-9_A_m2"] = b["terms"]["ferromagnetic"]
    extra["ferro_term_vs_iron"] = {f"{ppm:g} ppm Fe": b["terms"]["ferromagnetic"] * v / 1e-9 for ppm, v in moments.items()}
    extra["iron_ppm_at_spec"] = 1e-9 / moments[1]
    # charging
    a_t = S.PEND["a_t"]
    act_g = NUM["nuclides"]["U-238"]["act"]                         # Bq per gram, 238U (234mPa in equilibrium)
    if mat == "U":
        rho = H.MATERIALS["U"]["rho"]
        alpha_bare = act_g * rho * 1e3 * 4 * math.pi * a_t ** 2 * 10e-6 / 4     # Bq/m^3 x area x 10 um range / 4
        beta_rate = act_g * mt * 1e3
    else:
        alpha_bare = beta_rate = 0.0
    kappa = S.PEND["kappa"]; d = 17e-6
    L = math.pi * 161e9 * d ** 4 / (32 * kappa)                     # tungsten fibre length for the pendulum's kappa
    R_fibre = 5.6e-8 * L / (math.pi * d * d / 4)
    I_beta = beta_rate * E_CHARGE
    extra["charging"] = dict(alpha_escape_bare_per_s=alpha_bare, alpha_current_bare_A=2 * alpha_bare * E_CHARGE,
                             alpha_with_gold_coat=0.0, beta_escape_max_per_s=beta_rate, beta_current_max_A=I_beta,
                             fibre_length_m=L, fibre_resistance_ohm=R_fibre, potential_V=I_beta * R_fibre)
    # beta momentum from the source (upper bound: every 234mPa beta leaves the capsule; the test body stops it)
    n_beta = act_g * s["M"] * 1e3
    T = 0.82; p = math.sqrt(T * T + 2 * T * 0.511) * 1e6 * E_CHARGE / S.c       # mean 234mPa beta, kg m/s

    def F(dist, ab):
        return n_beta / (4 * math.pi * dist * dist) * math.pi * ab * ab * p
    extra["beta_push_source_upper"] = abs(S.radial_push(s, F) / tau1)
    sys_all = math.sqrt(b["s_sys"] ** 2 + extra["beta_push_source_upper"] ** 2)
    return dict(paper_terms=b["terms"], s_sys_paper=b["s_sys"], s_sys_with_beta_upper=sys_all, tau1=tau1,
                tau_lambda={str(l): S.unit_torque(s, l) for l in LAMS}, extra=extra)


def run(args):
    mat, days, lam, p_inj, bias_sd, seed = args
    H.set_test_body(mat)
    s = S.SOURCES[SRC]
    tau = S.unit_torque(s, lam)
    ss = np.random.SeedSequence(seed)
    bias = float(np.random.default_rng(ss.spawn(1)[0]).normal(0, bias_sd)) if bias_sd > 0 else 0.0
    t, th, _ = S.simulate((p_inj + bias) * tau, days, H.DT, ss)
    m, e, _, _ = S.demodulate(t, S.reconstruct_torque(th, H.DT), H.DT)
    return m / tau


def main():
    n_inj = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    n_null = 4 * n_inj
    t0 = time.time()
    res = dict(seed=SEED, n_null=n_null, n_inj=n_inj, budget={m: budget_extra(m) for m in ("Cu", "U")})
    print(f"budgets done [{time.time() - t0:.0f} s]", flush=True)
    root = np.random.SeedSequence(SEED)
    null_seeds = {d: [int(x.generate_state(1)[0]) for x in root.spawn(n_null)] for d in DAYS}
    inj_seeds = {d: [int(x.generate_state(1)[0]) for x in root.spawn(n_inj)] for d in DAYS}
    camp = {}
    with mp.Pool(os.cpu_count()) as pool:
        for days in DAYS:
            # null runs at 0.5 m: the statistical error in torque, the same for both test bodies and every range
            null = np.array(pool.map(run, [("U", days, 0.5, 0.0, 0.0, sd) for sd in null_seeds[days]], chunksize=16))
            s_stat_05 = float(np.std(null, ddof=1))
            for mat in ("Cu", "U"):
                b = res["budget"][mat]
                for lam in LAMS:
                    k = b["tau1"] / b["tau_lambda"][str(lam)]
                    s_stat, s_sys = s_stat_05 * k, b["s_sys_paper"] * k
                    s_tot = math.hypot(s_stat, s_sys)
                    T5 = 5 * s_tot
                    rec = np.array(pool.map(run, [(mat, days, lam, T5, s_sys, sd) for sd in inj_seeds[days]],
                                            chunksize=16))
                    camp[f"{mat} {days} d {lam:g} m"] = dict(
                        test_body=mat, days=days, lam=lam, sigma_stat=s_stat, sigma_sys=s_sys, sigma_tot=s_tot,
                        S=2 * s_tot, T5=T5, recovered_over_injected=float(rec.mean() / T5),
                        bias_se=float((rec.mean() - T5) / (rec.std(ddof=1) / math.sqrt(len(rec)))),
                        sd_ratio=float(rec.std(ddof=1) / s_tot), detected=float(np.mean(rec > T5)))
                    c = camp[f"{mat} {days} d {lam:g} m"]
                    print(f"{mat:2s} {days:3d} d {lam:5g} m | S {c['S']:.3e} | rec/inj {c['recovered_over_injected']:.3f} "
                          f"sd ratio {c['sd_ratio']:.3f} detected {c['detected']:.3f}  [{time.time() - t0:.0f} s]",
                          flush=True)
    res["campaigns"] = camp

    # 3. what the reach means
    meaning = {}
    for lam in LAMS:
        for days in DAYS:
            S_u = camp[f"U {days} d {lam:g} m"]["S"]
            S_cu = camp[f"Cu {days} d {lam:g} m"]["S"]
            fy = 2 * math.pi * S.G * 2400 * lam * math.exp(-0.5 / lam) / 9.80
            qU2_ff = 5e-10 / (1e-6 * fy)
            r = min(0.10, lam)
            qU2_bulk = 1e-3 * 1e11 / (2 * math.pi / 3 * S.G * 19050.0 ** 2 * r * r)
            meaning[f"{days} d {lam:g} m"] = dict(
                S_copper=S_cu, S_uranium=S_u, B_U=b_u(lam), PU_improvement_copper=b_u(lam) / S_cu,
                qU2_limit_uranium_pair=S_u, qU2_free_fall_1ppm=qU2_ff, qU2_bulk=qU2_bulk,
                improvement_over_free_fall=qU2_ff / S_u, improvement_over_bulk=qU2_bulk / S_u,
                q0_below_which_pair_beats_PU_route=b_u(lam) / math.sqrt(S_u),
                q0_bound=math.sqrt(b_u(lam)))
    res["meaning"] = meaning
    json.dump(res, open(OUT / "u238_pair_campaign.json", "w"), indent=1, default=float)
    write_md(res)
    print(f"total {time.time() - t0:.0f} s")


def e(x, d=1):
    m, p = f"{x:.{d}e}".split("e")
    return f"{m}e{int(p)}"


def write_md(res):
    L = []
    a = L.append
    bu, bc = res["budget"]["U"], res["budget"]["Cu"]
    a("# Stage 1 with a depleted-uranium test body")
    a("")
    a(f"{res['n_null']} null and {res['n_inj']} injection runs per cell, common random numbers (seed {res['seed']}).")
    a("")
    a("## Error budget at 0.5 m (anomaly units)")
    a("")
    a("| Term | Copper test body | Uranium test body |")
    a("|---|---|---|")
    for k in bu["paper_terms"]:
        a(f"| {k} | {e(bc['paper_terms'][k])} | {e(bu['paper_terms'][k])} |")
    a(f"| beta momentum from the source, no capsule attenuation (not in the paper) | "
      f"{e(bc['extra']['beta_push_source_upper'])} | {e(bu['extra']['beta_push_source_upper'])} |")
    a(f"| total, paper's terms | {e(bc['s_sys_paper'], 2)} | {e(bu['s_sys_paper'], 2)} |")
    a(f"| total with the beta upper bound | {e(bc['s_sys_with_beta_upper'], 2)} | {e(bu['s_sys_with_beta_upper'], 2)} |")
    a("")
    pv = bu["extra"]["paramagnetic_vs_field"]
    a("Paramagnetic term against the residual field (uranium test body): "
      + ", ".join(f"{k}: {e(v)}" for k, v in pv["terms"].items())
      + f". It stays below 1% of the systematic error for fields under {pv['B0_max_for_1pct_T'] * 1e6:.1f} uT, i.e. a "
      f"shielding factor of {pv['shielding_from_50uT']:.0f} against the Earth's field.")
    a("Test-body remanent moment if its iron sits in 1%-remanent precipitates (the paper's worst case): "
      + ", ".join(f"{k} {e(v)} A m^2" for k, v in bu["extra"]["test_body_moment_A_m2"].items())
      + f". The paper requires below 1e-9 A m^2, which this worst case meets only below "
      f"{bu['extra']['iron_ppm_at_spec']:.2f} ppm. The faked anomaly at each iron level, with the source at its own "
      "requirement: " + ", ".join(f"{k} {e(v)}" for k, v in bu["extra"]["ferro_term_vs_iron"].items())
      + ". The same holds for a copper test body; the requirement is met by measuring the moment (SQUID) and "
      "demagnetizing, not by assuming purity.")
    ch = bu["extra"]["charging"]
    a(f"Charging: a bare uranium surface releases {ch['alpha_escape_bare_per_s']:.0f} alphas per second "
      f"({e(ch['alpha_current_bare_A'])} A), none through a gold coat thicker than ~10 um; the betas of 234mPa, at most "
      f"{e(ch['beta_escape_max_per_s'])} per second ({e(ch['beta_current_max_A'])} A), flow out through the "
      f"{ch['fibre_length_m']:.1f} m tungsten fibre ({ch['fibre_resistance_ohm']:.0f} ohm), raising the test body by "
      f"{e(ch['potential_V'])} V.")
    a("")
    a("## Reach and what it means")
    a("")
    a("| Range | Days | Reach S | Present bound on P_U | Copper: P_U improved x | Uranium pair: q_U^2 limit "
      "| over free fall (1 ppm) | over bulk density | pair beats the P_U route if q0 < |")
    a("|---|---|---|---|---|---|---|---|---|")
    for k, m in res["meaning"].items():
        days, lam = k.split(" d ")
        a(f"| {lam} | {days} | {e(m['S_uranium'], 2)} | {e(m['B_U'])} | {m['PU_improvement_copper']:.2g} | "
          f"{e(m['qU2_limit_uranium_pair'])} | {e(m['improvement_over_free_fall'], 0)} | "
          f"{e(m['improvement_over_bulk'], 0)} | {e(m['q0_below_which_pair_beats_PU_route'])} (bound {e(m['q0_bound'])}) |")
    a("")
    a("Time-domain checks (recovered/injected at the five-sigma threshold, scatter over predicted, detection rate):")
    for k, c in res["campaigns"].items():
        a(f"- {k}: {c['recovered_over_injected']:.3f}, {c['sd_ratio']:.3f}, {c['detected']:.3f}")
    (OUT / "u238_pair_campaign.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
