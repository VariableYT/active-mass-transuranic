#!/usr/bin/env python3
"""
detection_map.py
Detection-probability maps for the transuranic active-mass experiment over the
parameter space of the hypothetical force, using the noise and systematics of
active_mass_sim.py (results in out/results.json).

Detection rule (deterministic): a campaign detects if |P_hat| > 5 sigma_tot, where
P_hat ~ Normal(P_true, sigma_tot), sigma_tot^2 = sigma_stat(T)^2 + sigma_sys^2.
Probability = Phi((|P|-5s)/s) + Phi((-|P|-5s)/s). All systematics are torques, so in
P units they scale with range lambda as 1/u(lambda), u = unit torque ratio to lambda=0.5 m.
The analytic rule is validated against full time-domain simulations (seeded).
"""
import json, math, sys, os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.argv = [sys.argv[0], "out"]
import active_mass_sim as S

R = json.load(open("out/results.json"))
OUT = "out"
DAY = 86400.0

# ----------------------------------------------------------------------------- campaigns
CAMP = {
    "238U 100 g, 30 d":   dict(src="238U",  days=30),
    "242Pu 1 g, 90 d":    dict(src="242Pu", days=90),
    "244Pu 1 g, 90 d":    dict(src="244Pu", days=90),
    "248Cm 10 mg, 180 d": dict(src="248Cm", days=180),
}
ZN = {"238U": (92, 146), "242Pu": (94, 148), "244Pu": (94, 150), "248Cm": (96, 152)}
A = {k: z + n for k, (z, n) in ZN.items()}

CAL = {"factor": 1.0}   # multiplies sigma_day; set from full time-domain simulations in calibrate()

def red_inflation(T):
    return CAL["factor"]

def sigma_sys(src):
    b = R["budget"][src]; nm = R["sources"][src]["newton_mc"]["flip_p95_good"] / 2
    terms = [nm,
             1e-5,                                      # ferromagnetic, at the stated spec
             0.1 * b["Fission-neutron momentum"],       # neutron push modelled to 10%
             b["Decay-heat radiation (heater 1%)"],
             b["241Am gamma pressure"] or 0.0]
    if src == "248Cm":
        # weighing improved to 0.1 ug (mass term /10); capsule flip removes in-plane COM terms
        cr = R["sources"][src]["newton_mc"]["components_rms_good"]
        terms[0] = math.hypot(cr["M"] / 10, cr["dim"])
    return math.sqrt(sum(t * t for t in terms))

def sigma_tot(src, days, u=1.0):
    st = R["sources"][src]["sigmaP_1day"] / math.sqrt(days) * red_inflation(days)
    return math.hypot(st, sigma_sys(src)) / u

def p_detect(P, s, k=5.0):
    P = np.abs(P)
    return norm.cdf((P - k * s) / s) + norm.cdf((-P - k * s) / s)

# ----------------------------------------------------------------------------- uranium bound vs lambda
# Paper Sec. II C: 7.3e-6 at 0.5 m, (6.5-11)e-6 for lambda >~ 0.2 m, 2.5e-5 at 10 cm, 1.1e-4 at 5 cm.
LB = np.array([0.05, 0.10, 0.20, 0.50, 1.0, 10.0]); BB = np.array([1.1e-4, 2.5e-5, 1.1e-5, 7.3e-6, 6.5e-6, 6.5e-6])
def B_U(lam):
    lam = np.asarray(lam, float)
    out = np.exp(np.interp(np.log(lam), np.log(LB), np.log(BB)))
    lo = lam < LB[0]
    slope = math.log(BB[1] / BB[0]) / math.log(LB[1] / LB[0])
    out = np.where(lo, BB[0] * (lam / LB[0]) ** slope, out)
    return out

def u_ratio(src, lam):
    s = S.SOURCES[src]
    return S.unit_torque(s, lam) / S.unit_torque(s, 0.5)

def main():
    res = dict(campaigns={})
    for c, d in CAMP.items():
        res["campaigns"][c] = dict(sigma_stat=R["sources"][d["src"]]["sigmaP_1day"] / math.sqrt(d["days"]) * red_inflation(d["days"]),
                                   sigma_sys=sigma_sys(d["src"]), sigma_tot=sigma_tot(d["src"], d["days"]),
                                   threshold_5sigma=5 * sigma_tot(d["src"], d["days"]))
    camps = [c for c in CAMP if CAMP[c]["src"] != "238U"]
    # ------------------------------------------------ Validation against time-domain simulation
    # Runs 60 full 90-day simulations (244Pu) with injected P and a random systematic bias.
    # Step 1 calibrates sigma_stat from their scatter; step 2 compares detection rates.
    src = "244Pu"; days = 90; tau1 = R["sources"][src]["unit_torque"]
    s_prov = sigma_tot(src, days)
    seeds = np.random.SeedSequence(S.SEED + 7).spawn(60)
    groups = []; j = 0
    for mult in (4.0, 5.0, 6.0):
        P = mult * s_prov; recs = []
        for i in range(20):
            bias = np.random.default_rng(seeds[j]).normal(0, sigma_sys(src))
            t, th, _ = S.simulate((P + bias) * tau1, days, 10.0, seeds[j]); j += 1
            m, e, _, _ = S.demodulate(t, S.reconstruct_torque(th, 10.0), 10.0)
            recs.append(m / tau1)
        groups.append((P, np.array(recs)))
    resid = np.concatenate([g[1] - g[0] for g in groups])
    sd_tot = float(np.std(resid, ddof=1))
    sd_stat = math.sqrt(max(sd_tot ** 2 - sigma_sys(src) ** 2, 1e-30))
    CAL["factor"] = sd_stat / (R["sources"][src]["sigmaP_1day"] / math.sqrt(days))
    res["calibration"] = dict(n_runs=len(resid), sd_total_90d=sd_tot, sd_stat_90d=sd_stat,
                              factor_vs_120d_estimate=CAL["factor"],
                              factor_se=CAL["factor"] / math.sqrt(2 * (len(resid) - 1)))
    st = sigma_tot(src, days)
    val = []
    # detection test at a FIXED threshold of 5 * calibrated sigma
    for P, recs in groups:
        hits = int(np.sum(np.abs(recs) > 5 * st)); n = len(recs)
        val.append(dict(P_over_sigma=P / st, P=P, predicted=float(p_detect(P, st)), simulated=hits / n, n=n,
                        binomial_se=math.sqrt(max(hits / n * (1 - hits / n), 1 / n) / n)))
    res["validation"] = val
    # recompute campaign table and all maps with the calibrated sigma
    for c, d in CAMP.items():
        res["campaigns"][c] = dict(sigma_stat=R["sources"][d["src"]]["sigmaP_1day"] / math.sqrt(d["days"]) * red_inflation(d["days"]),
                                   sigma_sys=sigma_sys(d["src"]), sigma_tot=sigma_tot(d["src"], d["days"]),
                                   threshold_5sigma=5 * sigma_tot(d["src"], d["days"]))


    # ------------------------------------------------ Map 1: Gaussian, lambda = 0.5 m
    BU = 7.3e-6
    sig = np.linspace(2.5, 12, 381); lpc = np.linspace(-6, 26, 641)
    SG, LP = np.meshgrid(sig, lpc)
    d2U = (92 - 114) ** 2 + (146 - 184) ** 2
    PU = 10 ** LP * np.exp(-d2U / (2 * SG ** 2))
    allowed = PU <= BU
    pmaps = {}
    for c in CAMP:
        src = CAMP[c]["src"]; z, n = ZN[src]; d2 = (z - 114) ** 2 + (n - 184) ** 2
        PX = 10 ** LP * np.exp(-d2 / (2 * SG ** 2))
        pmaps[c] = p_detect(PX, sigma_tot(src, CAMP[c]["days"]))
    p_any = 1 - np.prod([1 - pmaps[c] for c in camps], axis=0)
    p_any = np.where(allowed, p_any, np.nan)
    # coverage in the allowed region (measure: uniform in sigma and log10 Pc within the box,
    # restricted to Pc >= 1, i.e. at least gravity strength at the island centre)
    box = allowed & (LP >= 0)
    cov = {c: float(np.mean(pmaps[c][box] > 0.5)) for c in CAMP}
    cov["any transuranic"] = float(np.mean(p_any[box] > 0.5))
    # how far the 50% contour sits below sigma_max, per Pc
    gap = {}
    for lp_ in (0, 2, 4, 6, 8, 10, 12, 16, 20, 25):
        i = np.argmin(abs(lpc - lp_)); row = p_any[i]; ok = np.where(row > 0.5)[0]
        smax = math.sqrt(d2U / (2 * math.log(10 ** lp_ / BU))) if 10 ** lp_ > BU else None
        gap[lp_] = dict(sigma_max=smax, sigma_min_detected=float(sig[ok].min()) if len(ok) else None)
    res["gaussian"] = dict(coverage_fraction_Pc_ge_1=cov, sigma_window=gap)

    # ------------------------------------------------ Map 2: power law, P_X = P_U (A/238)^k, P_U <= B_U
    ks = np.linspace(0, 400, 401); lpu = np.linspace(-12, math.log10(BU), 301)
    K, LPU = np.meshgrid(ks, lpu)
    pl = {}
    for c in CAMP:
        src = CAMP[c]["src"]; PX = 10 ** LPU * (A[src] / 238) ** K
        pl[c] = p_detect(PX, sigma_tot(src, CAMP[c]["days"]))
    pl_any = 1 - np.prod([1 - pl[c] for c in camps], axis=0)
    # minimum k detected at P_U = B_U
    kmin = {c: float(ks[np.argmax(pl[c][-1] > 0.5)]) if np.any(pl[c][-1] > 0.5) else None for c in CAMP}
    res["power_law"] = dict(k_min_detect_at_PU_eq_BU=kmin,
                            k_paper_Pc1e4=93.6, k_paper_Pc1e10=155.0)

    # ------------------------------------------------ Map 3: range dependence, widest Gaussian
    lams = np.logspace(-2.3, 1, 60)
    U = {src: np.array([u_ratio(src, l) for l in lams]) for src in ("242Pu", "244Pu", "248Cm")}
    lpc3 = np.linspace(-2, 26, 300)
    LL, LP3 = np.meshgrid(lams, lpc3)
    BUl = B_U(LL)
    with np.errstate(invalid="ignore", divide="ignore"):
        sw2 = d2U / (2 * np.log(10 ** LP3 / BUl))          # sigma_max^2 at each (lambda, Pc)
    lam_maps = {}
    for c in camps:
        src = CAMP[c]["src"]; z, n = ZN[src]; d2 = (z - 114) ** 2 + (n - 184) ** 2
        w = d2 / d2U
        PX = (10 ** LP3) ** (1 - w) * BUl ** w
        UU = np.interp(np.log(LL), np.log(lams), U[src])
        lam_maps[c] = np.where(10 ** LP3 > BUl, p_detect(PX, sigma_tot(src, CAMP[c]["days"], 1.0) / np.maximum(UU, 1e-30)), 0)
    lam_any = 1 - np.prod([1 - lam_maps[c] for c in camps], axis=0)
    # threshold Pc vs lambda for 50% detection
    thr = []
    for j, l in enumerate(lams):
        col = lam_any[:, j]; ok = np.where(col > 0.5)[0]
        thr.append(float(10 ** lpc3[ok.min()]) if len(ok) else None)
    res["range"] = dict(lam=lams.tolist(), Pc_threshold_50pct=thr, B_U=B_U(lams).tolist())

    # ------------------------------------------------ Figures
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    fig, ax = plt.subplots(figsize=(7.2, 5))
    im = ax.pcolormesh(SG, LP, p_any, cmap="viridis", vmin=0, vmax=1, shading="auto")
    ax.contourf(SG, LP, ~allowed, levels=[0.5, 1.5], colors=["#cfcfcf"])
    cols = {"242Pu 1 g, 90 d": "#f2b8a0", "244Pu 1 g, 90 d": "#ff6a3d", "248Cm 10 mg, 180 d": "#7fc8ff"}
    for c in camps:
        cs = ax.contour(SG, LP, np.where(allowed, pmaps[c], np.nan), levels=[0.5], colors=[cols[c]], linewidths=1.2)
        ax.plot([], [], color=cols[c], label=f"{c}: 50% contour")
    ax.text(9, 20, "excluded by existing\nuranium bound", ha="center", fontsize=8)
    ax.text(4.0, 2, "allowed but\nundetectable", ha="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(detect at 5σ), any transuranic stage")
    ax.set_xlabel("switch width σ (nucleons)"); ax.set_ylabel("log₁₀ P_c (strength at island centre)")
    ax.set_title("Gaussian switch, λ = 0.5 m: where the experiment detects the force", fontsize=9)
    ax.set_xlim(2.5, 12); ax.set_ylim(-6, 26); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/map_gaussian.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.pcolormesh(K, LPU, pl_any, cmap="viridis", vmin=0, vmax=1, shading="auto")
    for c in camps:
        ax.contour(K, LPU, pl[c], levels=[0.5], colors=[cols[c]], linewidths=1.2); ax.plot([], [], color=cols[c], label=c)
    for kk, lab in ((93.6, "paper k, Pc=1e4"), (155, "paper k, Pc=1e10")):
        ax.axvline(kk, color="w", ls=":", lw=.8); ax.text(kk + 3, -11.5, lab, color="w", fontsize=7, rotation=90)
    fig.colorbar(im, ax=ax, label="P(detect at 5σ), any transuranic stage")
    ax.set_xlabel("power-law steepness k  (P_X = P_U (A/238)^k)"); ax.set_ylabel("log₁₀ P_U (uranium's own anomaly, ≤ bound)")
    ax.set_title("Power-law switch, λ = 0.5 m", fontsize=9); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/map_powerlaw.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im = ax[0].pcolormesh(LL, LP3, lam_any, cmap="viridis", vmin=0, vmax=1, shading="auto")
    ax[0].set_xscale("log"); ax[0].axvspan(lams[0], 0.05, color="w", alpha=.15)
    ax[0].text(0.012, 24, "uranium bound\nextrapolated", color="w", fontsize=7)
    fig.colorbar(im, ax=ax[0], label="P(detect), widest allowed Gaussian")
    ax[0].set_xlabel("range λ (m)"); ax[0].set_ylabel("log₁₀ P_c")
    ax[0].set_title("Widest Gaussian vs range", fontsize=9)
    for src, c in (("242Pu", "#f2b8a0"), ("244Pu", "#ff6a3d"), ("248Cm", "#7fc8ff")):
        ax[1].loglog(lams, U[src], color=c, label=src)
    ax[1].set_xlabel("range λ (m)"); ax[1].set_ylabel("signal relative to λ = 0.5 m"); ax[1].legend(fontsize=7)
    ax[1].set_title("Signal suppression at short range (finite geometry)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/map_range.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    xs = np.linspace(2, 8, 200); ax.plot(xs, p_detect(xs, 1.0), color="k", label="analytic rule")
    for v in val:
        ax.errorbar(v["P_over_sigma"], v["simulated"], v["binomial_se"], fmt="o", color="#c4532c")
    ax.plot([], [], "o", color="#c4532c", label="full time-domain sims (20 × 90 d each)")
    ax.set_xlabel("true P / σ_tot (calibrated)  (244Pu, 90 d)"); ax.set_ylabel("detection probability"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/map_validation.png"); plt.close(fig)

    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
        return o
    json.dump(clean(res), open(f"{OUT}/detection_results.json", "w"), indent=1)
    print(json.dumps(clean({k: res[k] for k in ("calibration", "campaigns", "gaussian", "power_law", "validation")}), indent=1))
    print("range thresholds:", [(f"{l:.3g}", (f"{t:.2g}" if t else None)) for l, t in zip(lams[::6], thr[::6])])

if __name__ == "__main__":
    main()
