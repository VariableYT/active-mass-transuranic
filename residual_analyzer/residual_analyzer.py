"""residual_analyzer.py - empirical residual anomaly search for heavy and superheavy nuclei.

    python -I ra_ingest.py            raw published tables -> cache/ (run once, or after adding a table)
    python residual_analyzer.py       first pass: nuclear binding energies and Q_alpha, Z >= 82
    python residual_analyzer.py --validate   only check that the paper's Sec. II D fit is reproduced

Writes out/results.json (everything) and out/REPORT.md (the reading of it).

Passes, each run against every mass model in cache/models.json:
  1. residuals and their fitted noise model (correlated model error + experimental error)
  2. generic search: single nuclides, bumps, steps and kinks in (Z, N), with look-elsewhere correction
  3. cross-model consensus: features that every model shows with the same sign
  4. the paper's window: self-energy of a charge switched on at the island of stability, and a scan of where
     such an island could sit
  5. precision gaps: what each observable would need to see the largest effect the paper still allows
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import ra_atomic
import ra_nuclear as rn
import ra_stats as rs

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = 20261116
WIDTHS = [4.5, 5.0, 7.0, 10.0]                      # the paper's quoted widths
WIDTH_GRID = [round(2.5 + 0.25 * i, 2) for i in range(39)]   # 2.5 .. 12, the paper's scan
B_U = 7.3e-6                                        # uranium bound, baryon-number reading (paper Eq. for B_U)
INTEREST = [("U-238", 92, 146), ("Pu-242", 94, 148), ("Pu-244", 94, 150), ("Cm-248", 96, 152),
            ("Cf-252", 98, 154), ("No-254", 102, 152), ("Fl-289", 114, 175), ("Mc-290", 115, 175),
            ("Lv-293", 116, 177), ("Ts-294", 117, 177), ("Og-294", 118, 176)]


def physical_fit(pr, d, obs, width, Zc=114, Nc=184):
    t = rn.self_energy_template(d["Z"], d["N"], obs, width, Zc, Nc)
    sc = float(np.max(np.abs(t)))
    if sc == 0 or not np.isfinite(sc):
        return None
    b, sb = pr.amplitude(t / sc, d["y"])
    b, sb = b / sc, sb / sc
    qc = math.sqrt(rs.upper_limit_nonneg(b, sb))
    return dict(width=width, beta_hat=b, beta_sd=sb, z=b / sb, qc95=qc, Pc95_baryon=qc * rn.Q0_B,
                Pc95_mass=qc * rn.Q0_M, template_max_keV_per_qc2=sc)


def analyse(nuc, model, me_alpha, obs, zmin, nsim, rng):
    d = rn.dataset(nuc, model, me_alpha, obs, zmin)
    Z, N, y, se = d["Z"], d["N"], d["y"], d["se"]
    X0 = rn.nuisance(Z)
    gp = rs.fit_gp(rs.pairwise_distance(Z, N), se, y, X0,
                   starts=((300, 3, 150), (200, 8, 250), (350, 1.5, 80), (600, 5, 100)))
    pr = rs.Projected(gp.pop("K"), X0)
    she = Z >= 104
    res = dict(n=len(y), n_she=int(she.sum()), Zmax=int(Z.max()), rms_keV=float(np.sqrt(np.mean(y ** 2))),
               rms_she_keV=float(np.sqrt(np.mean(y[she] ** 2))) if she.any() else None,
               median_sigma_exp_keV=float(np.median(se)), noise=gp)

    # --- generic bank
    T, keys = rn.template_bank(Z, N)
    z, ok = pr.z(T, y)
    fam = np.array([k[0] for k in keys])
    groups = dict(all=np.ones(len(keys), bool), point=fam == "point", bump=fam == "bump",
                  change=np.char.startswith(fam, "step") | np.char.startswith(fam, "kink"))
    null = pr.null_max(T, groups, nsim, rng)
    gen = {}
    for g, mask in groups.items():
        m = mask & ok
        j = int(np.flatnonzero(m)[np.argmax(np.abs(z[m]))])
        gen[g] = dict(templates=int(m.sum()), max_abs_z=float(abs(z[j])), z=float(z[j]), key=list(keys[j]),
                      what=rn.describe(keys[j]), p_global=rs.p_global(null[g], abs(z[j])),
                      null_median_max=float(np.median(null[g])), null_95_max=float(np.quantile(null[g], 0.95)))
    res["generic"] = gen
    res["n_local_3sigma"] = int(np.sum(np.abs(z[ok]) >= 3))
    res["n_local_3sigma_expected_null"] = float(ok.sum() * 0.0026997960632601866)
    zmap = {keys[j]: float(z[j]) for j in np.flatnonzero(ok)}

    # --- the paper's window, fixed centre (114, 184)
    res["window"] = [physical_fit(pr, d, obs, w) for w in WIDTHS]
    res["window_scan"] = [physical_fit(pr, d, obs, w) for w in WIDTH_GRID]

    # --- where could an island sit: centre scan, physical sign only
    cols, ckeys = [], []
    for w in (3.0, 5.0, 7.0):
        for zc in range(100, 127):
            for nc in range(150, 197, 2):
                t = rn.self_energy_template(Z, N, obs, w, zc, nc)
                sc = float(np.max(np.abs(t)))
                if sc > 0 and np.isfinite(sc):
                    cols.append(t / sc); ckeys.append((zc, nc, w))
    Tc = np.array(cols).T
    zc_, okc = pr.z(Tc, y)
    nullc = pr.null_max(Tc, dict(all=np.ones(len(ckeys), bool)), nsim, rng, two_sided=False)["all"]
    j = int(np.flatnonzero(okc)[np.argmax(zc_[okc])])
    res["island_scan"] = dict(centres=int(okc.sum()), max_z=float(zc_[j]), Zc=ckeys[j][0], Nc=ckeys[j][1],
                              width=ckeys[j][2], p_global=rs.p_global(nullc, zc_[j]),
                              null_95_max=float(np.quantile(nullc, 0.95)),
                              z_at_114_184_w5=float(zc_[ckeys.index((114, 184, 5.0))])
                              if (114, 184, 5.0) in ckeys else None)
    res["island_map_w5"] = [[zc, nc, float(zc_[i])] for i, (zc, nc, w) in enumerate(ckeys) if w == 5.0 and okc[i]]
    imap = {ckeys[i]: float(zc_[i]) for i in np.flatnonzero(okc)}
    return res, zmap, d, imap


def island_consensus(imaps, models):
    """Island centres scored by the smallest z among the given models (physical sign only)."""
    common = set.intersection(*(set(imaps[m]) for m in models))
    rows = [dict(Zc=k[0], Nc=k[1], width=k[2], min_z=min(imaps[m][k] for m in models),
                 z={m: round(imaps[m][k], 2) for m in models}) for k in common]
    rows.sort(key=lambda r: -r["min_z"])
    return dict(models=list(models), centres=len(rows), n_min_z_ge_3=sum(r["min_z"] >= 3 for r in rows),
                n_min_z_ge_2=sum(r["min_z"] >= 2 for r in rows), top=rows[:10])


def window_all_models(nuc, mods, me_alpha, zmin):
    """The paper's fit (fixed centre, Q_alpha, its own selection Z >= zmin) repeated for every model."""
    out = {}
    for name, model in mods.items():
        d = rn.dataset(nuc, model, me_alpha, "Qa", zmin)
        X0 = rn.nuisance(d["Z"])
        gp = rs.fit_gp(rs.pairwise_distance(d["Z"], d["N"]), d["se"], d["y"], X0)
        pr = rs.Projected(gp.pop("K"), X0)
        out[name] = dict(n=len(d["y"]), Zmax=int(d["Z"].max()), noise=gp,
                         window=[physical_fit(pr, d, "Qa", w) for w in WIDTHS])
    return out


def consensus(zmaps, min_models):
    """Features present in at least min_models models, scored by the smallest |z| if all share a sign, else 0."""
    keys = {}
    for m, zm in zmaps.items():
        for k, v in zm.items():
            keys.setdefault(k, {})[m] = v
    rows = []
    for k, per in keys.items():
        if len(per) < min_models:
            continue
        v = np.array(list(per.values()))
        score = float(np.min(np.abs(v))) if (np.all(v > 0) or np.all(v < 0)) else 0.0
        rows.append(dict(key=list(k), what=rn.describe(k), consensus=score, sign=int(np.sign(v[0])) if score else 0,
                         z={m: round(x, 2) for m, x in per.items()}))
    rows.sort(key=lambda r: -r["consensus"])
    return rows


def residual_correlations(datasets):
    names = list(datasets)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = datasets[a], datasets[b]
            ia = {(z, n): k for k, (z, n) in enumerate(zip(da["Z"], da["N"]))}
            pairs = [(ia[(z, n)], k) for k, (z, n) in enumerate(zip(db["Z"], db["N"])) if (z, n) in ia]
            ja, jb = np.array(pairs).T
            out[f"{a} vs {b}"] = round(float(np.corrcoef(da["y"][ja], db["y"][jb])[0, 1]), 3)
    return out


def precision_gaps(nuc, results):
    """Largest self-energy shift the paper's window still allows in each observable, against what is measured."""
    full = [m for m, r in results["Qa"].items() if r["Zmax"] >= 118]
    env = {w: max(results["Qa"][m]["window_scan"][i]["qc95"] for m in full) for i, w in enumerate(WIDTH_GRID)}
    d2u = (92 - 114) ** 2 + (146 - 184) ** 2
    model_err = float(np.median([results["Qa"][m]["noise"]["amp"] for m in full]))
    rows = []
    for label, Z, N in INTEREST:
        e = nuc.get((Z, N))
        best = dict(B=0.0, Qa=0.0); at = dict(B=None, Qa=None)
        for w in WIDTH_GRID:
            qc = min(env[w], B_U * math.exp(min(d2u / (2 * w * w), 700.0)) / rn.Q0_B)   # Q_alpha ceiling, U bound
            for obs in ("B", "Qa"):
                s = abs(float(rn.self_energy_template(np.array([float(Z)]), np.array([float(N)]), obs, w)[0])) * qc * qc
                if s > best[obs]:
                    best[obs], at[obs] = s, w
        rec = dict(nuclide=label, Z=Z, N=N)
        for obs, val, sig, est in (("B", "ME", "dME", "ME_est"), ("Qa", "Qa", "dQa", "Qa_est")):
            have = e is not None and val in e and not e[est]
            sigma = e[sig] if have else None
            rec[obs] = dict(
                status="measured" if have else "DATA_GAP",
                value_keV=e[val] if have else None, sigma_exp_keV=sigma,
                ame_estimate_keV=(e[val] if (e is not None and val in e and e[est]) else None),
                max_allowed_shift_keV=best[obs], at_width=at[obs],
                shift_over_sigma_exp=(best[obs] / sigma if have and sigma > 0 else None),
                shift_over_model_error=best[obs] / model_err,
                precision_needed_keV=best[obs])
        rows.append(rec)
    return dict(models_used=full, model_error_keV=model_err, qc95_envelope={str(w): env[w] for w in WIDTHS},
                Pc95_envelope_baryon={str(w): env[w] * rn.Q0_B for w in WIDTHS},
                Pc95_envelope_mass={str(w): env[w] * rn.Q0_M for w in WIDTHS}, nuclides=rows)


def validate(nuc, mods, me_alpha):
    ref_path = HERE.parent / "2_indirect_search" / "data_test_robust.json"
    if not ref_path.exists():
        return dict(checked=False, reason="reference file not found")
    ref = json.load(open(ref_path))
    d = rn.dataset(nuc, mods["WS-Ma2019"], me_alpha, "Qa", zmin=84)
    X0 = rn.nuisance(d["Z"])
    gp = rs.fit_gp(rs.pairwise_distance(d["Z"], d["N"]), d["se"], d["y"], X0)
    pr = rs.Projected(gp["K"], X0)
    ratios = {str(w): physical_fit(pr, d, "Qa", w)["qc95"] / ref["limits"]["gp"][str(w)]["qc95"] for w in WIDTHS}
    ok = len(d["y"]) == ref["n"] and all(abs(v - 1) < 1e-4 for v in ratios.values())
    return dict(checked=True, passed=bool(ok), n=len(d["y"]), n_ref=ref["n"], qc95_ratio_to_paper=ratios,
                gp=dict(amp=gp["amp"], length=gp["length"], white=gp["white"]),
                gp_paper=dict(amp=ref["gp"]["amp_keV"], length=ref["gp"]["length_nucleons"],
                              white=ref["gp"]["white_keV"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--zmin", type=int, default=82)
    ap.add_argument("--nsim", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    nuc, mods, me_alpha = rn.load_cache()
    t0 = time.time()
    val = validate(nuc, mods, me_alpha)
    print("reproduces the paper's Sec. II D fit:", val.get("passed"), val.get("qc95_ratio_to_paper"))
    if args.validate:
        return 0 if val.get("passed") else 1
    rng = np.random.default_rng(SEED)
    results, zmaps, data, imaps = {}, {}, {}, {}
    for obs in ("B", "Qa"):
        results[obs], zmaps[obs], data[obs], imaps[obs] = {}, {}, {}, {}
        for name, model in mods.items():
            r, zm, d, im = analyse(nuc, model, me_alpha, obs, args.zmin, args.nsim, rng)
            results[obs][name], zmaps[obs][name], data[obs][name], imaps[obs][name] = r, zm, d, im
            g = r["generic"]["all"]
            print(f"{obs:2s} {name:10s} n={r['n']:3d} rms {r['rms_keV']:6.1f} keV | noise amp {r['noise']['amp']:6.1f} "
                  f"len {r['noise']['length']:4.2f} white {r['noise']['white']:6.1f} | max|z| {g['max_abs_z']:.2f} "
                  f"p_global {g['p_global']:.3f} | island max z {r['island_scan']['max_z']:.2f} "
                  f"p {r['island_scan']['p_global']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    cons = {obs: consensus(zmaps[obs], min_models=4) for obs in results}
    out = dict(
        seed=SEED, zmin=args.zmin, nsim=args.nsim, validation=val,
        sources=json.load(open(HERE / "cache" / "SOURCES.json")),
        coverage={str(k): v for k, v in rn.coverage(nuc, args.zmin).items()},
        results=results,
        residual_correlation_between_models={obs: residual_correlations(data[obs]) for obs in results},
        consensus={obs: dict(features_compared=len(c), n_consensus_ge_3=sum(r["consensus"] >= 3 for r in c),
                             n_consensus_ge_2=sum(r["consensus"] >= 2 for r in c), top=c[:25]) for obs, c in cons.items()},
        precision_gaps=precision_gaps(nuc, results),
        island_consensus=dict(
            Qa=island_consensus(imaps["Qa"], [m for m, r in results["Qa"].items() if r["Zmax"] >= 118]),
            B=island_consensus(imaps["B"], list(results["B"]))),
        window_paper_selection_Z_ge_84=window_all_models(nuc, mods, me_alpha, 84),
    )
    pc_mass = max(out["precision_gaps"]["Pc95_envelope_mass"].values())
    out["atomic"] = dict(Pc_ceiling_mass_like=pc_mass, prefactor_gravity_over_coulomb=ra_atomic.PREFACTOR,
                         best_clock_fraction=ra_atomic.BEST_CLOCK_FRACTION,
                         nuclides=ra_atomic.report(INTEREST + [("island centre (114,184)", 114, 184)], pc_mass))
    json.dump(out, open(OUT / "results.json", "w"), indent=1, default=float)
    np.savez_compressed(OUT / "residuals.npz", **{f"{obs}|{m}|{k}": v for obs in data for m, d in data[obs].items()
                                                  for k, v in d.items()})
    json.dump({obs: {m: [[*k, v] for k, v in zm.items() if k[0] == "point"] for m, zm in zmaps[obs].items()}
               for obs in zmaps}, open(OUT / "point_z.json", "w"))
    print(f"wrote {OUT / 'results.json'}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
