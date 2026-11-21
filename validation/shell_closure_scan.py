"""Could a charge that follows shell closure reach the island and be seen with plutonium? (paper Sec. III D, rev 5n)

Charge model: q_nuc(X) = c f(X)^k for a shell measure f >= 0; a source's pull on ordinary matter is P_X = q_nuc(X) q0
(paper Eq. P). Existing limits, both from the uranium-attractor test of Smith et al. (PRD 61, 022001 (2000)), where
copper and lead test bodies fell toward the attractor at the same rate:
  - uranium, as the attractor:  |P_U| <= B_U(lambda)  (paper Eq. BU; seen through the test bodies' difference in q0)
  - lead, as a test body:  |P_Pb - P_Cu| <= B_Pb(lambda) = Delta(B/mu) B_U(lambda), Delta(B/mu) = 1.0e-3: a nuclear
    charge on lead is compared directly with copper's (natural isotopic abundances), barring a cancellation between q0
    and uranium's own nuclear charge.
Both fix the product c q0, so the predictions do not depend on q0 itself; q0 at its bound, sqrt(B_U), enters only the
self-energy check (paper Eq. self, excluded above 1 MeV in any AME2020 measured mass or Q_alpha).

For each shell measure, mass model and island centre (Z = 114, 120 or 126, N = 184), for a strength P_c at the
centre between 1e4 and 1e10 (the paper's range) and for ranges of 0.5 m down to 2 cm, the script finds the smallest
power k that reaches P_c within both limits (the best case for the sources), and the signals of 1 g of Pu-239, Pu-242,
Pu-244 and 10 mg of Cm-248 against the reach of paper Table III (divided at shorter range by the Yukawa factor at
each source's distance, 2 cm for plutonium and 1 cm for curium, as in paper Sec. V A). If the centre's value of the measure is not above lead's, no k reaches P_c.

Shell measures:
  shell binding     the stabilising part of the ground-state shell correction, max(-E_sh, 0): WS4 (E_sh), KTUY05
                    (E_sh), FRDM2012 (E_s+p, shell plus pairing)
  shell share       shell binding / (shell binding + liquid-drop fission barrier); and the ratio shell binding /
                    liquid-drop barrier (floor 0.1 MeV). The barrier is Myers and Swiatecki's, Nucl. Phys. 81, 1
                    (1966): B = E_s0 F(x), E_s0 = 17.9439 (1 - 1.7826 I^2) A^(2/3) MeV,
                    x = (Z^2/A) / [50.883 (1 - 1.7826 I^2)], F = 0.38 (0.75 - x) for x <= 2/3, 0.83 (1 - x)^3 to x = 1
  shell gaps        delta_2n, delta_2p and their sum from each model's masses (WS-Ma2019, WS4, FRDM2012, KTUY05;
                    HFB-24 stops at Z = 110 and cannot reach the island)
  valence product   1 / (1 + N_p N_n) (valence nucleons from the nearest closed shells; proton closure 114, 120 or 126)

Inputs (fetched, not redistributed; see README): residual_analyzer/cache/models.json and experiment.json (built by
residual_analyzer/ra_ingest.py), the WS4, KTUY05 and FRDM2012 tables in residual_analyzer/raw/, and
paper/numbers.json for the reach.

Run:  python validation/shell_closure_scan.py     (from the top folder; about 30 s)
Writes validation/out/shell_closure_scan.json and validation/out/shell_closure_scan.md.
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
RA = ROOT / "residual_analyzer"

G, U_KG, FM, MEV = 6.67430e-11, 1.66053906660e-27, 1e-15, 1.602176634e-13
ME_H, ME_N = 7288.971064, 8071.31806                          # keV
PASSIVE = 5.7e-13 / 9.2e-5                                     # Smith et al.: 95% limit / attractor's pull
SIGMA = {0.5: 8.44e-4, 0.1: 2.47e-4, 0.05: 5.45e-5, 0.02: 1.53e-6}   # their Table X, Sigma(lambda)
DBMU = 1.0e-3

TARGETS = (1e4, 1e6, 1e8, 1e10)
THR_MEV = 1.0
NUMS = json.loads((ROOT / "paper" / "numbers.json").read_text())["nuclides"]
SOURCES = {"Pu-239": (94, 145), "Pu-242": (94, 148), "Pu-244": (94, 150), "Cm-248": (96, 152)}
REACH = {s: NUMS[s]["S"] for s in SOURCES}
PB = {(82, 122): 0.014, (82, 124): 0.241, (82, 125): 0.221, (82, 126): 0.524}
CU = {(29, 34): 0.6915, (29, 36): 0.3085}
U238 = (92, 146)
CENTRES = {114: (114, 184), 120: (120, 184), 126: (126, 184)}

EXP = json.loads((RA / "cache" / "experiment.json").read_text())["nuclides"]
MASS = {m: {tuple(map(int, k.split(","))): v for k, v in d.items()}
        for m, d in json.loads((RA / "cache" / "models.json").read_text()).items()}


def rows(path, ncol):
    for line in path.read_text(encoding="latin-1").splitlines():
        f = line.split()
        if len(f) == ncol:
            try:
                yield [float(x) for x in f]
            except ValueError:
                pass


ESH = {"WS4": {}, "KTUY05": {}, "FRDM2012": {}}
for A, Z, b2, b4, b6, esh, *_ in rows(RA / "raw" / "ws4" / "WS4.txt", 11):
    ESH["WS4"][(int(Z), int(A - Z))] = esh
for Z, N, mcal, esh, *_ in rows(RA / "raw" / "ktuy05" / "KTUY05_m246.dat", 7):
    ESH["KTUY05"][(int(Z), int(N))] = esh
# FRDM2012 (arXiv:1508.06294v1 as text): N A eps2 eps3 eps4 eps6 b2 b3 b4 b6 Es+p Emic Ebind Mth ...; the same
# repair and row check as residual_analyzer/ra_ingest.read_frdm2012
for line in (RA / "raw" / "frdm2012" / "frdm_text.txt").read_text(encoding="utf-8").splitlines():
    toks, f = line.replace("−", "-").split(), []
    for t in toks:
        if f and re.fullmatch(r"-?\d+", f[-1]) and t.startswith("."):
            f[-1] += t
        else:
            f.append(t)
    if len(f) not in (16, 18) or not (re.fullmatch(r"\d+", f[0]) and re.fullmatch(r"\d+", f[1])):
        continue
    try:
        N, A = int(f[0]), int(f[1]); esp, ebind, mth = float(f[10]), float(f[12]), float(f[13])
    except ValueError:
        continue
    Z = A - N
    if Z < 8 or abs(mth - (Z * 7.289 + N * 8.0713 - ebind)) > 0.03 + 2e-5 * A or (Z, N) in ESH["FRDM2012"]:
        continue
    ESH["FRDM2012"][(Z, N)] = esp


def bind(m, Z, N):
    me = MASS[m].get((Z, N))
    return None if me is None else (Z * ME_H + N * ME_N - me) / 1e3


def d2n(m, Z, N):
    b = [bind(m, Z, N + d) for d in (-2, 0, 2)]
    return None if None in b else max(2 * b[1] - b[0] - b[2], 0.0)


def d2p(m, Z, N):
    b = [bind(m, Z + d, N) for d in (-2, 0, 2)]
    return None if None in b else max(2 * b[1] - b[0] - b[2], 0.0)


def gaps(m, Z, N):
    a, b = d2n(m, Z, N), d2p(m, Z, N)
    return None if a is None or b is None else a + b


def b_ldm(Z, N):
    A = Z + N
    I = (N - Z) / A
    w = 1 - 1.7826 * I * I
    x = Z * Z / A / (50.883 * w)
    F = 0.38 * (0.75 - x) if x <= 2 / 3 else (0.83 * (1 - x) ** 3 if x < 1 else 0.0)
    return max(F, 0.0) * 17.9439 * w * A ** (2 / 3)


def shell_bind(m, Z, N):
    e = ESH[m].get((Z, N))
    return None if e is None else max(-e, 0.0)


def shell_share(m, Z, N):
    s = shell_bind(m, Z, N)
    return None if s is None else (s / (s + b_ldm(Z, N)) if s > 0 else 0.0)


def shell_ratio(m, Z, N):
    s = shell_bind(m, Z, N)
    return None if s is None else s / max(b_ldm(Z, N), 0.1)


MAGIC_N = (2, 8, 20, 28, 50, 82, 126, 184, 258)


def valence(n, magic):
    lo = max(x for x in magic if x <= n)
    hi = min(x for x in magic if x > n) if n < magic[-1] else lo
    return min(n - lo, hi - n)


def valence_prox(zc):
    mz = (2, 8, 20, 28, 50, 82, zc, 184)
    return lambda Z, N: 1.0 / (1.0 + valence(Z, mz) * valence(N, MAGIC_N))


CAND = []
for m in ("WS4", "KTUY05", "FRDM2012"):
    CAND += [("shell binding", m, None, (lambda m: lambda Z, N: shell_bind(m, Z, N))(m)),
             ("shell share", m, None, (lambda m: lambda Z, N: shell_share(m, Z, N))(m)),
             ("shell / liquid-drop barrier", m, None, (lambda m: lambda Z, N: shell_ratio(m, Z, N))(m))]
for m in ("WS-Ma2019", "WS4", "FRDM2012", "KTUY05", "HFB-24"):
    CAND += [("gap delta_2n", m, None, (lambda m: lambda Z, N: d2n(m, Z, N))(m)),
             ("gap delta_2p", m, None, (lambda m: lambda Z, N: d2p(m, Z, N))(m)),
             ("gaps delta_2n + delta_2p", m, None, (lambda m: lambda Z, N: gaps(m, Z, N))(m))]
for zc in CENTRES:
    CAND.append(("valence product", f"closure Z={zc}", zc, valence_prox(zc)))


def lse(terms):
    terms = [t for t in terms if t > -math.inf]
    if not terms:
        return -math.inf
    mx = max(terms)
    return mx + math.log(sum(math.exp(t - mx) for t in terms))


def mean_log(abund, fn, k):
    terms = []
    for zn, a in abund.items():
        v = fn(*zn)
        if v is None:
            return None
        terms.append(math.log(a) + (k * math.log(v) if v > 0 else -math.inf))
    return lse(terms)


def log_cq0(fn, k, b_u, b_pb):
    fu = fn(*U238)
    lp, lc = mean_log(PB, fn, k), mean_log(CU, fn, k)
    if fu is None or lp is None or lc is None:
        return None, None
    lim_u = math.log(b_u) - k * math.log(fu) if fu > 0 else math.inf
    hi, lo = max(lp, lc), min(lp, lc)
    diff = hi + math.log1p(-math.exp(lo - hi)) if hi > -math.inf and lo < hi else -math.inf
    lim_pb = math.log(b_pb) - diff if diff > -math.inf else math.inf
    return min(lim_u, lim_pb), ("uranium" if lim_u <= lim_pb else "lead")


K_GRID = [10 ** (e / 400) for e in range(-1200, 1401)]


def worst_self_energy(fn, k, lcq, q0):
    worst = -math.inf

    def lE(Z, N):
        v = fn(Z, N)
        if v is None or v <= 0:
            return -math.inf
        A = Z + N
        lq = lcq + k * math.log(v) - math.log(q0)
        return math.log(0.6 * G / (1.2 * A ** (1 / 3) * FM) / MEV) + 2 * (lq + math.log(A * U_KG))

    for e in EXP.values():
        Z, N = e["Z"], e["N"]
        if Z < 3 or N < 3:
            continue
        if not e.get("ME_est"):
            worst = max(worst, lE(Z, N))
        if e.get("Qa") is not None and not e.get("Qa_est"):
            a, b = lE(Z, N), lE(Z - 2, N - 2)
            if max(a, b) > -math.inf:
                hi, lo = max(a, b), min(a, b)
                worst = max(worst, hi + (math.log1p(-math.exp(lo - hi)) if lo < hi else -math.inf))
    return math.exp(min(worst, 700)) if worst > -math.inf else 0.0


def analyse(fn, centre, lam):
    b_u = PASSIVE / SIGMA[lam]
    b_pb = b_u * DBMU
    yuk = {s: (1 + NUMS[s]["r"] / lam) * math.exp(-NUMS[s]["r"] / lam) for s in SOURCES}   # each source's distance
    fi = fn(*centre)
    if fi is None or fi <= 0 or fn(*U238) is None or mean_log(PB, fn, 1.0) is None or mean_log(CU, fn, 1.0) is None:
        return None
    out = {}
    for pc in TARGETS:
        hit = None
        for k in K_GRID:
            lcq, by = log_cq0(fn, k, b_u, b_pb)
            if lcq + k * math.log(fi) >= math.log(pc):
                hit = (k, lcq, by)
                break
        if hit is None:
            out[f"{pc:.0e}"] = dict(reached=False)
            continue
        k, lcq, by = hit
        sig = {}
        for s, zn in SOURCES.items():
            v = fn(*zn)
            sig[s] = None if v is None else (0.0 if v <= 0 else math.exp(min(lcq + k * math.log(v), 700)) * yuk[s] / REACH[s])
        se = worst_self_energy(fn, k, lcq, math.sqrt(b_u))
        out[f"{pc:.0e}"] = dict(reached=True, k=k, bound_by=by, signal_over_reach=sig, self_energy_MeV=se,
                                excluded=se > THR_MEV)
    return out


RES = []
for fam, model, zc_only, fn in CAND:
    for zc, centre in CENTRES.items():
        if zc_only is not None and zc != zc_only:
            continue
        row = dict(family=fam, model=model, centre=zc, values=dict(
            island=fn(*centre), uranium=fn(*U238),
            lead=None if mean_log(PB, fn, 1.0) is None else math.exp(mean_log(PB, fn, 1.0)),
            copper=None if mean_log(CU, fn, 1.0) is None else math.exp(mean_log(CU, fn, 1.0)),
            **{s: fn(*zn) for s, zn in SOURCES.items()}), ranges={})
        for lam in SIGMA:
            row["ranges"][str(lam)] = analyse(fn, centre, lam)
        RES.append(row)

# summary: per range, over every row and every target in the paper's range, the largest signal / reach among couplings
# the masses allow (all k, and k <= 20), and the smallest k reaching 1e4 per family
SUM = dict(models_reaching_island=sorted({r["model"] for r in RES if r["values"]["island"]}), by_range={},
           blocked={}, min_k={})
for lam in SIGMA:
    best, best20 = None, None
    for r in RES:
        a = r["ranges"][str(lam)]
        if not a:
            continue
        for tn, t in a.items():
            if not t["reached"] or t["excluded"]:
                continue
            for s, v in t["signal_over_reach"].items():
                if v is None:
                    continue
                cand = (v, s, r["family"], r["model"], r["centre"], tn, t["k"])
                if best is None or v > best[0]:
                    best = cand
                if t["k"] <= 20 and (best20 is None or v > best20[0]):
                    best20 = cand
    per = {}
    for src in SOURCES:
        vals = [(t["signal_over_reach"][src], t["k"]) for r in RES if r["ranges"][str(lam)]
                for t in r["ranges"][str(lam)].values()
                if t["reached"] and not t["excluded"] and t["signal_over_reach"].get(src) is not None]
        per[src] = dict(all=max((v for v, k in vals), default=None),
                        k_le_20=max((v for v, k in vals if k <= 20), default=None))
    SUM["by_range"][str(lam)] = dict(B_U=PASSIVE / SIGMA[lam], B_Pb=PASSIVE / SIGMA[lam] * DBMU, best=best,
                                     best_k_le_20=best20, per_source=per)
for fam in dict.fromkeys(r["family"] for r in RES):
    rr = [r for r in RES if r["family"] == fam and r["ranges"]["0.5"]]
    SUM["blocked"][fam] = dict(rows=len(rr), blocked_at_1e4=sum(1 for r in rr if not r["ranges"]["0.5"]["1e+04"]["reached"]))
    ks = [r["ranges"]["0.5"]["1e+04"]["k"] for r in rr if r["ranges"]["0.5"]["1e+04"]["reached"]]
    SUM["min_k"][fam] = min(ks) if ks else None

(OUT / "shell_closure_scan.json").write_text(json.dumps(dict(summary=SUM, reach=REACH, results=RES), indent=1,
                                                        default=str))


def g(x):
    return "-" if x is None else f"{x:.2g}"


L = ["# Shell-closure scan (paper Sec. III D)", "",
     "| family | rows with an island value | blocked by lead at 0.5 m (P_c = 1e4) | smallest k reaching 1e4 |",
     "|---|---|---|---|"]
for fam, b in SUM["blocked"].items():
    L.append(f"| {fam} | {b['rows']} | {b['blocked_at_1e4']} | {g(SUM['min_k'][fam])} |")
L += ["", "Largest signal / reach over every row and every P_c from 1e4 to 1e10, couplings the masses allow:", "",
      "| range | uranium limit | lead limit | best | where | best with k <= 20 | where |", "|---|---|---|---|---|---|---|"]
for lam, x in SUM["by_range"].items():
    cells = []
    for b in (x["best"], x["best_k_le_20"]):
        cells += [g(b[0]) if b else "-", f"{b[1]}, {b[2]}, {b[3]}, centre {b[4]}, P_c {b[5]}, k {b[6]:.3g}" if b else "none"]
    L.append(f"| {float(lam) * 100:g} cm | {x['B_U']:.2g} | {x['B_Pb']:.2g} | " + " | ".join(cells) + " |")
L += ["", "Per source, largest signal / reach (all k; k <= 20):", "", "| range | " + " | ".join(SOURCES) + " |",
      "|---" * (len(SOURCES) + 1) + "|"]
for lam, x in SUM["by_range"].items():
    L.append(f"| {float(lam) * 100:g} cm | " + " | ".join(
        f"{g(x['per_source'][s]['all'])}; {g(x['per_source'][s]['k_le_20'])}" for s in SOURCES) + " |")
# every case that reaches its source's reach
L += ["", "Cases at or above a source's reach (couplings the masses allow):", ""]
for r in RES:
    for lam, a in r["ranges"].items():
        for tn, t in (a or {}).items():
            if t["reached"] and not t["excluded"]:
                for s, v in t["signal_over_reach"].items():
                    if v is not None and v >= 1:
                        L.append(f"- {float(lam) * 100:g} cm, {r['family']}, {r['model']}, centre {r['centre']}, "
                                 f"P_c {tn}, k {t['k']:.3g}: {s} {v:.2g} x reach")
(OUT / "shell_closure_scan.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
