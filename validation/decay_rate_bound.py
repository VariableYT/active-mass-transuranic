"""How steeply can a nuclear charge follow the alpha-decay rate, or the fission rate? (paper Sec. III C-D, rev 5m, 5o)

Benchmark:  |P_X| = B_U (lambda_alpha,X / lambda_alpha,U-238)^k,  lambda_alpha the partial alpha-decay rate.
With ordinary matter's charge at its largest allowed value, q0 = Q0_B = sqrt(B_U) (the baryon reading of Sec. II C),
the smallest nuclear charge that gives this pull is q_nuc = |P_X| / Q0_B, and Eq. (self) gives every nucleus a scalar
self-energy  E_self = (3/5) G (q_nuc M)^2 / R,  M = A u, R = 1.2 A^(1/3) fm. It grows as lambda^(2k), so the fastest
alpha emitters whose masses are measured bound k.

For every nuclide with a measured mass in AME2020 (not estimated) and a measured half-life and alpha branch in
NUBASE2020, the script asks at which k the self-energy, or its difference between parent and daughter for a measured
Q_alpha, first reaches a threshold, for thresholds of 0.1, 1 and 10 MeV, and reports the binding nuclide. A second
pass sets aside nuclides (parents or daughters) that live less than a nanosecond.

Inputs (not redistributed; see README):
  residual_analyzer/cache/experiment.json   AME2020 masses and Q_alpha, built by residual_analyzer/residual_analyzer.py
  nubase_4.mas20.txt                        NUBASE2020, https://www-nds.iaea.org/amdc/ (Kondev et al., Chin. Phys. C
                                            45, 030001 (2021)), in residual_analyzer/raw/nubase2020/ or
                                            deep_audit/raw/nubase2020/

Run:  python validation/decay_rate_bound.py     (from the top folder)
Writes validation/out/decay_rate_bound.json.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

G, U_KG, FM, MEV, YEAR = 6.67430e-11, 1.66053906660e-27, 1e-15, 1.602176634e-13, 3.15576e7
B_U = (5.7e-13 / 9.2e-5) / 8.44e-4          # Eq. (BU): Smith et al. at 0.5 m, Sigma = 8.44e-4
Q0_B = math.sqrt(B_U)
T_FAST = 1e-9
THRESHOLDS = (0.1, 1.0, 10.0)                 # MeV

EXP = json.loads((ROOT / "residual_analyzer" / "cache" / "experiment.json").read_text())["nuclides"]
NUB_PATH = next(p for p in (ROOT / "residual_analyzer" / "raw" / "nubase2020" / "nubase_4.mas20.txt",
                            ROOT / "deep_audit" / "raw" / "nubase2020" / "nubase_4.mas20.txt") if p.exists())

T_UNIT = {"ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3,
          "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
for _p, _f in (("", 1), ("k", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12), ("P", 1e15), ("E", 1e18), ("Z", 1e21),
               ("Y", 1e24)):
    T_UNIT[_p + "y"] = _f * YEAR
W_UNIT = {"eV": 1.0, "keV": 1e3, "MeV": 1e6}
HBAR_EV_S = 6.582119569e-16


def half_life(ts, unit):
    """Seconds; inf for stable; None for estimates ('#'), limits and 'p-unst'."""
    ts, u = ts.strip(), unit.strip()
    if ts.startswith("stbl"):
        return math.inf
    if not ts or ts.startswith("p-unst") or ts[0] in "<>" or "#" in ts:
        return None
    try:
        v = float(ts.lstrip("~"))
    except ValueError:
        return None
    if u in T_UNIT:
        return v * T_UNIT[u]
    if u in W_UNIT and v > 0:                                   # a width: T1/2 = hbar ln2 / Gamma
        return HBAR_EV_S * math.log(2) / (v * W_UNIT[u])
    return None


def alpha_branch(s):
    """Fraction of decays by alpha emission; 0 if none listed; None if listed without a value."""
    for item in s.split(";"):
        item = item.strip()
        if item.startswith("A") and (len(item) == 1 or not item[1].isalpha()):
            rest = item[1:].strip().lstrip("=~<>").strip().split()
            try:
                return float(rest[0]) / 100.0 if rest and rest[0] != "?" else None
            except ValueError:
                return None
    return 0.0


NUB = {}
for line in NUB_PATH.read_text(encoding="latin-1").splitlines():
    if line.startswith("#") or len(line) < 80:
        continue
    try:
        A, Z, i = int(line[0:3]), int(line[4:7]), int(line[7])
    except ValueError:
        continue
    if i:
        continue
    t = half_life(line[69:78], line[78:80])
    if t is not None:
        NUB[(Z, A - Z)] = (t, 0.0 if math.isinf(t) else alpha_branch(line[119:209]))

T_U, BR_U = NUB[(92, 146)]
LAM_U = math.log(2) / T_U * BR_U


def log_eself0(Z, N):
    """(log10 of E_self at k = 0 in MeV, log10 of lambda_alpha / lambda_alpha,U), or None if it carries no charge."""
    t, br = NUB.get((Z, N), (None, None))
    if t is None or br is None or math.isinf(t) or br == 0:
        return None
    lam = math.log(2) / t * br
    A = Z + N
    q = B_U / Q0_B
    le0 = math.log10(0.6 * G * (q * A * U_KG) ** 2 / (1.2 * A ** (1 / 3) * FM) / MEV)
    return le0, math.log10(lam / LAM_U)


def eself(Z, N, k):
    x = log_eself0(Z, N)
    if x is None:
        return 0.0
    le = x[0] + 2 * k * x[1]
    return math.inf if le > 300 else 10 ** le


def witnesses(slow_only):
    out = []
    for e in EXP.values():
        Z, N = e["Z"], e["N"]
        if Z < 3 or N < 3:
            continue
        tp = NUB.get((Z, N), (None, None))[0]
        if slow_only and tp is not None and tp < T_FAST:
            continue
        if not e.get("ME_est") and log_eself0(Z, N) is not None:
            out.append(("mass", Z, N))
        if e.get("Qa") is not None and not e.get("Qa_est"):
            td = NUB.get((Z - 2, N - 2), (None, None))[0]
            if slow_only and td is not None and td < T_FAST:
                continue
            if log_eself0(Z, N) is not None or log_eself0(Z - 2, N - 2) is not None:
                out.append(("Q_alpha", Z, N))
    return out


def shift(w, k):
    kind, Z, N = w
    if kind == "mass":
        return eself(Z, N, k)
    a, b = eself(Z, N, k), eself(Z - 2, N - 2, k)
    return math.inf if math.isinf(a) or math.isinf(b) else abs(a - b)


def k_max(ws, thr):
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if max(shift(w, mid) for w in ws) > thr:
            hi = mid
        else:
            lo = mid
    worst = max(ws, key=lambda w: shift(w, hi))
    t = NUB.get((worst[1], worst[2]), (None,))[0]
    return dict(k_max=lo, nuclide=f"Z={worst[1]} N={worst[2]} (A={worst[1] + worst[2]})", kind=worst[0],
                half_life_s=t)


R = dict(B_U=B_U, Q0_B=Q0_B, U238_half_life_s=T_U, nuclides_with_alpha_rate=sum(
    1 for v in NUB.values() if v[1] and not math.isinf(v[0])), results={})
for label, slow in (("all measured", False), ("living at least 1 ns", True)):
    ws = witnesses(slow)
    R["results"][label] = dict(witnesses=len(ws), **{f"{thr:g} MeV": k_max(ws, thr) for thr in THRESHOLDS})
# the self-energy of 8Be at k = 0.6, and the factor per 0.1 in k
_x = log_eself0(4, 4)
R["Be8"] = dict(half_life_s=NUB[(4, 4)][0], eself_MeV_at_k06=10 ** (_x[0] + 1.2 * _x[1]),
                factor_per_0p1_in_k=10 ** (0.2 * _x[1]))
# how far the alpha-decay rate climbs from uranium to the heaviest measured nuclei (Z = 113-118, ground states with a
# measured half-life and alpha branch), in decades
_sh = [math.log10(math.log(2) / t * br / LAM_U) for (Z, N), (t, br) in NUB.items()
       if 113 <= Z <= 118 and br and not math.isinf(t)]
R["superheavy_alpha_rate_decades"] = dict(n=len(_sh), min=min(_sh), max=max(_sh))


# ---------------------------------------------------------------- rev 5o: the same bound for the fission rate
# |P_X| = B_U (lambda_SF,X / lambda_SF,U-238)^k. Only branches NUBASE2020 gives as a value ('=' or '~') count; a limit
# ('<'), an unknown ('?') or an absent SF branch gives that nuclide no fission charge, so it cannot serve as a
# witness. The helpers above read NUB and LAM_U, which are swapped here for their fission versions.
def sf_branch(s):
    for item in s.split(";"):
        item = item.strip()
        if item.startswith("SF") and (len(item) == 2 or not item[2].isalpha()):
            rest = item[2:].strip()
            if rest[:1] in ("=", "~"):
                try:
                    return float(rest[1:].strip().split()[0]) / 100.0
                except (ValueError, IndexError):
                    return None
            return None
    return 0.0


NUB_ALPHA = NUB
NUB = {}
for line in NUB_PATH.read_text(encoding="latin-1").splitlines():
    if line.startswith("#") or len(line) < 80:
        continue
    try:
        A, Z, i = int(line[0:3]), int(line[4:7]), int(line[7])
    except ValueError:
        continue
    if i:
        continue
    t = half_life(line[69:78], line[78:80])
    if t is not None:
        NUB[(Z, A - Z)] = (t, 0.0 if math.isinf(t) else sf_branch(line[119:209]))
T_U, BR_U = NUB[(92, 146)]
LAM_U = math.log(2) / T_U * BR_U
F = dict(U238_sf_branch=BR_U, nuclides_with_fission_rate=sum(1 for v in NUB.values() if v[1] and not math.isinf(v[0])),
         results={})
for label, slow in (("all measured", False), ("living at least 1 ns", True)):
    ws = witnesses(slow)
    F["results"][label] = dict(witnesses=len(ws), **{f"{thr:g} MeV": k_max(ws, thr) for thr in THRESHOLDS})
_sf = [math.log10(math.log(2) / t * br / LAM_U) for (Z, N), (t, br) in NUB.items()
       if 104 <= Z <= 118 and br and not math.isinf(t)]
F["superheavy_fission_rate_decades"] = dict(n=len(_sf), min=min(_sf), max=max(_sf))
R["fission"] = F
(OUT / "decay_rate_bound.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
