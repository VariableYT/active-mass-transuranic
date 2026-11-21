"""ra_ingest.py - raw published tables -> one normalised cache, with provenance.

Standard library only, so it can run isolated from the data directories:

    python -I ra_ingest.py

Reads   raw/<source>/<file>     (each source in its own folder, files exactly as published)
Writes  cache/experiment.json   AME2020 masses, Q_alpha, S_2n for every nuclide (estimated '#' values flagged)
        cache/models.json       predicted mass excess (keV) per model per nuclide
        cache/SOURCES.json      file, SHA-256, reference, rows parsed / rejected, spot checks

Nothing here is synthetic. A value that a table does not give is absent from the cache, never filled in.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW, CACHE = HERE / "raw", HERE / "cache"

# AME2020 mass excesses of 1H and the neutron (keV); used only to turn a model mass excess into a binding energy.
ME_H, ME_N = 7288.971064, 8071.31806


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find(rel, *alternatives):
    """raw/<rel>, or the first alternative path that exists (the paper's archive keeps some tables in ../data)."""
    for p in (RAW / rel, *alternatives):
        if p.exists():
            return p
    raise FileNotFoundError(f"missing input table raw/{rel}: see README.md for where each table comes from")


def num(field):
    """AME number field -> (value, estimated?) or (None, False). '#' marks a value from systematics, '*' = none."""
    s = field.strip()
    if not s or "*" in s:
        return None, False
    est = "#" in s
    try:
        return float(s.replace("#", ".")), est
    except ValueError:
        return None, False


# ----------------------------------------------------------------------------------------------- experiment
def read_ame_mass(path):
    out, started = {}, False
    for line in path.read_text(encoding="latin-1").splitlines():
        if not started:
            started = line.startswith("1N-Z")
            continue
        if len(line) < 80:
            continue
        try:
            N, Z, A = int(line[4:9]), int(line[9:14]), int(line[14:19])
        except ValueError:
            continue
        me, est = num(line[28:42]); dme, _ = num(line[42:54])
        bea, _ = num(line[54:69])
        if me is None or dme is None or N + Z != A:
            continue
        out[(Z, N)] = dict(el=line[20:23].strip(), ME=me, dME=dme, ME_est=est, B=None if bea is None else bea * A)
    return out


def read_ame_rct1(path):
    out, started = {}, False
    for line in path.read_text(encoding="latin-1").splitlines():
        if not started:
            started = line.startswith("1 A  elt")
            continue
        if len(line) < 78:
            continue
        try:
            A, Z = int(line[1:4]), int(line[8:11])
        except ValueError:
            continue
        rec = {}
        for key, a in (("S2n", 12), ("Qa", 56)):
            v, est = num(line[a:a + 12]); d, _ = num(line[a + 12:a + 22])
            if v is not None and d is not None:
                rec[key], rec["d" + key], rec[key + "_est"] = v, d, est
        if rec:
            out[(Z, A - Z)] = rec
    return out


# ----------------------------------------------------------------------------------------------- models
def read_ws_ma2019(path):
    """Columns: Z N A Bcal Bth (Bcal-Bth) Bexp ... (MeV); 888888 = not available. Bth is the pure model."""
    out, bad = {}, 0
    for line in path.read_text(encoding="latin-1").splitlines():
        f = line.split()
        if len(f) != 17:
            bad += bool(f)
            continue
        Z, N, A = int(f[0]), int(f[1]), int(f[2])
        bth = float(f[4])
        if Z + N != A or bth > 8e5:
            bad += 1
            continue
        out[(Z, N)] = Z * ME_H + N * ME_N - bth * 1e3
    return out, bad


def read_ws4(path):
    """Columns: A Z Beta2 Beta4 Beta6 Esh Edef Eexp Eth Mexp Mth (MeV)."""
    out, bad = {}, 0
    for line in path.read_text(encoding="latin-1").splitlines():
        f = line.split()
        if len(f) != 11:
            continue
        try:
            A, Z = int(f[0]), int(f[1]); mth = float(f[10])
        except ValueError:
            continue
        if not 0 < Z < A:
            bad += 1
            continue
        out[(Z, A - Z)] = mth * 1e3
    return out, bad


def read_frdm2012(path):
    """Table of arXiv:1508.06294, as text extracted from the PDF. Rows: N A eps2 eps3 eps4 eps6 b2 b3 b4 b6 Es+p
    Emic Ebind Mth [Mexp sigma] EFLmic MFLth (MeV, two decimals). The extraction splits numbers ("59 .39") and uses
    U+2212 for minus; both are repaired. A row is kept only if it has 16 or 18 fields and its own Ebind and Mth
    agree (Mth = Z M_H + N M_n - Ebind to 0.03 MeV), which rejects any row the extraction garbled."""
    out, bad, seen = {}, 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        toks, f = line.replace("−", "-").split(), []
        for t in toks:
            if f and re.fullmatch(r"-?\d+", f[-1]) and t.startswith("."):
                f[-1] += t
            else:
                f.append(t)
        if len(f) not in (16, 18) or not (re.fullmatch(r"\d+", f[0]) and re.fullmatch(r"\d+", f[1])):
            continue
        seen += 1
        try:
            N, A = int(f[0]), int(f[1]); ebind, mth = float(f[12]), float(f[13])
        except ValueError:
            bad += 1
            continue
        Z = A - N
        if Z < 8 or abs(mth - (Z * 7.289 + N * 8.0713 - ebind)) > 0.03 + 2e-5 * A or (Z, N) in out:
            bad += 1
            continue
        out[(Z, N)] = mth * 1e3
    return out, bad


def read_hfb24(path):
    """Columns: Z A bet2 bet4 Rch Edef Sn Sp Qbet Mcal Mexp-Mcal Jexp Jth Pexp Pth (MeV)."""
    out, bad = {}, 0
    for line in path.read_text(encoding="latin-1").splitlines():
        f = line.split()
        if len(f) != 15:
            continue
        try:
            Z, A = int(f[0]), int(f[1]); mcal = float(f[9])
        except ValueError:
            continue
        if not 0 < Z < A:
            bad += 1
            continue
        out[(Z, A - Z)] = mcal * 1e3
    return out, bad


def read_ktuy05(path):
    """Columns: Z N Mcal Esh alpha2 alpha4 alpha6 (MeV)."""
    out, bad = {}, 0
    for line in path.read_text(encoding="latin-1").splitlines():
        f = line.split()
        if len(f) != 7:
            continue
        try:
            Z, N = int(f[0]), int(f[1]); mcal = float(f[2])
        except ValueError:
            continue
        out[(Z, N)] = mcal * 1e3
    return out, bad


MODELS = {
    "WS-Ma2019": dict(
        file="ws_ma2019/WS_mass_table.txt", reader=read_ws_ma2019, family="macroscopic-microscopic",
        ref="N.-N. Ma et al., Chin. Phys. C 43, 044105 (2019); pure model column (no RBF correction)",
        url="https://github.com/lukeronger/NuclearData-LZU (NuclearData/data_table.txt)"),
    "WS4": dict(
        file="ws4/WS4.txt", reader=read_ws4, family="macroscopic-microscopic",
        ref="N. Wang, M. Liu, X. Wu, J. Meng, Phys. Lett. B 734, 215 (2014)",
        url="http://www.imqmd.com/mass/WS4.txt"),
    "FRDM2012": dict(
        file="frdm2012/frdm_text.txt", reader=read_frdm2012, family="macroscopic-microscopic",
        ref="P. Moller, A. J. Sierk, T. Ichikawa, H. Sagawa, At. Data Nucl. Data Tables 109-110, 1 (2016)",
        url="arXiv:1508.06294v1; text from: pdftotext -raw -enc UTF-8 (reproduces every row of the file used)"),
    "HFB-24": dict(
        file="hfb24/hfb24-dat", reader=read_hfb24, family="self-consistent mean field (Skyrme HFB)",
        ref="S. Goriely, N. Chamel, J. M. Pearson, Phys. Rev. C 88, 024308 (2013)",
        url="https://www.astro.ulb.ac.be/bruslib/nucdata/hfb24-dat"),
    "KTUY05": dict(
        file="ktuy05/KTUY05_m246.dat", reader=read_ktuy05, family="gross + spherical-basis shell terms",
        ref="H. Koura, T. Tachibana, M. Uno, M. Yamada, Prog. Theor. Phys. 113, 305 (2005)",
        url="https://wwwndc.jaea.go.jp/nucldata/mass/KTUY05_m246.dat"),
}

SPOT = [(8, 8), (82, 126), (92, 146), (94, 150), (96, 152), (114, 175)]  # 16O 208Pb 238U 244Pu 248Cm 289Fl


def main():
    CACHE.mkdir(exist_ok=True)
    sources = {}
    shared = HERE.parent / "data"
    fm = find("ame2020/mass.mas20", shared / "mass.mas20")
    fr = find("ame2020/rct1.mas20", shared / "rct1.mas20")
    mass, rct = read_ame_mass(fm), read_ame_rct1(fr)
    exp = {}
    for (Z, N), m in mass.items():
        rec = dict(Z=Z, N=N, **m)
        rec.update(rct.get((Z, N), {}))
        exp[f"{Z},{N}"] = rec
    # the two AME files must agree: Q_alpha recomputed from the mass table against the rct1 value
    me_a = mass[(2, 2)]["ME"]
    worst = 0.0
    for (Z, N), r in rct.items():
        if "Qa" in r and not r["Qa_est"] and (Z, N) in mass and (Z - 2, N - 2) in mass \
                and not mass[(Z, N)]["ME_est"] and not mass[(Z - 2, N - 2)]["ME_est"]:
            worst = max(worst, abs(mass[(Z, N)]["ME"] - mass[(Z - 2, N - 2)]["ME"] - me_a - r["Qa"]))
    assert worst < 0.5, f"mass.mas20 and rct1.mas20 disagree on Q_alpha by {worst} keV"
    assert abs(mass[(92, 146)]["ME"] - 47307.732) < 1e-6 and abs(rct[(92, 146)]["Qa"] - 4269.8581) < 1e-6
    n_meas = sum(not m["ME_est"] for m in mass.values())
    sources["AME2020"] = dict(
        ref="M. Wang et al., Chin. Phys. C 45, 030003 (2021)", url="https://www-nds.iaea.org/amdc/",
        files={"mass.mas20": sha256(fm), "rct1.mas20": sha256(fr)}, nuclides=len(mass), measured_masses=n_meas,
        estimated_masses=len(mass) - n_meas, ME_4He_keV=me_a, qalpha_cross_check_max_keV=worst)
    json.dump(dict(ME_H=ME_H, ME_N=ME_N, ME_alpha=me_a, nuclides=exp), open(CACHE / "experiment.json", "w"))

    models = {}
    for name, m in MODELS.items():
        path = find(m["file"], shared / Path(m["file"]).name)
        table, bad = m["reader"](path)
        meas = [(k, v) for k, v in table.items() if k in mass and not mass[k]["ME_est"]]
        rms_all = (sum((mass[k]["ME"] - v) ** 2 for k, v in meas) / len(meas)) ** 0.5
        heavy = [(k, v) for k, v in meas if k[0] >= 82]
        rms_heavy = (sum((mass[k]["ME"] - v) ** 2 for k, v in heavy) / len(heavy)) ** 0.5
        models[name] = {f"{Z},{N}": v for (Z, N), v in table.items()}
        sources[name] = dict(
            ref=m["ref"], url=m["url"], family=m["family"], file=m["file"], sha256=sha256(path),
            bytes=path.stat().st_size, rows=len(table), rows_rejected=bad, Zmax=max(k[0] for k in table),
            rms_vs_AME2020_keV=dict(all_measured=round(rms_all, 1), n_all=len(meas), Z_ge_82=round(rms_heavy, 1),
                                    n_Z_ge_82=len(heavy)),
            spot_ME_minus_AME_keV={f"{Z},{N}": (round(table[(Z, N)] - mass[(Z, N)]["ME"], 1)
                                               if (Z, N) in table and (Z, N) in mass else None) for Z, N in SPOT})
    json.dump(models, open(CACHE / "models.json", "w"))
    json.dump(sources, open(CACHE / "SOURCES.json", "w"), indent=1)

    a = sources["AME2020"]
    print(f"AME2020: {a['nuclides']} nuclides, {a['measured_masses']} measured masses, "
          f"{a['estimated_masses']} estimated (#); mass/rct1 Q_alpha agree to {worst:.3f} keV")
    for name in MODELS:
        s = sources[name]
        print(f"{name:10s} rows {s['rows']:5d} rejected {s['rows_rejected']:3d} Zmax {s['Zmax']:3d}  "
              f"rms vs AME2020: all {s['rms_vs_AME2020_keV']['all_measured']:6.1f} keV, "
              f"Z>=82 {s['rms_vs_AME2020_keV']['Z_ge_82']:6.1f} keV (n={s['rms_vs_AME2020_keV']['n_Z_ge_82']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
