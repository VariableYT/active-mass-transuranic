"""Regenerate every number, table and figure in the paper and compare them with the shipped results.

    python run_all.py                 quick: paper/compute.py only, from the shipped intermediate JSON files
    python run_all.py --full          rerun every simulation and fit first, then paper/compute.py
    python run_all.py --mass-models   rerun only the mass-model check of Sec. II D, then paper/compute.py
    python run_all.py --check         only compare the current outputs with the shipped ones

Before anything runs, the shipped outputs are copied to _shipped/ (once). Afterwards every regenerated JSON file
is compared number by number with its shipped copy, and the tables and CSV as text (ignoring line endings).
A file is "identical" if every number agrees to 1e-12, "agrees" if every number agrees to 1e-4 (floating-point
differences between platforms and library builds; the worst relative difference is printed), and "DIFFERS"
otherwise. Figures are regenerated in paper/figs/ but not compared (PDFs carry timestamps).

The mass-model check needs four third-party tables that are not redistributed here. data/fetch_data.py tries to
download them; if any is missing, that one step is skipped and its shipped result is used.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "_shipped"
JSONS = [
    "1_experiment_sim/results.json",
    "1_experiment_sim/detection_results.json",
    "1_experiment_sim/u_stage_refine.json",
    "1_experiment_sim/calibration_check.json",
    "1_experiment_sim/instrument_calibration.json",
    "2_indirect_search/channels.json",
    "2_indirect_search/data_test.json",
    "2_indirect_search/data_test_robust.json",
    "residual_analyzer/out/results.json",
    "heavy_pair/out/heavy_pair_sim.json",
    "heavy_pair/out/u238_pair_campaign.json",
    "paper/numbers.json",
]
# For this file only the parts the paper uses are compared: the rest holds simulated look-elsewhere p-values,
# which can move in the last digit between platforms.
SUBTREE = {"residual_analyzer/out/results.json": ["validation", "window_paper_selection_Z_ge_84"]}
TEXTS = ["paper/tab_predictions.tex", "paper/tab_practical.tex", "paper/tab_campaign.tex", "paper/tab_heavy.tex",
         "paper/tab_fingerprint.tex", "2_indirect_search/residuals_she.csv"]
MODEL_TABLES = ["ws4/WS4.txt", "frdm2012/frdm_text.txt", "hfb24/hfb24-dat", "ktuy05/KTUY05_m246.dat"]


def snapshot():
    if SNAP.exists():
        return
    for rel in JSONS + TEXTS:
        dst = SNAP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)


def run(folder, *args, isolated=False):
    t0 = time.time()
    print(f"[{folder}] python {' '.join(args)}", flush=True)
    cmd = [sys.executable] + (["-I"] if isolated else []) + list(args)
    subprocess.run(cmd, cwd=ROOT / folder, check=True, stdout=subprocess.DEVNULL)
    print(f"    done in {time.time() - t0:.0f} s", flush=True)


def mass_models():
    missing = [t for t in MODEL_TABLES if not (ROOT / "residual_analyzer" / "raw" / t).exists()]
    if missing:
        print("[residual_analyzer] SKIPPED, shipped result used. Missing tables: " + ", ".join(missing)
              + " (see data/fetch_data.py)", flush=True)
        return
    run("residual_analyzer", "ra_ingest.py", isolated=True)   # the only step that reads downloaded files
    run("residual_analyzer", "residual_analyzer.py")


def worst_rel(lines):
    w = 0.0
    for line in lines:
        try:
            a, b = (float(x) for x in line.split(": ", 1)[1].split(" vs shipped "))
        except ValueError:
            return math.inf  # a non-numeric difference
        w = max(w, abs(a - b) / max(abs(b), 1e-300))
    return w


def diff(a, b, path="", out=None, rtol=1e-12, atol=1e-300):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one file")
            else:
                diff(a[k], b[k], f"{path}/{k}", out, rtol, atol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, rtol, atol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=rtol, abs_tol=atol) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} vs shipped {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs shipped {b!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="rerun the simulations and fits (about 11 minutes)")
    ap.add_argument("--mass-models", action="store_true", help="rerun only the mass-model check (about 2 minutes)")
    ap.add_argument("--check", action="store_true", help="compare only; run nothing")
    args = ap.parse_args()
    snapshot()
    t0 = time.time()
    if args.check:
        return compare()
    if args.full or args.mass_models:
        if subprocess.run([sys.executable, "data/fetch_data.py"], cwd=ROOT).returncode:
            sys.exit("a required input table is missing or changed; see data/fetch_data.py")
    if args.full:
        sim = ROOT / "1_experiment_sim"
        run("1_experiment_sim", "active_mass_sim.py", "out")
        shutil.copy2(sim / "out" / "results.json", sim / "results.json")
        run("1_experiment_sim", "detection_map.py")
        shutil.copy2(sim / "out" / "detection_results.json", sim / "detection_results.json")
        run("1_experiment_sim", "u_stage_refine.py")
        run("1_experiment_sim", "calibration_check.py")
        run("1_experiment_sim", "instrument_calibration.py")
        run("2_indirect_search", "channels.py")
        run("2_indirect_search", "data_test.py")
        run("2_indirect_search", "data_test_robust.py")
    if args.full or args.mass_models:
        mass_models()
    run(".", "paper/compute.py")
    print(f"total {time.time() - t0:.0f} s\n")
    return compare()


def compare():
    bad = 0
    for rel in JSONS:
        new, old = json.load(open(ROOT / rel)), json.load(open(SNAP / rel))
        if rel in SUBTREE:
            new, old = ({k: d.get(k) for k in SUBTREE[rel]} for d in (new, old))
        d = diff(new, old)
        w = worst_rel(d)
        if not d:
            print(f"identical  {rel}")
        elif w <= 1e-4:
            print(f"agrees     {rel}  (worst relative difference {w:.1e})")
        else:
            print(f"DIFFERS    {rel}")
            for line in d[:10]:
                print("           " + line)
            bad += 1
    for rel in TEXTS:
        same = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n") == (SNAP / rel).read_bytes().replace(b"\r\n", b"\n")
        print(f"{'identical' if same else 'DIFFERS  '}  {rel}")
        bad += not same
    print("\nall outputs reproduce" if not bad else f"\n{bad} file(s) differ from the shipped results")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
