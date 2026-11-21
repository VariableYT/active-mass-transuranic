"""Stage-1 (100 g U-238 capsule) Newtonian residual vs machining/centring tolerance, using the paper's own
Monte Carlo (1_experiment_sim/active_mass_sim.py: measure/park exchange plus 180-degree flip,
oxide pellet with a 1 %/mm density gradient). Only the dimension and centring tolerances are changed.

Run:  python validation/tolerance_scan.py   -> validation/out/tolerance_scan.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIMDIR = HERE.parent / "1_experiment_sim"
spec = importlib.util.spec_from_file_location("active_mass_sim", SIMDIR / "active_mass_sim.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

N = 1500
out = {}
for tol_um in (1.0, 2.0, 3.0, 5.0, 10.0 / np.sqrt(3), 10.0):
    sim.TOL.update(sig_com=tol_um * 1e-6, sig_dim=tol_um * 1e-6)
    raw, flip, parts = sim.newton_mc("238U", N, 10.0, 20261118)
    out[f"{tol_um:.3g}"] = dict(sigma_um=tol_um, raw_p95=float(np.percentile(np.abs(raw), 95)),
                                flip_p95=float(np.percentile(np.abs(flip), 95)), parts_rms=parts)
    print(f"sigma {tol_um:5.2f} um: raw p95 {out[f'{tol_um:.3g}']['raw_p95']:.2e}, "
          f"flip p95 {out[f'{tol_um:.3g}']['flip_p95']:.2e}")
(HERE / "out" / "tolerance_scan.json").write_text(json.dumps(out, indent=1))
