"""ra_figures.py - figures for the first pass, from out/results.json, out/residuals.npz and out/point_z.json.

    python ra_figures.py

  out/fig1_qalpha_residual_maps.png   Q_alpha residuals of each model on the nuclide chart + what all models share
  out/fig2_ceiling_by_model.png       95% ceiling on the central strength against the switch width, per model
  out/fig3_consensus.png              the strongest features each model shows, and how the other models see them
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# palette: surfaces, ink and hues (light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, GRAY_SERIES = "#2a78d6", "#a9a79f"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#8f2423"])
DIVERGING.set_bad(SURFACE)

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "legend.frameon": False})


def grid_of(Z, N, v, zlim, nlim):
    g = np.full((zlim[1] - zlim[0] + 1, nlim[1] - nlim[0] + 1), np.nan)
    for z, n, x in zip(Z.astype(int), N.astype(int), v):
        if zlim[0] <= z <= zlim[1] and nlim[0] <= n <= nlim[1]:
            g[z - zlim[0], n - nlim[0]] = x
    return np.ma.masked_invalid(g)


def fig1(res, npz, pz):
    models = list(res["results"]["Qa"])
    zlim, nlim = (82, 119), (98, 186)
    ext = (nlim[0] - 0.5, nlim[1] + 0.5, zlim[0] - 0.5, zlim[1] + 0.5)
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.2), sharex=True, sharey=True)
    norm = Normalize(-1000, 1000)
    for ax, m in zip(axes.flat, models):
        Z, N, y = (npz[f"Qa|{m}|{k}"] for k in ("Z", "N", "y"))
        im = ax.imshow(grid_of(Z, N, y, zlim, nlim), origin="lower", extent=ext, cmap=DIVERGING, norm=norm,
                       aspect="auto", interpolation="nearest")
        r = res["results"]["Qa"][m]
        ax.set_title(f"{m}   rms {r['rms_keV']:.0f} keV" + ("   (table stops at Z = 110)" if r["Zmax"] < 118 else ""))
    # last panel: single-nuclide z-scores that every model agrees on (same sign), smallest |z| shown
    ax = axes.flat[5]
    per = {}
    for m in models:
        for _, z, n, v in pz["Qa"][m]:
            per.setdefault((z, n), []).append(v)
    Zc, Nc, vc = [], [], []
    for (z, n), v in per.items():
        v = np.array(v)
        if len(v) >= 4:
            Zc.append(z); Nc.append(n)
            vc.append(np.sign(v[0]) * np.min(np.abs(v)) if (np.all(v > 0) or np.all(v < 0)) else 0.0)
    im2 = ax.imshow(grid_of(np.array(Zc), np.array(Nc), np.array(vc), zlim, nlim), origin="lower", extent=ext,
                    cmap=DIVERGING, norm=Normalize(-3, 3), aspect="auto", interpolation="nearest")
    ax.set_title("Shared by every model: smallest |z|, where all agree in sign")
    k = int(np.argmax(np.abs(vc)))
    ax.annotate(f"largest: Z={Zc[k]}, N={Nc[k]}  (z = {vc[k]:+.1f})", (Nc[k] - 0.6, Zc[k] + 0.3), xytext=(104, 112),
                color=INK2, fontsize=8, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for a in axes.flat:
        a.plot([184], [114], marker="o", ms=7, mfc="none", mec=INK2, mew=1.0)
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
        a.set_xticks([100, 126, 152, 162, 184]); a.set_yticks([82, 92, 100, 108, 114, 118])
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    axes.flat[0].annotate("island centre\n(114, 184)", (184, 114), xytext=(165, 100), color=INK2, fontsize=8,
                          ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for a in axes[-1]:
        a.set_xlabel("Neutron number N")
    for a in axes[:, 0]:
        a.set_ylabel("Proton number Z")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.14, hspace=0.3, wspace=0.06)
    c1 = fig.add_axes([0.07, 0.055, 0.40, 0.014]); c2 = fig.add_axes([0.585, 0.055, 0.40, 0.014])
    cb = fig.colorbar(im, cax=c1, orientation="horizontal", extend="both")
    cb.set_label("Measured minus model, alpha-decay energy (keV)  [five model panels]", color=INK2); cb.outline.set_visible(False)
    cb = fig.colorbar(im2, cax=c2, orientation="horizontal")
    cb.set_label("z-score shared by all models  [last panel]", color=INK2); cb.outline.set_visible(False)
    fig.suptitle("Each mass model misses in different places; almost nothing is shared", x=0.07, y=0.965, ha="left",
                 fontsize=13, fontweight="semibold")
    fig.text(0.07, 0.928, "Measured alpha-decay energies (AME2020) minus five published mass models, Z ≥ 82. "
             "Last panel: grey means the models agree with the data or disagree with each other.", color=INK2, fontsize=9)
    fig.savefig(OUT / "fig1_qalpha_residual_maps.png"); plt.close(fig)


def fig2(res):
    R = res["results"]["Qa"]
    full = [m for m, r in R.items() if r["Zmax"] >= 118]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ends = {}
    for m in full:
        w = [x["width"] for x in R[m]["window_scan"]]; p = [x["Pc95_baryon"] for x in R[m]["window_scan"]]
        lead = m == "WS-Ma2019"
        ax.plot(w, p, color=BLUE if lead else GRAY_SERIES, lw=2.0 if lead else 1.4, solid_capstyle="round",
                zorder=3 if lead else 2)
        ends[m] = p[-1]
    # direct labels at the right end, spread so they do not collide
    order = sorted(full, key=lambda m: ends[m])
    ys = np.array([ends[m] for m in order]); lo = np.log10(ys)
    for i in range(1, len(lo)):
        lo[i] = max(lo[i], lo[i - 1] + 0.085)
    for m, y0, yl in zip(order, ys, 10 ** lo):
        ax.plot([12.0, 12.25], [y0, yl], color=MUTED, lw=0.6)
        ax.text(12.3, yl, m + (" (the paper's)" if m == "WS-Ma2019" else ""), va="center", color=INK, fontsize=8.5)
    ax.set_yscale("log"); ax.set_xlim(2.5, 15.6); ax.set_ylim(3e13, 3e15)
    ax.set_xticks([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Width of the switch (nucleons)")
    ax.set_ylabel("95% ceiling on central strength $P_c$ (× gravity)")
    fig.suptitle("The ceiling on the central strength barely depends on the mass model", x=0.02, y=0.975, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.text(0.02, 0.905, "From superheavy alpha-decay energies, four mass models that reach Z = 118. "
             "HFB-24 stops at Z = 110 and cannot test this.", color=INK2, fontsize=8.5)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.125)
    fig.savefig(OUT / "fig2_ceiling_by_model.png"); plt.close(fig)


def fig3(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)
    for ax, obs, name in zip(axes, ("Qa", "B"), ("Alpha-decay energies", "Binding energies")):
        top = res["consensus"][obs]["top"][:10][::-1]
        for i, t in enumerate(top):
            v = np.array(list(t["z"].values()))
            ax.plot([v.min(), v.max()], [i, i], color=AXIS, lw=2.0, solid_capstyle="round", zorder=1)
            ax.scatter(v, [i] * len(v), s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.6, zorder=3)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([t["what"].replace("single nuclide ", "").replace(" (A=", ", A=").replace(")", "")
                            for t in top], fontsize=8.2, color=INK)
        for x in (-3, 3):
            ax.axvline(x, color=INK2, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.text(3.1, -0.62, "+3σ", color=INK2, fontsize=8.5, va="bottom")
        ax.text(-3.1, -0.62, "−3σ", color=INK2, fontsize=8.5, va="bottom", ha="right")
        n = res["consensus"][obs]
        ax.set_title(f"{name}  ({n['features_compared']:,} features compared)")
        ax.set_xlim(-5.2, 5.2); ax.set_ylim(-0.7, len(top) - 0.3)
        ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("z-score of the feature in each mass model (one dot per model)")
    fig.suptitle("No feature reaches 3σ in every mass model", x=0.02, y=0.975, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.text(0.02, 0.905, "The ten features the models agree on most. A real effect would put every dot of a row "
             "beyond a 3σ line on the same side.", color=INK2, fontsize=8.8)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.80, bottom=0.12, wspace=0.62)
    fig.savefig(OUT / "fig3_consensus.png"); plt.close(fig)


if __name__ == "__main__":
    res = json.load(open(OUT / "results.json"))
    npz = np.load(OUT / "residuals.npz")
    pz = json.load(open(OUT / "point_z.json"))
    fig1(res, npz, pz); fig2(res); fig3(res)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("fig*.png"))))
