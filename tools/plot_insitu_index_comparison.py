"""Figure 11: the in situ retrieved dispersions against the single-layer reference corridor.

The retrieved spline is read from ``results/insitu_index_corrections.json``, which
``plot_insitu_convergence.py`` writes from the converged unregularized solution. Run that
script first; nothing here is typed in by hand.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

PACKAGE = Path(__file__).resolve().parent.parent
WORKSPACE = PACKAGE.parent
RESULTS = PACKAGE / "results"

# Load published reference single-layer indices (companion paper)
df_pub_nb = pd.read_csv(PACKAGE / "studies" / "volet2" / "indices" / "Nb2O5_H800_published.csv", comment="#")
df_pub_si = pd.read_csv(PACKAGE / "studies" / "volet2" / "indices" / "SiO2_H800_published.csv", comment="#")

# Load literature Franta indices
df_lit_nb = pd.read_csv(PACKAGE / "studies" / "volet2" / "indices" / "Nb2O5_Franta2024.csv", comment="#")
df_lit_si = pd.read_csv(PACKAGE / "studies" / "volet2" / "indices" / "SiO2_Franta2016.csv", comment="#")

# Converged knot corrections of the unregularized run, relative to the literature tables.
_knot_file = RESULTS / "insitu_index_corrections.json"
if not _knot_file.is_file():
    raise SystemExit(f"{_knot_file.name} is missing: run tools/plot_insitu_convergence.py first.")
_knots = json.loads(_knot_file.read_text(encoding="utf-8"))
knots_grid = np.asarray(_knots["knots_nm"], dtype=float)
delta_nb = np.asarray(_knots["delta_n"]["Nb2O5"], dtype=float)
delta_si = np.asarray(_knots["delta_n"]["SiO2"], dtype=float)

cs_delta_nb = CubicSpline(knots_grid, delta_nb)
cs_delta_si = CubicSpline(knots_grid, delta_si)

wls = np.linspace(1500.0, 4000.0, 501)

# Interpolate onto wls
n_pub_nb = np.interp(wls, df_pub_nb["Wavelength_nm"], df_pub_nb["n"])
sigma_pub_nb = np.interp(wls, df_pub_nb["Wavelength_nm"], df_pub_nb["sigma_n"])

n_pub_si = np.interp(wls, df_pub_si["Wavelength_nm"], df_pub_si["n"])
sigma_pub_si = np.interp(wls, df_pub_si["Wavelength_nm"], df_pub_si["sigma_n"])

n_lit_nb = np.interp(wls, df_lit_nb["Wavelength_nm"], df_lit_nb["n"])
n_lit_si = np.interp(wls, df_lit_si["Wavelength_nm"], df_lit_si["n"])

n_insitu_nb = n_lit_nb + cs_delta_nb(wls)
n_insitu_si = n_lit_si + cs_delta_si(wls)

# Plotting: 2 columns, each with top (n) and bottom (delta n / sigma)
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})

import pandas as _pd
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_insitu_convergence import panel_indices as _panel_indices

fig = plt.figure(figsize=(11, 8.2))
_grid = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.35], hspace=0.30, wspace=0.22)
axes = {(0, 0): fig.add_subplot(_grid[0, 0]), (0, 1): fig.add_subplot(_grid[0, 1])}

# Panel (a): Nb2O5 index
ax = axes[0, 0]
ax.fill_between(wls, n_pub_nb - sigma_pub_nb, n_pub_nb + sigma_pub_nb, color="#BDD7EE", alpha=0.6, label=r"Optics Continuum corridor ($\pm 1\sigma$)")
ax.plot(wls, n_pub_nb, color="#1F4E79", lw=2.0, label=r"Companion-paper nominal (reference single layer)")
ax.plot(wls, n_lit_nb, color="#C00000", lw=1.8, ls="--", label=r"Literature starting guess (Franta 2024)")
ax.plot(wls, n_insitu_nb, color="#E36C09", lw=2.2, label=r"In situ retrieved (4-knot spline, 3 multilayers)")
_knot_text = ", ".join(f"{w:.0f}" for w in knots_grid)
ax.scatter(knots_grid, np.interp(knots_grid, wls, n_insitu_nb), color="#E36C09", edgecolor="black", s=40, zorder=5, label=f"Spline knots ({_knot_text} nm)")
ax.set_title(r"(a) $\mathrm{Nb}_2\mathrm{O}_5$ refractive index dispersion", fontsize=11, fontweight="bold", pad=8)
ax.set_ylabel("Refractive index $n$")
ax.set_xlim(1500, 4000)
ax.set_ylim(2.13, 2.26)
ax.grid(True, ls=":", alpha=0.6)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

# Panel (b): SiO2 index
ax = axes[0, 1]
ax.fill_between(wls, n_pub_si - sigma_pub_si, n_pub_si + sigma_pub_si, color="#BDD7EE", alpha=0.6, label=r"Optics Continuum corridor ($\pm 1\sigma$)")
_gap_si = float(n_lit_si[0] - n_pub_si[0])
_climb_si = float(n_insitu_si[0] - n_lit_si[0])
ax.plot(wls, n_pub_si, color="#1F4E79", lw=2.0, label=r"Companion-paper nominal (reference single layer)")
ax.plot(wls, n_lit_si, color="#C00000", lw=1.8, ls="--", label=rf"Literature starting guess (Franta 2016, $\Delta n = {_gap_si:+.4f}$)")
ax.plot(wls, n_insitu_si, color="#E36C09", lw=2.2, label=r"In situ retrieved (4-knot spline, 3 multilayers)")
ax.scatter(knots_grid, np.interp(knots_grid, wls, n_insitu_si), color="#E36C09", edgecolor="black", s=40, zorder=5, label="Spline knots (4 knots)")
ax.annotate(rf"${_climb_si:+.4f}$ from the literature start", xy=(1500, n_lit_si[0]), xytext=(1700, 1.445),
            arrowprops=dict(arrowstyle="->", color="#C00000", lw=1.5), fontsize=8.5, color="#C00000")
ax.set_title(r"(b) $\mathrm{SiO}_2$ refractive index dispersion", fontsize=11, fontweight="bold", pad=8)
ax.set_ylabel("Refractive index $n$")
ax.set_xlim(1500, 4000)
ax.set_ylim(1.41, 1.49)
ax.grid(True, ls=":", alpha=0.6)
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

# Panel (c): what every run retrieved at 1500 nm
ax = fig.add_subplot(_grid[1, :])
_runs = _pd.read_csv(PACKAGE / "results" / "insitu_multistart_noprior.csv")
_panel_indices(ax, _runs)
ax.set_title(r"(c) retrieved index of every run, at 1500 nm", fontsize=11,
             fontweight="bold", loc="left", pad=8)

out_pdf = WORKSPACE / "03_FIGURES_PUBLICATION" / "fig11_insitu_index_comparison.pdf"
out_png = WORKSPACE / "03_FIGURES_PUBLICATION" / "fig11_insitu_index_comparison.png"
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved figure to {out_pdf} and {out_png}")
