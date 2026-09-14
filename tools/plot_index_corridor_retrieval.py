"""Generate Figure 8 of the manuscript (``fig08_index_corridor.pdf``).

Self-consistent retrieval of the refractive index dispersion from the three production
multilayer filters alone, without the reference single layers, compared against the
*Optics Continuum* corridor of Volet 1.

Shows the index sensitivity bundle as a function of the thickness process prior
sigma_proc (allowed delta OT, from 0.1% to unconstrained / no prior).
Data are read directly from ``results/index_sigma_sensitivity.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import CubicSpline

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE))

# Palette
C_REF = "#1F4E79"       # reference single layer, dark navy
C_CORRIDOR = "#D6E4F0"  # Optics Continuum corridor, soft blue
C_FLAT = "#7F7F7F"      # non-dispersive starting point, grey
C_OFFSET = "#C00000"    # offset retrieval, crimson
C_SHEAF_FILL = "#FFE0B2" # light amber fill for the sheaf envelope

# Colormap for the spline sheaf as sigma_proc varies from 0.1% to unconstrained
SIGMA_COLORS = {
    0.1: "#FFB74D",   # light amber
    0.2: "#FFA726",
    0.5: "#F57C00",   # nominal 0.5% (medium orange)
    1.0: "#E65100",
    2.0: "#D84315",
    None: "#BF360C",  # unconstrained (deep rust)
}

# Non-dispersive starting points: the published index frozen at its 1500 nm value, k = 0.
FLAT = {"Nb2O5": 2.23870, "SiO2": 1.47050}

PANELS = [
    ("Nb2O5", r"(a) $\mathrm{Nb}_2\mathrm{O}_5$ refractive index", (2.135, 2.255)),
    ("SiO2", r"(b) $\mathrm{SiO}_2$ refractive index", (1.410, 1.488)),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_published(material: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wavelengths in nm, index and its one-sigma envelope over the inversion window."""
    path = PACKAGE / "studies" / "volet2" / "indices" / f"{material}_H800_published.csv"
    table = pd.read_csv(path, comment="#")
    band = table[(table["Wavelength_nm"] >= 1500) & (table["Wavelength_nm"] <= 4000)]
    return band["Wavelength_nm"].to_numpy(), band["n"].to_numpy(), band["sigma_n"].to_numpy()


def load_sensitivity_data() -> dict:
    path = PACKAGE / "results" / "index_sigma_sensitivity.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_panel(ax, material: str, title: str, ylim: tuple[float, float], data: dict) -> None:
    wl_nm, n_ref, sigma = read_published(material)
    wl_um = wl_nm / 1000.0
    knots_wl = np.array(data["wavelength_knots_nm"])

    # 1. Reference corridor
    ax.fill_between(wl_um, n_ref - sigma, n_ref + sigma, color=C_CORRIDOR, zorder=1)

    # 2. Non-dispersive start
    ax.plot(wl_um, np.full_like(wl_um, FLAT[material]), "--", color=C_FLAT, lw=1.1, zorder=2)

    # 3. Spline sheaf across sigma_proc values
    spline_runs = data["spline_model"]
    all_splines = []
    for run in spline_runs:
        s_val = run["sigma_proc_pct"]
        knots = np.array(run[f"knots_{material}"])
        n_sp = CubicSpline(knots_wl, knots, bc_type="natural")(wl_nm)
        all_splines.append(n_sp)
        color = SIGMA_COLORS.get(s_val, "#E65100")
        if s_val == 0.5:
            # Highlight nominal publication run
            ax.plot(wl_um, n_sp, "-.", color=color, lw=1.6, zorder=5)
        elif s_val is None:
            # Highlight unconstrained run
            ax.plot(wl_um, n_sp, "-", color=color, lw=1.2, zorder=4, alpha=0.85)
        else:
            ax.plot(wl_um, n_sp, ":", color=color, lw=1.0, zorder=4, alpha=0.7)

    # Envelope shading of the spline sheaf
    spline_matrix = np.array(all_splines)
    n_min = np.min(spline_matrix, axis=0)
    n_max = np.max(spline_matrix, axis=0)
    ax.fill_between(wl_um, n_min, n_max, color=C_SHEAF_FILL, alpha=0.5, zorder=3)

    # 4. Reference single layer
    ax.plot(wl_um, n_ref, "-", color=C_REF, lw=1.6, zorder=6)

    # 5. Offset model (nominal sigma_proc = 0.5%)
    nominal_offset = next(r for r in data["offset_model"] if r["sigma_proc_pct"] == 0.5)
    dn = nominal_offset[f"dn_{material}"]
    ax.plot(wl_um, n_ref + dn, ":", color=C_OFFSET, lw=1.8, zorder=7)

    ax.set_xlabel(r"Wavelength $\lambda$ ($\mathrm{\mu m}$)")
    ax.set_ylabel(r"Refractive index $n$")
    ax.set_title(title, loc="left", fontsize=9.0, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(1.5, 4.0)
    ax.set_ylim(*ylim)


def legend_handles() -> tuple[list, list]:
    """One common legend for both panels."""
    handles = [
        Patch(facecolor=C_CORRIDOR, edgecolor="none"),
        Line2D([], [], color=C_REF, lw=1.6, linestyle="-"),
        Patch(facecolor=C_SHEAF_FILL, edgecolor="#F57C00", linewidth=1.0),
        Line2D([], [], color="#F57C00", lw=1.6, linestyle="-."),
        Line2D([], [], color=C_FLAT, lw=1.1, linestyle="--"),
        Line2D([], [], color=C_OFFSET, lw=1.8, linestyle=":"),
    ]
    labels = [
        r"Optics Continuum corridor ($\pm\sigma_n$)",
        r"Reference single layer",
        r"Spline sheaf ($\sigma_{\mathrm{proc}} = 0.1\%\dots\infty$)",
        r"Spline nominal ($\sigma_{\mathrm{proc}} = 0.5\%$)",
        r"Non-dispersive start ($\mathrm{d}n/\mathrm{d}\lambda = 0$)",
        r"Offset sensitivity ($\Delta n$, $\sigma_{\mathrm{proc}}=0.5\%$)",
    ]
    return handles, labels


def main() -> int:
    set_style()
    data = load_sensitivity_data()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, (material, title, ylim) in zip(axes, PANELS):
        draw_panel(ax, material, title, ylim, data)

    handles, labels = legend_handles()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_dir = PACKAGE.parent / "03_FIGURES_PUBLICATION"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300})):
        target = out_dir / f"fig08_index_corridor{suffix}"
        fig.savefig(target, **kwargs)
        print(f"Saved: {target}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
