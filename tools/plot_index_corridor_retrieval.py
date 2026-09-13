"""Generate Figure 7 of the manuscript (``fig7_index_corridor.pdf``).

Self-consistent retrieval of the refractive index dispersion from the three production
multilayer filters alone, without the reference single layers, compared against the
*Optics Continuum* corridor of Volet 1.

Two retrievals are drawn against the witness reference:

* **offset** -- 39 thicknesses plus one constant :math:`\\Delta n` per material, started from
  the published dispersion;
* **spline** -- 39 thicknesses plus three dispersion knots per material, started from a
  strictly non-dispersive index pinned at its 1500 nm value.

The retrieved values below are the converged results of those two inversions, quoted here so
the figure is a plot of published numbers rather than a re-run; ``ladder_summary.py`` and the
study files in ``studies/volet2/`` carry the runs themselves.
"""

from __future__ import annotations

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
C_REF = "#1F4E79"       # witness reference, dark navy
C_CORRIDOR = "#D6E4F0"  # Optics Continuum corridor, soft blue
C_FLAT = "#7F7F7F"      # non-dispersive starting point, grey
C_OFFSET = "#C00000"    # offset retrieval, crimson
C_SPLINE = "#ED7D31"    # spline retrieval, amber

# Non-dispersive starting points: the published index frozen at its 1500 nm value, k = 0.
FLAT = {"Nb2O5": 2.23870, "SiO2": 1.47050}

# Converged constant offsets of the offset retrieval.
OFFSET = {"Nb2O5": 0.00617, "SiO2": 0.00044}

# Converged spline knots of the flat-start retrieval, at 1500, 2750 and 4000 nm.
KNOT_NM = np.array([1500.0, 2750.0, 4000.0])
KNOTS = {
    "Nb2O5": np.array([2.24392, 2.20006, 2.16328]),
    "SiO2": np.array([1.47500, 1.45862, 1.42228]),
}

PANELS = [
    ("Nb2O5", r"(a) $\mathrm{Nb}_2\mathrm{O}_5$ refractive index", (2.135, 2.252)),
    ("SiO2", r"(b) $\mathrm{SiO}_2$ refractive index", (1.412, 1.485)),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.8,
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


def draw_panel(ax, material: str, title: str, ylim: tuple[float, float]) -> None:
    wl_nm, n_ref, sigma = read_published(material)
    wl_um = wl_nm / 1000.0
    n_spline = CubicSpline(KNOT_NM, KNOTS[material], bc_type="natural")(wl_nm)

    ax.fill_between(wl_um, n_ref - sigma, n_ref + sigma, color=C_CORRIDOR, zorder=1)
    ax.plot(wl_um, np.full_like(wl_um, FLAT[material]), "--", color=C_FLAT, lw=1.1, zorder=2)
    ax.plot(wl_um, n_ref, "-", color=C_REF, lw=1.6, zorder=3)
    ax.plot(wl_um, n_ref + OFFSET[material], ":", color=C_OFFSET, lw=1.8, zorder=5)
    ax.plot(wl_um, n_spline, "-.", color=C_SPLINE, lw=1.4, zorder=4)

    ax.set_xlabel(r"Wavelength $\lambda$ ($\mathrm{\mu m}$)")
    ax.set_ylabel(r"Refractive index $n$")
    ax.set_title(title, loc="left", fontsize=9.0, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(1.5, 4.0)
    ax.set_ylim(*ylim)


def legend_handles() -> tuple[list, list]:
    """One legend for both panels, so no curve is ever hidden behind it."""
    handles = [
        Patch(facecolor=C_CORRIDOR, edgecolor="none"),
        Line2D([], [], color=C_REF, lw=1.6, linestyle="-"),
        Line2D([], [], color=C_SPLINE, lw=1.4, linestyle="-."),
        Line2D([], [], color=C_FLAT, lw=1.1, linestyle="--"),
        Line2D([], [], color=C_OFFSET, lw=1.8, linestyle=":"),
    ]
    labels = [
        r"Optics Continuum corridor ($\pm\sigma_n$)",
        r"Witness reference",
        r"Spline retrieval (flat start)",
        r"Non-dispersive start ($\mathrm{d}n/\mathrm{d}\lambda = 0$)",
        r"Offset retrieval ($\Delta n$)",
    ]
    return handles, labels


def main() -> int:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (material, title, ylim) in zip(axes, PANELS):
        draw_panel(ax, material, title, ylim)

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
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_dir = PACKAGE.parent / "03_FIGURES_PUBLICATION"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300})):
        target = out_dir / f"fig7_index_corridor{suffix}"
        fig.savefig(target, **kwargs)
        print(f"Saved: {target}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
