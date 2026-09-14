"""Figure 10: what the in situ multi-start inversion actually does.

Four panels, because the result of Section 6 has four parts and a table shows none of them.

(a) Every traced inversion, residual against function evaluation. The starts are spread over a
    factor of two in residual and all of them fall onto the same value.
(b) The final residual of every run in the benchmark, one dot each. The runs that succeed and
    the runs that fail are separated by an empty decade, which is what makes a residual
    threshold a sufficient filter in practice.
(c) The retrieved refractive index of each oxide at 1500 nm, against the corridor measured on
    the reference single layers. The inversions start far outside it and land inside.
(d) The retrieved layer thicknesses, against their nominal values. They do not land on the
    design, and that is the point: the spectra fix the indices, not the geometry.

    python tools/plot_insitu_convergence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
WORKSPACE = PACKAGE.parent
INDICES = PACKAGE / "studies" / "volet2" / "indices"
RESULTS = PACKAGE / "results"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE = "#2F5D8C"
ORANGE = "#D9772B"
GREEN = "#3A7D44"
RED = "#B44A4A"
GREY = "#8A8A8A"
CORRIDOR = "#D6E4F0"

SUCCESS_RMSE = 0.60  # per cent; the threshold the article quotes


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.0,
        "axes.labelsize": 8.0, "axes.titlesize": 8.5,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "lines.linewidth": 1.0,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def layer_departures() -> pd.DataFrame:
    """Per-layer departure of the converged unregularized solution, cached after the first run.

    The benchmark reports only the aggregate thickness error, so the converged design is
    recomputed here once. The setup below mirrors ``insitu_multistart_noprior`` exactly:
    literature starting indices, no process prior, four spline knots per oxide.
    """
    cache = RESULTS / "insitu_layer_departures.csv"
    knot_cache = RESULTS / "insitu_index_corrections.json"
    if cache.is_file() and knot_cache.is_file():
        return pd.read_csv(cache, comment="#")

    import importlib

    bench = importlib.import_module("insitu_multistart_noprior")
    from certus_re.dispersion import TabulatedIndex
    from certus_re.forward import Evaluator
    from certus_re.solve import invert

    study = bench.get_3components_study()

    # The nominal the departures are counted against is the one of the deposition sheets,
    # converted to nanometres with the campaign indices -- as insitu_multistart_noprior does it,
    # and before the literature tables are substituted. Reading it after the substitution would
    # move the nominal itself by 1.7 per cent on SiO2, and the bars of this panel would no longer
    # be the numbers of Table 4.
    nominal = Evaluator(study).nominal_thicknesses()

    for material, fname in (("Nb2O5", "Nb2O5_Franta2024.csv"), ("SiO2", "SiO2_Franta2016.csv")):
        df = pd.read_csv(INDICES / fname, comment="#")
        study.materials[material] = TabulatedIndex(
            wavelength_nm=df["Wavelength_nm"].to_numpy(),
            n=df["n"].to_numpy(), k=df["k"].to_numpy(),
            name=f"{material}_literature",
            uncertainty_n=df["sigma_n"].to_numpy(),
            valid_range_nm=(250.0, 4840.0),
        )
    study.free.thickness_tolerance = 0.25
    study.free.process_prior_pct = None
    study.free.index_correction = "bounded"
    study.free.index_tube_delta = 0.08
    study.free.index_n_knots = 4

    print("computing the converged solution once, for panel (b)...", flush=True)
    result = invert(study, start={k: v.copy() for k, v in nominal.items()})

    rows = []
    for stack in ("AR6", "BS45", "BS17"):
        for j, material in enumerate(study.stacks[stack].materials):
            d_nom = float(nominal[stack][j])
            d_ret = float(result.thicknesses[stack][j])
            rows.append({"stack": stack, "layer": j + 1, "material": material,
                         "d_nominal_nm": d_nom, "d_retrieved_nm": d_ret,
                         "departure_pct": 100.0 * (d_ret - d_nom) / d_nom})
    frame = pd.DataFrame(rows)
    header = ("# Per-layer departure of the unregularized in situ solution from the nominal "
              "design, for Fig. 10(b). Written by tools/plot_insitu_convergence.py.\n")
    cache.write_text(header + frame.to_csv(index=False), encoding="utf-8")

    # The same solution carries the retrieved dispersions. Figure 11 reads them from here
    # rather than carrying a copy of the knot values in its own source.
    knot_cache.write_text(json.dumps({
        "description": "Spline corrections to the literature dispersions at the four knots, "
                       "from the converged unregularized in situ solution. Written by "
                       "tools/plot_insitu_convergence.py.",
        "knots_nm": list(np.linspace(1500.0, 4000.0, 4)),
        "delta_n": {m: [float(v) for v in result.index_corrections[m]]
                    for m in ("Nb2O5", "SiO2")},
    }, indent=1), encoding="utf-8")
    return frame


def panel_funnel(ax) -> None:
    traces = json.loads((RESULTS / "insitu_convergence_traces.json").read_text(encoding="utf-8"))
    for t in traces:
        history = np.asarray(t["history"], dtype=float)
        if history.size < 2:
            continue
        # The weighting is fixed within a run, so the history scales onto the reported RMSE.
        scale = t["rmse_pct"] / history[-1]
        curve = history * scale
        converged = t["rmse_pct"] < SUCCESS_RMSE
        ax.plot(np.arange(curve.size), curve,
                color=BLUE if converged else RED,
                alpha=0.55 if converged else 0.9,
                lw=0.8 if converged else 1.2, zorder=2 if converged else 3)
    ax.axhline(0.20, color=GREEN, lw=1.0, ls="--", zorder=4)
    ax.text(0.985, 0.205, "photometric noise floor", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6.5, color=GREEN)
    ax.set_yscale("log")
    ax.set_xlabel("function evaluation")
    ax.set_ylabel("RMSE (%)")
    ax.set_title("(a) every start that succeeds ends on the same value",
                 loc="left", fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.2, label="reaches the global minimum"),
                       Line2D([], [], color=RED, lw=1.2, label="trapped in a false minimum")],
              loc="upper right", framealpha=0.95)


def panel_gap(ax, runs: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    levels = sorted(runs["perturbation_pct"].unique())
    for level in levels:
        sub = runs[runs["perturbation_pct"] == level]
        x = level + rng.uniform(-1.4, 1.4, size=len(sub))
        good = sub["rmse_pct"] < SUCCESS_RMSE
        ax.scatter(x[good.to_numpy()], sub["rmse_pct"][good], s=7, color=BLUE,
                   alpha=0.55, edgecolors="none", zorder=3)
        ax.scatter(x[~good.to_numpy()], sub["rmse_pct"][~good], s=16, color=RED,
                   alpha=0.9, marker="x", lw=0.9, zorder=4)
        rate = 100.0 * good.sum() / len(sub)
        se = 100.0 * np.sqrt((rate / 100) * (1 - rate / 100) / len(sub))
        label = f"{rate:.0f}%" if se == 0 else f"{rate:.0f}$\\pm${se:.0f}%"
        ax.text(level, 0.975, label, transform=ax.get_xaxis_transform(), rotation=90,
                ha="center", va="top", fontsize=6.8, color=BLUE)

    lo = runs.loc[runs["rmse_pct"] >= SUCCESS_RMSE, "rmse_pct"].min()
    hi = runs.loc[runs["rmse_pct"] < SUCCESS_RMSE, "rmse_pct"].max()
    ax.axhspan(hi, lo, color=GREY, alpha=0.16, zorder=1)
    ax.text(10, np.sqrt(hi * lo), "no run lands here", ha="center", va="center",
            fontsize=7.0, color="#555555", style="italic", zorder=5)
    ax.annotate("", xy=(-2.2, lo), xytext=(-2.2, hi),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.9), zorder=5)
    ax.text(-2.8, np.sqrt(hi * lo), f"$\\times${lo / hi:.1f}", rotation=90, ha="right",
            va="center", fontsize=6.8, color="#555555")
    ax.set_xlabel("perturbation of the start (%)")
    ax.set_title("and where they land", loc="left", fontweight="bold")
    ax.set_xticks(levels)
    ax.set_xlim(-3.0, 23.0)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.tick_params(labelleft=False)


def panel_indices(ax, runs: pd.DataFrame) -> None:
    """Retrieved index at 1500 nm, one marker per run, against the reference corridor.

    The runs that converge pile onto a single marker inside the corridor; the trapped ones
    scatter over the whole width of the index tube the solver is allowed. The residual
    threshold therefore separates the two populations in index space as well, which is what
    makes it usable as a filter. The axis is set from the data so that no marker is clipped.
    """
    good = runs[runs["rmse_pct"] < SUCCESS_RMSE]
    bad = runs[runs["rmse_pct"] >= SUCCESS_RMSE]

    span = 0.0
    for k, (label, tag, stem) in enumerate((("SiO$_2$", "si", "SiO2"),
                                            ("Nb$_2$O$_5$", "nb", "Nb2O5"))):
        pub = pd.read_csv(INDICES / f"{stem}_H800_published.csv", comment="#")
        lit = pd.read_csv(INDICES / (f"{stem}_Franta2016.csv" if stem == "SiO2"
                                     else f"{stem}_Franta2024.csv"), comment="#")
        i = (pub["Wavelength_nm"] - 1500).abs().idxmin()
        n_pub, sigma = float(pub["n"][i]), float(pub["sigma_n"][i])
        j = (lit["Wavelength_nm"] - 1500).abs().idxmin()
        n_lit = float(lit["n"][j]) - n_pub

        ax.add_patch(plt.Rectangle((-sigma, k - 0.19), 2 * sigma, 0.38,
                                   facecolor=CORRIDOR, edgecolor="none", zorder=1))
        ax.plot([0, 0], [k - 0.19, k + 0.19], color=BLUE, lw=1.8, zorder=4)

        ax.scatter(bad[f"n_ret_{tag}_1500"] - n_pub, np.full(len(bad), k - 0.075),
                   marker="x", s=13, color=RED, alpha=0.55, lw=0.8, zorder=5)
        ax.scatter(good[f"n_ret_{tag}_1500"] - n_pub, np.full(len(good), k + 0.075),
                   marker="D", s=20, color=ORANGE, alpha=0.9, edgecolors="none", zorder=6)
        ax.scatter([n_lit], [k], marker="o", s=40, facecolors="none",
                   edgecolors="#4A4A4A", lw=1.3, zorder=7)

        jump = float(good[f"n_ret_{tag}_1500"].mean()) - n_pub - n_lit
        ax.annotate("", xy=(n_lit + jump, k + 0.25), xytext=(n_lit, k + 0.25),
                    arrowprops=dict(arrowstyle="-|>", color="#4A4A4A", lw=1.0,
                                    shrinkA=1, shrinkB=1), zorder=7)
        ax.text(n_lit + jump / 2, k + 0.285, f"{jump:+.4f}", ha="center", va="bottom",
                fontsize=6.8, color="#4A4A4A")
        ax.text(0.985, k + 0.30, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5)
        span = max(span, float(np.max(np.abs(bad[f"n_ret_{tag}_1500"] - n_pub))),
                   abs(n_lit))

    ax.axvline(0, color=BLUE, lw=0.7, alpha=0.3, zorder=2)
    ax.set_yticks([])
    ax.set_ylim(-0.55, 2.35)
    limit = 0.01 * np.ceil(100 * 1.08 * span)
    ax.set_xlim(-limit, limit)
    ax.set_xticks(np.round(np.arange(-limit, limit + 1e-9, 0.02), 3))
    ax.set_xlabel(r"retrieved $n$ at 1500 nm, minus the reference single-layer value")
    ax.set_title("(b) the indices land in the reference corridor",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="x", ls=":", alpha=0.45)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="none", mec="#4A4A4A", ms=6,
               label="literature start"),
        Line2D([], [], marker="D", ls="", color=ORANGE, ms=5, label="converged runs"),
        Line2D([], [], marker="x", ls="", color=RED, ms=5, label="trapped runs"),
        Line2D([], [], color=CORRIDOR, lw=7,
               label=r"reference $\pm\sigma_n$"),
    ], loc="upper left", framealpha=0.95, ncol=2, handletextpad=0.5, columnspacing=1.0)


def panel_thicknesses(ax, departures: pd.DataFrame) -> None:
    colours = {"AR6": ORANGE, "BS45": BLUE, "BS17": GREEN}

    # What the chamber actually does, for scale: the rms departure of the regularized fit of
    # Section 5, component by component. Without it a reader reads the bars as manufacturing
    # error, when they are the width of the family the spectra cannot separate.
    designs = pd.read_csv(RESULTS / "designs.csv", comment="#")
    process = {
        stack: float(np.sqrt(np.mean(g["departure_pct"].to_numpy() ** 2)))
        for stack, g in designs.groupby("stack")
        if stack in colours
    }

    centres: list = []
    x = 0
    for stack, group in departures.groupby("stack", sort=False):
        idx = np.arange(x, x + len(group))
        band = process.get(stack)
        if band is not None:
            ax.fill_between([idx[0] - 0.6, idx[-1] + 0.6], -band, band,
                            color=GREY, alpha=0.22, lw=0, zorder=1)
        ax.bar(idx, group["departure_pct"], width=0.78, color=colours.get(stack, GREY),
               edgecolor="none", zorder=3)
        centres.append((idx.mean(), stack))
        x += len(group) + 2
    ax.set_ylim(-17.5, 17.5)
    for centre, stack in centres:
        ax.text(centre, 15.4, stack, ha="center", va="center", fontsize=7.2,
                color="#333333")
    ax.axhline(0, color="black", lw=0.7, zorder=4)
    rms = float(np.sqrt(np.mean(departures["departure_pct"].to_numpy() ** 2)))
    for level in (rms, -rms):
        ax.axhline(level, color=RED, lw=0.8, ls="--", alpha=0.8, zorder=2)
    ax.text(0.015, rms + 0.3, f"rms departure, {rms:.1f}%", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.5, color=RED)
    ax.legend(handles=[Line2D([], [], color=GREY, alpha=0.5, lw=7,
                              label="what the run does: rms of the regularized fit")],
              loc="lower right", framealpha=0.95, fontsize=6.5)
    ax.set_xticks([])
    ax.set_xlabel("layer, substrate to air, for each component")
    ax.set_ylabel("departure from nominal (%)")
    ax.set_title("(b) the layer thicknesses, meanwhile, do not settle on the design",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="y", ls=":", alpha=0.45)


def main() -> int:
    set_style()
    runs = pd.read_csv(RESULTS / "insitu_multistart_noprior.csv")
    departures = layer_departures()

    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.44)
    top = grid[0, 0].subgridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.04)

    ax_funnel = fig.add_subplot(top[0, 0])
    ax_gap = fig.add_subplot(top[0, 1], sharey=ax_funnel)
    panel_funnel(ax_funnel)
    panel_gap(ax_gap, runs)
    panel_thicknesses(fig.add_subplot(grid[1, 0]), departures)

    out = WORKSPACE / "03_FIGURES_PUBLICATION"
    out.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300})):
        target = out / f"fig10_insitu_convergence{suffix}"
        fig.savefig(target, **kwargs)
        print(f"Saved: {target}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
