"""Generate the four publication figures from the deposited studies and results.

Run ``python tools/article_tables.py`` and ``python tools/search_window_scan.py`` first,
then run this script from the package root.  Every figure is written as vector PDF for the
manuscript and as a 300 dpi PNG for quick inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from certus_re.forward import Evaluator  # noqa: E402
from certus_re.io_study import load_study  # noqa: E402
from certus_re.solve import invert  # noqa: E402


PACKAGE = Path(__file__).resolve().parent.parent
PROJECT = PACKAGE.parent
RESULTS = PACKAGE / "results"
JOINT_STUDY = PACKAGE / "studies" / "volet2" / "07_joint_campaign.json"

BLUE = "#2F5D8C"
ORANGE = "#D9772B"
GREEN = "#3A7D44"
PURPLE = "#76528B"
RED = "#B44A4A"
GREY = "#666666"
LIGHT = "#EEF3F7"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def box(ax, xy, width, height, title, body, *, face=LIGHT, edge=BLUE, body_size=7.3) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.70, title, ha="center", va="center", weight="bold")
    ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center", fontsize=body_size)


def arrow(ax, start, end, *, color=GREY, style="-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=1.0,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def figure_workflow(out: Path) -> None:
    """What the user declares, what the solver does, what it returns.

    Three columns rather than one tall stack, because the two questions a reader actually has
    are "what do I have to provide?" and "what do I get back?".  Only the centre column carries
    arrows between its boxes: the inputs are a declaration and the outputs are a report, and
    neither is a sequence.  Box heights follow the number of body lines and each column is laid
    out from its own top, so adding a line to a step cannot make it overflow its frame.
    """
    # Drawn at the width it is printed, so the type size here is the type size on the page.
    HEAD, TITLE, BODY = 7.6, 6.4, 5.9
    LINE, PAD, GAP = 0.042, 0.019, 0.030
    AX0, AW = 0.000, 0.240
    BX0, BW = 0.325, 0.385
    CX0, CW = 0.760, 0.240
    XLOOP = 0.301
    TOP = 0.895

    declared = [
        ("measured spectra",
         [r"$R$ or $T$ | angle | polarization",
          r"noise floor $\sigma$"]),
        ("nominal design",
         [r"in quarter waves at",
          r"$\lambda_0 = 1500$ nm, as deposited"]),
        ("optical constants",
         [r"$n(\lambda)$, $k(\lambda)$ of the reference",
          r"single layers: fixed, or",
          r"released in a declared tube"]),
        ("substrate, rear face",
         [r"rear face semi-infinite,",
          r"bare, or coated"]),
        ("instrument",
         [r"beam cone half-angle $h(\lambda)$,",
          r"from a preset or by hand"]),
        ("process prior",
         [r"$\sigma_{\mathrm{proc}} = 0.5\%$ in optical",
          r"thickness, from the plant"]),
    ]

    solver = [
        ("load and check", "step",
         [r"refuses a band the index tables do not",
          r"cover, or a block nothing constrains"]),
        ("forward model, per spectrum", "step",
         [r"$N = n - ik$, layer 1 on the substrate",
          r"tilted admittances $\eta$ | beam cone $h(\lambda)$",
          r"rear face summed incoherently"]),
        ("objective", "core",
         [r"$\Phi = \sum_s N_s \left(\frac{\mathrm{RMSE}_s}{\sigma_s}\right)^2 + \sum_j \left("
          r"\frac{\mathrm{OT}_j - \mathrm{OT}_j^{\mathrm{nom}}}"
          r"{\mathrm{OT}_j^{\mathrm{nom}}\,\sigma_{\mathrm{proc}}}\right)^2$",
          "",
          r"the spectra, plus what the plant allows",
          r"no box bounds; only $\mathrm{OT}_j > 0$ is enforced"]),
        ("trust-region step", "step",
         [r"deterministic: one start at the nominal",
          r"design, no restart, no perturbation"]),
        ("at the minimum", "exit",
         [r"covariance $\frac{m}{m-n}\mathrm{RMSE}^2 (J^{\mathsf{T}}J)^{-1}$",
          r"by SVD, defined for every layer"]),
    ]

    obtained = [
        ("retrieved thicknesses",
         [r"optical and physical, each",
          r"with its own $1\sigma$"]),
        ("departure from design",
         [r"in percent, and in units of $\sigma$"]),
        ("residual per spectrum",
         [r"RMSE, and how many $\sigma$ that is"]),
        ("fitted spectra",
         [r"measured, nominal design",
          r"and inverted, on one axis"]),
        ("index corrections",
         [r"$\Delta n(\lambda)$ per material,",
          r"only if you released them"]),
        ("parameter budget",
         [r"free parameters, points,",
          r"points per parameter"]),
    ]

    def height(body):
        return 2 * PAD + LINE * (1 + len(body))

    def stack(items, body_of):
        tops, y = [], TOP
        for it in items:
            h = height(body_of(it))
            y -= h
            tops.append((y, h))
            y -= GAP
        return tops, y + GAP

    a_tops, a_end = stack(declared, lambda it: it[1])
    b_tops, b_end = stack(solver, lambda it: it[2])
    c_tops, c_end = stack(obtained, lambda it: it[1])
    floor = min(a_end, b_end, c_end)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(floor - 0.055, 1.0)
    ax.axis("off")

    def panel(x, w, top, h, title, body, face, edge):
        ax.add_patch(FancyBboxPatch(
            (x, top), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=0.9, edgecolor=edge, facecolor=face, zorder=3))
        ax.text(x + w / 2, top + h - PAD - LINE * 0.42, title, ha="center", va="center",
                weight="bold", fontsize=TITLE, zorder=4)
        for j, line in enumerate(body):
            if not line:
                continue
            ax.text(x + w / 2, top + h - PAD - LINE * (1.34 + j), line, ha="center",
                    va="center", fontsize=BODY, zorder=4)

    for (title, body), (top, h) in zip(declared, a_tops):
        panel(AX0, AW, top, h, title, body, "#FAF4EA", ORANGE)
    for (title, kind, body), (top, h) in zip(solver, b_tops):
        face = {"step": "#EAF2F8", "core": "#DCE9F5", "exit": "#EDF5ED"}[kind]
        edge = {"step": BLUE, "core": BLUE, "exit": GREEN}[kind]
        panel(BX0, BW, top, h, title, body, face, edge)
        if kind == "core":
            ax.add_patch(FancyBboxPatch(
                (BX0 - 0.007, top - 0.007), BW + 0.014, h + 0.014,
                boxstyle="round,pad=0.002,rounding_size=0.010",
                linewidth=1.3, edgecolor=BLUE, facecolor="none", zorder=2))
    for (title, body), (top, h) in zip(obtained, c_tops):
        panel(CX0, CW, top, h, title, body, "#EDF5ED", GREEN)

    # Arrows only inside the solver column: that is the only part that is a sequence.
    for i in range(len(b_tops) - 1):
        ax.add_patch(FancyArrowPatch(
            (BX0 + BW / 2, b_tops[i][0]), (BX0 + BW / 2, b_tops[i + 1][0] + b_tops[i + 1][1]),
            arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=GREY, zorder=5))

    # The iteration: forward model, objective and step repeat until convergence.
    lo = b_tops[3][0] + 0.012
    hi = b_tops[1][0] + b_tops[1][1] - 0.012
    ax.add_patch(FancyArrowPatch(
        (XLOOP, lo), (XLOOP, hi), arrowstyle="-|>", mutation_scale=8,
        linewidth=1.0, color=BLUE, zorder=5))
    ax.text(XLOOP - 0.013, (lo + hi) / 2, "not converged", rotation=90, ha="center",
            va="center", fontsize=5.6, color=BLUE)

    def bus(x, tops, x_from, x_to, color, gather, y_reach=None):
        """A vertical rail beside a column, gathering it into one arrow or fanning out of it.

        ``y_reach`` extends the rail so the horizontal feed meets it instead of stopping short.
        """
        lo_rail = tops[-1][0] + 0.012
        hi_rail = tops[0][0] + tops[0][1] - 0.012
        if y_reach is not None:
            lo_rail, hi_rail = min(lo_rail, y_reach), max(hi_rail, y_reach)
        ax.plot([x, x], [lo_rail, hi_rail], color=color, linewidth=1.0, zorder=6)
        for top, h in tops:
            a, b = (x_from, x) if gather else (x, x_to)
            ax.add_patch(FancyArrowPatch(
                (a, top + h / 2), (b, top + h / 2), arrowstyle="-|>" if not gather else "-",
                mutation_scale=7, linewidth=0.9, color=color, zorder=6))

    y_in = b_tops[0][0] + b_tops[0][1] / 2
    y_out = b_tops[-1][0] + b_tops[-1][1] / 2

    bus(AX0 + AW + 0.020, a_tops, AX0 + AW + 0.004, None, ORANGE, True)
    ax.add_patch(FancyArrowPatch((AX0 + AW + 0.020, y_in), (BX0 - 0.006, y_in),
                                 arrowstyle="-|>", mutation_scale=9, linewidth=1.2,
                                 color=ORANGE, zorder=7))
    bus(CX0 - 0.020, c_tops, None, CX0 - 0.004, GREEN, False, y_reach=y_out)
    ax.add_patch(FancyArrowPatch((BX0 + BW + 0.006, y_out), (CX0 - 0.020, y_out),
                                 arrowstyle="-", mutation_scale=9, linewidth=1.2,
                                 color=GREEN, zorder=7))

    for x, w, label, color in ((AX0, AW, "1. what you declare", ORANGE),
                               (BX0, BW, "2. what certus_re does", BLUE),
                               (CX0, CW, "3. what you obtain", GREEN)):
        ax.text(x + w / 2, TOP + 0.028, label, ha="center", va="bottom",
                fontsize=HEAD, weight="bold", color=color)

    ax.text(AX0 + AW / 2, a_end - 0.014,
            "one JSON study file,\nevery path relative to it",
            ha="center", va="top", fontsize=5.6, color=GREY, style="italic")
    save(fig, out, "fig1_algorithm")


def fitted_spectra():
    """Measured, the nominal design's own response, and the inversion.

    The nominal curve is the one the article calls rung 0: the design as it left the coating
    shop, the transferred optical constants, and not one released parameter. The distance from
    it to the measurement is what the deposition run actually did; the distance from the
    inversion to the measurement is what is left over afterwards.
    """
    study = load_study(JOINT_STUDY)
    result = invert(study)
    evaluator = Evaluator(study)
    nominal = evaluator.predict(evaluator.nominal_thicknesses())
    predicted = evaluator.predict(
        result.thicknesses,
        aperture_deg=result.aperture_deg,
        crosstalk=result.crosstalk,
    )
    return evaluator.plans, predicted, nominal


def spectra_entries():
    plans, predicted, nominal = fitted_spectra()
    entries = []
    for plan, fit, nom in zip(plans, predicted, nominal):
        entries.append(
            {
                "sample": plan.sample.name,
                "pol": plan.measurement.polarization,
                "w": plan.measurement.wavelength_nm / 1000.0,
                "measured": plan.measurement.value,
                "fit": fit,
                "nominal": nom,
            }
        )
    return entries


COMPONENTS = [
    dict(
        stem="fig3_spectra_AR6",
        sample="AR6_alone",
        channels=[("a", GREEN, "unpolarized")],
        unpolarized=False,
        title=r"Filter 1: Six-layer antireflection coating, $R$ at $8^\circ$",
        target=dict(band=(3.0, 4.0), lo=0.0, hi=1.0, label=r"specification $R<1\%$"),
        ylim=(-3, 68),
        zoom=dict(span=(2.5, 3.3), ylim=(-0.4, 8.0),
                  note=r"Filter 1 detail, 2.5--3.3 $\mu$m"),
    ),
    dict(
        stem="fig4_spectra_BS45",
        sample="BS45_alone",
        channels=[("s", BLUE, "$s$"), ("p", ORANGE, "$p$")],
        unpolarized=True,
        title=r"Filter 2: Sixteen-layer beam splitter, $R$ at $45^\circ$",
        target=dict(band=(2.0, 4.0), lo=48.0, hi=52.0,
                    label=r"specification $48<R_a<52\%$"),
        ylim=(25, 102),
        zoom=dict(span=(2.0, 4.0), ylim=(46.0, 54.0), only_unpolarized=True,
                  note=r"Filter 2 unpolarized average, 2--4 $\mu$m"),
    ),
    dict(
        stem="fig5_spectra_BS17",
        sample="BS17_alone",
        channels=[("s", BLUE, "$s$"), ("p", ORANGE, "$p$")],
        unpolarized=False,
        title=r"Filter 3: Seventeen-layer beam splitter, $R$ at $45^\circ$",
        target=None,
        ylim=(-3, 103),
        zoom=dict(span=(2.1, 2.53), ylim=None,
                  note=r"Filter 3 detail, 2.10--2.53 $\mu$m"),
    ),
]


def _draw_target(ax, target, label=False):
    if target is None:
        return
    ax.fill_between(target["band"], target["lo"], target["hi"], color=RED, alpha=0.13,
                    lw=0, zorder=0)
    for level in (target["lo"], target["hi"]):
        ax.plot(target["band"], [level] * 2, color=RED, lw=0.9, ls="--", zorder=1)
    if label:
        ax.plot([], [], color=RED, lw=0.9, ls="--", label=target["label"])


def figure_spectra(out: Path) -> None:
    """One figure per component: the full spectrum with its directly attached residual panel
    (sharing the identical wavelength axis 1.5--4.0 um), followed by a dedicated detail panel
    with its own explicit scale."""
    entries = spectra_entries()
    for spec in COMPONENTS:
        selected = [e for e in entries if e["sample"] == spec["sample"]]
        if not selected:
            continue
        zoom = spec["zoom"]
        fig = plt.figure(figsize=(7.15, 5.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.6], hspace=0.35)
        gs_top = gs[0].subgridspec(2, 1, height_ratios=[2.7, 0.9], hspace=0.04)
        ax = fig.add_subplot(gs_top[0])
        ar = fig.add_subplot(gs_top[1], sharex=ax)
        az = fig.add_subplot(gs[1])

        def channel_curves(panel, span=None, faint=False):
            for pol, colour, label in spec["channels"]:
                e = next((x for x in selected if x["pol"] == pol), None)
                if e is None:
                    continue
                k = slice(None) if span is None else (
                    (e["w"] >= span[0]) & (e["w"] <= span[1]))
                panel.plot(e["w"][k], 100 * e["nominal"][k], color=GREY, lw=0.85,
                           ls=(0, (3, 2)), alpha=0.95, zorder=2,
                           label=("nominal design" if (pol in ("a", "s") and not faint) else None))
                panel.plot(e["w"][k], 100 * e["measured"][k], linestyle="none", marker="o",
                           ms=2.1, mfc="none", mew=0.55, color=colour, alpha=0.85, zorder=3,
                           label=(f"{label}, measured" if not faint else None))
                panel.plot(e["w"][k], 100 * e["fit"][k], color=colour, lw=1.35, zorder=4,
                           label=(f"{label}, retrieved" if not faint else None))

        def unpolarized_curves(panel, span=None, faint=False):
            es = next((x for x in selected if x["pol"] == "s"), None)
            ep = next((x for x in selected if x["pol"] == "p"), None)
            if es is None or ep is None:
                return
            k = slice(None) if span is None else ((es["w"] >= span[0]) & (es["w"] <= span[1]))
            panel.plot(es["w"][k], 50 * (es["nominal"][k] + ep["nominal"][k]), color=GREY,
                       lw=0.85, ls=(0, (3, 2)), zorder=2,
                       label=("nominal design" if faint else None))
            panel.plot(es["w"][k], 50 * (es["measured"][k] + ep["measured"][k]),
                       linestyle="none", marker="s", ms=1.9, mfc="none", mew=0.6,
                       color=PURPLE, alpha=0.9, zorder=5, label=r"$R_a$, measured")
            panel.plot(es["w"][k], 50 * (es["fit"][k] + ep["fit"][k]), color=PURPLE, lw=1.5,
                       zorder=6, label=r"$R_a$, retrieved")

        _draw_target(ax, spec["target"], label=True)
        channel_curves(ax)
        if spec["unpolarized"]:
            unpolarized_curves(ax)
        ax.set_title(spec["title"], loc="left", pad=3)
        ax.set_ylabel(r"$R$ (%)")
        ax.set_ylim(*spec["ylim"])
        ax.set_xlim(1.5, 4.0)
        ax.grid(color="#E4E4E4", lw=0.5)
        ax.tick_params(direction="in", top=True, right=True, labelbottom=False)
        ax.legend(frameon=False, ncol=3, fontsize=6.8, loc="best", handlelength=1.6,
                  columnspacing=1.1)

        # Highlight detail span in top panel
        ax.axvspan(zoom["span"][0], zoom["span"][1], color="#E8EEF5", alpha=0.55, zorder=0)

        for pol, colour, _ in spec["channels"]:
            e = next((x for x in selected if x["pol"] == pol), None)
            if e is not None:
                ar.plot(e["w"], 100 * (e["fit"] - e["measured"]), color=colour, lw=0.8)
        ar.axhline(0, color="black", lw=0.6)
        ar.set_ylabel("resid.\n(pts)", fontsize=7.0)
        ar.set_xlabel(r"Wavelength ($\mu$m)", fontsize=7.8)
        ar.set_xlim(1.5, 4.0)
        ar.grid(color="#E4E4E4", lw=0.5)
        ar.tick_params(direction="in", top=True, right=True, labelsize=7.2)

        for edge in (2.53, 3.70):
            for panel in (ax, ar):
                panel.axvline(edge, color="#BBBBBB", lw=0.55, ls=":", zorder=0)

        # Detail panel
        _draw_target(az, spec["target"])
        if zoom.get("only_unpolarized"):
            unpolarized_curves(az, zoom["span"], faint=True)
        else:
            channel_curves(az, zoom["span"], faint=True)
        az.set_facecolor("#FAFCFF")
        az.set_title(zoom["note"], loc="left", pad=2, fontsize=7.4)
        az.set_ylabel(r"$R$ (%)")
        az.set_xlabel(r"Wavelength ($\mu$m)", fontsize=7.8)
        az.set_xlim(*zoom["span"])
        if zoom["ylim"] is not None:
            az.set_ylim(*zoom["ylim"])
        az.grid(color="#E4E4E4", lw=0.5)
        az.tick_params(direction="in", top=True, right=True, labelsize=7.2)

        save(fig, out, spec["stem"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = (line for line in handle if not line.startswith("#"))
        return list(csv.DictReader(rows))


def figure_designs(out: Path) -> None:
    records = read_csv(RESULTS / "designs.csv")
    fig, axes = plt.subplots(3, 1, figsize=(7.15, 6.4), gridspec_kw={"height_ratios": [0.78, 1.18, 1.22], "hspace": 0.34})
    for ax, stack, title in zip(axes, ("AR6", "BS45", "BS17"), ("(a) Filter 1 (six-layer antireflection coating)", "(b) Filter 2 (sixteen-layer beam splitter)", "(c) Filter 3 (seventeen-layer beam splitter)")):
        rows = [r for r in records if r["stack"] == stack]
        x = np.asarray([int(r["layer"]) for r in rows])
        y = np.asarray([float(r["departure_pct"]) for r in rows])
        err = np.asarray([
            np.nan if not r["qwot_sigma"].strip()
            else 100 * float(r["qwot_sigma"]) / float(r["qwot_nominal"])
            for r in rows
        ])
        colors = [BLUE if r["material"] == "Nb2O5" else ORANGE for r in rows]
        ax.bar(x, y, color=colors, width=0.72, alpha=0.88, zorder=2)
        finite = np.isfinite(err)
        ax.errorbar(x[finite], y[finite], yerr=err[finite], fmt="none", ecolor="black",
                    elinewidth=0.75, capsize=2.2, zorder=3)
        ax.set_ylim(-7.0, 7.0)
        ax.axhline(0, color="black", lw=0.65, zorder=1)
        ax.set_title(title, loc="left", pad=3)
        ax.tick_params(labelsize=7.2)
        ax.set_xticks(x)
        ax.set_xlim(0.3, x[-1] + 0.7)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
        ax.tick_params(direction="in", top=True, right=True)
    axes[-1].set_xlabel("Layer number (substrate to air)")
    fig.supylabel(r"Departure from nominal optical thickness $\mathrm{OT}$ (%)", fontsize=8.5, x=0.045)
    legend = [
        Patch(facecolor=BLUE, label=r"Nb$_2$O$_5$"),
        Patch(facecolor=ORANGE, label=r"SiO$_2$"),
        Line2D([0], [0], color="black", marker="|", markersize=9, lw=0,
               label=r"local $1\sigma$ uncertainty"),
    ]
    axes[0].legend(handles=legend, ncol=3, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 1.18), handlelength=1.4, columnspacing=1.5,
                   fontsize=7.2)
    save(fig, out, "fig6_departures")


def figure_zooms(out: Path) -> None:
    """Three regions where the comparison actually decides something.

    (a) The antireflection coating around 2800 nm. The design slide of the campaign names this
        bump "le juge de paix des erreurs" -- the arbiter of errors -- because it is where a
        thickness error first becomes visible. It is drawn here against the specification.
    (b) The sixteen-layer beam splitter across its 48-52 % specification.
    (c) The seventeen-layer component between 2100 and 2530 nm, where its measured fringe
        contrast falls to 0.75 of the computed value and where its residual is largest.
    """
    entries = spectra_entries()
    panels = [
        ("AR6_alone", "a", (2.45, 3.25), r"(a) AR, the bump at 2.8\,$\mu$m",
         dict(band=(3.0, 3.25), lo=0.0, hi=1.0)),
        ("BS45_alone", "s", (2.0, 3.2), r"(b) BS 16 layers, across the 48--52\% band",
         dict(band=(2.0, 3.2), lo=48.0, hi=52.0)),
        ("BS17_alone", "s", (2.1, 2.53), r"(c) BS 17 layers, where the contrast is lost", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.2), gridspec_kw={"wspace": 0.26})
    for ax, (sample, pol, span, title, spec) in zip(axes, panels):
        e = next((x for x in entries if x["sample"] == sample and x["pol"] == pol), None)
        if e is None:
            continue
        k = (e["w"] >= span[0]) & (e["w"] <= span[1])
        if spec is not None:
            ax.fill_between(spec["band"], spec["lo"], spec["hi"], color=RED, alpha=0.13, lw=0,
                            zorder=0)
            for level in (spec["lo"], spec["hi"]):
                ax.plot(spec["band"], [level] * 2, color=RED, lw=0.8, ls="--", zorder=1)
        ax.plot(e["w"][k], 100 * e["nominal"][k], color=GREY, lw=0.9, ls=(0, (3, 2)),
                label="nominal design")
        ax.plot(e["w"][k], 100 * e["measured"][k], linestyle="none", marker="o", ms=2.6,
                mfc="none", mew=0.7, color=BLUE, label="measured")
        ax.plot(e["w"][k], 100 * e["fit"][k], color=ORANGE, lw=1.2, label="reconstructed")
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel(r"Wavelength ($\mu$m)")
        ax.grid(color="#DDDDDD", lw=0.5)
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"$R$ (%)")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.5, fontsize=6.4)
    save(fig, out, "fig8_zooms")


STACK_TITLES = {
    "AR6": "(a) Filter 1 (antireflection coating, 6 layers)",
    "BS45": "(b) Filter 2 (beam splitter, 16 layers)",
    "BS17": "(c) Filter 3 (beam splitter, 17 layers)",
}


def figure_nominal_designs(out: Path) -> None:
    """The three components as they were designed, drawn to scale.

    Each stack is shown as deposited, substrate on the left and air on the right, with the
    physical thickness of every layer to scale. This is the object the article is about, and
    it is worth one look before any residual is quoted: the seventeen-layer component carries
    a single thick layer of 2.7 um (10.6 quarter waves) -- which is what makes it both
    the hardest component to deposit and the easiest one to invert.
    """
    records = read_csv(RESULTS / "designs.csv")
    stacks = ("AR6", "BS45", "BS17")
    fig, axes = plt.subplots(
        3, 1, figsize=(7.15, 4.3), gridspec_kw={"hspace": 0.62}
    )
    widest = max(
        sum(float(r["d_nominal_nm"]) for r in records if r["stack"] == s) for s in stacks
    )
    for ax, stack in zip(axes, stacks):
        rows = [r for r in records if r["stack"] == stack]
        left = 0.0
        for row in rows:
            width = float(row["d_nominal_nm"])
            colour = BLUE if row["material"] == "Nb2O5" else ORANGE
            ax.barh(0, width, left=left, height=0.78, color=colour,
                    edgecolor="white", linewidth=0.7, zorder=2)
            # Label only the layers wide enough to carry a number without overlapping.
            if width > 0.030 * widest:
                ax.text(left + width / 2, 0, f"{width:.0f}", ha="center", va="center",
                        fontsize=6.2, color="white", zorder=3)
            left += width
        ax.set_title(f"{STACK_TITLES[stack]}, {left:.0f} nm in all", loc="left", pad=4)
        ax.set_xlim(-0.012 * widest, widest * 1.012)
        ax.set_ylim(-0.62, 0.62)
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(direction="in")
        ax.text(-0.008 * widest, 0, "Si", ha="right", va="center", fontsize=7.5)
        ax.text(left + 0.008 * widest, 0, "air", ha="left", va="center", fontsize=7.5)
    axes[-1].set_xlabel("Physical thickness from the substrate (nm)")
    legend = [
        Patch(facecolor=BLUE, label=r"Nb$_2$O$_5$ (high index)"),
        Patch(facecolor=ORANGE, label=r"SiO$_2$ (low index)"),
    ]
    axes[0].legend(handles=legend, ncol=2, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 1.62), handlelength=1.3, columnspacing=1.6)
    save(fig, out, "fig2_nominal_designs")


def figure_aperture(out: Path) -> None:
    """Why the beam cone matters, and how it is determined.

    Panel (a) is the effect itself: the same design computed with a collimated beam and with
    the 2-degree cone the instrument actually delivers. Panel (b) is the determination -- the
    aperture imposed and the layer thicknesses RE-INVERTED at each value, so that the curve
    shows what the aperture is worth once the thicknesses have had every chance to absorb it.
    The two components answer differently, and that is the point of the figure.
    """
    study = load_study(PACKAGE / "studies" / "volet2" / "05_BS17_resolved.json")
    study.instrument.aperture_per_band_deg = None
    study.instrument.aperture_band_edges_nm = ()
    study.instrument.beam_aperture_deg = 1e-6
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    collimated = evaluator.predict(nominal)

    study.instrument.beam_aperture_deg = 2.0
    evaluator = Evaluator(study)
    cone = evaluator.predict(nominal)
    grid = evaluator.plans[0].measurement.wavelength_nm

    fig, (ax, aw) = plt.subplots(
        1, 2, figsize=(7.15, 3.0),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.30},
    )

    ax.plot(grid / 1000, 100 * collimated[0], color=GREY, lw=0.9,
            label=r"collimated ($0^\circ$)")
    ax.plot(grid / 1000, 100 * cone[0], color=BLUE, lw=0.9,
            label=r"$2^\circ$ cone")
    ax.set_title("(a) Effect of the beam cone", loc="left")
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(r"$R_s$ of the 17-layer design (%)")
    ax.set_xlim(1.2, 2.2)
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    worst = float(np.max(np.abs(100 * (cone[0] - collimated[0]))))
    where = float(grid[int(np.argmax(np.abs(cone[0] - collimated[0])))])
    ax.annotate(
        rf"up to {worst:.1f} points, at {where:.0f}\,nm",
        xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7.0, color=RED,
    )

    for stem, label, colour in (
        ("aperture_04_BS45_resolved", "16 layers", ORANGE),
        ("aperture_05_BS17_resolved", "17 layers", BLUE),
    ):
        path = RESULTS / f"{stem}.csv"
        if not path.is_file():
            continue
        rows = read_csv(path)
        h = np.asarray([float(r["aperture_deg"]) for r in rows])
        rms = np.asarray([float(r["rms_inverted_pct"]) for r in rows])
        aw.plot(h, rms - rms.min(), color=colour, lw=1.15, marker="o", ms=2.4,
                label=f"{label}, thicknesses re-inverted")
    aw.axhline(0.136, color=GREEN, lw=0.8, ls="--")
    aw.text(0.06, 0.145, "one measured repeatability", fontsize=6.8, color=GREEN)
    aw.set_title("(b) Sensitivity to the imposed half-angle", loc="left")
    aw.set_xlabel(r"Imposed beam half-angle (deg)")
    aw.set_ylabel("rms above its own minimum (points)")
    aw.set_ylim(-0.02, 0.9)
    aw.grid(color="#DDDDDD", lw=0.5)
    aw.tick_params(direction="in", top=True, right=True)
    aw.legend(frameon=False, loc="upper center", handlelength=1.6)
    save(fig, out, "fig5_aperture")


def figure_aperture_scan(out: Path) -> None:
    """Plot the stored aperture scans without evaluating or re-inverting the optical model."""
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for stem, label, colour, marker in (
        ("aperture_04_BS45_resolved", "16-layer beam splitter", ORANGE, "o"),
        ("aperture_05_BS17_resolved", "17-layer beam splitter", BLUE, "s"),
    ):
        rows = read_csv(RESULTS / f"{stem}.csv")
        h = np.asarray([float(r["aperture_deg"]) for r in rows])
        rms = np.asarray([float(r["rms_inverted_pct"]) for r in rows])
        ax.plot(h, rms - rms.min(), color=colour, lw=1.2, marker=marker, ms=3.0,
                label=label)
    ax.axhline(0.136, color=GREEN, lw=0.85, ls="--", label="reference repeatability")
    ax.axvline(2.0, color=GREY, lw=0.8, ls=":", label=r"imposed $2.0^\circ$")
    ax.set_xlabel("Imposed beam half-angle (deg)")
    ax.set_ylabel("rms above scan minimum (percentage points)")
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.02, 0.9)
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    save(fig, out, "fig8_aperture_scan")


def figure_uniqueness(out: Path) -> None:
    """Where many scattered starting points land, as the prior is loosened."""
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9),
                             gridspec_kw={"wspace": 0.26})
    for ax, stem, title in (
        (axes[0], "uniqueness_BS45", "(a) 16 layers, 500 points"),
        (axes[1], "uniqueness_BS17", "(b) 17 layers, 1000 points"),
    ):
        path = RESULTS / f"{stem}.json"
        if not path.is_file():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        windows = sorted({r["window"] for r in rows})
        for offset, window in enumerate(windows):
            group = [r for r in rows if r["window"] == window]
            best = min(r["rms"] for r in group)
            y = [100 * (r["rms"] - best) for r in group]
            x = np.full(len(y), offset) + np.linspace(-0.22, 0.22, len(y))
            reached = sum(1 for v in y if v < 0.1 * 0.136)
            ax.scatter(x, y, s=5.5, color=BLUE, alpha=0.65, edgecolors="none", zorder=2)
            ax.text(offset, 24.0, rf"{reached}/{len(y)}", ha="center", fontsize=7.0,
                    color=GREEN if reached == len(y) else RED)
        ax.axhline(0.136, color=GREEN, lw=0.8, ls="--")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(-0.03, 40)
        ax.set_xticks(range(len(windows)),
                      [rf"$\pm{100 * w:.0f}\%$" for w in windows])
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel("Search window on each thickness")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel("rms above the best residual (points)")
    save(fig, out, "fig6_uniqueness")


def figure_ladder_and_window(out: Path) -> None:
    summary = json.loads((RESULTS / "article_numbers.json").read_text(encoding="utf-8"))
    ladder = summary["ladder"]
    x = np.arange(len(ladder))
    labels = [
        "pred.\n0",
        "witness\n2",
        "AR6\n6",
        "BS45\n16",
        "BS17\n17",
        "joint\n41",
    ]
    keys = [
        ("witness_SiO2", "a", r"SiO$_2$ witness", "#777777", "o"),
        ("witness_Nb2O5", "a", r"Nb$_2$O$_5$ witness", "#AAAAAA", "s"),
        ("AR6_alone", "a", "AR6", GREEN, "D"),
        ("BS45_alone", "s", "BS45 s", BLUE, "^"),
        ("BS45_alone", "p", "BS45 p", "#6F9DC6", "v"),
        ("BS17_alone", "s", "BS17 s", RED, "P"),
        ("BS17_alone", "p", "BS17 p", ORANGE, "X"),
    ]

    fig, (ax, aw) = plt.subplots(1, 2, figsize=(7.15, 3.7), gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.34})
    for sample, pol, label, color, marker in keys:
        values = []
        for entry in ladder:
            found = next((r for r in entry["residuals"] if r["sample"] == sample and r["polarization"] == pol), None)
            values.append(np.nan if found is None else 100 * found["rms_final"])
        ax.plot(x, values, color=color, marker=marker, ms=4.0, lw=1.0, label=label)
    ax.set_title("(a) Residuals across the inversion sequence", loc="left")
    ax.set_ylabel("rms residual (percentage points)")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Study and number of released parameters")
    ax.set_xlim(-0.2, len(x) - 0.8)
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.4)
    ax.tick_params(direction="in", top=True, right=True)

    rows = read_csv(RESULTS / "search_window_07_joint_campaign.csv")
    window = np.asarray([float(r["window_pct"]) for r in rows])
    residual = np.asarray([float(r["rms_components_pct"]) for r in rows])
    disp_ar = np.asarray([float(r["dispersion_pct_AR6"]) for r in rows])
    disp_bs = np.asarray([float(r["dispersion_pct_BS45"]) for r in rows])
    aw.plot(window, residual, color="black", marker="o", ms=3.3, label="component residual")
    aw.set_xscale("log")
    aw.set_xticks([1, 2, 3, 5, 10, 20, 50], ["1", "2", "3", "5", "10", "20", "50"])
    aw.set_xlabel("Half-width of thickness window (%)")
    aw.set_ylabel("Combined component rms (%)")
    aw.set_ylim(0.50, 0.66)
    aw.axvline(3, color=GREEN, lw=0.9, ls="--")
    aw.text(3.15, 0.655, "retained", color=GREEN, va="top", fontsize=7.2)
    aw.grid(axis="y", color="#DDDDDD", lw=0.5)
    aw.tick_params(direction="in", top=True)
    ad = aw.twinx()
    ad.plot(window, disp_ar, color=GREEN, marker="s", ms=3.0, label="AR6 dispersion")
    ad.plot(window, disp_bs, color=BLUE, marker="^", ms=3.0, label="BS45 dispersion")
    ad.set_ylabel("Layer-wise QWOT dispersion (%)")
    ad.set_ylim(0, 7.0)
    ad.tick_params(direction="in", right=True)
    aw.set_title("(b) Sensitivity to the thickness search window", loc="left")
    handles = aw.get_lines()[:1] + ad.get_lines()
    aw.legend(handles=handles, labels=[h.get_label() for h in handles], frameon=False, loc="center right")
    save(fig, out, "fig7_ladder_and_window")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT / "03_FIGURES_PUBLICATION")
    parser.add_argument(
        "--extras",
        action="store_true",
        help="also draw the exploratory figures the manuscript does not carry "
        "(aperture, aperture scan, multistart, ladder and window)",
    )
    args = parser.parse_args(argv)
    required = [RESULTS / "designs.csv", RESULTS / "article_numbers.json", RESULTS / "search_window_07_joint_campaign.csv"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print("missing generated inputs:\n  " + "\n  ".join(map(str, missing)), file=sys.stderr)
        return 2

    set_style()
    figure_workflow(args.out)
    figure_nominal_designs(args.out)
    figure_spectra(args.out)
    figure_designs(args.out)
    stems = [
        "fig1_algorithm",
        "fig2_nominal_designs",
        "fig3_spectra_AR6",
        "fig4_spectra_BS45",
        "fig5_spectra_BS17",
        "fig6_departures",
    ]
    # The exploratory figures were dropped from the manuscript on 13 September; they
    # stay available behind a flag so the deposit can still reproduce them on request.
    if args.extras:
        figure_aperture(args.out)
        figure_aperture_scan(args.out)
        figure_uniqueness(args.out)
        figure_ladder_and_window(args.out)
        stems += ["fig5_aperture", "fig8_aperture_scan", "fig6_uniqueness",
                  "fig7_ladder_and_window"]
    print(f"written to {args.out}")
    for stem in stems:
        print(f"  {stem}.pdf")
        print(f"  {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
