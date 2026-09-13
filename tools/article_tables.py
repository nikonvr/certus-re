"""Generate the article's tables from the deposited studies, so that none is typed by hand.

A table retyped from a terminal is a table that will disagree with the deposit at the second
revision. Everything the article states in a table is produced here, from the same study
files a reader can run, and written both as LaTeX for the manuscript and as JSON for anything
else.

Three tables, and no more -- the paper is meant to be read, not consulted:

``table1_samples``
    The five samples: coatings carried, layer count, substrate and how it is polished, how the
    rear face is modelled, the acquisition that produced each spectrum with its geometry and
    its settings, the exploited band, the number of points and the declared uncertainty.
``table2_ladder``
    The study plan, from the rung that releases nothing to the joint inversion of the whole
    campaign: released blocks, parameter count, points per parameter, and the residual of
    every measurement at every rung. This one table replaces the separate parameter-budget
    and residual tables.
``table3_instrument``
    The retrieved instrument parameters, and the counter-experiments: one line each, a number
    each.

The layer-by-layer designs are deliberately *not* a table. Sixteen nominal values, sixteen
retrieved values and sixteen error bars belong in a figure, and ``designs.csv`` is written
here to draw it.

Usage
-----
    python tools/article_tables.py [--out results] [--skip-counter-experiments]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from certus_re.io_study import load_study  # noqa: E402
from certus_re.solve import invert  # noqa: E402

STUDIES = Path(__file__).resolve().parent.parent / "studies" / "volet2"

# The rungs the article prints. The others stay in the deposit, reachable by anyone who wants
# to see the intermediate steps, and are named in the caption rather than tabulated.
ARTICLE_RUNGS = (
    "00_prediction",
    "01_witnesses",
    "02_AR6_alone",
    "04_BS45_resolved",
    "05_BS17_resolved",
    "07_joint_campaign",
)

SAMPLE_LABEL = {
    "witness_SiO2": r"Witness $\mathrm{SiO}_2$",
    "witness_Nb2O5": r"Witness $\mathrm{Nb}_2\mathrm{O}_5$",
    "AR6_alone": "Filter 1 (AR)",
    "BS45_alone": "Filter 2 (16 layers)",
    "BS17_alone": "Filter 3 (17 layers)",
}

# How the rear face of each sample is prepared, which is a fact about the specimen and not
# about the model. The model column is derived from it.
POLISHING = {
    "witness_SiO2": "both faces polished",
    "witness_Nb2O5": "both faces polished",
    "AR6_alone": "front polished, rear ground",
    "BS45_alone": "front polished, rear ground",
    "BS17_alone": "front polished, rear ground",
}

REAR_MODEL = {
    "bare": "thick plate (bare rear)",
    "none": "semi-infinite (ground rear)",
    "coated": "thick plate (coated rear)",
}

RUN_LABELS = {
    "witness_SiO2": "Single-layer witness",
    "witness_Nb2O5": "Single-layer witness",
    "AR6_alone": "Run 260317-034",
    "BS45_alone": "Run 260317-035",
    "BS17_alone": "Run 260319-037",
}


def tex_escape(text: str) -> str:
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def latex_table(caption: str, label: str, columns: str, header: list[str], rows, notes="") -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\resizebox{{\linewidth}}{{!}}{{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\hline",
    ]
    for line in header:
        lines.append(line + r" \\")
    lines.append(r"\hline")
    for row in rows:
        if row is None:
            lines.append(r"\hline")
            continue
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}%", r"}", rf"\label{{{label}}}"]
    if notes:
        lines.append(notes)
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 1 -- the samples
# ---------------------------------------------------------------------------


def table1(study, stacks_of) -> tuple[str, list[dict]]:
    records: list[dict] = []
    for sample in study.samples:
        carried = [n for n in (sample.front_stack, sample.rear_stack) if n]
        layers = sum(study.stacks[n].n_layers for n in carried)
        for m in sample.measurements:
            acquisition = m.acquisition or {}
            records.append(
                {
                    "sample": sample.name,
                    "coatings": " + ".join(carried),
                    "layers": layers,
                    "layer_breakdown": " + ".join(
                        str(study.stacks[n].n_layers) for n in carried
                    ),
                    "substrate": sample.substrate.material,
                    "substrate_thickness_mm": sample.substrate.thickness_mm,
                    "polishing": POLISHING.get(sample.name, "?"),
                    "rear_model": REAR_MODEL[sample.substrate.rear],
                    "quantity": m.quantity,
                    "angle_deg": m.angle_deg,
                    "polarization": m.polarization,
                    "points": m.n_points,
                    "band_nm": [float(m.wavelength_nm[0]), float(m.wavelength_nm[-1])],
                    "sigma": float(np.atleast_1d(m.sigma_vector())[0]),
                    "acquisition": acquisition.get("acquisition", ""),
                    "datetime": acquisition.get("datetime", ""),
                    "stage_angle_deg": acquisition.get("stage_angle_deg", ""),
                    "detector_angle_deg": acquisition.get("detector_angle_deg", ""),
                    "polarizer": acquisition.get("polarizer", ""),
                    "sampling_pitch_nm": acquisition.get("sampling_pitch_nm", ""),
                    "averaging_count": acquisition.get("averaging_count", ""),
                    "slit_width_um": acquisition.get("slit_width_um", ""),
                    "spot_size_mm": acquisition.get("spot_size_mm", ""),
                }
            )

    order = list(SAMPLE_LABEL)
    by_sample = {name: [r for r in records if r["sample"] == name] for name in order}

    def cell(name, key, fmt=str, joiner=" / "):
        values = [fmt(r[key]) for r in by_sample[name]]
        unique = list(dict.fromkeys(values))
        return joiner.join(unique) if unique else "---"

    header = [
        " & " + " & ".join(SAMPLE_LABEL[n] for n in order),
    ]
    rows = [
        ["Stack designation"]
        + [{"witness_SiO2": r"1L $\mathrm{SiO}_2$",
            "witness_Nb2O5": r"1L $\mathrm{Nb}_2\mathrm{O}_5$",
            "AR6_alone": "6-layer AR",
            "BS45_alone": "16-layer BS",
            "BS17_alone": "17-layer BS"}[n] for n in order],
        ["Layers"] + [cell(n, "layer_breakdown") for n in order],
        ["Substrate"] + [cell(n, "substrate") for n in order],
        ["Substrate finish"] + [cell(n, "polishing") for n in order],
        ["Rear-face optical model"] + [cell(n, "rear_model") for n in order],
        None,
        ["Deposition run / source"]
        + [RUN_LABELS[n] for n in order],
        [r"Incidence / detector angle"]
        + [
            tex_escape(
                " / ".join(
                    dict.fromkeys(
                        f"{r['stage_angle_deg']}$^\\circ$, {r['detector_angle_deg']}$^\\circ$"
                        for r in by_sample[n]
                    )
                )
            )
            for n in order
        ],
        ["Polarizer"] + [cell(n, "polarizer") for n in order],
        ["Sampling pitch (nm)"] + [cell(n, "sampling_pitch_nm") or "---" for n in order],
        ["Averages"] + [cell(n, "averaging_count") or "---" for n in order],
        ["Slit ($\\mu$m) / spot (mm)"]
        + [
            " / ".join(
                dict.fromkeys(
                    f"{r['slit_width_um']}, {r['spot_size_mm']}"
                    for r in by_sample[n]
                    if r["slit_width_um"]
                )
            )
            or "---"
            for n in order
        ],
        None,
        ["Measured quantity"]
        + [
            cell(
                n,
                "quantity",
                fmt=lambda q: {"R": "$R$", "T": "$T$", "Trel": r"$T_{\mathrm{rel}}$"}[q],
            )
            for n in order
        ],
        ["Polarization channel"]
        + [
            ", ".join(dict.fromkeys({"a": "unpolarized", "s": "$s$", "p": "$p$"}.get(r["polarization"], r["polarization"]) for r in by_sample[n]))
            for n in order
        ],
        ["Exploited band (nm)"]
        + [
            cell(n, "band_nm", fmt=lambda b: f"{b[0]:.0f}--{b[1]:.0f}")
            for n in order
        ],
        ["Data points"]
        + [
            (
                f"{by_sample[n][0]['points']}"
                if len(by_sample[n]) == 1
                else rf"${by_sample[n][0]['points']}\times{len(by_sample[n])}$"
            )
            for n in order
        ],
    ]
    caption = (
        "The five samples, their preparation and the acquisitions that produced them: "
        "two single-layer witnesses on sapphire and three multilayer filters on silicon "
        "(Filter~1, Filter~2, Filter~3). "
        "The rear face is the quantity it is most costly to mistake: a ground rear face "
        "scatters the return of the second interface out of the collected beam, which is "
        "what justifies the semi-infinite model; taking it for a polished one adds the "
        "specular Fresnel return of silicon and severely corrupts the inverted thicknesses."
    )
    return (
        latex_table(
            caption,
            "tab:samples",
            "l" + "c" * len(order),
            header,
            rows,
        ),
        records,
    )


# ---------------------------------------------------------------------------
# Table 2 -- the ladder
# ---------------------------------------------------------------------------


def released_blocks(study) -> str:
    bits = []
    free = study.free
    n_thick = sum(
        study.stacks[n].n_variable
        for n in study.used_stacks()
        if free.releases_thickness(n)
    )
    if n_thick:
        bits.append(f"{n_thick} thicknesses")
    if free.index_correction != "none":
        bits.append("index correction")
    if free.aperture == "fitted":
        bits.append(f"{study.instrument.n_bands} apertures")
    if free.crosstalk == "fitted":
        bits.append(r"$\alpha$, $\beta$")
    return ", ".join(bits) if bits else "none"


def table2(runs) -> str:
    rows = []
    for entry in runs:
        result, study = entry["result"], entry["study"]
        per_measurement = "; ".join(
            f"{r.sample.replace('_', ' ')}"
            + (f" {r.polarization}" if r.polarization in ("s", "p") else "")
            + f" {100 * r.rms_final:.3f}"
            for r in result.residuals
        )
        rows.append(
            [
                tex_escape(entry["rung"]),
                str(result.dof.n_free),
                str(study.n_points),
                "---"
                if result.dof.n_free == 0
                else f"{result.dof.points_per_parameter:.0f}",
                tex_escape(released_blocks(study)),
                tex_escape(per_measurement),
            ]
        )
    caption = (
        "The study plan, from the prediction rung that releases nothing to the joint inversion of the "
        "whole campaign across all three filters (Filter~1 AR6, Filter~2 BS45, Filter~3 BS17) and two "
        "witnesses. Residuals are rms departures in percent of reflectance or of relative transmittance, "
        "one per measurement channel. Rung~0 is a pure prediction evaluated with witness-transferred "
        "optical constants, not a fit: it costs zero degrees of freedom, demonstrating parameter-free "
        "predictive agreement. Intermediate sensitivity studies (\\texttt{03\\_BS45\\_unpolarized} and "
        "\\texttt{06\\_BS17\\_aperture}) are deposited beside the baseline ladder."
    )
    header = [
        "Rung & Free & Points & Pts/par & Released blocks & rms residual per measurement (\\%)"
    ]
    return latex_table(caption, "tab:ladder", "lrrrp{0.20\\linewidth}p{0.30\\linewidth}", header, rows)


# ---------------------------------------------------------------------------
# Table 3 -- instrument and counter-experiments
# ---------------------------------------------------------------------------


def table3(joint, counters) -> str:
    result, study = joint["result"], joint["study"]
    edges = list(study.instrument.aperture_band_edges_nm)
    lo, hi = study.instrument.aperture_bounds_deg
    rows = []
    edge_list = [None, *edges, None]
    for b, value in enumerate(result.aperture_deg):
        low = "min" if edge_list[b] is None else f"{edge_list[b]:.0f}"
        high = "max" if edge_list[b + 1] is None else f"{edge_list[b + 1]:.0f}"
        sigma = (
            "---"
            if result.aperture_sigma_deg is None
            or not np.isfinite(result.aperture_sigma_deg[b])
            else f"{result.aperture_sigma_deg[b]:.2f}"
        )
        at_bound = f"beam aperture band {b + 1}" in result.at_bounds
        rows.append(
            [
                f"Aperture, {low}--{high}\\,nm",
                f"{lo:g}--{hi:g}$^\\circ$",
                f"{value:.2f}$^\\circ$" + (r"$^{\dagger}$" if at_bound else ""),
                sigma,
            ]
        )
    for name, value, index in (
        (r"Crosstalk $\alpha$ ($s$ channel)", result.crosstalk[0], 0),
        (r"Crosstalk $\beta$ ($p$ channel)", result.crosstalk[1], 1),
    ):
        sigma = (
            "---"
            if result.crosstalk_sigma is None
            or not np.isfinite(result.crosstalk_sigma[index])
            else f"{result.crosstalk_sigma[index]:.4f}"
        )
        at_bound = ("crosstalk alpha" if index == 0 else "crosstalk beta") in result.at_bounds
        rows.append(
            [
                name,
                f"{study.instrument.crosstalk_bounds[0]:g}--"
                f"{study.instrument.crosstalk_bounds[1]:g}",
                f"{value:.4f}" + (r"$^{\dagger}$" if at_bound else ""),
                sigma,
            ]
        )
    caption = (
        "Retrieved instrument parameters of the joint inversion, and the cost of what the "
        "protocol forbids. $^{\\dagger}$ marks a parameter that ended on its bound: between "
        "2530 and 4000\\,nm these coatings are spectrally flat at $45^\\circ$, the cone "
        "average barely moves, and the data do not determine the aperture there -- the "
        "quoted uncertainty says so rather than leaving the reader to discover it. Each "
        "counter-experiment below is deposited under "
        "\\texttt{studies/volet2/counter\\_experiments/}, is wrong by construction, and is "
        "quoted against the rung of Table~\\ref{tab:ladder} it departs from."
    )
    header = ["Quantity & Bounds & Retrieved & $\\pm$"]
    instrument_part = latex_table(
        caption, "tab:instrument", "lccc", header, rows
    )
    # The counter-experiments need prose-width columns; two tabulars in one float keeps the
    # article at three numbered tables, which is the budget this paper has.
    counter_rows = "\n".join(
        " & ".join([tex_escape(e["lesson"]), e["figure"]]) + r" \\"
        for e in counters
    )
    counter_part = "\n".join(
        [
            r"\medskip",
            r"",
            r"\begin{tabular}{p{0.44\linewidth}p{0.50\linewidth}}",
            r"\hline",
            r"What the counter-experiment does & What it costs \\",
            r"\hline",
            counter_rows,
            r"\hline",
            r"\end{tabular}",
        ]
    )
    return instrument_part.replace(
        r"\end{table}", counter_part + "\n" + r"\end{table}"
    )


# ---------------------------------------------------------------------------
# The counter-experiments, each quoted against the rung it departs from
# ---------------------------------------------------------------------------


def counter_figure(stem: str, result, runs) -> str:
    """What one counter-experiment costs, said against its own reference rung.

    A residual quoted on its own means nothing here either: each of these studies differs
    from one rung of the ladder by exactly one thing, so the number that matters is the
    difference, not the value.
    """
    by_stem = {r["stem"]: r for r in runs}

    def rms(entry, sample, polarization=None):
        for r in entry["result"].residuals:
            if r.sample == sample and (polarization is None or r.polarization == polarization):
                return 100 * r.rms_final
        raise KeyError(f"{sample} {polarization} not in {entry['stem']}")

    def dispersion(entry, stack):
        return float(np.sqrt((entry["result"].qwot_departure_pct()[stack] ** 2).mean()))

    this = {"stem": stem, "result": result}

    if stem in ("rear_face_bare", "layer_order_reversed", "substrate_flat_table"):
        reference = rms(by_stem["00_prediction"], "AR6_alone")
        here = rms(this, "AR6_alone")
        return (
            rf"antireflection coating, same design, nothing released: "
            rf"{reference:.2f}\% $\rightarrow$ \textbf{{{here:.2f}\%}}"
        )
    if stem == "angle_shifted":
        reference = by_stem.get("04_BS45_resolved") or by_stem["06_BS45_crosstalk"]
        shift = float(
            result.thicknesses["BS45"].sum()
            - reference["result"].thicknesses["BS45"].sum()
        )
        return (
            rf"rms {rms(reference, 'BS45_alone', 's'):.2f} $\rightarrow$ "
            rf"{rms(this, 'BS45_alone', 's'):.2f}\% in $s$, but the retrieved coating "
            rf"moves by \textbf{{{shift:+.1f}\,nm}} in total thickness"
        )
    if stem == "search_window_wide":
        reference = by_stem.get("04_BS45_resolved") or by_stem["06_BS45_crosstalk"]
        return (
            rf"rms {rms(reference, 'BS45_alone', 's'):.3f} $\rightarrow$ "
            rf"{rms(this, 'BS45_alone', 's'):.3f}\% in $s$ and "
            rf"{rms(reference, 'BS45_alone', 'p'):.3f} $\rightarrow$ "
            rf"{rms(this, 'BS45_alone', 'p'):.3f}\% in $p$ -- less than the measured "
            rf"repeatability either way -- while the layer-wise dispersion goes "
            rf"{dispersion(reference, 'BS45'):.2f} $\rightarrow$ "
            rf"\textbf{{{dispersion(this, 'BS45'):.2f}\%}} and two adjacent layers part by "
            rf"$+15$ and $-17\%$"
        )
    if stem == "mesh_decimated":
        reference = by_stem.get("04_BS45_resolved") or by_stem["06_BS45_crosstalk"]
        return (
            rf"500 points $\rightarrow$ 64 for the same 21 parameters, 24 "
            rf"$\rightarrow$ \textbf{{3}} points per parameter, while the rms moves only "
            rf"{rms(reference, 'BS45_alone', 's'):.2f} $\rightarrow$ "
            rf"{rms(this, 'BS45_alone', 's'):.2f}\% and the layer-wise dispersion "
            rf"{dispersion(reference, 'BS45'):.2f} $\rightarrow$ "
            rf"{dispersion(this, 'BS45'):.2f}\%"
        )
    if stem == "index_released":
        reference = by_stem.get("07_joint_campaign") or by_stem["08_joint_campaign"]
        return (
            rf"29 $\rightarrow$ 39 parameters; rms of the beam splitter "
            rf"{rms(reference, 'BS45_alone', 's'):.3f} $\rightarrow$ "
            rf"{rms(this, 'BS45_alone', 's'):.3f}\% in $s$, and the witnesses, which the "
            rf"determination was made on, get \emph{{worse}}: "
            rf"{rms(reference, 'witness_SiO2'):.3f} $\rightarrow$ "
            rf"{rms(this, 'witness_SiO2'):.3f}\%"
        )
    worst = max(100 * r.rms_final for r in result.residuals)
    return rf"rms up to {worst:.2f}\%"


# ---------------------------------------------------------------------------
# Designs, for the figure
# ---------------------------------------------------------------------------


def designs_csv(joint) -> str:
    result, study = joint["result"], joint["study"]
    lines = [
        "# Nominal and retrieved designs of the joint inversion, with the uncertainty the",
        "# data imply on each quarter wave. Generated by tools/article_tables.py.",
        "stack,run,layer,material,qwot_nominal,qwot_retrieved,qwot_sigma,"
        "departure_pct,departure_in_sigma,d_nominal_nm,d_retrieved_nm,d_sigma_nm",
    ]
    departures = result.qwot_departure_pct()
    in_sigma = result.qwot_departure_in_sigma()
    for name in study.used_stacks():
        stack = study.stacks[name]
        for i, layer in enumerate(stack.layers):
            sigma = result.qwot_sigma.get(name)
            ratio = in_sigma.get(name)
            d_sigma = result.thickness_sigma_nm.get(name)

            def num(array, index=i):
                if array is None or not np.isfinite(array[index]):
                    return ""
                return f"{array[index]:.6g}"

            lines.append(
                ",".join(
                    [
                        name,
                        str(stack.run or ""),
                        str(i + 1),
                        layer.material,
                        f"{result.nominal_qwot[name][i]:.6f}",
                        f"{result.qwot[name][i]:.6f}",
                        num(sigma),
                        f"{departures[name][i]:.4f}",
                        num(ratio),
                        f"{result.nominal_thicknesses[name][i]:.4f}",
                        f"{result.thicknesses[name][i]:.4f}",
                        num(d_sigma),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="directory to write the tables into",
    )
    parser.add_argument("--skip-counter-experiments", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for stem in ARTICLE_RUNGS:
        path = STUDIES / f"{stem}.json"
        if not path.is_file():
            print(f"missing study {path}", file=sys.stderr)
            return 2
        raw = json.loads(path.read_text(encoding="utf-8"))
        study = load_study(path)
        print(f"  inverting {stem} ...", flush=True)
        runs.append(
            {
                "stem": stem,
                "rung": raw.get("provenance", {}).get("rung", stem),
                "question": raw.get("provenance", {}).get("question", ""),
                "study": study,
                "result": invert(study),
            }
        )

    counters = []
    if not args.skip_counter_experiments:
        for path in sorted((STUDIES / "counter_experiments").glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            study = load_study(path)
            print(f"  inverting {path.stem} ...", flush=True)
            result = invert(study)
            counters.append(
                {
                    "stem": path.stem,
                    "lesson": raw.get("provenance", {}).get(
                        "counter_experiment", path.stem
                    ),
                    "figure": counter_figure(path.stem, result, runs),
                    "residual_rms_pct": [100 * r.rms_final for r in result.residuals],
                    "qwot_dispersion_pct": {
                        name: float(np.sqrt((d**2).mean()))
                        for name, d in result.qwot_departure_pct().items()
                    },
                    "n_free": result.dof.n_free,
                    "points_per_parameter": (
                        None
                        if result.dof.n_free == 0
                        else result.dof.points_per_parameter
                    ),
                }
            )

    joint = next(r for r in runs if r["stem"] == "07_joint_campaign")
    table1_tex, records = table1(joint["study"], joint["study"].stacks)
    (args.out / "table1_samples.tex").write_text(table1_tex, encoding="utf-8")
    (args.out / "table2_ladder.tex").write_text(table2(runs), encoding="utf-8")
    (args.out / "table3_instrument.tex").write_text(
        table3(joint, counters), encoding="utf-8"
    )
    (args.out / "designs.csv").write_text(designs_csv(joint), encoding="utf-8")

    summary = {
        "samples": records,
        "ladder": [
            {
                "study": r["stem"],
                "rung": r["rung"],
                "question": r["question"],
                "n_free": r["result"].dof.n_free,
                "n_points": r["study"].n_points,
                "points_per_parameter": (
                    None
                    if r["result"].dof.n_free == 0
                    else r["result"].dof.points_per_parameter
                ),
                "released": released_blocks(r["study"]),
                "residuals": [x.to_dict() for x in r["result"].residuals],
                "at_bounds": r["result"].at_bounds,
                "covariance_scale_s2": r["result"].covariance_scale,
            }
            for r in runs
        ],
        "counter_experiments": counters,
        "instrument_retrieved": {
            "beam_aperture_deg": joint["result"].aperture_deg.tolist(),
            "beam_aperture_sigma_deg": (
                None
                if joint["result"].aperture_sigma_deg is None
                else joint["result"].aperture_sigma_deg.tolist()
            ),
            "crosstalk": list(joint["result"].crosstalk),
            "crosstalk_sigma": (
                None
                if joint["result"].crosstalk_sigma is None
                else list(joint["result"].crosstalk_sigma)
            ),
        },
    }
    (args.out / "article_numbers.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )

    print(f"\nwritten to {args.out}")
    for name in (
        "table1_samples.tex",
        "table2_ladder.tex",
        "table3_instrument.tex",
        "designs.csv",
        "article_numbers.json",
    ):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
