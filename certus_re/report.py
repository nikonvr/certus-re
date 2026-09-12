"""Turning a run into something a reader can check.

A reverse-engineering result is only as good as what is reported with it. Three things are
printed next to every number, because without them the number cannot be judged:

* **what was released** -- the parameter budget, block by block, held blocks included;
* **what each measurement costs** -- one residual per measurement, not a single pooled
  figure, so that a dataset the model fails on cannot hide inside an average;
* **what the run leaned on** -- parameters that ended on their bound, dispersions queried
  outside the range they were determined over, measurements with no declared uncertainty.

Departures are expressed in quarter-wave optical thickness. It is the unit the design is
written in, the quantity optical monitoring controls, and it is invariant to first order
under the index-thickness degeneracy, so it is the only one in which two runs using
different dispersion datasets can be compared.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from .model import Study
from .solve import InversionResult

__all__ = ["text_report", "json_report"]


def _rule(width: int = 78) -> str:
    return "-" * width


def text_report(result: InversionResult, study: Study) -> str:
    """A complete run report, readable in a terminal and quotable in a manuscript."""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"certus_re -- {result.study_name}")
    add("=" * 78)
    if study.comment:
        add("")
        for chunk in _wrap(study.comment, 78):
            add(chunk)
    add("")
    add(f"samples          {len(study.samples)}")
    add(f"measurements     {sum(len(s.measurements) for s in study.samples)}")
    add(f"data points      {study.n_points}")
    add(f"reference        lambda0 = {study.lambda0_nm:.1f} nm")
    add(f"instrument       {study.instrument.name}")
    add("")

    # -- parameter budget --------------------------------------------------
    add("PARAMETERS RELEASED")
    add(_rule())
    add(result.dof.to_text())
    add("")

    # -- shared stacks -----------------------------------------------------
    shared = {
        name: [s.name for s in study.samples_using(name)]
        for name in study.used_stacks()
    }
    if any(len(users) > 1 for users in shared.values()):
        add("COATINGS SHARED BETWEEN SAMPLES")
        add(_rule())
        for name, users in shared.items():
            points = sum(s.n_points for s in study.samples_using(name))
            add(
                f"  {name:<10s} {study.stacks[name].n_layers:2d} layers, constrained by "
                f"{points:5d} points from {', '.join(users)}"
            )
        add("")
        add(
            "  A sample that names a coating already named by another adds measured points"
        )
        add("  and no unknown: its residual is a consistency test, not a fit.")
        add("")

    # -- residuals ---------------------------------------------------------
    add("RESIDUALS, PER MEASUREMENT")
    add(_rule())
    add(
        f"  {'sample':<20s} {'quantity':>8s} {'pol':>4s} {'points':>7s} "
        f"{'rms before':>11s} {'rms after':>10s} {'chi2/pt':>8s}"
    )
    for r in result.residuals:
        add(
            f"  {r.sample:<20s} {r.quantity:>8s} {r.polarization:>4s} {r.n_points:7d} "
            f"{100 * r.rms_initial:10.3f}% {100 * r.rms_final:9.3f}% "
            f"{r.chi2_per_point:8.2f}"
        )
    add("")
    add(
        f"  weighted residual over the whole study: "
        f"{result.rms_initial:.4f} -> {result.rms_final:.4f}"
    )
    add(
        "  chi2 per point near one means the model reproduces the data to within the"
    )
    add("  declared uncertainty; far above one means it does not.")
    add("")

    # -- retrieved coatings ------------------------------------------------
    departures = result.qwot_departure_pct()
    in_sigma = result.qwot_departure_in_sigma()
    add("RETRIEVED COATINGS")
    add(_rule(92))
    add(
        "  The uncertainty on each quarter wave is the square root of the diagonal of"
    )
    add(
        f"  s^2 (J^T J)^-1 at the solution, with s^2 = chi2 / (points - parameters) = "
        f"{result.covariance_scale:.3f},"
    )
    add(
        "  converted from nanometres by 4 n(lambda0) / lambda0. A departure smaller than its"
    )
    add("  own error bar is not a departure; the last column is the one to read.")
    for name in study.used_stacks():
        stack = study.stacks[name]
        qwot0 = result.nominal_qwot[name]
        qwot1 = result.qwot[name]
        d0 = result.nominal_thicknesses[name]
        d1 = result.thicknesses[name]
        departure = departures[name]
        sigma = result.qwot_sigma.get(name)
        ratio = in_sigma.get(name)
        total = 100.0 * (qwot1.sum() - qwot0.sum()) / qwot0.sum()
        add("")
        run = f", run {stack.run}" if stack.run else ""
        add(f"  {name}{run} -- {stack.n_layers} layers")
        add(
            f"  {'layer':>5s} {'material':<8s} {'QWOT nom':>9s} {'QWOT ret':>9s} "
            f"{'+-sigma':>9s} {'delta %':>8s} {'d/sigma':>8s} {'d nom/nm':>9s} "
            f"{'d ret/nm':>9s} {'delta nm':>9s}"
        )
        for i, layer in enumerate(stack.layers):
            s = "        -" if sigma is None or not np.isfinite(sigma[i]) else f"{sigma[i]:9.4f}"
            r = "       -" if ratio is None or not np.isfinite(ratio[i]) else f"{ratio[i]:+8.1f}"
            add(
                f"  {i + 1:5d} {layer.material:<8s} {qwot0[i]:9.4f} {qwot1[i]:9.4f} "
                f"{s} {departure[i]:+8.2f} {r} {d0[i]:9.2f} {d1[i]:9.2f} "
                f"{d1[i] - d0[i]:+9.2f}"
            )
        rms = float(np.sqrt((departure**2).mean()))
        add(
            f"  total QWOT departure {total:+.3f} %   "
            f"layer-wise rms {rms:.2f} %   max {np.abs(departure).max():.2f} %"
        )
        if sigma is not None and np.any(np.isfinite(sigma)):
            finite = sigma[np.isfinite(sigma)]
            add(
                f"  uncertainty on one quarter wave: median {np.median(finite):.4f}, "
                f"worst {finite.max():.4f}"
            )
        if ratio is not None and np.any(np.isfinite(ratio)):
            significant = int(np.count_nonzero(np.abs(ratio[np.isfinite(ratio)]) > 3.0))
            add(
                f"  layers whose departure exceeds three times its own uncertainty: "
                f"{significant} of {int(np.count_nonzero(np.isfinite(ratio)))}"
            )
        thinnest = int(np.argmin(qwot0))
        if int(np.argmax(np.abs(departure))) == thinnest:
            add(
                f"  the largest departure falls on layer {thinnest + 1}, which is the "
                f"thinnest of the stack"
            )
    add("")

    # -- instrument ---------------------------------------------------------
    released_aperture = study.free.aperture == "fitted"
    released_crosstalk = study.free.crosstalk == "fitted"
    if released_aperture or released_crosstalk:
        add("INSTRUMENT PARAMETERS RETRIEVED")
        add(_rule())
        if released_aperture:
            edges = list(study.instrument.aperture_band_edges_nm)
            lo, hi = study.instrument.aperture_bounds_deg
            add(
                f"  beam aperture, one total aperture per band, bounds {lo:g}-{hi:g} deg,"
            )
            add(
                "  step wavelengths imposed at "
                + (", ".join(f"{e:g}" for e in edges) or "none")
                + " nm:"
            )
            edge_list = [None, *edges, None]
            for b, value in enumerate(result.aperture_deg):
                low = "min" if edge_list[b] is None else f"{edge_list[b]:g}"
                high = "max" if edge_list[b + 1] is None else f"{edge_list[b + 1]:g}"
                s = (
                    ""
                    if result.aperture_sigma_deg is None
                    or not np.isfinite(result.aperture_sigma_deg[b])
                    else f" +- {result.aperture_sigma_deg[b]:.3f}"
                )
                add(f"    band {b + 1}  {low:>5s} - {high:<5s} nm   {value:.3f} deg{s}")
        if released_crosstalk:
            alpha, beta = result.crosstalk
            sa, sb = (
                ("", "")
                if result.crosstalk_sigma is None
                else (
                    f" +- {result.crosstalk_sigma[0]:.4f}",
                    f" +- {result.crosstalk_sigma[1]:.4f}",
                )
            )
            add("")
            add(f"  polarizer crosstalk   alpha = {alpha:.4f}{sa}   beta = {beta:.4f}{sb}")
            add(
                f"  applied to the computed spectra as Rs_meas = (1-alpha) Rs + alpha Rp"
            )
            add("  and Rp_meas = (1-beta) Rp + beta Rs; the measurements are left as recorded.")
            add(
                f"  alpha - beta = {alpha - beta:+.4f}: it is this difference, not the "
                f"leakage itself,"
            )
            add(
                "  that an unpolarized measurement could have seen, the half-sum being"
            )
            add("  rigorously invariant when the two are equal.")
        add("")

    # -- index corrections -------------------------------------------------
    if result.index_corrections:
        add("INDEX CORRECTION RELEASED")
        add(_rule())
        for material, knots in result.index_corrections.items():
            add(
                f"  {material:<8s} max |delta n| = {np.abs(knots).max():.5f} "
                f"(tube half-width {study.free.index_tube_delta:.5f})"
            )
        add(
            "  A run that moves the index cannot be used to conclude that the index was"
        )
        add("  correct. This block is reported so that conclusion is not drawn.")
        add("")

    # -- what the run leaned on -------------------------------------------
    warnings: list[str] = []
    if result.at_bounds:
        warnings.append(
            "parameters that ended on their bound (the search window, not the data, is "
            "what stopped them, and the uncertainty quoted for them is meaningless): "
            + ", ".join(result.at_bounds)
        )
    if result.window_override:
        warnings.append(result.window_override)
    if result.covariance_note:
        warnings.append(result.covariance_note)
    if np.isfinite(result.covariance_scale) and result.covariance_scale > 4.0:
        warnings.append(
            f"chi2 per degree of freedom is {result.covariance_scale:.1f}: the model does "
            f"not reproduce the data to within the declared photometric uncertainty, and "
            f"the error bars above are widened by that factor rather than trusted as they "
            f"stand"
        )
    for report in result.extrapolation:
        beyond = report.get("beyond_validated_range")
        detail = (
            f"queried {report['queried_range_nm'][0]:.0f}-"
            f"{report['queried_range_nm'][1]:.0f} nm against a table covering "
            f"{report['tabulated_range_nm'][0]:.0f}-"
            f"{report['tabulated_range_nm'][1]:.0f} nm"
        )
        if report["points_outside_table"]:
            warnings.append(
                f"dispersion of {report['material']}: {detail}; "
                f"{report['points_outside_table']} evaluations fell outside and were held "
                f"at the edge value"
            )
        elif beyond:
            warnings.append(
                f"dispersion of {report['material']}: queried up to "
                f"{beyond['queried_range_nm'][1]:.0f} nm, beyond the "
                f"{beyond['validated_range_nm'][1]:.0f} nm up to which the determination "
                f"is stated to be constrained"
            )
    undeclared = [
        f"{s.name}/{m.label or m.quantity}"
        for s in study.samples
        for m in s.measurements
        if m.sigma is None
    ]
    if undeclared and len(study.samples) > 1:
        warnings.append(
            "measurements without a declared uncertainty in a joint inversion: "
            + ", ".join(undeclared)
        )
    if not result.success:
        warnings.append(f"the optimiser did not converge: {result.message}")

    add("WHAT THIS RUN LEANED ON")
    add(_rule())
    if warnings:
        for w in warnings:
            for k, chunk in enumerate(_wrap(w, 74)):
                add(("  - " if k == 0 else "    ") + chunk)
    else:
        add("  Nothing outside its declared scope.")
    add("")

    add("REPRODUCIBILITY")
    add(_rule())
    add(
        "  The inversion is a trust-region least squares started from the nominal design."
    )
    add(
        "  It contains no random element: no restarts, no shakes, no seeded initialisation."
    )
    add(f"  Function evaluations {result.n_function_evaluations}, {result.message}")
    add("")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out or [""]


def json_report(result: InversionResult, study: Study) -> dict:
    """The same content, machine-readable, for figures and downstream checks."""
    departures = result.qwot_departure_pct()
    in_sigma = result.qwot_departure_in_sigma()

    def _finite(values) -> list:
        """None rather than NaN: a JSON reader should not have to guess what nan means."""
        return [None if not np.isfinite(v) else float(v) for v in np.atleast_1d(values)]

    return {
        "study": result.study_name,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lambda0_nm": study.lambda0_nm,
        "instrument": {
            "name": study.instrument.name,
            "beam_aperture_deg": study.instrument.aperture_values_deg().tolist(),
            "aperture_band_edges_nm": list(study.instrument.aperture_band_edges_nm),
            "aperture_mode": study.instrument.aperture_mode,
            "n_aperture_nodes": study.instrument.n_aperture_nodes,
            "crosstalk_mode": study.instrument.crosstalk_mode,
        },
        "instrument_retrieved": {
            "beam_aperture_deg": result.aperture_deg.tolist(),
            "beam_aperture_sigma_deg": (
                None
                if result.aperture_sigma_deg is None
                else _finite(result.aperture_sigma_deg)
            ),
            "crosstalk_alpha": result.crosstalk[0],
            "crosstalk_beta": result.crosstalk[1],
            "crosstalk_sigma": (
                None if result.crosstalk_sigma is None else _finite(result.crosstalk_sigma)
            ),
        },
        "free_parameters": result.dof.to_dict(),
        "shared_stacks": {
            name: [s.name for s in study.samples_using(name)]
            for name in study.used_stacks()
        },
        "residuals": [r.to_dict() for r in result.residuals],
        "weighted_residual": {
            "initial": result.rms_initial,
            "final": result.rms_final,
            "chi2_per_point": result.chi2_per_point,
        },
        "stacks": {
            name: {
                "run": study.stacks[name].run,
                "materials": study.stacks[name].materials,
                "qwot_nominal": result.nominal_qwot[name].tolist(),
                "qwot_retrieved": result.qwot[name].tolist(),
                "qwot_departure_pct": departures[name].tolist(),
                "qwot_departure_total_pct": float(
                    100.0
                    * (result.qwot[name].sum() - result.nominal_qwot[name].sum())
                    / result.nominal_qwot[name].sum()
                ),
                "qwot_departure_rms_pct": float(np.sqrt((departures[name] ** 2).mean())),
                "qwot_departure_max_pct": float(np.abs(departures[name]).max()),
                "qwot_sigma": _finite(result.qwot_sigma.get(name, [])),
                "qwot_departure_in_sigma": _finite(in_sigma.get(name, [])),
                "thickness_nominal_nm": result.nominal_thicknesses[name].tolist(),
                "thickness_retrieved_nm": result.thicknesses[name].tolist(),
                "thickness_sigma_nm": _finite(result.thickness_sigma_nm.get(name, [])),
            }
            for name in study.used_stacks()
        },
        "index_corrections": {
            material: knots.tolist()
            for material, knots in result.index_corrections.items()
        },
        "diagnostics": {
            "at_bounds": result.at_bounds,
            "extrapolation": result.extrapolation,
            "converged": result.success,
            "message": result.message,
            "n_function_evaluations": result.n_function_evaluations,
            "covariance_scale_s2": (
                None
                if not np.isfinite(result.covariance_scale)
                else float(result.covariance_scale)
            ),
            "covariance_note": result.covariance_note,
            "deterministic": True,
        },
    }


def write_json_report(path, result: InversionResult, study: Study) -> None:
    """Write :func:`json_report` to ``path``."""
    from pathlib import Path

    Path(path).write_text(
        json.dumps(json_report(result, study), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
