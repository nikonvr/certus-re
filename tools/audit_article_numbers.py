"""Recompute the numbers of the manuscript that no other script deposits.

Several claims in the text were, until now, produced by hand at the terminal: the fringe
contrast of the four angular models, what removing the process prior costs on each component,
the residual when each oxide is forced to a constant index, the collimated-beam test, and the
unpolarized average of Filter 2 against its specification. This script computes them all and
writes results/audit_numbers.json, so that every number in the paper has a file behind it.

    python tools/audit_article_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
RESULTS = PACKAGE / "results"
STUDIES = PACKAGE / "studies" / "volet2"
sys.path.insert(0, str(PACKAGE))

import numpy as np  # noqa: E402

import certus_re.forward as forward  # noqa: E402
from certus_re.dispersion import TabulatedIndex  # noqa: E402
from certus_re.forward import Evaluator  # noqa: E402
from certus_re.io_study import load_study  # noqa: E402
from certus_re.physics import aperture_nodes as exact_nodes  # noqa: E402
from certus_re.solve import invert  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aperture_quadrature_check import predict_with, round_pupil_nodes  # noqa: E402


def channel_rms(residuals) -> dict[str, float]:
    return {r.label: round(float(r.rms_final * 100.0), 4) for r in residuals}


# ---------------------------------------------------------------------------
# 1. Fringe contrast of the four angular models, against the measurement
# ---------------------------------------------------------------------------
def fringe_contrast() -> dict:
    study = load_study(STUDIES / "07_joint_campaign.json")
    ev = Evaluator(study)
    d = ev.nominal_thicknesses()
    collimated = np.zeros(study.instrument.n_bands, dtype=float)
    curves = {
        "two_rays": predict_with(ev, d, nodes=2, rule=exact_nodes),
        "converged_in_plane": predict_with(ev, d, nodes=21, rule=exact_nodes),
        "round_pupil": predict_with(ev, d, nodes=2, rule=round_pupil_nodes),
        "single_ray": predict_with(ev, d, nodes=2, rule=exact_nodes, aperture=collimated),
    }

    def contrast(wl, y, lo, hi):
        """Fringe amplitude: rms of the curve about a cubic local mean, in percentage points."""
        m = (wl >= lo) & (wl <= hi)
        x, v = wl[m], y[m]
        return 100.0 * float(np.std(v - np.polyval(np.polyfit(x, v, 3), x)))

    out = {"window_nm": [2100, 2530], "note": "rms fringe amplitude about a cubic local mean"}
    for i, plan in enumerate(ev.plans):
        if plan.sample.name != "BS17_alone":
            continue
        wl = plan.measurement.wavelength_nm
        key = f"BS17_{plan.measurement.polarization}"
        out[key] = {"measured": round(contrast(wl, plan.measurement.value, 2100, 2530), 3)}
        for name, c in curves.items():
            out[key][name] = round(contrast(wl, c[i], 2100, 2530), 3)
    return out


# ---------------------------------------------------------------------------
# 2. What removing the process prior costs, component by component
# ---------------------------------------------------------------------------
def prior_removed() -> dict:
    out = {}
    for stem, name in (("02_AR6_alone", "Filter 1"), ("04_BS45_resolved", "Filter 2"),
                       ("05_BS17_resolved", "Filter 3")):
        got = {}
        for tag, prior in (("with_prior", 0.5), ("no_prior", None)):
            study = load_study(STUDIES / f"{stem}.json")
            study.free.process_prior_pct = prior
            nominal = Evaluator(study).nominal_thicknesses()
            res = invert(study, start={k: v.copy() for k, v in nominal.items()})
            stack = next(iter(res.thicknesses))
            dep = 100.0 * (res.thicknesses[stack] - nominal[stack]) / nominal[stack]
            got[tag] = {"channels": channel_rms(res.residuals),
                        "departures_pct": [round(float(v), 2) for v in dep]}
        a = np.array(list(got["with_prior"]["channels"].values()))
        b = np.array(list(got["no_prior"]["channels"].values()))
        dep = np.array(got["no_prior"]["departures_pct"])
        j = int(np.argmax(np.abs(dep)))
        got["largest_residual_change_pp"] = round(float(np.max(np.abs(a - b))), 4)
        got["worst_layer"] = {"index": j + 1, "departure_pct": float(dep[j])}
        if j > 0:
            got["worst_layer"]["neighbour_below_pct"] = float(dep[j - 1])
        out[name] = got
    return out


# ---------------------------------------------------------------------------
# 3. Each oxide forced to a constant index, against the dispersive reference
# ---------------------------------------------------------------------------
def dispersionless() -> dict:
    from insitu_multistart_noprior import get_3components_study

    def joint(res):
        r = np.array([x.rms_final * 100 for x in res.residuals])
        n = np.array([x.n_points for x in res.residuals], dtype=float)
        return (round(float(np.sqrt(np.mean(r ** 2))), 4),
                round(float(np.sqrt((n * r ** 2).sum() / n.sum())), 4))

    study = get_3components_study()
    study.free.process_prior_pct = 0.5
    ref = invert(study, start=Evaluator(study).nominal_thicknesses())
    ref_chan, ref_pt = joint(ref)

    study = get_3components_study()
    wl = np.linspace(1000.0, 5000.0, 401)
    flat = {}
    for m in ("SiO2", "Nb2O5"):
        n0 = float(study.materials[m].n_at(np.array([1500.0]))[0])
        flat[m] = round(n0, 5)
        study.materials[m] = TabulatedIndex(
            wavelength_nm=wl, n=np.full_like(wl, n0), k=np.zeros_like(wl),
            name=f"{m}_flat", uncertainty_n=np.full_like(wl, 0.01),
            valid_range_nm=(1000.0, 5000.0))
    study.free.process_prior_pct = 0.5
    res = invert(study, start=Evaluator(study).nominal_thicknesses())
    chan, pt = joint(res)
    return {"constant_index_value": flat,
            "reference_dispersion": {"channel_mean_pct": ref_chan, "point_weighted_pct": ref_pt},
            "materials_forced_flat": {"channel_mean_pct": chan, "point_weighted_pct": pt}}


# ---------------------------------------------------------------------------
# 4. Filter 3 with the beam cone dropped
# ---------------------------------------------------------------------------
def collimated() -> dict:
    out = {}
    for tag, aperture in (("two_ray", None), ("collimated", 0.0)):
        study = load_study(STUDIES / "05_BS17_resolved.json")
        if aperture is not None:
            study.instrument.beam_aperture_deg = 0.0
        nominal = Evaluator(study).nominal_thicknesses()
        res = invert(study, start={k: v.copy() for k, v in nominal.items()})
        stack = next(iter(res.thicknesses))
        dep = 100.0 * (res.thicknesses[stack] - nominal[stack]) / nominal[stack]
        out[tag] = {"channels": channel_rms(res.residuals),
                    "departure_rms_pct": round(float(np.sqrt(np.mean(dep ** 2))), 3),
                    "departure_max_abs_pct": round(float(np.max(np.abs(dep))), 3)}
    return out


# ---------------------------------------------------------------------------
# 5. Filter 2, unpolarized average against its specification
# ---------------------------------------------------------------------------
def unpolarized_average() -> dict:
    study = load_study(STUDIES / "04_BS45_resolved.json")
    nominal = Evaluator(study).nominal_thicknesses()
    res = invert(study, start={k: v.copy() for k, v in nominal.items()})
    ev = Evaluator(study)
    pred = ev.predict(res.thicknesses)
    wl = ev.plans[0].measurement.wavelength_nm
    ra = 0.5 * (np.asarray(pred[0]) + np.asarray(pred[1])) * 100.0
    band = (wl >= 2000.0) & (wl <= 4000.0)
    return {"band_nm": [2000, 4000],
            "R_a_min_pct": round(float(ra[band].min()), 2),
            "R_a_max_pct": round(float(ra[band].max()), 2),
            "spec_pct": [48.0, 52.0],
            "misses_low_by_pp": round(48.0 - float(ra[band].min()), 2),
            "misses_high_by_pp": round(float(ra[band].max()) - 52.0, 2)}


# ---------------------------------------------------------------------------
# 6. How much of the Filter 3 residual survives averaging the two polarizations
# ---------------------------------------------------------------------------
def residual_common_mode() -> dict:
    study = load_study(STUDIES / "05_BS17_resolved.json")
    nominal = Evaluator(study).nominal_thicknesses()
    res = invert(study, start={k: v.copy() for k, v in nominal.items()})
    ev = Evaluator(study)
    pred = ev.predict(res.thicknesses)
    rs = np.asarray(pred[0]) - ev.plans[0].measurement.value
    rp = np.asarray(pred[1]) - ev.plans[1].measurement.value
    # rs = c + d, rp = c - d, so rs^2 + rp^2 = 2 c^2 + 2 d^2 and the split is exact.
    c, d = 0.5 * (rs + rp), 0.5 * (rs - rp)
    total = float(np.mean(rs ** 2) + np.mean(rp ** 2))
    return {"common_mode_share_of_power": round(2 * float(np.mean(c ** 2)) / total, 3),
            "differential_share_of_power": round(2 * float(np.mean(d ** 2)) / total, 3),
            "correlation_s_p": round(float(np.corrcoef(rs, rp)[0, 1]), 3)}


# ---------------------------------------------------------------------------
# 7. How far the spline solution moves the thicknesses, against fixed indices
# ---------------------------------------------------------------------------
def spline_thickness_shift() -> dict:
    from insitu_multistart_noprior import get_3components_study

    study = get_3components_study()
    study.free.process_prior_pct = 0.5
    fixed = invert(study, start=Evaluator(study).nominal_thicknesses())

    study = get_3components_study()
    study.free.process_prior_pct = 0.5
    study.free.index_correction = "bounded"
    study.free.index_tube_delta = 0.08
    study.free.index_n_knots = 3
    spline = invert(study, start=Evaluator(study).nominal_thicknesses())

    shifts = []
    for stack in ("AR6", "BS45", "BS17"):
        a = np.asarray(fixed.thicknesses[stack], dtype=float)
        b = np.asarray(spline.thicknesses[stack], dtype=float)
        shifts.append(100.0 * (b - a) / a)
    shifts = np.concatenate(shifts)
    return {"mean_abs_pct": round(float(np.mean(np.abs(shifts))), 3),
            "rms_pct": round(float(np.sqrt(np.mean(shifts ** 2))), 3),
            "max_abs_pct": round(float(np.max(np.abs(shifts))), 3)}


# ---------------------------------------------------------------------------
# 8. Does the Filter 3 residual follow the level, or the spectral derivative?
# ---------------------------------------------------------------------------
def residual_shape() -> dict:
    """A wavelength-scale error shifts the spectrum, so its residual follows dR/dlambda.

    A photometric or contrast error scales with R itself. Correlating the residual with each
    tells the two apart, and decides whether the discrepancy sits in the position of the
    fringes or in their level.
    """
    study = load_study(STUDIES / "05_BS17_resolved.json")
    nominal = Evaluator(study).nominal_thicknesses()
    res = invert(study, start={k: v.copy() for k, v in nominal.items()})
    ev = Evaluator(study)
    pred = ev.predict(res.thicknesses)

    out = {}
    for i, plan in enumerate(ev.plans):
        wl = plan.measurement.wavelength_nm
        model = np.asarray(pred[i])
        r = model - plan.measurement.value
        out[f"BS17_{plan.measurement.polarization}"] = {
            "corr_with_reflectance": round(float(np.corrcoef(r, model)[0, 1]), 3),
            "corr_with_spectral_derivative":
                round(float(np.corrcoef(r, np.gradient(model, wl))[0, 1]), 3),
        }
    return out


def main() -> int:
    out = {
        "description": "Numbers quoted in the manuscript that no other deposited script "
                       "produces. Written by tools/audit_article_numbers.py.",
        "fringe_contrast": fringe_contrast(),
        "process_prior_removed": prior_removed(),
        "dispersionless_control": dispersionless(),
        "collimated_beam": collimated(),
        "filter2_unpolarized_average": unpolarized_average(),
        "filter3_residual_common_mode": residual_common_mode(),
        "spline_thickness_shift": spline_thickness_shift(),
        "filter3_residual_shape": residual_shape(),
    }
    target = RESULTS / "audit_numbers.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
