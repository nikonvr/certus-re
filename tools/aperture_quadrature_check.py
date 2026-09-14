"""How good is the two-ray beam-aperture model? Measured, not asserted.

Section 3 of the article makes two claims about the forward model: that averaging over the two
extreme rays reproduces a full two-dimensional pupil quadrature to better than a stated tolerance
in reflectance, and that below ten degrees the angular spread can be ignored. Both are cheap to
check on the real spectra, so they are checked here rather than asserted.

Four angular-averaging models are compared, all at the declared half-angle:

* **single**      -- one central ray, the collimated assumption;
* **two rays**    -- the baseline of the article: theta0 +/- h, equal weights;
* **in plane**    -- Gauss-Legendre over the same interval with many nodes, that is the exact
                     in-plane average;
* **round pupil** -- a uniformly illuminated circular pupil of half-angle h, integrated over its
                     area. The in-plane tilt moves the angle of incidence directly; the
                     out-of-plane tilt enters only through the projection, at second order.

The pupil model is imposed by substituting the quadrature rule the forward model calls, so the
optics, the dispersion and the spectra are identical across the four cases and only the angular
weighting differs.

    python tools/aperture_quadrature_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
RESULTS = PACKAGE / "results"
sys.path.insert(0, str(PACKAGE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import certus_re.forward as forward  # noqa: E402
from certus_re.forward import Evaluator  # noqa: E402
from certus_re.io_study import load_study  # noqa: E402
from certus_re.physics import aperture_nodes as exact_nodes  # noqa: E402

STUDY = PACKAGE / "studies" / "volet2" / "07_joint_campaign.json"
NEAR_NORMAL_LIMIT_DEG = 10.0


def round_pupil_nodes(total_aperture_deg: float, _n_nodes: int, n_radial: int = 16,
                      n_azimuth: int = 64, incidence_deg: float = 45.0):
    """In-plane equivalent offsets and weights of a uniformly illuminated circular pupil.

    A ray through the pupil at polar coordinates (rho, phi) is tilted by
    (theta_x, theta_y) = rho h (cos phi, sin phi). Only theta_x changes the angle of incidence
    directly. The out-of-plane part theta_y raises the effective angle at second order, by
    approximately theta_y^2 tan(theta_0) / 2, which is the term kept here.
    """
    h = np.deg2rad(0.5 * total_aperture_deg)
    if h <= 0.0:
        return np.zeros(1), np.ones(1)
    rho = np.sqrt((np.arange(n_radial) + 0.5) / n_radial)   # equal-area rings
    phi = (np.arange(n_azimuth) + 0.5) * 2.0 * np.pi / n_azimuth
    r, p = np.meshgrid(rho, phi, indexing="ij")
    theta_x = r * h * np.cos(p)
    theta_y = r * h * np.sin(p)
    offsets = np.rad2deg(theta_x + 0.5 * theta_y**2 * np.tan(np.deg2rad(incidence_deg)))
    offsets = offsets.ravel()
    return offsets, np.full(offsets.size, 1.0 / offsets.size)


def predict_with(evaluator: Evaluator, thicknesses, *, nodes, rule=None, aperture=None):
    """Predict every measurement under a given quadrature rule and node count."""
    original_rule = forward.aperture_nodes
    original_counts = [plan.n_aperture_nodes for plan in evaluator.plans]
    try:
        if rule is not None:
            forward.aperture_nodes = rule
        for plan in evaluator.plans:
            plan.n_aperture_nodes = nodes
            plan._angles.clear()
        return evaluator.predict(thicknesses, aperture_deg=aperture)
    finally:
        forward.aperture_nodes = original_rule
        for plan, count in zip(evaluator.plans, original_counts):
            plan.n_aperture_nodes = count
            plan._angles.clear()


def main() -> int:
    study = load_study(STUDY)
    evaluator = Evaluator(study)
    thicknesses = evaluator.nominal_thicknesses()

    collimated = np.zeros(study.instrument.n_bands, dtype=float)

    curves = {
        "two": predict_with(evaluator, thicknesses, nodes=2, rule=exact_nodes),
        "fine": predict_with(evaluator, thicknesses, nodes=21, rule=exact_nodes),
        "pupil": predict_with(evaluator, thicknesses, nodes=2, rule=round_pupil_nodes),
        "single": predict_with(evaluator, thicknesses, nodes=2, rule=exact_nodes,
                               aperture=collimated),
    }

    rows = []
    for i, plan in enumerate(evaluator.plans):
        oblique = plan.nominal_angle_deg >= NEAR_NORMAL_LIMIT_DEG

        def worst(a: str, b: str) -> float:
            return 100.0 * float(np.max(np.abs(curves[a][i] - curves[b][i])))

        def rms(a: str, b: str) -> float:
            return 100.0 * float(np.sqrt(np.mean((curves[a][i] - curves[b][i]) ** 2)))

        rows.append({
            "measurement": f"{plan.sample.name} ({plan.measurement.polarization})",
            "angle_deg": plan.nominal_angle_deg,
            "oblique": oblique,
            "two_vs_fine_pct": worst("two", "fine"),
            "two_vs_pupil_pct": worst("two", "pupil"),
            "single_vs_fine_pct": worst("single", "fine"),
            "fine_vs_pupil_pct": worst("fine", "pupil"),
            "two_vs_fine_rms_pct": rms("two", "fine"),
            "two_vs_pupil_rms_pct": rms("two", "pupil"),
            "single_vs_fine_rms_pct": rms("single", "fine"),
            "fine_vs_pupil_rms_pct": rms("fine", "pupil"),
        })

    frame = pd.DataFrame(rows)
    target = RESULTS / "aperture_quadrature_check.csv"
    header = ("# Difference in reflectance, in percentage points, between angular-averaging "
              "models of the beam aperture, on the nominal designs of the joint campaign. "
              "The '_pct' columns are the largest absolute difference over the band, the "
              "'_rms_pct' columns the rms difference over the same points -- the quantity "
              "comparable to a fit residual. Written by tools/aperture_quadrature_check.py.\n"
              "# The near-normal rows are identically zero by construction, not by physics: "
              "below the 10 degree threshold the forward model disables the aperture, so the "
              "single-ray and quadrature calculations are the same calculation.\n")
    target.write_text(header + frame.to_csv(index=False), encoding="utf-8")

    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    oblique = frame[frame["oblique"]]
    print()
    if len(oblique):
        print(f"oblique: two rays vs converged in-plane quadrature, worst "
              f"{oblique['two_vs_fine_pct'].max():.3f} points, rms "
              f"{oblique['two_vs_fine_rms_pct'].max():.3f}")
        print(f"oblique: two rays vs round pupil,                   worst "
              f"{oblique['two_vs_pupil_pct'].max():.3f} points, rms "
              f"{oblique['two_vs_pupil_rms_pct'].max():.3f}")
        print(f"oblique: collimated vs converged quadrature,        worst "
              f"{oblique['single_vs_fine_pct'].max():.3f} points, rms "
              f"{oblique['single_vs_fine_rms_pct'].max():.3f}")
        print(f"oblique: out of plane only (in plane vs pupil),     worst "
              f"{oblique['fine_vs_pupil_pct'].max():.3f} points, rms "
              f"{oblique['fine_vs_pupil_rms_pct'].max():.3f}")
    print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
