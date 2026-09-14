"""Does a more exact instrument model change the in situ retrieved indices?

The in situ inversion of Section 6 is repeated under four instrument models, everything else
held fixed: three components alone, literature starting dispersions, four spline knots per
oxide, no process prior, start at the nominal thicknesses.

    A   two rays, no polarizer crosstalk      -- the model of the paper
    B   Gauss-Legendre quadrature, no crosstalk
    C   two rays, crosstalk released
    D   Gauss-Legendre quadrature, crosstalk released

The quadrature converges by nine nodes (see tools/aperture_quadrature_check.py), so B and D
are the exact average over a beam uniform in the plane of incidence, not merely a finer one.

Each retrieved dispersion is then read against the single-layer determination of the
companion paper, in units of that determination's own sigma_n. That comparison is the point
of the exercise: the in situ curves are obtained without any single-layer datum, so moving
closer to the published corridor is evidence that the instrument model was the thing in the
way.

    python tools/insitu_model_refinement.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
INDICES = PACKAGE / "studies" / "volet2" / "indices"
RESULTS = PACKAGE / "results"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

from certus_re.dispersion import TabulatedIndex  # noqa: E402
from certus_re.forward import Evaluator  # noqa: E402
from certus_re.solve import invert  # noqa: E402

from insitu_multistart_noprior import get_3components_study  # noqa: E402

N_NODES_CONVERGED = 9
PROBE_NM = (1500.0, 2500.0, 4000.0)

VARIANTS = (
    ("A", "two rays, no crosstalk", 2, "none"),
    ("B", "quadrature, no crosstalk", N_NODES_CONVERGED, "none"),
    ("C", "two rays, crosstalk fitted", 2, "fitted"),
    ("D", "quadrature, crosstalk fitted", N_NODES_CONVERGED, "fitted"),
)


def literature(material: str) -> TabulatedIndex:
    fname = "SiO2_Franta2016.csv" if material == "SiO2" else "Nb2O5_Franta2024.csv"
    df = pd.read_csv(INDICES / fname, comment="#")
    return TabulatedIndex(
        wavelength_nm=df["Wavelength_nm"].to_numpy(),
        n=df["n"].to_numpy(), k=df["k"].to_numpy(),
        name=f"{material}_literature",
        uncertainty_n=df["sigma_n"].to_numpy(),
        valid_range_nm=(250.0, 4840.0),
    )


def published(material: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(INDICES / f"{material}_H800_published.csv", comment="#")
    return (df["Wavelength_nm"].to_numpy(), df["n"].to_numpy(), df["sigma_n"].to_numpy())


def run(n_nodes: int, crosstalk: str) -> dict:
    study = get_3components_study()

    # Nominal thicknesses before the literature tables go in, as everywhere else in Section 6.
    nominal = Evaluator(study).nominal_thicknesses()

    lit = {m: literature(m) for m in ("Nb2O5", "SiO2")}
    for m, table in lit.items():
        study.materials[m] = table

    study.instrument.n_aperture_nodes = n_nodes
    study.instrument.crosstalk_mode = crosstalk
    study.free.crosstalk = crosstalk
    study.free.thickness_tolerance = 0.25
    study.free.process_prior_pct = None
    study.free.index_correction = "bounded"
    study.free.index_tube_delta = 0.08
    study.free.index_n_knots = 4

    t0 = time.time()
    res = invert(study, start={k: v.copy() for k, v in nominal.items()})
    elapsed = time.time() - t0

    knots = np.linspace(1500.0, 4000.0, 4)
    probe = np.asarray(PROBE_NM)
    indices = {}
    for m in ("SiO2", "Nb2O5"):
        spline = CubicSpline(knots, res.index_corrections[m])
        n_ret = lit[m].n_at(probe) + spline(probe)
        wl_pub, n_pub, sig_pub = published(m)
        n_ref = np.interp(probe, wl_pub, n_pub)
        sig = np.interp(probe, wl_pub, sig_pub)
        indices[m] = {
            "wavelength_nm": [float(w) for w in probe],
            "n_retrieved": [round(float(v), 5) for v in n_ret],
            "n_reference": [round(float(v), 5) for v in n_ref],
            "sigma_n": [round(float(v), 5) for v in sig],
            "departure": [round(float(v), 5) for v in (n_ret - n_ref)],
            "departure_in_sigma": [round(float(v), 3) for v in (n_ret - n_ref) / sig],
        }

    departures = []
    for stack in ("AR6", "BS45", "BS17"):
        d_nom = np.asarray(nominal[stack], dtype=float)
        d_ret = np.asarray(res.thicknesses[stack], dtype=float)
        departures.append(100.0 * (d_ret - d_nom) / d_nom)
    departures = np.concatenate(departures)

    rms = np.array([r.rms_final * 100.0 for r in res.residuals])
    alpha, beta = study.instrument.crosstalk_values()
    fitted = getattr(res, "crosstalk", None)
    if fitted is not None:
        alpha, beta = (float(fitted[0]), float(fitted[1]))

    return {
        "n_aperture_nodes": n_nodes,
        "crosstalk_mode": crosstalk,
        "elapsed_s": round(elapsed, 1),
        "channels": {r.label: round(float(r.rms_final * 100.0), 4) for r in res.residuals},
        "joint_rmse_pct": round(float(np.sqrt(np.mean(rms ** 2))), 4),
        "crosstalk_alpha": round(float(alpha), 5),
        "crosstalk_beta": round(float(beta), 5),
        "thickness_rms_pct": round(float(np.sqrt(np.mean(departures ** 2))), 3),
        "thickness_max_abs_pct": round(float(np.max(np.abs(departures))), 3),
        "indices": indices,
    }


def main() -> int:
    out = {
        "description": "In situ index retrieval on the three components under four instrument "
                       "models. Written by tools/insitu_model_refinement.py.",
        "common": {"components": ["AR6", "BS45", "BS17"], "reference_single_layers": 0,
                   "process_prior": None, "index_n_knots": 4, "index_tube_delta": 0.08,
                   "thickness_tolerance": 0.25, "start": "nominal thicknesses, "
                                                         "literature dispersions"},
        "variants": {},
    }
    for tag, label, nodes, crosstalk in VARIANTS:
        print(f"[{tag}] {label} ...", flush=True)
        record = run(nodes, crosstalk)
        record["label"] = label
        out["variants"][tag] = record
        si = record["indices"]["SiO2"]["departure_in_sigma"]
        nb = record["indices"]["Nb2O5"]["departure_in_sigma"]
        print(f"    joint {record['joint_rmse_pct']:.4f}%  "
              f"SiO2 {si}  Nb2O5 {nb}  "
              f"alpha={record['crosstalk_alpha']:.4f} beta={record['crosstalk_beta']:.4f}  "
              f"({record['elapsed_s']:.0f}s)", flush=True)

    target = RESULTS / "insitu_model_refinement.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
