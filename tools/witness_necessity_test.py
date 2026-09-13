"""Are reference single layers strictly indispensable in a multi-component campaign?

This diagnostic tool reproduces the experiment described in Section 5.3 of the article:
it performs an unconstrained joint inversion using solely the three multilayer coatings
(Filter 1, Filter 2, and Filter 3 -- 39 layer thicknesses across 1750 spectral points)
with no reference single layers provided, simultaneously retrieving layer thicknesses
and material refractive index offsets (delta_n_SiO2, delta_n_Nb2O5).

Usage
-----
    python tools/witness_necessity_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE))

from certus_re.forward import Evaluator
from certus_re.io_study import load_study
from certus_re.model import Study


def main() -> None:
    print("=" * 80)
    print("WITNESS NECESSITY TEST: SELF-CONSISTENT RETRIEVAL OF INDICES & THICKNESSES")
    print("=" * 80)

    study_file = PACKAGE / "studies" / "volet2" / "07_joint_campaign.json"
    study = load_study(study_file)

    # Filter to keep only the three multilayer filters (exclude reference single layers)
    filter_samples = [
        s for s in study.samples if s.name in ("AR6_alone", "BS45_alone", "BS17_alone")
    ]
    study_3f = Study(
        name="3_filters_only",
        lambda0_nm=study.lambda0_nm,
        materials=study.materials,
        substrates=study.substrates,
        stacks={k: study.stacks[k] for k in ("AR6", "BS45", "BS17")},
        samples=filter_samples,
        instrument=study.instrument,
        free=study.free,
    )

    evaluator = Evaluator(study_3f)
    nominal_th = evaluator.nominal_thicknesses()
    stack_names = ["AR6", "BS45", "BS17"]
    layer_counts = {k: len(study.stacks[k].layers) for k in stack_names}

    measured = np.concatenate([p.measurement.value for p in evaluator.plans])
    sigma = np.concatenate([p.measurement.sigma_vector() for p in evaluator.plans])
    x0_th = np.concatenate([nominal_th[k] for k in stack_names])
    base_front = [p.n_front.copy() for p in evaluator.plans]

    def unpack_th(x: np.ndarray) -> dict[str, np.ndarray]:
        d = {}
        idx = 0
        for name in stack_names:
            cnt = layer_counts[name]
            d[name] = x[idx : idx + cnt]
            idx += cnt
        return d

    # 1. Reference: 39 thicknesses inverted with witness-fixed indices
    def res_th_only(x: np.ndarray) -> np.ndarray:
        pred = evaluator.predict(unpack_th(x))
        r_data = (np.concatenate(pred) - measured) / sigma
        r_prior = (x - x0_th) / (x0_th * 0.005)
        return np.concatenate([r_data, r_prior])

    opt_th = least_squares(
        res_th_only,
        x0_th,
        bounds=(np.zeros_like(x0_th), x0_th * 2.0),
        ftol=1e-6,
        xtol=1e-6,
    )
    pred_ref = evaluator.predict(unpack_th(opt_th.x))
    rmse_ref = np.sqrt(np.mean((np.concatenate(pred_ref) - measured) ** 2)) * 100.0

    print(f"\n1. REFERENCE SOLUTION (Indices fixed to witness single layers):")
    print(f"   RMSE = {rmse_ref:.3f} % across 1750 spectral points")

    # 2. Test: 39 thicknesses + 2 index offsets without any witness data
    def apply_offset(dn_L: float, dn_H: float) -> None:
        for i, plan in enumerate(evaluator.plans):
            stack = study.stacks[plan.front_stack]
            n_mat = base_front[i].copy()
            for j, layer in enumerate(stack.layers):
                if layer.material == "SiO2":
                    n_mat[:, j] += dn_L
                elif layer.material == "Nb2O5":
                    n_mat[:, j] += dn_H
            plan.n_front = n_mat

    def res_test(x: np.ndarray) -> np.ndarray:
        th = unpack_th(x[:39])
        dn_L, dn_H = x[39], x[40]
        apply_offset(dn_L, dn_H)
        pred = evaluator.predict(th)
        r_data = (np.concatenate(pred) - measured) / sigma
        r_prior = (x[:39] - x0_th) / (x0_th * 0.005)
        return np.concatenate([r_data, r_prior])

    # Initialized from intentionally perturbed offsets (+0.02 / -0.02)
    x0 = np.concatenate([x0_th, [0.02, -0.02]])
    lb = np.concatenate([np.zeros_like(x0_th), [-0.10, -0.10]])
    ub = np.concatenate([x0_th * 2.0, [0.10, 0.10]])

    opt = least_squares(res_test, x0, bounds=(lb, ub), ftol=1e-6, xtol=1e-6)
    dn_L_found = opt.x[39]
    dn_H_found = opt.x[40]
    apply_offset(dn_L_found, dn_H_found)
    pred_test = evaluator.predict(unpack_th(opt.x[:39]))
    rmse_test = np.sqrt(np.mean((np.concatenate(pred_test) - measured) ** 2)) * 100.0

    print(f"\n2. SELF-CONSISTENT RETRIEVAL (3 filters alone, no witness data):")
    print(f"   Converged in {opt.nfev} evaluations ({opt.message})")
    print(f"   Recovered index offsets from witness values:")
    print(f"     delta_n_SiO2  = {dn_L_found:+.5f} ({dn_L_found / 1.47050 * 100:+.2f}%)")
    print(f"     delta_n_Nb2O5 = {dn_H_found:+.5f} ({dn_H_found / 2.23870 * 100:+.2f}%)")
    print(f"   Global RMSE = {rmse_test:.3f} %")

    # Layer-by-layer comparison
    th_ref = opt_th.x
    th_test = opt.x[:39]
    diff_pct = (th_test - th_ref) / th_ref * 100.0
    print(f"\n3. RECONSTRUCTED LAYER THICKNESS ACCURACY:")
    print(f"   Mean absolute thickness departure: {np.mean(np.abs(diff_pct)):.2f} %")
    print(f"   Max absolute thickness departure:  {np.max(np.abs(diff_pct)):.2f} %")
    print("=" * 80)


if __name__ == "__main__":
    main()
