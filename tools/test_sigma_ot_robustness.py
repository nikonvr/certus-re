"""Test robustness of the index retrieval (Figure 8) against sigma_proc (allowed delta OT).

Tests both:
1. The 2-parameter offset model (dn_SiO2, dn_Nb2O5)
2. The 6-parameter nodal spline model (3 knots per material from flat start)
across a range of sigma_proc values (from tight prior 0.1% to unconstrained / no prior).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE))

from certus_re.forward import Evaluator
from certus_re.io_study import load_study
from certus_re.model import Study


def setup_evaluator():
    study_file = PACKAGE / "studies" / "volet2" / "07_joint_campaign.json"
    study = load_study(study_file)

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
    sigma_meas = np.concatenate([p.measurement.sigma_vector() for p in evaluator.plans])
    x0_th = np.concatenate([nominal_th[k] for k in stack_names])
    base_front = [p.n_front.copy() for p in evaluator.plans]

    return study, evaluator, stack_names, layer_counts, measured, sigma_meas, x0_th, base_front


def main():
    (
        study,
        evaluator,
        stack_names,
        layer_counts,
        measured,
        sigma_meas,
        x0_th,
        base_front,
    ) = setup_evaluator()

    def unpack_th(x: np.ndarray) -> dict[str, np.ndarray]:
        d = {}
        idx = 0
        for name in stack_names:
            cnt = layer_counts[name]
            d[name] = x[idx : idx + cnt]
            idx += cnt
        return d

    # Range of sigma_proc (in percent): 0.1%, 0.2%, 0.5%, 1.0%, 2.0%, 5.0%, and None (no prior)
    sigma_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, None]

    print("=" * 90)
    print("TEST DE ROBUSTESSE A SIGMA (TOLERANCE DELTA OT) - MODELE D'OFFSET (2 PARAMETRES)")
    print("=" * 90)
    print(f"{'Sigma_proc (%)':<15} | {'RMSE (%)':<10} | {'dn_SiO2':<12} | {'dn_Nb2O5':<12} | {'Moy |d-d0| (%)':<16} | {'Max |d-d0| (%)':<16}")
    print("-" * 90)

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

    for s_val in sigma_list:
        def res_offset(x: np.ndarray) -> np.ndarray:
            th = unpack_th(x[:39])
            dn_L, dn_H = x[39], x[40]
            apply_offset(dn_L, dn_H)
            pred = evaluator.predict(th)
            r_data = (np.concatenate(pred) - measured) / sigma_meas
            if s_val is not None and s_val > 0:
                r_prior = (x[:39] - x0_th) / (x0_th * (s_val / 100.0))
                return np.concatenate([r_data, r_prior])
            return r_data

        x0 = np.concatenate([x0_th, [0.0, 0.0]])
        lb = np.concatenate([x0_th * 0.1, [-0.10, -0.10]])
        ub = np.concatenate([x0_th * 2.0, [0.10, 0.10]])

        opt = least_squares(res_offset, x0, bounds=(lb, ub), ftol=1e-6, xtol=1e-6)
        dn_L_found, dn_H_found = opt.x[39], opt.x[40]
        apply_offset(dn_L_found, dn_H_found)
        pred = evaluator.predict(unpack_th(opt.x[:39]))
        rmse = np.sqrt(np.mean((np.concatenate(pred) - measured) ** 2)) * 100.0

        diff_pct = np.abs(opt.x[:39] - x0_th) / x0_th * 100.0
        mean_diff = np.mean(diff_pct)
        max_diff = np.max(diff_pct)

        s_str = f"{s_val:.1f}%" if s_val is not None else "None (libre)"
        print(f"{s_str:<15} | {rmse:<10.3f} | {dn_L_found:+12.5f} | {dn_H_found:+12.5f} | {mean_diff:<16.2f} | {max_diff:<16.2f}")

    print("\n" + "=" * 90)
    print("TEST DE ROBUSTESSE A SIGMA (TOLERANCE DELTA OT) - MODELE SPLINE (6 PARAMETRES, DEPART PLAT)")
    print("=" * 90)

    # Flat reference values at 1500 nm
    FLAT_NB = 2.23870
    FLAT_SI = 1.47050
    KNOT_NM = np.array([1500.0, 2750.0, 4000.0])

    def apply_spline(knots_si: np.ndarray, knots_nb: np.ndarray) -> None:
        cs_si = CubicSpline(KNOT_NM, knots_si, bc_type="natural")
        cs_nb = CubicSpline(KNOT_NM, knots_nb, bc_type="natural")
        for i, plan in enumerate(evaluator.plans):
            wl_nm = plan.measurement.wavelength_nm
            n_si_wl = cs_si(wl_nm)
            n_nb_wl = cs_nb(wl_nm)
            stack = study.stacks[plan.front_stack]
            n_mat = np.zeros((len(wl_nm), len(stack.layers)), dtype=np.float64)
            for j, layer in enumerate(stack.layers):
                if layer.material == "SiO2":
                    n_mat[:, j] = n_si_wl
                elif layer.material == "Nb2O5":
                    n_mat[:, j] = n_nb_wl
            plan.n_front = n_mat

    print(f"{'Sigma_proc (%)':<15} | {'RMSE (%)':<10} | {'Nb2O5 @1.5/2.75/4.0 um':<28} | {'SiO2 @1.5/2.75/4.0 um':<28} | {'Moy |d-d0|':<10}")
    print("-" * 90)

    for s_val in sigma_list:
        def res_spline(x: np.ndarray) -> np.ndarray:
            th = unpack_th(x[:39])
            knots_si = x[39:42]
            knots_nb = x[42:45]
            apply_spline(knots_si, knots_nb)
            pred = evaluator.predict(th)
            r_data = (np.concatenate(pred) - measured) / sigma_meas
            if s_val is not None and s_val > 0:
                r_prior = (x[:39] - x0_th) / (x0_th * (s_val / 100.0))
                return np.concatenate([r_data, r_prior])
            return r_data

        x0 = np.concatenate([x0_th, [FLAT_SI]*3, [FLAT_NB]*3])
        lb = np.concatenate([x0_th * 0.1, [1.35]*3, [2.00]*3])
        ub = np.concatenate([x0_th * 2.0, [1.60]*3, [2.45]*3])

        opt = least_squares(res_spline, x0, bounds=(lb, ub), ftol=1e-6, xtol=1e-6)
        knots_si_found = opt.x[39:42]
        knots_nb_found = opt.x[42:45]
        apply_spline(knots_si_found, knots_nb_found)
        pred = evaluator.predict(unpack_th(opt.x[:39]))
        rmse = np.sqrt(np.mean((np.concatenate(pred) - measured) ** 2)) * 100.0

        diff_pct = np.abs(opt.x[:39] - x0_th) / x0_th * 100.0
        mean_diff = np.mean(diff_pct)

        nb_str = f"[{knots_nb_found[0]:.3f}, {knots_nb_found[1]:.3f}, {knots_nb_found[2]:.3f}]"
        si_str = f"[{knots_si_found[0]:.3f}, {knots_si_found[1]:.3f}, {knots_si_found[2]:.3f}]"

        s_str = f"{s_val:.1f}%" if s_val is not None else "None (libre)"
        print(f"{s_str:<15} | {rmse:<10.3f} | {nb_str:<28} | {si_str:<28} | {mean_diff:<10.2f}%")

    print("=" * 90)


if __name__ == "__main__":
    main()
