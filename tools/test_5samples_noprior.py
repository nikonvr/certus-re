"""Test reverse engineering on the full 5-sample campaign with sigma = infinity (no process prior).

5 samples:
- 3 multilayer filters (AR6, BS45, BS17: 39 layers)
- 2 reference single layers (REF_SiO2, REF_Nb2O5: 2 layers)
Total: 41 layer thicknesses across 7 spectral channels.

Condition:
sigma_proc = None (unconstrained / infinity).

Evaluates:
1. Baseline with reference indices fixed.
2. Offset model (dn_SiO2, dn_Nb2O5).
3. 3-knot cubic spline model from flat start.
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


def main():
    study_file = PACKAGE / "studies" / "volet2" / "07_joint_campaign.json"
    study = load_study(study_file)

    evaluator = Evaluator(study)
    nominal_th = evaluator.nominal_thicknesses()
    stack_names = ["AR6", "BS45", "BS17", "REF_SiO2", "REF_Nb2O5"]
    layer_counts = {k: len(study.stacks[k].layers) for k in stack_names}

    measured = np.concatenate([p.measurement.value for p in evaluator.plans])
    sigma_meas = np.concatenate([p.measurement.sigma_vector() for p in evaluator.plans])
    x0_th = np.concatenate([nominal_th[k] for k in stack_names])
    base_front = [p.n_front.copy() for p in evaluator.plans]

    n_layers_total = len(x0_th)  # 41

    def unpack_th(x: np.ndarray) -> dict[str, np.ndarray]:
        d = {}
        idx = 0
        for name in stack_names:
            cnt = layer_counts[name]
            d[name] = x[idx : idx + cnt]
            idx += cnt
        return d

    print("=" * 90)
    print("REVERSE ENGINEERING SUR 5 ECHANTILLONS (3 FILTRES + 2 MONOCOUCHES DE REFERENCE)")
    print("CONDITION : SIGMA = INFINI (AUCUN PRIOR D'EPAISSEUR)")
    print("=" * 90)
    print(f"Nombre d'echantillons : {len(study.samples)}")
    print(f"Nombre de canaux spectraux : {len(evaluator.plans)}")
    print(f"Nombre de points spectraux : {len(measured)}")
    print(f"Nombre d'epaisseurs libres : {n_layers_total} (39 multicouches + 2 monocouches de reference)")

    # 1. Baseline: Indices fixes (valeurs de reference Volet 1), epaisseurs libres (sans prior)
    def res_fixed(x: np.ndarray) -> np.ndarray:
        pred = evaluator.predict(unpack_th(x))
        return (np.concatenate(pred) - measured) / sigma_meas

    lb_th = x0_th * 0.1
    ub_th = x0_th * 2.5
    opt_fixed = least_squares(res_fixed, x0_th, bounds=(lb_th, ub_th), ftol=1e-6, xtol=1e-6)
    pred_fixed = evaluator.predict(unpack_th(opt_fixed.x))
    rmse_fixed = np.sqrt(np.mean((np.concatenate(pred_fixed) - measured) ** 2)) * 100.0

    diff_th_fixed = (opt_fixed.x - x0_th) / x0_th * 100.0
    print("\n1. INDICES FIXES (REFERENCE NOMINALE) - EPAISSEURS LIBRES SANS PRIOR:")
    print(f"   RMSE global       = {rmse_fixed:.3f} %")
    print(f"   Moyenne |d - d0|  = {np.mean(np.abs(diff_th_fixed[:39])):.2f} % sur multicouches")
    print(f"   Ecart d REF_SiO2  = {diff_th_fixed[39]:+.2f} % (d = {opt_fixed.x[39]:.1f} nm vs nominal {x0_th[39]:.1f} nm)")
    print(f"   Ecart d REF_Nb2O5 = {diff_th_fixed[40]:+.2f} % (d = {opt_fixed.x[40]:.1f} nm vs nominal {x0_th[40]:.1f} nm)")

    # 2. Offset Model: 41 epaisseurs + 2 offsets (dn_SiO2, dn_Nb2O5), sans prior
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

    def res_offset(x: np.ndarray) -> np.ndarray:
        th = unpack_th(x[:n_layers_total])
        dn_L, dn_H = x[n_layers_total], x[n_layers_total + 1]
        apply_offset(dn_L, dn_H)
        pred = evaluator.predict(th)
        return (np.concatenate(pred) - measured) / sigma_meas

    x0_off = np.concatenate([x0_th, [0.0, 0.0]])
    lb_off = np.concatenate([lb_th, [-0.10, -0.10]])
    ub_off = np.concatenate([ub_th, [0.10, 0.10]])

    opt_off = least_squares(res_offset, x0_off, bounds=(lb_off, ub_off), ftol=1e-6, xtol=1e-6)
    dn_L_found, dn_H_found = opt_off.x[n_layers_total], opt_off.x[n_layers_total + 1]
    apply_offset(dn_L_found, dn_H_found)
    pred_off = evaluator.predict(unpack_th(opt_off.x[:n_layers_total]))
    rmse_off = np.sqrt(np.mean((np.concatenate(pred_off) - measured) ** 2)) * 100.0
    diff_th_off = (opt_off.x[:n_layers_total] - x0_th) / x0_th * 100.0

    print("\n2. MODELE D'OFFSET (41 EPAISSEURS + 2 OFFSETS D'INDICE) - SANS PRIOR:")
    print(f"   RMSE global       = {rmse_off:.3f} %")
    print(f"   delta_n_SiO2      = {dn_L_found:+.5f} ({dn_L_found / 1.47050 * 100:+.2f}%)")
    print(f"   delta_n_Nb2O5     = {dn_H_found:+.5f} ({dn_H_found / 2.23870 * 100:+.2f}%)")
    print(f"   Moyenne |d - d0|  = {np.mean(np.abs(diff_th_off[:39])):.2f} % sur multicouches")
    print(f"   Max |d - d0|      = {np.max(np.abs(diff_th_off[:39])):.2f} % sur multicouches")
    print(f"   Ecart d REF_SiO2  = {diff_th_off[39]:+.2f} % (d = {opt_off.x[39]:.1f} nm)")
    print(f"   Ecart d REF_Nb2O5 = {diff_th_off[40]:+.2f} % (d = {opt_off.x[40]:.1f} nm)")

    # 3. Nodal Spline Model: 41 epaisseurs + 6 noeuds spline (depart plat a 1500 nm), sans prior
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

    def res_spline(x: np.ndarray) -> np.ndarray:
        th = unpack_th(x[:n_layers_total])
        knots_si = x[n_layers_total : n_layers_total + 3]
        knots_nb = x[n_layers_total + 3 : n_layers_total + 6]
        apply_spline(knots_si, knots_nb)
        pred = evaluator.predict(th)
        return (np.concatenate(pred) - measured) / sigma_meas

    x0_sp = np.concatenate([x0_th, [FLAT_SI]*3, [FLAT_NB]*3])
    lb_sp = np.concatenate([lb_th, [1.35]*3, [2.00]*3])
    ub_sp = np.concatenate([ub_th, [1.60]*3, [2.45]*3])

    opt_sp = least_squares(res_spline, x0_sp, bounds=(lb_sp, ub_sp), ftol=1e-6, xtol=1e-6)
    knots_si_found = opt_sp.x[n_layers_total : n_layers_total + 3]
    knots_nb_found = opt_sp.x[n_layers_total + 3 : n_layers_total + 6]
    apply_spline(knots_si_found, knots_nb_found)
    pred_sp = evaluator.predict(unpack_th(opt_sp.x[:n_layers_total]))
    rmse_sp = np.sqrt(np.mean((np.concatenate(pred_sp) - measured) ** 2)) * 100.0
    diff_th_sp = (opt_sp.x[:n_layers_total] - x0_th) / x0_th * 100.0

    print("\n3. MODELE SPLINE A 3 NOEUDS (DEPART PLAT, 41 EPAISSEURS + 6 NOEUDS) - SANS PRIOR:")
    print(f"   RMSE global       = {rmse_sp:.3f} %")
    print(f"   Noeuds Nb2O5      = [{knots_nb_found[0]:.5f}, {knots_nb_found[1]:.5f}, {knots_nb_found[2]:.5f}]")
    print(f"   Noeuds SiO2       = [{knots_si_found[0]:.5f}, {knots_si_found[1]:.5f}, {knots_si_found[2]:.5f}]")
    print(f"   Moyenne |d - d0|  = {np.mean(np.abs(diff_th_sp[:39])):.2f} % sur multicouches")
    print(f"   Max |d - d0|      = {np.max(np.abs(diff_th_sp[:39])):.2f} % sur multicouches")
    print(f"   Ecart d REF_SiO2  = {diff_th_sp[39]:+.2f} % (d = {opt_sp.x[39]:.1f} nm)")
    print(f"   Ecart d REF_Nb2O5 = {diff_th_sp[40]:+.2f} % (d = {opt_sp.x[40]:.1f} nm)")

    # Detail par canal
    print("\n4. DETAIL PAR CANAL (POUR LE MODELE SPLINE SANS PRIOR) :")
    for plan, pred_arr in zip(evaluator.plans, pred_sp):
        m = plan.measurement
        r = pred_arr - m.value
        rms_chan = np.sqrt(np.mean(r**2)) * 100.0
        print(f"   {plan.sample.name:<12} | {m.quantity} ({m.polarization}) @ {m.angle_deg:4.1f} deg | RMSE = {rms_chan:.3f} %")
    print("=" * 90)


if __name__ == "__main__":
    main()
