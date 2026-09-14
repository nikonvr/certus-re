"""The same multi-start benchmark with the process prior left in place.

Its only role in the article is the comparison point quoted in Section 6.5:
the regularized solution sits at a marginally higher residual than the free one.

Paths are relative to this file, so the script runs from any clone of the deposit.
"""

import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
WORKSPACE = PACKAGE.parent
INDICES = PACKAGE / "studies" / "volet2" / "indices"
RESULTS = PACKAGE / "results"
sys.path.insert(0, str(PACKAGE))

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from scipy.interpolate import CubicSpline

from certus_re.io_study import load_study
from certus_re.dispersion import TabulatedIndex
from certus_re.solve import invert
from certus_re.forward import Evaluator

study_path = PACKAGE / "studies" / "volet2" / "07_joint_campaign.json"

def get_3components_study():
    s = load_study(study_path)
    s.samples = [samp for samp in s.samples if not samp.name.startswith("ref_")]
    del s.stacks["REF_SiO2"]
    del s.stacks["REF_Nb2O5"]
    return s

def run_single_inversion(args):
    p_level, draw_idx, gt_thicknesses, nom_d = args
    t0 = time.time()
    
    # Reload in worker
    sys.path.insert(0, str(PACKAGE))
    from certus_re.io_study import load_study
    from certus_re.dispersion import TabulatedIndex
    from certus_re.solve import invert
    
    # Load literature indices
    df_nb = pd.read_csv(INDICES / "Nb2O5_Franta2024.csv", comment="#")
    lit_nb = TabulatedIndex(
        wavelength_nm=df_nb["Wavelength_nm"].to_numpy(),
        n=df_nb["n"].to_numpy(),
        k=df_nb["k"].to_numpy(),
        name="Nb2O5_Franta",
        uncertainty_n=df_nb["sigma_n"].to_numpy(),
        valid_range_nm=(250.0, 4840.0)
    )
    df_si = pd.read_csv(INDICES / "SiO2_Franta2016.csv", comment="#")
    lit_si = TabulatedIndex(
        wavelength_nm=df_si["Wavelength_nm"].to_numpy(),
        n=df_si["n"].to_numpy(),
        k=df_si["k"].to_numpy(),
        name="SiO2_Franta",
        uncertainty_n=df_si["sigma_n"].to_numpy(),
        valid_range_nm=(250.0, 4840.0)
    )
    
    # Perturb thicknesses
    rng = np.random.default_rng(int(1000 * p_level + draw_idx * 73 + 42))
    start_d = {}
    for stack, d in nom_d.items():
        if p_level == 0.0:
            pert = np.ones_like(d)
        else:
            pert = 1.0 + rng.uniform(-p_level, p_level, size=len(d))
        start_d[stack] = d * pert
        
    study = get_3components_study()
    study.materials["Nb2O5"] = lit_nb
    study.materials["SiO2"] = lit_si
    
    tol = max(0.25, p_level + 0.15)
    study.free.thickness_tolerance = tol
    study.free.index_correction = "bounded"
    study.free.index_tube_delta = 0.08
    study.free.index_n_knots = 4
    
    res = invert(study, start=start_d)
    rms = np.sqrt(np.mean([r.rms_final**2 for r in res.residuals])) * 100
    
    knots_grid = np.linspace(1500.0, 4000.0, 4)
    cs_si = CubicSpline(knots_grid, res.index_corrections["SiO2"])
    cs_nb = CubicSpline(knots_grid, res.index_corrections["Nb2O5"])
    
    # Evaluate at key wavelengths
    wls = np.array([1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0])
    n_ret_si = lit_si.n_at(wls) + cs_si(wls)
    n_ret_nb = lit_nb.n_at(wls) + cs_nb(wls)
    
    errs_all, errs_si, errs_nb = [], [], []
    for stack in ("AR6", "BS45", "BS17"):
        for idx, m in enumerate(study.stacks[stack].materials):
            d_sol = res.thicknesses[stack][idx]
            d_ref = gt_thicknesses[stack][idx]
            rel_err = 100.0 * (d_sol - d_ref) / d_ref
            errs_all.append(rel_err)
            if m == "SiO2":
                errs_si.append(rel_err)
            else:
                errs_nb.append(rel_err)
                
    errs_all = np.array(errs_all)
    errs_si = np.array(errs_si)
    errs_nb = np.array(errs_nb)
    
    elapsed = time.time() - t0
    
    return {
        "perturbation_pct": p_level * 100,
        "draw_idx": draw_idx,
        "elapsed_s": elapsed,
        "rmse_pct": float(rms),
        "n_ret_si_1500": float(n_ret_si[0]),
        "n_ret_si_2500": float(n_ret_si[2]),
        "n_ret_si_4000": float(n_ret_si[5]),
        "n_ret_nb_1500": float(n_ret_nb[0]),
        "n_ret_nb_2500": float(n_ret_nb[2]),
        "n_ret_nb_4000": float(n_ret_nb[5]),
        "d_err_all_rms_pct": float(np.sqrt(np.mean(errs_all**2))),
        "d_err_all_mean_pct": float(np.mean(errs_all)),
        "d_err_all_max_abs_pct": float(np.max(np.abs(errs_all))),
        "d_err_si_rms_pct": float(np.sqrt(np.mean(errs_si**2))),
        "d_err_nb_rms_pct": float(np.sqrt(np.mean(errs_nb**2))),
    }

if __name__ == "__main__":
    t_start = time.time()
    
    # 1. Precompute ground truth ONCE
    print("Precomputing ground truth with HELIOS indices...")
    s_gt = get_3components_study()
    eval_gt = Evaluator(s_gt)
    nom_d = eval_gt.nominal_thicknesses()
    res_gt = invert(s_gt)
    gt_thicknesses = {st: res_gt.thicknesses[st].copy() for st in res_gt.thicknesses}
    rms_gt = np.sqrt(np.mean([r.rms_final**2 for r in res_gt.residuals])) * 100
    print(f"Ground truth computed in {time.time()-t_start:.1f}s, RMSE = {rms_gt:.3f}%")
    
    tasks = []
    # 0%: 1 draw
    tasks.append((0.0, 0, gt_thicknesses, nom_d))
    # 5%, 10%, 15%, 20%: 10 draws each
    for p in [0.05, 0.10, 0.15, 0.20]:
        for d in range(10):
            tasks.append((p, d, gt_thicknesses, nom_d))
            
    print(f"Running {len(tasks)} inversions with 4-knot spline across 10 CPU workers...")
    t_pool = time.time()
    
    results = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(run_single_inversion, t): t for t in tasks}
        for future in as_completed(futures):
            t = futures[future]
            try:
                res = future.result()
                results.append(res)
                print(f"[Done {len(results)}/{len(tasks)}] Error={res['perturbation_pct']:.0f}% Draw={res['draw_idx']} -> RMSE={res['rmse_pct']:.3f}%, n_Si(1500)={res['n_ret_si_1500']:.4f}, d_err_RMS={res['d_err_all_rms_pct']:.2f}% ({res['elapsed_s']:.1f}s)")
            except Exception as exc:
                print(f"Task {t} generated an exception: {exc}")
                
    total_time = time.time() - t_start
    print(f"\nAll {len(results)} runs completed in {total_time:.1f}s ({total_time/60:.2f} min)")
    
    df_all = pd.DataFrame(results).sort_values(by=["perturbation_pct", "draw_idx"])
    out_all_csv = RESULTS / "insitu_multistart_prior.csv"
    df_all.to_csv(out_all_csv, index=False)
    print(f"Saved full results to {out_all_csv}")
    
    # Aggregate summary
    summary = []
    for p in [0.0, 5.0, 10.0, 15.0, 20.0]:
        sub = df_all[df_all["perturbation_pct"] == p]
        succ = sub[sub["rmse_pct"] < 0.60]
        summary.append({
            "perturbation_pct": p,
            "n_draws": len(sub),
            "success_rate_pct": len(succ) / len(sub) * 100.0,
            "rmse_succ_mean": succ["rmse_pct"].mean() if len(succ) > 0 else np.nan,
            "rmse_succ_std": succ["rmse_pct"].std() if len(succ) > 1 else 0.0,
            "n_ret_si_1500_mean": succ["n_ret_si_1500"].mean() if len(succ) > 0 else np.nan,
            "n_ret_nb_1500_mean": succ["n_ret_nb_1500"].mean() if len(succ) > 0 else np.nan,
            "d_err_rms_succ_mean": succ["d_err_all_rms_pct"].mean() if len(succ) > 0 else np.nan,
            "d_err_rms_succ_std": succ["d_err_all_rms_pct"].std() if len(succ) > 1 else 0.0,
            "d_err_max_succ_mean": succ["d_err_all_max_abs_pct"].mean() if len(succ) > 0 else np.nan,
        })
        
    df_sum = pd.DataFrame(summary)
    out_sum_csv = RESULTS / "insitu_multistart_prior_summary.csv"
    df_sum.to_csv(out_sum_csv, index=False)
    print(f"Saved summary to {out_sum_csv}\n")
    print("=" * 85)
    print("STATISTICAL SUMMARY ACROSS DRAWS (4 KNOTS):")
    print("=" * 85)
    print(df_sum.to_string())
