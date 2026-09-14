"""Figure 9: one representative run of the unregularized in situ inversion.

The run drawn here must be the same experiment as the one tabulated by
``insitu_multistart_noprior.py``: three multilayer components alone, literature starting
dispersions, thicknesses perturbed by 10 per cent, and **no process prior**. Every residual
printed on the figure is computed from that run, never typed in by hand; the values are also
written to ``results/fig09_insitu_representative.json`` so the manuscript can quote them.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

PACKAGE = Path(__file__).resolve().parent.parent
WORKSPACE = PACKAGE.parent
RESULTS = PACKAGE / "results"
sys.path.insert(0, str(PACKAGE))

from certus_re.io_study import load_study
from certus_re.dispersion import TabulatedIndex
from certus_re.forward import Evaluator
from certus_re.solve import invert

study_path = PACKAGE / "studies" / "volet2" / "07_joint_campaign.json"

def get_clean_study():
    s = load_study(study_path)
    s.samples = [samp for samp in s.samples if not samp.name.startswith("ref_")]
    del s.stacks["REF_SiO2"]
    del s.stacks["REF_Nb2O5"]
    return s

s = get_clean_study()

# Load literature indices
df_nb = pd.read_csv(PACKAGE / "studies" / "volet2" / "indices" / "Nb2O5_Franta2024.csv", comment="#")
lit_nb = TabulatedIndex(
    wavelength_nm=df_nb["Wavelength_nm"].to_numpy(),
    n=df_nb["n"].to_numpy(),
    k=df_nb["k"].to_numpy(),
    name="Nb2O5_Franta",
    uncertainty_n=df_nb["sigma_n"].to_numpy(),
    valid_range_nm=(250.0, 4840.0)
)
df_si = pd.read_csv(PACKAGE / "studies" / "volet2" / "indices" / "SiO2_Franta2016.csv", comment="#")
lit_si = TabulatedIndex(
    wavelength_nm=df_si["Wavelength_nm"].to_numpy(),
    n=df_si["n"].to_numpy(),
    k=df_si["k"].to_numpy(),
    name="SiO2_Franta",
    uncertainty_n=df_si["sigma_n"].to_numpy(),
    valid_range_nm=(250.0, 4840.0)
)

# The nominal thicknesses the perturbed start is drawn around are the ones of the deposition
# sheets, converted with the campaign (reference) indices -- exactly as insitu_multistart_noprior
# does it. Substituting the literature indices first would move the nominal itself by 1.7 per cent
# on SiO2 and make the reported thickness departures incomparable between the two.
nom_d = Evaluator(get_clean_study()).nominal_thicknesses()

s.materials["Nb2O5"] = lit_nb
s.materials["SiO2"] = lit_si

evaluator_init = Evaluator(s)

# 10% perturbation (draw 0)
rng = np.random.default_rng(42)
start_d = {st: nom_d[st] * (1.0 + rng.uniform(-0.10, 0.10, size=len(nom_d[st]))) for st in nom_d}

pred_init = evaluator_init.predict(start_d)

# Invert with free spline, under the conditions of the unregularized stress test
s.free.thickness_tolerance = 0.25
s.free.process_prior_pct = None   # STRICTLY NO PROCESS PRIOR, as in insitu_multistart_noprior
s.free.index_correction = "bounded"
s.free.index_tube_delta = 0.08
s.free.index_n_knots = 4

res = invert(s, start=start_d)

# ---------------------------------------------------------------------------
# Residuals of this run, in percentage points. Channel order follows the plans:
# AR6 (a), BS45 (s), BS45 (p), BS17 (s), BS17 (p).
# ---------------------------------------------------------------------------
rms_init = np.array([r.rms_initial * 100.0 for r in res.residuals])
rms_fin = np.array([r.rms_final * 100.0 for r in res.residuals])
n_pts = np.array([r.n_points for r in res.residuals], dtype=float)


def quad(values):
    """Unweighted quadratic mean, the convention used for the per-component figures."""
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def label_polarizations(ax, wl, curve_s, curve_p, colour):
    """Write Rs and Rp beside their own curves, left of the upper-right legend.

    The position is read off the curves rather than fixed, so the labels stay put when the
    spectra change. Only the left half of the band is searched, which keeps them clear of the
    legend box.
    """
    left = wl <= 0.5 * (wl[0] + wl[-1])
    k = int(np.argmax((curve_s - curve_p)[left]))
    x = float(wl[left][k])
    for curve, name, offset in ((curve_s, r"$R_s$", 5.0), (curve_p, r"$R_p$", -5.0)):
        y = float(curve[left][k]) + offset
        ax.text(x, y, name, fontsize=8.5, weight="bold", color=colour,
                ha="center", va="bottom" if offset > 0 else "top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


joint_init = quad(rms_init)
joint_fin = quad(rms_fin)
joint_init_w = float(np.sqrt(np.sum(n_pts * rms_init ** 2) / n_pts.sum()))
joint_fin_w = float(np.sqrt(np.sum(n_pts * rms_fin ** 2) / n_pts.sum()))

numbers = {
    "description": "Representative 10% draw of the unregularized in situ inversion, "
                   "drawn in Fig. 9. Written by tools/plot_insitu_robustness.py.",
    "process_prior": None,
    "thickness_tolerance": 0.25,
    "index_tube_delta": 0.08,
    "index_n_knots": 4,
    "channels": [
        {"label": r.label, "n_points": int(r.n_points),
         "rms_initial_pct": round(float(r.rms_initial * 100.0), 4),
         "rms_final_pct": round(float(r.rms_final * 100.0), 4)}
        for r in res.residuals
    ],
    "per_component": {
        "AR6": {"rms_initial_pct": round(rms_init[0], 3), "rms_final_pct": round(rms_fin[0], 3)},
        "BS45": {"rms_initial_pct": round(quad(rms_init[1:3]), 3),
                 "rms_final_pct": round(quad(rms_fin[1:3]), 3)},
        "BS17": {"rms_initial_pct": round(quad(rms_init[3:5]), 3),
                 "rms_final_pct": round(quad(rms_fin[3:5]), 3)},
    },
    "joint_channel_quadratic_mean_pct": {"initial": round(joint_init, 4),
                                         "final": round(joint_fin, 4)},
    "joint_point_weighted_pct": {"initial": round(joint_init_w, 4),
                                 "final": round(joint_fin_w, 4)},
}
(RESULTS / "fig09_insitu_representative.json").write_text(
    json.dumps(numbers, indent=1), encoding="utf-8")

for r in res.residuals:
    print(f"{r.label:34s} {r.n_points:5d}  {r.rms_initial * 100:8.3f} -> {r.rms_final * 100:7.4f}")
print(f"joint, channel quadratic mean : {joint_init:.3f} -> {joint_fin:.4f} %")
print(f"joint, point weighted         : {joint_init_w:.3f} -> {joint_fin_w:.4f} %")

# Predict final using corrected materials
knots_wl = np.linspace(1500.0, 4000.0, 4)
cs_si = CubicSpline(knots_wl, res.index_corrections["SiO2"])
cs_nb = CubicSpline(knots_wl, res.index_corrections["Nb2O5"])

s_final = get_clean_study()
wl_nb = df_nb["Wavelength_nm"].to_numpy()
s_final.materials["Nb2O5"] = TabulatedIndex(
    wavelength_nm=wl_nb,
    n=df_nb["n"].to_numpy() + cs_nb(wl_nb),
    k=df_nb["k"].to_numpy(),
    name="Nb2O5_retrieved"
)
wl_si = df_si["Wavelength_nm"].to_numpy()
s_final.materials["SiO2"] = TabulatedIndex(
    wavelength_nm=wl_si,
    n=df_si["n"].to_numpy() + cs_si(wl_si),
    k=df_si["k"].to_numpy(),
    name="SiO2_retrieved"
)
evaluator_final = Evaluator(s_final)
pred_final = evaluator_final.predict(res.thicknesses)

# Ground truth study for index reference
study_true = load_study(study_path)

# Colors
CLR_MEAS = "#111111"        # Black solid
CLR_INIT = "#D95F02"        # Orange/red dashed
CLR_FIT = "#1B9E77"         # Green solid
CLR_HELIOS = "#2166AC"      # Blue solid

fig = plt.figure(figsize=(7.2, 7.8), dpi=300)
# 4 rows: Filter 1 (AR), Filter 2 (BS45), Filter 3 (BS17), Index dispersions
gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.2, 1.2, 1.2], hspace=0.38)

# Panel 1: Filter 1, reflectance at 8 degrees, unpolarized
ax1 = fig.add_subplot(gs[0])
wl1 = evaluator_init.plans[0].measurement.wavelength_nm
meas1 = evaluator_init.plans[0].measurement.value * 100
init1 = pred_init[0] * 100
fit1 = pred_final[0] * 100

ax1.plot(wl1, init1, color=CLR_INIT, linestyle="--", linewidth=1.2, alpha=0.9,
         label=rf"Initial start (Literature $n$ + 10% $d$ error, RMSE = {rms_init[0]:.2f}%)")
ax1.plot(wl1, fit1, color=CLR_FIT, linestyle="-", linewidth=1.6,
         label=rf"Final joint inversion (Retrieved $n(\lambda)$ & $d$, RMSE = {rms_fin[0]:.2f}%)")
ax1.plot(wl1, meas1, color=CLR_MEAS, linestyle=":", linewidth=1.1, label="Measured spectrum", alpha=0.75)
ax1.set_ylabel(r"Reflectance (%)", fontsize=8.5, weight="bold")
ax1.set_title(r"(a) Filter 1 (Antireflection, 6 layers, 8$^\circ$ incidence, unpolarized)",
              fontsize=9.0, weight="bold", loc="left")
ax1.set_xlim(1500, 4000)
top1 = float(np.max([init1.max(), fit1.max(), meas1.max()]))
ax1.set_ylim(0, 1.55 * top1)
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.legend(loc="upper right", fontsize=7.0, framealpha=0.9)

# Panel 2: Filter 2 (Beam Splitter 45 deg)
ax2 = fig.add_subplot(gs[1])
wl2 = evaluator_init.plans[1].measurement.wavelength_nm
meas2_s = evaluator_init.plans[1].measurement.value * 100
meas2_p = evaluator_init.plans[2].measurement.value * 100
init2_s = pred_init[1] * 100
init2_p = pred_init[2] * 100
fit2_s = pred_final[1] * 100
fit2_p = pred_final[2] * 100

ax2.plot(wl2, init2_s, color=CLR_INIT, linestyle="--", linewidth=1.1, alpha=0.85,
         label=rf"Initial start (RMSE $s$/$p$ = {rms_init[1]:.2f}/{rms_init[2]:.2f}%)")
ax2.plot(wl2, init2_p, color=CLR_INIT, linestyle="--", linewidth=1.1, alpha=0.85)
ax2.plot(wl2, fit2_s, color=CLR_FIT, linestyle="-", linewidth=1.5,
         label=rf"Final joint inversion (RMSE $s$/$p$ = {rms_fin[1]:.2f}/{rms_fin[2]:.2f}%)")
ax2.plot(wl2, fit2_p, color=CLR_FIT, linestyle="-", linewidth=1.5)
ax2.plot(wl2, meas2_s, color=CLR_MEAS, linestyle=":", linewidth=1.0, alpha=0.75, label=r"Measured ($s$ and $p$)")
ax2.plot(wl2, meas2_p, color=CLR_MEAS, linestyle=":", linewidth=1.0, alpha=0.75)
label_polarizations(ax2, wl2, fit2_s, fit2_p, CLR_FIT)
ax2.set_ylabel(r"Reflectance (%)", fontsize=8.5, weight="bold")
ax2.set_title(r"(b) Filter 2 (Beam Splitter 50/50, 16 layers, 45$^\circ$ oblique incidence)", fontsize=9.0, weight="bold", loc="left")
ax2.set_xlim(1500, 4000)
ax2.set_ylim(0, 100)
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(loc="upper right", fontsize=7.0, framealpha=0.9)

# Panel 3: Filter 3 (Beam Splitter 17 layers with 10.6 QW layer)
ax3 = fig.add_subplot(gs[2])
wl3 = evaluator_init.plans[3].measurement.wavelength_nm
meas3_s = evaluator_init.plans[3].measurement.value * 100
meas3_p = evaluator_init.plans[4].measurement.value * 100
init3_s = pred_init[3] * 100
init3_p = pred_init[4] * 100
fit3_s = pred_final[3] * 100
fit3_p = pred_final[4] * 100

ax3.plot(wl3, init3_s, color=CLR_INIT, linestyle="--", linewidth=1.0, alpha=0.85,
         label=rf"Initial start (RMSE $s$/$p$ = {rms_init[3]:.2f}/{rms_init[4]:.2f}%, fringes shifted)")
ax3.plot(wl3, init3_p, color=CLR_INIT, linestyle="--", linewidth=1.0, alpha=0.85)
ax3.plot(wl3, fit3_s, color=CLR_FIT, linestyle="-", linewidth=1.4,
         label=rf"Final joint inversion (RMSE $s$/$p$ = {rms_fin[3]:.2f}/{rms_fin[4]:.2f}%)")
ax3.plot(wl3, fit3_p, color=CLR_FIT, linestyle="-", linewidth=1.4)
ax3.plot(wl3, meas3_s, color=CLR_MEAS, linestyle=":", linewidth=0.9, alpha=0.75, label=r"Measured ($s$ and $p$)")
ax3.plot(wl3, meas3_p, color=CLR_MEAS, linestyle=":", linewidth=0.9, alpha=0.75)
label_polarizations(ax3, wl3, fit3_s, fit3_p, CLR_FIT)
ax3.set_ylabel(r"Reflectance (%)", fontsize=8.5, weight="bold")
ax3.set_title(r"(c) Filter 3 (Beam Splitter, 17 layers with a 10.6 $\lambda_0/4$ layer, "
              r"45$^\circ$ oblique incidence)", fontsize=9.0, weight="bold", loc="left")
ax3.set_xlim(1500, 4000)
ax3.set_ylim(0, 100)
ax3.grid(True, linestyle=":", alpha=0.5)
ax3.legend(loc="upper right", fontsize=7.0, framealpha=0.9)

# Panel 4: In situ Retrieved Refractive Indices vs Literature Start vs single-layer reference
ax4 = fig.add_subplot(gs[3])
wl_grid = np.linspace(1500, 4000, 251)
n_helios_si = study_true.materials["SiO2"].n_at(wl_grid)
n_helios_nb = study_true.materials["Nb2O5"].n_at(wl_grid)
n_lit_si = s.materials["SiO2"].n_at(wl_grid)
n_lit_nb = s.materials["Nb2O5"].n_at(wl_grid)

n_ret_si = n_lit_si + cs_si(wl_grid)
n_ret_nb = n_lit_nb + cs_nb(wl_grid)

# Plot Nb2O5
ax4.plot(wl_grid, n_lit_nb, color=CLR_INIT, linestyle="--", linewidth=1.3, label=r"Literature start (Franta 2024 / 2016)")
ax4.plot(wl_grid, n_ret_nb, color=CLR_FIT, linestyle="-", linewidth=1.8, label=r"Retrieved in situ spline (from 3 multilayers alone)")
ax4.plot(wl_grid, n_helios_nb, color=CLR_HELIOS, linestyle=":", linewidth=1.5, label=r"Single-layer determination (companion paper)")

# Plot SiO2
ax4.plot(wl_grid, n_lit_si, color=CLR_INIT, linestyle="--", linewidth=1.3)
ax4.plot(wl_grid, n_ret_si, color=CLR_FIT, linestyle="-", linewidth=1.8)
ax4.plot(wl_grid, n_helios_si, color=CLR_HELIOS, linestyle=":", linewidth=1.5)

# Annotation: the gap the SiO2 fit has to cross, from the literature film to the sputtered one
climb = float(n_ret_si[0] - n_lit_si[0])
ax4.annotate(rf"$\mathbf{{+{climb:.4f}}}$ at 1500 nm," + "\n" + r"literature film $\rightarrow$ sputtered film",
             xy=(1720, 0.5 * (n_lit_si[np.argmin(np.abs(wl_grid - 1720))]
                              + n_ret_si[np.argmin(np.abs(wl_grid - 1720))])),
             xytext=(1560, 1.63),
             arrowprops=dict(arrowstyle="->", color=CLR_FIT, lw=1.2),
             fontsize=7.5, weight="bold", color=CLR_FIT,
             bbox=dict(boxstyle="round,pad=0.3", fc="#F0FFF0", ec=CLR_FIT, lw=0.8))

ax4.text(1550, 2.26, r"$\mathbf{Nb_2O_5}$", fontsize=9, weight="bold", color="#153250")
ax4.text(1550, 1.49, r"$\mathbf{SiO_2}$", fontsize=9, weight="bold", color="#153250")
ax4.set_xlabel(r"Wavelength $\lambda$ (nm)", fontsize=8.5, weight="bold")
ax4.set_ylabel(r"Refractive index $n(\lambda)$", fontsize=8.5, weight="bold")
ax4.set_title(r"(d) Simultaneous in situ refractive index determination, from three multilayers alone",
              fontsize=9.0, weight="bold", loc="left")
ax4.set_xlim(1500, 4000)
ax4.set_ylim(1.35, 2.32)
ax4.grid(True, linestyle=":", alpha=0.5)
ax4.legend(loc="center right", fontsize=7.0, framealpha=0.9)

plt.tight_layout()
out_pdf = WORKSPACE / "03_FIGURES_PUBLICATION" / "fig09_insitu_robustness.pdf"
out_png = WORKSPACE / "03_FIGURES_PUBLICATION" / "fig09_insitu_robustness.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved figure successfully to {out_pdf} and {out_png}")
