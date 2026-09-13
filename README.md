# certus_re — reverse engineering of multilayer coatings, with the degrees of freedom on the table

Open-source implementation, measured spectra and nominal designs accompanying:

> **Transferability of Single-Layer Optical Constants to Complex Multilayer Reverse Engineering in the MWIR**,
> F. Lemarchand and J. Lumeau (2026).

This is the second deposit of a two-part study. The first, *Certus Index Spline*, determined
the complex refractive index of the materials on single layers. This one asks whether that
determination transfers, unchanged, to the multilayer coatings produced on the same machine —
and it is built so that the question can be answered rather than assumed.

---

### Status — September 2026

The model specified for the article is now complete: every parameter block it calls for is
wired, counted and reported, and `python reproduce.py` runs the whole deposit end to end.

| | State |
|---|---|
| Optical model, solver, reports, CLI, **124 tests** | **done and verified** — see Section 6 |
| Joint inversion with shared coatings | **done** — the five samples of the campaign in one run (41 thicknesses, 2266 points) |
| Technological process prior (Tikhonov / MAP) | **done** — `process_prior_pct: 0.5%`, bounding layer departures in optical thickness OT |
| Two-side-coated components | **supported and verified in test suite** (all 3 experimental filters are single-side coated) |
| Silicon dispersion | **done** — evaluated from Li's 1980 formula in `certus_re.dispersion`, tabulated into the deposit by the build tool, and checked against its published control values by the test suite |
| Beam aperture, piecewise-constant in wavelength | **done**, imposed or released, one total aperture per band |
| Polarizer crosstalk `alpha`, `beta` | **done** — applied to the computed spectra, one pair for the whole instrument |
| Deposited studies | **`s` and `p` resolved** for both 45° beam splitters (BS45 and BS17) |
| Inversion window | fixed at **1500–4000 nm** for every sample; 1000 nm falls below the pole of Li's formula at 1107 nm |
| Uncertainty on every retrieved layer thickness | **done** — from `s²(JᵀJ)⁻¹` at the solution |
| Photometric uncertainty | **measured**, not specified — 0.0014 from a repeated acquisition, where the manufacturer's figure is 0.0053 |
| Parameter search space | **unconstrained MAP** ($D > 0$), regularized by the technological process prior $\sigma_{\mathrm{proc}} = 0.5\%$ in optical thickness |

Two things a reader should know before quoting a number from it:

- **The campaign covers three manufactured optical components.** Alongside the two reference single
  layers (SiO₂ and Nb₂O₅), three multilayer filters produced in the same chamber are
  analyzed:
  - **Filter 1 (AR6)**: a 6-layer antireflection coating targeting $R < 1\%$ across $3.0\text{--}4.0\,\mu\mathrm{m}$;
  - **Filter 2 (BS45)**: a 16-layer beam splitter targeting a $50/50$ split ($48\% < R_a < 52\%$) across $2.0\text{--}5.0\,\mu\mathrm{m}$;
  - **Filter 3 (BS17)**: a 17-layer beam splitter containing an unusually thick layer of $2.7\,\mu\mathrm{m}$ ($10.6$ quarter waves), much thicker than all other layers, generating rapid spectral fringes.
  All were acquired on raw spectrophotometric data with identical 220 µm slit and 5 mm spot settings, without empirical photometric scaling.
- **The problem is ill-posed, and the deposit regularizes it physically.** Sputtered multilayer
  reverse engineering exhibits parameter compensation. By declaring a technological process prior
  `process_prior_pct: 0.5%` grounded in the repeatability of magnetron sputtering plants,
  stable quasi-solutions (Tikhonov regularization / Bayesian MAP) are selected in the vicinity of
  the nominal designs, keeping layer departures below 4–6% in optical thickness across all
  three filters. Widening the search window to ±50% without prior increases
  layer-wise dispersion fourfold while barely changing photometric residuals.
- **Reference single layers remain strictly indispensable.** Releasing refractive indices during 3-filter inversion starting from non-dispersive flat lines recovers the optical constants directly within the *Optics Continuum* uncertainty corridor ($\pm 0.005$). However, because the photometric residual difference ($\Delta\mathrm{RMSE} \le 0.029\%$) is buried under instrumental repeatability noise ($\sigma_{\mathrm{phot}} \approx 0.14\%$), multilayer spectrophotometry alone cannot untangle the physical indices from a subtle $0.3\%$ thickness trade-off. Reference single layers provide the necessary uncoupled metrological anchor.
- **Two of the three aperture parameters end on their bound** in every study that releases
  them, with uncertainties of one to two degrees. Between 2 530 and 4 000 nm these coatings
  are spectrally flat at 45°, so the cone average barely moves there and the data do not
  determine the aperture. The run report says so; the number is not a measurement.

---

## 0. Why this package exists

Multilayer reverse engineering conventionally adjusts layer thicknesses **and** dispersion
corrections inside the same optimization. When the residual falls, the improvement cannot be
attributed to the thickness correction — the physically meaningful outcome — rather than to
the index correction, which silently rewrites the quantity one claims to validate.

> *A reverse-engineering procedure that is free to reshape n(λ) cannot, by construction,
> establish that a previously determined n(λ) was correct.*

So this package does not let a run move anything it has not declared. The parameter blocks a
run is allowed to release are written in the study file, counted before the run starts by the
same code that assembles the parameter vector, and printed next to every residual:

```
parameter block      count  status
-----------------------------------------
layer thicknesses       41  released
                             WIT_SiO2: 1, WIT_Nb2O5: 1, Filter 1 (AR6): 6, Filter 2 (BS45): 16, Filter 3 (BS17): 17;
                             unconstrained MAP (process prior 0.5 % in OT)
index correction         0  tabulated
                             n and k transferred from reference single layers, held fixed
beam aperture            0  imposed
polarizer crosstalk      0  none
substrate index          0  literature
                             propagated, not adjusted (sapphire, silicon)
angle of incidence       0  nominal
                             angles used as set: 0, 8, 45 deg
-----------------------------------------
total free parameters     41
data points             2266
points per parameter    55.3
```

A held block still appears, with the reason it is zero. `0 (imposed)` and `0 (not used)` are
different statements, and the difference is the whole argument. A block that is *countable but
not implemented* is refused outright rather than counted at zero — that trap was live here
once, when declaring the aperture released counted three parameters that could not move.

The same discipline runs to the end of the report: every released parameter comes back with
the uncertainty the data imply on it, and every layer's departure from its design is quoted in
units of its own error bar. A departure smaller than its own uncertainty is not a departure.

---

## 1. What it handles

One component or several at once, each coated on one face or on both. That is the whole
matrix, and every cell of it is exercised by a deposited study — `tests/test_capabilities.py`
fails if one stops being covered, because a claim in a README that nothing checks is worth
nothing.

| | **One face coated** | **Both faces coated** |
|---|---|---|
| **One component** | `02_AR6_alone` (6 layers), `04_BS45_resolved` (16 layers, `s` and `p`), `05_BS17_resolved` (17 layers, `s` and `p`) | synthesized in test fixtures — two coherent stacks, incoherent substrate, exact multiple-reflection summation |
| **Several components** | `01_witnesses` (two specimens, 1 thickness each), `07_joint_campaign` (five specimens, **41 unknowns**, 2266 data points) | demonstrated in test fixtures (`examples/several_components_two_faces.json`, `tests/test_capabilities.py`) |

Two-side coated components and synthetic campaigns are demonstrated on data generated in
the test fixture: two samples, one of them two-sided, sharing a coating, and all its thicknesses
required to come back. **In the experimental campaign itself, there is no two-side-coated component**:
all three filters are single-side coated on silicon with ground rear faces.

| | |
|---|---|
| **Several samples at once** | inverted jointly under a shared technological process prior ($\sigma_{\mathrm{proc}} = 0.5\%$), or sharing coatings across multiple specimens: all unknowns constrained jointly |
| **Two-side-coated components** | supported by the core: a coherent stack on each face, the substrate incoherent between them, exact multiple-reflection summation — not `R_front + R_rear` |
| **Any angle of incidence** | in `s`, `p`, or the unpolarized channel `a` |
| **Reflectance and transmittance** | including transmittance **relative to the bare substrate**, the quantity single-layer witnesses are measured in |
| **Finite beam aperture** | modelled as a **piecewise-constant staircase function of wavelength**, stepped at the instrument's documented switchovers (InGaAs/PbSe detector at 2530 nm, halogen/IR source at 3700 nm); the steps are imposed, the heights are imposed (nominal 2.0°) or released |
| **Polarizer leakage** | `R_s^meas = (1−α)R_s + αR_p`, `R_p^meas = (1−β)R_p + βR_s`, applied to the **computed** spectra, with one pair of coefficients for the whole instrument |
| **Absorbing layers and substrates** | `n − ik` throughout, with the substrate's internal path treated explicitly |
| **What the data determine** | the uncertainty on every retrieved thickness and quarter wave, from `s²(JᵀJ)⁻¹` at the solution |

Dependencies: **NumPy and SciPy**. Nothing else. The optical model is plain NumPy — the
sixteen-layer beam splitter inverts in a few seconds, the whole five-sample campaign in under
twenty — so there is no just-in-time compiler to install and no extension to build. A deposit
a reviewer cannot install is a deposit nobody checks.

---

## 2. Provenance of the data — read this first

> The spectra distributed under `studies/` are **not** generic examples. They characterize
> specific coatings, produced on a specific machine, with a specific process, at a specific
> time:
>
> | | |
> |---|---|
> | **Laboratory** | Institut Fresnel, Marseille, France |
> | **Deposition system** | Bühler Leybold Optics **HELIOS 800** |
> | **Process** | Plasma-assisted reactive magnetron sputtering (PARMS), SiO₂ / Nb₂O₅ |
> | **Runs** | 260317-034 (Filter~1 AR6, 6 layers), 260317-035 (Filter~2 BS45, 16 layers), 260319-037 (Filter~3 BS17, 17 layers) |
> | **Substrates** | silicon, 1 mm: **ground on the rear face** for all three multilayer filters; sapphire, polished on both faces, for the single-layer witnesses (bare rear face) |
> | **Instrument** | EssentOptics PHOTON RT, SN 38425, 185–5200 nm configuration, 220 µm slit, 5 mm spot, ten averages |
> | **Campaign** | March 2026; the single-layer witnesses were deposited **less than one week** before the components, with no target change or chamber reconditioning in between |
>
> That last line is the condition that makes the transferability test legitimate. The
> witnesses are not archival data: they are witness samples of the production run, which is
> also the configuration of ordinary coating-shop practice.

Three things about the data are worth knowing before using them.

**The rear face is not the same on every sample, and it is the most expensive thing to get
wrong.** All three multilayer components sit on silicon substrates whose rear face is *ground*: it
scatters the return of the second interface out of the collected beam, which is the classical
way to isolate the response of the front face and is what justifies modelling them as
semi-infinite. **There is no two-side-coated component in this study**: all three filters are
single-side coated. Modelling a ground rear face as a polished one takes the computed reflectance of the
antireflection coating from 0.6 % to 25.1 % at 8° — and the counter-experiment that measures that is
deposited under `studies/volet2/counter_experiments/`. The single-layer witnesses sit on double-side-polished
sapphire plates and are modelled with a bare rear face.

**One determination of the optical constants is distributed**, the one published with Volet 1,
from the relative transmittance of the witnesses, 250–5200 nm, with its uncertainty envelope.
It is the determination whose transferability the article tests; distributing a second one
beside it would blur that statement. Beyond the last interference extremum, near 4840 nm, the
index is no longer constrained by the fringe positions; the study files declare that as
`valid_range_nm`, and a run that reaches past it says so.

**Silicon and sapphire are computed, not tabulated by hand.** Li's 1980 formula and Malitson's
Sellmeier coefficients are evaluated in `certus_re.dispersion`, checked against their published
control values by the test suite, and written into the deposit by the build tool so that the
archived CSV can still be read without running anything.

Every measurement also carries the settings the spectrophotometer recorded for it — the
acquisition identifier, the date and time, the stage and detector angles, the polarizer, the
range, the sampling pitch, the averaging count, the slit width and the spot size. A second acquisition of the
antireflection coating exists, taken the day before with a **240 µm slit and a 6 mm spot**
where every other spectrum of the campaign used 220 µm and 5 mm. That is a different beam
geometry, deposited beside the retained one, as a second column, and not used.

---

## 3. Install and run

```bash
pip install -e .
python reproduce.py          # the whole self-check, a few minutes
```

Invert one study and print its report:

```bash
python -m certus_re run studies/volet2/07_joint_campaign.json
```

Validate a study without inverting it:

```bash
python -m certus_re check studies/volet2/07_joint_campaign.json
```

From Python:

```python
from certus_re import load_study, count_free_parameters
from certus_re.solve import invert
from certus_re.report import text_report

study = load_study("studies/volet2/02_AR6_alone.json")
print(count_free_parameters(study).to_text())
result = invert(study)
print(text_report(result, study))
```

### The studies are a ladder, not a folder

Each rung adds **exactly one thing** to the one before it, so that every change in a residual
can be attributed to one cause. That is the whole method: a procedure that changes several
things at once cannot say which one paid.

| Rung | Study | Free | Points | Pts/par | What it answers |
|---|---|---|---|---|---|
| 0 | `00_prediction` | **0** | 2266 | — | does the determination transfer at all, with nothing adjusted? |
| 1 | `01_witnesses` | 2 | 516 | 258 | does this implementation reproduce the determination it is testing? |
| 2 | `02_AR6_alone` | 6 | 250 | 42 | what does ordinary reverse engineering give on six layers? |
| 3 | `03_BS45_unpolarized` | 16 | 250 | 16 | and on sixteen, in the unpolarized channel? |
| 3 | `04_BS45_resolved` | 16 | 500 | 31 | what appears when the two channels are separated? |
| 3 | `05_BS17_resolved` | 17 | 1000 | 59 | how does the seventeen-layer filter with its thick layer behave? |
| 4 | `06_BS17_aperture` | 20 | 1000 | 50 | how much of it is the finite beam cone? |
| 5 | `07_joint_campaign` | **41** | 2266 | 55 | can one set of unknowns account for all five samples at once? |

Rung 0 is not a fit. It releases nothing, so its residual cannot have been bought, and every
other rung is read against it.

`studies/volet2/counter_experiments/` holds seven studies that are **wrong by construction** —
the index released, the angle shifted by a quarter of a degree, the search window opened wide,
the spectral mesh decimated by eight, the rear face mismodelled, the design reversed, the flat silicon
table used. They exist so that the cost of each mistake is a number a reader can reproduce rather than
a warning in a paper.

```bash
python tools/article_tables.py     # runs the ladder, writes results/ and the article's tables
```

---

## 4. The input format

One JSON file describes the problem; CSV files hold the numbers. Both are readable and
diffable without this package, which matters for an archive: a reader in ten years should be
able to see what was measured without running anything. **Every path is relative to the JSON
file**, so a study directory can be moved or archived as a unit.

```json
{
  "lambda0_nm": 1500.0,
  "materials":  { "Nb2O5": { "file": "indices/Nb2O5_H800_published.csv",
                             "uncertainty_column": "sigma_n",
                             "valid_range_nm": [250.0, 4840.0] } },
  "substrates": { "silicon": { "file": "indices/Si_Li1980.csv",
                               "valid_range_nm": [1200.0, 6000.0] } },
  "stacks": {
    "BS45": { "run": "260317-035", "layers": [["Nb2O5", 1.691], ["SiO2", 2.358395]] }
  },
  "samples": [
    { "name": "BS45_alone",
      "substrate": { "material": "silicon", "thickness_mm": 1.0, "rear": "none" },
      "front_stack": "BS45",
      "measurements": [
        { "file": "spectra/BS45_R45deg.csv", "column": "R_s_pct", "units": "percent",
          "quantity": "R", "angle_deg": 45.0, "polarization": "s",
          "band_nm": [1500, 4000], "sigma": 0.0014,
          "acquisition": { "acquisition": "260317-035s",
                           "datetime": "17-03-2026 16:31:24",
                           "stage_angle_deg": "45", "detector_angle_deg": "90",
                           "polarizer": "S", "sampling_pitch_nm": "10",
                           "slit_width_um": "220", "spot_size_mm": "5" } }
      ] }
  ],
  "instrument": { "preset": "photon_rt_5200" },
  "free_parameters": { "thicknesses": ["*"],
                       "thickness_tolerance": null,
                       "process_prior_pct": 0.5,
                       "index_correction": "none",
                       "aperture": "imposed", "crosstalk": "none" }
}
```

Points worth knowing:

- **Columns are selected by name, never by position.** Two workbooks of one campaign were
  found to order their index columns differently; substituting by rank produced a
  forty-percent residual.
- **`units` is explicit.** A percent column read as a fraction is caught by the study
  validation, not by the physics.
- **`rear`** is `"bare"`, `"coated"` or `"none"`. `"none"` is a semi-infinite substrate, which
  is what a ground rear face requires.
- **A stack named by two samples is one set of unknowns.** That is the mechanism behind joint
  inversion; the report lists which samples constrain which coating.
- **`sigma`** is the photometric uncertainty. In a joint inversion it is what weights each
  dataset by what it is worth rather than by how many points it happens to contain — a study
  with several samples and no declared uncertainty is refused. Here it is *measured*, from a
  repeated acquisition, not taken from a specification; the difference was a factor of four
  and it reversed a conclusion.
- **The problem is ill-posed, so something has to select one solution out of the many that fit
  equally well.** Two mechanisms are available, and they are alternatives rather than companions.

  - **`process_prior_pct`** — what every study of this deposit uses. It is the relative standard
    deviation, in percent, with which the plant reproduces an optical thickness, and it turns into a
    smooth quadratic cost: a layer sitting one $\sigma_{\mathrm{proc}}$ from its design costs as much
    as one measured point missing by one noise floor. Nothing is forbidden, so nothing is ever
    clipped against a wall, and every layer keeps a meaningful uncertainty. Set it from how the
    coating was made — $0.5\%$ here — and never from what lowers the residual.
  - **`thickness_tolerance`** — the older, harder mechanism: a search window on each thickness, as a
    fraction of nominal, outside which the optimizer may not go. It still works and the
    counter-experiment `search_window_wide` uses it, but a layer that ends on its bound has no
    usable local uncertainty, and the report says so. Set it to `null` when using the prior.

  Declaring both is accepted; the prior then acts inside the window. Declaring neither leaves the
  problem unregularized, which is legitimate only as an experiment — the deposit contains one.
- **`instrument` can be named rather than described.** Two presets ship with the package.
  `"preset": "photon_rt_5200"` carries the EssentOptics PHOTON RT geometry used throughout the
  article — a $2.0^\circ$ beam half-angle, band edges at 2530 and 3700 nm where the detector and
  the source change over, and the interval within which that half-angle may be released.
  `"preset": "manual"` asserts nothing and leaves every field to you, which is the one to start from
  on any other spectrophotometer. Any field can be overridden beside the preset —
  `{"preset": "manual", "beam_aperture_deg": 1.2}` — and the override is echoed in the report. An
  unknown preset is refused by name rather than quietly replaced by a default, and a study that
  names no preset keeps the explicit fields it gives.
- **`free_parameters` and `instrument` must agree.** Declaring `aperture: "fitted"` against an
  instrument in `"imposed"` mode is refused, not silently ignored. So is releasing the leakage
  in a study with no polarization-resolved measurement, or releasing a per-band aperture when
  one of the bands carries no oblique point: a block nothing can constrain is a block that
  widens every error bar for nothing.
- **`acquisition`** carries the instrument settings recorded with the spectrum. Nothing reads
  it during an inversion; it is there because a measurement without its conditions is not a
  measurement, and because the article's table of samples is generated from it.
- **`stride`** keeps one point in *n*. It exists for one counter-experiment and says so.

---

## 5. Conventions of the optical model

Stated once, used everywhere, and asserted in the test suite.

| | |
|---|---|
| Layer order | `layers[0]` is **adjacent to the substrate** — the first layer deposited |
| Refractive index | `N = n − ik`, `k ≥ 0`, inside the model; tables are stored as `n + ik` and conjugated once, on entry |
| Phase | `δ = 2π N d cos θ ⁄ λ` |
| Tilted admittance | `η = N cos θ` (s), `η = N ⁄ cos θ` (p) |
| Angles | Snell applied with the complex layer index; incident and exit media use the real part, absorption in the substrate being carried by an explicit path factor |
| Incoherent substrate plate | `R = R_f + T_f T_f′ R_b τ² ⁄ (1 − R_f′ R_b τ²)` — exact, not `R_f + R_b` |
| Aperture | steps at the declared switchover wavelengths themselves; two-angle mean by default, Gauss–Legendre quadrature over the cone with `n_aperture_nodes` odd ≥ 3 |

---

## 6. What is verified, and how

`python reproduce.py` runs all of it. Nothing below is asserted in prose only.

**Against results known in closed form** — Fresnel reflectance at normal incidence; the
vanishing of `R_p` at Brewster's angle; the reflectance of a quarter-wave layer; the absentee
half-wave layer; `R + T = 1` for a transparent stack at every angle and both polarizations,
to 10⁻¹²; `T(air→substrate) = T(substrate→air)` by reciprocity, which is what allows the
incoherent substrate plate formula to be written as it is; a two-side-coated plate conserving energy to
10⁻¹⁵; a rear stack of zero thickness reproducing a bare rear face exactly.

**Against the data** — over the inversion window, the two single-layer witnesses are
reproduced to **0.091 %** and **0.153 %**, which is the fit quality the Volet 1 determination
reports for itself. Reproducing an independent result with an independent implementation is a
stronger statement than any internal check.

**By round trip** — thicknesses are perturbed by one percent, the spectrum they produce is
computed, and the inversion is required to return them. It does, to **2·10⁻¹³ nm** on the
six-layer coating and **5·10⁻¹⁰ nm** on the sixteen-layer one. The same test on the
two-side-coated sample recovers all **twenty-two** thicknesses of both faces from its single
pair of spectra, to 2·10⁻¹² nm.

**On the instrument model** — a symmetric polarizer leakage leaves the half-sum
`(R_s + R_p)/2` invariant to **2·10⁻¹⁶**, which is the statement that makes the leakage
observable only in the resolved channels; an asymmetric one shifts it by exactly
`(α−β)(R_p−R_s)/2`; and a spectrum generated from a known aperture and a known pair of
coefficients returns all three, to 10⁻⁵, together with its sixteen thicknesses.

**By determinism** — every study of the ladder is inverted twice and the results compared bit
for bit. There is no stochastic element anywhere: no random restarts, no shakes, no seeded
initialisation. The optimiser is a trust-region least squares started from the nominal design.

**Against the reference implementation** — see below.

---

## 7. Relation to the laboratory's production tool

The coatings of this study were first inverted with the laboratory's own tool. Its numerical
core was rewritten here for clarity and for the joint, two-sided model. `tests/test_reference.py`
imports that tool — **read-only** — and requires the two forward models to agree.

They agree to **6.6·10⁻¹⁵** on the real coatings, at 8° and 45°, in both polarizations, once
this package is put in *compatibility mode*. That mode exists for one reason, and the reason
is worth stating plainly:

> The reference kernel for oblique incidence is written for `N = n − ik`, and is fed indices
> stored as `n + ik`. With the `+i` phase convention it therefore **amplifies instead of
> absorbing**, and returns `R + T > 1` — up to 1.048 on a test stack. The test suite asserts
> both facts: that compatibility mode reproduces the reference exactly, and that the
> reference convention creates energy.

Pinning the difference down to exactly one cause is what makes the rewrite auditable. The
correction is not cosmetic, and its size scales with `k`: on the coatings of this campaign,
with the published dispersions, it reaches **4.2·10⁻²** in reflectance on the beam splitter at
45° — about an order of magnitude more than the residual such an inversion reports.

This package uses the physical convention by default. Compatibility mode is reachable only
from the comparison tools and is never used to produce a result.

---

## 8. Known limitations, stated rather than discovered

- **Two of the three aperture parameters are not determined by these data.** Between 2530 and
  4000 nm both 45° coatings are spectrally flat, so the cone average barely moves and the
  fitted aperture walks to a bound with an uncertainty of one to two degrees. The run report
  lists it among the parameters that ended on their bound, and the quoted uncertainty says the
  rest. Only the first band, below 2530 nm, is measured: 2.38 ± 0.08°.
- **The forward model treats each layer as homogeneous, isotropic and abruptly bounded.**
  Roughness, thickness non-uniformity across the spot, index gradients along the growth
  direction and interfacial transition layers are not represented; where they are present the
  optimizer compensates for the model deficiency by displacing the retrieved thicknesses.
- **The default aperture average is over two angles.** It is what the reference
  implementation does, and it is kept as the default so the two can be compared. Its error
  grows with the curvature of `R(θ)` and therefore differs between polarizations. Set
  `n_aperture_nodes` to 5 or 9 for a quadrature over the cone.
- **The angle of incidence is the most sensitive quantity of an oblique inversion**, and it is
  not released: a fitted angle would absorb almost any model error. Its effect is measured
  instead, by imposing it a quarter of a degree away — which moves the retrieved total
  thickness of the beam splitter by 6.4 nm while leaving the residual where it was. Study
  `counter_experiments/angle_shifted.json`.
- **The antireflection design is recorded to two decimals of a quarter wave.** On its third
  layer, whose nominal is 0.41, a rounding of 0.005 is 1.2 % — the order of magnitude of the
  departures reported for that coating. The campaign's own publication workbook quotes
  physical thicknesses to three decimals, but they are derived from the same two-decimal
  quarter waves, so they carry no further information.
- **The photometric uncertainty is estimated from one repeated acquisition**, the pair
  `2600316-033-mono` / `mono repet`, recorded four minutes apart at 8° without a polarizer.
  It gives $\sigma=0.00136$ over 1500--4000 nm and replaces the manufacturer's 0.0053 figure
  in the article studies. Because the polarized 45° channels transmit less flux, this estimate
  is explicitly treated as a lower bound on their uncertainty rather than as a complete
  repeatability study.
- **Uncertainties are scaled by `s² = χ²/(m−n)`.** That makes them independent of an overall
  factor on the declared `σ` and widens them when the model does not reproduce the data to
  within it — the honest direction, but it means an error bar here reports what the *fit*
  achieved, not what the photometry promised. `s²` is printed next to the table.

---

## 9. Contents

```
certus_re/          the package
  model.py            stacks, samples, measurements, instrument, free parameters
  dispersion.py       tabulated indices and the formulas of the substrates
  physics.py          transfer matrices, incoherent plate, beam aperture
  forward.py          predicting each measurement: cone average, polarizer leakage
  solve.py            the inversion, built from the declared blocks, and its covariance
  dof.py              the parameter budget
  report.py           text and JSON reports
  io_study.py         reading a study from JSON and CSV
  cli.py              run / check / predict
studies/volet2/     the ladder: designs, spectra, dispersions, eight study files
  counter_experiments/  seven studies that are wrong on purpose
tests/              the test suite, 124 tests
tools/              build_study_volet2.py       rebuild the deposit from the archives
                    article_tables.py           run the ladder, write the article's tables
                    article_figures.py          generate the 7 publication figures
                    witness_necessity_test.py   verify self-consistent index retrieval vs witnesses
                    search_window_scan.py       scan ill-posedness and prior regularization
                    compare_with_reference.py   non-regression against the laboratory tool
results/            what article_tables.py writes: three LaTeX tables, designs.csv,
                    article_numbers.json
reproduce.py        one command that checks the whole deposit
```

---

## 10. Citation

See `CITATION.cff`.

**There is no DOI in this repository yet, and none is quoted.** Zenodo mints the concept DOI
when the first GitHub release is published; until that happens, stating one here would be the
only invented number in the deposit. `RELEASE.md` is the checklist that fills it in — in
`CITATION.cff`, in this section, and in the manuscript — and the concept DOI, once minted,
always resolves to the latest version.

Code is released under the MIT licence, data under CC BY 4.0; see `LICENSE`. The optical
constants of the witnesses are reproduced from the Volet 1 deposit and should be cited
through it.
