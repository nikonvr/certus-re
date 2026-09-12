# certus_re — reverse engineering of multilayer coatings, with the degrees of freedom on the table

Open-source implementation, measured spectra and nominal designs accompanying:

> **Transferability of broadband UV–MWIR optical constants to the reverse engineering of
> sputtered multilayer filters on silicon**, F. Lemarchand and J. Lumeau (2026).

This is the second deposit of a two-part study. The first, *Certus Index Spline*, determined
the complex refractive index of the materials on single layers. This one asks whether that
determination transfers, unchanged, to the multilayer coatings produced on the same machine —
and it is built so that the question can be answered rather than assumed.

---

## Status — work in progress, 12 September 2026

**This repository is not the released deposit yet.** The optical model, the solver, the
reporting and the test suite are complete and verified; three items of the target
specification are not yet wired, and one deposited data file is known to be wrong. They are
listed here rather than discovered later.

| | State |
|---|---|
| Optical model, solver, reports, CLI, 99 tests | **done and verified** — see Section 6 |
| Joint inversion with shared coatings | **done** |
| Two-side-coated components | **done** |
| Beam aperture, piecewise-constant in wavelength | **done**, but only with *imposed* heights |
| `studies/volet2/indices/Si_Li1980.csv` | **WRONG** — flat at 3.5545, it is not Li's dispersion. To be regenerated from the formula, which gives 3.4804 at 1500 nm and 3.4271 at 4000 nm |
| Aperture heights released as parameters | declared and counted, **not yet assembled** into the parameter vector |
| Polarizer crosstalk `alpha`, `beta` | fields and documentation in `model.py`, **not yet used** by the forward model or the solver |
| Deposited studies | still in the unpolarized channel; the target is `s` and `p` resolved |
| Inversion window | to be fixed at **1200–4000 nm** for every sample (1000 nm falls below the pole of Li's formula at 1107 nm) |

Do not quote numbers produced by this revision.

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
PARAMETERS RELEASED
------------------------------------------------------------------------------
parameter block     count  status
----------------------------------------
layer thicknesses      17  released
                           BS45: 17
index correction        0  tabulated
                           n and k used as determined on the witness samples, not adjusted
beam aperture           0  imposed
                           total aperture 2.00, 2.00, 2.00 deg; step wavelengths (nm): 2530, 3700
substrate index         0  literature
                           propagated, not adjusted (silicon)
angle of incidence      0  nominal
                           angles used as set: 45 deg
----------------------------------------
total free parameters     17
data points           600
points per parameter   35.3
```

A held block still appears, with the reason it is zero. `0 (imposed)` and `0 (not used)` are
different statements, and the difference is the whole argument.

---

## 1. What it handles

| | |
|---|---|
| **Several samples at once** | inverted jointly, **sharing the coatings they have in common**: a run measured alone and the same run measured as one face of an assembled component are one set of unknowns constrained by two spectra, not two sets |
| **Two-side-coated components** | a coherent stack on each face, the substrate incoherent between them, exact cavity summation — not `R_front + R_rear` |
| **Any angle of incidence** | in `s`, `p`, or the unpolarized half-sum `a` |
| **Reflectance and transmittance** | including transmittance **relative to the bare substrate**, the quantity single-layer witnesses are measured in |
| **Finite beam aperture** | modelled as a **piecewise-constant function of wavelength**, stepped at the instrument's documented source and detector switchovers, and *imposed* rather than fitted |
| **Absorbing layers and substrates** | `n − ik` throughout, with the substrate's internal path treated explicitly |

Dependencies: **NumPy and SciPy**. Nothing else. The optical model is plain NumPy — a
seventeen-layer inversion takes about three seconds — so there is no just-in-time compiler to
install and no extension to build. A deposit a reviewer cannot install is a deposit nobody
checks.

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
> | **Runs** | 260317-034 (antireflection, 6 layers), 260317-035 (beam splitter, 17 layers) |
> | **Substrates** | double-side-polished silicon, 1 mm, for the components; sapphire for the witnesses |
> | **Instrument** | EssentOptics PHOTON RT, SN 38425, 185–5200 nm configuration |
> | **Campaign** | March 2026; the single-layer witnesses were deposited **less than one week** before the components, with no target change or chamber reconditioning in between |
>
> That last line is the condition that makes the transferability test legitimate. The
> witnesses are not archival data: they are witness samples of the production run, which is
> also the configuration of ordinary coating-shop practice.

Two determinations of the same films are distributed, because the article compares them:

- `*_published.csv` — from the relative transmittance of the witnesses, 250–5200 nm, with its
  uncertainty envelope, reproduced from the Volet 1 deposit. Beyond the last interference
  extremum, near 4840 nm, the index is no longer constrained by the fringe positions; the
  study files declare that as `valid_range_nm`, and a run that reaches past it says so.
- `*_campaign.csv` — the determination carried in the campaign workbooks, exploiting R and T,
  1000–5200 nm only.

Neither is silently preferred: each study names the one it uses.

---

## 3. Install and run

```bash
pip install -e .
python reproduce.py          # the whole self-check, about one minute
```

Invert one study and print its report:

```bash
python -m certus_re run studies/volet2/joint_campaign.json
```

Validate a study without inverting it — the fastest way to find out that a dispersion table
does not span a measurement, or that a declared free-parameter block contradicts the
instrument description:

```bash
python -m certus_re check studies/volet2/joint_campaign.json
```

From Python:

```python
from certus_re import load_study, count_free_parameters
from certus_re.solve import invert
from certus_re.report import text_report

study = load_study("studies/volet2/AR6_alone.json")
print(count_free_parameters(study).to_text())
result = invert(study)
print(text_report(result, study))
```

---

## 4. The input format

One JSON file describes the problem; CSV files hold the numbers. Both are readable and
diffable without this package, which matters for an archive: a reader in ten years should be
able to see what was measured without running anything. **Every path is relative to the JSON
file**, so a study directory can be moved or archived as a unit.

```json
{
  "name": "joint_campaign",
  "lambda0_nm": 1500.0,
  "materials":  { "Nb2O5": { "file": "indices/Nb2O5_H800_published.csv",
                             "uncertainty_column": "sigma_n",
                             "valid_range_nm": [250.0, 4840.0] } },
  "substrates": { "silicon": { "file": "indices/Si_Li1980.csv" } },
  "stacks": {
    "BS45": { "run": "260317-035", "layers": [["SiO2", 1.520588], ["Nb2O5", 1.698967]] }
  },
  "samples": [
    { "name": "BS45_alone",
      "substrate": { "material": "silicon", "thickness_mm": 1.0, "rear": "none" },
      "front_stack": "BS45",
      "measurements": [
        { "file": "spectra/BS45_R45deg.csv", "column": "R_a_pct", "units": "percent",
          "quantity": "R", "angle_deg": 45.0, "polarization": "a",
          "band_nm": [1000, 4000], "sigma": 0.0053 }
      ] }
  ],
  "instrument": { "beam_aperture_deg": 2.0, "aperture_band_edges_nm": [2530, 3700],
                  "aperture_mode": "imposed" },
  "free_parameters": { "thicknesses": ["*"], "index_correction": "none" }
}
```

Points worth knowing:

- **Columns are selected by name, never by position.** Two workbooks of one campaign were
  found to order their index columns differently; substituting by rank produced a
  forty-percent residual.
- **`units` is explicit.** A percent column read as a fraction is caught by the study
  validation, not by the physics.
- **`rear`** is `"bare"`, `"coated"` or `"none"`. `"none"` is a semi-infinite substrate, which
  is what a measurement whose rear-face contribution was removed at acquisition requires.
- **A stack named by two samples is one set of unknowns.** That is the mechanism behind joint
  inversion; the report lists which samples constrain which coating.
- **`sigma`** is the photometric uncertainty. In a joint inversion it is what weights each
  dataset by what it is worth rather than by how many points it happens to contain — a study
  with several samples and no declared uncertainty is refused.

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
| Incoherent cavity | `R = R_f + T_f T_f′ R_b τ² ⁄ (1 − R_f′ R_b τ²)` — exact, not `R_f + R_b` |
| Aperture | steps at the declared switchover wavelengths themselves; two-angle mean by default, Gauss–Legendre quadrature over the cone with `n_aperture_nodes` odd ≥ 3 |

---

## 6. What is verified, and how

`python reproduce.py` runs all of it. Nothing below is asserted in prose only.

**Against results known in closed form** — Fresnel reflectance at normal incidence; the
vanishing of `R_p` at Brewster's angle; the reflectance of a quarter-wave layer; the absentee
half-wave layer; `R + T = 1` for a transparent stack at every angle and both polarizations,
to 10⁻¹²; `T(air→substrate) = T(substrate→air)` by reciprocity, which is what allows the
incoherent cavity to be written as it is; a two-side-coated plate conserving energy to
10⁻¹⁵; a rear stack of zero thickness reproducing a bare rear face exactly.

**Against the data** — the two single-layer witnesses are reproduced to 0.18 % and 0.33 %,
which is the fit quality the Volet 1 determination reports for itself. Reproducing an
independent result with an independent implementation is a stronger statement than any
internal check.

**By round trip** — thicknesses are perturbed by one percent, the spectrum they produce is
computed, and the inversion is required to return them. It does, to **2·10⁻¹³ nm**. The same
test on the two-side-coated sample recovers all twenty-three thicknesses of both faces from
its single spectrum, to the same precision.

**By determinism** — every study is inverted twice and the results compared bit for bit.
There is no stochastic element anywhere: no random restarts, no shakes, no seeded
initialisation. The optimiser is a trust-region least squares started from the nominal design.

**Against the reference implementation** — see below.

---

## 7. Relation to the laboratory's production tool

The coatings of this study were first inverted with the laboratory's own tool. Its numerical
core was rewritten here for clarity and for the joint, two-sided model. `tests/test_reference.py`
imports that tool — **read-only** — and requires the two forward models to agree.

They agree to **1.1·10⁻¹⁴** on the real coatings, at 8° and 45°, in both polarizations, once
this package is put in *compatibility mode*. That mode exists for one reason, and the reason
is worth stating plainly:

> The reference kernel for oblique incidence is written for `N = n − ik`, and is fed indices
> stored as `n + ik`. With the `+i` phase convention it therefore **amplifies instead of
> absorbing**, and returns `R + T > 1` — up to 1.048 on a test stack. The test suite asserts
> both facts: that compatibility mode reproduces the reference exactly, and that the
> reference convention creates energy.

Pinning the difference down to exactly one cause is what makes the rewrite auditable. The
correction is not cosmetic, and its size scales with `k`, so it is reported per dispersion
dataset:

| dispersions used | largest change in computed R or T |
|---|---|
| campaign determination | **1.7·10⁻²** |
| published determination (larger `k`) | **1.9·10⁻¹** |

On the seventeen-layer coating at 45° the change in reflectance alone reaches 1.0·10⁻² with
the campaign dispersions — about twice the residual such an inversion reports — and 1.9·10⁻¹
with the published ones, where the thick central cavity amplifies the effect.

This package uses the physical convention by default. Compatibility mode is reachable only
from the comparison tools and is never used to produce a result.

---

## 8. Known limitations, stated rather than discovered

- **The two-side-coated component does not fit.** `studies/volet2/biface_unresolved.json`
  names the same two coating runs as the components measured alone, so it should add measured
  points and no unknown. It does not: the nominal design reproduces its spectrum to 30 %, and
  releasing all twenty-three thicknesses over a ±50 % window reaches only 5 %, with seven
  layers ending on their bound. The pairing of designs and measurements for that sample needs
  to be confirmed at the bench. It is deposited alone, and excluded from the joint study, so
  that the inconsistency is reproducible rather than averaged away.
- **The forward model treats each layer as homogeneous, isotropic and abruptly bounded.**
  Roughness, thickness non-uniformity across the spot, index gradients along the growth
  direction and interfacial transition layers are not represented; where they are present the
  optimizer compensates for the model deficiency by displacing the retrieved thicknesses.
- **The default aperture average is over two angles.** It is what the reference
  implementation does, and it is kept as the default so the two can be compared. Its error
  grows with the curvature of `R(θ)` and therefore differs between polarizations. Set
  `n_aperture_nodes` to 5 or 9 for a quadrature over the cone.
- **The angle of incidence is the most sensitive quantity of an oblique inversion.** At 45°
  the retrieved total thickness moves by about 35 nm per degree. Releasing `angle_offset`
  will absorb almost any model error; the report says so when it is released.
- **The silicon dispersion is tabulated over 1000–4000 nm only**, which bounds the exploitable
  band. The study validation refuses a measurement that reaches outside it rather than
  holding the index at its edge value.
- **The photometric uncertainty declared for the components is the manufacturer's
  specification**, not a measured repeatability. The campaign contains a repeated acquisition
  of one specimen from which a proper figure can be derived; until it is, the declared value
  is visible in the study file rather than buried in the solver.

---

## 9. Contents

```
certus_re/          the package
  model.py            stacks, samples, measurements, instrument, free parameters
  dispersion.py       tabulated indices, with provenance and traced extrapolation
  physics.py          transfer matrices, incoherent plate, beam aperture
  forward.py          predicting each measurement from a set of thicknesses
  solve.py            the inversion, built from the declared blocks
  dof.py              the parameter budget
  report.py           text and JSON reports
  io_study.py         reading a study from JSON and CSV
  cli.py              run / check / predict
studies/volet2/     the campaign: designs, spectra, dispersions, study files
tests/              the test suite
tools/              rebuilding the study from the laboratory archives; comparing
                    with the reference implementation
reproduce.py        one command that checks the whole deposit
```

---

## 10. Citation

See `CITATION.cff`. The concept DOI always resolves to the latest version.

Code is released under the MIT licence, data under CC BY 4.0; see `LICENSE`. The optical
constants of the witnesses are reproduced from the Volet 1 deposit and should be cited
through it.
