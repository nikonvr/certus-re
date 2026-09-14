# certus_re 1.1.1

Metadata only. 1.1.0 was archived before Zenodo had minted its DOI, so the `CITATION.cff` and
`README.md` inside that archive could not state it. This release carries them completed:
concept DOI [10.5281/zenodo.22756244](https://doi.org/10.5281/zenodo.22756244), which always
resolves to the latest version.

No code, data or result changes from 1.1.0. What follows describes what 1.1.0 brought, and
still applies.

---

Companion code, measured spectra and reproduction scripts for the Optics Express paper
*Open-source reverse engineering of MWIR multilayer coatings: multi-angle polarized reflectance
determines the optical constants, the individual layer thicknesses require a priori process
knowledge*.

Certus RE inverts measured spectrophotometric data for the layer thicknesses of a finished
coating, and optionally for the optical constants of its materials. Every quantity the fit is
allowed to move is named in a study file rather than chosen inside the optimizer, so that a
number can be re-derived rather than believed.

Two runtime dependencies, NumPy and SciPy. No compiled extension. Source under MIT, data under
CC BY 4.0.

---

## What 1.1.0 adds

### The result the paper is built on

Four hundred and one inversions of three dissimilar components together — a six-layer
antireflection coating and two beam splitters — started from refractive indices measured on
foreign equipment, with the thickness prior switched off entirely and the starting geometries
perturbed by up to 20 per cent.

Of those, 319 converge, and every one returns the same dispersions: 1.4764 for SiO₂ and 2.2432
for Nb₂O₅ at 1500 nm, agreeing between runs to better than 10⁻⁶, and landing within 1.3 standard
deviations of the curves measured on dedicated single layers. The same runs leave the individual
layer thicknesses 4.2 per cent rms from the design, in anti-correlated adjacent pairs reaching
+11.9 and −12.6 per cent.

Multi-angle polarization-resolved photometry therefore determines the material and not the
geometry. Reference single layers can be dropped for the optical constants; a declared process
prior is needed for the thicknesses.

    tools/insitu_multistart_noprior.py     the 401 runs
    tools/insitu_multistart_prior.py       the same with the prior restored
    tools/insitu_convergence_traces.py     residual histories
    tools/plot_insitu_robustness.py        figure 9
    tools/plot_insitu_convergence.py       figure 10
    tools/plot_insitu_index_comparison.py  figure 11
    tools/test_sigma_ot_robustness.py      index retrieval against the prior width
    tools/test_5samples_noprior.py         the whole campaign, unconstrained

### Every number now has a file behind it

`tools/audit_article_numbers.py` computes the eight families of figures that the manuscript
quoted but no script deposited: fringe contrast of the four angular models, what removing the
prior costs on each component, a constant-index control, the collimated-beam test, Filter 2
against its specification, the common-mode split of the Filter 3 residual, the thickness shift
due to the spline, and the shape of that residual. It writes `results/audit_numbers.json`.

Four of those numbers were wrong and are corrected in this release.

### Naming

Witness samples are called reference single layers throughout — file names, study identifiers,
sample labels, the data licence and the help page. No numerical content changes with the rename.

---

## Corrections

| | was | is |
|---|---|---|
| Filter 2 against its specification | misses by 0.4 point, band 47.6–52.5 % | misses by 0.26 point, band 47.74–52.45 % |
| thickness shift from the spline solution | 0.3 % on average | 0.95 % |
| Filter 3 residual under polarization averaging | three quarters of its power | three fifths |
| shape of that residual | does not follow the wavelength scale | follows the spectral derivative at −0.45 in *p*, against −0.05 for the level |
| two-ray versus converged quadrature | 0.14 point | 1.08 point rms; 0.14 was the out-of-plane term alone |

Three defects in the figures, found the same way:

- **figure 9** drew Filter 1 on a transmittance axis, so the panel was empty; it was generated
  with the process prior still active, which is not the experiment its caption describes; and its
  six residuals were typed into the legend labels, two of them wrong;
- **figure 10** counted thickness departures against a nominal converted with the literature
  indices while the table used the campaign ones, so the bars and the table disagreed by a factor
  1.7 on SiO₂;
- **figure 1** still said *witness samples* weeks after the script had been corrected.

Scripts now compute what they print and deposit the computation. Figure 10 carries the rms
departure of the regularized fit as a grey band, so that the width of the family of equivalent
stacks can be told apart from the error of the deposition run.

`tools/aperture_quadrature_check.py` gains rms columns and an out-of-plane comparison.

---

## Reproducing

```bash
python -m venv .venv && . .venv/bin/activate     # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python reproduce.py                              # self-check, ends with "every check passed"
pytest                                           # 124 tests
```

The figures and tables of the paper are regenerated by

```bash
python tools/article_tables.py
python tools/article_figures.py
python tools/audit_article_numbers.py
```

The 401 multi-start inversions take a few hours on one core and are not part of the self-check;
their outputs are deposited in `results/`.

---

## Provenance

The optical constants used here were determined on dedicated single layers in the companion
paper *Certus Index Spline* (Opt. Continuum, 2026), deposited at
[10.5281/zenodo.21216491](https://doi.org/10.5281/zenodo.21216491). The samples of both studies
come from a single deposition campaign on a Bühler Leybold Optics HELIOS 800, less than a week
apart, with no target change or chamber reconditioning in between.
