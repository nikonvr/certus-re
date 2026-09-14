# results — generated, not written

Everything in this directory is produced by

```bash
python tools/article_tables.py
```

which runs the ladder of `studies/volet2/` and the sensitivity studies beside it, and
writes the numerical results used by the article. If a number is
wrong, the study file or the code is wrong, and the fix belongs there.

The four `insitu_multistart_*` files are the exception: they come from

```bash
python tools/insitu_multistart_noprior.py --draws 100   # ~40 min on sixteen cores
python tools/insitu_multistart_prior.py
python tools/insitu_convergence_traces.py --draws 8     # residual histories, ~10 min
python tools/aperture_quadrature_check.py               # seconds
```

which are kept separate because they take minutes rather than seconds.

| File | What it is |
|---|---|
| `table1_samples.tex` | the five samples: coatings, layers, substrate, polishing, rear-face model, acquisition and its settings, band, points, declared σ |
| `table2_ladder.tex` | the study plan in six lines — released blocks, parameter count, points per parameter, and the residual of every measurement at every rung. Replaces a separate parameter-budget table and a separate residual table |
| `table3_instrument.tex` | the instrument block of the joint campaign, followed by model-sensitivity comparisons retained in the reproducibility package |
| `designs.csv` | nominal and retrieved optical thicknesses layer by layer, with the local data-Jacobian sensitivity estimate and relative departure (\%) — the file used for the design figure |
| `article_numbers.json` | all of the above, machine-readable, including every rung's residuals and `s²` |
| `insitu_multistart_noprior.csv` | one line per multi-start inversion of the in situ test of Section 6: perturbation level, draw, wall time, residual, retrieved index of each oxide at 1500, 2500 and 4000 nm, and the thickness departures. This is the file Table 4 and Figs. 8 and 9 are built from |
| `insitu_multistart_noprior_summary.csv` | the same, aggregated per perturbation level: convergence rate, and the mean and spread of every quantity over the runs that converged |
| `insitu_multistart_prior.csv`, `..._summary.csv` | the same benchmark with the process prior left on, which is the comparison point quoted in Section 6.5 |
| `insitu_convergence_traces.json` | the residual history of a traced subset of those inversions, one list per run, in multiples of the noise floor. `certus_re` does not record the optimizer's path, so this is captured from outside; it is what the funnel of Fig. 10 is drawn from |
| `insitu_layer_departures.csv` | the per-layer departure of the converged unregularized solution, which the benchmark reports only as an aggregate |
| `aperture_quadrature_check.csv` | how far the two-ray beam-aperture model sits from a converged in-plane quadrature and from a two-dimensional round-pupil integration, per measurement. This is the number Section 3 quotes |

The designs are deliberately **not** a table: the nominal and retrieved values with error
bars across all three components (Filter 1 AR6, Filter 2 BS45, Filter 3 BS17) are presented as a figure, and `designs.csv`
is what draws it.

**What the article shows, and what it leaves here.** The sample table and analysis ladder are inlined in
the manuscript. A compact manuscript table summarizes the prior, index, and collimated-beam sensitivities.
The more detailed instrument table, alternative settings, multistart probe, and aperture scan remain in
the reproducibility package.

Re-splice generated tables into the manuscript after regenerating:

```bash
python ../06_REDACTION/update_tables.py
```
