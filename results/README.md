# results — generated, not written

Everything in this directory is produced by

```bash
python tools/article_tables.py
```

which runs the ladder of `studies/volet2/` and the six counter-experiments beside it, and
writes what the article states in a table. Nothing here is edited by hand; if a number is
wrong, the study file or the code is wrong, and the fix belongs there.

| File | What it is |
|---|---|
| `table1_samples.tex` | the five samples: coatings, layers, substrate, polishing, rear-face model, acquisition and its settings, band, points, declared σ |
| `table2_ladder.tex` | the study plan in six lines — released blocks, parameter count, points per parameter, and the residual of every measurement at every rung. Replaces a separate parameter-budget table and a separate residual table |
| `table3_instrument.tex` | the retrieved beam aperture and polarizer leakage with their uncertainties, and what each counter-experiment costs |
| `designs.csv` | nominal and retrieved quarter waves layer by layer, with the uncertainty on each and the departure expressed in units of that uncertainty — the file the design figure is drawn from |
| `article_numbers.json` | all of the above, machine-readable, including every rung's residuals and `s²` |

The designs are deliberately **not** a table: sixteen nominal values, sixteen retrieved values
and sixteen error bars are a figure, and `designs.csv` is what draws it.

The manuscript inlines the three `.tex` tables rather than `\input`-ing them, so that it stays
self-contained for submission. Re-splice them after regenerating:

```bash
python ../06_REDACTION/update_tables.py
```
