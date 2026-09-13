# results — generated, not written

Everything in this directory is produced by

```bash
python tools/article_tables.py
```

which runs the ladder of `studies/volet2/` and the seven counter-experiments beside it, and
writes what the article states in a table. Nothing here is edited by hand; if a number is
wrong, the study file or the code is wrong, and the fix belongs there.

| File | What it is |
|---|---|
| `table1_samples.tex` | the five samples: coatings, layers, substrate, polishing, rear-face model, acquisition and its settings, band, points, declared σ |
| `table2_ladder.tex` | the study plan in six lines — released blocks, parameter count, points per parameter, and the residual of every measurement at every rung. Replaces a separate parameter-budget table and a separate residual table |
| `table3_instrument.tex` | the instrument block of the joint campaign -- beam aperture and polarizer crosstalk -- with the value each one carries and whether it was imposed or fitted, followed by what each of the seven counter-experiments costs |
| `designs.csv` | nominal and retrieved optical thicknesses layer by layer, with the local 1$\sigma$ uncertainty on each and the relative departure (\%) — the file the design figure is drawn from |
| `article_numbers.json` | all of the above, machine-readable, including every rung's residuals and `s²` |

The designs are deliberately **not** a table: the nominal and retrieved values with error
bars across all three components (Filter 1 AR6, Filter 2 BS45, Filter 3 BS17) are presented as a figure, and `designs.csv`
is what draws it.

**What the article shows, and what it leaves here.** To keep the manuscript on the optical physics and
the three filters, only `table1_samples.tex` is inlined. The ladder (`table2_ladder.tex`), the instrument
and counter-experiment table (`table3_instrument.tex`), the multistart probe and the aperture scan are
summarized in a sentence each and left here in full, so that a reviewer who wants the benchmark can read it
without the manuscript carrying it.

Re-splice generated tables into the manuscript after regenerating:

```bash
python ../06_REDACTION/update_tables.py
```
