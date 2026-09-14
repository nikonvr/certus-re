# certus_re 1.0.0 — Companion code for Volet 2

Reverse engineering of multilayer optical coatings from spectrophotometric measurements, with an explicit parameter budget and joint multi-sample inversion.

### Key Capabilities
- **Joint multi-sample inversion**: Inverts several samples simultaneously, sharing common thin-film stacks without duplicating unknowns.
- **Two-side coated components**: Coherent transfer-matrix model on each face, coupled through an incoherent thick substrate.
- **Instrument modeling**: Wavelength-stepped finite beam cone aperture (polarizers treated as ideal; crosstalk excluded from investigation).
- **Continuous MAP Bayesian prior**: Technology-motivated prior (0.5% optical thickness repeatability) avoiding artificial search box boundaries.
- **Autonomous in situ index determination**: Support for spline-parameterized dispersion fitting across dissimilar multilayer coatings, demonstrated in a quasi-blind multi-start test without reference single layers.
- **Rigorous degrees-of-freedom budget**: Parameter counting reported alongside fit residuals.

### Complementarity
- Companion code to Volet 2: *Transferability of Single-Layer Optical Constants to Complex Multilayer Reverse Engineering in the MWIR*.
- Reuses and references optical constants established in Volet 1 (*Certus Index Spline*, DOI: 10.5281/zenodo.21216491).
