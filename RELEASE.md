# Releasing this deposit

The steps that turn this repository into an archived, citable deposit. They are written down
because the last of them fills in numbers that several files claim, and a deposit whose
metadata disagrees with itself is the kind of thing this package exists to prevent.

Nothing here is automatic. Each step is a decision.

---

## Before anything

```bash
python reproduce.py                        # 41 checks, must end with "every check passed"
python -m pytest tests/ -q                 # 123 tests
python tools/compare_with_reference.py     # non-regression against the laboratory tool
```

The last one needs the laboratory tool on the machine and is skipped elsewhere; it is not a
condition of the release, but a discrepancy in it is a reason to stop.

Then check that the deposit still says true things about itself:

- the **Status** table at the top of `README.md` matches what is actually wired;
- `results/` was regenerated after the last change to the model or the studies
  (`python tools/article_tables.py`, then `python tools/search_window_scan.py`);
- the version is the same in `pyproject.toml`, `certus_re/__init__.py` and `CITATION.cff`.

---

## The release itself

1. **Make the repository public.** Zenodo's GitHub integration cannot see a private one.

   ```bash
   gh repo edit nikonvr/certus-re --visibility public --accept-visibility-change-consequences
   ```

2. **Enable the repository in Zenodo**, at <https://zenodo.org/account/settings/github/>, and
   only then create the release. Zenodo archives releases made *after* the switch is on.

3. **Tag and release.**

   ```bash
   git tag -a v1.0.0 -m "certus_re 1.0.0"
   git push origin v1.0.0
   gh release create v1.0.0 --title "certus_re 1.0.0" --notes-file RELEASE_NOTES.md
   ```

4. **Collect the two DOIs.** Zenodo mints a *concept* DOI, which always resolves to the latest
   version, and a *version* DOI for this release. The concept DOI is the one to cite.

---

## Filling in what the deposit currently leaves blank

These are the only places that state a DOI, and they must be changed together. Until they are,
each of them says plainly that the DOI does not exist yet, which is why none of them has to be
corrected — only completed.

| File | What to add |
|---|---|
| `CITATION.cff` | `doi:` with the **concept** DOI, and `date-released:` |
| `README.md`, section 10 | replace the paragraph that says no DOI exists yet with the concept DOI |
| `CITATION.cff`, `preferred-citation` | journal volume, pages and article DOI, on acceptance |
| `06_REDACTION/lemarchand_lumeau_volet2.tex` | the three `\TBD{concept DOI Zenodo}` markers and the `\TBD{vol., pages}` of the self-reference |

A useful order: submit the article, release the deposit, put the concept DOI in the manuscript
before the proofs, and add the article DOI to `CITATION.cff` once it is assigned.

---

## What a reviewer will do with it

Not the same things the authors do, so they are worth testing once from a clean state:

```bash
git clone https://github.com/nikonvr/certus-re.git
cd certus-re
python -m venv .venv && . .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -e .
python reproduce.py
```

That is the whole contract: two dependencies, no compiler, no database, no absolute path, and
one command that re-derives every number the article quotes. It has been checked from an empty
virtual environment; keep it that way.

---

## What is deliberately not in the deposit

Stated here so that their absence is a decision on record rather than an omission.

- **The laboratory archives.** `tools/build_study_volet2.py` reads the campaign workbooks, the
  session log and the design slide to *produce* `studies/volet2/`; those sources are not
  redistributed. The script is the provenance trail, not a dependency of the deposit.
- **The laboratory's production tool.** `tools/compare_with_reference.py` imports it read-only
  for the non-regression test and skips cleanly when it is absent.
- **The manuscript.** It lives with the article, not with the code.
