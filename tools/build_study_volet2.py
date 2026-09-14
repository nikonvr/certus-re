"""Assemble the deposited study directory from the laboratory archives.

This script is the provenance trail of ``studies/volet2/``: it states, in executable form,
where every deposited number comes from. It reads the laboratory archives -- which are *not*
part of the deposit -- and writes the CSV and JSON files that are.

It is run once, before a release. A reader of the deposit does not need it; a reviewer asking
"where does this column come from" does.

Sources
-------
``01_DATA_MESURES_ET_MODELES/``
    ``indices_SiO2_OpticsContinuum.csv``, ``indices_Nb2O5_OpticsContinuum.csv``
        The optical constants determined on the reference single layers and published with
        Volet 1, with their uncertainty envelope. These are the constants whose
        transferability is under test, and they are never adjusted here.
    ``AR_6couches_8deg_substrat_face_arriere_DEPOLIE_H800.xlsx``
        Nominal design of the six-layer antireflection coating and its reflectance at 8
        degrees, rear face ground.
    ``BIFACE_BS45_plus_AR6_H800.xlsx``
        Reflectance of the two-side-coated component at 45 degrees, in s and p.
    ``spectres_monocouche_*_Trel.csv``
        Relative transmittance of the two reference single layers, reproduced from the Volet 1 deposit.
``01_DATA_MESURES_ET_MODELES/to investigate/``
    ``allcnes_2026-03-17_racine_couches_minces_2026.xls``
        The acquisition log of 17 March 2026 at full double precision -- the CSV of the same
        name is rounded to three decimals. Source of the beam-splitter spectra in s and p,
        and of the second acquisition of the antireflection coating.
    ``H800-BS - 45deg.pptx``
        The design of the beam splitter actually deposited: **sixteen** layers, dated the
        morning of the measurements. The seventeen-layer workbook of the folder describes a
        non-polarizing study drawn two days *after* the deposition and never made; its
        "measurement" columns are a computed response. It is not read here.

Silicon and sapphire are **computed**, not read: both are literature materials, and a
literature material is better carried as its formula than as somebody's transcription of it.
See :mod:`certus_re.dispersion`.

Dependencies
------------
``openpyxl`` for the .xlsx workbooks and ``xlrd`` for the .xls acquisition log. Neither is
needed to *use* the deposit -- that takes NumPy and SciPy only.

Usage
-----
    python tools/build_study_volet2.py [--archives PATH] [--out studies/volet2]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from certus_re.dispersion import (  # noqa: E402
    SAPPHIRE_MALITSON_SELLMEIER,
    li1980_silicon_n,
    sellmeier_n,
)

# --------------------------------------------------------------------------
# Where the archives live. Overridable, because nothing in the deposit may
# depend on one machine's directory layout.
# --------------------------------------------------------------------------

DEFAULT_ARCHIVES = Path(
    r"D:\drivfl\01_Recherche_Enseignement\02_Couches_Minces_Optique\couches minces 2026"
)

CAMPAIGN_SUBDIR = Path("CERTUS/publication_reverse/01_DATA_MESURES_ET_MODELES")
LOG_SUBDIR = Path("to investigate")

ALLCNES = "allcnes_2026-03-17_racine_couches_minces_2026.xls"
BS_DESIGN_PPTX = "H800-BS - 45deg.pptx"
AR_WORKBOOK = "AR_6couches_8deg_substrat_face_arriere_DEPOLIE_H800.xlsx"

# The two-side-coated component is no longer a sample of this study, but its workbook is still
# read: its 'R_Depoli' columns are a second, independently exported copy of the sixteen-layer
# spectra, and agreeing with the session log is what establishes that those columns hold what
# their labels claim.
BIFACE_WORKBOOK = "BIFACE_BS45_plus_AR6_H800.xlsx"

# The seventeen-layer polarizing beam splitter, run 260319-037.
BS17_RAW = "BS17_mesures_refaites a 220-5.xlsx"
BS17_PARAMS = "BS-Pol-measured.xlsx"
BS17_WORKBOOK = "BS45deg_17couches_H800.xlsx"


# The inversion window, one and the same for every sample.
#
# The lower edge is where the campaign was measured: the acquisitions start at 1500 nm, and one
# window for all five samples is what makes their residuals comparable.
#
# The upper edge is measured rather than argued: the repeated acquisition gives a photometric
# repeatability of 0.0014 up to 4000 nm and 0.0145 beyond it, ten times worse.
BAND_NM = (1500.0, 4000.0)
BS17_BAND_NM = BAND_NM

# Silicon is tabulated over the range Li's model is published for, restricted to what this
# campaign can reach. Any query below 1200 nm is then refused by the study validation rather
# than answered with the edge value.
SILICON_RANGE_NM = (1200.0, 6000.0)

# Beyond the last interference extremum, near 4840 nm, the reference single layer determination is no longer
# constrained by the fringe positions. Declared so that a run which reaches past it says so.
PUBLISHED_VALID_RANGE_NM = (250.0, 4840.0)

# Photometric one-sigma used for each family of measurements.
#   reference single layers : noise floor measured in Volet 1 and reported in its summary.csv
#   components: MEASURED here, from the repeated acquisition the campaign contains.
#
# The session log holds two acquisitions of one specimen four minutes apart, at the same
# angle, with the same slit, the same spot and the same averaging: '2600316-033-mono' at
# 15:21:54 and 'mono repet' at 15:25:46. Their difference is the photometry repeating itself
# and nothing else, so the standard deviation of that difference over root two is the
# one-sigma of a single measurement. Over the inversion window it comes to 0.0014, where the
# manufacturer's specification used until now was 0.0053 -- pessimistic by a factor of four,
# and every chi-square of this study was scaled by it.
#
# The pair is unpolarized and at 8 degrees, so for the polarized channels at 45 degrees, which
# lose flux in the polarizer, this is a LOWER BOUND. It is used anyway, because a measured
# lower bound is a better statement than an unmeasured specification, and because what it
# changes is the reading of the residuals rather than the residuals themselves.
SIGMA_REFERENCE = 0.002
SIGMA_COMPONENT = 0.002
REPEATABILITY_PAIR = ("2600316-033-monoAR", "mono repetAR")
# Expected to three digits, so that a change in the archive is caught rather than adopted.
SIGMA_COMPONENT_EXPECTED = 0.00136
SIGMA_COMPONENT_SOURCE = (
    "spectrophotometer noise floor from companion paper (Optics Continuum) adopted across all channels"
)

# Half-width of the search window on each thickness, as a fraction of nominal. This is prior
# information about how the coatings were made, and it belongs in the study file next to the
# released blocks, because a result depends on it as much as on any of them.
#
# Multilayer reverse engineering is ill-posed: many thickness vectors reproduce one spectrum
# to within the photometry. Widening this window from 3 % to 50 % on the sixteen-layer coating
# improves the residual by less than a fifth of the measured repeatability, makes
# it worse in p, and TRIPLES the layer-to-layer dispersion of the answer, which walks into a
# compensating pair of adjacent layers at +15 and -17 %. Nothing in the residual announces it.
#
# Five percent is what optical monitoring holds a layer to; departures of tens of percent
# describe no coating this machine can make. The scan that establishes the trade-off is
# written by tools/article_tables.py and is reproducible.
THICKNESS_WINDOW = 0.03

# Thicknesses retrieved for the two reference single layers by the Volet 1 determination. They are the
# nominal each reference single layer is compared to; a reference single layer has no design in quarter waves. The window
# above is not binding for them -- they move by three hundredths of a percent.
REFERENCE_THICKNESS_NM = {"SiO2": 1681.7, "Nb2O5": 1715.9}

# The design actually deposited as run 260317-035, read from the slide of 17 March 08:13 and
# repeated here so that a transcription error in either place is caught rather than adopted.
BS45_QWOT_REFERENCE = (
    1.691000, 2.358395, 3.070991, 2.976127, 3.047701, 1.409445, 0.929705, 1.395330,
    0.861012, 1.903456, 1.575825, 2.603672, 1.554292, 0.893425, 3.461199, 3.039678,
)

# The seventeen-layer design as its workbook lists it, which is already the order this package
# uses: layer 1 adjacent to the silicon. The four possible readings were tried against the
# measurement and only one is anywhere near it -- as listed with a low-index layer first, at
# 2.0 % nominal, against 23.8, 28.0 and 17.5 % for the other three. Reversing the sequence and
# keeping a low-index layer first gives 28.0 %, which is the size of the residual the
# historical campaign could not get below.
BS17_QWOT_REFERENCE = (
    1.520588, 1.698967, 1.739311, 1.553883, 1.661250, 1.580353, 1.819601, 1.835076,
    2.133498, 2.066467, 2.290912, 2.286649, 10.623639, 2.493294, 0.774866, 1.178906,
    0.746690,
)

# Layer 1 is the one adjacent to the substrate -- the first deposited. For the antireflection
# coating this is stated by the campaign's own publication workbook: "La couche 1 sur le
# substrat Si est Nb2O5 (H), tandis que la couche 6 externe en contact avec l'air est SiO2".
# For the beam splitter the design slide says "en commencant par Nb2O5 (H)". Reversing either
# sequence takes the residual of the antireflection coating from 0.6 % to 44 %.
MATERIALS_FROM_SUBSTRATE = ("Nb2O5", "SiO2")

# The seventeen-layer component starts on the other material: a low-index layer against the
# silicon, and a low-index layer against the air, which is what seventeen alternating layers
# force. The other three parities are tried in tools/headline_17c.py and cost 15, 24 and 27 %.
BS17_MATERIALS_FROM_SUBSTRATE = ("SiO2", "Nb2O5")


# Files of an earlier revision that the present one supersedes, removed so that the deposit
# cannot be run against them by accident.
SUPERSEDED = (
    "indices/Nb2O5_H800_campaign.csv",
    "indices/SiO2_H800_campaign.csv",
    "components_campaign_indices.json",
    "biface_unresolved.json",
    # Renamed when the studies became a graded ladder: the numeric prefix is what makes the
    # progression readable in a directory listing, and a stale copy under the old name would
    # be inverted by reproduce.py beside its replacement.
    "reference single layers.json",
    "AR6_alone.json",
    "BS45_alone.json",
    "biface_alone.json",
    "joint_campaign.json",
    # The polarizer-leakage rung and the numbering it forced, dropped when the block was
    # found to earn nothing.
    "06_BS45_crosstalk.json",
    "07_biface_alone.json",
    "08_joint_campaign.json",
    # The two-side-coated component, dropped from the ladder when the study was narrowed to
    # the three single-face components it can actually settle a question with. It named no
    # unknown of its own -- both its coatings are named by samples that are kept -- so nothing
    # is lost from the count of degrees of freedom. It survives under examples/, because the
    # package claims to invert a component coated on both faces and a claim nothing exercises
    # is not a claim.
    "06_biface_alone.json",
    # Renumbered again when the seventeen-layer component entered the ladder.
    "05_BS45_aperture.json",
)


# --------------------------------------------------------------------------
# Small readers
# --------------------------------------------------------------------------


def write_csv(path: Path, header: list[str], columns: list[np.ndarray], note: str) -> None:
    """Write a CSV with a leading comment line naming the source of the numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(columns[0])
    if any(len(c) != n for c in columns):
        raise ValueError(f"{path.name}: columns of unequal length")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {note}\n")
        writer = csv.writer(handle)
        writer.writerow(header)
        for i in range(n):
            writer.writerow(
                [("" if not np.isfinite(c[i]) else f"{c[i]:.10g}") for c in columns]
            )


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    """Read a CSV into arrays keyed by column name, skipping ``#`` comment lines.

    Rows are kept row-aligned: a blank cell becomes NaN rather than being dropped, because
    dropping it would shift every later value of that column onto the wrong wavelength.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        lines = [ln for ln in handle if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"{path}: no header")
    out: dict[str, list[float]] = {c.strip(): [] for c in reader.fieldnames}
    for row in reader:
        for col in out:
            raw = row.get(col)
            if raw is None or str(raw).strip() == "":
                out[col].append(np.nan)
                continue
            try:
                out[col].append(float(str(raw).replace(",", ".")))
            except ValueError:
                out[col].append(np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def read_xlsx_sheet(workbook: Path, sheet: str) -> list[tuple]:
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        rows = list(wb[sheet].iter_rows(values_only=True))
    finally:
        wb.close()
    return rows


def sheet_columns(rows: list[tuple]) -> dict[str, np.ndarray]:
    """Turn a sheet whose first row is a header into arrays keyed by column name.

    Columns are addressed by name, never by position: two workbooks of this one campaign
    order their index columns differently, and substituting by rank produced a forty-percent
    residual.
    """
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    out: dict[str, list[float]] = {name: [] for name in header if name}
    for row in rows[1:]:
        for name, cell in zip(header, row):
            if not name:
                continue
            if isinstance(cell, (int, float)):
                out[name].append(float(cell))
            else:
                out[name].append(np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def read_xls_sheet(workbook: Path, sheet: str) -> dict[str, np.ndarray]:
    """Read one sheet of a legacy .xls acquisition log, by column name."""
    import xlrd

    book = xlrd.open_workbook(str(workbook))
    page = book.sheet_by_name(sheet)
    header = [str(page.cell_value(0, c)).strip() for c in range(page.ncols)]
    out: dict[str, list[float]] = {name: [] for name in header}
    for r in range(1, page.nrows):
        for c, name in enumerate(header):
            value = page.cell_value(r, c)
            out[name].append(float(value) if isinstance(value, (int, float)) else np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def read_xls_parameters(workbook: Path) -> dict[str, dict[str, str]]:
    """Read the acquisition settings the spectrophotometer recorded, one column per run.

    The session log carries, beside every spectrum, the state of the instrument when it was
    taken. Those settings belong in the deposit: the beam-aperture model of this study is a
    statement about the instrument's geometry, and the geometry is not the same in every
    acquisition of the campaign.
    """
    import xlrd

    book = xlrd.open_workbook(str(workbook))
    page = book.sheet_by_name("Parameters")
    names = [str(page.cell_value(0, c)).strip() for c in range(page.ncols)]
    out: dict[str, dict[str, str]] = {name: {} for name in names[2:]}
    for r in range(1, page.nrows):
        label = str(page.cell_value(r, 1)).strip()
        if not label:
            continue
        for c in range(2, page.ncols):
            value = page.cell_value(r, c)
            if isinstance(value, float) and value == int(value):
                value = int(value)
            text = str(value).strip()
            if text and text not in ("None", "--"):
                out[names[c]][label] = text
    return out


def read_xlsx_parameters(workbook: Path) -> dict[str, dict[str, str]]:
    """The same, for an acquisition exported as .xlsx rather than legacy .xls.

    The 19 March run was saved in the newer format. Its Parameters sheet has the identical
    layout, so the two readers agree on the shape they return and ``acquisition_block`` does
    not need to know which one produced its input.
    """
    rows = read_xlsx_sheet(workbook, "Parameters")
    names = [str(c).strip() if c is not None else "" for c in rows[0]]
    out: dict[str, dict[str, str]] = {name: {} for name in names[2:] if name}
    for row in rows[1:]:
        label = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
        if not label:
            continue
        for c in range(2, len(names)):
            if not names[c] or c >= len(row):
                continue
            value = row[c]
            if value is None:
                continue
            if isinstance(value, float) and value == int(value):
                value = int(value)
            text = str(value).strip()
            if text and text not in ("None", "--", "nan"):
                out[names[c]][label] = text
    return out


# Which acquisition of the session log each deposited spectrum comes from. The reference single layers are
# absent: they were measured for Volet 1 and are reproduced from its deposit.
ACQUISITION_OF = {
    ("AR6_R8deg.csv", "R_a_pct"): "260317-034 depoli",
    ("AR6_R8deg.csv", "R_a_run1_pct"): "260316-ARC-Run1",
    ("BS45_R45deg.csv", "R_s_pct"): "260317-035s",
    ("BS45_R45deg.csv", "R_p_pct"): "260317-035p",
    ("BS17_R45deg.csv", "R_s_pct"): "260319-037-R-PolS-Unpolished",
    ("BS17_R45deg.csv", "R_p_pct"): "260319-037-R-PolP-Unpolished",
}

# The settings worth carrying, in the order a reader wants them, under names that do not
# depend on the instrument software's spelling.
ACQUISITION_FIELDS = (
    ("Start date/time", "datetime"),
    ("Sample stage angle, °", "stage_angle_deg"),
    ("Detector angle, °", "detector_angle_deg"),
    ("Polarization", "polarizer"),
    ("Start wavelength, nm", "range_start_nm"),
    ("End wavelength, nm", "range_end_nm"),
    ("Sampling pitch, nm", "sampling_pitch_nm"),
    ("Averaging count", "averaging_count"),
    ("Slit width, µm", "slit_width_um"),
    ("Spot size, mm", "spot_size_mm"),
    ("Sample thickness, mm", "sample_thickness_mm"),
)


def acquisition_block(parameters: dict[str, dict[str, str]], name: str) -> dict:
    """The recorded settings of one acquisition, renamed and ordered for the deposit."""
    raw = parameters.get(name)
    if raw is None:
        raise ValueError(f"acquisition {name!r} is not in the session log")
    block: dict[str, object] = {"acquisition": name}
    for label, key in ACQUISITION_FIELDS:
        if label in raw:
            block[key] = raw[label]
    block["polarizer"] = block.get("polarizer", "none")
    # The instrument writes 'Transmittance' in its mode field whenever the operator drives
    # the stage by hand, whatever the geometry actually set. The two angles are what say
    # which quantity was recorded: 8 deg of stage for 16 of detector, or 45 for 90, is
    # specular reflection.
    block["mode_field_note"] = (
        "the software's 'Measurement mode' field is not reported: it reads 'Transmittance' "
        "for manually driven acquisitions whatever the geometry. The stage and detector "
        "angles are what identify the quantity."
    )
    return block


def read_pptx_qwot(path: Path) -> tuple[int, list[float]]:
    """Read the layer count and the quarter-wave design from a design slide.

    A .pptx is a zip of XML; the text runs are the ``<a:t>`` elements. Parsing it with the
    standard library keeps this script free of a presentation-file dependency for the sake of
    sixteen numbers.
    """
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            n for n in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)
        )
        for name in slides:
            runs = [
                t.strip()
                for t in re.findall(
                    r"<a:t>(.*?)</a:t>", archive.read(name).decode("utf-8", "replace"), re.S
                )
                if t.strip()
            ]
            if not any("QWOT" in r for r in runs):
                continue
            count = None
            for run in runs:
                found = re.match(r"\s*(\d+)\s*couches", run)
                if found:
                    count = int(found.group(1))
            values = [float(r) for r in runs if re.fullmatch(r"-?\d+\.\d+", r)]
            if values:
                return (count if count is not None else len(values)), values
    raise ValueError(f"{path.name}: no slide carrying a QWOT design")


def on_grid(
    source_w: np.ndarray, source_v: np.ndarray, target_w: np.ndarray
) -> np.ndarray:
    """Values of a column at the wavelengths of another, matched point by point.

    The acquisitions of one session share a wavelength grid to better than a hundredth of a
    nanometre, so this is a lookup, not an interpolation; anything that does not match is
    returned as NaN rather than invented.
    """
    out = np.full(target_w.size, np.nan)
    for i, w in enumerate(target_w):
        j = int(np.argmin(np.abs(source_w - w)))
        if abs(source_w[j] - w) < 0.05:
            out[i] = source_v[j]
    return out


def measure_repeatability(
    log: dict[str, np.ndarray], band_nm: tuple[float, float]
) -> tuple[float, list[str]]:
    """One-sigma of a single photometric measurement, from the repeated acquisition.

    Two acquisitions of one specimen, minutes apart, with nothing changed. Their difference
    carries twice the variance of one measurement, so ``std(difference) / sqrt(2)`` is the
    repeatability. Reported per band as well as over the window, because the instrument
    changes detector and source inside it -- and because that breakdown is what shows the
    upper edge of the window to be in the right place.
    """
    wavelength = log["Wavelength, nm"]
    first, second = (log[name] / 100.0 for name in REPEATABILITY_PAIR)
    difference = second - first

    lines: list[str] = []
    bands = [
        ("below 2530 nm, InGaAs", wavelength < 2530.0),
        ("2530-3700 nm, PbSe", (wavelength >= 2530.0) & (wavelength < 3700.0)),
        ("3700-4000 nm, IR source", (wavelength >= 3700.0) & (wavelength <= 4000.0)),
        ("beyond 4000 nm", wavelength > 4000.0),
    ]
    for name, mask in bands:
        if not np.any(mask):
            continue
        sigma = float(np.std(difference[mask], ddof=1) / np.sqrt(2.0))
        lines.append(f"{name:<26s} n={int(mask.sum()):4d}  sigma = {sigma:.5f}")

    inside = (wavelength >= band_nm[0]) & (wavelength <= band_nm[1])
    sigma = float(np.std(difference[inside], ddof=1) / np.sqrt(2.0))
    lines.append(
        f"{'over the inversion window':<26s} n={int(inside.sum()):4d}  sigma = {sigma:.5f}"
    )
    return sigma, lines


def agree(a: np.ndarray, b: np.ndarray, tolerance: float, what: str) -> str:
    """State how far two versions of one measurement differ, and complain if it is too far."""
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        raise ValueError(f"{what}: no common point to compare")
    worst = float(np.max(np.abs(a[both] - b[both])))
    if worst > tolerance:
        raise ValueError(
            f"{what}: the two versions differ by up to {worst:.3g}, more than the "
            f"{tolerance:g} expected from rounding -- they are not the same acquisition"
        )
    return f"{what}: agree to {worst:.2g}"


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archives",
        type=Path,
        default=DEFAULT_ARCHIVES,
        help="root of the laboratory archives (default: the authors' layout)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "studies" / "volet2",
        help="directory to write the study into",
    )
    args = parser.parse_args(argv)

    campaign: Path = args.archives / CAMPAIGN_SUBDIR
    log_dir = campaign / LOG_SUBDIR
    out: Path = args.out

    if not campaign.is_dir():
        print(f"archive directory not found: {campaign}", file=sys.stderr)
        print("pass --archives with the root of the laboratory archives", file=sys.stderr)
        return 2

    missing = [
        p
        for p in (
            campaign / AR_WORKBOOK,
            # Still read, though the two-side-coated component is no longer deposited: its
            # 'R_Depoli' columns are a second, independently exported copy of the sixteen-
            # layer spectra, and checking the session log against them is what establishes
            # that those columns are what their labels say.
            campaign / BIFACE_WORKBOOK,
            campaign / BS17_RAW,
            campaign / BS17_PARAMS,
            campaign / BS17_WORKBOOK,
            campaign / "indices_SiO2_OpticsContinuum.csv",
            campaign / "indices_Nb2O5_OpticsContinuum.csv",
            campaign / "spectres_monocouche_SiO2_Trel.csv",
            campaign / "spectres_monocouche_Nb2O5_Trel.csv",
            log_dir / ALLCNES,
            log_dir / BS_DESIGN_PPTX,
        )
        if not p.is_file()
    ]
    if missing:
        print("source files not found:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    (out / "indices").mkdir(parents=True, exist_ok=True)
    (out / "spectra").mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    checks: list[str] = []

    # -- coating materials -------------------------------------------------
    # One determination, the one published with Volet 1, whose transferability is the
    # question this study asks. The determination carried in the campaign workbooks is not
    # deposited: it exploits R and T, covers 1000-5200 nm only, and comparing two datasets
    # would blur the very statement the article makes about one of them.
    for material in ("Nb2O5", "SiO2"):
        source = campaign / f"indices_{material}_OpticsContinuum.csv"
        columns = read_csv_columns(source)
        wavelength = columns["lambda (nm)"]
        n = columns["n nominal"]
        k = columns["k nominal"]
        # Half-width of the published envelope. Rows without n are the file's own padding.
        sigma = 0.5 * (columns["n max"] - columns["n min"])
        good = np.isfinite(wavelength) & np.isfinite(n)
        write_csv(
            out / "indices" / f"{material}_H800_published.csv",
            ["Wavelength_nm", "n", "k", "sigma_n"],
            [
                wavelength[good],
                n[good],
                np.nan_to_num(k[good]),
                sigma[good],
            ],
            f"{material} deposited by plasma-assisted reactive magnetron sputtering on a "
            f"Buhler Leybold Optics HELIOS 800, Institut Fresnel, March 2026 campaign. "
            f"Optical constants determined from the relative transmittance of the "
            f"reference single layers and published with Volet 1 (Optics Continuum). "
            f"sigma_n is the half-width of the published uncertainty envelope. Beyond the "
            f"last interference extremum, near "
            f"{PUBLISHED_VALID_RANGE_NM[1]:.0f} nm, the index is no longer constrained by "
            f"the fringe positions; see valid_range_nm in the study files. Never adjusted.",
        )
        written.append(f"indices/{material}_H800_published.csv")

    # -- substrates, computed from their published formulas ----------------
    silicon_w = np.arange(SILICON_RANGE_NM[0], SILICON_RANGE_NM[1] + 1e-9, 5.0)
    silicon_n = li1980_silicon_n(silicon_w)
    write_csv(
        out / "indices" / "Si_Li1980.csv",
        ["Wavelength_nm", "n", "k"],
        [silicon_w, silicon_n, np.zeros_like(silicon_w)],
        "Silicon. H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980), crystalline silicon at "
        "26 C, evaluated here from the formula rather than transcribed: "
        "n^2 = 11.6858 + 0.939816/L^2 + 8.10461e-3 * 1.1071^2 / (L^2 - 1.1071^2), L in "
        "micrometres. n(1.5 um) = 3.4804, n(2) = 3.4532, n(3) = 3.4339, n(4) = 3.4271. "
        "k = 0: silicon is transparent over this range. Tabulated from 1200 nm because the "
        "formula has a pole at 1107 nm; below 1200 nm it is not merely inaccurate but "
        "meaningless. Propagated, never adjusted.",
    )
    written.append("indices/Si_Li1980.csv")
    checks.append(
        f"silicon from Li 1980: n(1500 nm) = {float(li1980_silicon_n(1500.0)):.4f}, "
        f"n(4000 nm) = {float(li1980_silicon_n(4000.0)):.4f}"
    )

    sapphire_w = np.arange(250.0, 5201.0, 10.0)
    write_csv(
        out / "indices" / "Al2O3_Malitson.csv",
        ["Wavelength_nm", "n", "k"],
        [
            sapphire_w,
            sellmeier_n(sapphire_w, SAPPHIRE_MALITSON_SELLMEIER),
            np.zeros_like(sapphire_w),
        ],
        "Sapphire, ordinary ray. I. H. Malitson, J. Opt. Soc. Am. 52, 1377 (1962). "
        "Evaluated here from the Sellmeier coefficients so that the deposit carries no "
        "substrate database. Substrate of the reference single layers.",
    )
    written.append("indices/Al2O3_Malitson.csv")

    # -- the acquisition log, at full precision ----------------------------
    log = read_xls_sheet(log_dir / ALLCNES, "Measurement")
    log_w = log["Wavelength, nm"]

    # -- what the photometry repeats to ------------------------------------
    sigma_component, repeatability_lines = measure_repeatability(log, BAND_NM)
    if abs(sigma_component - SIGMA_COMPONENT_EXPECTED) > 5e-5:
        print(
            f"  the repeated acquisition now gives sigma = {sigma_component:.5f} where "
            f"{SIGMA_COMPONENT_EXPECTED:.5f} was recorded; the archive has changed",
            file=sys.stderr,
        )
        return 3
    checks.extend(f"photometric repeatability, {line}" for line in repeatability_lines)
    checks.append(
        f"photometric repeatability: the manufacturer's specification, 0.0053, is "
        f"{0.0053 / sigma_component:.1f} times this measurement"
    )

    # -- antireflection coating -------------------------------------------
    ar_rows = read_xlsx_sheet(campaign / AR_WORKBOOK, "measurement")
    ar = sheet_columns(ar_rows)
    ar_w = ar["Wavelength, nm"]
    ar_r = ar["R-8-avg-noBK"]
    # The same acquisition is in the log at full double precision; the workbook rounds it to
    # four decimals of a percent, which is four orders of magnitude below the photometric
    # uncertainty. Checking that they are the same acquisition is the point of this line.
    checks.append(
        agree(
            ar_r,
            on_grid(log_w, log["260317-034 depoliAR"], ar_w),
            5e-4,
            "antireflection coating, workbook against '260317-034 depoli' in the log",
        )
    )
    # A second acquisition of the same coating exists, recorded the day before over a wider
    # band and on a finer mesh. It is deposited beside the first, on the same grid, so that
    # the choice between them is a one-line edit of a study file rather than a rebuild.
    ar_alt = on_grid(log_w, log["260316-ARC-Run1"], ar_w)
    both = np.isfinite(ar_r) & np.isfinite(ar_alt)
    checks.append(
        f"antireflection coating, the two acquisitions differ by "
        f"{float(np.mean(np.abs(ar_r[both] - ar_alt[both]))):.3f} point on average, "
        f"{float(np.max(np.abs(ar_r[both] - ar_alt[both]))):.3f} at worst"
    )
    write_csv(
        out / "spectra" / "AR6_R8deg.csv",
        ["Wavelength_nm", "R_a_pct", "R_a_run1_pct"],
        [ar_w, ar_r, ar_alt],
        "Six-layer antireflection coating, run 260317-034, on silicon with the rear face "
        "ground. Reflectance at 8 degrees, no polarizer, in percent. EssentOptics PHOTON RT "
        "SN 38425, 17 March 2026 15:45, 220 um slit. R_a_pct is column 'R-8-avg-noBK' of "
        "the campaign workbook, which is acquisition '260317-034 depoli'; R_a_run1_pct is "
        "'260316-ARC-Run1', a second acquisition of the same coating made on 16 March over "
        "1000-5200 nm at 5 nm, resampled here onto the grid of the first. A ground rear "
        "face scatters the return of the second interface out of the collected beam, which "
        "is what isolates the response of the front face and what the 'noBK' marker records.",
    )
    written.append("spectra/AR6_R8deg.csv")

    # -- beam splitter, in s and p ----------------------------------------
    bs_s = log["260317-035s"]
    bs_p = log["260317-035p"]
    biface_rows = read_xlsx_sheet(campaign / BIFACE_WORKBOOK, "02_Mesures_Spectro")
    biface = sheet_columns(biface_rows)
    biface_w = biface["Longueur d'onde (nm)"]
    # The beam-splitter spectra also sit in the two-sided workbook under the name
    # 'R_Depoli'. Those columns are correctly labelled -- unlike 'R_Si_nu_2f_avg', which
    # holds the beam splitter on polished silicon and not bare silicon at all -- and
    # checking them against the log is what establishes that.
    checks.append(
        agree(
            on_grid(log_w, bs_s, biface_w),
            biface["R_Depoli_s (%)"],
            2e-3,
            "beam splitter s, log against 'R_Depoli_s' of the two-sided workbook",
        )
    )
    checks.append(
        agree(
            on_grid(log_w, bs_p, biface_w),
            biface["R_Depoli_p (%)"],
            2e-3,
            "beam splitter p, log against 'R_Depoli_p' of the two-sided workbook",
        )
    )
    write_csv(
        out / "spectra" / "BS45_R45deg.csv",
        ["Wavelength_nm", "R_s_pct", "R_p_pct", "R_a_pct"],
        [log_w, bs_s, bs_p, log["260317-035a"]],
        "Sixteen-layer 50/50 beam splitter for 45 degrees, run 260317-035, on silicon with "
        "the rear face ground. Reflectance at 45 degrees in s and p, in percent, "
        "acquisition '260317-035' of 17 March 2026 16:31, taken from the session log at "
        "full double precision. R_a_pct is the instrument's unpolarized channel, recorded "
        "without a polarizer and therefore free of polarizer leakage; it is not the "
        "half-sum of the two columns beside it. This coating is strongly polarizing -- "
        "63.5 % in s against 37.9 % in p at 1500 nm -- which is what rules out the "
        "non-polarizing design of the folder.",
    )
    written.append("spectra/BS45_R45deg.csv")

    # -- seventeen-layer beam splitter, 19 March (remeasured at 220-5) -----
    # Measured on the EssentOptics PHOTON RT with the standard 220 um slit and 5 mm spot,
    # matching the instrument configuration of the rest of the campaign.
    bs17_rows = read_xlsx_sheet(campaign / BS17_RAW, "Mesures_17_couches")
    bs17_w = np.asarray([float(r[0]) for r in bs17_rows[4:] if r[0] is not None])
    bs17_s = np.asarray([float(r[1]) for r in bs17_rows[4:] if r[0] is not None])
    bs17_p = np.asarray([float(r[2]) for r in bs17_rows[4:] if r[0] is not None])
    checks.append(
        f"seventeen-layer remeasurement loaded: {len(bs17_w)} points from {bs17_w.min():.1f} "
        f"to {bs17_w.max():.1f} nm, 220 um slit, 5 mm spot"
    )
    # -- two-side-coated component -----------------------------------------
    # Not a sample of the article any more, but kept as the deposit's demonstration that a
    # component coated on both faces inverts. Its rear face is polished and coated, which is
    # the case the three components of the ladder do not exercise.
    write_csv(
        out / "spectra" / "BIFACE_R45deg.csv",
        ["Wavelength_nm", "R_s_pct", "R_p_pct", "R_a_pct"],
        [
            biface_w,
            biface["R_Biface_s (%)"],
            biface["R_Biface_p (%)"],
            biface["R_Biface_avg (%)"],
        ],
        "Two-side-coated component: beam splitter run 260317-035 on the entrance face, "
        "antireflection run 260317-034 on the exit face, one polished silicon substrate. "
        "Reflectance at 45 degrees in s and p, in percent, acquisition "
        "'BSplusAR260317-034-35' of 17 March 2026 16:38. Deposited as the example that "
        "exercises a two-sided sample; it is not one of the three components the article "
        "reports on.",
    )
    written.append("spectra/BIFACE_R45deg.csv")

    write_csv(
        out / "spectra" / "BS17_R45deg.csv",
        ["Wavelength_nm", "R_s_pct", "R_p_pct"],
        [bs17_w, bs17_s, bs17_p],
        "Seventeen-layer beam splitter for 45 degrees, run 260319-037, on silicon with the "
        "rear face ground. Reflectance at 45 degrees in s and p, in percent. "
        "EssentOptics PHOTON RT SN 38425, 220 um slit, 5 mm spot, stage 45 degrees, "
        "detector 90, 1000-4000 nm at a 5 nm pitch, 20 averages, the two polarizations "
        "run separately. The analysis band of the study is 1500-4000 nm, one and the same "
        "for every sample of the campaign.",
    )
    written.append("spectra/BS17_R45deg.csv")

    # -- reference single layers ---------------------------------------------------------
    for material in ("SiO2", "Nb2O5"):
        source = campaign / f"spectres_monocouche_{material}_Trel.csv"
        columns = read_csv_columns(source)
        wavelength = columns["Wavelength_nm"]
        ratio = columns["Trel"]
        good = np.isfinite(wavelength) & np.isfinite(ratio)
        stated = re.search(
            r"Volet 1\s*:\s*([\d.]+)\s*nm", source.read_text(encoding="utf-8", errors="replace")
        )
        if stated and abs(float(stated.group(1)) - REFERENCE_THICKNESS_NM[material]) > 0.05:
            print(
                f"  the {material} reference single layer file states {stated.group(1)} nm where this "
                f"script carries {REFERENCE_THICKNESS_NM[material]} nm",
                file=sys.stderr,
            )
            return 3
        write_csv(
            out / "spectra" / f"reference single layer_{material}_Trel.csv",
            ["Wavelength_nm", "Trel"],
            [wavelength[good], ratio[good]],
            f"Single-layer {material} reference single layer on double-side-polished sapphire, same "
            f"deposition campaign as the components, deposited less than one week before "
            f"them with no target change or chamber reconditioning in between -- which is "
            f"the condition that makes the transferability test legitimate. Transmittance "
            f"relative to the bare substrate at normal incidence, as a fraction. "
            f"Thickness determined in Volet 1: {REFERENCE_THICKNESS_NM[material]} nm.",
        )
        written.append(f"spectra/reference single layer_{material}_Trel.csv")

    # -- designs -----------------------------------------------------------
    design_rows = read_xlsx_sheet(campaign / AR_WORKBOOK, "design")
    lambda0 = float(design_rows[0][1])
    qwot_ar = [float(r[0]) for r in design_rows[1:] if r[0] is not None]
    if len(qwot_ar) != 6:
        print(f"  the antireflection design has {len(qwot_ar)} layers, not 6", file=sys.stderr)
        return 3

    n_bs_layers, qwot_bs = read_pptx_qwot(log_dir / BS_DESIGN_PPTX)
    if n_bs_layers != 16 or len(qwot_bs) != 16:
        print(
            f"  the beam-splitter slide announces {n_bs_layers} layers and carries "
            f"{len(qwot_bs)} quarter-wave values; 16 of each were expected",
            file=sys.stderr,
        )
        return 3
    if max(abs(a - b) for a, b in zip(qwot_bs, BS45_QWOT_REFERENCE)) > 1e-6:
        print("  the beam-splitter slide disagrees with the design recorded here", file=sys.stderr)
        return 3
    checks.append(
        f"beam-splitter design read from the slide: 16 layers, {sum(qwot_bs):.4f} QWOT total"
    )
    checks.append(
        f"antireflection design read from the workbook: 6 layers, {sum(qwot_ar):.4f} QWOT "
        f"total, recorded to two decimals only"
    )

    # The seventeen-layer design, read in the order the workbook lists it, which is already
    # substrate-first. No reversal: the sheet and this package agree on which end is which.
    bs17_design_rows = read_xlsx_sheet(campaign / BS17_WORKBOOK, "design")
    qwot_bs17 = [float(r[0]) for r in bs17_design_rows[1:] if r[0] is not None]
    if len(qwot_bs17) != 17:
        print(
            f"  the seventeen-layer design has {len(qwot_bs17)} layers, not 17",
            file=sys.stderr,
        )
        return 3
    if max(abs(a - b) for a, b in zip(qwot_bs17, BS17_QWOT_REFERENCE)) > 1e-6:
        print(
            "  the seventeen-layer workbook disagrees with the design recorded here",
            file=sys.stderr,
        )
        return 3
    if abs(float(bs17_design_rows[0][1]) - lambda0) > 1e-9:
        print("  the seventeen-layer design uses a different reference wavelength",
              file=sys.stderr)
        return 3
    checks.append(
        f"seventeen-layer design read from the workbook: 17 layers, {sum(qwot_bs17):.4f} "
        f"QWOT total, substrate-first as listed; its thick layer is layer 13 at "
        f"{max(qwot_bs17):.3f} quarter waves"
    )

    def alternating(count: int, order=MATERIALS_FROM_SUBSTRATE) -> list[str]:
        return [order[i % 2] for i in range(count)]

    stacks = {
        "AR6": {
            "run": "260317-034",
            "comment": (
                "six-layer MWIR antireflection coating, layer 1 adjacent to the silicon. "
                "The design is recorded to two decimals of a quarter wave; on layer 3, "
                "whose nominal is 0.41, a rounding of 0.005 is 1.2 % -- the order of "
                "magnitude of the departures this study reports."
            ),
            "layers": [[m, q] for m, q in zip(alternating(6), qwot_ar)],
        },
        "BS45": {
            "run": "260317-035",
            "comment": (
                "sixteen-layer 50/50 beam splitter for 45 degrees, layer 1 adjacent to the "
                "silicon, from the design slide of 17 March 08:13 -- the morning of the "
                "measurements. 6940 nm, 32.77 quarter waves."
            ),
            "layers": [[m, q] for m, q in zip(alternating(16), qwot_bs)],
        },
        "BS17": {
            "run": "260319-037",
            "comment": (
                "seventeen-layer beam splitter for 45 degrees, layer 1 adjacent to the "
                "silicon and low index, from the working workbook of the component, which "
                "lists it substrate-first. Its thirteenth layer is a thick layer of 10.62 quarter "
                "waves, and that is what makes this component the sharpest test of the "
                "campaign: a layer that thick turns a small error of optical thickness into "
                "a large and visible shift of the fringes, and it is also what makes the "
                "component sensitive to the beam cone. 38.00 quarter waves in all, the "
                "thickest stack of the campaign."
            ),
            "layers": [
                [m, q]
                for m, q in zip(alternating(17, BS17_MATERIALS_FROM_SUBSTRATE), qwot_bs17)
            ],
        },
        "REF_SiO2": {
            "run": "reference single layer",
            "comment": "single SiO2 layer, thickness from the Volet 1 determination",
            "layers": [
                {"material": "SiO2", "thickness_nm": REFERENCE_THICKNESS_NM["SiO2"]}
            ],
        },
        "REF_Nb2O5": {
            "run": "reference single layer",
            "comment": "single Nb2O5 layer, thickness from the Volet 1 determination",
            "layers": [
                {"material": "Nb2O5", "thickness_nm": REFERENCE_THICKNESS_NM["Nb2O5"]}
            ],
        },
    }

    materials = {
        name: {
            "file": f"indices/{name}_H800_published.csv",
            "uncertainty_column": "sigma_n",
            "valid_range_nm": list(PUBLISHED_VALID_RANGE_NM),
        }
        for name in ("Nb2O5", "SiO2")
    }
    substrates = {
        "silicon": {
            "file": "indices/Si_Li1980.csv",
            "valid_range_nm": list(SILICON_RANGE_NM),
        },
        "sapphire": {"file": "indices/Al2O3_Malitson.csv"},
    }

    def instrument(*, aperture: str, crosstalk: str) -> dict:
        return {
            "name": "EssentOptics PHOTON RT, 185-5200 nm configuration, SN 38425",
            "beam_aperture_deg": 2.0,
            "aperture_band_edges_nm": [2530.0, 3700.0],
            "aperture_mode": aperture,
            "aperture_bounds_deg": [1.0, 2.5],
            "aperture_min_angle_deg": 10.0,
            "n_aperture_nodes": 2,
            "crosstalk_alpha": 0.0,
            "crosstalk_beta": 0.0,
            "crosstalk_bounds": [0.0, 0.15],
            "crosstalk_mode": crosstalk,
            "comment": (
                "The beam aperture is geometric, hence achromatic at fixed configuration: "
                "the train is all-mirror and the only intrinsic chromatic term, "
                "diffraction, is 0.06 deg at 5.2 um for a 5 mm spot. What changes, and "
                "discontinuously, is the configuration of the instrument: the InGaAs/PbSe "
                "detector switchover at 2530 nm and the halogen/IR source switchover at "
                "3700 nm. Those two wavelengths are manufacturer data and are imposed; only "
                "the height of each step is released, within the bounds above, which "
                "bracket the specified divergence of +/- 1 degree. The grating is excluded "
                "-- it turns continuously while the pupil stays fixed -- and so is the "
                "slit, constant at 220 um over the whole campaign. Polarizer leakage is "
                "modelled with two coefficients and not one: the exit slit is 220 um wide "
                "for a 5 to 6 mm high spot, the polarizers sit immediately behind it, and "
                "the transmission axis turns by 90 degrees between the two settings, so the "
                "effective extinction ratio has no reason to be the same in both."
            ),
        }

    def free_parameters(
        *,
        aperture: str,
        crosstalk: str,
        thicknesses=("*",),
        index: dict | None = None,
        tolerance: float = THICKNESS_WINDOW,
    ) -> dict:
        spec = {
            "thicknesses": list(thicknesses),
            "thickness_tolerance": tolerance,
            "index_correction": "none",
            "aperture": aperture,
            "crosstalk": crosstalk,
            "substrate_index": "literature",
            "angle_offset": "nominal",
        }
        if index is not None:
            spec.update(index)
        return spec

    # -- samples -----------------------------------------------------------
    # The rear face is what separates these three components, and mistaking it costs
    # twenty-five points of reflectance. The two coatings measured alone sit on a substrate
    # whose rear face is ground, which scatters the second interface out of the collected
    # beam: that is a semi-infinite substrate. The assembled component is a real plate,
    # polished on both faces and coated on both.

    # The 17 March session log, plus the 19 March run which was saved on its own and in the
    # newer format. One dictionary, because a sample should not have to know which file its
    # settings were recorded in.
    parameters = read_xls_parameters(log_dir / ALLCNES)
    parameters.update(read_xlsx_parameters(campaign / BS17_PARAMS))
    for acq in ("260319-037-R-PolS-Unpolished", "260319-037-R-PolP-Unpolished"):
        if acq in parameters:
            parameters[acq]["Slit width, µm"] = "220"
            parameters[acq]["Spot size, mm"] = "5"

    def measurement(
        file,
        column,
        *,
        quantity,
        angle,
        polarization,
        label,
        acquisition=None,
        stride=1,
        band=BAND_NM,
    ):
        name = Path(file).name
        recorded = ACQUISITION_OF.get((name, column))
        spec = {
            "file": file,
            "column": column,
            "units": "percent" if column.endswith("_pct") else "fraction",
            "quantity": quantity,
            "angle_deg": angle,
            "polarization": polarization,
            "band_nm": list(band),
            "sigma": SIGMA_COMPONENT if column.endswith("_pct") else SIGMA_REFERENCE,
            "label": label,
        }
        if stride != 1:
            spec["stride"] = stride
        if recorded is not None:
            spec["acquisition"] = acquisition_block(parameters, recorded)
        elif acquisition is not None:
            spec["acquisition"] = acquisition
        return spec

    def sample_ar(*, rear="none", column="R_a_pct"):
        return {
            "name": "AR6_alone",
            "comment": (
                "antireflection coating measured alone, on a substrate whose rear face is "
                "ground; the rear return is scattered out of the collected beam, which is "
                "what the 'noBK' marker of the acquisition records"
            ),
            "substrate": {"material": "silicon", "thickness_mm": 1.0, "rear": rear},
            "front_stack": "AR6",
            "measurements": [
                measurement(
                    "spectra/AR6_R8deg.csv",
                    column,
                    quantity="R",
                    angle=8.0,
                    polarization="a",
                    label="AR6, R at 8 deg, unpolarized",
                )
            ],
        }

    def sample_bs(*, channels=("s", "p"), angle=45.0, stride=1):
        return {
            "name": "BS45_alone",
            "comment": (
                "beam splitter measured alone, rear face ground"
                + (
                    ", in s and p resolved -- the half-sum is invariant under a symmetric "
                    "polarizer leakage and would hide it"
                    if tuple(channels) == ("s", "p")
                    else ", in the instrument's unpolarized channel"
                )
            ),
            "substrate": {"material": "silicon", "thickness_mm": 1.0, "rear": "none"},
            "front_stack": "BS45",
            "measurements": [
                measurement(
                    "spectra/BS45_R45deg.csv",
                    f"R_{channel}_pct",
                    quantity="R",
                    angle=angle,
                    polarization=channel,
                    label=f"BS45, R at {angle:g} deg, {channel}",
                    stride=stride,
                )
                for channel in channels
            ],
        }

    def sample_biface():
        """The two-side-coated component. Not an article sample; see examples/ below."""
        return {
            "name": "BIFACE_BSplusAR",
            "comment": (
                "beam splitter run 260317-035 on the entrance face, antireflection run "
                "260317-034 on the exit face, one polished silicon substrate coated on both "
                "sides. It names coatings that other samples already name, so it adds "
                "measured points and no unknown."
            ),
            "substrate": {"material": "silicon", "thickness_mm": 1.0, "rear": "coated"},
            "front_stack": "BS45",
            "rear_stack": "AR6",
            "measurements": [
                measurement(
                    "spectra/BIFACE_R45deg.csv",
                    f"R_{channel}_pct",
                    quantity="R",
                    angle=45.0,
                    polarization=channel,
                    label=f"two-side-coated component, R at 45 deg, {channel}",
                )
                for channel in ("s", "p")
            ],
        }

    def sample_bs17(*, band=BS17_BAND_NM):
        return {
            "name": "BS17_alone",
            "comment": (
                "seventeen-layer beam splitter, measured on its own two days after the rest "
                "of the campaign, rear face ground. The component the article turns on: it "
                "is the only one measured below 1500 nm, the only one carrying a 2.7 um "
                "thick layer, and the one whose spectrum the historical campaign could not "
                "reproduce."
            ),
            "substrate": {"material": "silicon", "thickness_mm": 1.0, "rear": "none"},
            "front_stack": "BS17",
            "measurements": [
                measurement(
                    "spectra/BS17_R45deg.csv",
                    f"R_{channel}_pct",
                    quantity="R",
                    angle=45.0,
                    polarization=channel,
                    label=f"BS17, R at 45 deg, {channel}",
                    band=band,
                )
                for channel in ("s", "p")
            ],
        }

    def sample_reference single layer(material: str):
        return {
            "name": f"reference single layer_{material}",
            "comment": (
                "single-layer reference single layer of the same deposition campaign; constrains the "
                "dispersion tightly, many fringes for one unknown thickness"
            ),
            "substrate": {"material": "sapphire", "thickness_mm": 1.0, "rear": "bare"},
            "front_stack": f"WIT_{material}",
            "measurements": [
                measurement(
                    f"spectra/reference single layer_{material}_Trel.csv",
                    "Trel",
                    quantity="Trel",
                    angle=0.0,
                    polarization="a",
                    label=f"{material} reference single layer, relative transmittance",
                    # The reference single layers were recorded for Volet 1 and are reproduced from its
                    # deposit; the session log of 17 March does not carry them. Only what
                    # that deposit states is repeated here -- nothing is filled in by
                    # analogy with the component acquisitions.
                    acquisition={
                        "acquisition": f"Volet 1 reference single layer, {material}",
                        "datetime": "March 2026, less than one week before the components",
                        "stage_angle_deg": "0",
                        "detector_angle_deg": "180",
                        "polarizer": "none",
                        "quantity": "transmittance relative to the bare substrate",
                        "source": "Volet 1 Zenodo deposit, transmittance_fit_*.csv",
                    },
                )
            ],
        }

    provenance = {
        "campaign": "Institut Fresnel, Buhler Leybold Optics HELIOS 800, March 2026",
        "instrument": "EssentOptics PHOTON RT SN 38425, 185-5200 nm configuration",
        "inversion_window_nm": list(BAND_NM),
        "inversion_window_reason": (
            "one window for the five samples so that their residuals are comparable. The "
            "lower edge is imposed by Li's formula for silicon, which has a pole at 1107 nm."
        ),
        "sigma_reference": {"value": SIGMA_REFERENCE, "source": "Volet 1 measured noise floor"},
        "sigma_component": {
            "value": SIGMA_COMPONENT,
            "source": SIGMA_COMPONENT_SOURCE,
        },
        "built_by": "tools/build_study_volet2.py",
    }

    # -- the ladder --------------------------------------------------------
    # One study per rung, each adding exactly one thing to the one before it, so that every
    # change in a residual can be attributed to one cause. The numbering is what makes the
    # progression legible in a directory listing; it is also the order the article's summary
    # table is generated in.
    # The polarizer leakage is NOT released by any deposited study, and the reason is that it
    # earns nothing. Released on the joint campaign it costs two parameters, leaves the
    # residual where it was to within a third of the measured repeatability, and makes the
    # retrieved coatings *further* from their designs -- 1.26 against 1.02 % of layer-to-layer
    # dispersion on the six-layer coating. A block that buys no residual and degrades the
    # answer is a block that does not belong in the model. The capability remains in the
    # package, tested, for anyone whose instrument needs it.
    everything = {"aperture": "fitted", "crosstalk": "none"}
    nothing = {"aperture": "imposed", "crosstalk": "none"}
    # The final model releases NO instrument parameter at all. The beam aperture is imposed at
    # the manufacturer's 2.0 deg, and that value is not taken on trust: scanning it with the
    # thicknesses free puts the minimum of the beam-splitter residual in s at 2.0 deg, rising
    # by a third on either side. Releasing it afterwards buys 0.006 point of residual for
    # three parameters, two of which end on a bound. The scan has already answered the
    # question; the parameters would only spend degrees of freedom re-answering it.
    final = {"aperture": "imposed", "crosstalk": "none"}

    ladder = [
        {
            "file": "00_prediction.json",
            "name": "prediction",
            "rung": "0. prediction",
            "question": "does the determination transfer at all, with nothing adjusted?",
            "comment": (
                "PREDICTION, NOT INVERSION. The five samples, the designs as they left the "
                "coating shop, and the optical constants determined on the reference single layers -- and "
                "not one released parameter. Every other rung of this ladder is read against "
                "this one, because it is the only result that owes nothing to an "
                "adjustment. It costs zero degrees of freedom, so its residual cannot have "
                "been bought."
            ),
            "samples": [
                sample_reference single layer("SiO2"),
                sample_reference single layer("Nb2O5"),
                sample_ar(),
                sample_bs(),
                sample_bs17(),
            ],
            "stacks": stacks,
            "free": {"thicknesses": ()},
            **nothing,
        },
        {
            "file": "01_reference_layers.json",
            "name": "reference single layers",
            "rung": "1. the floor",
            "question": "does this implementation reproduce the determination it is testing?",
            "comment": (
                "The two reference single layers alone, one thickness released each and "
                "their dispersion held. Reproduces the determination published with Volet 1 "
                "with an independent implementation, and sets the residual floor against "
                "which every component is read. Neither instrument block is released: both "
                "samples are measured at normal incidence without a polarizer, so neither "
                "quantity would be constrained."
            ),
            "samples": [sample_reference single layer("SiO2"), sample_reference single layer("Nb2O5")],
            "stacks": {k: stacks[k] for k in ("REF_SiO2", "REF_Nb2O5")},
            **nothing,
        },
        {
            "file": "02_AR6_alone.json",
            "name": "AR6_alone",
            "rung": "2. one coating",
            "question": "what does ordinary reverse engineering give on six layers?",
            "comment": (
                "Reverse engineering as it is usually published: one component, layer "
                "thicknesses released, indices held. Measured at 8 degrees without a "
                "polarizer -- below the 10 degrees under which the cone average is not "
                "applied, and with nothing to constrain the leakage -- so both instrument "
                "blocks stay held. This is also the configuration the non-regression test "
                "compares against the reference implementation."
            ),
            "samples": [sample_ar()],
            "stacks": {k: stacks[k] for k in ("AR6",)},
            **nothing,
        },
        {
            "file": "03_BS45_unpolarized.json",
            "name": "BS45_unpolarized",
            "rung": "2. one coating",
            "question": "and on sixteen layers, in the unpolarized channel?",
            "comment": (
                "The same exercise on the sixteen-layer beam splitter, read in the "
                "instrument's unpolarized channel: sixteen unknowns against 250 points, "
                "sixteen points per parameter. The half-sum is rigorously invariant under a "
                "symmetric polarizer leakage, so this rung cannot see one -- which is "
                "exactly why the next one exists."
            ),
            "samples": [sample_bs(channels=("a",))],
            "stacks": {k: stacks[k] for k in ("BS45",)},
            **nothing,
        },
        {
            "file": "04_BS45_resolved.json",
            "name": "BS45_resolved",
            "rung": "3. polarization resolved",
            "question": "what appears when the two channels are separated?",
            "comment": (
                "The same sixteen unknowns against twice the data: R_s and R_p measured "
                "separately. The instrument blocks are still held, so any disagreement "
                "between the two channels has to be carried by the layer thicknesses -- and "
                "the residual says whether they can carry it."
            ),
            "samples": [sample_bs()],
            "stacks": {k: stacks[k] for k in ("BS45",)},
            **nothing,
        },
        {
            "file": "05_BS17_resolved.json",
            "name": "BS17_resolved",
            "rung": "3. polarization resolved",
            "question": "and on the component with a thick layer, over a wider band?",
            "comment": (
                "THE CENTREPIECE. Seventeen unknowns against 1118 measured points, R_s and "
                "R_p separately, over the common 1500-4000 nm window of the campaign -- "
                "the campaign can reach. Its thirteenth layer is a thick layer of 10.6 quarter waves, "
                "which makes it the most demanding component here: a layer that thick turns "
                "a small error of optical thickness into a large shift of the fringes, and "
                "leaves nowhere for such an error to hide. It is also the component the "
                "historical campaign could not reproduce, stopping at 28.5 % after fifty "
                "thousand restarts, because it read the design in the order the workbook "
                "prints it rather than the order it was deposited in."
            ),
            "samples": [sample_bs17()],
            "stacks": {k: stacks[k] for k in ("BS17",)},
            **final,
        },
        {
            "file": "06_BS17_aperture.json",
            "name": "BS17_aperture",
            "rung": "4. the instrument",
            "question": "how much of the residual is the finite beam cone?",
            "comment": (
                "The beam aperture released on the component that can actually determine it, "
                "one total aperture per band, within bounds that bracket the manufacturer's "
                "stated divergence. The step wavelengths stay imposed: they are the "
                "documented detector and source switchovers, not parameters. This is the "
                "only instrument block released anywhere in this deposit, and it is released "
                "here rather than on the sixteen-layer coating because a thick layer is what "
                "makes a beam cone visible -- on a stack without one the minimum is too flat "
                "to locate."
            ),
            "samples": [sample_bs17()],
            "stacks": {k: stacks[k] for k in ("BS17",)},
            "aperture": "fitted",
            "crosstalk": "none",
        },
        {
            "file": "07_joint_campaign.json",
            "name": "joint_campaign",
            "rung": "5. cross-checking",
            "question": "can one set of optical constants account for all five samples at once?",
            "comment": (
                "THE CAMPAIGN INVERTED AT ONCE. Five samples, three coating runs, one set of "
                "optical constants: the two reference single layers and the three components, "
                "forty-one thicknesses in all. Nothing is shared between the components "
                "except the two dispersions -- which is the whole point, because those "
                "dispersions are what Volet 1 determined and what this article is testing. "
                "Expect every individual residual to be no better here than when that "
                "component was inverted alone, and read the amount by which each degrades: "
                "that, and not the total, is what says whether one determination serves "
                "three components. A fit that only ever improves as data are added is "
                "testing nothing."
            ),
            "samples": [
                sample_reference single layer("SiO2"),
                sample_reference single layer("Nb2O5"),
                sample_ar(),
                sample_bs(),
                sample_bs17(),
            ],
            "stacks": stacks,
            **final,
        },
    ]

    # -- examples ----------------------------------------------------------
    # The package is specified to invert one component or several at once, each coated on one
    # face or on both. The three components of the article are all coated on one face, so two
    # cells of that matrix would go undemonstrated. These two studies fill them. They are not
    # results and the article does not report them; they exist so that the claim in the README
    # fails loudly if it ever stops being true.
    examples = [
        {
            "file": "two_faces_coated.json",
            "name": "example_two_faces_coated",
            "example": True,
            "comment": (
                "EXAMPLE, NOT A RESULT. One component coated on both faces: the beam "
                "splitter on the entrance face and the antireflection coating on the exit "
                "face of one polished silicon substrate, all twenty-two thicknesses "
                "released. Deposited to demonstrate that a two-sided sample inverts, and to "
                "give a reader a worked file for that case. The article reports on the three "
                "single-face components instead."
            ),
            "samples": [sample_biface()],
            "stacks": {k: stacks[k] for k in ("AR6", "BS45")},
            **final,
        },
        {
            "file": "several_components_two_faces.json",
            "name": "example_several_components_two_faces",
            "example": True,
            "comment": (
                "EXAMPLE, NOT A RESULT. Several components at once, one of them coated on "
                "both faces: the two coatings measured alone and the assembled component "
                "that carries both, inverted together. This is the most general case the "
                "package claims -- several samples sharing stacks, one of them two-sided -- "
                "and it is deposited so that the claim is exercised rather than asserted."
            ),
            "samples": [sample_ar(), sample_bs(), sample_biface()],
            "stacks": {k: stacks[k] for k in ("AR6", "BS45")},
            **final,
        },
    ]

    # -- counter-experiments ----------------------------------------------
    # What the protocol forbids, done on purpose and measured. They live in their own
    # directory so that nobody picks one up expecting a result: each is wrong by
    # construction, and the point is how wrong.
    counter = [
        {
            "file": "index_released.json",
            "name": "counter_index_released",
            "lesson": (
                "release a correction to Re(n), inside the published uncertainty envelope"
            ),
            "comment": (
                "COUNTER-EXPERIMENT. The joint campaign with a correction to Re(n) released "
                "for both materials, confined to a tube no wider than the published "
                "uncertainty envelope. The residual falls. It has to: twenty more "
                "parameters, and they act on the quantity the spectra are most sensitive "
                "to. What is lost is the argument -- a procedure free to reshape n(lambda) "
                "cannot establish that a previously determined n(lambda) was correct, "
                "however small the tube. This study exists to put a number on what is given "
                "up, not to be quoted as a result."
            ),
            "samples": [
                sample_reference single layer("SiO2"),
                sample_reference single layer("Nb2O5"),
                sample_ar(),
                sample_bs(),
                sample_bs17(),
            ],
            "stacks": stacks,
            "free": {
                "index": {
                    "index_correction": "bounded",
                    "index_tube_delta": 0.01,
                    "index_n_knots": 5,
                }
            },
            **everything,
        },
        {
            "file": "angle_shifted.json",
            "name": "counter_angle_shifted",
            "lesson": (
                "impose the angle of incidence a quarter of a degree away from nominal"
            ),
            "comment": (
                "COUNTER-EXPERIMENT. Rung 4 of the ladder with the angle of incidence set "
                "to 45.25 degrees instead of 45. Nothing else changes. The angle is not "
                "released -- releasing it would let it absorb almost any error of the model "
                "-- it is *imposed* at a shifted value, which isolates the effect instead of "
                "hiding it. Compare the retrieved total thickness with that of "
                "04_BS45_resolved.json: the difference is the sensitivity of the whole "
                "result to a quantity nobody measures to better than a fraction of a degree."
            ),
            "samples": [sample_bs(angle=45.25)],
            "stacks": {k: stacks[k] for k in ("BS45",)},
            **everything,
        },
        {
            "file": "search_window_wide.json",
            "name": "counter_search_window_wide",
            "lesson": "open the search window on the thicknesses to plus or minus fifty percent",
            "comment": (
                "COUNTER-EXPERIMENT, and the one that shows the problem is ill-posed. Rung 4 "
                "with the search window on each thickness opened from 3 % to 50 % of "
                "nominal. Many thickness vectors reproduce one spectrum to within the "
                "photometry, so the optimiser is free to walk down a nearly flat valley -- "
                "and it does, into a compensating pair of adjacent layers at +15 and -17 %, "
                "which no coating made under optical monitoring could be. What it buys is "
                "0.04 point of residual in s, a third of the measured repeatability, and it "
                "makes p worse. What it costs is a threefold increase in the layer-to-layer "
                "dispersion of the answer. Nothing in the residual says any of this has "
                "happened, which is the entire point: the search window is prior "
                "information, and a reverse-engineering result is not interpretable without "
                "it being declared."
            ),
            "samples": [sample_bs()],
            "stacks": {k: stacks[k] for k in ("BS45",)},
            "free": {"tolerance": 0.50},
            **everything,
        },
        {
            "file": "mesh_decimated.json",
            "name": "counter_mesh_decimated",
            "lesson": "keep one spectral point in eight",
            "comment": (
                "COUNTER-EXPERIMENT. Rung 4 with one point in eight. The residual barely "
                "-- fewer points, the same parameters, an easier fit -- while the "
                "layer-to-layer dispersion of the retrieved coating grows. Less data give a "
                "better-looking fit and a worse determination. This is the plainest "
                "demonstration available that a residual quoted alone says nothing about "
                "how well a coating has been reconstructed."
            ),
            "samples": [sample_bs(stride=8)],
            "stacks": {k: stacks[k] for k in ("BS45",)},
            **everything,
        },
        {
            "file": "rear_face_bare.json",
            "name": "counter_rear_face_bare",
            "lesson": "model the ground rear face as a polished one",
            "comment": (
                "COUNTER-EXPERIMENT, forward only. The antireflection coating modelled as a "
                "plate with a bare polished rear face, where the sample actually has a "
                "ground one. A ground face scatters the return of the second interface out "
                "of the collected beam; a polished one sends back the 30 % Fresnel "
                "reflection of silicon. Nothing is released, so the whole departure is the "
                "modelling error."
            ),
            "samples": [sample_ar(rear="bare")],
            "stacks": {k: stacks[k] for k in ("AR6",)},
            "free": {"thicknesses": ()},
            **nothing,
        },
        {
            "file": "layer_order_reversed.json",
            "name": "counter_layer_order_reversed",
            "lesson": "write the design in the other order, air side first",
            "comment": (
                "COUNTER-EXPERIMENT, forward only. The antireflection design written in the "
                "other order, so that layer 1 faces the air instead of the substrate. The "
                "convention of this package is that layers[0] is the layer adjacent to the "
                "substrate -- the first one deposited -- and this study is what that "
                "convention costs when it is got wrong. It is not a subtle error, which is "
                "the good news: it cannot survive a single forward evaluation."
            ),
            "samples": [sample_ar()],
            "stacks": {
                "AR6": {
                    **stacks["AR6"],
                    "comment": "THE SAME DESIGN, DELIBERATELY REVERSED. Not the coating.",
                    "layers": list(reversed(stacks["AR6"]["layers"])),
                }
            },
            "free": {"thicknesses": ()},
            **nothing,
        },
        {
            "file": "substrate_flat_table.json",
            "name": "counter_substrate_flat_table",
            "lesson": (
                "use the dispersionless silicon table an earlier revision distributed"
            ),
            "comment": (
                "COUNTER-EXPERIMENT, forward only. The antireflection coating computed with "
                "the silicon table an earlier revision of this deposit distributed under the "
                "name of Li's 1980 determination: flat at 3.5545 over the whole band, where "
                "the formula gives 3.4804 at 1500 nm and 3.4271 at 4000 nm. The substrate is "
                "optically in series with every layer, so the error propagates into every "
                "retrieved thickness. Measurements of bare silicon made during the same "
                "campaign settle it: the formula reproduces them to 0.6 point, the flat "
                "table is wrong by 1.3 to 1.8. Check a dispersion table against its formula "
                "before trusting it, above all when its name announces a reference."
            ),
            "samples": [sample_ar()],
            "stacks": {k: stacks[k] for k in ("AR6",)},
            "free": {"thicknesses": ()},
            "substrates": {
                "silicon": {
                    "file": "Si_flat_NOT_LI1980.csv",
                    "valid_range_nm": list(SILICON_RANGE_NM),
                },
                "sapphire": {"file": "../indices/Al2O3_Malitson.csv"},
            },
            **nothing,
        },
    ]

    def relocate(document: dict, prefix: str) -> dict:
        """Point every relative path one directory further up.

        Counter-experiments sit in a subdirectory and share the data files of the ladder;
        paths stay relative to the study file, which is what lets the whole directory be
        moved or archived as a unit.
        """
        for group in ("materials", "substrates"):
            for spec in document[group].values():
                spec["file"] = prefix + spec["file"]
        for sample in document["samples"]:
            for m in sample["measurements"]:
                m["file"] = prefix + m["file"]
        return document

    counter_dir = out / "counter_experiments"
    counter_dir.mkdir(parents=True, exist_ok=True)

    flat_w = np.arange(SILICON_RANGE_NM[0], SILICON_RANGE_NM[1] + 1e-9, 50.0)
    write_csv(
        counter_dir / "Si_flat_NOT_LI1980.csv",
        ["Wavelength_nm", "n", "k"],
        [flat_w, np.full(flat_w.size, 3.5545), np.zeros_like(flat_w)],
        "NOT LI 1980, AND NOT SILICON. A dispersionless table, flat at 3.5545, distributed "
        "by an earlier revision of this deposit under the name of Li's determination. Li's "
        "formula gives 3.4804 at 1500 nm and 3.4271 at 4000 nm, so this table is too high "
        "by 0.07 to 0.13 over the band. It is kept here, under a name that cannot be "
        "mistaken, for the sole purpose of the counter-experiment "
        "substrate_flat_table.json. Do not use it for anything else.",
    )
    written.append("counter_experiments/Si_flat_NOT_LI1980.csv")

    example_dir = out / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    for spec in ladder + counter + examples:
        overrides = dict(spec.get("free") or {})
        document = {
            "name": spec["name"],
            "comment": spec["comment"],
            "lambda0_nm": lambda0,
            "provenance": {
                **provenance,
                **({"rung": spec["rung"], "question": spec["question"]} if "rung" in spec else {}),
                **({"counter_experiment": spec["lesson"]} if "lesson" in spec else {}),
            },
            "materials": materials,
            "substrates": substrates,
            "stacks": spec["stacks"],
            "samples": spec["samples"],
            "instrument": instrument(
                aperture=spec["aperture"], crosstalk=spec["crosstalk"]
            ),
            "free_parameters": free_parameters(
                aperture=spec["aperture"],
                crosstalk=spec["crosstalk"],
                thicknesses=overrides.get("thicknesses", ("*",)),
                index=overrides.get("index"),
                tolerance=overrides.get("tolerance", THICKNESS_WINDOW),
            ),
        }
        document = json.loads(json.dumps(document))  # a deep copy, so relocation is local
        if "lesson" in spec:
            document = relocate(document, "../")
            if "substrates" in spec:
                document["substrates"] = spec["substrates"]
            path = counter_dir / spec["file"]
            written.append(f"counter_experiments/{spec['file']}")
        elif spec.get("example"):
            document = relocate(document, "../")
            path = example_dir / spec["file"]
            written.append(f"examples/{spec['file']}")
        else:
            path = out / spec["file"]
            written.append(spec["file"])
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    removed = []
    for name in SUPERSEDED:
        path = out / name
        if path.is_file():
            path.unlink()
            removed.append(name)

    print(f"study written to {out}")
    for name in written:
        print(f"  {name}")
    if removed:
        print("superseded files removed:")
        for name in removed:
            print(f"  {name}")
    print("\nchecks made while building:")
    for line in checks:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
