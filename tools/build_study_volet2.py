"""Assemble the deposited study directory from the laboratory archives.

This script is the provenance trail of ``studies/volet2/``: it states, in executable form,
where every deposited number comes from. It reads the laboratory archives -- which are *not*
part of the deposit -- and writes the CSV and JSON files that are.

It is run once, before a release. A reader of the deposit does not need it; a reviewer
asking "where does this column come from" does.

Sources
-------
campaign workbooks (``01_DATA_MESURES_ET_MODELES``)
    Nominal designs in QWOT, measured spectra of the antireflection coating, of the
    beam splitter and of the two-side-coated component, tabulated indices of the two coating
    materials and of the silicon substrate.
Volet 1 Zenodo deposit (``certus_index_spline``)
    Measured relative-transmittance spectra of the two single-layer witnesses, their
    retrieved thicknesses, and the uncertainty envelope of the published optical constants.
Malitson's Sellmeier coefficients
    Sapphire, ordinary ray -- the witness substrate, tabulated here so that the deposit
    needs no substrate database.

Usage
-----
    python tools/build_study_volet2.py [--archives PATH] [--out studies/volet2]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Where the archives live. Overridable, because nothing in the deposit may
# depend on one machine's directory layout.
# --------------------------------------------------------------------------

DEFAULT_ARCHIVES = Path(
    r"D:\drivfl\01_Recherche_Enseignement\02_Couches_Minces_Optique\couches minces 2026"
)

CAMPAIGN_SUBDIR = Path("CERTUS/publication_reverse/01_DATA_MESURES_ET_MODELES")
VOLET1_SUBDIR = Path(
    "optics continuum/05_CODE_ET_ZENODO/zenodo_release/certus_index_spline/results"
)

# Sapphire, ordinary ray. I. H. Malitson, J. Opt. Soc. Am. 52, 1377 (1962).
# Sellmeier form n^2 - 1 = sum_i B_i lambda^2 / (lambda^2 - C_i), lambda in micrometres.
SAPPHIRE_SELLMEIER = (
    (1.4313493, 0.00527992877161),
    (0.65054713, 0.01423826448164),
    (5.3414021, 325.0188308009999),
)

# Photometric one-sigma used for each family of measurements.
#   witnesses : noise floor measured in Volet 1 and reported in its summary.csv
#   components: manufacturer's MWIR photometric accuracy, worst case of the quoted range.
# The component value is a specification, not a measurement: the repeated acquisition of the
# same specimen recorded during the campaign gives a proper repeatability figure and should
# replace it. Until then the number is declared here rather than hidden in the solver.
SIGMA_WITNESS = 0.002
SIGMA_COMPONENT = 0.0053
SIGMA_COMPONENT_SOURCE = (
    "EssentOptics PHOTON RT specification, MWIR photometric accuracy; "
    "to be replaced by the measured repeatability of the repeated acquisition"
)


def sellmeier_n(wavelength_nm: np.ndarray, terms) -> np.ndarray:
    """Refractive index from Sellmeier coefficients, wavelength given in nanometres."""
    lam_um2 = (np.asarray(wavelength_nm, dtype=np.float64) / 1000.0) ** 2
    n2 = np.ones_like(lam_um2)
    for b, c in terms:
        n2 = n2 + b * lam_um2 / (lam_um2 - c)
    return np.sqrt(n2)


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
                [
                    ("" if not np.isfinite(c[i]) else f"{c[i]:.10g}")
                    for c in columns
                ]
            )


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    """Read a CSV into arrays keyed by column name, skipping ``#`` comment lines."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        lines = [ln for ln in handle if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"{path}: no header")
    out: dict[str, list[float]] = {c.strip(): [] for c in reader.fieldnames}
    keep: list[bool] = []
    rows = list(reader)
    for row in rows:
        vals = {}
        ok = True
        for col in out:
            raw = row.get(col)
            if raw is None or str(raw).strip() == "":
                vals[col] = np.nan
                continue
            try:
                vals[col] = float(str(raw).replace(",", "."))
            except ValueError:
                ok = False
                break
        keep.append(ok)
        if ok:
            for col, v in vals.items():
                out[col].append(v)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def read_design_qwot(workbook: Path) -> tuple[float, list[float]]:
    """Read the nominal design from a campaign workbook.

    Returns the reference wavelength and the QWOT of each layer, in the order written in the
    workbook -- which is the order the reconstruction results are reported in.
    """
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        sheet = wb["design"]
        rows = list(sheet.iter_rows(values_only=True))
    finally:
        wb.close()
    lambda0 = float(rows[0][1])
    qwots = [float(r[0]) for r in rows[1:] if r[0] is not None]
    return lambda0, qwots


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

    archives: Path = args.archives
    campaign = archives / CAMPAIGN_SUBDIR
    volet1 = archives / VOLET1_SUBDIR
    out: Path = args.out

    missing = [p for p in (campaign, volet1) if not p.is_dir()]
    if missing:
        print("archive directories not found:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print("\npass --archives with the root of the laboratory archives", file=sys.stderr)
        return 2

    (out / "indices").mkdir(parents=True, exist_ok=True)
    (out / "spectra").mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # -- coating materials ------------------------------------------------
    # Two determinations of the same films are available and the article compares them.
    # Both are deposited, under names that say what they are, so that a study can select
    # one explicitly instead of inheriting whichever happened to be on disk.
    #
    #   *_published : the determination published with Volet 1, from relative transmittance
    #                 of the single-layer witnesses, 250-5200 nm, with its uncertainty
    #                 envelope. This is the dataset whose transferability is under test.
    #   *_campaign  : the determination carried in the campaign workbooks, which exploits
    #                 R and T, and covers 1000-5200 nm only.
    idx = read_csv_columns(campaign / "indices_H800_SiO2_Nb2O5.csv")
    for material, n_col, k_col in (
        ("Nb2O5", "n_Nb2O5_H800", "k_Nb2O5_H800"),
        ("SiO2", "n_SiO2_H800", "k_SiO2_H800"),
    ):
        wavelength = idx["Wavelength_nm"]
        write_csv(
            out / "indices" / f"{material}_H800_campaign.csv",
            ["Wavelength_nm", "n", "k"],
            [wavelength, idx[n_col], idx[k_col]],
            f"{material} deposited by PARMS on a Buhler Leybold Optics HELIOS H800, "
            f"Institut Fresnel, March 2026 campaign. Determination exploiting reflectance "
            f"and transmittance, as carried in the campaign workbooks. "
            f"Tabulated over {wavelength.min():.0f}-{wavelength.max():.0f} nm only.",
        )
        written.append(f"indices/{material}_H800_campaign.csv")

        v1_file = volet1 / f"optical_constants_{material}.csv"
        if not v1_file.is_file():
            print(f"  published optical constants missing: {v1_file}", file=sys.stderr)
            continue
        v1 = read_csv_columns(v1_file)
        w1 = v1["lambda (nm)"]
        n1, k1 = v1["n nominal"], v1["k nominal"]
        good = np.isfinite(w1) & np.isfinite(n1)
        sigma = 0.5 * (v1["n max"] - v1["n min"])
        write_csv(
            out / "indices" / f"{material}_H800_published.csv",
            ["Wavelength_nm", "n", "k", "sigma_n"],
            [w1[good], n1[good], np.nan_to_num(k1[good]), sigma[good]],
            f"{material}, same films, as published with the Volet 1 deposit: determined "
            f"from the relative transmittance of the single-layer witnesses. sigma_n is "
            f"the half-width of the published uncertainty envelope. Beyond the last "
            f"interference extremum the index is no longer constrained by the fringe "
            f"positions; see the valid_range_nm declared in the study files.",
        )
        written.append(f"indices/{material}_H800_published.csv")

    # -- substrates -------------------------------------------------------
    si = read_csv_columns(campaign / "indice_substrat_Silicium_Li1980.csv")
    write_csv(
        out / "indices" / "Si_Li1980.csv",
        ["Wavelength_nm", "n", "k"],
        [si["Wavelength_nm"], si["n_Silicon_Li1980"], si["k_Silicon"]],
        "Silicon. H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980). "
        "Propagated as given, never adjusted.",
    )
    written.append("indices/Si_Li1980.csv")

    sapphire_w = np.arange(250.0, 5201.0, 10.0)
    write_csv(
        out / "indices" / "Al2O3_Malitson.csv",
        ["Wavelength_nm", "n", "k"],
        [
            sapphire_w,
            sellmeier_n(sapphire_w, SAPPHIRE_SELLMEIER),
            np.zeros_like(sapphire_w),
        ],
        "Sapphire, ordinary ray. I. H. Malitson, J. Opt. Soc. Am. 52, 1377 (1962). "
        "Tabulated here from the Sellmeier coefficients so that the deposit carries no "
        "substrate database. Substrate of the single-layer witnesses.",
    )
    written.append("indices/Al2O3_Malitson.csv")

    # -- component spectra ------------------------------------------------
    ar = read_csv_columns(campaign / "spectres_mesures_AR_6layers.csv")
    write_csv(
        out / "spectra" / "AR6_R8deg.csv",
        ["Wavelength_nm", "R_a_pct"],
        [ar["Wavelength_nm"], ar["R_8deg_avg_pct"]],
        "Six-layer antireflection coating, run 260317-034, on double-side polished "
        "silicon. Reflectance at 8 degrees, no polarizer. EssentOptics PHOTON RT "
        "SN 38425, March 2026, 220-240 um slit.",
    )
    written.append("spectra/AR6_R8deg.csv")

    # The archive file names keep the laboratory's internal label for this component; the
    # deposit calls it by what it is, a beam splitter for 45 degrees.
    bs = read_csv_columns(campaign / "spectres_mesures_BSCNES_17layers.csv")
    bs_avg = 0.5 * (bs["R_45deg_s_pct"] + bs["R_45deg_p_pct"])
    write_csv(
        out / "spectra" / "BS45_R45deg.csv",
        ["Wavelength_nm", "R_s_pct", "R_p_pct", "R_a_pct"],
        [bs["Wavelength_nm"], bs["R_45deg_s_pct"], bs["R_45deg_p_pct"], bs_avg],
        "Seventeen-layer 50/50 beam splitter, run 260317-035, on double-side polished "
        "silicon. Reflectance at 45 degrees in s and p. R_a is the half-sum, which is "
        "what an unpolarized measurement returns and what this study inverts; it is "
        "rigorously invariant under a symmetric mixing of the two channels.",
    )
    written.append("spectra/BS45_R45deg.csv")

    bf = read_csv_columns(campaign / "spectres_mesures_BIFACE_BSplusAR.csv")
    write_csv(
        out / "spectra" / "BIFACE_R45deg.csv",
        ["Wavelength_nm", "R_s_pct", "R_p_pct", "R_a_pct"],
        [
            bf["Wavelength_nm"],
            bf["R_biface_s_pct"],
            bf["R_biface_p_pct"],
            bf["R_biface_avg_pct"],
        ],
        "Two-side-coated component: beam splitter run 260317-035 on the entrance face, "
        "antireflection run 260317-034 on the exit face, one silicon substrate. "
        "Reflectance at 45 degrees. Same two coating runs as the two components measured "
        "alone, hence the same layer thicknesses.",
    )
    written.append("spectra/BIFACE_R45deg.csv")

    # -- witness spectra, from the Volet 1 deposit -------------------------
    witness_thickness = {}
    summary = volet1 / "summary.csv"
    if summary.is_file():
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                witness_thickness[row["material"]] = float(row["d_opt_nm"])

    for material in ("SiO2", "Nb2O5"):
        fit = volet1 / f"transmittance_fit_{material}.csv"
        if not fit.is_file():
            print(f"  witness spectrum missing: {fit}", file=sys.stderr)
            continue
        cols = read_csv_columns(fit)
        wavelength = cols["T exp_x"]
        ratio = cols["Experimental ratio"]
        good = np.isfinite(wavelength) & np.isfinite(ratio)
        write_csv(
            out / "spectra" / f"witness_{material}_Trel.csv",
            ["Wavelength_nm", "Trel"],
            [wavelength[good], ratio[good]],
            f"Single-layer {material} witness on double-side polished sapphire, same "
            f"deposition campaign as the components, less than one week apart, no target "
            f"change or chamber reconditioning in between. Transmittance relative to the "
            f"bare substrate at normal incidence, as a fraction. Reproduced from the "
            f"Volet 1 deposit.",
        )
        written.append(f"spectra/witness_{material}_Trel.csv")

    # -- designs ----------------------------------------------------------
    lambda0_ar, qwot_ar = read_design_qwot(campaign / "AR_filter_6layers_H800.xlsx")
    lambda0_bs, qwot_bs = read_design_qwot(campaign / "BSCNES_filter_17layers_H800.xlsx")
    if lambda0_ar != lambda0_bs:
        print(
            f"  the two designs use different reference wavelengths "
            f"({lambda0_ar} and {lambda0_bs} nm)",
            file=sys.stderr,
        )
        return 3
    lambda0 = lambda0_ar

    # Material sequence of each run. The antireflection coating alternates starting with the
    # high index, the beam splitter starting with the low index; both were verified against
    # the layer-by-layer reconstruction tables of the campaign.
    mats_ar = ["Nb2O5", "SiO2"] * 3
    mats_bs = (["SiO2", "Nb2O5"] * 9)[:17]
    if len(qwot_ar) != 6 or len(qwot_bs) != 17:
        print(
            f"  unexpected layer counts: {len(qwot_ar)} and {len(qwot_bs)}",
            file=sys.stderr,
        )
        return 3

    stacks = {
        "AR6": {
            "run": "260317-034",
            "comment": "six-layer MWIR antireflection coating",
            "layers": [[m, q] for m, q in zip(mats_ar, qwot_ar)],
        },
        "BS45": {
            "run": "260317-035",
            "comment": "seventeen-layer 50/50 beam splitter for 45 degrees",
            "layers": [[m, q] for m, q in zip(mats_bs, qwot_bs)],
        },
        "WIT_SiO2": {
            "run": "witness",
            "comment": "single SiO2 layer, thickness from the Volet 1 determination",
            "layers": [
                {
                    "material": "SiO2",
                    "thickness_nm": witness_thickness.get("SiO2", 1681.7),
                }
            ],
        },
        "WIT_Nb2O5": {
            "run": "witness",
            "comment": "single Nb2O5 layer, thickness from the Volet 1 determination",
            "layers": [
                {
                    "material": "Nb2O5",
                    "thickness_nm": witness_thickness.get("Nb2O5", 1715.9),
                }
            ],
        },
    }

    # The published determination is the one whose transferability the study tests, and it
    # is the only one that spans the witness range. Beyond the last interference extremum,
    # near 4840 nm, Volet 1 states that the index is no longer constrained by the fringe
    # positions; that is declared here so the report can say when a run leaned on it.
    materials = {
        "Nb2O5": {
            "file": "indices/Nb2O5_H800_published.csv",
            "uncertainty_column": "sigma_n",
            "valid_range_nm": [250.0, 4840.0],
        },
        "SiO2": {
            "file": "indices/SiO2_H800_published.csv",
            "uncertainty_column": "sigma_n",
            "valid_range_nm": [250.0, 4840.0],
        },
    }
    materials_campaign = {
        "Nb2O5": {"file": "indices/Nb2O5_H800_campaign.csv"},
        "SiO2": {"file": "indices/SiO2_H800_campaign.csv"},
    }
    substrates = {
        "silicon": {"file": "indices/Si_Li1980.csv"},
        "sapphire": {"file": "indices/Al2O3_Malitson.csv"},
    }
    instrument = {
        "name": "EssentOptics PHOTON RT, 185-5200 nm configuration, SN 38425",
        "beam_aperture_deg": 2.0,
        "aperture_band_edges_nm": [2530.0, 3700.0],
        "aperture_mode": "imposed",
        "aperture_min_angle_deg": 10.0,
        "n_aperture_nodes": 2,
        "comment": (
            "Total cone aperture from the manufacturer's beam divergence of +/- 1 degree. "
            "Steps imposed at the documented switchovers: InGaAs/PbSe detector at 2530 nm, "
            "halogen/IR source at 3700 nm. The all-mirror train is achromatic, so nothing "
            "else can make the aperture vary with wavelength."
        ),
    }

    # The single-coating measurements are recorded with the rear-face contribution removed
    # -- the column headers carry the 'noBK' marker of the acquisition software -- so the
    # model must see a semi-infinite substrate. Taking them for plate measurements adds the
    # 30 % Fresnel return of a bare silicon face and moves the predicted reflectance of the
    # antireflection coating from 14 % to 39 %.
    def sample_ar(band=(1500.0, 4000.0)):
        return {
            "name": "AR6_alone",
            "comment": (
                "antireflection coating measured alone, rear-face contribution removed "
                "at acquisition (noBK)"
            ),
            "substrate": {"material": "silicon", "thickness_mm": 1.0, "rear": "none"},
            "front_stack": "AR6",
            "measurements": [
                {
                    "file": "spectra/AR6_R8deg.csv",
                    "column": "R_a_pct",
                    "units": "percent",
                    "quantity": "R",
                    "angle_deg": 8.0,
                    "polarization": "a",
                    "band_nm": list(band),
                    "sigma": SIGMA_COMPONENT,
                    "label": "AR6 R 8 deg unpolarized",
                }
            ],
        }

    def sample_bs(band=(1000.0, 4000.0)):
        return {
            "name": "BS45_alone",
            "comment": (
                "beam splitter measured alone, rear-face contribution removed at "
                "acquisition (noBK)"
            ),
            "substrate": {"material": "silicon", "thickness_mm": 1.0, "rear": "none"},
            "front_stack": "BS45",
            "measurements": [
                {
                    "file": "spectra/BS45_R45deg.csv",
                    "column": "R_a_pct",
                    "units": "percent",
                    "quantity": "R",
                    "angle_deg": 45.0,
                    "polarization": "a",
                    "band_nm": list(band),
                    "sigma": SIGMA_COMPONENT,
                    "label": "beam splitter 45 deg, unpolarized R",
                }
            ],
        }

    def sample_biface(band=(1500.0, 4000.0)):
        return {
            "name": "BIFACE_BSplusAR",
            "comment": (
                "operational component: the two runs above, one on each face of a single "
                "silicon substrate; adds no unknown, only constraint"
            ),
            "substrate": {"material": "silicon", "thickness_mm": 1.0, "rear": "coated"},
            "front_stack": "BS45",
            "rear_stack": "AR6",
            "measurements": [
                {
                    "file": "spectra/BIFACE_R45deg.csv",
                    "column": "R_a_pct",
                    "units": "percent",
                    "quantity": "R",
                    "angle_deg": 45.0,
                    "polarization": "a",
                    "band_nm": list(band),
                    "sigma": SIGMA_COMPONENT,
                    "label": "two-side-coated component, R 45 deg unpolarized",
                }
            ],
        }

    def sample_witness(material: str, band):
        return {
            "name": f"witness_{material}",
            "comment": (
                "single-layer witness of the same deposition campaign; constrains the "
                "dispersion tightly, many fringes for one unknown thickness"
            ),
            "substrate": {"material": "sapphire", "thickness_mm": 1.0, "rear": "bare"},
            "front_stack": f"WIT_{material}",
            "measurements": [
                {
                    "file": f"spectra/witness_{material}_Trel.csv",
                    "column": "Trel",
                    "units": "fraction",
                    "quantity": "Trel",
                    "angle_deg": 0.0,
                    "polarization": "a",
                    "band_nm": list(band),
                    "sigma": SIGMA_WITNESS,
                    "label": f"{material} witness, relative transmittance",
                }
            ],
        }

    provenance = {
        "campaign": "Institut Fresnel, Buhler Leybold Optics HELIOS H800, March 2026",
        "instrument": "EssentOptics PHOTON RT SN 38425, 185-5200 nm configuration",
        "sigma_witness": {"value": SIGMA_WITNESS, "source": "Volet 1 measured noise floor"},
        "sigma_component": {
            "value": SIGMA_COMPONENT,
            "source": SIGMA_COMPONENT_SOURCE,
        },
        "built_by": "tools/build_study_volet2.py",
    }

    studies = {
        "AR6_alone.json": {
            "name": "AR6_alone",
            "comment": (
                "Single-sample reference inversion of the antireflection coating, indices "
                "held fixed. This is the configuration the non-regression test compares "
                "against the reference implementation."
            ),
            "samples": [sample_ar()],
            "stacks": {k: stacks[k] for k in ("AR6",)},
        },
        "BS45_alone.json": {
            "name": "BS45_alone",
            "comment": (
                "Single-sample reference inversion of the beam splitter in the unpolarized "
                "channel, indices held fixed."
            ),
            "samples": [sample_bs()],
            "stacks": {k: stacks[k] for k in ("BS45",)},
        },
        "witnesses.json": {
            "name": "witnesses",
            "comment": (
                "The two single-layer witnesses alone, with their thickness released and "
                "their dispersion held. Reproduces the determination published with "
                "Volet 1 and is the reference point against which the components are read."
            ),
            "samples": [
                sample_witness("SiO2", (252.0, 5200.0)),
                sample_witness("Nb2O5", (350.0, 5200.0)),
            ],
            "stacks": {k: stacks[k] for k in ("WIT_SiO2", "WIT_Nb2O5")},
        },
        "biface_unresolved.json": {
            "name": "biface_unresolved",
            "comment": (
                "THE TWO-SIDE-COATED COMPONENT, AND AN UNRESOLVED INCONSISTENCY. This "
                "sample names the same two coating runs as the two components measured "
                "alone, so it should add measured points and no unknown. It does not: the "
                "nominal design reproduces its spectrum to 30 %, and releasing all "
                "twenty-three thicknesses over a plus or minus fifty percent window only "
                "reaches 5 %, with seven layers ending on their bound. The three "
                "measurements of run 260317-035 held in the campaign archive are flat near "
                "70 % and are not reproduced by the seventeen-layer design either, while "
                "the spectrum deposited for that run is reproduced to 1.2 %. The pairing "
                "of designs and measurements therefore needs to be confirmed at the bench "
                "before this sample can enter a joint inversion. It is kept here, alone, "
                "so that the inconsistency is reproducible rather than hidden."
            ),
            "samples": [sample_biface()],
            "stacks": {k: stacks[k] for k in ("AR6", "BS45")},
        },
        "joint_campaign.json": {
            "name": "joint_campaign",
            "comment": (
                "The campaign inverted at once: the two single-layer witnesses and the two "
                "coatings measured alone, with the dispersions held fixed and only layer "
                "thicknesses released. This is the configuration in which the "
                "transferability of the optical constants from witness to component is "
                "tested. The two-side-coated component is deliberately absent; see "
                "biface_unresolved.json."
            ),
            # Witness bands start where their own dispersion table starts: the published
            # determination of SiO2 begins at 252 nm and that of Nb2O5 at 350 nm. Asking
            # for a point outside the table would silently hold the index at its edge
            # value, which the study validation refuses.
            "samples": [
                sample_witness("SiO2", (252.0, 5200.0)),
                sample_witness("Nb2O5", (350.0, 5200.0)),
                sample_ar(),
                sample_bs(),
            ],
            "stacks": stacks,
        },
        "components_campaign_indices.json": {
            "name": "components_campaign_indices",
            "comment": (
                "The two coatings and the two-side-coated component, inverted with the "
                "campaign determination of the optical constants instead of the published "
                "one. Same samples, same bands, same imposed aperture: the only difference "
                "is the index dataset, which is what makes the two runs comparable. The "
                "witnesses are absent because that determination does not extend below "
                "1000 nm."
            ),
            "samples": [sample_ar(), sample_bs()],
            "stacks": {k: stacks[k] for k in ("AR6", "BS45")},
            "materials": materials_campaign,
        },
    }

    for filename, spec in studies.items():
        document = {
            "name": spec["name"],
            "comment": spec["comment"],
            "lambda0_nm": lambda0,
            "provenance": provenance,
            "materials": spec.get("materials", materials),
            "substrates": substrates,
            "stacks": spec["stacks"],
            "samples": spec["samples"],
            "instrument": instrument,
            "free_parameters": {
                "thicknesses": ["*"],
                "index_correction": "none",
                "aperture": "imposed",
                "substrate_index": "literature",
                "angle_offset": "nominal",
            },
        }
        path = out / filename
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.append(filename)

    print(f"study written to {out}")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
