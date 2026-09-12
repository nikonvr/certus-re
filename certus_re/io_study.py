"""Reading a study from a JSON description and CSV data files.

The input format is deliberately plain: one JSON file describing the problem, and CSV files
holding the numbers. Both are readable and diffable without this package, which matters for
an archived deposit -- a reader in ten years should be able to see what was measured without
running anything.

Every path in the JSON is resolved relative to the JSON file itself, so a study directory
can be moved, copied or archived as a unit. There are no absolute paths anywhere.

Example
-------
.. code-block:: json

    {
      "name": "volet2_joint",
      "lambda0_nm": 1500.0,
      "materials": {
        "H": {"file": "indices/Nb2O5_H800.csv", "valid_range_nm": [250, 4840]},
        "L": {"file": "indices/SiO2_H800.csv"}
      },
      "substrates": {
        "silicon":  {"file": "indices/Si_Li1980.csv"},
        "sapphire": {"file": "indices/Al2O3.csv"}
      },
      "stacks": {
        "AR6":  {"run": "260317-034", "layers": [["H", 2.55], ["L", 2.67]]},
        "BS45": {"run": "260317-035", "layers": [["L", 1.520588], ["H", 1.698967]]}
      },
      "samples": [
        {
          "name": "AR6_alone",
          "substrate": {"material": "silicon", "thickness_mm": 1.0, "rear": "bare"},
          "front_stack": "AR6",
          "measurements": [
            {"file": "spectra/AR.csv", "column": "R_8_avg_pct", "units": "percent",
             "quantity": "R", "angle_deg": 8.0, "polarization": "a",
             "band_nm": [1500, 4000], "sigma": 0.0015}
          ]
        }
      ],
      "instrument": {"beam_aperture_deg": 2.0, "aperture_band_edges_nm": [2530, 3700],
                     "aperture_mode": "fitted", "crosstalk_mode": "fitted"},
      "free_parameters": {"thicknesses": ["*"], "thickness_tolerance": 0.03,
                          "index_correction": "none",
                          "aperture": "fitted", "crosstalk": "fitted"}
    }

``thickness_tolerance`` is the search window on each thickness, as a fraction of nominal. It
is prior information about how the coating was made, not a setting of the optimiser, and it
belongs in this file for the same reason the released blocks do: the problem is ill-posed, and
a result depends on the window as much as on what was released.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dispersion import TabulatedIndex, load_index_csv
from .model import (
    FreeParameters,
    Instrument,
    Layer,
    Measurement,
    Sample,
    Stack,
    Study,
    Substrate,
)

__all__ = ["load_study", "read_spectrum_csv", "StudyError"]


class StudyError(ValueError):
    """A study description that cannot be turned into a runnable problem."""


def _require(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise StudyError(f"{where}: missing required field {key!r}")
    return mapping[key]


def read_spectrum_csv(
    path: str | Path,
    column: str,
    *,
    wavelength_column: str | None = None,
    units: str = "fraction",
    band_nm: tuple[float, float] | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one spectral column from a CSV file, by column name.

    Parameters
    ----------
    path:
        CSV file with a header row.
    column:
        Name of the column holding the measured quantity.
    wavelength_column:
        Name of the wavelength column. Guessed when omitted.
    units:
        ``"percent"`` or ``"fraction"``. Percent values are divided by 100 here, once, so
        that nothing downstream has to wonder.
    band_nm:
        Optional restriction applied immediately; points outside never reach the solver.
    stride:
        Keep one point in ``stride``. Provided for one purpose: to make the effect of a
        coarser spectral mesh reproducible. Decimating by eight *improves* the residual of
        an inversion while tripling the layer-to-layer dispersion of the result, which is
        the plainest demonstration available that a residual is not a measure of how well a
        coating has been reconstructed. A study that uses it says so in its own file.

    Returns
    -------
    (wavelength_nm, value)
        Sorted by wavelength, with blank and non-numeric entries dropped.
    """
    path = Path(path)
    if units not in ("percent", "fraction"):
        raise StudyError(f"{path}: unknown units {units!r}, expected percent or fraction")
    if int(stride) < 1:
        raise StudyError(f"{path}: stride must be at least 1, got {stride!r}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        # A leading '#' line is the provenance note of a deposited file, not data.
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise StudyError(f"{path}: no header row")
    columns = [c.strip() for c in reader.fieldnames]
    rows = [{(k.strip() if k else k): v for k, v in row.items()} for row in reader]

    if column not in columns:
        raise StudyError(f"{path}: column {column!r} not found; available: {columns}")

    if wavelength_column is None:
        lowered = {c.lower(): c for c in columns}
        for candidate in (
            "wavelength_nm",
            "wavelength, nm",
            "wavelength nm",
            "wavelength",
            "lambda_nm",
            "lambda (nm)",
            "lambda",
        ):
            if candidate in lowered:
                wavelength_column = lowered[candidate]
                break
        else:
            raise StudyError(
                f"{path}: cannot identify the wavelength column among {columns}"
            )
    elif wavelength_column not in columns:
        raise StudyError(f"{path}: column {wavelength_column!r} not found")

    wavelengths: list[float] = []
    values: list[float] = []
    for i, row in enumerate(rows, start=2):
        raw_w = row.get(wavelength_column)
        raw_v = row.get(column)
        if raw_w is None or raw_v is None:
            continue
        sw, sv = str(raw_w).strip(), str(raw_v).strip()
        if not sw or not sv:
            continue
        try:
            w = float(sw.replace(",", "."))
            v = float(sv.replace(",", "."))
        except ValueError as exc:
            raise StudyError(
                f"{path} line {i}: non-numeric entry in {wavelength_column!r} "
                f"or {column!r}: {raw_w!r}, {raw_v!r}"
            ) from exc
        if not (np.isfinite(w) and np.isfinite(v)):
            continue
        wavelengths.append(w)
        values.append(v)

    w_arr = np.asarray(wavelengths, dtype=np.float64)
    v_arr = np.asarray(values, dtype=np.float64)
    if w_arr.size == 0:
        raise StudyError(f"{path}: column {column!r} holds no usable point")
    if units == "percent":
        v_arr = v_arr / 100.0
    order = np.argsort(w_arr, kind="mergesort")
    w_arr, v_arr = w_arr[order], v_arr[order]
    if band_nm is not None:
        lo, hi = float(band_nm[0]), float(band_nm[1])
        keep = (w_arr >= lo) & (w_arr <= hi)
        if not np.any(keep):
            raise StudyError(
                f"{path}: column {column!r} has no point inside the requested band "
                f"{lo:g}-{hi:g} nm (data span {w_arr[0]:g}-{w_arr[-1]:g} nm)"
            )
        w_arr, v_arr = w_arr[keep], v_arr[keep]
    if int(stride) > 1:
        w_arr, v_arr = w_arr[:: int(stride)], v_arr[:: int(stride)]
    return w_arr, v_arr


def _load_index(spec: dict | str, base: Path, name: str) -> TabulatedIndex:
    if isinstance(spec, str):
        spec = {"file": spec}
    where = f"index {name!r}"
    file = _require(spec, "file", where)
    valid = spec.get("valid_range_nm")
    return load_index_csv(
        base / file,
        wavelength_column=spec.get("wavelength_column"),
        n_column=spec.get("n_column"),
        k_column=spec.get("k_column"),
        uncertainty_column=spec.get("uncertainty_column"),
        name=name,
        valid_range_nm=(float(valid[0]), float(valid[1])) if valid else None,
    )


def _load_stack(
    name: str,
    spec: dict,
    materials: dict[str, TabulatedIndex],
    lambda0_nm: float,
) -> Stack:
    """Build a stack, accepting layers specified in QWOT or in nanometres.

    A layer given by ``thickness_nm`` is converted to QWOT once, using its own material at
    the study reference wavelength, and the declared thickness is kept as the nominal so
    that reports compare the solution to what was actually specified.
    """
    where = f"stack {name!r}"
    raw_layers = _require(spec, "layers", where)
    layers: list[Layer] = []
    for i, entry in enumerate(raw_layers, start=1):
        thickness = None
        if isinstance(entry, (list, tuple)):
            if len(entry) < 2:
                raise StudyError(f"{where} layer {i}: expected [material, qwot]")
            material, qwot = entry[0], entry[1]
            variable = bool(entry[2]) if len(entry) > 2 else True
        elif isinstance(entry, dict):
            material = _require(entry, "material", f"{where} layer {i}")
            variable = bool(entry.get("variable", True))
            if "qwot" in entry and "thickness_nm" in entry:
                raise StudyError(
                    f"{where} layer {i}: give either qwot or thickness_nm, not both"
                )
            if "thickness_nm" in entry:
                thickness = float(entry["thickness_nm"])
                if str(material) not in materials:
                    raise StudyError(
                        f"{where} layer {i}: material {material!r} must be declared "
                        f"before a thickness can be converted to QWOT"
                    )
                n0 = float(materials[str(material)].n_at(lambda0_nm))
                qwot = 4.0 * n0 * thickness / lambda0_nm
            else:
                qwot = _require(entry, "qwot", f"{where} layer {i}")
        else:
            raise StudyError(f"{where} layer {i}: expected a list or an object")
        layers.append(
            Layer(
                material=str(material),
                qwot=float(qwot),
                variable=variable,
                nominal_thickness_nm=thickness,
            )
        )
    return Stack(
        name=name,
        layers=layers,
        run=spec.get("run"),
        comment=spec.get("comment", ""),
    )


def _load_measurement(spec: dict, base: Path, where: str) -> Measurement:
    file = _require(spec, "file", where)
    column = _require(spec, "column", where)
    band = spec.get("band_nm")
    band_t = (float(band[0]), float(band[1])) if band else None
    stride = int(spec.get("stride", 1))
    wavelength, value = read_spectrum_csv(
        base / file,
        str(column),
        wavelength_column=spec.get("wavelength_column"),
        units=str(spec.get("units", "fraction")),
        band_nm=band_t,
        stride=stride,
    )
    sigma = spec.get("sigma")
    if isinstance(sigma, str):
        sigma_w, sigma_v = read_spectrum_csv(
            base / file,
            sigma,
            units=str(spec.get("units", "fraction")),
            band_nm=band_t,
            stride=stride,
        )
        if sigma_w.size != wavelength.size:
            raise StudyError(
                f"{where}: uncertainty column {sigma!r} has {sigma_w.size} points "
                f"against {wavelength.size} for the measurement"
            )
        sigma = sigma_v
    elif sigma is not None:
        sigma = float(sigma)
    return Measurement(
        quantity=str(_require(spec, "quantity", where)),  # type: ignore[arg-type]
        angle_deg=float(_require(spec, "angle_deg", where)),
        polarization=str(spec.get("polarization", "a")),  # type: ignore[arg-type]
        wavelength_nm=wavelength,
        value=value,
        sigma=sigma,
        band_nm=band_t,
        label=str(spec.get("label") or f"{Path(str(file)).name}:{column}"),
        acquisition=dict(spec.get("acquisition") or {}),
    )


def _load_sample(spec: dict, base: Path) -> Sample:
    name = str(_require(spec, "name", "sample"))
    where = f"sample {name!r}"
    sub_spec = _require(spec, "substrate", where)
    substrate = Substrate(
        material=str(_require(sub_spec, "material", f"{where} substrate")),
        thickness_mm=float(sub_spec.get("thickness_mm", 1.0)),
        rear=str(sub_spec.get("rear", "bare")),  # type: ignore[arg-type]
    )
    measurements = [
        _load_measurement(m, base, f"{where} measurement {i}")
        for i, m in enumerate(spec.get("measurements", []), start=1)
    ]
    return Sample(
        name=name,
        substrate=substrate,
        front_stack=spec.get("front_stack"),
        rear_stack=spec.get("rear_stack"),
        measurements=measurements,
        comment=str(spec.get("comment", "")),
    )


def load_study(path: str | Path, *, strict: bool = True) -> Study:
    """Read a study from its JSON description.

    Parameters
    ----------
    path:
        The JSON file. Every path inside it is resolved relative to its directory.
    strict:
        When true, a study that fails :meth:`~certus_re.model.Study.validate` raises
        :class:`StudyError` listing every problem found. When false, the study is returned
        anyway and the caller is responsible for reporting; useful for tooling that wants to
        show the problems rather than stop.
    """
    path = Path(path)
    base = path.parent
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StudyError(f"{path}: invalid JSON ({exc})") from exc

    materials = {
        name: _load_index(spec, base, name)
        for name, spec in _require(raw, "materials", str(path)).items()
    }
    substrates = {
        name: _load_index(spec, base, name)
        for name, spec in _require(raw, "substrates", str(path)).items()
    }
    lambda0_nm = float(_require(raw, "lambda0_nm", str(path)))
    stacks = {
        name: _load_stack(name, spec, materials, lambda0_nm)
        for name, spec in _require(raw, "stacks", str(path)).items()
    }
    samples = [_load_sample(s, base) for s in _require(raw, "samples", str(path))]

    inst_spec = raw.get("instrument", {})
    per_band = inst_spec.get("aperture_per_band_deg")
    instrument = Instrument(
        name=str(inst_spec.get("name", "unspecified")),
        beam_aperture_deg=float(inst_spec.get("beam_aperture_deg", 2.0)),
        aperture_band_edges_nm=tuple(
            float(x) for x in inst_spec.get("aperture_band_edges_nm", ())
        ),
        aperture_per_band_deg=(
            tuple(float(x) for x in per_band) if per_band is not None else None
        ),
        aperture_mode=str(inst_spec.get("aperture_mode", "imposed")),  # type: ignore[arg-type]
        aperture_bounds_deg=tuple(  # type: ignore[arg-type]
            float(x) for x in inst_spec.get("aperture_bounds_deg", (1.0, 2.5))
        ),
        aperture_min_angle_deg=float(inst_spec.get("aperture_min_angle_deg", 10.0)),
        n_aperture_nodes=int(inst_spec.get("n_aperture_nodes", 2)),
        crosstalk_alpha=float(inst_spec.get("crosstalk_alpha", 0.0)),
        crosstalk_beta=float(inst_spec.get("crosstalk_beta", 0.0)),
        crosstalk_bounds=tuple(  # type: ignore[arg-type]
            float(x) for x in inst_spec.get("crosstalk_bounds", (0.0, 0.15))
        ),
        crosstalk_mode=str(inst_spec.get("crosstalk_mode", "none")),  # type: ignore[arg-type]
        comment=str(inst_spec.get("comment", "")),
    )

    free_spec = raw.get("free_parameters", {})
    free = FreeParameters(
        thicknesses=tuple(free_spec.get("thicknesses", ("*",))),
        thickness_tolerance=(
            dict(free_spec["thickness_tolerance"])
            if isinstance(free_spec.get("thickness_tolerance"), dict)
            else float(free_spec.get("thickness_tolerance", 0.5))
        ),
        index_correction=str(free_spec.get("index_correction", "none")),  # type: ignore[arg-type]
        index_tube_delta=float(free_spec.get("index_tube_delta", 0.0)),
        index_n_knots=int(free_spec.get("index_n_knots", 0)),
        aperture=str(free_spec.get("aperture", instrument.aperture_mode)),  # type: ignore[arg-type]
        crosstalk=str(free_spec.get("crosstalk", instrument.crosstalk_mode)),  # type: ignore[arg-type]
        substrate_index=str(free_spec.get("substrate_index", "literature")),  # type: ignore[arg-type]
        angle_offset=str(free_spec.get("angle_offset", "nominal")),  # type: ignore[arg-type]
    )

    study = Study(
        name=str(raw.get("name", path.stem)),
        lambda0_nm=lambda0_nm,
        materials=materials,
        substrates=substrates,
        stacks=stacks,
        samples=samples,
        instrument=instrument,
        free=free,
        comment=str(raw.get("comment", "")),
    )

    problems = study.validate()
    if problems and strict:
        raise StudyError(
            f"{path}: the study is not consistent:\n  - " + "\n  - ".join(problems)
        )
    return study
