"""Tabulated complex refractive indices, with provenance and honest extrapolation.

A dispersion dataset is not a curve: it is a measurement, made on a given machine at a given
time, valid over a given range. This module keeps that context attached to the numbers, and
refuses to lose it silently.

Two behaviours are deliberate.

* **Interpolation is linear, with constant extrapolation at the boundaries.** This matches
  the reference implementation exactly, which is what makes the non-regression test against
  it meaningful.
* **Extrapolation is recorded.** Clamping to the first or last tabulated value is a
  defensible default and a dangerous one: beyond the last interference extremum the index of
  a single-layer determination is no longer constrained by the fringe positions, so a study
  that reaches past the table is leaning on a value the measurement never established.
  :meth:`TabulatedIndex.extrapolation_report` states how far, and on how many points.

Sign convention
---------------
``n_complex = n + i k`` with ``k >= 0`` for an absorbing medium (Macleod convention), which
is the convention of the transfer-matrix kernels used downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "TabulatedIndex",
    "load_index_csv",
    "li1980_silicon_n",
    "sellmeier_n",
    "silicon_li1980",
    "sapphire_malitson",
    "LI1980_VALID_RANGE_NM",
    "LI1980_POLE_NM",
    "SAPPHIRE_MALITSON_SELLMEIER",
]


@dataclass(slots=True)
class TabulatedIndex:
    """A complex refractive index sampled on a wavelength grid.

    Parameters
    ----------
    wavelength_nm:
        Strictly increasing grid, in nanometres.
    n:
        Real part of the refractive index.
    k:
        Extinction coefficient, non-negative. Defaults to zero.
    name:
        Material identifier, used in reports.
    source:
        Where the numbers come from: file, article, deposition run. Printed in the report so
        that a published result names its own index dataset.
    uncertainty_n:
        Optional one-sigma envelope on ``n``, per grid point. When available it bounds the
        tube of a bounded-excursion inversion instead of an arbitrary constant.
    valid_range_nm:
        Optional ``(lambda_min, lambda_max)`` over which the determination is considered
        constrained. Distinct from the tabulated span: a table may extend beyond the range
        the measurement actually constrained.
    """

    wavelength_nm: np.ndarray
    n: np.ndarray
    k: np.ndarray | None = None
    name: str = ""
    source: str = ""
    uncertainty_n: np.ndarray | None = None
    valid_range_nm: tuple[float, float] | None = None
    # Aggregate of what this dataset has been asked for, kept as three scalars rather than
    # a growing list so that a long run cannot accumulate memory in a diagnostic.
    _query_lo: float = field(default=np.inf, repr=False)
    _query_hi: float = field(default=-np.inf, repr=False)
    _query_outside: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=np.float64).ravel()
        self.n = np.asarray(self.n, dtype=np.float64).ravel()
        if self.wavelength_nm.size != self.n.size:
            raise ValueError(
                f"{self.name or 'index'}: {self.wavelength_nm.size} wavelengths for "
                f"{self.n.size} values of n"
            )
        if self.wavelength_nm.size < 2:
            raise ValueError(f"{self.name or 'index'}: need at least two grid points")
        if np.any(np.diff(self.wavelength_nm) <= 0.0):
            order = np.argsort(self.wavelength_nm, kind="mergesort")
            self.wavelength_nm = self.wavelength_nm[order]
            self.n = self.n[order]
            if self.k is not None:
                self.k = np.asarray(self.k, dtype=np.float64).ravel()[order]
            if self.uncertainty_n is not None:
                self.uncertainty_n = np.asarray(
                    self.uncertainty_n, dtype=np.float64
                ).ravel()[order]
        if self.k is None:
            self.k = np.zeros_like(self.n)
        else:
            self.k = np.asarray(self.k, dtype=np.float64).ravel()
            if self.k.size != self.n.size:
                raise ValueError(
                    f"{self.name or 'index'}: {self.k.size} values of k for "
                    f"{self.n.size} values of n"
                )
        if np.any(self.k < 0.0):
            raise ValueError(
                f"{self.name or 'index'}: negative k -- the convention here is n + ik "
                f"with k >= 0 for an absorbing medium"
            )
        if self.uncertainty_n is not None:
            self.uncertainty_n = np.asarray(self.uncertainty_n, dtype=np.float64).ravel()
            if self.uncertainty_n.size != self.n.size:
                raise ValueError(
                    f"{self.name or 'index'}: uncertainty has "
                    f"{self.uncertainty_n.size} values for {self.n.size} points"
                )

    # -- span -------------------------------------------------------------

    @property
    def span_nm(self) -> tuple[float, float]:
        """Range actually tabulated."""
        return float(self.wavelength_nm[0]), float(self.wavelength_nm[-1])

    # -- evaluation -------------------------------------------------------

    def n_at(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        """Real part of the index, by linear interpolation, clamped outside the table."""
        w = np.asarray(wavelength_nm, dtype=np.float64)
        return np.interp(w, self.wavelength_nm, self.n, left=self.n[0], right=self.n[-1])

    def k_at(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        """Extinction coefficient, same interpolation."""
        w = np.asarray(wavelength_nm, dtype=np.float64)
        return np.interp(w, self.wavelength_nm, self.k, left=self.k[0], right=self.k[-1])

    def complex_at(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        """``n + ik`` on the requested wavelengths, as complex128.

        The query is recorded so that :meth:`extrapolation_report` can tell, after a run,
        whether the study leaned on values outside the tabulated or the validated range.
        """
        w = np.atleast_1d(np.asarray(wavelength_nm, dtype=np.float64))
        if w.size:
            lo, hi = self.span_nm
            self._query_outside += int(np.count_nonzero((w < lo) | (w > hi)))
            self._query_lo = min(self._query_lo, float(w.min()))
            self._query_hi = max(self._query_hi, float(w.max()))
        n_i = self.n_at(w)
        k_i = self.k_at(w)
        return (n_i + 1j * k_i).astype(np.complex128)

    def uncertainty_at(self, wavelength_nm: float | np.ndarray) -> np.ndarray | None:
        """One-sigma envelope on ``n``, or ``None`` when the dataset carries none."""
        if self.uncertainty_n is None:
            return None
        w = np.asarray(wavelength_nm, dtype=np.float64)
        return np.interp(
            w,
            self.wavelength_nm,
            self.uncertainty_n,
            left=self.uncertainty_n[0],
            right=self.uncertainty_n[-1],
        )

    # -- provenance -------------------------------------------------------

    def extrapolation_report(self) -> dict | None:
        """What this dataset was asked for outside the range it covers.

        Returns ``None`` when every query fell inside the tabulated span *and* inside the
        validated range. Otherwise a dictionary stating how far outside, and on how many
        points -- to be printed in the run report rather than discovered later.
        """
        if not np.isfinite(self._query_lo):
            return None
        q_lo, q_hi, n_out = self._query_lo, self._query_hi, self._query_outside
        lo, hi = self.span_nm
        beyond_valid = None
        if self.valid_range_nm is not None:
            v_lo, v_hi = self.valid_range_nm
            if q_lo < v_lo or q_hi > v_hi:
                beyond_valid = {
                    "validated_range_nm": [v_lo, v_hi],
                    "queried_range_nm": [q_lo, q_hi],
                }
        if n_out == 0 and beyond_valid is None:
            return None
        return {
            "material": self.name,
            "tabulated_range_nm": [lo, hi],
            "queried_range_nm": [q_lo, q_hi],
            "points_outside_table": n_out,
            "beyond_validated_range": beyond_valid,
        }

    def describe(self) -> str:
        lo, hi = self.span_nm
        bits = [f"{self.name or 'index'}: {self.n.size} points, {lo:.1f}-{hi:.1f} nm"]
        if self.valid_range_nm is not None:
            bits.append(
                f"validated {self.valid_range_nm[0]:.0f}-{self.valid_range_nm[1]:.0f} nm"
            )
        if np.any(self.k > 0.0):
            bits.append(f"k up to {float(self.k.max()):.2e}")
        else:
            bits.append("non-absorbing")
        if self.source:
            bits.append(self.source)
        return " | ".join(bits)


def load_index_csv(
    path: str | Path,
    *,
    wavelength_column: str | None = None,
    n_column: str | None = None,
    k_column: str | None = None,
    uncertainty_column: str | None = None,
    name: str = "",
    valid_range_nm: tuple[float, float] | None = None,
) -> TabulatedIndex:
    """Read a tabulated index from a CSV file with a header row.

    Columns are selected **by name**, never by position. That is not pedantry: two workbooks
    of the same campaign were found to order their index columns differently, and
    substituting by position silently produced a forty-percent residual. When a column name
    is not given, it is guessed from a small set of conventional spellings, and the guess is
    recorded in :attr:`TabulatedIndex.source`.

    Parameters
    ----------
    path:
        CSV file, comma-separated, first row a header.
    wavelength_column, n_column, k_column, uncertainty_column:
        Column names. ``k`` and the uncertainty are optional.
    name:
        Material identifier. Defaults to the file stem.
    valid_range_nm:
        Range over which the determination is considered constrained, if narrower than the
        tabulated span.
    """
    import csv

    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        # Lines opening with '#' carry the provenance note of a deposited file and are not
        # data; dropping them here is what lets a CSV document its own origin.
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"{path}: no header row")
    columns = [c.strip() for c in reader.fieldnames]
    rows = [{k.strip() if k else k: v for k, v in row.items()} for row in reader]

    def pick(explicit: str | None, candidates: tuple[str, ...], what: str) -> str:
        if explicit is not None:
            if explicit not in columns:
                raise ValueError(
                    f"{path}: column {explicit!r} not found; available: {columns}"
                )
            return explicit
        lowered = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        raise ValueError(
            f"{path}: cannot identify the {what} column among {columns}; "
            f"pass it explicitly"
        )

    w_col = pick(
        wavelength_column,
        ("wavelength_nm", "wavelength, nm", "wavelength", "lambda_nm", "lambda (nm)", "lambda"),
        "wavelength",
    )
    n_col = pick(n_column, ("n", "n_real", "index"), "refractive index")
    k_col = None
    if k_column is not None or "k" in {c.lower() for c in columns}:
        k_col = pick(k_column, ("k", "kappa", "extinction"), "extinction coefficient")
    u_col = None
    if uncertainty_column is not None:
        u_col = pick(uncertainty_column, (), "uncertainty")

    def column(col: str) -> np.ndarray:
        """Read one column, row-aligned, with blanks as NaN.

        Alignment matters: an optional column such as the uncertainty envelope may be blank
        over part of the range, and dropping its blanks instead of marking them would
        silently shift every subsequent value onto the wrong wavelength.
        """
        out = []
        for i, row in enumerate(rows, start=2):
            raw = row.get(col)
            if raw is None or str(raw).strip() == "":
                out.append(np.nan)
                continue
            try:
                out.append(float(str(raw).replace(",", ".")))
            except ValueError as exc:
                raise ValueError(
                    f"{path} line {i}: column {col!r} is not a number: {raw!r}"
                ) from exc
        return np.asarray(out, dtype=np.float64)

    wavelengths = column(w_col)
    n_values = column(n_col)
    k_values = column(k_col) if k_col else None
    u_values = column(u_col) if u_col else None

    # A row without a wavelength or without n is not a data point.
    keep = np.isfinite(wavelengths) & np.isfinite(n_values)
    if not np.any(keep):
        raise ValueError(f"{path}: no usable row in columns {w_col!r} and {n_col!r}")
    wavelengths = wavelengths[keep]
    n_values = n_values[keep]
    if k_values is not None:
        k_values = np.nan_to_num(k_values[keep], nan=0.0)
    if u_values is not None:
        u_values = u_values[keep]

    guessed = [
        f"{w_col} -> wavelength",
        f"{n_col} -> n",
        *([f"{k_col} -> k"] if k_col else []),
    ]
    return TabulatedIndex(
        wavelength_nm=wavelengths,
        n=n_values,
        k=k_values,
        name=name or path.stem,
        source=f"{path.name} ({'; '.join(guessed)})",
        uncertainty_n=u_values,
        valid_range_nm=valid_range_nm,
    )


# ---------------------------------------------------------------------------
# Dispersions given by a formula
# ---------------------------------------------------------------------------
#
# The two substrates of this campaign are literature materials, and a literature material is
# better carried as its formula than as somebody's transcription of it. A table whose name
# announces a reference is exactly the kind of file that goes wrong silently: the one
# distributed with an earlier revision of this deposit was flat at 3.5545 over the whole
# band, which is not Li's dispersion at all, and being optically in series with every layer
# it displaced every retrieved thickness. The formulas are therefore evaluated here, checked
# against their published control values by the test suite, and tabulated into the deposit by
# ``tools/build_study_volet2.py`` so that the archived CSV can be read without running
# anything.

# H. H. Li, "Refractive index of silicon and germanium and its wavelength and temperature
# derivatives", J. Phys. Chem. Ref. Data 9, 561 (1980). Crystalline silicon at 26 C.
LI1980_POLE_NM = 1107.1
LI1980_VALID_RANGE_NM = (1200.0, 14000.0)

# Sapphire, ordinary ray. I. H. Malitson, J. Opt. Soc. Am. 52, 1377 (1962).
# Sellmeier form n^2 - 1 = sum_i B_i lambda^2 / (lambda^2 - C_i), lambda in micrometres.
SAPPHIRE_MALITSON_SELLMEIER = (
    (1.4313493, 0.00527992877161),
    (0.65054713, 0.01423826448164),
    (5.3414021, 325.0188308009999),
)


def li1980_silicon_n(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Refractive index of crystalline silicon from Li's 1980 dispersion formula.

    With the wavelength in micrometres,

    .. math::

        n^2 = 11.6858 + \\frac{0.939816}{\\lambda^2}
              + \\frac{8.10461\\times10^{-3}\\,\\lambda_0^2}{\\lambda^2 - \\lambda_0^2},
        \\qquad \\lambda_0 = 1.1071\\ \\mu\\mathrm{m} .

    Control values: ``n(2 um) = 3.4532``, ``n(3 um) = 3.4339``, ``n(4 um) = 3.4271``, against
    3.4522, 3.4320 and 3.4255 in the literature.

    Raises
    ------
    ValueError
        Outside :data:`LI1980_VALID_RANGE_NM`. The model is published for 1.2-14 um and has a
        pole at :data:`LI1980_POLE_NM`; between 1050 and 1200 nm it returns values that are
        not monotonic and have no physical meaning, and silicon starts absorbing near 1.1 um
        in any case. Refusing is the point: an index formula evaluated just on the wrong side
        of its pole returns a number, not an error, and nothing downstream would notice.
    """
    w = np.asarray(wavelength_nm, dtype=np.float64)
    lo, hi = LI1980_VALID_RANGE_NM
    if w.size and (np.min(w) < lo - 1e-9 or np.max(w) > hi + 1e-9):
        raise ValueError(
            f"Li 1980 is published for {lo:.0f}-{hi:.0f} nm and has a pole at "
            f"{LI1980_POLE_NM:.1f} nm; asked for "
            f"{float(np.min(w)):.1f}-{float(np.max(w)):.1f} nm"
        )
    lam2 = (w / 1000.0) ** 2
    pole2 = (LI1980_POLE_NM / 1000.0) ** 2
    n2 = 11.6858 + 0.939816 / lam2 + 8.10461e-3 * pole2 / (lam2 - pole2)
    return np.sqrt(n2)


def sellmeier_n(wavelength_nm: float | np.ndarray, terms) -> np.ndarray:
    """Refractive index from Sellmeier coefficients, wavelength given in nanometres.

    ``terms`` is a sequence of ``(B_i, C_i)`` pairs with ``C_i`` in squared micrometres.
    """
    lam2 = (np.asarray(wavelength_nm, dtype=np.float64) / 1000.0) ** 2
    n2 = np.ones_like(lam2)
    for b, c in terms:
        n2 = n2 + b * lam2 / (lam2 - c)
    return np.sqrt(n2)


def silicon_li1980(
    wavelength_nm: np.ndarray | None = None,
    *,
    valid_range_nm: tuple[float, float] = (1200.0, 6000.0),
) -> TabulatedIndex:
    """Silicon, sampled from :func:`li1980_silicon_n`.

    ``k = 0``: silicon is transparent from 1.2 to 6 um, and the campaign never queries it
    outside that. The default grid starts at 1200 nm, which is what forces the low edge of
    the inversion window -- 1000 nm would fall on the wrong side of the pole.
    """
    lo, hi = valid_range_nm
    grid = np.arange(lo, hi + 1e-9, 5.0) if wavelength_nm is None else np.asarray(
        wavelength_nm, dtype=np.float64
    )
    return TabulatedIndex(
        wavelength_nm=grid,
        n=li1980_silicon_n(grid),
        k=np.zeros(grid.size),
        name="silicon",
        source="H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980), evaluated by certus_re",
        valid_range_nm=(float(lo), float(hi)),
    )


def sapphire_malitson(wavelength_nm: np.ndarray | None = None) -> TabulatedIndex:
    """Sapphire, ordinary ray, sampled from Malitson's Sellmeier coefficients."""
    grid = (
        np.arange(250.0, 5201.0, 10.0)
        if wavelength_nm is None
        else np.asarray(wavelength_nm, dtype=np.float64)
    )
    return TabulatedIndex(
        wavelength_nm=grid,
        n=sellmeier_n(grid, SAPPHIRE_MALITSON_SELLMEIER),
        k=np.zeros(grid.size),
        name="sapphire",
        source="I. H. Malitson, J. Opt. Soc. Am. 52, 1377 (1962), evaluated by certus_re",
    )
