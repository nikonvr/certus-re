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

__all__ = ["TabulatedIndex", "load_index_csv"]


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
