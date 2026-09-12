"""Explicit accounting of the parameters an inversion is allowed to move.

Reverse-engineering results are routinely published as a residual and a set of retrieved
thicknesses, without stating how many quantities were released to obtain them. The residual
alone says nothing: enough free parameters will fit anything, and the index correction that
silently improves a fit is the very quantity one usually claims to be validating.

This module turns the declaration in :class:`~certus_re.model.FreeParameters` into a
countable budget, block by block, and derives the two numbers that make a result readable:

* the **number of free parameters**, which the run report prints and which the article's
  parameter table is generated from;
* the **number of independent data points per free parameter**, which is what actually
  limits an inversion -- far more than the layer count does.

The count is computed from the declaration *before* the solver starts, and the solver builds
its parameter vector from the same blocks, so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Study

__all__ = ["ParameterBlock", "DoFReport", "count_free_parameters"]


@dataclass(frozen=True, slots=True)
class ParameterBlock:
    """One named group of parameters, released or not.

    Attributes
    ----------
    name:
        Short label used in the report and in the article's parameter table.
    count:
        Number of scalar parameters released by this block. Zero means the block is held.
    status:
        Why the block has the count it has, in words: ``"released"``, ``"imposed"``,
        ``"literature"``, ``"nominal"``, ``"tabulated"``. Printed next to the count so a
        zero is never ambiguous between "not used" and "held fixed on purpose".
    detail:
        Optional expansion, e.g. the per-stack breakdown of released thicknesses.
    """

    name: str
    count: int
    status: str
    detail: str = ""


@dataclass(slots=True)
class DoFReport:
    """The parameter budget of one inversion."""

    blocks: list[ParameterBlock] = field(default_factory=list)
    n_points: int = 0

    @property
    def n_free(self) -> int:
        return sum(b.count for b in self.blocks)

    @property
    def points_per_parameter(self) -> float:
        """Independent data points per released parameter.

        Returns infinity when nothing is released, which is the honest answer for a purely
        forward evaluation.
        """
        if self.n_free == 0:
            return float("inf")
        return self.n_points / self.n_free

    def released(self) -> list[ParameterBlock]:
        return [b for b in self.blocks if b.count > 0]

    def held(self) -> list[ParameterBlock]:
        return [b for b in self.blocks if b.count == 0]

    def to_dict(self) -> dict:
        return {
            "blocks": [
                {
                    "name": b.name,
                    "count": b.count,
                    "status": b.status,
                    "detail": b.detail,
                }
                for b in self.blocks
            ],
            "n_free_parameters": self.n_free,
            "n_data_points": self.n_points,
            "points_per_free_parameter": (
                None
                if self.points_per_parameter == float("inf")
                else round(self.points_per_parameter, 2)
            ),
        }

    def to_text(self) -> str:
        """A table fit to be read in a terminal and pasted into a manuscript."""
        width = max((len(b.name) for b in self.blocks), default=10)
        lines = [f"{'parameter block'.ljust(width)}  count  status"]
        lines.append("-" * (width + 22))
        for b in self.blocks:
            lines.append(f"{b.name.ljust(width)}  {b.count:5d}  {b.status}")
            if b.detail:
                lines.append(f"{' ' * width}         {b.detail}")
        lines.append("-" * (width + 22))
        lines.append(f"{'total free parameters'.ljust(width)}  {self.n_free:5d}")
        lines.append(f"{'data points'.ljust(width)}  {self.n_points:5d}")
        if self.n_free:
            lines.append(
                f"{'points per parameter'.ljust(width)}  "
                f"{self.points_per_parameter:5.1f}"
            )
        return "\n".join(lines)


def count_free_parameters(study: Study) -> DoFReport:
    """Derive the parameter budget of ``study`` from its declaration.

    Every block appears in the report, including the held ones, so that a reader can see
    what was *not* released as plainly as what was.
    """
    report = DoFReport(n_points=study.n_points)

    # -- layer thicknesses ------------------------------------------------
    per_stack: list[str] = []
    n_thickness = 0
    for name in study.used_stacks():
        stack = study.stacks[name]
        if study.free.releases_thickness(name):
            n_thickness += stack.n_variable
            per_stack.append(f"{name}: {stack.n_variable}")
        else:
            per_stack.append(f"{name}: 0 (held)")
    released_stacks = [
        name for name in study.used_stacks() if study.free.releases_thickness(name)
    ]
    detail = ", ".join(per_stack)
    if n_thickness:
        # The search window is prior information and belongs beside the count: two runs
        # releasing the same number of thicknesses in windows of 5 and 50 percent are not
        # the same run, and on an ill-posed problem they do not return the same coating.
        detail += "; " + study.free.tolerances_text(released_stacks)
    report.blocks.append(
        ParameterBlock(
            name="layer thicknesses",
            count=n_thickness,
            status="released" if n_thickness else "held",
            detail=detail,
        )
    )

    # -- index correction -------------------------------------------------
    free = study.free
    if free.index_correction == "none":
        report.blocks.append(
            ParameterBlock(
                "index correction",
                0,
                "tabulated",
                "n and k used as determined on the witness samples, not adjusted",
            )
        )
    else:
        n_materials = len({m for s in study.used_stacks() for m in study.stacks[s].materials})
        count = n_materials * free.index_n_knots
        status = (
            f"released, tube |n - n_tab| <= {free.index_tube_delta:g}"
            if free.index_correction == "bounded"
            else "released, unconstrained"
        )
        report.blocks.append(
            ParameterBlock(
                "index correction",
                count,
                status,
                f"{free.index_n_knots} spline knots on Re(n) for each of "
                f"{n_materials} materials",
            )
        )

    # -- beam aperture ----------------------------------------------------
    inst = study.instrument
    edges = (
        ", ".join(f"{e:g}" for e in inst.aperture_band_edges_nm)
        if inst.aperture_band_edges_nm
        else "none"
    )
    if free.aperture == "imposed":
        values = ", ".join(f"{v:.2f}" for v in inst.aperture_values_deg())
        report.blocks.append(
            ParameterBlock(
                "beam aperture",
                0,
                "imposed",
                f"total aperture {values} deg; step wavelengths (nm): {edges}",
            )
        )
    else:
        lo, hi = inst.aperture_bounds_deg
        report.blocks.append(
            ParameterBlock(
                "beam aperture",
                inst.n_bands,
                f"released within {lo:g}-{hi:g} deg",
                f"one total aperture per band, {inst.n_bands} bands; step wavelengths "
                f"(nm): {edges} -- imposed, not fitted; applied only at incidence >= "
                f"{inst.aperture_min_angle_deg:g} deg",
            )
        )

    # -- polarizer crosstalk ----------------------------------------------
    n_polarized = study.n_polarized_points
    if free.crosstalk == "none":
        report.blocks.append(
            ParameterBlock(
                "polarizer crosstalk",
                0,
                "not modelled",
                "the polarizer is taken as perfect; alpha = beta = 0",
            )
        )
    else:
        lo, hi = inst.crosstalk_bounds
        report.blocks.append(
            ParameterBlock(
                "polarizer crosstalk",
                2,
                f"released within {lo:g}-{hi:g}",
                f"alpha and beta, one pair for the whole study, shared by every "
                f"polarization-resolved measurement; {n_polarized} polarized points "
                f"constrain them",
            )
        )

    # -- substrate --------------------------------------------------------
    if free.substrate_index == "literature":
        subs = ", ".join(sorted({s.substrate.material for s in study.samples}))
        report.blocks.append(
            ParameterBlock(
                "substrate index",
                0,
                "literature",
                f"propagated, not adjusted ({subs})",
            )
        )
    else:
        n_sub = len({s.substrate.material for s in study.samples})
        report.blocks.append(
            ParameterBlock(
                "substrate index",
                3 * n_sub,
                "released, Cauchy",
                f"three Cauchy coefficients for each of {n_sub} substrates",
            )
        )

    # -- angle of incidence -----------------------------------------------
    angles = sorted(
        {
            round(float(m.angle_deg), 6)
            for s in study.samples
            for m in s.measurements
        }
    )
    if free.angle_offset == "nominal":
        report.blocks.append(
            ParameterBlock(
                "angle of incidence",
                0,
                "nominal",
                "angles used as set: " + ", ".join(f"{a:g}" for a in angles) + " deg",
            )
        )
    else:
        report.blocks.append(
            ParameterBlock(
                "angle of incidence",
                len(angles),
                "released",
                "one offset per distinct angle; note that at 45 deg the retrieved "
                "thickness moves by about 35 nm per degree",
            )
        )

    return report
