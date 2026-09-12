"""The inversion: least squares on the blocks the study declares, and nothing else.

The parameter vector is assembled from :class:`~certus_re.model.FreeParameters`, block by
block, by the same code that counts them for the report. A block that is not declared has no
entry in the vector, so it cannot move -- the declaration is not documentation of the run, it
*is* the run.

Determinism
-----------
There is no stochastic element: no random restarts, no shakes, no seeded initialisation. The
optimiser is a trust-region least-squares started from the nominal design. Two runs of the
same study on the same data return the same numbers, bit for bit, and the report says so
rather than asking the reader to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from .dof import DoFReport, count_free_parameters
from .forward import Evaluator
from .model import Study

__all__ = ["InversionResult", "SampleResidual", "invert", "ParameterLayout"]


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParameterLayout:
    """Which scalar of the parameter vector is what.

    Built from the study's declaration, so that the vector handed to the optimiser and the
    budget printed in the report are two views of one object.
    """

    study: Study
    thickness_stacks: list[str] = field(default_factory=list)
    thickness_slices: dict[str, slice] = field(default_factory=dict)
    index_materials: list[str] = field(default_factory=list)
    index_slices: dict[str, slice] = field(default_factory=dict)
    index_knots_nm: np.ndarray | None = None
    n_parameters: int = 0

    @classmethod
    def build(cls, study: Study) -> "ParameterLayout":
        layout = cls(study=study)
        cursor = 0
        for name in study.used_stacks():
            if not study.free.releases_thickness(name):
                continue
            count = study.stacks[name].n_variable
            layout.thickness_stacks.append(name)
            layout.thickness_slices[name] = slice(cursor, cursor + count)
            cursor += count

        if study.free.index_correction != "none":
            materials = sorted(
                {m for s in study.used_stacks() for m in study.stacks[s].materials}
            )
            lo, hi = study.wavelength_span_nm()
            layout.index_knots_nm = np.linspace(lo, hi, study.free.index_n_knots)
            for material in materials:
                layout.index_materials.append(material)
                layout.index_slices[material] = slice(
                    cursor, cursor + study.free.index_n_knots
                )
                cursor += study.free.index_n_knots

        layout.n_parameters = cursor
        return layout

    # -- packing ----------------------------------------------------------

    def initial_vector(self, nominal: dict[str, np.ndarray]) -> np.ndarray:
        """Start from the nominal design, with no index correction."""
        x = np.zeros(self.n_parameters, dtype=np.float64)
        for name, sl in self.thickness_slices.items():
            stack = self.study.stacks[name]
            variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
            x[sl] = nominal[name][variable]
        return x

    def bounds(
        self, nominal: dict[str, np.ndarray], thickness_tolerance: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Box constraints: a relative window on thicknesses, a tube on the index.

        The thickness window exists to keep the search in the interference order the design
        was made in; a solution that reaches it is reported rather than quietly accepted.
        """
        lower = np.full(self.n_parameters, -np.inf)
        upper = np.full(self.n_parameters, np.inf)
        for name, sl in self.thickness_slices.items():
            stack = self.study.stacks[name]
            variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
            d0 = nominal[name][variable]
            lower[sl] = np.maximum(d0 * (1.0 - thickness_tolerance), 1e-3)
            upper[sl] = d0 * (1.0 + thickness_tolerance)
        delta = self.study.free.index_tube_delta
        for _, sl in self.index_slices.items():
            lower[sl] = -delta
            upper[sl] = +delta
        return lower, upper

    def unpack(
        self, x: np.ndarray, nominal: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Turn a parameter vector into thicknesses and index-knot corrections."""
        thicknesses = {name: np.array(value, dtype=np.float64) for name, value in nominal.items()}
        for name, sl in self.thickness_slices.items():
            stack = self.study.stacks[name]
            variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
            thicknesses[name][variable] = x[sl]
        corrections = {
            material: np.asarray(x[sl], dtype=np.float64)
            for material, sl in self.index_slices.items()
        }
        return thicknesses, corrections


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SampleResidual:
    """How well one measurement is reproduced, before and after the inversion."""

    sample: str
    label: str
    quantity: str
    n_points: int
    rms_initial: float
    rms_final: float
    bias_final: float
    max_abs_final: float
    chi2_per_point: float

    def to_dict(self) -> dict:
        return {
            "sample": self.sample,
            "measurement": self.label,
            "quantity": self.quantity,
            "n_points": self.n_points,
            "rms_initial": self.rms_initial,
            "rms_final": self.rms_final,
            "bias_final": self.bias_final,
            "max_abs_final": self.max_abs_final,
            "chi2_per_point": self.chi2_per_point,
        }


@dataclass(slots=True)
class InversionResult:
    """Everything a run produces, in the units a coating engineer reads."""

    study_name: str
    dof: DoFReport
    thicknesses: dict[str, np.ndarray]
    nominal_thicknesses: dict[str, np.ndarray]
    qwot: dict[str, np.ndarray]
    nominal_qwot: dict[str, np.ndarray]
    residuals: list[SampleResidual]
    index_corrections: dict[str, np.ndarray]
    rms_initial: float
    rms_final: float
    chi2_per_point: float
    n_function_evaluations: int
    optimality: float
    success: bool
    message: str
    at_bounds: list[str] = field(default_factory=list)
    extrapolation: list[dict] = field(default_factory=list)

    def qwot_departure_pct(self) -> dict[str, np.ndarray]:
        """Departure from the nominal design, per layer, in percent of quarter waves."""
        return {
            name: 100.0 * (self.qwot[name] - self.nominal_qwot[name]) / self.nominal_qwot[name]
            for name in self.qwot
        }

    def thickness_departure_nm(self) -> dict[str, np.ndarray]:
        return {
            name: self.thicknesses[name] - self.nominal_thicknesses[name]
            for name in self.thicknesses
        }


# ---------------------------------------------------------------------------
# Index correction basis
# ---------------------------------------------------------------------------


def _spline_basis(knots_nm: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Matrix mapping knot values to a natural cubic spline sampled on ``wavelengths``.

    A natural cubic spline is a *linear* function of its knot values, so the mapping is a
    matrix that can be built once and reused at every iteration. That is the whole reason
    for preferring it to a shape-preserving interpolant here.
    """
    n_knots = knots_nm.size
    basis = np.zeros((wavelengths.size, n_knots), dtype=np.float64)
    for j in range(n_knots):
        unit = np.zeros(n_knots)
        unit[j] = 1.0
        basis[:, j] = CubicSpline(knots_nm, unit, bc_type="natural")(wavelengths)
    return basis


# ---------------------------------------------------------------------------
# The inversion
# ---------------------------------------------------------------------------


def invert(
    study: Study,
    *,
    thickness_tolerance: float = 0.5,
    max_iterations: int = 200,
    verbose: bool = False,
    legacy_gain_sign: bool = False,
) -> InversionResult:
    """Invert a study and return everything needed to report it.

    Parameters
    ----------
    study:
        The problem, already validated.
    thickness_tolerance:
        Half-width of the box constraint on each thickness, as a fraction of its nominal
        value. The default of 0.5 keeps the search inside the interference order of the
        design while leaving ample room; a layer that ends on its bound is listed in
        :attr:`InversionResult.at_bounds`.
    max_iterations:
        Passed to the optimiser as its function-evaluation budget per parameter.
    legacy_gain_sign:
        Reproduce the sign convention of the reference implementation, for comparison only.
    """
    problems = study.validate()
    if problems:
        raise ValueError(
            "the study is not consistent:\n  - " + "\n  - ".join(problems)
        )

    evaluator = Evaluator(study, legacy_gain_sign=legacy_gain_sign)
    layout = ParameterLayout.build(study)
    nominal = evaluator.nominal_thicknesses()

    # Precompute, for every measurement and every corrected material, the matrix that maps
    # the correction knots onto that measurement's wavelength grid.
    bases: list[dict[str, np.ndarray]] = []
    if layout.index_knots_nm is not None:
        for plan in evaluator.plans:
            bases.append(
                {
                    material: _spline_basis(
                        layout.index_knots_nm, plan.measurement.wavelength_nm
                    )
                    for material in layout.index_materials
                }
            )

    measured = np.concatenate([p.measurement.value for p in evaluator.plans])
    sigma = np.concatenate([p.measurement.sigma_vector() for p in evaluator.plans])

    base_front = [plan.n_front.copy() for plan in evaluator.plans]
    base_rear = [
        None if plan.n_rear is None else plan.n_rear.copy() for plan in evaluator.plans
    ]

    def apply_corrections(corrections: dict[str, np.ndarray]) -> None:
        """Write the corrected indices into the plans, in place."""
        for i, plan in enumerate(evaluator.plans):
            for which, base in (("front", base_front[i]), ("rear", base_rear[i])):
                if base is None:
                    continue
                stack_name = plan.front_stack if which == "front" else plan.rear_stack
                if stack_name is None:
                    continue
                materials = study.stacks[stack_name].materials
                corrected = base.copy()
                for column, material in enumerate(materials):
                    if material not in corrections:
                        continue
                    delta = bases[i][material] @ corrections[material]
                    corrected[:, column] = corrected[:, column] + delta
                if which == "front":
                    plan.n_front = corrected
                else:
                    plan.n_rear = corrected

    def residual(x: np.ndarray) -> np.ndarray:
        thicknesses, corrections = layout.unpack(x, nominal)
        if corrections:
            apply_corrections(corrections)
        predicted = np.concatenate(evaluator.predict(thicknesses))
        return (predicted - measured) / sigma

    x0 = layout.initial_vector(nominal)
    lower, upper = layout.bounds(nominal, thickness_tolerance)
    residual_initial = residual(x0)

    if layout.n_parameters == 0:
        solution = None
        x_final = x0
    else:
        solution = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max_iterations * max(1, layout.n_parameters),
            verbose=2 if verbose else 0,
        )
        x_final = solution.x

    thicknesses, corrections = layout.unpack(x_final, nominal)
    if corrections:
        apply_corrections(corrections)
    predicted = evaluator.predict(thicknesses)
    residual_final = (np.concatenate(predicted) - measured) / sigma

    # Which parameters ended on a bound: a silent bound hit is a wrong answer that looks
    # like a converged one.
    at_bounds: list[str] = []
    for name, sl in layout.thickness_slices.items():
        stack = study.stacks[name]
        variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
        for k, layer_index in enumerate(variable):
            value = x_final[sl][k]
            if value <= lower[sl][k] * (1 + 1e-6) or value >= upper[sl][k] * (1 - 1e-6):
                at_bounds.append(f"{name} layer {layer_index + 1}")

    # Per-measurement residuals, in the natural units of each measurement. The starting
    # point is recomputed rather than cached, so that "before" and "after" are produced by
    # the same code path and cannot drift apart.
    residuals: list[SampleResidual] = []
    thicknesses_initial, corrections_initial = layout.unpack(x0, nominal)
    if corrections_initial:
        apply_corrections(corrections_initial)
    predicted_initial = evaluator.predict(thicknesses_initial)
    if corrections:
        apply_corrections(corrections)

    for plan, before, after in zip(evaluator.plans, predicted_initial, predicted):
        measurement = plan.measurement
        r0 = before - measurement.value
        r1 = after - measurement.value
        s = measurement.sigma_vector()
        residuals.append(
            SampleResidual(
                sample=plan.sample.name,
                label=measurement.label or measurement.quantity,
                quantity=measurement.quantity,
                n_points=measurement.n_points,
                rms_initial=float(np.sqrt(np.mean(r0**2))),
                rms_final=float(np.sqrt(np.mean(r1**2))),
                bias_final=float(np.mean(r1)),
                max_abs_final=float(np.max(np.abs(r1))),
                chi2_per_point=float(np.mean((r1 / s) ** 2)),
            )
        )

    extrapolation = [
        report
        for index in list(study.materials.values()) + list(study.substrates.values())
        if (report := index.extrapolation_report()) is not None  # type: ignore[attr-defined]
    ]

    return InversionResult(
        study_name=study.name,
        dof=count_free_parameters(study),
        thicknesses=thicknesses,
        nominal_thicknesses=nominal,
        qwot=evaluator.qwot(thicknesses),
        nominal_qwot=evaluator.qwot(nominal),
        residuals=residuals,
        index_corrections=corrections,
        rms_initial=float(np.sqrt(np.mean(residual_initial**2))),
        rms_final=float(np.sqrt(np.mean(residual_final**2))),
        chi2_per_point=float(np.mean(residual_final**2)),
        n_function_evaluations=0 if solution is None else int(solution.nfev),
        optimality=0.0 if solution is None else float(solution.optimality),
        success=True if solution is None else bool(solution.success),
        message="no free parameter" if solution is None else str(solution.message),
        at_bounds=at_bounds,
        extrapolation=extrapolation,
    )
