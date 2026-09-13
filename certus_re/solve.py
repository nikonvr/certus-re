"""The inversion: least squares on the blocks the study declares, and nothing else.

The parameter vector is assembled from :class:`~certus_re.model.FreeParameters`, block by
block, by the same code that counts them for the report. A block that is not declared has no
entry in the vector, so it cannot move -- the declaration is not documentation of the run, it
*is* the run.

What a run returns is not only a set of thicknesses but what the data actually determine
about them. The covariance of the solution is read off the Jacobian the optimiser already
computed, ``sigma^2 (J^T J)^{-1}``, and converted from nanometres to quarter waves layer by
layer. A retrieved thickness without its uncertainty cannot be compared to a design tolerance,
and a departure smaller than its own error bar is not a departure.

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

__all__ = [
    "InversionResult",
    "SampleResidual",
    "ParameterValues",
    "invert",
    "ParameterLayout",
]


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParameterValues:
    """One point of the parameter space, in the units each block is written in."""

    thicknesses: dict[str, np.ndarray]
    index_corrections: dict[str, np.ndarray]
    aperture_deg: np.ndarray
    crosstalk: tuple[float, float]


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
    aperture_slice: slice | None = None
    crosstalk_slice: slice | None = None
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

        # One aperture per band. The band edges are the documented switchovers of the
        # instrument and are never released: what is uncertain is how wide the cone is, not
        # where the instrument changes configuration.
        if study.free.aperture == "fitted":
            count = study.instrument.n_bands
            layout.aperture_slice = slice(cursor, cursor + count)
            cursor += count

        # One pair for the whole study. A pair per sample would have no discriminating power:
        # the point of modelling the leakage as an instrument property is that the same two
        # numbers must absorb the s/p disagreement of every polarized spectrum at once.
        if study.free.crosstalk == "fitted":
            layout.crosstalk_slice = slice(cursor, cursor + 2)
            cursor += 2

        layout.n_parameters = cursor
        return layout

    # -- naming -----------------------------------------------------------

    def parameter_names(self) -> list[str]:
        """One label per scalar of the vector, in vector order.

        Used to say which parameter ended on a bound and which one an uncertainty belongs
        to, without anyone having to re-derive the layout by hand.
        """
        names = [""] * self.n_parameters
        for stack_name, sl in self.thickness_slices.items():
            stack = self.study.stacks[stack_name]
            variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
            for k, layer_index in enumerate(variable):
                names[sl.start + k] = f"{stack_name} layer {layer_index + 1}"
        for material, sl in self.index_slices.items():
            for k in range(sl.stop - sl.start):
                names[sl.start + k] = f"index {material} knot {k + 1}"
        if self.aperture_slice is not None:
            for k in range(self.aperture_slice.stop - self.aperture_slice.start):
                names[self.aperture_slice.start + k] = f"beam aperture band {k + 1}"
        if self.crosstalk_slice is not None:
            names[self.crosstalk_slice.start] = "crosstalk alpha"
            names[self.crosstalk_slice.start + 1] = "crosstalk beta"
        return names

    # -- packing ----------------------------------------------------------

    def initial_vector(self, nominal: dict[str, np.ndarray]) -> np.ndarray:
        """Start from the nominal design, the declared aperture and the declared leakage."""
        x = np.zeros(self.n_parameters, dtype=np.float64)
        for name, sl in self.thickness_slices.items():
            stack = self.study.stacks[name]
            variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
            x[sl] = nominal[name][variable]
        if self.aperture_slice is not None:
            x[self.aperture_slice] = self.study.instrument.aperture_values_deg()
        if self.crosstalk_slice is not None:
            x[self.crosstalk_slice] = self.study.instrument.crosstalk_values()
        return x

    def bounds(
        self,
        nominal: dict[str, np.ndarray],
        thickness_tolerance: float | dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Box constraints: a relative window on thicknesses, a tube on the index.

        The thickness window is prior information about how the coating was made, declared in
        the study file; passing it here overrides the declaration, which is what a sensitivity
        scan does. A solution that reaches the window is reported rather than quietly
        accepted: it means the data and the prior disagree.
        """
        lower = np.full(self.n_parameters, -np.inf)
        upper = np.full(self.n_parameters, np.inf)
        for name, sl in self.thickness_slices.items():
            stack = self.study.stacks[name]
            variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
            d0 = nominal[name][variable]
            if thickness_tolerance is None:
                window = self.study.free.tolerance_for(name)
            elif isinstance(thickness_tolerance, dict):
                val = thickness_tolerance.get(name, 0.5)
                window = None if val is None else float(val)
            else:
                window = float(thickness_tolerance)
            if window is None or np.isinf(window):
                lower[sl] = 1e-3
                upper[sl] = np.inf
            else:
                lower[sl] = np.maximum(d0 * (1.0 - window), 1e-3)
                upper[sl] = d0 * (1.0 + window)
        delta = self.study.free.index_tube_delta
        for _, sl in self.index_slices.items():
            lower[sl] = -delta
            upper[sl] = +delta
        if self.aperture_slice is not None:
            lo, hi = self.study.instrument.aperture_bounds_deg
            lower[self.aperture_slice] = lo
            upper[self.aperture_slice] = hi
        if self.crosstalk_slice is not None:
            lo, hi = self.study.instrument.crosstalk_bounds
            lower[self.crosstalk_slice] = lo
            upper[self.crosstalk_slice] = hi
        return lower, upper

    def unpack(self, x: np.ndarray, nominal: dict[str, np.ndarray]) -> ParameterValues:
        """Turn a parameter vector into the quantities the forward model takes."""
        thicknesses = {
            name: np.array(value, dtype=np.float64) for name, value in nominal.items()
        }
        for name, sl in self.thickness_slices.items():
            stack = self.study.stacks[name]
            variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
            thicknesses[name][variable] = x[sl]
        corrections = {
            material: np.asarray(x[sl], dtype=np.float64)
            for material, sl in self.index_slices.items()
        }
        aperture = (
            self.study.instrument.aperture_values_deg()
            if self.aperture_slice is None
            else np.asarray(x[self.aperture_slice], dtype=np.float64)
        )
        crosstalk = (
            self.study.instrument.crosstalk_values()
            if self.crosstalk_slice is None
            else (float(x[self.crosstalk_slice][0]), float(x[self.crosstalk_slice][1]))
        )
        return ParameterValues(thicknesses, corrections, aperture, crosstalk)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SampleResidual:
    """How well one measurement is reproduced, before and after the inversion."""

    sample: str
    label: str
    quantity: str
    polarization: str
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
            "polarization": self.polarization,
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
    aperture_deg: np.ndarray
    crosstalk: tuple[float, float]
    thickness_sigma_nm: dict[str, np.ndarray] = field(default_factory=dict)
    qwot_sigma: dict[str, np.ndarray] = field(default_factory=dict)
    aperture_sigma_deg: np.ndarray | None = None
    crosstalk_sigma: tuple[float, float] | None = None
    covariance_scale: float = float("nan")
    covariance_note: str = ""
    window_override: str = ""
    at_bounds: list[str] = field(default_factory=list)
    extrapolation: list[dict] = field(default_factory=list)
    process_prior_pct: float | None = None

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

    def qwot_departure_in_sigma(self) -> dict[str, np.ndarray]:
        """Departure from the design, per layer, in units of its own uncertainty.

        This is the number that decides whether a layer moved: a departure of one percent
        means nothing until it is compared with what the measurement determines. Layers whose
        thickness was held return ``nan``.
        """
        out: dict[str, np.ndarray] = {}
        for name in self.qwot:
            sigma = self.qwot_sigma.get(name)
            if sigma is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                out[name] = (self.qwot[name] - self.nominal_qwot[name]) / sigma
        return out


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
# Covariance
# ---------------------------------------------------------------------------


def _covariance(
    jac: np.ndarray, weighted_residual: np.ndarray, n_parameters: int
) -> tuple[np.ndarray | None, float, str]:
    """Parameter covariance ``s^2 (J^T J)^{-1}`` from the Jacobian at the solution.

    ``J`` is the Jacobian of the residual *already divided by the declared uncertainty*, so
    ``(J^T J)^{-1}`` is the covariance the declared photometry implies. It is scaled by
    ``s^2 = chi^2 / (m - n)``, which widens the error bars when the model does not reproduce
    the data to within the declared sigma -- the honest direction, and the reason ``s^2`` is
    reported next to the uncertainties rather than folded into them silently.

    The inverse is taken through the singular values, so that a direction the data do not
    constrain shows up as a rank deficiency and is reported, instead of producing a very
    large number that looks like a result.
    """
    m = int(weighted_residual.size)
    dof = max(m - n_parameters, 1)
    scale = float(weighted_residual @ weighted_residual) / dof
    if jac is None or jac.size == 0:
        return None, scale, "no Jacobian available"
    _, singular, vt = np.linalg.svd(np.asarray(jac, dtype=np.float64), full_matrices=False)
    threshold = np.finfo(np.float64).eps * max(jac.shape) * (
        singular[0] if singular.size else 0.0
    )
    keep = singular > threshold
    note = ""
    if not np.all(keep):
        note = (
            f"the Jacobian is rank deficient: {int(np.count_nonzero(~keep))} of "
            f"{singular.size} directions are not constrained by the data, and the "
            f"uncertainties below are those of the constrained subspace only"
        )
    v = vt[keep].T
    covariance = scale * (v * (1.0 / singular[keep] ** 2)) @ v.T
    return covariance, scale, note


# ---------------------------------------------------------------------------
# The inversion
# ---------------------------------------------------------------------------


def invert(
    study: Study,
    *,
    thickness_tolerance: float | dict[str, float] | None = None,
    process_prior_pct: float | None = None,
    start: dict[str, np.ndarray] | None = None,
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
        Overrides the window declared by the study, for a sensitivity scan. Leave it at
        ``None`` to use the declaration, which is where it belongs: the window is prior
        information about how the coating was made, and a result depends on it. A layer that
        ends on it is listed in :attr:`InversionResult.at_bounds`.
    process_prior_pct:
        Process reproducibility standard deviation (in percent of nominal thickness) defining
        a Tikhonov / Bayesian MAP penalty towards the nominal design.
    start:
        Thicknesses to start the search from, per stack, instead of the nominal design.

        **This does not make the inversion stochastic.** A run started from a given point is
        as deterministic as any other, and the deposited studies never use it: they start from
        the nominal design, which is the only starting point that needs no justification. It
        exists so that the shape of the landscape can be *probed* -- many starts scattered
        through the search window, to find out whether the solution reached from the nominal
        design is the best one or merely the nearest one. That question deserves an answer,
        and answering it must not change what a deposited run does. See
        ``tools/multistart_probe.py``.
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

    def evaluate(x: np.ndarray) -> tuple[ParameterValues, list[np.ndarray]]:
        values = layout.unpack(x, nominal)
        if values.index_corrections:
            apply_corrections(values.index_corrections)
        predicted = evaluator.predict(
            values.thicknesses,
            aperture_deg=values.aperture_deg,
            crosstalk=values.crosstalk,
        )
        return values, predicted

    proc_prior = (
        process_prior_pct
        if process_prior_pct is not None
        else study.free.process_prior_pct
    )
    nom_vars: dict[str, np.ndarray] = {}
    if proc_prior is not None and proc_prior > 0:
        for name in layout.thickness_stacks:
            stack = study.stacks[name]
            var_indices = [i for i, layer in enumerate(stack.layers) if layer.variable]
            nom_vars[name] = nominal[name][var_indices]

    def residual(x: np.ndarray) -> np.ndarray:
        _, predicted = evaluate(x)
        res_data = (np.concatenate(predicted) - measured) / sigma
        if proc_prior is not None and proc_prior > 0:
            res_priors = [
                (x[sl] - nom_vars[name]) / (nom_vars[name] * (proc_prior / 100.0))
                for name, sl in layout.thickness_slices.items()
            ]
            if res_priors:
                return np.concatenate([res_data, *res_priors])
        return res_data

    x0 = layout.initial_vector(nominal if start is None else {**nominal, **start})
    lower, upper = layout.bounds(nominal, thickness_tolerance)
    # A start outside the window is not a start, it is a bound violation; clip rather than
    # let the optimiser reject the whole run.
    x0 = np.clip(x0, lower, upper)
    _, predicted_initial = evaluate(x0)
    residual_initial = (np.concatenate(predicted_initial) - measured) / sigma
    # A window other than the declared one is a different run, and the report says so rather
    # than leaving a reader to compare two sets of thicknesses obtained under two priors.
    window_override = (
        ""
        if thickness_tolerance is None
        else (
            f"the search window on the thicknesses was overridden at the call site to "
            f"{thickness_tolerance}, instead of the "
            f"{study.free.thickness_tolerance} the study declares"
        )
    )

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

    values, predicted = evaluate(x_final)
    residual_final = (np.concatenate(predicted) - measured) / sigma

    # -- what the data determine ------------------------------------------
    jac_data = None if solution is None else solution.jac[: measured.size, :]
    covariance, covariance_scale, covariance_note = (
        (None, float("nan"), "no free parameter")
        if solution is None
        else _covariance(jac_data, residual_final, layout.n_parameters)
    )
    sigma_x = (
        np.full(layout.n_parameters, np.nan)
        if covariance is None
        else np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    )

    thickness_sigma: dict[str, np.ndarray] = {}
    for name in study.used_stacks():
        stack = study.stacks[name]
        column = np.full(stack.n_layers, np.nan)
        sl = layout.thickness_slices.get(name)
        if sl is not None:
            variable = [i for i, layer in enumerate(stack.layers) if layer.variable]
            column[variable] = sigma_x[sl]
        thickness_sigma[name] = column
    factors = evaluator.qwot_per_nm()
    qwot_sigma = {
        name: factors[name] * value
        for name, value in thickness_sigma.items()
        if name in factors
    }

    aperture_sigma = (
        None if layout.aperture_slice is None else sigma_x[layout.aperture_slice]
    )
    crosstalk_sigma = (
        None
        if layout.crosstalk_slice is None
        else (
            float(sigma_x[layout.crosstalk_slice][0]),
            float(sigma_x[layout.crosstalk_slice][1]),
        )
    )

    # Which parameters ended on a bound: a silent bound hit is a wrong answer that looks
    # like a converged one, and it also makes the covariance above meaningless for that
    # direction, so every released block is checked, not only the thicknesses.
    at_bounds: list[str] = []
    names = layout.parameter_names()
    for i in range(layout.n_parameters):
        span = upper[i] - lower[i]
        if not np.isfinite(span):
            continue
        margin = 1e-6 * max(abs(span), 1.0)
        if x_final[i] <= lower[i] + margin or x_final[i] >= upper[i] - margin:
            at_bounds.append(names[i])

    # Per-measurement residuals, in the natural units of each measurement. The starting
    # point is recomputed rather than cached, so that "before" and "after" are produced by
    # the same code path and cannot drift apart.
    residuals: list[SampleResidual] = []
    _, predicted_initial = evaluate(x0)
    if values.index_corrections:
        apply_corrections(values.index_corrections)

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
                polarization=measurement.polarization,
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
        thicknesses=values.thicknesses,
        nominal_thicknesses=nominal,
        qwot=evaluator.qwot(values.thicknesses),
        nominal_qwot=evaluator.qwot(nominal),
        residuals=residuals,
        index_corrections=values.index_corrections,
        rms_initial=float(np.sqrt(np.mean(residual_initial**2))),
        rms_final=float(np.sqrt(np.mean(residual_final**2))),
        chi2_per_point=float(np.mean(residual_final**2)),
        n_function_evaluations=0 if solution is None else int(solution.nfev),
        optimality=0.0 if solution is None else float(solution.optimality),
        success=True if solution is None else bool(solution.success),
        message="no free parameter" if solution is None else str(solution.message),
        aperture_deg=values.aperture_deg,
        crosstalk=values.crosstalk,
        thickness_sigma_nm=thickness_sigma,
        qwot_sigma=qwot_sigma,
        aperture_sigma_deg=aperture_sigma,
        crosstalk_sigma=crosstalk_sigma,
        covariance_scale=covariance_scale,
        covariance_note=covariance_note,
        window_override=window_override,
        at_bounds=at_bounds,
        extrapolation=extrapolation,
        process_prior_pct=proc_prior,
    )
