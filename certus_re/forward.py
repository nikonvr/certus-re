"""Predicting what each measurement of a study should read, given a set of thicknesses.

This is the layer between the study description and the optical model: it resolves which
stack sits on which face of which sample, interpolates the dispersions onto each measured
grid once and for all, applies the piecewise-constant beam aperture, and returns one
predicted spectrum per measurement, in the same order the study declares them.

Everything that does not depend on the thicknesses is computed once, at construction:
index tables on the measured grids, aperture band membership, and the transmittance of the
bare substrate that relative measurements are divided by. What remains inside the loop is
the transfer matrix itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Measurement, Sample, Study
from .physics import aperture_nodes, assemble_plate

__all__ = ["Evaluator", "MeasurementPlan"]


@dataclass(slots=True)
class MeasurementPlan:
    """Everything needed to predict one measurement, resolved once.

    Attributes
    ----------
    sample, measurement:
        The objects this plan predicts.
    n_front, n_rear:
        Complex indices of the layers of each face, shape ``(W, L)``, ordered as the stack
        is -- column 0 adjacent to the substrate.
    n_substrate:
        Complex index of the substrate on the measured grid.
    angles_deg, weights:
        The angles at which the model is evaluated and their weights, one row per aperture
        band. A measurement below the aperture threshold has a single angle of weight one.
    band_masks:
        Which measured points belong to each aperture band.
    reference_transmittance:
        Transmittance of the bare substrate, for a relative measurement; ``None`` otherwise.
    """

    sample: Sample
    measurement: Measurement
    n_front: np.ndarray
    n_rear: np.ndarray | None
    n_substrate: np.ndarray
    front_stack: str | None
    rear_stack: str | None
    angles_deg: list[np.ndarray]
    weights: list[np.ndarray]
    band_masks: list[np.ndarray]
    reference_transmittance: np.ndarray | None = None
    polarizations: tuple[str, ...] = ("s", "p")
    pol_weights: tuple[float, ...] = (0.5, 0.5)


@dataclass(slots=True)
class Evaluator:
    """Predict every measurement of a study from a set of layer thicknesses.

    Parameters
    ----------
    study:
        The problem. Its measurements are flattened, in declaration order, into
        :attr:`plans`; predictions come back in that same order.
    legacy_gain_sign:
        Reproduce the sign convention of the reference implementation. For comparison only;
        see :func:`certus_re.physics.as_macleod`.
    """

    study: Study
    legacy_gain_sign: bool = False
    plans: list[MeasurementPlan] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for sample in self.study.samples:
            for measurement in sample.measurements:
                self.plans.append(self._plan(sample, measurement))

    # -- construction -----------------------------------------------------

    def _stack_indices(self, stack_name: str | None, wavelength: np.ndarray) -> np.ndarray:
        if stack_name is None:
            return np.zeros((wavelength.size, 0), dtype=np.complex128)
        materials = self.study.stacks[stack_name].materials
        return np.stack(
            [self.study.materials[m].complex_at(wavelength) for m in materials], axis=1
        )

    def _plan(self, sample: Sample, measurement: Measurement) -> MeasurementPlan:
        wavelength = measurement.wavelength_nm
        instrument = self.study.instrument

        n_front = self._stack_indices(sample.front_stack, wavelength)
        n_rear = (
            self._stack_indices(sample.rear_stack, wavelength)
            if sample.rear_stack is not None
            else None
        )
        n_substrate = self.study.substrates[sample.substrate.material].complex_at(wavelength)

        # Aperture: one angle set per band. Below the threshold angle the response is
        # stationary in theta to first order, so averaging over the cone changes nothing
        # measurable and a single evaluation is both faster and more honest.
        apply_aperture = abs(measurement.angle_deg) >= instrument.aperture_min_angle_deg
        band_of_point = instrument.band_index(wavelength)
        apertures = instrument.aperture_values_deg()
        band_masks: list[np.ndarray] = []
        angles: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for band in range(instrument.n_bands):
            mask = band_of_point == band
            if not np.any(mask):
                continue
            band_masks.append(mask)
            if not apply_aperture:
                angles.append(np.array([float(measurement.angle_deg)]))
                weights.append(np.array([1.0]))
                continue
            offsets, node_weights = aperture_nodes(
                float(apertures[band]), instrument.n_aperture_nodes
            )
            theta = float(measurement.angle_deg) + offsets
            # Keep every ray inside the physical range of incidence angles.
            if theta.min() <= 0.0 or theta.max() >= 90.0:
                half = min(
                    float(measurement.angle_deg) - 1e-6,
                    90.0 - 1e-6 - float(measurement.angle_deg),
                )
                offsets, node_weights = aperture_nodes(
                    2.0 * max(half, 0.0), instrument.n_aperture_nodes
                )
                theta = float(measurement.angle_deg) + offsets
            angles.append(theta)
            weights.append(node_weights)

        if measurement.polarization == "s":
            polarizations, pol_weights = ("s",), (1.0,)
        elif measurement.polarization == "p":
            polarizations, pol_weights = ("p",), (1.0,)
        else:
            polarizations, pol_weights = ("s", "p"), (0.5, 0.5)

        plan = MeasurementPlan(
            sample=sample,
            measurement=measurement,
            n_front=n_front,
            n_rear=n_rear,
            n_substrate=n_substrate,
            front_stack=sample.front_stack,
            rear_stack=sample.rear_stack,
            angles_deg=angles,
            weights=weights,
            band_masks=band_masks,
            polarizations=polarizations,
            pol_weights=pol_weights,
        )

        if measurement.quantity == "Trel":
            plan.reference_transmittance = self._bare_substrate_transmittance(plan)
        return plan

    def _bare_substrate_transmittance(self, plan: MeasurementPlan) -> np.ndarray:
        """Transmittance of the uncoated substrate, the reference of a relative measurement.

        It does not depend on any thickness, so it is computed once. A relative
        transmittance measured against a bare witness divides by exactly this.
        """
        out = np.zeros(plan.measurement.wavelength_nm.size, dtype=np.float64)
        for mask, angles, weights in zip(plan.band_masks, plan.angles_deg, plan.weights):
            w = plan.measurement.wavelength_nm[mask]
            accumulated = np.zeros(w.size, dtype=np.float64)
            for angle, weight in zip(angles, weights):
                for pol, pol_weight in zip(plan.polarizations, plan.pol_weights):
                    result = assemble_plate(
                        w,
                        n_substrate=plan.n_substrate[mask],
                        substrate_thickness_mm=plan.sample.substrate.thickness_mm,
                        angle_deg=float(angle),
                        polarization=pol,
                        rear="bare",
                        legacy_gain_sign=self.legacy_gain_sign,
                    )
                    accumulated += weight * pol_weight * result.T
            out[mask] = accumulated
        return out

    # -- evaluation -------------------------------------------------------

    def predict(self, thicknesses: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Predicted value of every measurement, in study order.

        Parameters
        ----------
        thicknesses:
            Physical thicknesses in nanometres, keyed by stack name. Every stack referenced
            by a sample must be present.
        """
        return [self.predict_one(plan, thicknesses) for plan in self.plans]

    def predict_one(
        self, plan: MeasurementPlan, thicknesses: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Predicted value of one measurement."""
        measurement = plan.measurement
        d_front = (
            np.zeros(0)
            if plan.front_stack is None
            else np.asarray(thicknesses[plan.front_stack], dtype=np.float64)
        )
        d_rear = (
            None
            if plan.rear_stack is None
            else np.asarray(thicknesses[plan.rear_stack], dtype=np.float64)
        )
        rear = plan.sample.substrate.rear

        out = np.zeros(measurement.wavelength_nm.size, dtype=np.float64)
        for mask, angles, weights in zip(plan.band_masks, plan.angles_deg, plan.weights):
            w = measurement.wavelength_nm[mask]
            accumulated = np.zeros(w.size, dtype=np.float64)
            for angle, weight in zip(angles, weights):
                for pol, pol_weight in zip(plan.polarizations, plan.pol_weights):
                    result = assemble_plate(
                        w,
                        n_substrate=plan.n_substrate[mask],
                        substrate_thickness_mm=plan.sample.substrate.thickness_mm,
                        angle_deg=float(angle),
                        polarization=pol,
                        n_front_layers=plan.n_front[mask],
                        d_front_nm=d_front,
                        n_rear_layers=None if plan.n_rear is None else plan.n_rear[mask],
                        d_rear_nm=d_rear,
                        rear=rear,
                        legacy_gain_sign=self.legacy_gain_sign,
                    )
                    value = result.R if measurement.quantity == "R" else result.T
                    accumulated += weight * pol_weight * value
            out[mask] = accumulated

        if measurement.quantity == "Trel":
            reference = plan.reference_transmittance
            if reference is None:
                raise RuntimeError("relative measurement without a reference transmittance")
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(reference > 1e-12, out / np.maximum(reference, 1e-12), 0.0)
        return out

    # -- convenience ------------------------------------------------------

    def nominal_thicknesses(self) -> dict[str, np.ndarray]:
        """Nominal physical thickness of every stack, from its design and its indices.

        A layer specified in quarter waves is converted through the index of its own
        material at the study reference wavelength; a layer specified in nanometres keeps
        the thickness it was given.
        """
        lambda0 = self.study.lambda0_nm
        out: dict[str, np.ndarray] = {}
        for name, stack in self.study.stacks.items():
            values = []
            for layer in stack.layers:
                if layer.nominal_thickness_nm is not None:
                    values.append(float(layer.nominal_thickness_nm))
                else:
                    n0 = float(self.study.materials[layer.material].n_at(lambda0))
                    values.append(layer.thickness_nm(lambda0, n0))
            out[name] = np.asarray(values, dtype=np.float64)
        return out

    def qwot(self, thicknesses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Optical thickness in quarter waves, per layer, for a set of thicknesses."""
        lambda0 = self.study.lambda0_nm
        out: dict[str, np.ndarray] = {}
        for name, stack in self.study.stacks.items():
            if name not in thicknesses:
                continue
            n0 = np.asarray(
                [
                    float(self.study.materials[layer.material].n_at(lambda0))
                    for layer in stack.layers
                ]
            )
            out[name] = 4.0 * n0 * np.asarray(thicknesses[name], dtype=np.float64) / lambda0
        return out
