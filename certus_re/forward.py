"""Predicting what each measurement of a study should read, given a set of thicknesses.

This is the layer between the study description and the optical model: it resolves which
stack sits on which face of which sample, interpolates the dispersions onto each measured
grid once and for all, averages over the beam cone band by band, mixes the two polarization
channels by the leakage of the polarizer, and returns one predicted spectrum per measurement,
in the order the study declares them.

Everything that does not depend on the parameters is computed once, at construction: index
tables on the measured grids and aperture band membership. What depends on the beam aperture
is computed per call and memoised on the aperture value, so that an imposed aperture costs
exactly what it did before the aperture could be released.

Two instrument effects are applied to the **computed** spectrum, never to the measurement:

* the finite beam aperture, as a weighted average of the model over a cone of angles, with
  the aperture constant inside each declared wavelength band;
* the polarizer leakage, as the mixture ``R_s^meas = (1-alpha) R_s + alpha R_p`` and
  ``R_p^meas = (1-beta) R_p + beta R_s``.

Correcting the data instead would amplify the photometric noise and would leave an archive of
spectra that are no longer what the instrument recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Measurement, Sample, Study
from .physics import aperture_nodes, assemble_plate

__all__ = ["Evaluator", "MeasurementPlan"]

_BOTH = ("s", "p")


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
    band_masks, band_indices:
        Which measured points belong to each aperture band, and which band that is. Bands
        carrying no point of this measurement are dropped, so the two lists are parallel and
        never empty.
    apply_aperture:
        Whether the cone average is applied at all. Below
        :attr:`~certus_re.model.Instrument.aperture_min_angle_deg` it is not: at near-normal
        incidence ``dR/dtheta`` vanishes to first order and averaging changes nothing
        measurable.
    polarizations:
        The **pure** polarizations the optical model is evaluated in. Always ``("s", "p")``
        for an unpolarized measurement, and also for a polarized one as soon as leakage is
        modelled -- ``R_s`` alone cannot be mixed with an ``R_p`` that was never computed.
    crosstalk_role:
        ``"none"``, ``"s"`` or ``"p"``: which line of the mixing applies to this measurement.
    """

    sample: Sample
    measurement: Measurement
    n_front: np.ndarray
    n_rear: np.ndarray | None
    n_substrate: np.ndarray
    front_stack: str | None
    rear_stack: str | None
    band_masks: list[np.ndarray]
    band_indices: list[int]
    apply_aperture: bool
    nominal_angle_deg: float
    n_aperture_nodes: int
    polarizations: tuple[str, ...] = _BOTH
    crosstalk_role: str = "none"
    _angles: dict = field(default_factory=dict, repr=False)
    _reference: dict = field(default_factory=dict, repr=False)

    # -- aperture ---------------------------------------------------------

    def aperture_key(self, aperture_deg: np.ndarray) -> tuple:
        """Cache key for the angle sets implied by a vector of per-band apertures.

        A measurement that does not apply the aperture has one key for every aperture: its
        angles do not depend on it, and neither does anything derived from them.
        """
        if not self.apply_aperture:
            return ()
        return tuple(float(aperture_deg[b]) for b in self.band_indices)

    def angles_and_weights(
        self, aperture_deg: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Angles and quadrature weights for each band, for a given aperture vector."""
        key = self.aperture_key(aperture_deg)
        cached = self._angles.get(key)
        if cached is not None:
            return cached
        angles: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for band in self.band_indices:
            if not self.apply_aperture:
                angles.append(np.array([self.nominal_angle_deg]))
                weights.append(np.array([1.0]))
                continue
            total = float(aperture_deg[band])
            offsets, node_weights = aperture_nodes(total, self.n_aperture_nodes)
            theta = self.nominal_angle_deg + offsets
            # Keep every ray inside the physical range of incidence angles.
            if theta.min() <= 0.0 or theta.max() >= 90.0:
                half = min(
                    self.nominal_angle_deg - 1e-6,
                    90.0 - 1e-6 - self.nominal_angle_deg,
                )
                offsets, node_weights = aperture_nodes(
                    2.0 * max(half, 0.0), self.n_aperture_nodes
                )
                theta = self.nominal_angle_deg + offsets
            angles.append(theta)
            weights.append(node_weights)
        # A released aperture takes a new value at every iteration, so this cache would grow
        # with the run. It exists to make an imposed aperture free, not to remember a search
        # path; dropping it wholesale keeps the memory bounded and changes no result.
        if len(self._angles) >= 64:
            self._angles.clear()
        self._angles[key] = (angles, weights)
        return angles, weights

    # -- polarization mixing ----------------------------------------------

    def pol_weights(self, alpha: float, beta: float) -> tuple[float, ...]:
        """Weight of each pure polarization in the predicted value, in :attr:`polarizations`.

        Unpolarized: the half-sum, whatever the leakage. A channel recorded without a
        polarizer cannot be contaminated by one, and this is also the statement that makes
        the leakage observable only in the resolved channels.
        """
        if self.crosstalk_role == "s":
            return (1.0 - alpha, alpha)
        if self.crosstalk_role == "p":
            return (beta, 1.0 - beta)
        if len(self.polarizations) == 1:
            return (1.0,)
        return (0.5, 0.5)


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

        band_of_point = instrument.band_index(wavelength)
        band_masks: list[np.ndarray] = []
        band_indices: list[int] = []
        for band in range(instrument.n_bands):
            mask = band_of_point == band
            if np.any(mask):
                band_masks.append(mask)
                band_indices.append(band)

        polarization = measurement.polarization
        if polarization == "a":
            polarizations, role = _BOTH, "none"
        elif instrument.models_crosstalk:
            polarizations, role = _BOTH, polarization
        else:
            polarizations, role = (polarization,), "none"

        return MeasurementPlan(
            sample=sample,
            measurement=measurement,
            n_front=n_front,
            n_rear=n_rear,
            n_substrate=n_substrate,
            front_stack=sample.front_stack,
            rear_stack=sample.rear_stack,
            band_masks=band_masks,
            band_indices=band_indices,
            apply_aperture=abs(measurement.angle_deg)
            >= instrument.aperture_min_angle_deg,
            nominal_angle_deg=float(measurement.angle_deg),
            n_aperture_nodes=instrument.n_aperture_nodes,
            polarizations=polarizations,
            crosstalk_role=role,
        )

    # -- evaluation -------------------------------------------------------

    def _cone_average(
        self,
        plan: MeasurementPlan,
        aperture_deg: np.ndarray,
        thicknesses: dict[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        """Model output per pure polarization, averaged over the beam cone.

        ``thicknesses = None`` evaluates the bare substrate, which is the reference of a
        relative transmittance. Doing it through the same code path is what guarantees that
        the numerator and the denominator of a ratio see the same angles.
        """
        measurement = plan.measurement
        size = measurement.wavelength_nm.size
        quantity = "T" if thicknesses is None else measurement.quantity
        out = {pol: np.zeros(size, dtype=np.float64) for pol in plan.polarizations}

        if thicknesses is None:
            d_front: np.ndarray | None = None
            d_rear: np.ndarray | None = None
            rear = "bare"
        else:
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

        angles, weights = plan.angles_and_weights(aperture_deg)
        for mask, band_angles, band_weights in zip(plan.band_masks, angles, weights):
            w = measurement.wavelength_nm[mask]
            for pol in plan.polarizations:
                accumulated = np.zeros(w.size, dtype=np.float64)
                for angle, weight in zip(band_angles, band_weights):
                    if thicknesses is None:
                        result = assemble_plate(
                            w,
                            n_substrate=plan.n_substrate[mask],
                            substrate_thickness_mm=plan.sample.substrate.thickness_mm,
                            angle_deg=float(angle),
                            polarization=pol,
                            rear="bare",
                            legacy_gain_sign=self.legacy_gain_sign,
                        )
                    else:
                        result = assemble_plate(
                            w,
                            n_substrate=plan.n_substrate[mask],
                            substrate_thickness_mm=plan.sample.substrate.thickness_mm,
                            angle_deg=float(angle),
                            polarization=pol,
                            n_front_layers=plan.n_front[mask],
                            d_front_nm=d_front,
                            n_rear_layers=(
                                None if plan.n_rear is None else plan.n_rear[mask]
                            ),
                            d_rear_nm=d_rear,
                            rear=rear,
                            legacy_gain_sign=self.legacy_gain_sign,
                        )
                    accumulated += weight * (result.R if quantity == "R" else result.T)
                out[pol][mask] = accumulated
        return out

    def _reference_transmittance(
        self, plan: MeasurementPlan, aperture_deg: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Transmittance of the uncoated substrate, per pure polarization.

        It does not depend on any thickness, so it is computed once per angle set. The
        polarizer mixes the flux measured on the reference exactly as it mixes the flux
        measured on the sample, so the ratio must be formed **after** the mixing, not before;
        keeping the two channels separate here is what makes that possible.
        """
        key = plan.aperture_key(aperture_deg)
        cached = plan._reference.get(key)
        if cached is None:
            cached = self._cone_average(plan, aperture_deg, None)
            plan._reference[key] = cached
        return cached

    def predict(
        self,
        thicknesses: dict[str, np.ndarray],
        *,
        aperture_deg: np.ndarray | None = None,
        crosstalk: tuple[float, float] | None = None,
    ) -> list[np.ndarray]:
        """Predicted value of every measurement, in study order.

        Parameters
        ----------
        thicknesses:
            Physical thicknesses in nanometres, keyed by stack name. Every stack referenced
            by a sample must be present.
        aperture_deg:
            Total cone aperture per band, in degrees. Defaults to the instrument's declared
            values.
        crosstalk:
            ``(alpha, beta)`` polarizer leakage. Defaults to the instrument's declared pair.
        """
        aperture, mixing = self._instrument_state(aperture_deg, crosstalk)
        return [
            self._predict_one(plan, thicknesses, aperture, mixing) for plan in self.plans
        ]

    def predict_one(
        self,
        plan: MeasurementPlan,
        thicknesses: dict[str, np.ndarray],
        *,
        aperture_deg: np.ndarray | None = None,
        crosstalk: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Predicted value of one measurement."""
        aperture, mixing = self._instrument_state(aperture_deg, crosstalk)
        return self._predict_one(plan, thicknesses, aperture, mixing)

    def _instrument_state(
        self,
        aperture_deg: np.ndarray | None,
        crosstalk: tuple[float, float] | None,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        instrument = self.study.instrument
        aperture = (
            instrument.aperture_values_deg()
            if aperture_deg is None
            else np.asarray(aperture_deg, dtype=np.float64)
        )
        if aperture.size != instrument.n_bands:
            raise ValueError(
                f"{aperture.size} aperture values for {instrument.n_bands} bands"
            )
        mixing = instrument.crosstalk_values() if crosstalk is None else (
            float(crosstalk[0]),
            float(crosstalk[1]),
        )
        return aperture, mixing

    def _predict_one(
        self,
        plan: MeasurementPlan,
        thicknesses: dict[str, np.ndarray],
        aperture_deg: np.ndarray,
        crosstalk: tuple[float, float],
    ) -> np.ndarray:
        measurement = plan.measurement
        pure = self._cone_average(plan, aperture_deg, thicknesses)
        weights = plan.pol_weights(*crosstalk)

        out = np.zeros(measurement.wavelength_nm.size, dtype=np.float64)
        for pol, weight in zip(plan.polarizations, weights):
            out = out + weight * pure[pol]

        if measurement.quantity == "Trel":
            reference_pure = self._reference_transmittance(plan, aperture_deg)
            reference = np.zeros_like(out)
            for pol, weight in zip(plan.polarizations, weights):
                reference = reference + weight * reference_pure[pol]
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

    def qwot_per_nm(self) -> dict[str, np.ndarray]:
        """Conversion factor ``4 n(lambda0) / lambda0`` from nanometres to quarter waves.

        One value per layer, because the factor is the index of that layer's own material.
        It converts a thickness and, with it, the uncertainty on a thickness.
        """
        lambda0 = self.study.lambda0_nm
        return {
            name: np.asarray(
                [
                    4.0
                    * float(self.study.materials[layer.material].n_at(lambda0))
                    / lambda0
                    for layer in stack.layers
                ],
                dtype=np.float64,
            )
            for name, stack in self.study.stacks.items()
        }

    def qwot(self, thicknesses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Optical thickness in quarter waves, per layer, for a set of thicknesses."""
        factors = self.qwot_per_nm()
        return {
            name: factors[name] * np.asarray(value, dtype=np.float64)
            for name, value in thicknesses.items()
            if name in factors
        }
