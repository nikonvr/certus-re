"""Data model for a multi-sample reverse-engineering study.

The model is deliberately explicit about one thing: **which unknowns are shared between
samples**. A coating run produces a stack; that stack may be measured alone, or as one face
of a two-side-coated component. Both measurements constrain the *same* layer thicknesses.
Representing stacks as named objects, and samples as assemblies referencing them, is what
makes a joint inversion meaningful rather than a concatenation of independent fits.

Vocabulary
----------
Stack
    An ordered list of layers produced by one deposition run, identified by a name. Layers
    are specified in quarter-wave optical thickness (QWOT) at a reference wavelength, which
    is the unit the design is written in and the quantity optical monitoring controls.
Sample
    A physical specimen: a substrate, a front stack, optionally a rear stack, and the
    measurements recorded on it. Two samples may reference the same stack.
Measurement
    One measured spectral quantity on one sample: reflectance, transmittance or relative
    transmittance, at one angle of incidence, in one polarization state.
Study
    The whole set: materials, substrates, stacks, samples, instrument model, and the
    declaration of which parameter blocks are released.

Conventions
-----------
* Wavelengths are in nanometres, angles in degrees, thicknesses in nanometres.
* Photometric quantities are fractions in [0, 1], never percent. Readers convert on input.
* Polarization is ``"s"``, ``"p"`` or ``"a"``. ``"a"`` is the unpolarized channel, defined
  as the half-sum (R_s + R_p) / 2, which is what a spectrophotometer without a polarizer
  measures and what this study uses throughout.
* The angle of incidence is the nominal angle of the chief ray. The finite beam aperture is
  an instrument property, described by :class:`Instrument`, not by the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Polarization = Literal["s", "p", "a"]
Quantity = Literal["R", "T", "Trel"]
RearFace = Literal["bare", "coated", "none"]

__all__ = [
    "Layer",
    "Stack",
    "Substrate",
    "Measurement",
    "Sample",
    "Instrument",
    "FreeParameters",
    "Study",
    "Polarization",
    "Quantity",
    "RearFace",
]


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layer:
    """One layer of a deposition run.

    Parameters
    ----------
    material:
        Key into :attr:`Study.materials`.
    qwot:
        Nominal optical thickness in quarter waves at the study reference wavelength.
    variable:
        Whether the thickness of this layer is released during inversion. A layer held
        fixed still contributes to the spectrum; it just does not count as a free
        parameter.

    Notes
    -----
    The physical thickness is *derived*, ``d = qwot * lambda0 / (4 n(lambda0))``, and
    therefore depends on the index dataset in use. This is why two index datasets cannot be
    compared on physical thickness: each solution must be compared to its own nominal.
    """

    material: str
    qwot: float
    variable: bool = True
    nominal_thickness_nm: float | None = None

    @property
    def nominal_is_thickness(self) -> bool:
        """Whether this layer was specified in nanometres rather than in quarter waves.

        A multilayer design is written in QWOT; a single witness layer is specified by the
        thickness its own determination returned. Both are legitimate nominals, and a report
        must compare each solution to the one actually declared -- otherwise a change of
        index dataset moves the reference and manufactures a departure that is not there.
        """
        return self.nominal_thickness_nm is not None

    def thickness_nm(self, lambda0_nm: float, n_at_lambda0: float) -> float:
        """Physical thickness implied by this layer's QWOT under a given index."""
        if not np.isfinite(n_at_lambda0) or n_at_lambda0 <= 0.0:
            raise ValueError(
                f"material {self.material!r}: non-physical index {n_at_lambda0!r} at "
                f"lambda0 = {lambda0_nm} nm"
            )
        return float(self.qwot) * float(lambda0_nm) / (4.0 * float(n_at_lambda0))


@dataclass(slots=True)
class Stack:
    """An ordered list of layers produced by one deposition run.

    Layer 1 is the first layer deposited, hence the one adjacent to the substrate. This is
    the order a coating engineer writes a design in, and the order the monitoring follows.
    The transfer-matrix code uses the opposite convention internally; the conversion is done
    once, at the boundary, by the forward model.
    """

    name: str
    layers: list[Layer]
    run: str | None = None
    comment: str = ""

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError(f"stack {self.name!r} has no layer")

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_variable(self) -> int:
        """Number of layers released during inversion."""
        return sum(1 for layer in self.layers if layer.variable)

    @property
    def materials(self) -> list[str]:
        return [layer.material for layer in self.layers]

    def qwot_vector(self) -> np.ndarray:
        return np.asarray([layer.qwot for layer in self.layers], dtype=np.float64)

    def total_qwot(self) -> float:
        return float(self.qwot_vector().sum())


# ---------------------------------------------------------------------------
# Substrates and samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Substrate:
    """A substrate slab.

    Parameters
    ----------
    material:
        Key into :attr:`Study.substrates`.
    thickness_mm:
        Physical thickness. Only used to state that the slab is thick enough to be treated
        incoherently; no interference is computed inside it.
    rear:
        ``"bare"``   -- uncoated rear face, Fresnel interface with air, incoherent cavity;
        ``"coated"`` -- rear face carries the stack named by :attr:`Sample.rear_stack`;
        ``"none"``   -- semi-infinite substrate, no rear return at all (the sample was
        measured on a wedged or index-matched slab, or the rear return was baffled out).
    """

    material: str
    thickness_mm: float = 1.0
    rear: RearFace = "bare"


@dataclass(slots=True)
class Measurement:
    """One measured spectrum on one sample.

    Parameters
    ----------
    quantity:
        ``"R"``    -- reflectance of the assembled sample;
        ``"T"``    -- transmittance of the assembled sample;
        ``"Trel"`` -- transmittance divided by that of the bare substrate, the quantity the
        single-layer witnesses are measured in.
    angle_deg:
        Nominal angle of incidence of the chief ray.
    polarization:
        ``"s"``, ``"p"`` or ``"a"``. See module docstring.
    wavelength_nm, value:
        The measured spectrum, as fractions in [0, 1]. Readers are responsible for the
        percent-to-fraction conversion and for sorting by wavelength.
    sigma:
        One-sigma photometric uncertainty, either a scalar or a per-point array, in the same
        units as ``value``. Residuals are divided by it, so a joint inversion weights each
        dataset by what it is actually worth rather than by how many points it happens to
        contain. Required for joint inversions; optional for single-sample runs, where a
        constant cancels out of the minimization.
    band_nm:
        Optional ``(lambda_min, lambda_max)`` restriction applied on load. Points outside it
        are dropped and never reach the solver.
    label:
        Free-form provenance string: source file, column name, acquisition date.
    """

    quantity: Quantity
    angle_deg: float
    polarization: Polarization
    wavelength_nm: np.ndarray
    value: np.ndarray
    sigma: np.ndarray | float | None = None
    band_nm: tuple[float, float] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=np.float64).ravel()
        self.value = np.asarray(self.value, dtype=np.float64).ravel()
        if self.wavelength_nm.size != self.value.size:
            raise ValueError(
                f"measurement {self.label or self.quantity!r}: "
                f"{self.wavelength_nm.size} wavelengths for {self.value.size} values"
            )
        if self.wavelength_nm.size == 0:
            raise ValueError(f"measurement {self.label or self.quantity!r} is empty")
        if np.any(np.diff(self.wavelength_nm) <= 0.0):
            order = np.argsort(self.wavelength_nm, kind="mergesort")
            self.wavelength_nm = self.wavelength_nm[order]
            self.value = self.value[order]
            if isinstance(self.sigma, np.ndarray) and self.sigma.size == order.size:
                self.sigma = self.sigma[order]
        if self.polarization not in ("s", "p", "a"):
            raise ValueError(f"unknown polarization {self.polarization!r}")
        if self.quantity not in ("R", "T", "Trel"):
            raise ValueError(f"unknown quantity {self.quantity!r}")
        if isinstance(self.sigma, np.ndarray):
            self.sigma = np.asarray(self.sigma, dtype=np.float64).ravel()
            if self.sigma.size not in (1, self.value.size):
                raise ValueError(
                    f"measurement {self.label!r}: sigma has {self.sigma.size} entries "
                    f"for {self.value.size} points"
                )

    @property
    def n_points(self) -> int:
        return int(self.value.size)

    def sigma_vector(self) -> np.ndarray:
        """One-sigma uncertainty broadcast to one value per point.

        A measurement with no declared uncertainty gets ``1.0``, which makes the residual a
        plain least-squares residual. That is legitimate for a single dataset and wrong for
        a joint one; :meth:`Study.validate` reports it.
        """
        if self.sigma is None:
            return np.ones(self.value.size, dtype=np.float64)
        arr = np.asarray(self.sigma, dtype=np.float64).ravel()
        if arr.size == 1:
            return np.full(self.value.size, float(arr[0]), dtype=np.float64)
        return arr


@dataclass(slots=True)
class Sample:
    """A physical specimen and everything measured on it.

    Parameters
    ----------
    name:
        Identifier, used in reports and in the free-parameter declaration.
    substrate:
        The slab.
    front_stack:
        Name of the stack on the illuminated face, or ``None`` for a bare substrate.
    rear_stack:
        Name of the stack on the rear face, for a two-side-coated component. Requires
        ``substrate.rear == "coated"``.
    measurements:
        The spectra recorded on this specimen.

    Notes
    -----
    ``front_stack`` and ``rear_stack`` are *names*, not objects. Two samples naming the same
    stack share its thickness unknowns: the seventeen-layer beamsplitter measured alone and
    the same run measured as the front face of the two-side-coated component are one set of
    seventeen unknowns constrained by two spectra, not two sets of seventeen.
    """

    name: str
    substrate: Substrate
    front_stack: str | None = None
    rear_stack: str | None = None
    measurements: list[Measurement] = field(default_factory=list)
    comment: str = ""

    def __post_init__(self) -> None:
        if self.rear_stack is not None and self.substrate.rear != "coated":
            raise ValueError(
                f"sample {self.name!r} names a rear stack {self.rear_stack!r} but its "
                f'substrate rear face is {self.substrate.rear!r}; set rear="coated"'
            )
        if self.substrate.rear == "coated" and self.rear_stack is None:
            raise ValueError(
                f'sample {self.name!r} declares a coated rear face but names no rear stack'
            )

    @property
    def n_points(self) -> int:
        return sum(m.n_points for m in self.measurements)

    @property
    def stack_names(self) -> list[str]:
        return [s for s in (self.front_stack, self.rear_stack) if s is not None]

    @property
    def is_two_sided(self) -> bool:
        return self.rear_stack is not None


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Instrument:
    """The spectrophotometer, described by what is documented about it.

    The beam aperture is modelled as a **piecewise-constant function of wavelength**. The
    angular distribution seen by the sample is set by the pupil of the illumination system;
    for an all-mirror train it is achromatic at fixed configuration, so it cannot drift
    continuously with wavelength. What does change, and discontinuously, is the instrument
    configuration: the switchovers of source, detector and grating. The steps are therefore
    placed at the wavelengths the manufacturer documents for those switchovers, and are
    *imposed*, not fitted.

    Parameters
    ----------
    beam_aperture_deg:
        Total cone aperture, in degrees. The manufacturer specification of a beam divergence
        of +/- 1 degree is a total aperture of 2.0 degrees. Stored as the total, never as the
        half-angle, because that confusion silently halves or doubles the effect.
    aperture_band_edges_nm:
        Wavelengths at which the aperture is allowed to step, in increasing order. An empty
        tuple means a single band, hence a constant aperture over the whole range.
    aperture_per_band_deg:
        One total aperture per band, of length ``len(aperture_band_edges_nm) + 1``. If
        ``None``, ``beam_aperture_deg`` applies to every band.
    aperture_mode:
        ``"imposed"`` -- the values above are used as given and no aperture parameter is
        released; ``"fitted"`` -- one aperture per band is released, within
        ``aperture_bounds_deg``. Only ``"imposed"`` supports the claim that the inversion
        did not spend degrees of freedom on the instrument.
    aperture_bounds_deg:
        Bounds used when ``aperture_mode == "fitted"``.
    aperture_min_angle_deg:
        Below this angle of incidence the aperture term is not applied: at near-normal
        incidence dR/dtheta vanishes to first order, so averaging over the cone changes
        nothing measurable.
    n_aperture_nodes:
        Number of angles used for the quadrature over the cone. ``2`` reproduces the
        historical two-angle mean ``(theta - h, theta + h)``; an odd value >= 3 uses a
        Gauss-Legendre quadrature over the cone, which is the correct averaging and differs
        between polarizations wherever R(theta) is curved.
    crosstalk_alpha, crosstalk_beta:
        Polarizer leakage. A polarizer does not separate the two states perfectly, so the
        measured reflectances are mixtures of the pure ones:

        .. math::

            R_s^{meas} = (1-\\alpha) R_s + \\alpha R_p , \\qquad
            R_p^{meas} = (1-\\beta) R_p + \\beta R_s .

        Two coefficients rather than one, because the exit slit is strongly anisotropic --
        220 um wide for a 5 to 6 mm high spot -- and the polarizers sit immediately behind
        it, where the beam still has the aperture of the monochromator. The transmission
        axis being rotated by 90 degrees between the two settings, the effective extinction
        ratio has no reason to be the same in both.

        The distinction decides what is observable: if ``alpha == beta`` the half-sum
        ``(R_s + R_p) / 2`` is rigorously invariant, so a symmetric crosstalk is entirely
        invisible to an unpolarized measurement; if they differ, the residue is
        ``(alpha - beta)(R_p - R_s) / 2``.

        The mixing is applied to the **computed** spectra, before comparison. The
        measurements are never corrected: the instrument is modelled, which avoids
        amplifying the photometric noise and leaves the archived spectra untouched.

        These are instrument parameters, not sample parameters: one value of each for the
        whole study, shared by every polarization-resolved measurement.
    crosstalk_bounds:
        Bounds used when ``crosstalk_mode == "fitted"``.
    crosstalk_mode:
        ``"none"`` -- no leakage is modelled and none is released; ``"fitted"`` -- alpha and
        beta are released within ``crosstalk_bounds``.
    """

    name: str = "unspecified"
    beam_aperture_deg: float = 2.0
    aperture_band_edges_nm: tuple[float, ...] = ()
    aperture_per_band_deg: tuple[float, ...] | None = None
    aperture_mode: Literal["imposed", "fitted"] = "imposed"
    aperture_bounds_deg: tuple[float, float] = (1.0, 2.5)
    aperture_min_angle_deg: float = 10.0
    n_aperture_nodes: int = 2
    crosstalk_alpha: float = 0.0
    crosstalk_beta: float = 0.0
    crosstalk_bounds: tuple[float, float] = (0.0, 0.15)
    crosstalk_mode: Literal["none", "fitted"] = "none"
    comment: str = ""

    def __post_init__(self) -> None:
        edges = tuple(float(x) for x in self.aperture_band_edges_nm)
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(
                f"aperture band edges must be strictly increasing, got {edges!r}"
            )
        self.aperture_band_edges_nm = edges
        if self.aperture_per_band_deg is not None:
            per_band = tuple(float(x) for x in self.aperture_per_band_deg)
            if len(per_band) != len(edges) + 1:
                raise ValueError(
                    f"{len(per_band)} aperture values for {len(edges)} band edges; "
                    f"expected {len(edges) + 1}"
                )
            self.aperture_per_band_deg = per_band
        if self.n_aperture_nodes < 2:
            raise ValueError("n_aperture_nodes must be at least 2")
        if self.n_aperture_nodes > 2 and self.n_aperture_nodes % 2 == 0:
            raise ValueError(
                "a quadrature over the cone needs an odd number of nodes (3, 5, 7, ...); "
                "n_aperture_nodes = 2 selects the historical two-angle mean"
            )
        lo, hi = self.crosstalk_bounds
        for name, value in (("alpha", self.crosstalk_alpha), ("beta", self.crosstalk_beta)):
            if not (lo - 1e-12 <= value <= hi + 1e-12):
                raise ValueError(
                    f"crosstalk {name} = {value} lies outside its bounds {self.crosstalk_bounds}"
                )
        if self.crosstalk_alpha + self.crosstalk_beta >= 1.0:
            raise ValueError(
                "crosstalk alpha + beta must stay below 1: at unity the two channels carry "
                "the same mixture and the polarization information is gone"
            )

    @property
    def n_bands(self) -> int:
        return len(self.aperture_band_edges_nm) + 1

    def aperture_values_deg(self) -> np.ndarray:
        """Total aperture per band, in degrees, one value per band."""
        if self.aperture_per_band_deg is not None:
            return np.asarray(self.aperture_per_band_deg, dtype=np.float64)
        return np.full(self.n_bands, float(self.beam_aperture_deg), dtype=np.float64)

    def band_index(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """Index of the aperture band each wavelength falls in.

        Boundaries are the declared switchover wavelengths themselves -- not midpoints
        between interpolation nodes. A point exactly at an edge belongs to the upper band.
        """
        w = np.asarray(wavelength_nm, dtype=np.float64)
        edges = np.asarray(self.aperture_band_edges_nm, dtype=np.float64)
        if edges.size == 0:
            return np.zeros(w.shape, dtype=np.int64)
        return np.searchsorted(edges, w, side="right").astype(np.int64)

    def aperture_for(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """Total aperture, in degrees, at each wavelength."""
        return self.aperture_values_deg()[self.band_index(wavelength_nm)]


# ---------------------------------------------------------------------------
# Free parameters
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FreeParameters:
    """Declaration of which parameter blocks the inversion is allowed to move.

    This is the centrepiece of the protocol. A reverse-engineering procedure that is free to
    reshape ``n(lambda)`` cannot, by construction, establish that a previously determined
    ``n(lambda)`` was correct. The only way to make the distinction auditable is to declare
    the released blocks up front and to have the solver refuse anything else -- which is
    what :func:`certus_re.dof.count_free_parameters` and the run report enforce.

    Parameters
    ----------
    thicknesses:
        Names of the stacks whose layer thicknesses are released. ``"*"`` releases every
        stack of the study.
    index_correction:
        ``"none"``     -- tabulated indices are used as given, nothing is adjusted;
        ``"bounded"``  -- a correction to Re(n) is released per material, confined to a tube
        ``|n - n_tab| <= delta`` set by :attr:`index_tube_delta`;
        ``"free"``     -- unconstrained correction. Reported, but it forfeits any claim that
        the tabulated indices were validated.
    index_tube_delta:
        Half-width of the tube, in index units, when ``index_correction == "bounded"``.
    index_n_knots:
        Number of spline knots per material when an index correction is released.
    aperture:
        ``"imposed"`` or ``"fitted"``, mirroring :attr:`Instrument.aperture_mode`.
    substrate_index:
        ``"literature"`` -- propagated, never adjusted. An index optically in series with
        the whole stack would absorb the errors of the layers, reintroducing at the
        substrate level the circularity the protocol sets out to avoid. ``"cauchy"``
        releases three Cauchy coefficients and is provided only to quantify that effect.
    angle_offset:
        ``"nominal"`` -- the angle of incidence is the nominal one; ``"fitted"`` releases one
        offset per distinct angle. At 45 degrees the retrieved thickness moves by about
        35 nm per degree, so this block is the most dangerous of all: released, it will
        absorb almost any model error.
    """

    thicknesses: tuple[str, ...] = ("*",)
    index_correction: Literal["none", "bounded", "free"] = "none"
    index_tube_delta: float = 0.0
    index_n_knots: int = 0
    aperture: Literal["imposed", "fitted"] = "imposed"
    crosstalk: Literal["none", "fitted"] = "none"
    substrate_index: Literal["literature", "cauchy"] = "literature"
    angle_offset: Literal["nominal", "fitted"] = "nominal"

    def __post_init__(self) -> None:
        if self.index_correction == "bounded" and self.index_tube_delta <= 0.0:
            raise ValueError(
                'index_correction="bounded" requires a positive index_tube_delta'
            )
        if self.index_correction != "none" and self.index_n_knots <= 0:
            raise ValueError(
                f"index_correction={self.index_correction!r} requires index_n_knots > 0"
            )

    def releases_thickness(self, stack_name: str) -> bool:
        return "*" in self.thicknesses or stack_name in self.thicknesses


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Study:
    """A complete reverse-engineering problem: what was made, what was measured, what moves.

    Attributes
    ----------
    materials:
        Layer materials, by key, as tabulated dispersions. Values are
        :class:`certus_re.dispersion.TabulatedIndex` instances.
    substrates:
        Substrate materials, by key, same type.
    stacks:
        Deposition runs, by name.
    samples:
        Physical specimens.
    """

    name: str
    lambda0_nm: float
    materials: dict[str, object]
    substrates: dict[str, object]
    stacks: dict[str, Stack]
    samples: list[Sample]
    instrument: Instrument = field(default_factory=Instrument)
    free: FreeParameters = field(default_factory=FreeParameters)
    comment: str = ""

    # -- consistency ------------------------------------------------------

    def validate(self) -> list[str]:
        """Check the study for internal consistency.

        Returns a list of human-readable problems. An empty list means the study can be
        inverted as declared. Problems are returned rather than raised so that a caller can
        report all of them at once instead of one per run.
        """
        problems: list[str] = []

        for stack in self.stacks.values():
            for i, layer in enumerate(stack.layers, start=1):
                if layer.material not in self.materials:
                    problems.append(
                        f"stack {stack.name!r} layer {i} uses material "
                        f"{layer.material!r}, which is not declared"
                    )

        seen: set[str] = set()
        for sample in self.samples:
            if sample.name in seen:
                problems.append(f"duplicate sample name {sample.name!r}")
            seen.add(sample.name)
            if sample.substrate.material not in self.substrates:
                problems.append(
                    f"sample {sample.name!r} uses substrate "
                    f"{sample.substrate.material!r}, which is not declared"
                )
            for stack_name in sample.stack_names:
                if stack_name not in self.stacks:
                    problems.append(
                        f"sample {sample.name!r} references stack {stack_name!r}, "
                        f"which is not declared"
                    )
            if not sample.measurements:
                problems.append(f"sample {sample.name!r} carries no measurement")
            for m in sample.measurements:
                if m.quantity == "Trel" and sample.substrate.rear == "coated":
                    problems.append(
                        f"sample {sample.name!r}: relative transmittance is defined "
                        f"against a bare substrate and is meaningless for a two-side-"
                        f"coated component"
                    )
                if m.quantity in ("T", "Trel") and sample.substrate.rear == "none":
                    problems.append(
                        f"sample {sample.name!r}, measurement {m.label or m.quantity!r}: "
                        f"a transmittance is measured through a real plate, but the "
                        f'substrate rear face is declared "none" (semi-infinite); the '
                        f"model would return the flux entering the substrate, not the "
                        f"flux leaving the sample"
                    )
                # R and T are fractions of the incident flux and cannot exceed one.
                # Relative transmittance is a ratio to the bare substrate and legitimately
                # exceeds one whenever the layer antireflects the substrate -- which is the
                # normal case for a low-index film. Two is still far beyond anything
                # physical and catches a percent-valued column.
                upper = 2.0 if m.quantity == "Trel" else 1.0
                if np.nanmin(m.value) < -1e-9 or np.nanmax(m.value) > upper + 1e-9:
                    problems.append(
                        f"sample {sample.name!r}, measurement {m.label or m.quantity!r}: "
                        f"values outside [0, {upper:g}] -- percent not converted to "
                        f"fraction?"
                    )

        # A dispersion table that does not span the measured range is the most damaging
        # silent failure available here: interpolation clamps to the edge value, the model
        # keeps running, and the residual is wrong by whatever the dispersion does outside
        # the table. It is checked rather than trusted.
        for sample in self.samples:
            used = [self.substrates.get(sample.substrate.material)]
            for stack_name in sample.stack_names:
                stack = self.stacks.get(stack_name)
                if stack is None:
                    continue
                used.extend(self.materials.get(m) for m in set(stack.materials))
            for m in sample.measurements:
                lo, hi = float(m.wavelength_nm[0]), float(m.wavelength_nm[-1])
                for index in used:
                    if index is None:
                        continue
                    t_lo, t_hi = index.span_nm  # type: ignore[attr-defined]
                    if lo < t_lo - 1e-9 or hi > t_hi + 1e-9:
                        problems.append(
                            f"sample {sample.name!r}, measurement "
                            f"{m.label or m.quantity!r}: spans {lo:.1f}-{hi:.1f} nm but "
                            f"the dispersion of "
                            f"{getattr(index, 'name', '?')!r} is tabulated only over "
                            f"{t_lo:.1f}-{t_hi:.1f} nm; outside it the value would be "
                            f"held constant at the edge"
                        )

        for name in self.free.thicknesses:
            if name != "*" and name not in self.stacks:
                problems.append(
                    f"free parameters release thicknesses of stack {name!r}, "
                    f"which is not declared"
                )

        if self.free.aperture != self.instrument.aperture_mode:
            problems.append(
                f"free parameters declare aperture={self.free.aperture!r} while the "
                f"instrument declares aperture_mode="
                f"{self.instrument.aperture_mode!r}; the two must agree"
            )

        if len(self.samples) > 1:
            undeclared = [
                f"{s.name}/{m.label or m.quantity}"
                for s in self.samples
                for m in s.measurements
                if m.sigma is None
            ]
            if undeclared:
                problems.append(
                    "joint inversion without declared uncertainties on "
                    + ", ".join(undeclared)
                    + " -- datasets would be weighted by their point count instead of by "
                    "what they are worth"
                )

        return problems

    # -- convenience ------------------------------------------------------

    @property
    def n_points(self) -> int:
        return sum(s.n_points for s in self.samples)

    def used_stacks(self) -> list[str]:
        """Stacks actually referenced by at least one sample, in declaration order."""
        used = {name for s in self.samples for name in s.stack_names}
        return [name for name in self.stacks if name in used]

    def samples_using(self, stack_name: str) -> list[Sample]:
        """Samples whose front or rear face carries the named stack."""
        return [s for s in self.samples if stack_name in s.stack_names]

    def wavelength_span_nm(self) -> tuple[float, float]:
        """Smallest interval containing every measured point of the study."""
        lo = min(float(m.wavelength_nm[0]) for s in self.samples for m in s.measurements)
        hi = max(float(m.wavelength_nm[-1]) for s in self.samples for m in s.measurements)
        return lo, hi
