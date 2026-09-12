"""Forward optical model: coherent stacks, an incoherent substrate, a finite beam aperture.

The model computes what a spectrophotometer measures on a coated plate: a coherent multilayer
on the entrance face, a substrate thick enough that no interference survives inside it, and
an exit face that is either bare or carries a second coherent multilayer. Both faces are
handled on the same footing, which is what makes a two-side-coated component computable
rather than a special case.

Conventions, stated once and used everywhere
--------------------------------------------
Layer order
    ``layers[0]`` is the layer **adjacent to the substrate**, i.e. the first one deposited.
    This is the order a design is written in and the order a monitoring sequence follows.
Refractive index
    ``N = n - i k`` with ``k >= 0`` inside the model (Macleod). Tabulated data are supplied
    as ``n + i k`` and conjugated on entry, once, by :func:`as_macleod`.
Phase
    ``delta = 2 pi N d cos(theta) / lambda``, with the characteristic matrix
    ``[[cos d, i sin d / eta], [i eta sin d, cos d]]``.
Tilted admittance
    ``eta = N cos(theta)`` for s polarization, ``eta = N / cos(theta)`` for p.
Angles
    Snell's law is applied with the complex layer index; the incident and exit media use the
    real part of their index, absorption in the substrate being carried by an explicit path
    factor rather than by its admittance.

Energy
------
For a non-absorbing assembly ``R + T = 1`` to machine precision, at every angle and in both
polarizations. The test suite asserts it rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "as_macleod",
    "snell_cos",
    "admittance",
    "stack_matrix",
    "coherent_rt",
    "fresnel_rt",
    "assemble_plate",
    "PlateResult",
    "aperture_nodes",
]


# ---------------------------------------------------------------------------
# Elementary quantities
# ---------------------------------------------------------------------------


def as_macleod(n_complex: np.ndarray, *, legacy_gain_sign: bool = False) -> np.ndarray:
    """Return the index in the ``n - i k`` convention used throughout the model.

    Tabulated dispersions are conventionally stored as ``n + i k`` with ``k >= 0``. With the
    ``+i`` phase convention of the characteristic matrix, that sign describes a medium with
    **gain**: the wave grows as it propagates, and the assembly returns ``R + T > 1``.
    Conjugating here, once, at the boundary of the model, is what keeps every downstream
    formula in one convention.

    Parameters
    ----------
    legacy_gain_sign:
        Reproduce the behaviour of the reference implementation, which feeds ``n + i k``
        into a kernel written for ``n - i k`` and therefore amplifies instead of absorbing.
        Provided for one purpose only: showing that this package reproduces that
        implementation to machine precision, so that the two can be compared term by term.
        It must not be used to produce results. See ``tools/compare_with_reference.py``.
    """
    arr = np.asarray(n_complex)
    if not np.iscomplexobj(arr):
        return arr.astype(np.complex128)
    if legacy_gain_sign:
        return arr.astype(np.complex128)
    return np.where(arr.imag > 0.0, arr.conj(), arr).astype(np.complex128)


def snell_cos(n_medium: np.ndarray, sin_theta_incident: float | np.ndarray) -> np.ndarray:
    """Cosine of the propagation angle in a medium, from the invariant ``n sin(theta)``.

    ``sin_theta_incident`` is the invariant ``n_0 sin(theta_0)`` of the whole assembly, which
    is conserved across every interface. The square root is taken in the complex plane, so
    total internal reflection and absorbing media are handled without a branch.
    """
    n = np.asarray(n_medium, dtype=np.complex128)
    s = np.asarray(sin_theta_incident, dtype=np.complex128)
    sin_theta = s / n
    return np.sqrt(1.0 - sin_theta * sin_theta)


def admittance(n_medium: np.ndarray, cos_theta: np.ndarray, polarization: str) -> np.ndarray:
    """Tilted optical admittance, in units of the free-space admittance.

    Only the ratio of admittances enters the reflectance, so the free-space factor cancels
    and is omitted.
    """
    n = np.asarray(n_medium, dtype=np.complex128)
    c = np.asarray(cos_theta, dtype=np.complex128)
    if polarization == "s":
        return n * c
    if polarization == "p":
        return n / c
    raise ValueError(f"admittance is defined for 's' or 'p', not {polarization!r}")


# ---------------------------------------------------------------------------
# Coherent stack
# ---------------------------------------------------------------------------


def stack_matrix(
    wavelength_nm: np.ndarray,
    n_layers: np.ndarray,
    d_nm: np.ndarray,
    sin_theta_incident: float | np.ndarray,
    polarization: str,
    *,
    reverse: bool,
    legacy_gain_sign: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Characteristic matrix of a coherent stack, for every wavelength at once.

    Parameters
    ----------
    wavelength_nm:
        Wavelengths, shape ``(W,)``.
    n_layers:
        Complex indices, shape ``(W, L)``, column ``j`` being ``layers[j]`` -- that is,
        column 0 is the layer adjacent to the substrate.
    d_nm:
        Physical thicknesses, shape ``(L,)``, same ordering.
    sin_theta_incident:
        The invariant ``n_0 sin(theta_0)``.
    polarization:
        ``"s"`` or ``"p"``.
    reverse:
        ``False`` when light enters from the substrate side, so that ``layers[0]`` is the
        first layer met; ``True`` when light enters from the air side, so that
        ``layers[-1]`` is met first. The characteristic matrix is the ordered product of the
        layer matrices *in the order the light meets them*, so this flag is the only
        difference between illuminating a stack from the front or from the back.

    Returns
    -------
    (M00, M01, M10, M11)
        Each of shape ``(W,)``.
    """
    n = as_macleod(
        np.asarray(n_layers, dtype=np.complex128), legacy_gain_sign=legacy_gain_sign
    )
    d = np.asarray(d_nm, dtype=np.float64)
    w = np.asarray(wavelength_nm, dtype=np.float64)
    if n.ndim != 2:
        raise ValueError(f"n_layers must be (n_wavelengths, n_layers), got {n.shape}")
    if n.shape[0] != w.size:
        raise ValueError(f"{n.shape[0]} index rows for {w.size} wavelengths")
    if n.shape[1] != d.size:
        raise ValueError(f"{n.shape[1]} index columns for {d.size} thicknesses")

    # The characteristic matrix is the product taken from the incident medium towards the
    # exit medium. Building it by multiplying each new layer on the left means applying the
    # factors in the reverse of their position in that product: the layer adjacent to the
    # *exit* medium is applied first, the one adjacent to the incident medium last.
    #
    #   reverse=True  (light from the air side) : product L[-1] ... L[0], apply 0 -> L-1
    #   reverse=False (light from the substrate): product L[0] ... L[-1], apply L-1 -> 0
    order = range(n.shape[1]) if reverse else range(n.shape[1] - 1, -1, -1)

    one = np.ones(w.size, dtype=np.complex128)
    zero = np.zeros(w.size, dtype=np.complex128)
    M00, M01, M10, M11 = one, zero, zero, one

    two_pi_over_lambda = (2.0 * np.pi / w).astype(np.complex128)
    for j in order:
        n_j = n[:, j]
        cos_j = snell_cos(n_j, sin_theta_incident)
        eta_j = admittance(n_j, cos_j, polarization)
        delta = two_pi_over_lambda * n_j * d[j] * cos_j
        cos_d = np.cos(delta)
        i_sin_d = 1j * np.sin(delta)
        # A layer of vanishing admittance would divide by zero; it is also physically
        # meaningless, so it is reported rather than silently regularized.
        if np.any(np.abs(eta_j) < 1e-30):
            raise ValueError(
                f"layer {j}: vanishing optical admittance -- check the index table"
            )
        L01 = i_sin_d / eta_j
        L10 = i_sin_d * eta_j
        # New = L @ M, the incoming layer multiplying on the left.
        M00, M01, M10, M11 = (
            cos_d * M00 + L01 * M10,
            cos_d * M01 + L01 * M11,
            L10 * M00 + cos_d * M10,
            L10 * M01 + cos_d * M11,
        )
    return M00, M01, M10, M11


def coherent_rt(
    wavelength_nm: np.ndarray,
    n_layers: np.ndarray,
    d_nm: np.ndarray,
    n_incident: np.ndarray,
    n_exit: np.ndarray,
    sin_theta_incident: float | np.ndarray,
    polarization: str,
    *,
    reverse: bool,
    legacy_gain_sign: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflectance and transmittance of a coherent stack between two semi-infinite media.

    The admittances of the incident and exit media are formed from the **real part** of
    their indices. Inside this function the two media are half-spaces; any absorption they
    carry belongs to the incoherent assembly, where it appears as a path factor, and
    counting it twice would be wrong.

    Returns
    -------
    (R, T)
        Power reflectance and transmittance, each of shape ``(W,)``.
    """
    n_inc = np.real(np.asarray(n_incident, dtype=np.complex128)).astype(np.complex128)
    n_ext = np.real(np.asarray(n_exit, dtype=np.complex128)).astype(np.complex128)
    cos_inc = snell_cos(n_inc, sin_theta_incident)
    cos_ext = snell_cos(n_ext, sin_theta_incident)
    eta_inc = admittance(n_inc, cos_inc, polarization)
    eta_ext = admittance(n_ext, cos_ext, polarization)

    if np.asarray(d_nm).size == 0:
        return fresnel_rt(eta_inc, eta_ext)

    M00, M01, M10, M11 = stack_matrix(
        wavelength_nm,
        n_layers,
        d_nm,
        sin_theta_incident,
        polarization,
        reverse=reverse,
        legacy_gain_sign=legacy_gain_sign,
    )
    B = M00 + M01 * eta_ext
    C = M10 + M11 * eta_ext
    denominator = eta_inc * B + C
    r = (eta_inc * B - C) / denominator
    t = 2.0 * eta_inc / denominator
    R = np.real(r * np.conj(r))
    T = np.real(eta_ext) / np.real(eta_inc) * np.real(t * np.conj(t))
    return _clip01(R), _clip01(T)


def fresnel_rt(eta_incident: np.ndarray, eta_exit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflectance and transmittance of a single interface, from tilted admittances.

    Beyond the critical angle the exit admittance is imaginary, ``|r|`` is one and ``T`` is
    zero; the complex arithmetic delivers that without a special case.
    """
    eta_i = np.asarray(eta_incident, dtype=np.complex128)
    eta_e = np.asarray(eta_exit, dtype=np.complex128)
    r = (eta_i - eta_e) / (eta_i + eta_e)
    R = np.real(r * np.conj(r))
    T = 1.0 - R
    # An evanescent exit medium carries no flux; guard the rounding.
    T = np.where(np.abs(np.imag(eta_e)) > 1e-12 * np.abs(eta_e), 0.0, T)
    return _clip01(R), _clip01(T)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Plate assembly
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PlateResult:
    """Reflectance and transmittance of the assembled plate, plus its internal pieces.

    The intermediate quantities are kept because they are what one inspects when a result
    looks wrong: ``R_front`` alone tells whether the coating or the rear face is at fault.
    """

    R: np.ndarray
    T: np.ndarray
    R_front: np.ndarray
    T_front: np.ndarray
    R_front_from_substrate: np.ndarray
    T_front_from_substrate: np.ndarray
    R_rear: np.ndarray
    T_rear: np.ndarray
    internal_transmittance: np.ndarray


def assemble_plate(
    wavelength_nm: np.ndarray,
    *,
    n_substrate: np.ndarray,
    substrate_thickness_mm: float,
    angle_deg: float,
    polarization: str,
    n_front_layers: np.ndarray | None = None,
    d_front_nm: np.ndarray | None = None,
    n_rear_layers: np.ndarray | None = None,
    d_rear_nm: np.ndarray | None = None,
    rear: str = "bare",
    n_incident: float = 1.0,
    legacy_gain_sign: bool = False,
) -> PlateResult:
    """Reflectance and transmittance of a coated plate, substrate treated incoherently.

    The plate is: incident medium, coherent front stack, substrate, rear face, incident
    medium again. The substrate is thick, so the multiple reflections inside it are summed
    in intensity, not in amplitude:

    .. math::

        R = R_f + \\frac{T_f T_f' R_b \\tau^2}{1 - R_f' R_b \\tau^2}, \\qquad
        T = \\frac{T_f T_b \\tau}{1 - R_f' R_b \\tau^2}

    where primed quantities are the front stack seen from inside the substrate and
    :math:`\\tau` is the single-pass internal transmittance of the slab. This is the exact
    incoherent cavity, not the sum ``R_f + R_b``.

    Parameters
    ----------
    rear:
        ``"bare"`` -- uncoated rear face, a Fresnel interface with the incident medium;
        ``"coated"`` -- the rear face carries the stack given by ``n_rear_layers`` and
        ``d_rear_nm``, deposited on the substrate, so its ``layers[0]`` is adjacent to the
        substrate exactly as on the front face;
        ``"none"`` -- semi-infinite substrate: nothing returns from the rear at all.
    """
    w = np.asarray(wavelength_nm, dtype=np.float64)
    n_sub = np.asarray(n_substrate, dtype=np.complex128)
    if n_sub.size == 1:
        n_sub = np.full(w.size, complex(n_sub.reshape(-1)[0]))
    n_inc = np.full(w.size, complex(n_incident, 0.0))

    sin_theta_incident = float(n_incident) * np.sin(np.deg2rad(float(angle_deg)))

    empty_n = np.zeros((w.size, 0), dtype=np.complex128)
    empty_d = np.zeros(0, dtype=np.float64)
    nf = empty_n if n_front_layers is None else np.asarray(n_front_layers, dtype=np.complex128)
    df = empty_d if d_front_nm is None else np.asarray(d_front_nm, dtype=np.float64)

    # Front face, illuminated from the incident medium: the light meets layers[-1] first.
    R_f, T_f = coherent_rt(
        w,
        nf,
        df,
        n_inc,
        n_sub,
        sin_theta_incident,
        polarization,
        reverse=True,
        legacy_gain_sign=legacy_gain_sign,
    )
    # Front face seen from inside the substrate: the light meets layers[0] first.
    R_fp, T_fp = coherent_rt(
        w,
        nf,
        df,
        n_sub,
        n_inc,
        sin_theta_incident,
        polarization,
        reverse=False,
        legacy_gain_sign=legacy_gain_sign,
    )

    # Rear face, illuminated from inside the substrate.
    if rear == "none":
        R_b = np.zeros(w.size)
        T_b = np.ones(w.size)
    elif rear == "bare":
        cos_sub = snell_cos(np.real(n_sub).astype(np.complex128), sin_theta_incident)
        cos_out = snell_cos(n_inc, sin_theta_incident)
        eta_sub = admittance(np.real(n_sub).astype(np.complex128), cos_sub, polarization)
        eta_out = admittance(n_inc, cos_out, polarization)
        R_b, T_b = fresnel_rt(eta_sub, eta_out)
    elif rear == "coated":
        if n_rear_layers is None or d_rear_nm is None:
            raise ValueError('rear="coated" requires n_rear_layers and d_rear_nm')
        nr = np.asarray(n_rear_layers, dtype=np.complex128)
        dr = np.asarray(d_rear_nm, dtype=np.float64)
        # Deposited on the substrate, so layers[0] touches the substrate and is met first.
        R_b, T_b = coherent_rt(
            w,
            nr,
            dr,
            n_sub,
            n_inc,
            sin_theta_incident,
            polarization,
            reverse=False,
            legacy_gain_sign=legacy_gain_sign,
        )
    else:
        raise ValueError(f"unknown rear face {rear!r}")

    # Single-pass internal transmittance of the slab, along the tilted path. The magnitude
    # of the extinction coefficient is what matters here, so this line is independent of the
    # storage convention -- including in compatibility mode, where a substrate must still
    # absorb.
    k_sub = np.abs(np.imag(n_sub))
    cos_sub = snell_cos(np.real(n_sub).astype(np.complex128), sin_theta_incident)
    path_nm = float(substrate_thickness_mm) * 1.0e6 / np.maximum(np.real(cos_sub), 1e-12)
    tau = np.exp(-4.0 * np.pi * k_sub * path_nm / w)
    tau = np.clip(tau, 0.0, 1.0)

    denominator = 1.0 - R_fp * R_b * tau * tau
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)
    R_total = R_f + (T_f * T_fp * R_b * tau * tau) / denominator
    T_total = (T_f * T_b * tau) / denominator

    return PlateResult(
        R=_clip01(R_total),
        T=_clip01(T_total),
        R_front=R_f,
        T_front=T_f,
        R_front_from_substrate=R_fp,
        T_front_from_substrate=T_fp,
        R_rear=R_b,
        T_rear=T_b,
        internal_transmittance=tau,
    )


# ---------------------------------------------------------------------------
# Beam aperture
# ---------------------------------------------------------------------------


def aperture_nodes(
    total_aperture_deg: float, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Angular offsets and weights describing the finite beam cone.

    Two rules are available, and the choice is a modelling decision worth stating in a
    publication rather than burying.

    ``n_nodes = 2``
        The two extreme rays, ``theta -/+ h`` with equal weight. This is the historical
        approximation. Its error grows with the curvature of ``R(theta)``, and therefore
        differs between polarizations wherever that curvature does -- at 45 degrees,
        markedly.
    ``n_nodes`` odd, >= 3
        Gauss-Legendre quadrature of the mean of ``R`` over the angular interval, which
        converges to the true average over a beam whose angular distribution is uniform in
        the plane of incidence.

    Returns
    -------
    (offsets_deg, weights)
        Weights sum to one.
    """
    h = 0.5 * float(total_aperture_deg)
    if n_nodes == 2:
        return np.array([-h, +h]), np.array([0.5, 0.5])
    if n_nodes < 2 or n_nodes % 2 == 0:
        raise ValueError(
            f"n_nodes must be 2 or an odd integer >= 3, got {n_nodes}"
        )
    x, wgt = np.polynomial.legendre.leggauss(n_nodes)
    return h * x, wgt / 2.0
