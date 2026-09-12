"""The optical model, checked against results that can be written down in closed form.

Every assertion here has an answer known independently of this code: a Fresnel coefficient,
Brewster's angle, the reflectance of a quarter-wave layer, the absentee half-wave, energy
conservation, reciprocity. A transfer-matrix implementation that passes all of them is not
merely self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from certus_re.physics import (
    aperture_nodes,
    as_macleod,
    assemble_plate,
    coherent_rt,
    fresnel_rt,
    snell_cos,
)

WAVELENGTH = np.array([1000.0])
AIR = np.array([1.0 + 0j])
GLASS = np.array([1.5 + 0j])
NO_LAYERS = np.zeros((1, 0), dtype=complex)
NO_THICKNESS = np.zeros(0)


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


def test_macleod_conjugates_a_gain_convention_index():
    assert as_macleod(np.array([2.0 + 1e-3j]))[0].imag == pytest.approx(-1e-3)
    assert as_macleod(np.array([2.0 - 1e-3j]))[0].imag == pytest.approx(-1e-3)


def test_macleod_legacy_flag_leaves_the_sign_alone():
    value = as_macleod(np.array([2.0 + 1e-3j]), legacy_gain_sign=True)[0]
    assert value.imag == pytest.approx(+1e-3)


def test_snell_is_the_invariant_not_the_local_angle():
    # n sin(theta) is conserved: entering glass from air at 45 degrees gives
    # sin(theta_glass) = sin(45)/1.5.
    cos_glass = snell_cos(GLASS, np.sin(np.deg2rad(45.0)))[0]
    sin_glass = np.sqrt(1.0 - cos_glass**2)
    assert sin_glass.real == pytest.approx(np.sin(np.deg2rad(45.0)) / 1.5)


# ---------------------------------------------------------------------------
# Single interface
# ---------------------------------------------------------------------------


def test_fresnel_at_normal_incidence():
    R, T = coherent_rt(
        WAVELENGTH, NO_LAYERS, NO_THICKNESS, AIR, GLASS, 0.0, "s", reverse=True
    )
    assert R[0] == pytest.approx(((1.0 - 1.5) / (1.0 + 1.5)) ** 2)
    assert R[0] + T[0] == pytest.approx(1.0)


def test_p_polarization_vanishes_at_brewster_angle():
    brewster = np.degrees(np.arctan(1.5))
    R, _ = coherent_rt(
        WAVELENGTH,
        NO_LAYERS,
        NO_THICKNESS,
        AIR,
        GLASS,
        np.sin(np.radians(brewster)),
        "p",
        reverse=True,
    )
    assert R[0] == pytest.approx(0.0, abs=1e-15)


def test_total_internal_reflection_carries_no_flux():
    critical = np.degrees(np.arcsin(1.0 / 1.5))
    R, T = coherent_rt(
        WAVELENGTH,
        NO_LAYERS,
        NO_THICKNESS,
        GLASS,
        AIR,
        1.5 * np.sin(np.radians(critical + 5.0)),
        "s",
        reverse=False,
    )
    assert R[0] == pytest.approx(1.0)
    assert T[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Coherent stacks
# ---------------------------------------------------------------------------


def test_quarter_wave_layer_matches_the_closed_form():
    n_layer = 2.2
    thickness = 1000.0 / (4.0 * n_layer)
    R, T = coherent_rt(
        WAVELENGTH,
        np.full((1, 1), n_layer + 0j),
        np.array([thickness]),
        AIR,
        GLASS,
        0.0,
        "s",
        reverse=True,
    )
    expected = ((1.0 * 1.5 - n_layer**2) / (1.0 * 1.5 + n_layer**2)) ** 2
    assert R[0] == pytest.approx(expected)
    assert R[0] + T[0] == pytest.approx(1.0)


def test_half_wave_layer_is_absentee():
    n_layer = 2.2
    thickness = 2.0 * 1000.0 / (4.0 * n_layer)
    R, _ = coherent_rt(
        WAVELENGTH,
        np.full((1, 1), n_layer + 0j),
        np.array([thickness]),
        AIR,
        GLASS,
        0.0,
        "s",
        reverse=True,
    )
    assert R[0] == pytest.approx(((1.0 - 1.5) / (1.0 + 1.5)) ** 2)


@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("angle", [0.0, 8.0, 30.0, 45.0, 60.0, 80.0])
def test_energy_is_conserved_by_a_transparent_stack(polarization, angle):
    rng = np.random.default_rng(0)
    n_values = rng.uniform(1.4, 2.4, 7)
    thickness = rng.uniform(50.0, 500.0, 7)
    wavelengths = np.linspace(800.0, 4000.0, 40)
    R, T = coherent_rt(
        wavelengths,
        np.tile(n_values, (wavelengths.size, 1)).astype(complex),
        thickness,
        np.full(wavelengths.size, 1 + 0j),
        np.full(wavelengths.size, 3.43 + 0j),
        np.sin(np.radians(angle)),
        polarization,
        reverse=True,
    )
    assert np.max(np.abs(R + T - 1.0)) < 1e-12


@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("angle", [0.0, 45.0, 70.0])
def test_transmittance_is_the_same_in_both_directions(polarization, angle):
    """Reciprocity. It is what allows the incoherent cavity to be written with T T'."""
    rng = np.random.default_rng(1)
    n_values = rng.uniform(1.4, 2.4, 5)
    thickness = rng.uniform(50.0, 500.0, 5)
    wavelengths = np.linspace(800.0, 4000.0, 25)
    layers = np.tile(n_values, (wavelengths.size, 1)).astype(complex)
    air = np.full(wavelengths.size, 1 + 0j)
    substrate = np.full(wavelengths.size, 3.43 + 0j)
    _, forward = coherent_rt(
        wavelengths, layers, thickness, air, substrate,
        np.sin(np.radians(angle)), polarization, reverse=True,
    )
    _, backward = coherent_rt(
        wavelengths, layers, thickness, substrate, air,
        np.sin(np.radians(angle)), polarization, reverse=False,
    )
    assert np.max(np.abs(forward - backward)) < 1e-12


def test_layer_order_changes_the_reflectance():
    """A stack and its reverse must not be confused: R distinguishes them, T does not."""
    wavelengths = np.linspace(1000.0, 2000.0, 11)
    n_values = np.array([2.2, 1.46, 2.2])
    thickness = np.array([120.0, 300.0, 90.0])
    layers = np.tile(n_values, (wavelengths.size, 1)).astype(complex)
    air = np.full(wavelengths.size, 1 + 0j)
    substrate = np.full(wavelengths.size, 3.43 + 0j)
    R_direct, _ = coherent_rt(
        wavelengths, layers, thickness, air, substrate, 0.0, "s", reverse=True
    )
    R_reversed, _ = coherent_rt(
        wavelengths, layers[:, ::-1], thickness[::-1], air, substrate, 0.0, "s",
        reverse=True,
    )
    assert np.max(np.abs(R_direct - R_reversed)) > 1e-3


# ---------------------------------------------------------------------------
# Plate assembly
# ---------------------------------------------------------------------------


def test_bare_plate_matches_the_two_surface_formula():
    wavelengths = np.array([3000.0])
    n_substrate = np.array([3.43 + 0j])
    plate = assemble_plate(
        wavelengths,
        n_substrate=n_substrate,
        substrate_thickness_mm=1.0,
        angle_deg=0.0,
        polarization="s",
        rear="bare",
    )
    single = ((1.0 - 3.43) / (1.0 + 3.43)) ** 2
    expected = single + (1.0 - single) ** 2 * single / (1.0 - single * single)
    assert plate.R[0] == pytest.approx(expected)
    assert plate.R[0] + plate.T[0] == pytest.approx(1.0)


def test_a_rear_stack_of_zero_thickness_is_a_bare_rear_face():
    wavelengths = np.array([3000.0])
    n_substrate = np.array([3.43 + 0j])
    common = dict(
        n_substrate=n_substrate,
        substrate_thickness_mm=1.0,
        angle_deg=45.0,
        polarization="p",
    )
    coated = assemble_plate(
        wavelengths,
        n_rear_layers=np.full((1, 3), 2.0 + 0j),
        d_rear_nm=np.zeros(3),
        rear="coated",
        **common,
    )
    bare = assemble_plate(wavelengths, rear="bare", **common)
    assert coated.R[0] == pytest.approx(bare.R[0], abs=1e-15)
    assert coated.T[0] == pytest.approx(bare.T[0], abs=1e-15)


@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("angle", [0.0, 8.0, 45.0, 60.0])
def test_two_side_coated_plate_conserves_energy(polarization, angle):
    rng = np.random.default_rng(2)
    wavelengths = np.linspace(1500.0, 5000.0, 30)
    front = np.tile(rng.uniform(1.45, 2.3, 6), (wavelengths.size, 1)).astype(complex)
    rear = np.tile(rng.uniform(1.45, 2.3, 4), (wavelengths.size, 1)).astype(complex)
    plate = assemble_plate(
        wavelengths,
        n_substrate=np.full(wavelengths.size, 3.43 + 0j),
        substrate_thickness_mm=1.0,
        angle_deg=angle,
        polarization=polarization,
        n_front_layers=front,
        d_front_nm=rng.uniform(80.0, 800.0, 6),
        n_rear_layers=rear,
        d_rear_nm=rng.uniform(80.0, 800.0, 4),
        rear="coated",
    )
    assert np.max(np.abs(plate.R + plate.T - 1.0)) < 1e-12


def test_an_absorbing_substrate_removes_flux():
    wavelengths = np.array([3000.0])
    plate = assemble_plate(
        wavelengths,
        n_substrate=np.array([3.43 + 1e-5j]),
        substrate_thickness_mm=1.0,
        angle_deg=0.0,
        polarization="s",
        rear="bare",
    )
    assert 0.0 < plate.internal_transmittance[0] < 1.0
    assert plate.R[0] + plate.T[0] < 1.0


def test_semi_infinite_substrate_returns_nothing_from_the_rear():
    wavelengths = np.array([3000.0])
    common = dict(
        n_substrate=np.array([3.43 + 0j]),
        substrate_thickness_mm=1.0,
        angle_deg=8.0,
        polarization="s",
    )
    none = assemble_plate(wavelengths, rear="none", **common)
    bare = assemble_plate(wavelengths, rear="bare", **common)
    assert none.T[0] == pytest.approx(1.0 - none.R[0])
    assert none.R[0] < bare.R[0]


def test_energy_conservation_holds_with_the_gain_sign_flag_off_only():
    """The legacy convention creates energy; that is precisely why it is not the default."""
    wavelengths = np.linspace(1000.0, 4000.0, 7)
    layers = np.full((wavelengths.size, 4), 2.2 + 2e-3j)
    thickness = np.full(4, 500.0)
    args = (
        wavelengths,
        layers,
        thickness,
        np.full(wavelengths.size, 1 + 0j),
        np.full(wavelengths.size, 3.43 + 0j),
        np.sin(np.radians(45.0)),
        "s",
    )
    R_phys, T_phys = coherent_rt(*args, reverse=True)
    R_legacy, T_legacy = coherent_rt(*args, reverse=True, legacy_gain_sign=True)
    assert np.all(R_phys + T_phys <= 1.0 + 1e-12)
    assert np.any(R_legacy + T_legacy > 1.0 + 1e-3)


# ---------------------------------------------------------------------------
# Beam aperture
# ---------------------------------------------------------------------------


def test_two_angle_rule_is_the_pair_of_extreme_rays():
    offsets, weights = aperture_nodes(2.0, 2)
    assert offsets == pytest.approx([-1.0, 1.0])
    assert weights == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("n_nodes", [3, 5, 9])
def test_quadrature_weights_sum_to_one_and_stay_inside_the_cone(n_nodes):
    offsets, weights = aperture_nodes(2.0, n_nodes)
    assert weights.sum() == pytest.approx(1.0)
    assert np.abs(offsets).max() <= 1.0


def test_an_even_node_count_above_two_is_refused():
    with pytest.raises(ValueError):
        aperture_nodes(2.0, 4)
