"""The two instrument effects, and what makes each of them observable.

Both are applied to the **computed** spectrum and never to the measurement, so both can be
checked the same way: generate a spectrum from known instrument settings and require the
inversion to return them.

The centrepiece is the invariance test. If the two polarizer channels leak into each other by
the same amount, the half-sum ``(R_s + R_p) / 2`` is rigorously unchanged -- which means a
symmetric leakage is *invisible* to an unpolarized measurement, and that the leakage can only
be seen, or refuted, in the resolved channels. An implementation that got the mixing wrong
would break that invariance, so asserting it is how the wiring is checked rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from certus_re.dof import count_free_parameters
from certus_re.dispersion import TabulatedIndex
from certus_re.forward import Evaluator
from certus_re.io_study import load_study
from certus_re.model import (
    FreeParameters,
    Instrument,
    Layer,
    Measurement,
    Sample,
    Stack,
    Study,
    Substrate,
)
from certus_re.solve import ParameterLayout, invert


# ---------------------------------------------------------------------------
# Polarizer leakage: what it does, and what it cannot do
# ---------------------------------------------------------------------------


def test_a_symmetric_leakage_leaves_the_half_sum_rigorously_invariant(
    polarized_study_dir: Path,
):
    study = load_study(polarized_study_dir)
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    pure_s, pure_p = evaluator.predict(nominal, crosstalk=(0.0, 0.0))
    for leakage in (0.01, 0.05, 0.15):
        mixed_s, mixed_p = evaluator.predict(nominal, crosstalk=(leakage, leakage))
        assert np.max(np.abs((mixed_s + mixed_p) - (pure_s + pure_p))) < 1e-15


def test_an_asymmetric_leakage_shifts_it_by_exactly_the_predicted_residue(
    polarized_study_dir: Path,
):
    study = load_study(polarized_study_dir)
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    pure_s, pure_p = evaluator.predict(nominal, crosstalk=(0.0, 0.0))
    alpha, beta = 0.09, 0.02
    mixed_s, mixed_p = evaluator.predict(nominal, crosstalk=(alpha, beta))
    residue = 0.5 * (mixed_s + mixed_p) - 0.5 * (pure_s + pure_p)
    assert residue == pytest.approx((alpha - beta) * (pure_p - pure_s) / 2.0, abs=1e-15)
    # And it is not a small print: the shift must be large enough to matter.
    assert np.max(np.abs(residue)) > 1e-3


def test_the_mixing_is_the_declared_one_term_for_term(polarized_study_dir: Path):
    study = load_study(polarized_study_dir)
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    pure_s, pure_p = evaluator.predict(nominal, crosstalk=(0.0, 0.0))
    alpha, beta = 0.11, 0.03
    mixed_s, mixed_p = evaluator.predict(nominal, crosstalk=(alpha, beta))
    assert mixed_s == pytest.approx((1 - alpha) * pure_s + alpha * pure_p, abs=1e-15)
    assert mixed_p == pytest.approx((1 - beta) * pure_p + beta * pure_s, abs=1e-15)


def test_the_unpolarized_channel_carries_no_leakage(polarized_study_dir: Path):
    """A channel recorded without a polarizer cannot be contaminated by one."""
    study = load_study(polarized_study_dir)
    for measurement in study.samples[0].measurements:
        measurement.polarization = "a"
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    without = evaluator.predict(nominal, crosstalk=(0.0, 0.0))
    with_leak = evaluator.predict(nominal, crosstalk=(0.12, 0.01))
    for a, b in zip(without, with_leak):
        assert np.array_equal(a, b)


def test_the_leakage_is_recovered_from_the_spectra_it_produced(
    polarized_study_dir: Path, polarized_truth
):
    study = load_study(polarized_study_dir)
    # Start from a perfect polarizer, so that recovering the truth is a result and not the
    # initial value handed back.
    study.instrument.crosstalk_alpha = 0.0
    study.instrument.crosstalk_beta = 0.0
    study.instrument.beam_aperture_deg = 2.2
    result = invert(study)
    assert result.crosstalk[0] == pytest.approx(polarized_truth["alpha"], abs=1e-6)
    assert result.crosstalk[1] == pytest.approx(polarized_truth["beta"], abs=1e-6)
    assert result.aperture_deg[0] == pytest.approx(
        polarized_truth["aperture_deg"], abs=1e-5
    )


def test_the_leakage_is_two_parameters_shared_by_the_whole_study(
    polarized_study_dir: Path,
):
    study = load_study(polarized_study_dir)
    layout = ParameterLayout.build(study)
    # Five layers, one aperture band, one pair of leakage coefficients.
    assert layout.n_parameters == 5 + 1 + 2
    assert layout.n_parameters == count_free_parameters(study).n_free
    names = layout.parameter_names()
    assert names[-2:] == ["crosstalk alpha", "crosstalk beta"]


def test_the_budget_names_the_points_that_constrain_the_leakage(polarized_study_dir: Path):
    study = load_study(polarized_study_dir)
    block = next(
        b for b in count_free_parameters(study).blocks if b.name == "polarizer crosstalk"
    )
    assert block.count == 2
    assert str(study.n_polarized_points) in block.detail


# ---------------------------------------------------------------------------
# Beam aperture
# ---------------------------------------------------------------------------


def test_the_aperture_block_is_assembled_and_not_only_counted(polarized_study_dir: Path):
    """Declaring the aperture released used to count parameters that could not move."""
    study = load_study(polarized_study_dir)
    layout = ParameterLayout.build(study)
    assert layout.aperture_slice is not None
    lower, upper = layout.bounds(Evaluator(study).nominal_thicknesses(), 0.5)
    assert lower[layout.aperture_slice].tolist() == [1.0]
    assert upper[layout.aperture_slice].tolist() == [2.5]


def test_the_aperture_changes_the_prediction(polarized_study_dir: Path):
    study = load_study(polarized_study_dir)
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    narrow = evaluator.predict(nominal, aperture_deg=np.array([1.0]))
    wide = evaluator.predict(nominal, aperture_deg=np.array([2.5]))
    assert np.max(np.abs(narrow[0] - wide[0])) > 1e-4


def test_a_band_of_the_aperture_steps_where_the_instrument_switches(
    polarized_study_dir: Path,
):
    study = load_study(polarized_study_dir)
    study.instrument.aperture_band_edges_nm = (2530.0, 3700.0)
    study.instrument.aperture_per_band_deg = None
    layout = ParameterLayout.build(study)
    assert layout.aperture_slice is not None
    assert layout.aperture_slice.stop - layout.aperture_slice.start == 3
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    # Moving the aperture of the third band alone must move the points of that band alone.
    base = evaluator.predict(nominal, aperture_deg=np.array([2.0, 2.0, 2.0]))[0]
    moved = evaluator.predict(nominal, aperture_deg=np.array([2.0, 2.0, 1.0]))[0]
    wavelengths = study.samples[0].measurements[0].wavelength_nm
    assert np.array_equal(base[wavelengths < 3700.0], moved[wavelengths < 3700.0])
    assert not np.array_equal(base[wavelengths >= 3700.0], moved[wavelengths >= 3700.0])


def test_the_aperture_is_not_applied_at_near_normal_incidence(polarized_study_dir: Path):
    study = load_study(polarized_study_dir)
    for measurement in study.samples[0].measurements:
        measurement.angle_deg = 8.0
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    narrow = evaluator.predict(nominal, aperture_deg=np.array([1.0]))
    wide = evaluator.predict(nominal, aperture_deg=np.array([2.5]))
    for a, b in zip(narrow, wide):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


def _unpolarized_study(**overrides) -> Study:
    index = TabulatedIndex(np.array([1000.0, 4000.0]), np.array([2.0, 2.0]), name="H")
    substrate = TabulatedIndex(np.array([1000.0, 4000.0]), np.array([3.43, 3.43]), name="sub")
    measurement = Measurement(
        quantity="R",
        angle_deg=45.0,
        polarization="a",
        wavelength_nm=np.array([1000.0, 2000.0, 3000.0, 3900.0]),
        value=np.array([0.3, 0.3, 0.3, 0.3]),
        sigma=1e-3,
    )
    defaults = dict(
        name="t",
        lambda0_nm=1500.0,
        materials={"H": index},
        substrates={"sub": substrate},
        stacks={"A": Stack("A", [Layer("H", 1.0)])},
        samples=[
            Sample(
                name="s",
                substrate=Substrate("sub", 1.0, "bare"),
                front_stack="A",
                measurements=[measurement],
            )
        ],
        instrument=Instrument(),
        free=FreeParameters(),
    )
    defaults.update(overrides)
    return Study(**defaults)


def test_the_crosstalk_declaration_must_match_the_instrument():
    study = _unpolarized_study(free=FreeParameters(crosstalk="fitted"))
    assert any("must agree" in p for p in study.validate())


def test_leakage_released_against_unpolarized_data_alone_is_refused():
    study = _unpolarized_study(
        instrument=Instrument(crosstalk_mode="fitted"),
        free=FreeParameters(crosstalk="fitted"),
    )
    assert any("unconstrained" in p for p in study.validate())


def test_an_aperture_released_with_no_oblique_measurement_is_refused():
    study = _unpolarized_study(
        instrument=Instrument(aperture_mode="fitted"),
        free=FreeParameters(aperture="fitted"),
    )
    study.samples[0].measurements[0].angle_deg = 0.0
    assert any("aperture is released but no measurement" in p for p in study.validate())


def test_an_aperture_band_with_no_oblique_point_is_refused():
    """Three bands declared, one of them never measured: three parameters, two determined."""
    study = _unpolarized_study(
        instrument=Instrument(
            aperture_band_edges_nm=(2530.0, 3700.0), aperture_mode="fitted"
        ),
        free=FreeParameters(aperture="fitted"),
    )
    study.samples[0].measurements[0].wavelength_nm = np.array([1000.0, 2000.0])
    study.samples[0].measurements[0].value = np.array([0.3, 0.3])
    problems = study.validate()
    assert any("carry no oblique measurement" in p for p in problems)


def test_leakage_outside_its_bounds_is_refused():
    with pytest.raises(ValueError, match="outside its bounds"):
        Instrument(crosstalk_alpha=0.4)
