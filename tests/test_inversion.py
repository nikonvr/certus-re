"""The inversion, checked where the answer is known: on data it generated itself.

A round trip is the only test of an inverse problem that has no argument: take a set of
thicknesses, compute the spectrum they produce, hand that spectrum back, and require the
thicknesses to come out. Anything less than machine precision on noise-free data means
something in the chain is not consistent with itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from certus_re.forward import Evaluator
from certus_re.io_study import load_study
from certus_re.report import json_report, text_report
from certus_re.solve import ParameterLayout, invert


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_a_noise_free_round_trip_recovers_every_thickness(
    synthetic_study_dir: Path, synthetic_truth
):
    study = load_study(synthetic_study_dir)
    result = invert(study)
    for stack, truth in synthetic_truth.items():
        assert np.max(np.abs(result.thicknesses[stack] - truth)) < 1e-6
    # The residual is in units of the declared uncertainty; on data the model generated
    # itself it should be limited only by the arithmetic.
    for residual in result.residuals:
        assert residual.rms_final < 1e-10


def test_the_rear_coating_is_recovered_from_the_two_sided_sample_alone(
    synthetic_study_dir: Path, synthetic_truth
):
    """The two-side-coated sample constrains both of its coatings on its own."""
    study = load_study(synthetic_study_dir)
    study.samples = [s for s in study.samples if s.name == "both"]
    result = invert(study)
    for stack in ("A", "B"):
        assert np.max(np.abs(result.thicknesses[stack] - synthetic_truth[stack])) < 1e-6


def test_a_shared_coating_is_one_set_of_unknowns_not_two(
    synthetic_study_dir: Path, synthetic_truth
):
    study = load_study(synthetic_study_dir)
    layout = ParameterLayout.build(study)
    # Four layers in A and two in B: six unknowns for two samples, not ten.
    assert layout.n_parameters == 6
    result = invert(study)
    assert np.max(np.abs(result.thicknesses["A"] - synthetic_truth["A"])) < 1e-6


def test_noise_degrades_the_recovery_proportionately(synthetic_study_dir: Path, synthetic_truth):
    study = load_study(synthetic_study_dir)
    rng = np.random.default_rng(4)
    for sample in study.samples:
        for measurement in sample.measurements:
            measurement.value = np.clip(
                measurement.value + rng.normal(0.0, 5e-4, measurement.value.size), 0.0, 1.0
            )
    result = invert(study)
    error = np.abs(result.thicknesses["A"] - synthetic_truth["A"])
    # A photometric noise of 5e-4 leaves sub-nanometre errors; it must not leave none.
    assert 0.0 < error.max() < 2.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_runs_of_one_study_agree_bit_for_bit(synthetic_study_dir: Path):
    first = invert(load_study(synthetic_study_dir))
    second = invert(load_study(synthetic_study_dir))
    for stack in first.thicknesses:
        assert np.array_equal(first.thicknesses[stack], second.thicknesses[stack])
    assert first.rms_final == second.rms_final
    assert first.n_function_evaluations == second.n_function_evaluations


# ---------------------------------------------------------------------------
# Declared parameters are the ones that move
# ---------------------------------------------------------------------------


def test_a_coating_held_fixed_does_not_move(synthetic_study_dir: Path):
    study = load_study(synthetic_study_dir)
    study.free.thicknesses = ("A",)
    nominal = Evaluator(study).nominal_thicknesses()
    result = invert(study)
    assert np.array_equal(result.thicknesses["B"], nominal["B"])
    assert not np.array_equal(result.thicknesses["A"], nominal["A"])


def test_a_layer_marked_fixed_does_not_move(synthetic_study_dir: Path):
    import dataclasses

    study = load_study(synthetic_study_dir)
    layers = study.stacks["A"].layers
    layers[2] = dataclasses.replace(layers[2], variable=False)
    nominal = Evaluator(study).nominal_thicknesses()
    result = invert(study)
    assert result.thicknesses["A"][2] == pytest.approx(nominal["A"][2])


def test_the_budget_and_the_parameter_vector_cannot_disagree(synthetic_study_dir: Path):
    study = load_study(synthetic_study_dir)
    result = invert(study)
    assert ParameterLayout.build(study).n_parameters == result.dof.n_free


def test_an_index_correction_is_released_only_when_declared(synthetic_study_dir: Path):
    study = load_study(synthetic_study_dir)
    assert invert(study).index_corrections == {}
    study.free.index_correction = "bounded"
    study.free.index_tube_delta = 0.01
    study.free.index_n_knots = 3
    result = invert(study)
    assert set(result.index_corrections) == {"H", "L"}
    assert all(np.abs(v).max() <= 0.01 + 1e-9 for v in result.index_corrections.values())


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_a_thickness_stopped_by_its_bound_is_reported(synthetic_study_dir: Path):
    study = load_study(synthetic_study_dir)
    result = invert(study, thickness_tolerance=1e-4)
    assert result.at_bounds, "a search window this tight must pin several layers"


def test_the_report_states_what_was_released_and_what_was_not(synthetic_study_dir: Path):
    study = load_study(synthetic_study_dir)
    result = invert(study)
    text = text_report(result, study)
    assert "PARAMETERS RELEASED" in text
    assert "index correction" in text
    assert "COATINGS SHARED BETWEEN SAMPLES" in text
    document = json_report(result, study)
    assert document["free_parameters"]["n_free_parameters"] == 6
    assert document["diagnostics"]["deterministic"] is True
    assert document["shared_stacks"]["A"] == ["plate", "both"]


def test_qwot_departures_are_relative_to_each_solutions_own_nominal(synthetic_study_dir: Path):
    study = load_study(synthetic_study_dir)
    result = invert(study)
    for stack, departure in result.qwot_departure_pct().items():
        expected = 100.0 * (
            result.qwot[stack] - result.nominal_qwot[stack]
        ) / result.nominal_qwot[stack]
        assert departure == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The deposited studies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "00_prediction.json",
        "02_AR6_alone.json",
        "04_BS45_resolved.json",
        "08_joint_campaign.json",
    ],
)
def test_every_deposited_study_loads_and_is_consistent(name):
    from .conftest import deposited

    study = load_study(deposited(name))
    assert study.validate() == []
    assert study.n_points > 0


def test_the_deposited_antireflection_study_inverts(tmp_path):
    from .conftest import deposited

    study = load_study(deposited("02_AR6_alone.json"))
    result = invert(study)
    assert result.success
    assert result.residuals[0].rms_final < result.residuals[0].rms_initial
    assert result.dof.n_free == 6
