"""The study description: what it accepts, and what it refuses to let pass silently.

Most of these tests assert that a *wrong* study is rejected. That is deliberate. The failures
this package is built to prevent are not crashes: they are runs that complete and return a
plausible number computed from the wrong thing -- a percent column read as a fraction, a
dispersion held constant outside its table, a free parameter declared but never released.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from certus_re.dof import count_free_parameters
from certus_re.dispersion import TabulatedIndex, load_index_csv
from certus_re.io_study import StudyError, load_study, read_spectrum_csv
from certus_re.model import (
    FreeParameters,
    Instrument,
    Layer,
    Measurement,
    Sample,
    Stack,
    Substrate,
)


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------


def test_spectrum_is_read_by_column_name_and_comments_are_skipped(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text(
        "# provenance note\nWavelength_nm,R_a_pct,R_s_pct\n1000,50.0,60.0\n1010,51.0,61.0\n",
        encoding="utf-8",
    )
    w, v = read_spectrum_csv(path, "R_a_pct", units="percent")
    assert w.tolist() == [1000.0, 1010.0]
    assert v == pytest.approx([0.50, 0.51])


def test_a_missing_column_names_the_ones_available(tmp_path: Path):
    path = tmp_path / "s.csv"
    path.write_text("Wavelength_nm,R_a_pct\n1000,50.0\n", encoding="utf-8")
    with pytest.raises(StudyError, match="R_p_pct"):
        read_spectrum_csv(path, "R_p_pct")


def test_band_restriction_is_applied_on_load(tmp_path: Path):
    path = tmp_path / "s.csv"
    rows = "\n".join(f"{w},{50.0}" for w in range(1000, 2001, 10))
    path.write_text("Wavelength_nm,R\n" + rows + "\n", encoding="utf-8")
    w, _ = read_spectrum_csv(path, "R", units="percent", band_nm=(1200.0, 1400.0))
    assert w.min() == 1200.0 and w.max() == 1400.0


def test_index_columns_are_selected_by_name_not_position(tmp_path: Path):
    """Two workbooks of one campaign ordered their index columns differently; taking them
    by position produced a forty-percent residual. The loader refuses to guess by rank."""
    path = tmp_path / "i.csv"
    path.write_text("Wavelength_nm,k,n\n1000,0.001,2.2\n2000,0.002,2.1\n", encoding="utf-8")
    index = load_index_csv(path)
    assert index.n_at(1000.0) == pytest.approx(2.2)
    assert index.k_at(2000.0) == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def test_index_uses_the_n_plus_ik_storage_convention():
    index = TabulatedIndex(np.array([1000.0, 2000.0]), np.array([2.0, 2.0]), np.array([1e-3, 1e-3]))
    assert index.complex_at(1500.0)[0].imag == pytest.approx(1e-3)


def test_negative_k_is_refused():
    with pytest.raises(ValueError, match="negative k"):
        TabulatedIndex(np.array([1000.0, 2000.0]), np.array([2.0, 2.0]), np.array([-1e-3, 0.0]))


def test_extrapolation_is_reported_not_hidden():
    index = TabulatedIndex(
        np.array([1000.0, 2000.0]), np.array([2.0, 1.9]), name="X", valid_range_nm=(1000.0, 1800.0)
    )
    index.complex_at(np.array([900.0, 1500.0]))
    report = index.extrapolation_report()
    assert report is not None
    assert report["points_outside_table"] == 1


def test_a_query_inside_the_table_reports_nothing():
    index = TabulatedIndex(np.array([1000.0, 2000.0]), np.array([2.0, 1.9]))
    index.complex_at(np.array([1500.0]))
    assert index.extrapolation_report() is None


# ---------------------------------------------------------------------------
# Model consistency
# ---------------------------------------------------------------------------


def _minimal_study(**overrides):
    from certus_re.model import Study

    index = TabulatedIndex(np.array([1000.0, 4000.0]), np.array([2.0, 2.0]), name="H")
    substrate = TabulatedIndex(np.array([1000.0, 4000.0]), np.array([3.43, 3.43]), name="sub")
    measurement = Measurement(
        quantity="R",
        angle_deg=8.0,
        polarization="a",
        wavelength_nm=np.array([1000.0, 2000.0, 3000.0]),
        value=np.array([0.3, 0.3, 0.3]),
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


def test_a_consistent_study_reports_no_problem():
    assert _minimal_study().validate() == []


def test_percent_values_are_caught():
    study = _minimal_study()
    study.samples[0].measurements[0].value = np.array([30.0, 30.0, 30.0])
    assert any("percent" in p for p in study.validate())


def test_relative_transmittance_may_exceed_one():
    study = _minimal_study()
    study.samples[0].measurements[0].quantity = "Trel"
    study.samples[0].measurements[0].value = np.array([1.03, 1.04, 1.02])
    assert study.validate() == []


def test_a_dispersion_narrower_than_the_measurement_is_caught():
    study = _minimal_study()
    study.materials["H"] = TabulatedIndex(
        np.array([1500.0, 2500.0]), np.array([2.0, 2.0]), name="H"
    )
    assert any("tabulated only over" in p for p in study.validate())


def test_a_transmittance_through_a_semi_infinite_substrate_is_caught():
    study = _minimal_study()
    study.samples[0].substrate = Substrate("sub", 1.0, "none")
    study.samples[0].measurements[0].quantity = "T"
    assert any("semi-infinite" in p for p in study.validate())


def test_a_coated_rear_face_needs_a_rear_stack():
    with pytest.raises(ValueError, match="names no rear stack"):
        Sample(name="s", substrate=Substrate("sub", 1.0, "coated"), front_stack="A")


def test_a_rear_stack_needs_a_coated_rear_face():
    with pytest.raises(ValueError, match="rear"):
        Sample(
            name="s",
            substrate=Substrate("sub", 1.0, "bare"),
            front_stack="A",
            rear_stack="B",
        )


def test_aperture_declaration_must_match_the_instrument():
    study = _minimal_study(free=FreeParameters(aperture="fitted"))
    assert any("must agree" in p for p in study.validate())


def test_a_joint_study_without_uncertainties_is_flagged():
    study = _minimal_study()
    second = Sample(
        name="s2",
        substrate=Substrate("sub", 1.0, "bare"),
        front_stack="A",
        measurements=[
            Measurement(
                quantity="R",
                angle_deg=8.0,
                polarization="a",
                wavelength_nm=np.array([1000.0, 2000.0]),
                value=np.array([0.3, 0.3]),
            )
        ],
    )
    study.samples.append(second)
    assert any("weighted by their point count" in p for p in study.validate())


def test_aperture_bands_are_delimited_by_the_declared_switchovers():
    instrument = Instrument(aperture_band_edges_nm=(2530.0, 3700.0))
    bands = instrument.band_index(np.array([1000.0, 2529.9, 2530.0, 3699.9, 3700.0, 5000.0]))
    assert bands.tolist() == [0, 0, 1, 1, 2, 2]


def test_aperture_values_are_one_per_band():
    instrument = Instrument(
        aperture_band_edges_nm=(2530.0, 3700.0), aperture_per_band_deg=(2.0, 1.8, 1.6)
    )
    assert instrument.aperture_for(np.array([1000.0, 3000.0, 4000.0])).tolist() == [2.0, 1.8, 1.6]


def test_a_wrong_number_of_aperture_values_is_refused():
    with pytest.raises(ValueError, match="aperture values"):
        Instrument(aperture_band_edges_nm=(2530.0,), aperture_per_band_deg=(2.0, 1.8, 1.6))


# ---------------------------------------------------------------------------
# Degrees of freedom
# ---------------------------------------------------------------------------


def test_the_budget_counts_only_what_is_declared():
    report = count_free_parameters(_minimal_study())
    assert report.n_free == 1
    assert {b.name for b in report.held()} == {
        "index correction",
        "beam aperture",
        "polarizer crosstalk",
        "substrate index",
        "angle of incidence",
    }


def test_held_blocks_state_why_they_are_zero():
    report = count_free_parameters(_minimal_study())
    statuses = {b.name: b.status for b in report.blocks}
    assert statuses["index correction"] == "tabulated"
    assert statuses["beam aperture"] == "imposed"
    assert statuses["polarizer crosstalk"] == "not modelled"
    assert statuses["substrate index"] == "literature"


def test_releasing_the_index_adds_knots_per_material():
    study = _minimal_study(
        free=FreeParameters(index_correction="bounded", index_tube_delta=0.01, index_n_knots=4)
    )
    report = count_free_parameters(study)
    assert report.n_free == 1 + 4


def test_a_bounded_index_needs_a_positive_tube():
    with pytest.raises(ValueError, match="positive index_tube_delta"):
        FreeParameters(index_correction="bounded", index_tube_delta=0.0, index_n_knots=4)


def test_points_per_parameter_is_reported():
    report = count_free_parameters(_minimal_study())
    assert report.points_per_parameter == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Shared coatings
# ---------------------------------------------------------------------------


def test_two_samples_naming_one_coating_share_its_unknowns(synthetic_study_dir: Path):
    study = load_study(synthetic_study_dir)
    assert [s.name for s in study.samples_using("A")] == ["plate", "both"]
    assert count_free_parameters(study).n_free == 4 + 2


def test_a_study_file_keeps_every_path_relative(synthetic_study_dir: Path):
    document = json.loads(synthetic_study_dir.read_text(encoding="utf-8"))
    text = json.dumps(document)
    assert ":\\" not in text and ":/" not in text
