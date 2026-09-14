"""The four cases the deliverable claims, asserted against the deposited studies.

The package is specified to invert a **single component or several at once**, each carrying a
coating on **one face or on both**.  That is a two-by-two matrix, and a claim in a README is
worth nothing unless something fails when it stops being true.  These tests find, for each
cell, a deposited study that exercises it, and check it inverts.

The one cell the campaign cannot fill with real data -- *several* two-side-coated components
inverted jointly, since only one such component was made -- is covered by the synthetic
fixture, where two samples share a coating and one of them is two-sided.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from certus_re.io_study import load_study
from certus_re.solve import invert

from .conftest import STUDIES


def _classify(study) -> tuple[str, str]:
    several = "several components" if len(study.samples) > 1 else "one component"
    two_sided = any(s.is_two_sided for s in study.samples)
    return several, "both faces coated" if two_sided else "one face coated"


def _deposited_studies():
    """The ladder, plus the worked examples that fill the cells the campaign's own
    components do not reach. Counter-experiments are excluded: they are wrong by
    construction and must not be picked up as a demonstration of anything."""
    for path in sorted(STUDIES.glob("*.json")) + sorted(STUDIES.glob("examples/*.json")):
        yield path, load_study(path, strict=False)


@pytest.mark.parametrize(
    "components,faces",
    [
        ("one component", "one face coated"),
        ("one component", "both faces coated"),
        ("several components", "one face coated"),
        ("several components", "both faces coated"),
    ],
)
def test_every_cell_of_the_matrix_has_a_deposited_study(components, faces):
    if not STUDIES.is_dir():
        pytest.skip("the deposited studies are not present")
    matching = [
        path.stem
        for path, study in _deposited_studies()
        if _classify(study) == (components, faces)
    ]
    assert matching, (
        f"no deposited study exercises '{components}, {faces}'; the package claims to "
        f"handle it, so something must demonstrate it"
    )


def test_the_two_sided_component_is_modelled_as_a_coherent_stack_on_each_face():
    """Not R_front + R_rear: an exact incoherent plate between two coherent stacks."""
    path = STUDIES / "examples" / "two_faces_coated.json"
    if not path.is_file():
        pytest.skip("the two-side-coated example is not present")
    study = load_study(path)
    sample = study.samples[0]
    assert sample.front_stack == "BS45" and sample.rear_stack == "AR6"
    assert sample.substrate.rear == "coated"
    # Twenty-two unknowns on one specimen: sixteen on the entrance face, six on the exit one.
    assert sum(study.stacks[n].n_layers for n in sample.stack_names) == 22


def test_several_components_share_the_coatings_they_have_in_common():
    """The point of a joint inversion: one set of unknowns, not one per sample."""
    path = STUDIES / "examples" / "several_components_two_faces.json"
    if not path.is_file():
        pytest.skip("the shared-coating example is not present")
    study = load_study(path)
    shared = {n: [s.name for s in study.samples_using(n)] for n in study.used_stacks()}
    assert shared["BS45"] == ["BS45_alone", "BIFACE_BSplusAR"]
    assert shared["AR6"] == ["AR6_alone", "BIFACE_BSplusAR"]
    # Three samples carrying 6 + 16 + (16 + 6) layers between them, but only 22 unknowns,
    # because the assembled component names coatings the others already name.
    assert sum(study.stacks[n].n_layers for n in study.used_stacks()) == 22


def test_the_joint_campaign_shares_only_the_dispersions():
    """The article's joint rung: three components that share no layer, only two dispersions.

    That is the whole question of the study. If the components shared thicknesses the joint
    residual would say something about the coatings; sharing only n(lambda) and k(lambda)
    makes it say something about the determination those came from.
    """
    path = STUDIES / "07_joint_campaign.json"
    if not path.is_file():
        pytest.skip("the joint study is not present")
    study = load_study(path)
    for name in study.used_stacks():
        assert len(study.samples_using(name)) == 1, f"{name} is carried by more than one sample"
    # Two reference single layers, six, sixteen and seventeen layers: forty-one unknowns for five samples.
    assert sum(study.stacks[n].n_layers for n in study.used_stacks()) == 41
    assert len(study.samples) == 5


def test_several_two_sided_components_invert_jointly(
    synthetic_study_dir: Path, synthetic_truth
):
    """The cell the campaign cannot fill: two samples, one two-sided, sharing a coating.

    Only one two-side-coated component was made, so this is demonstrated on data the package
    generated itself -- which is the honest way to claim a capability the campaign does not
    exercise.
    """
    study = load_study(synthetic_study_dir)
    assert len(study.samples) == 2
    assert any(s.is_two_sided for s in study.samples)
    assert study.samples_using("A") == study.samples  # the shared coating
    result = invert(study)
    for stack, truth in synthetic_truth.items():
        assert np.max(np.abs(result.thicknesses[stack] - truth)) < 1e-6
