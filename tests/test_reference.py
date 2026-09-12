"""Non-regression against the reference implementation the coatings were inverted with.

This package's optical model was rewritten; these tests are what make that rewrite
defensible. They import the laboratory's production tool -- read-only, never modified -- and
require the two forward models to agree to machine precision on the real coatings.

The agreement is asserted in *compatibility mode*, in which this package reproduces the
reference's sign convention on the extinction coefficient. That convention is not the
physical one: fed an index stored as ``n + i k``, the reference kernel amplifies instead of
absorbing and returns ``R + T > 1``. Asserting agreement in compatibility mode, and
asserting non-conservation of energy in that same mode, pins down exactly one difference
between the two implementations and shows it is the only one.

The whole module is skipped when the reference is not installed, so the deposit remains
testable on a machine that only has the deposit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from certus_re.io_study import load_study
from certus_re.physics import coherent_rt

REFERENCE = Path(
    os.environ.get(
        "CERTUS_REFERENCE",
        r"D:\drivfl\01_Recherche_Enseignement\02_Couches_Minces_Optique"
        r"\couches minces 2026\CERTUS\0409",
    )
)

TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def reference_model():
    if not (REFERENCE / "certus").is_dir():
        pytest.skip(f"reference implementation not found at {REFERENCE}")
    os.environ.setdefault("CERTUS_RE_HEADLESS", "1")
    sys.path.insert(0, str(REFERENCE))
    try:
        from certus.physics.certus_tmm_oblique import (  # type: ignore
            calc_spectrum_oblique_vectorized,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import the reference forward model: {exc}")
    return calc_spectrum_oblique_vectorized


@pytest.fixture(scope="module")
def campaign():
    from .conftest import deposited

    study = load_study(deposited("08_joint_campaign.json"))
    return study


def _stack_arrays(study, stack_name: str):
    stack = study.stacks[stack_name]
    lambda0 = study.lambda0_nm
    thickness = np.asarray(
        [
            layer.thickness_nm(lambda0, float(study.materials[layer.material].n_at(lambda0)))
            for layer in stack.layers
        ]
    )
    sample = study.samples_using(stack_name)[0]
    wavelengths = sample.measurements[0].wavelength_nm
    layers = np.stack(
        [study.materials[m].complex_at(wavelengths) for m in stack.materials], axis=1
    )
    substrate = study.substrates[sample.substrate.material].complex_at(wavelengths)
    return wavelengths, layers, thickness, substrate


@pytest.mark.parametrize("stack_name", ["AR6", "BS45"])
@pytest.mark.parametrize("angle", [8.0, 45.0])
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_compatibility_mode_reproduces_the_reference(
    reference_model, campaign, stack_name, angle, polarization
):
    wavelengths, layers, thickness, substrate = _stack_arrays(campaign, stack_name)
    R_ref, T_ref = reference_model(
        wavelengths, layers, thickness, substrate, angle, polarization
    )
    R_new, T_new = coherent_rt(
        wavelengths,
        layers,
        thickness,
        np.ones(wavelengths.size, dtype=np.complex128),
        substrate,
        np.sin(np.deg2rad(angle)),
        polarization,
        reverse=True,
        legacy_gain_sign=True,
    )
    assert np.max(np.abs(R_ref - R_new)) < TOLERANCE
    assert np.max(np.abs(T_ref - T_new)) < TOLERANCE


def test_the_reference_convention_creates_energy(reference_model):
    """Why compatibility mode is not the default, stated as an assertion."""
    wavelengths = np.linspace(1000.0, 4000.0, 7)
    layers = np.full((wavelengths.size, 4), 2.2 + 2e-3j)  # stored as n + i k, k > 0
    thickness = np.full(4, 500.0)
    substrate = np.full(wavelengths.size, 3.43 + 0j)
    R, T = reference_model(wavelengths, layers, thickness, substrate, 45.0, "s")
    assert np.max(R + T) > 1.0 + 1e-3


@pytest.mark.parametrize("stack_name", ["AR6", "BS45"])
def test_the_physical_convention_differs_measurably(reference_model, campaign, stack_name):
    """The correction is not cosmetic: it is of the order of the residual being reported."""
    wavelengths, layers, thickness, substrate = _stack_arrays(campaign, stack_name)
    R_ref, _ = reference_model(wavelengths, layers, thickness, substrate, 45.0, "s")
    R_phys, _ = coherent_rt(
        wavelengths,
        layers,
        thickness,
        np.ones(wavelengths.size, dtype=np.complex128),
        substrate,
        np.sin(np.deg2rad(45.0)),
        "s",
        reverse=True,
        legacy_gain_sign=False,
    )
    assert np.max(np.abs(R_ref - R_phys)) > 1e-5
