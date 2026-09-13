"""One command that checks the deposit: physics, determinism, and every deposited study.

    python reproduce.py

It runs six things and prints what each returns, so that a reader can see the numbers
appear rather than take them on trust:

1. **the optical model against closed-form results** -- Fresnel, Brewster, quarter-wave,
   half-wave absentee, energy conservation, reciprocity;
2. **a round trip** -- thicknesses are perturbed, the spectrum they produce is computed, and
   the inversion is required to return the thicknesses;
3. **the instrument model** -- that a symmetric polarizer leakage is invisible in the
   half-sum and an asymmetric one shifts it by exactly the predicted amount, and that the
   beam aperture and both leakage coefficients come back out of a spectrum they produced;
4. **the ladder** -- every study of the deposit, from the one that releases nothing to the
   joint inversion of the whole campaign, each loaded, validated and inverted twice so that
   determinism is compared bit for bit;
5. **the counter-experiments** -- the things the protocol forbids, done on purpose, so that
   the cost of each is a number in this output rather than an assertion in a paper.

Exits non-zero if anything fails, so it can be used as a gate in a continuous-integration
job as well as by hand.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from certus_re import __version__  # noqa: E402
from certus_re.forward import Evaluator  # noqa: E402
from certus_re.io_study import load_study  # noqa: E402
from certus_re.physics import assemble_plate, coherent_rt  # noqa: E402
from certus_re.solve import invert  # noqa: E402

STUDIES = HERE / "studies" / "volet2"
PASS = "  ok  "
FAIL = " FAIL "


class Checks:
    def __init__(self) -> None:
        self.failures = 0

    def __call__(self, label: str, condition: bool, detail: str = "") -> None:
        mark = PASS if condition else FAIL
        if not condition:
            self.failures += 1
        print(f"[{mark}] {label}" + (f"   {detail}" if detail else ""))


def section(title: str) -> None:
    print()
    print(title)
    print("-" * 78)


def check_physics(check: Checks) -> None:
    section("1. The optical model against results known in closed form")

    wavelength = np.array([1000.0])
    air = np.array([1.0 + 0j])
    glass = np.array([1.5 + 0j])
    none = (np.zeros((1, 0), dtype=complex), np.zeros(0))

    R, T = coherent_rt(wavelength, *none, air, glass, 0.0, "s", reverse=True)
    check(
        "Fresnel reflectance at normal incidence",
        abs(R[0] - ((1 - 1.5) / (1 + 1.5)) ** 2) < 1e-14,
        f"R = {R[0]:.6f}",
    )
    check("energy is conserved at that interface", abs(R[0] + T[0] - 1.0) < 1e-14)

    brewster = np.degrees(np.arctan(1.5))
    R, _ = coherent_rt(
        wavelength, *none, air, glass, np.sin(np.radians(brewster)), "p", reverse=True
    )
    check(
        "p polarization vanishes at Brewster's angle",
        R[0] < 1e-15,
        f"theta_B = {brewster:.3f} deg, R_p = {R[0]:.2e}",
    )

    n_layer = 2.2
    R, _ = coherent_rt(
        wavelength,
        np.full((1, 1), n_layer + 0j),
        np.array([1000.0 / (4 * n_layer)]),
        air,
        glass,
        0.0,
        "s",
        reverse=True,
    )
    expected = ((1.5 - n_layer**2) / (1.5 + n_layer**2)) ** 2
    check(
        "quarter-wave layer matches its closed form",
        abs(R[0] - expected) < 1e-14,
        f"R = {R[0]:.6f}, expected {expected:.6f}",
    )

    R, _ = coherent_rt(
        wavelength,
        np.full((1, 1), n_layer + 0j),
        np.array([2000.0 / (4 * n_layer)]),
        air,
        glass,
        0.0,
        "s",
        reverse=True,
    )
    check(
        "half-wave layer is absentee",
        abs(R[0] - ((1 - 1.5) / (1 + 1.5)) ** 2) < 1e-14,
    )

    rng = np.random.default_rng(0)
    wavelengths = np.linspace(1000.0, 5000.0, 40)
    worst = 0.0
    for polarization in ("s", "p"):
        for angle in (0.0, 8.0, 45.0, 60.0, 80.0):
            plate = assemble_plate(
                wavelengths,
                n_substrate=np.full(wavelengths.size, 3.43 + 0j),
                substrate_thickness_mm=1.0,
                angle_deg=angle,
                polarization=polarization,
                n_front_layers=np.tile(
                    rng.uniform(1.45, 2.3, 6), (wavelengths.size, 1)
                ).astype(complex),
                d_front_nm=rng.uniform(80.0, 800.0, 6),
                n_rear_layers=np.tile(
                    rng.uniform(1.45, 2.3, 4), (wavelengths.size, 1)
                ).astype(complex),
                d_rear_nm=rng.uniform(80.0, 800.0, 4),
                rear="coated",
            )
            worst = max(worst, float(np.max(np.abs(plate.R + plate.T - 1.0))))
    check(
        "a two-side-coated plate conserves energy at every angle",
        worst < 1e-12,
        f"largest departure {worst:.1e}",
    )


def _hold_the_instrument(study) -> None:
    """Hold the instrument blocks and process prior, so that a round trip tests the thicknesses alone.

    Released, the aperture and the leakage are extra directions the optimiser can trade a
    thickness against, and a round trip would no longer be the clean statement it is meant to
    be. They get a round trip of their own, in section 3.
    """
    study.free.aperture = "imposed"
    study.instrument.aperture_mode = "imposed"
    study.free.crosstalk = "none"
    study.instrument.crosstalk_mode = "none"
    study.free.process_prior_pct = None


def check_round_trip(check: Checks) -> None:
    section("2. Round trip: thicknesses in, spectrum computed, thicknesses out")

    for name in (
        "02_AR6_alone.json",
        "04_BS45_resolved.json",
        "05_BS17_resolved.json",
        "examples/two_faces_coated.json",
    ):
        path = STUDIES / name
        if not path.is_file():
            continue
        study = load_study(path)
        _hold_the_instrument(study)
        evaluator = Evaluator(study)
        nominal = evaluator.nominal_thicknesses()
        rng = np.random.default_rng(12345)
        truth = {
            stack: value * (1.0 + rng.normal(0.0, 0.01, value.size))
            for stack, value in nominal.items()
        }
        for plan, predicted in zip(evaluator.plans, evaluator.predict(truth)):
            plan.measurement.value = predicted.copy()
        result = invert(study)
        error = max(
            float(np.max(np.abs(result.thicknesses[s] - truth[s]))) for s in truth
        )
        n_layers = sum(value.size for value in truth.values())
        check(
            f"{study.name}: all {n_layers} thicknesses recovered",
            error < 1e-6,
            f"largest error {error:.2e} nm",
        )


def check_instrument(check: Checks) -> None:
    section("3. The instrument model: beam aperture and polarizer leakage")

    path = STUDIES / "06_BS17_aperture.json"
    if not path.is_file():
        check("a polarization-resolved study is deposited", False, f"none at {path}")
        return

    # -- what the leakage can and cannot be seen in ------------------------
    # No deposited study releases the polarizer leakage: on this campaign it earns no residual
    # and moves the retrieved coatings away from their designs, so it is not part of the
    # model. The capability is kept in the package for instruments that do need it, and it is
    # checked here on purpose -- an unused feature that is never exercised is a feature that
    # quietly rots. It is switched on for this section and nowhere else.
    study = load_study(path)
    study.instrument.crosstalk_mode = "fitted"
    study.free.crosstalk = "fitted"
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    pure_s, pure_p = evaluator.predict(nominal, crosstalk=(0.0, 0.0))

    symmetric = evaluator.predict(nominal, crosstalk=(0.07, 0.07))
    departure = float(
        np.max(np.abs(0.5 * (symmetric[0] + symmetric[1]) - 0.5 * (pure_s + pure_p)))
    )
    check(
        "a symmetric leakage leaves the half-sum rigorously invariant",
        departure < 1e-15,
        f"largest departure {departure:.1e}",
    )

    alpha, beta = 0.07, 0.02
    asymmetric = evaluator.predict(nominal, crosstalk=(alpha, beta))
    expected = (alpha - beta) * (pure_p - pure_s) / 2.0
    residue = 0.5 * (asymmetric[0] + asymmetric[1]) - 0.5 * (pure_s + pure_p)
    check(
        "an asymmetric one shifts it by exactly (alpha - beta)(Rp - Rs)/2",
        float(np.max(np.abs(residue - expected))) < 1e-15,
        f"largest shift {float(np.max(np.abs(expected))):.2e}",
    )
    check(
        "the mixing is the declared one, term for term",
        float(np.max(np.abs(asymmetric[0] - ((1 - alpha) * pure_s + alpha * pure_p))))
        < 1e-15
        and float(np.max(np.abs(asymmetric[1] - ((1 - beta) * pure_p + beta * pure_s))))
        < 1e-15,
    )

    # -- round trip on the instrument parameters themselves ----------------
    # One aperture band rather than three: on this campaign the second and third bands are
    # spectrally flat at 45 degrees, so the cone average barely moves there and the data
    # cannot determine them. A round trip must ask for what the data contain.
    study = load_study(path)
    study.instrument.crosstalk_mode = "fitted"
    study.free.crosstalk = "fitted"
    study.instrument.aperture_band_edges_nm = ()
    study.instrument.aperture_per_band_deg = None
    truth_aperture, truth_alpha, truth_beta = 1.60, 0.040, 0.090
    study.instrument.beam_aperture_deg = truth_aperture
    evaluator = Evaluator(study)
    nominal = evaluator.nominal_thicknesses()
    generated = evaluator.predict(
        nominal,
        aperture_deg=np.array([truth_aperture]),
        crosstalk=(truth_alpha, truth_beta),
    )
    for plan, predicted in zip(evaluator.plans, generated):
        plan.measurement.value = predicted.copy()
    # Start elsewhere, so that recovering the truth is a result and not the initial value.
    study.instrument.beam_aperture_deg = 2.0
    study.instrument.crosstalk_alpha = 0.0
    study.instrument.crosstalk_beta = 0.0
    result = invert(study)
    check(
        "the beam aperture is recovered from the spectrum it produced",
        abs(float(result.aperture_deg[0]) - truth_aperture) < 1e-4,
        f"{float(result.aperture_deg[0]):.5f} deg against {truth_aperture} deg",
    )
    check(
        "alpha and beta are recovered from the spectra they produced",
        abs(result.crosstalk[0] - truth_alpha) < 1e-5
        and abs(result.crosstalk[1] - truth_beta) < 1e-5,
        f"alpha {result.crosstalk[0]:.6f} against {truth_alpha}, "
        f"beta {result.crosstalk[1]:.6f} against {truth_beta}",
    )
    error = max(
        float(np.max(np.abs(result.thicknesses[s] - nominal[s]))) for s in nominal
    )
    check(
        "and the sixteen thicknesses come back with them",
        error < 1e-4,
        f"largest error {error:.2e} nm",
    )


def check_studies(check: Checks) -> None:
    section("4. The ladder: every deposited study, loaded, validated, inverted, twice")

    paths = sorted(STUDIES.glob("*.json"))
    if not paths:
        check("deposited studies present", False, f"none found under {STUDIES}")
        return

    for path in paths:
        study = load_study(path, strict=False)
        problems = study.validate()
        check(f"{path.stem}: consistent", not problems, "; ".join(problems[:1]))
        if problems:
            continue

        rung = _provenance(path).get("rung")
        question = _provenance(path).get("question")
        if rung:
            print(f"         rung {rung} -- {question}")

        started = time.perf_counter()
        first = invert(study)
        elapsed = time.perf_counter() - started
        second = invert(load_study(path))
        identical = all(
            np.array_equal(first.thicknesses[s], second.thicknesses[s])
            for s in first.thicknesses
        )
        check(f"{path.stem}: deterministic", identical, f"{elapsed:.1f} s per run")
        _summarise(first, study)


def _provenance(path: Path) -> dict:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("provenance", {})


def _summarise(result, study) -> None:
    print(
        f"         {result.dof.n_free} free parameters, {study.n_points} points, "
        + (
            "forward evaluation only"
            if result.dof.n_free == 0
            else f"{result.dof.points_per_parameter:.0f} points per parameter"
        )
    )
    for residual in result.residuals:
        print(
            f"         {residual.sample:<20s} {residual.polarization} rms "
            f"{100 * residual.rms_initial:6.3f} % -> "
            f"{100 * residual.rms_final:6.3f} %   chi2/pt "
            f"{residual.chi2_per_point:7.2f}"
        )
    for stack, departure in result.qwot_departure_pct().items():
        total = 100.0 * (
            result.qwot[stack].sum() - result.nominal_qwot[stack].sum()
        ) / result.nominal_qwot[stack].sum()
        print(
            f"         {stack:<10s} QWOT departure {total:+7.3f} % total, "
            f"{float(np.sqrt((departure ** 2).mean())):5.2f} % rms per layer, "
            f"{float(np.abs(departure).max()):5.2f} % max"
        )
    if result.at_bounds:
        print(f"         parameters on their bound: {', '.join(result.at_bounds)}")


def check_counter_experiments(check: Checks) -> None:
    section("5. The counter-experiments: what the protocol forbids, measured")
    print(
        "  Each of these is wrong by construction. They are inverted here so that the cost"
    )
    print("  of the mistake is a number in this output and not an assertion in a paper.")

    paths = sorted((STUDIES / "counter_experiments").glob("*.json"))
    if not paths:
        check("counter-experiments present", False, "none found")
        return
    for path in paths:
        study = load_study(path, strict=False)
        problems = study.validate()
        check(f"{path.stem}: runs", not problems, "; ".join(problems[:1]))
        if problems:
            continue
        lesson = _provenance(path).get("counter_experiment", "")
        if lesson:
            print(f"         lesson: {lesson}")
        _summarise(invert(study), study)


def check_tests(check: Checks) -> None:
    section("6. The test suite")
    try:
        import pytest
    except ImportError:
        print("  pytest is not installed; skipping (pip install pytest)")
        return
    code = pytest.main(["-q", str(HERE / "tests")])
    check("test suite", code == 0, f"pytest exit code {code}")


def main() -> int:
    print("=" * 78)
    print(f"certus_re {__version__} -- deposit self-check")
    print("=" * 78)
    print(f"python  {platform.python_version()} on {platform.system()}")
    print(f"numpy   {np.__version__}")
    try:
        import scipy

        print(f"scipy   {scipy.__version__}")
    except ImportError:
        print("scipy   MISSING")

    check = Checks()
    check_physics(check)
    check_round_trip(check)
    check_instrument(check)
    check_studies(check)
    check_counter_experiments(check)
    check_tests(check)

    print()
    print("=" * 78)
    if check.failures:
        print(f"{check.failures} check(s) FAILED")
        return 1
    print("every check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
