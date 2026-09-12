"""One command that checks the deposit: physics, determinism, and every deposited study.

    python reproduce.py

It runs four things and prints what each returns, so that a reader can see the numbers
appear rather than take them on trust:

1. **the optical model against closed-form results** -- Fresnel, Brewster, quarter-wave,
   half-wave absentee, energy conservation, reciprocity;
2. **a round trip** -- thicknesses are perturbed, the spectrum they produce is computed, and
   the inversion is required to return the thicknesses;
3. **determinism** -- every study is inverted twice and the two results are compared bit for
   bit;
4. **every deposited study** -- loaded, validated, inverted, with its residuals and its
   parameter budget printed.

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


def check_round_trip(check: Checks) -> None:
    section("2. Round trip: thicknesses in, spectrum computed, thicknesses out")

    for name in ("AR6_alone.json", "BS45_alone.json"):
        path = STUDIES / name
        if not path.is_file():
            continue
        study = load_study(path)
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
        check(
            f"{study.name}: every thickness recovered",
            error < 1e-6,
            f"largest error {error:.2e} nm",
        )


def check_studies(check: Checks) -> None:
    section("3. Every deposited study: loaded, validated, inverted, twice")

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

        started = time.perf_counter()
        first = invert(study)
        elapsed = time.perf_counter() - started
        second = invert(load_study(path))
        identical = all(
            np.array_equal(first.thicknesses[s], second.thicknesses[s])
            for s in first.thicknesses
        )
        check(f"{path.stem}: deterministic", identical, f"{elapsed:.1f} s per run")

        print(
            f"         {first.dof.n_free} free parameters, {study.n_points} points, "
            f"{first.dof.points_per_parameter:.0f} points per parameter"
        )
        for residual in first.residuals:
            print(
                f"         {residual.sample:<20s} rms "
                f"{100 * residual.rms_initial:6.3f} % -> "
                f"{100 * residual.rms_final:6.3f} %   chi2/pt "
                f"{residual.chi2_per_point:7.2f}"
            )
        for stack, departure in first.qwot_departure_pct().items():
            total = 100.0 * (
                first.qwot[stack].sum() - first.nominal_qwot[stack].sum()
            ) / first.nominal_qwot[stack].sum()
            print(
                f"         {stack:<10s} QWOT departure {total:+7.3f} % total, "
                f"{float(np.sqrt((departure ** 2).mean())):5.2f} % rms per layer, "
                f"{float(np.abs(departure).max()):5.2f} % max"
            )
        if first.at_bounds:
            print(f"         thicknesses on their bound: {', '.join(first.at_bounds)}")


def check_tests(check: Checks) -> None:
    section("4. The test suite")
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
    check_studies(check)
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
