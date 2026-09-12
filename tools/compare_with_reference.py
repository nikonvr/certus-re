"""Compare the forward model of this package with the reference implementation.

The numerical core of this package was rewritten for clarity and for the joint, two-sided
model. What justifies that rewrite is not that it reads better: it is that it returns the
same numbers. This script computes the reflectance and transmittance of the campaign
coatings with both implementations, on the same designs, the same dispersions and the same
angles, and reports two quantities:

* **compatibility mode** -- this package reproducing the reference exactly, including its
  sign convention on the extinction coefficient. This is the non-regression figure.
* **cost of that convention** -- the same model with absorption restored. The reference
  kernel for oblique incidence is written for ``n - ik`` and is fed indices stored as
  ``n + ik``; it therefore amplifies instead of absorbing, and returns ``R + T > 1``. The
  difference is reported for each dispersion dataset, because it scales with ``k``.

The reference is the laboratory's production tool, which this script only ever **reads**.

Usage
-----
    python tools/compare_with_reference.py [--reference PATH] [--tolerance 1e-12]
                                           [--study FILE] ...

Exits non-zero if any compatibility-mode comparison exceeds the tolerance.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from certus_re.io_study import load_study  # noqa: E402
from certus_re.physics import coherent_rt  # noqa: E402

DEFAULT_REFERENCE = Path(
    r"D:\drivfl\01_Recherche_Enseignement\02_Couches_Minces_Optique"
    r"\couches minces 2026\CERTUS\0409"
)
STUDIES = Path(__file__).resolve().parent.parent / "studies" / "volet2"
ANGLES = (8.0, 45.0)
POLARIZATIONS = ("s", "p")


def stack_arrays(study, stack_name: str):
    """Design, dispersions and measured grid of one coating, as the model needs them."""
    stack = study.stacks[stack_name]
    lambda0 = study.lambda0_nm
    thickness = np.asarray(
        [
            layer.thickness_nm(
                lambda0, float(study.materials[layer.material].n_at(lambda0))
            )
            for layer in stack.layers
        ],
        dtype=np.float64,
    )
    sample = study.samples_using(stack_name)[0]
    wavelengths = sample.measurements[0].wavelength_nm
    layers = np.stack(
        [study.materials[m].complex_at(wavelengths) for m in stack.materials], axis=1
    )
    substrate = study.substrates[sample.substrate.material].complex_at(wavelengths)
    return wavelengths, layers, thickness, substrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--study",
        type=Path,
        action="append",
        default=None,
        help="study to take designs and dispersions from; may be repeated "
        "(default: one study per index dataset)",
    )
    args = parser.parse_args(argv)

    studies = args.study or [STUDIES / "08_joint_campaign.json"]

    if not (args.reference / "certus").is_dir():
        print(f"reference implementation not found at {args.reference}", file=sys.stderr)
        return 2

    os.environ.setdefault("CERTUS_RE_HEADLESS", "1")
    sys.path.insert(0, str(args.reference))
    try:
        from certus.physics.certus_tmm_oblique import (  # type: ignore
            calc_spectrum_oblique_vectorized as reference_model,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"cannot import the reference forward model: {exc}", file=sys.stderr)
        return 2

    print(f"tolerance {args.tolerance:g}")
    print(
        "\ncompatibility mode reproduces the reference exactly, sign convention included;\n"
        "the physical columns are the same model with absorption restored.\n"
    )
    header = (
        f"{'indices':<10s} {'stack':<6s} {'angle':>6s} {'pol':>4s} "
        f"{'|dR| compat':>12s} {'|dT| compat':>12s} "
        f"{'|dR| phys':>11s} {'|dT| phys':>11s}"
    )
    print(header)
    print("-" * len(header))

    worst_compat = 0.0
    failures = 0
    cost: dict[str, float] = {}

    for path in studies:
        if not Path(path).is_file():
            print(f"  study not found, skipped: {path}", file=sys.stderr)
            continue
        study = load_study(path)
        # One determination of the optical constants is deposited, the one published with
        # Volet 1. The cost of the sign convention scales with k, so it is still reported
        # per dataset: pass --study to compare another one.
        dataset = Path(path).stem
        cost.setdefault(dataset, 0.0)

        for stack_name in ("AR6", "BS45"):
            if stack_name not in study.stacks or not study.samples_using(stack_name):
                continue
            wavelengths, layers, thickness, substrate = stack_arrays(study, stack_name)
            air = np.ones(wavelengths.size, dtype=np.complex128)

            for angle in ANGLES:
                for polarization in POLARIZATIONS:
                    R_ref, T_ref = reference_model(
                        wavelengths, layers, thickness, substrate, angle, polarization
                    )
                    common = dict(
                        wavelength_nm=wavelengths,
                        n_layers=layers,
                        d_nm=thickness,
                        n_incident=air,
                        n_exit=substrate,
                        sin_theta_incident=np.sin(np.deg2rad(angle)),
                        polarization=polarization,
                        reverse=True,
                    )
                    R_compat, T_compat = coherent_rt(**common, legacy_gain_sign=True)
                    R_phys, T_phys = coherent_rt(**common, legacy_gain_sign=False)

                    dR = float(np.max(np.abs(R_ref - R_compat)))
                    dT = float(np.max(np.abs(T_ref - T_compat)))
                    pR = float(np.max(np.abs(R_ref - R_phys)))
                    pT = float(np.max(np.abs(T_ref - T_phys)))

                    worst_compat = max(worst_compat, dR, dT)
                    cost[dataset] = max(cost[dataset], pR, pT)
                    flag = "" if max(dR, dT) <= args.tolerance else "  <-- EXCEEDS"
                    if flag:
                        failures += 1
                    print(
                        f"{dataset:<10s} {stack_name:<6s} {angle:6.1f} {polarization:>4s} "
                        f"{dR:12.3e} {dT:12.3e} {pR:11.3e} {pT:11.3e}{flag}"
                    )

    print()
    print(f"compatibility mode, largest discrepancy : {worst_compat:.3e}")
    for dataset in sorted(cost):
        print(
            f"cost of the reference sign convention   : {cost[dataset]:.3e}  "
            f"({dataset} dispersions)"
        )
    if failures:
        print(f"{failures} comparison(s) exceeded the tolerance", file=sys.stderr)
        return 1
    print(
        "\nThe port is faithful: in compatibility mode the two implementations agree to the\n"
        "stated tolerance, so every remaining difference is the sign convention and nothing\n"
        "else. The cost of that convention scales with k, which is why it is reported per\n"
        "dispersion dataset."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
