"""Command line: run a study, or check one without running it.

    python -m certus_re run     studies/volet2/AR6_alone.json
    python -m certus_re check   studies/volet2/joint_campaign.json
    python -m certus_re predict studies/volet2/AR6_alone.json --out spectra.csv

``check`` validates a study and prints its parameter budget without inverting anything,
which is the fastest way to find out that a dispersion table does not span a measurement or
that a declared free-parameter block does not match the instrument description.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .dof import count_free_parameters
from .forward import Evaluator
from .io_study import StudyError, load_study
from .report import text_report, write_json_report
from .solve import invert


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("study", type=Path, help="JSON description of the study")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="certus_re", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="invert a study and report the result")
    _add_common(p_run)
    p_run.add_argument(
        "--json", type=Path, default=None, help="also write a machine-readable report here"
    )
    p_run.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="half-width of the box constraint on each thickness, as a fraction of "
        "nominal (default 0.5)",
    )
    p_run.add_argument("--verbose", action="store_true")
    p_run.add_argument(
        "--legacy-gain-sign",
        action="store_true",
        help=argparse.SUPPRESS,  # comparison with the reference implementation only
    )

    p_check = sub.add_parser(
        "check", help="validate a study and print its parameter budget, without inverting"
    )
    _add_common(p_check)

    p_predict = sub.add_parser(
        "predict", help="compute the spectra of the nominal design, without inverting"
    )
    _add_common(p_predict)
    p_predict.add_argument("--out", type=Path, default=None, help="write a CSV here")

    args = parser.parse_args(argv)

    try:
        study = load_study(args.study, strict=args.command != "check")
    except StudyError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.command == "check":
        problems = study.validate()
        print(f"{study.name}: {len(study.samples)} samples, {study.n_points} points")
        print()
        print(count_free_parameters(study).to_text())
        print()
        if problems:
            print("problems found:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("the study is consistent and can be inverted as declared.")
        return 0

    if args.command == "predict":
        evaluator = Evaluator(study)
        predicted = evaluator.predict(evaluator.nominal_thicknesses())
        rows: list[str] = ["sample,measurement,wavelength_nm,measured,predicted"]
        for plan, values in zip(evaluator.plans, predicted):
            m = plan.measurement
            for w, meas, pred in zip(m.wavelength_nm, m.value, values):
                rows.append(
                    f"{plan.sample.name},{m.label or m.quantity},{w:.4f},"
                    f"{meas:.8f},{pred:.8f}"
                )
            print(
                f"{plan.sample.name:<20s} {m.quantity:>5s} {m.n_points:5d} pts  "
                f"rms {100 * float(np.sqrt(np.mean((values - m.value) ** 2))):7.3f} %"
            )
        if args.out:
            args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
            print(f"\nwritten to {args.out}")
        return 0

    result = invert(
        study,
        thickness_tolerance=args.tolerance,
        verbose=args.verbose,
        legacy_gain_sign=args.legacy_gain_sign,
    )
    print(text_report(result, study))
    if args.json:
        write_json_report(args.json, result, study)
        print(f"machine-readable report written to {args.json}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
