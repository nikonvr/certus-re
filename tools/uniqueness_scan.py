"""Is the retrieved coating the best solution, or merely the nearest one?

A deposited run starts from the nominal design and is deterministic, which is the right thing
for a published result and leaves one question open. This script answers it the only way it
can be answered: start from many scattered points and see where they land.

It is a diagnostic, not an inversion mode. Nothing it does changes what
``python -m certus_re run`` returns.

The experiment is repeated for several widths of the search window, because that is the
interesting variable. The window is prior information about how the coatings were made, and
the claim worth testing is not "this problem has a unique solution" -- it does not -- but
"this problem together with this prior has one". Widening the window scatters the starting
points and loosens the bounds at the same time, which is exactly the regime in which a
well-posed-looking inversion stops being one.

Usage
-----
    python tools/uniqueness_scan.py --study 05_BS17_resolved.json --starts 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE))

from certus_re.forward import Evaluator  # noqa: E402
from certus_re.io_study import load_study  # noqa: E402
from certus_re.solve import invert  # noqa: E402

STUDIES = PACKAGE / "studies" / "volet2"
SIGMA_REPEAT = 0.00136

# "The same solution" means: a final residual within a tenth of what the photometry repeats
# to. Anything closer than that is not distinguishable by measurement.
SAME_SOLUTION = 0.1 * SIGMA_REPEAT


def one_start(path: Path, window: float, seed: int, nominal: dict) -> dict:
    study = load_study(path)
    study.free.thickness_tolerance = window
    if seed == 0:
        start = None  # the nominal design itself, which is what a deposited run uses
    else:
        rng = np.random.default_rng(seed)
        start = {
            stack: value * (1.0 + rng.uniform(-window, window, value.size))
            for stack, value in nominal.items()
        }
    result = invert(study, thickness_tolerance=window, start=start)
    quad = float(np.sqrt(np.mean([r.rms_final ** 2 for r in result.residuals])))
    stack = next(iter(result.thicknesses))
    departure = result.qwot_departure_pct()[stack]
    return dict(
        seed=seed,
        window=window,
        rms=quad,
        total_nm=float(result.thicknesses[stack].sum()),
        dispersion=float(np.sqrt((departure ** 2).mean())),
        at_bounds=len([b for b in result.at_bounds if "layer" in b]),
        thicknesses=[float(x) for x in result.thicknesses[stack]],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", default="05_BS17_resolved.json")
    ap.add_argument("--starts", type=int, default=60)
    ap.add_argument("--windows", type=float, nargs="+", default=[0.03, 0.10, 0.50])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    path = STUDIES / args.study
    nominal = Evaluator(load_study(path)).nominal_thicknesses()
    n_layers = sum(v.size for v in nominal.values())

    print(f"{args.study}: {n_layers} thicknesses, {args.starts} starting points per window,")
    print("optical constants held fixed, the instrument held as deposited.")
    print(f"Two starts count as the same solution when their residuals differ by less than")
    print(f"{SAME_SOLUTION:.5f}, a tenth of the measured repeatability.\n")

    print(f"{'window':>8s}{'starts':>8s}{'best rms':>11s}{'worst rms':>11s}"
          f"{'reached best':>14s}{'total nm spread':>17s}{'QWOT rms at best':>18s}")
    print("-" * 87)

    everything = []
    for window in args.windows:
        rows = [one_start(path, window, seed, nominal) for seed in range(args.starts)]
        everything.extend(rows)
        best = min(rows, key=lambda r: r["rms"])
        reached = sum(1 for r in rows if r["rms"] - best["rms"] < SAME_SOLUTION)
        totals = np.array([r["total_nm"] for r in rows])
        print(f"{100 * window:7.0f}%{len(rows):8d}{100 * best['rms']:10.4f}%"
              f"{100 * max(r['rms'] for r in rows):10.4f}%"
              f"{reached:9d}/{len(rows):<4d}{totals.max() - totals.min():16.2f}"
              f"{best['dispersion']:17.2f}%")

    print("\nRead it this way: if every start reaches the same residual, the inversion is")
    print("well posed GIVEN THAT WINDOW. If the spread of total thickness grows as the window")
    print("opens while the residual barely moves, the extra freedom is buying nothing that")
    print("the measurement can see, and the answer has become a choice rather than a result.")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(everything, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
