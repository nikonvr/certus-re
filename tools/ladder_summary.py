"""Every deposited inversion, on one page.

Runs each study of the ladder and prints what the article needs to quote: how well the
nominal design already reproduced the measurement, how much the inversion improved on that,
how far the retrieved coating moved from its design, and what the improvement is worth in
units of the photometry's own repeatability.

The last column is the one that decides things. An inversion that improves the residual by
less than the instrument repeats to has not learned anything about the coating.

Usage
-----
    python tools/ladder_summary.py
    python tools/ladder_summary.py --json results/ladder_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE))

from certus_re.io_study import load_study  # noqa: E402
from certus_re.solve import invert  # noqa: E402

STUDIES = PACKAGE / "studies" / "volet2"
SIGMA_REPEAT = 0.0020


def run(path: Path) -> dict:
    study = load_study(path)
    result = invert(study)
    rows = []
    for r in result.residuals:
        rows.append(
            dict(
                sample=r.sample,
                pol=r.polarization,
                points=r.n_points,
                before=r.rms_initial,
                after=r.rms_final,
                chi2=r.chi2_per_point,
            )
        )
    departures = {
        stack: float(np.sqrt((d ** 2).mean()))
        for stack, d in result.qwot_departure_pct().items()
    }
    worst = {
        stack: float(np.abs(d).max()) for stack, d in result.qwot_departure_pct().items()
    }
    return dict(
        name=study.name,
        file=path.name,
        n_free=result.dof.n_free,
        n_points=result.dof.n_points,
        residuals=rows,
        departures=departures,
        worst=worst,
        at_bounds=[b for b in result.at_bounds],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    paths = sorted(STUDIES.glob("[0-9]*.json"))
    out = []
    print("EVERY DEPOSITED INVERSION")
    print("=" * 96)
    print(f"{'study':<22s}{'free':>5s}{'points':>8s}{'pts/par':>8s}"
          f"{'nominal':>10s}{'inverted':>10s}{'gain/sig':>10s}{'QWOT rms':>10s}{'bounds':>8s}")
    print("-" * 96)
    for path in paths:
        r = run(path)
        out.append(r)
        before = float(np.sqrt(np.mean([x["before"] ** 2 for x in r["residuals"]])))
        after = float(np.sqrt(np.mean([x["after"] ** 2 for x in r["residuals"]])))
        gain = (before - after) / SIGMA_REPEAT
        disp = (max(r["departures"].values()) if r["departures"] else 0.0)
        per = r["n_points"] / r["n_free"] if r["n_free"] else float("inf")
        print(f"{r['file'][:-5]:<22s}{r['n_free']:5d}{r['n_points']:8d}{per:8.1f}"
              f"{100 * before:9.3f}%{100 * after:9.3f}%{gain:+10.1f}{disp:9.2f}%"
              f"{len(r['at_bounds']):8d}")

    print("\nPER MEASUREMENT")
    print("=" * 96)
    for r in out:
        print(f"\n{r['file'][:-5]}  ({r['n_free']} free, {r['n_points']} points)")
        for x in r["residuals"]:
            print(f"   {x['sample']:<22s} {x['pol']:>2s}  {x['points']:5d} pts   "
                  f"{100 * x['before']:7.3f} % -> {100 * x['after']:7.3f} %   "
                  f"chi2/pt {x['chi2']:9.2f}")
        for stack in sorted(r["departures"]):
            print(f"   {stack:<22s} QWOT departure rms {r['departures'][stack]:5.2f} %, "
                  f"worst {r['worst'][stack]:5.2f} %")
        if r["at_bounds"]:
            print(f"   on their bound: {', '.join(r['at_bounds'])}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
