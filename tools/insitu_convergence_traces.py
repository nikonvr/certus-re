"""Residual histories of the in situ multi-start inversions, for the convergence figure.

``certus_re`` does not record the path the optimizer takes, and it has no reason to: no result
depends on it. But the claim of Section 6 -- that starts scattered over twenty percent in
thickness, from a foreign index, all fall into one basin -- is a statement about paths, and a
table cannot show it.

So this tool reuses the benchmark of ``insitu_multistart_noprior`` unchanged, and only wraps
``least_squares`` from the outside to record the norm of the weighted residual at every
evaluation. Nothing in the package is modified, and the inversions performed here are the same
ones the benchmark performs.

The residual is reported in multiples of the declared photometric noise floor: one means the
model sits exactly at the repeatability of the instrument.

    python tools/insitu_convergence_traces.py --draws 6 --workers 4
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
RESULTS = PACKAGE / "results"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

PERTURBATIONS = [0.05, 0.10, 0.15, 0.20]


def one_run(task):
    """One benchmark inversion, with the optimizer's residual history recorded."""
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import certus_re.solve as solve_mod
    from scipy.optimize import least_squares as _least_squares

    history: list[float] = []

    def traced(fun, x0, **kwargs):
        def wrapped(x):
            r = np.asarray(fun(x), dtype=float)
            history.append(float(np.sqrt(np.mean(r * r))))
            return r

        return _least_squares(wrapped, x0, **kwargs)

    solve_mod.least_squares = traced

    bench = importlib.import_module("insitu_multistart_noprior")
    result = bench.run_single_inversion(task)
    return {
        "perturbation_pct": result["perturbation_pct"],
        "draw_idx": result["draw_idx"],
        "rmse_pct": result["rmse_pct"],
        "elapsed_s": result["elapsed_s"],
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=6,
                        help="draws per perturbation level (default 6: enough curves to show "
                             "the funnel without redoing the whole benchmark)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    args = parser.parse_args()

    bench = importlib.import_module("insitu_multistart_noprior")
    from certus_re.forward import Evaluator

    nominal = Evaluator(bench.get_3components_study()).nominal_thicknesses()

    tasks = [(0.0, 0, nominal)]
    for level in PERTURBATIONS:
        tasks += [(level, d, nominal) for d in range(args.draws)]

    print(f"tracing {len(tasks)} inversions on {args.workers} workers", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_run, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            out.append(res)
            h = res["history"]
            print(f"  [{len(out)}/{len(tasks)}] {res['perturbation_pct']:.0f}% "
                  f"draw {res['draw_idx']}: {len(h)} evaluations, "
                  f"{h[0]:.0f} -> {h[-1]:.2f} noise floors, "
                  f"RMSE {res['rmse_pct']:.3f}% ({res['elapsed_s']:.0f}s)", flush=True)

    out.sort(key=lambda r: (r["perturbation_pct"], r["draw_idx"]))
    target = RESULTS / "insitu_convergence_traces.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
