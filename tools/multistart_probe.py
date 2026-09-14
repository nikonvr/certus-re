"""Probe the landscape of the sixteen-layer inversion, and ask what the aperture model buys.

A deposited run of this package starts from the nominal design and is deterministic. That is
the right thing for a published result, and it leaves one question open: is the solution
reached from the nominal design the *best* one, or merely the *nearest* one? The only way to
find out is to start from many other points and see where they land.

This script does that, and it is a diagnostic, not an inversion mode. Nothing it does changes
what ``python -m certus_re run`` returns.

It compares four models of the beam aperture on the same data -- the sixteen-layer coating at
45 degrees, R_s and R_p inverted simultaneously, optical constants held fixed throughout:

``neglected``   the cone ignored altogether
``imposed``     imposed at the manufacturer's 2.0 degrees, no parameter spent
``single``      one aperture released for the whole band
``staircase``   one aperture per band, stepped at the documented switchovers

The comparison that matters is not which reaches the lowest residual -- more parameters
always can -- but whether the extra parameters buy more than the measured photometric
repeatability, and whether they leave the retrieved coating closer to or further from a design
the deposition process could have produced.

Usage
-----
    python tools/multistart_probe.py --config staircase --first-seed 0 --n-seeds 25
    python tools/multistart_probe.py --summarize results/multistart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from certus_re.io_study import load_study  # noqa: E402
from certus_re.solve import invert  # noqa: E402

PACKAGE = Path(__file__).resolve().parent.parent
STUDY = PACKAGE / "studies" / "volet2" / "04_BS45_resolved.json"
STACK = "BS45"

# Photometric noise floor of the instrument from Optics Continuum companion paper.
# Every residual difference below is quoted against it, because a difference smaller than what
# the instrument repeats to is not a difference.
SIGMA = 0.0020

CONFIGS = {
    "neglected": "the beam cone ignored altogether",
    "imposed": "imposed at 2.0 deg, the manufacturer's specification, no parameter spent",
    "single": "one aperture released for the whole band",
    "staircase": "one aperture per band, stepped at the documented switchovers",
}


def configure(study, config: str) -> None:
    """Apply one aperture model to a loaded study, in place."""
    instrument = study.instrument
    instrument.crosstalk_mode = "none"
    study.free.crosstalk = "none"
    instrument.aperture_per_band_deg = None
    if config == "neglected":
        instrument.aperture_mode = "imposed"
        study.free.aperture = "imposed"
        instrument.beam_aperture_deg = 1e-3
        instrument.aperture_band_edges_nm = ()
    elif config == "imposed":
        instrument.aperture_mode = "imposed"
        study.free.aperture = "imposed"
        instrument.beam_aperture_deg = 2.0
        instrument.aperture_band_edges_nm = ()
    elif config == "single":
        instrument.aperture_mode = "fitted"
        study.free.aperture = "fitted"
        instrument.beam_aperture_deg = 2.0
        instrument.aperture_band_edges_nm = ()
    elif config == "staircase":
        instrument.aperture_mode = "fitted"
        study.free.aperture = "fitted"
        instrument.beam_aperture_deg = 2.0
        instrument.aperture_band_edges_nm = (2530.0, 3700.0)
    else:
        raise SystemExit(f"unknown config {config!r}; choose from {sorted(CONFIGS)}")


def one_start(config: str, seed: int, nominal: np.ndarray, window: float) -> dict:
    """One inversion from one scattered starting point."""
    study = load_study(STUDY)
    configure(study, config)
    if seed == 0:
        start = None  # the nominal design itself, which is what a deposited run uses
    else:
        rng = np.random.default_rng(seed)
        start = {STACK: nominal * (1.0 + rng.uniform(-window, window, nominal.size))}
    result = invert(study, start=start)
    by_pol = {r.polarization: r for r in result.residuals}
    departure = result.qwot_departure_pct()[STACK]
    return {
        "config": config,
        "seed": seed,
        "n_free": result.dof.n_free,
        "rms_s": by_pol["s"].rms_final,
        "rms_p": by_pol["p"].rms_final,
        "rms_quadratic": float(
            np.sqrt((by_pol["s"].rms_final ** 2 + by_pol["p"].rms_final ** 2) / 2)
        ),
        "total_nm": float(result.thicknesses[STACK].sum()),
        "dispersion_pct": float(np.sqrt((departure**2).mean())),
        "max_departure_pct": float(np.abs(departure).max()),
        "aperture_deg": [float(x) for x in result.aperture_deg],
        "layers_at_bound": len([b for b in result.at_bounds if "layer" in b]),
        "converged": bool(result.success),
        "thicknesses_nm": [float(x) for x in result.thicknesses[STACK]],
    }


def run(config: str, first_seed: int, n_seeds: int, out_dir: Path) -> Path:
    study = load_study(STUDY)
    configure(study, config)
    from certus_re.forward import Evaluator

    nominal = Evaluator(study).nominal_thicknesses()[STACK]
    window = study.free.tolerance_for(STACK)

    rows = []
    for seed in range(first_seed, first_seed + n_seeds):
        row = one_start(config, seed, nominal, window)
        rows.append(row)
        print(
            f"  {config:<10s} seed {seed:4d}  rms {100 * row['rms_quadratic']:.4f} %  "
            f"disp {row['dispersion_pct']:.2f} %  total {row['total_nm']:.1f} nm",
            flush=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{config}_{first_seed:05d}_{n_seeds:04d}.json"
    target.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"written to {target}")
    return target


def summarize(out_dir: Path) -> int:
    rows: list[dict] = []
    for path in sorted(out_dir.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        print(f"no result under {out_dir}", file=sys.stderr)
        return 2

    print(f"{len(rows)} inversions of the sixteen-layer coating, R_s and R_p together,")
    print("optical constants held fixed, thicknesses started from scattered points.\n")
    header = (
        f"{'aperture model':<12s}{'starts':>7s}{'free':>6s}{'best rms %':>12s}"
        f"{'vs imposed':>12s}{'reached':>9s}{'disp % at best':>16s}{'aperture at best':>18s}"
    )
    print(header)
    print("-" * len(header))

    best_of: dict[str, dict] = {}
    for config in CONFIGS:
        group = [r for r in rows if r["config"] == config]
        if not group:
            continue
        best = min(group, key=lambda r: r["rms_quadratic"])
        best_of[config] = best
        # How many distinct starts land on the best solution, to within a tenth of the
        # measured repeatability: that is the practical meaning of "unique".
        reached = sum(
            1 for r in group if r["rms_quadratic"] - best["rms_quadratic"] < 0.1 * SIGMA
        )
        reference = best_of.get("imposed")
        delta = (
            "---"
            if reference is None
            else f"{(best['rms_quadratic'] - reference['rms_quadratic']) / SIGMA:+.2f} sig"
        )
        aperture = ", ".join(f"{a:.2f}" for a in best["aperture_deg"])
        print(
            f"{config:<12s}{len(group):7d}{best['n_free']:6d}"
            f"{100 * best['rms_quadratic']:12.4f}{delta:>12s}"
            f"{reached:6d}/{len(group):<3d}{best['dispersion_pct']:14.2f} %"
            f"{aperture:>18s}"
        )

    print()
    for config in CONFIGS:
        group = [r for r in rows if r["config"] == config]
        if not group:
            continue
        totals = np.array([r["total_nm"] for r in group])
        spread = np.array([r["rms_quadratic"] for r in group])
        print(
            f"{config:<12s} residual over all starts "
            f"{100 * spread.min():.4f}-{100 * spread.max():.4f} %, "
            f"total thickness {totals.min():.1f}-{totals.max():.1f} nm "
            f"(spread {totals.max() - totals.min():.1f} nm)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(CONFIGS))
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=25)
    parser.add_argument(
        "--out", type=Path, default=PACKAGE / "results" / "multistart"
    )
    parser.add_argument("--summarize", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.summarize is not None:
        return summarize(args.summarize)
    if args.config is None:
        parser.error("give --config, or --summarize a directory")
    run(args.config, args.first_seed, args.n_seeds, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
