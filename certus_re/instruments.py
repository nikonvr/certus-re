"""Named instrument presets, so that a study file need not describe a spectrophotometer.

A study written against this package must say how the beam that produced its spectra was
shaped: the half-angle of the cone, and the wavelengths at which the instrument changes
detector or source, because the cone changes with them. That is machine-specific knowledge, and
a user inverting their own coatings should not have to rediscover it for the instrument these
authors happen to own.

Two presets are provided.

``photon_rt_5200``
    The EssentOptics PHOTON RT in its 185-5200 nm configuration, the instrument of this study.
    Beam half-angle 2.0 degrees, the manufacturer's specified divergence; detector switchover at
    2530 nm (InGaAs to PbSe) and source switchover at 3700 nm (halogen to the IR emitter). Below
    10 degrees of incidence the cone average is not applied: the response is stationary in angle
    there and averaging over it changes nothing a measurement could see.

``manual``
    Nothing assumed. A single half-angle, no switchovers, and the user supplies the value. This
    is the preset to start from on any other spectrophotometer: measure or look up the
    divergence, set it, and add band edges only if the instrument really does change its optics
    part-way through a scan.

A study file selects one with

    "instrument": {"preset": "photon_rt_5200"}

and may override any field of it:

    "instrument": {"preset": "manual", "beam_aperture_deg": 1.2}

An unknown preset is refused by name rather than silently replaced by a default, and a study
that names no preset at all keeps the explicit fields it gives, which is how every study of this
deposit was written before presets existed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PRESETS", "apply_preset", "describe"]


PRESETS: dict[str, dict[str, Any]] = {
    "photon_rt_5200": {
        "name": "EssentOptics PHOTON RT, 185-5200 nm configuration",
        "beam_aperture_deg": 2.0,
        "aperture_band_edges_nm": [2530.0, 3700.0],
        "aperture_mode": "imposed",
        "aperture_bounds_deg": [1.0, 2.5],
        "aperture_min_angle_deg": 10.0,
        "crosstalk_mode": "none",
        "comment": (
            "Beam half-angle at the manufacturer's specified divergence. The two band edges are "
            "the documented detector (2530 nm) and source (3700 nm) switchovers, at which the "
            "illumination geometry changes discontinuously; they are imposed, never fitted. "
            "Below 10 degrees of incidence the cone average is not applied."
        ),
    },
    "manual": {
        "name": "user-specified spectrophotometer",
        "beam_aperture_deg": 1.0,
        "aperture_band_edges_nm": [],
        "aperture_mode": "imposed",
        "aperture_bounds_deg": [0.0, 5.0],
        "aperture_min_angle_deg": 10.0,
        "crosstalk_mode": "none",
        "comment": (
            "No instrument assumed. Set beam_aperture_deg to the half-angle of your own "
            "instrument's cone at the sample; add aperture_band_edges_nm only if it changes "
            "detector or source within the measured range."
        ),
    },
}


def apply_preset(block: dict[str, Any]) -> dict[str, Any]:
    """Expand a ``preset`` key into instrument fields, letting the study override any of them.

    Returns a new dictionary; the input is not modified. A block without a ``preset`` key is
    returned unchanged, so studies written before presets existed behave exactly as they did.
    """
    if "preset" not in block:
        return dict(block)
    name = block["preset"]
    if name not in PRESETS:
        raise ValueError(
            f"unknown instrument preset {name!r}; choose one of {sorted(PRESETS)} "
            f"or omit 'preset' and give the fields explicitly"
        )
    merged = dict(PRESETS[name])
    merged.update({k: v for k, v in block.items() if k != "preset"})
    merged["preset"] = name
    return merged


def describe(name: str) -> str:
    """One paragraph about a preset, for the command line and for error messages."""
    if name not in PRESETS:
        raise ValueError(f"unknown instrument preset {name!r}")
    preset = PRESETS[name]
    edges = preset["aperture_band_edges_nm"]
    return (
        f"{name}: {preset['name']}\n"
        f"  beam half-angle {preset['beam_aperture_deg']} deg, {preset['aperture_mode']}\n"
        f"  band edges: {', '.join(f'{e:.0f} nm' for e in edges) if edges else 'none'}\n"
        f"  {preset['comment']}"
    )
