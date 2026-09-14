"""certus_re -- reverse engineering of multilayer optical coatings from measured spectra.

The package inverts measured reflectance and transmittance spectra to retrieve the layer
thicknesses of one or several coatings, with the refractive indices supplied as an external
determination rather than adjusted along the way.

What it is for
--------------
Multilayer reverse engineering conventionally adjusts thicknesses and dispersion corrections
inside the same optimization. When the residual falls, the improvement cannot be attributed
to the thickness correction -- the physically meaningful outcome -- rather than to the index
correction, which silently rewrites the quantity one claims to validate. This package
separates the two: the parameter blocks a run is allowed to move are declared before it
starts, counted, and printed with the result.

What it handles
---------------
* several **samples inverted jointly**, sharing the coatings they have in common;
* **two-side-coated components**, with a coherent stack on each face and the substrate
  treated incoherently between them;
* any **angle of incidence**, in ``s``, ``p`` or the unpolarized half-sum;
* reflectance, transmittance, and transmittance relative to the bare substrate;
* a **piecewise-constant beam aperture**, stepped at the instrument's documented source and
  detector switchovers and imposed rather than fitted.

Entry points
------------
:func:`certus_re.io_study.load_study`
    Read a study from its JSON description.
:func:`certus_re.dof.count_free_parameters`
    The parameter budget of a study, block by block.

See ``README.md`` for the reproduction procedure and the provenance of the deposited data.
"""

from __future__ import annotations

from .dispersion import TabulatedIndex, load_index_csv
from .dof import DoFReport, ParameterBlock, count_free_parameters
from .io_study import StudyError, load_study, read_spectrum_csv
from .model import (
    FreeParameters,
    Instrument,
    Layer,
    Measurement,
    Sample,
    Stack,
    Study,
    Substrate,
)

__version__ = "1.1.1"

__all__ = [
    "DoFReport",
    "FreeParameters",
    "Instrument",
    "Layer",
    "Measurement",
    "ParameterBlock",
    "Sample",
    "Stack",
    "Study",
    "StudyError",
    "Substrate",
    "TabulatedIndex",
    "count_free_parameters",
    "load_index_csv",
    "load_study",
    "read_spectrum_csv",
    "__version__",
]
