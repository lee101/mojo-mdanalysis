from __future__ import annotations

import numpy as np

from .._lib import addr, lib
from ..lib.distances import _box_data, _coordinates


def contact_matrix(coord, cutoff=15.0, returntype="numpy", box=None):
    coordinates, _ = _coordinates(coord, "coord")
    dense = np.empty((len(coordinates), len(coordinates)), dtype=np.bool_)
    if dense.size:
        box_data, mode = _box_data(box)
        lib().mda_contact_matrix(
            addr(coordinates),
            addr(dense),
            len(coordinates),
            float(cutoff),
            addr(box_data),
            mode,
        )
    if returntype == "numpy":
        return dense
    if returntype == "sparse":
        from scipy.sparse import lil_matrix

        return lil_matrix(dense)
    raise ValueError(
        f"returntype must be 'numpy' or 'sparse', not {returntype!r}"
    )
