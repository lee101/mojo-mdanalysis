from __future__ import annotations

import numpy as np

from .._lib import addr, lib


def rmsd(a, b, weights=None, center=False, superposition=False):
    raw_a = np.asarray(a)
    raw_b = np.asarray(b)
    if np.issubdtype(raw_a.dtype, np.complexfloating) or np.issubdtype(
        raw_b.dtype, np.complexfloating
    ):
        raise TypeError("a and b must contain real-valued coordinates")
    a = np.require(raw_a, dtype=np.float64, requirements=("C", "A"))
    b = np.require(raw_b, dtype=np.float64, requirements=("C", "A"))
    if a.shape != b.shape:
        raise ValueError("a and b must have same shape")
    if a.ndim != 2 or a.shape[1:] != (3,):
        raise ValueError("a and b must have shape (n, 3)")
    if weights is None:
        weight_array = np.ones(max(len(a), 1), dtype=np.float64)
        weighted = 0
    else:
        raw_weights = np.asarray(weights)
        if np.issubdtype(raw_weights.dtype, np.complexfloating):
            raise TypeError("weights must be real-valued")
        weight_array = np.require(
            raw_weights, dtype=np.float64, requirements=("C", "A")
        )
        if weight_array.ndim != 1 or len(weight_array) != len(a):
            raise ValueError("weights must have same length as a and b")
        if weight_array.sum() == 0.0:
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        weighted = 1
    work = np.empty(16, dtype=np.float64)
    return lib().mda_rmsd(
        addr(a),
        addr(b),
        addr(weight_array),
        addr(work),
        len(a),
        weighted,
        int(bool(center)),
        int(bool(superposition)),
    )
