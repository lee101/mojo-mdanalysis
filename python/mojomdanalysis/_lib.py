from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "dist" / "libmojo-mdanalysis.so"

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mda_distance_array": ([I, I, I, I, I, I, I], None),
    "mda_self_distance_array": ([I, I, I, I, I], None),
    "mda_calc_bonds": ([I, I, I, I, I, I], None),
    "mda_calc_angles": ([I, I, I, I, I, I, I], None),
    "mda_calc_dihedrals": ([I, I, I, I, I, I, I, I], None),
    "mda_minimize_vectors_f32": ([I, I, I, I, I], None),
    "mda_minimize_vectors_f64": ([I, I, I, I, I], None),
    "mda_contact_matrix": ([I, I, I, F, I, I], None),
    "mda_rmsd": ([I, I, I, I, I, I, I, I], F),
}

_loaded: ctypes.CDLL | None = None

def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        if not LIBRARY.exists():
            raise RuntimeError(
                f"Mojo library not found at {LIBRARY}; run `pixi run build`"
            )
        _loaded = ctypes.CDLL(str(LIBRARY))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null pointer for a non-empty array")
    return address


