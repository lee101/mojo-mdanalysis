from __future__ import annotations

import numpy as np

from .._lib import addr, lib

_NO_BOX = np.zeros(18, dtype=np.float64)


def _coordinates(value, name: str) -> tuple[np.ndarray, bool]:
    if hasattr(value, "positions"):
        value = value.positions
    original = np.asarray(value)
    single = original.ndim == 1 and original.shape == (3,)
    if single:
        original = original.reshape(1, 3)
    if original.ndim != 2 or original.shape[1:] != (3,):
        raise ValueError(
            f"{name} must have shape (3,) or (n, 3), got {original.shape}"
        )
    if np.issubdtype(original.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real-valued coordinates")
    # MDAnalysis performs geometry in float32.  np.require also guarantees the
    # alignment assumed by Mojo's SIMD loads; ascontiguousarray alone does not.
    return np.require(original, dtype=np.float32, requirements=("C", "A")), single


def _result(result, shape: tuple[int, ...]) -> np.ndarray:
    if result is None:
        return np.empty(shape, dtype=np.float64)
    if not isinstance(result, np.ndarray):
        raise TypeError("result must be a numpy.ndarray")
    if result.dtype != np.float64:
        raise TypeError(
            f"Result array must be of type numpy.float64, got {result.dtype}."
        )
    if result.shape != shape:
        raise ValueError(
            f"Result array has incorrect shape, should be {shape}, got {result.shape}."
        )
    if not result.flags.c_contiguous:
        raise ValueError("Result array must be C-contiguous")
    if not result.flags.aligned:
        raise ValueError("Result array must be aligned")
    if not result.flags.writeable:
        raise ValueError("Result array must be writable")
    return result


def _box_data(box) -> tuple[np.ndarray, int]:
    if box is None:
        return _NO_BOX, 0
    storage = np.zeros(18, dtype=np.float64)
    original = np.asarray(box)
    if np.issubdtype(original.dtype, np.complexfloating):
        raise TypeError("box dimensions must be real-valued")
    # Upstream consumes dimensions as float32 before constructing its box matrix.
    dimensions = np.asarray(original, dtype=np.float32).astype(np.float64)
    if dimensions.shape != (6,):
        raise ValueError(
            "Invalid box information. Must be of the form "
            "[lx, ly, lz, alpha, beta, gamma]."
        )
    lx, ly, lz, alpha, beta, gamma = dimensions
    if np.array_equal(dimensions[3:], np.array([90.0, 90.0, 90.0])):
        if min(lx, ly, lz) <= 0.0:
            raise ValueError("box lengths must be positive")
        matrix = np.diag(dimensions[:3])
        storage[:9] = matrix.ravel()
        storage[9:] = np.linalg.inv(matrix).ravel()
        return storage, 1
    if (
        min(lx, ly, lz, alpha, beta, gamma) <= 0.0
        or max(alpha, beta, gamma) >= 180.0
    ):
        raise ValueError("box lengths and angles must define a valid unit cell")
    ar, br, gr = np.deg2rad([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos([ar, br, gr])
    sin_g = np.sin(gr)
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0, 0] = lx
    matrix[1, 0] = ly * cos_g
    matrix[1, 1] = ly * sin_g
    matrix[2, 0] = lz * cos_b
    matrix[2, 1] = lz * (cos_a - cos_b * cos_g) / sin_g
    discriminant = lz * lz - matrix[2, 0] ** 2 - matrix[2, 1] ** 2
    if not discriminant > 0.0:
        raise ValueError("box angles do not define a valid unit cell")
    matrix[2, 2] = np.sqrt(discriminant)
    matrix = matrix.astype(np.float32).astype(np.float64)
    storage[:9] = matrix.ravel()
    storage[9:] = np.linalg.inv(matrix).ravel()
    return storage, 2


def _backend(backend: str) -> None:
    if str(backend).lower() != "serial":
        raise ValueError(
            "mojomdanalysis implements the 'serial' backend; "
            f"backend {backend!r} is not available"
        )


def distance_array(
    reference,
    configuration,
    box=None,
    result=None,
    backend="serial",
):
    _backend(backend)
    reference, _ = _coordinates(reference, "reference")
    configuration, _ = _coordinates(configuration, "configuration")
    target = _result(result, (len(reference), len(configuration)))
    if target.size:
        box_data, mode = _box_data(box)
        lib().mda_distance_array(
            addr(reference),
            addr(configuration),
            addr(target),
            len(reference),
            len(configuration),
            addr(box_data),
            mode,
        )
    return target


def self_distance_array(reference, box=None, result=None, backend="serial"):
    _backend(backend)
    reference, _ = _coordinates(reference, "reference")
    count = len(reference) * (len(reference) - 1) // 2
    target = _result(result, (count,))
    if count:
        box_data, mode = _box_data(box)
        lib().mda_self_distance_array(
            addr(reference), addr(target), len(reference), addr(box_data), mode
        )
    return target


def _paired(inputs, names, result, backend):
    _backend(backend)
    converted = [_coordinates(value, name) for value, name in zip(inputs, names)]
    lengths = {len(value[0]) for value in converted}
    if len(lengths) != 1:
        raise ValueError("coordinate arrays must contain the same number of points")
    single = all(value[1] for value in converted)
    count = len(converted[0][0])
    return [value[0] for value in converted], _result(result, (count,)), single


def calc_bonds(coords1, coords2, box=None, result=None, backend="serial"):
    values, target, single = _paired(
        (coords1, coords2), ("coords1", "coords2"), result, backend
    )
    if target.size:
        box_data, mode = _box_data(box)
        lib().mda_calc_bonds(
            addr(values[0]),
            addr(values[1]),
            addr(target),
            len(target),
            addr(box_data),
            mode,
        )
    return target[0] if single else target


def calc_angles(
    coords1, coords2, coords3, box=None, result=None, backend="serial"
):
    values, target, single = _paired(
        (coords1, coords2, coords3),
        ("coords1", "coords2", "coords3"),
        result,
        backend,
    )
    if target.size:
        box_data, mode = _box_data(box)
        lib().mda_calc_angles(
            addr(values[0]),
            addr(values[1]),
            addr(values[2]),
            addr(target),
            len(target),
            addr(box_data),
            mode,
        )
    return target[0] if single else target


def calc_dihedrals(
    coords1,
    coords2,
    coords3,
    coords4,
    box=None,
    result=None,
    backend="serial",
):
    values, target, single = _paired(
        (coords1, coords2, coords3, coords4),
        ("coords1", "coords2", "coords3", "coords4"),
        result,
        backend,
    )
    if target.size:
        box_data, mode = _box_data(box)
        lib().mda_calc_dihedrals(
            addr(values[0]),
            addr(values[1]),
            addr(values[2]),
            addr(values[3]),
            addr(target),
            len(target),
            addr(box_data),
            mode,
        )
    return target[0] if single else target


def minimize_vectors(vectors, box):
    original = np.asarray(vectors)
    if original.dtype not in (np.float32, np.float64):
        raise TypeError("vectors must have dtype float32 or float64")
    values, single = _coordinates_preserving_dtype(original, "vectors")
    target = np.empty_like(values)
    if target.size:
        box_data, mode = _box_data(box)
        function = (
            lib().mda_minimize_vectors_f32
            if values.dtype == np.float32
            else lib().mda_minimize_vectors_f64
        )
        function(addr(values), addr(target), len(values), addr(box_data), mode)
    return target.reshape(3) if single else target


def _coordinates_preserving_dtype(value, name):
    single = value.ndim == 1 and value.shape == (3,)
    if single:
        value = value.reshape(1, 3)
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (3,) or (n, 3), got {value.shape}")
    return np.require(value, requirements=("C", "A")), single
