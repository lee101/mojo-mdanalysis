import numpy as np
import pytest

MDAnalysis = pytest.importorskip("MDAnalysis")
from MDAnalysis.analysis.distances import contact_matrix as upstream_contacts
from MDAnalysis.analysis.rms import rmsd as upstream_rmsd
from MDAnalysis.lib import distances as upstream

import mojomdanalysis as mojo


ORTHO = np.array([23.0, 27.0, 31.0, 90.0, 90.0, 90.0])
TRICLINIC = np.array([23.0, 27.0, 31.0, 72.0, 83.0, 76.0])
SKEW_TRICLINIC = np.array(
    [55.37764714, 97.74822795, 90.39235210, 64.06978464, 120.21080357, 56.18859357]
)


@pytest.fixture(scope="module")
def coordinates():
    rng = np.random.default_rng(741)
    return (
        rng.uniform(-35.0, 40.0, size=(73, 3)),
        rng.uniform(-30.0, 45.0, size=(51, 3)),
        rng.uniform(-25.0, 50.0, size=(73, 3)),
        rng.uniform(-20.0, 55.0, size=(73, 3)),
    )


@pytest.mark.parametrize("box", [None, ORTHO, TRICLINIC])
def test_distance_array_parity(coordinates, box):
    a, b, _, _ = coordinates
    actual = mojo.distance_array(a, b, box=box)
    expected = upstream.distance_array(a, b, box=box)
    assert actual.dtype == np.float64
    assert actual.shape == (73, 51)
    assert np.allclose(actual, expected, rtol=2e-6, atol=4e-6)


def test_distance_array_single_and_preallocated():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([[3.0, 2.0, 1.0], [1.0, 5.0, 3.0]])
    target = np.empty((1, 2), dtype=np.float64)
    returned = mojo.distance_array(a, b, result=target)
    assert returned is target
    assert np.array_equal(returned, upstream.distance_array(a, b))


@pytest.mark.parametrize("box", [None, ORTHO, TRICLINIC])
def test_self_distance_array_parity(coordinates, box):
    a = coordinates[0]
    actual = mojo.self_distance_array(a, box=box)
    expected = upstream.self_distance_array(a, box=box)
    assert actual.shape == (len(a) * (len(a) - 1) // 2,)
    assert np.allclose(actual, expected, rtol=2e-6, atol=4e-6)


def test_self_distance_condensed_order():
    xyz = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=np.float32)
    assert np.array_equal(mojo.self_distance_array(xyz), np.array([3.0, 4.0, 5.0]))


def test_simd_tail_paths():
    rng = np.random.default_rng(184)
    a = rng.uniform(-5.0, 5.0, size=(7, 3)).astype(np.float32)
    b = rng.uniform(-5.0, 5.0, size=(11, 3)).astype(np.float32)
    assert np.allclose(
        mojo.distance_array(a, b),
        upstream.distance_array(a, b),
        rtol=2e-6,
        atol=4e-6,
    )
    assert np.allclose(
        mojo.self_distance_array(b),
        upstream.self_distance_array(b),
        rtol=2e-6,
        atol=4e-6,
    )
    assert np.array_equal(
        mojo.contact_matrix(b, cutoff=3.75),
        upstream_contacts(b, cutoff=3.75),
    )
    assert np.allclose(
        mojo.calc_bonds(a, a[::-1]),
        upstream.calc_bonds(a, a[::-1]),
        rtol=2e-6,
        atol=4e-6,
    )


def test_parallel_pair_paths():
    rng = np.random.default_rng(918)
    a = rng.uniform(-15.0, 15.0, size=(513, 3)).astype(np.float32)
    actual_distances = mojo.distance_array(a, a)
    expected_distances = upstream.distance_array(a, a)
    assert np.allclose(
        actual_distances, expected_distances, rtol=2e-6, atol=4e-6
    )
    actual_contacts = mojo.contact_matrix(a, cutoff=4.25)
    expected_contacts = upstream_contacts(a, cutoff=4.25)
    assert np.array_equal(actual_contacts, expected_contacts)

    b = rng.uniform(-15.0, 15.0, size=(725, 3)).astype(np.float32)
    assert np.allclose(
        mojo.self_distance_array(b),
        upstream.self_distance_array(b),
        rtol=2e-6,
        atol=4e-6,
    )


def test_parallel_element_paths():
    rng = np.random.default_rng(321)
    values = [
        rng.uniform(-20.0, 20.0, size=(65_536, 3)).astype(np.float32)
        for _ in range(4)
    ]
    assert np.allclose(
        mojo.calc_bonds(values[0], values[1]),
        upstream.calc_bonds(values[0], values[1]),
        rtol=2e-6,
        atol=4e-6,
    )
    assert np.allclose(
        mojo.calc_bonds(values[0], values[1], box=ORTHO),
        upstream.calc_bonds(values[0], values[1], box=ORTHO),
        rtol=2e-6,
        atol=4e-6,
    )
    assert np.allclose(
        mojo.calc_angles(values[0], values[1], values[2]),
        upstream.calc_angles(values[0], values[1], values[2]),
        rtol=2e-6,
        atol=5e-6,
    )
    assert np.allclose(
        mojo.calc_dihedrals(*values),
        upstream.calc_dihedrals(*values),
        rtol=3e-6,
        atol=8e-6,
        equal_nan=True,
    )


@pytest.mark.parametrize("box", [None, ORTHO, TRICLINIC])
def test_calc_bonds_parity(coordinates, box):
    a, _, c, _ = coordinates
    actual = mojo.calc_bonds(a, c, box=box)
    expected = upstream.calc_bonds(a, c, box=box)
    assert np.allclose(actual, expected, rtol=2e-6, atol=4e-6)


@pytest.mark.parametrize("box", [None, ORTHO, TRICLINIC])
def test_calc_angles_parity(coordinates, box):
    a, _, c, d = coordinates
    actual = mojo.calc_angles(a, c, d, box=box)
    expected = upstream.calc_angles(a, c, d, box=box)
    assert np.allclose(actual, expected, rtol=2e-6, atol=5e-6)


@pytest.mark.parametrize("box", [None, ORTHO, TRICLINIC])
def test_calc_dihedrals_parity(coordinates, box):
    a, _, c, d = coordinates
    fourth = np.roll(a, 7, axis=0)
    actual = mojo.calc_dihedrals(a, c, d, fourth, box=box)
    expected = upstream.calc_dihedrals(a, c, d, fourth, box=box)
    assert np.allclose(actual, expected, rtol=3e-6, atol=8e-6, equal_nan=True)


def test_geometry_single_values_and_degenerate_cases():
    origin = np.zeros(3)
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([1.0, 0.0, 1.0])
    assert isinstance(mojo.calc_bonds(origin, x), np.float64)
    assert mojo.calc_angles(x, origin, y) == pytest.approx(np.pi / 2)
    assert mojo.calc_angles(origin, origin, y) == 0.0
    assert mojo.calc_dihedrals(y, origin, x, z) == pytest.approx(np.pi / 2)
    assert np.isnan(mojo.calc_dihedrals(origin, x, 2 * x, 3 * x))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("box", [ORTHO, TRICLINIC, SKEW_TRICLINIC])
def test_minimize_vectors_parity_and_dtype(coordinates, dtype, box):
    vectors = (coordinates[0] * 2.7).astype(dtype)
    actual = mojo.minimize_vectors(vectors, box)
    expected = upstream.minimize_vectors(vectors, box)
    tolerance = 3e-4 if dtype == np.float32 else 2e-12
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, rtol=2e-6, atol=tolerance)
    assert not np.shares_memory(actual, vectors)


def test_minimize_single_vector():
    vector = np.array([20.0, -20.0, 16.0], dtype=np.float64)
    actual = mojo.minimize_vectors(vector, ORTHO)
    expected = upstream.minimize_vectors(vector, ORTHO)
    assert actual.shape == (3,)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    ("center", "superposition"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_rmsd_parity(coordinates, center, superposition):
    a, _, c, _ = coordinates
    before = a.copy()
    actual = mojo.rmsd(a, c, center=center, superposition=superposition)
    expected = upstream_rmsd(a, c, center=center, superposition=superposition)
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)
    assert np.array_equal(a, before)


def test_weighted_rmsd_parity(coordinates):
    a, _, c, _ = coordinates
    weights = np.linspace(0.5, 4.0, len(a))
    for superposition in (False, True):
        actual = mojo.rmsd(
            a, c, weights=weights, center=True, superposition=superposition
        )
        expected = upstream_rmsd(
            a, c, weights=weights, center=True, superposition=superposition
        )
        assert actual == pytest.approx(expected, rel=3e-12, abs=3e-12)


def test_superposition_removes_rigid_transform():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(100, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    b = a @ q + np.array([7.0, -2.0, 5.0])
    assert mojo.rmsd(a, b, superposition=True) == pytest.approx(
        upstream_rmsd(a, b, superposition=True), abs=1e-7
    )
    assert mojo.rmsd(a, b, superposition=True) < 1e-6


@pytest.mark.parametrize("box", [None, ORTHO, TRICLINIC])
def test_contact_matrix_parity(coordinates, box):
    coord = coordinates[0].astype(np.float32)
    actual = mojo.contact_matrix(coord, cutoff=7.25, box=box)
    expected = upstream_contacts(coord, cutoff=7.25, box=box)
    assert actual.dtype == np.bool_
    assert np.array_equal(actual, expected)


def test_sparse_contact_matrix_parity(coordinates):
    coord = coordinates[0].astype(np.float32)
    actual = mojo.contact_matrix(coord, cutoff=6.0, returntype="sparse")
    expected = upstream_contacts(coord, cutoff=6.0, returntype="sparse")
    assert actual.getformat() == "lil"
    assert np.array_equal(actual.toarray(), expected.toarray())


def test_atomgroup_coordinates_are_accepted():
    universe = MDAnalysis.Universe.empty(4, trajectory=True)
    universe.atoms.positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32
    )
    assert np.array_equal(
        mojo.self_distance_array(universe.atoms),
        upstream.self_distance_array(universe.atoms),
    )


def test_empty_inputs():
    empty = np.empty((0, 3), dtype=np.float32)
    assert mojo.distance_array(empty, empty).shape == (0, 0)
    assert mojo.self_distance_array(empty).shape == (0,)
    assert mojo.calc_bonds(empty, empty).shape == (0,)
    assert mojo.contact_matrix(empty).shape == (0, 0)


def test_validation_errors():
    points = np.zeros((2, 3))
    with pytest.raises(ValueError):
        mojo.distance_array(np.zeros((2, 2)), points)
    with pytest.raises(TypeError):
        mojo.distance_array(points, points, result=np.empty((2, 2), np.float32))
    with pytest.raises(ValueError):
        mojo.distance_array(points, points, result=np.empty(4))
    with pytest.raises(ValueError):
        mojo.calc_bonds(points, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        mojo.distance_array(points, points, box=[1, 2, 3])
    with pytest.raises(ValueError):
        mojo.distance_array(points, points, backend="OpenMP")
    with pytest.raises(ValueError):
        mojo.contact_matrix(points, returntype="other")
    readonly = np.empty((2, 2), dtype=np.float64)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        mojo.distance_array(points, points, result=readonly)
    storage = np.empty(4 * 8 + 1, dtype=np.uint8)
    unaligned = np.ndarray((2, 2), dtype=np.float64, buffer=storage, offset=1)
    assert not unaligned.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        mojo.distance_array(points, points, result=unaligned)
    with pytest.raises(TypeError, match="real-valued"):
        mojo.distance_array(points.astype(np.complex128), points)
    with pytest.raises(ZeroDivisionError):
        mojo.rmsd(points, points, weights=np.zeros(2), center=True)


def test_strided_and_unaligned_inputs_are_copied_safely():
    storage = np.arange(6 * 8 + 1, dtype=np.uint8)
    unaligned = np.ndarray((2, 3), dtype=np.float64, buffer=storage, offset=1)
    strided = np.arange(18.0).reshape(3, 6)[:, ::2]
    assert not unaligned.flags.aligned
    assert not strided.flags.c_contiguous
    assert np.allclose(
        mojo.distance_array(unaligned, strided),
        upstream.distance_array(unaligned, strided),
    )


def test_public_module_layout():
    from mojomdanalysis.analysis.distances import contact_matrix
    from mojomdanalysis.analysis.rms import rmsd
    from mojomdanalysis.lib.distances import distance_array

    assert contact_matrix is mojo.contact_matrix
    assert rmsd is mojo.rmsd
    assert distance_array is mojo.distance_array
