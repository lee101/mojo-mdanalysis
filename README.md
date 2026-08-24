# mojo-mdanalysis

`mojo-mdanalysis` is a standalone Mojo implementation of the compute-heavy,
array-level geometry kernels used in MDAnalysis trajectory analysis. It is
callable from Python through NumPy arrays and mirrors the names, signatures,
return shapes, dtypes, periodic-boundary behavior, and preallocated-result
contract of the covered MDAnalysis functions.

The package name is `mojomdanalysis`. For the covered subset, migration is an
import-path change:

```python
from mojomdanalysis.lib.distances import distance_array
from mojomdanalysis.analysis.rms import rmsd
```

MDAnalysis is used by the test and benchmark environments for parity. It is not
needed by the compiled kernels or by the dense Python APIs at runtime.

## Covered subset

| MDAnalysis API | Coverage |
| --- | --- |
| `MDAnalysis.lib.distances.distance_array` | all-pairs distances; single coordinates; preallocated `float64` result |
| `MDAnalysis.lib.distances.self_distance_array` | condensed upper-triangle distances in upstream order |
| `MDAnalysis.lib.distances.calc_bonds` | batched and scalar bond lengths |
| `MDAnalysis.lib.distances.calc_angles` | batched and scalar angles; upstream zero result for undefined angles |
| `MDAnalysis.lib.distances.calc_dihedrals` | signed angles in `(-pi, pi)`; upstream `nan` result for undefined dihedrals |
| `MDAnalysis.lib.distances.minimize_vectors` | `float32` and `float64`, with dtype and shape preserved |
| `MDAnalysis.analysis.rms.rmsd` | unweighted and weighted RMSD, centering, and optimal rotational superposition |
| `MDAnalysis.analysis.distances.contact_matrix` | dense NumPy and SciPy LIL return types |

The distance, bond, angle, dihedral, and contact kernels support no box,
orthorhombic boxes, and triclinic boxes in MDAnalysis's
`[lx, ly, lz, alpha, beta, gamma]` format. `minimize_vectors`, like upstream,
requires an orthorhombic or triclinic box. Geometry coordinate arguments also
accept MDAnalysis `AtomGroup`-like objects exposing `.positions`.

The repository does not port the `Universe`, trajectory readers and writers,
selection language, topology model, or stateful analysis classes such as
`RMSD`, `InterRDF`, and `HydrogenBondAnalysis`. It also does not yet implement
the neighbor-search APIs (`capped_distance` and `FastNS`) or the `OpenMP` and
`distopia` backends. The covered functions expose the upstream `backend`
parameter but currently accept only its default value, `"serial"`.
Large independent workloads are still internally distributed over Mojo's CPU
thread pool; the parameter describes API/backend selection, not a promise of
single-threaded execution.

The `"sparse"` contact-matrix result has the correct LIL type and values but is
currently formed through a dense temporary; it is API-compatible, not the
low-memory implementation intended for very large sparse systems.

## Install

Clone this repository, then create the pinned development/runtime environment
and compile the shared library:

```bash
pixi install
pixi run build
```

The build task creates `dist/libmojo-mdanalysis.so`; the Python package loads
that library from the source checkout. A standalone wheel is not currently
published. Run the upstream parity suite with:

```bash
pixi run test
```

## Usage

This example applies the minimum-image convention to an all-pairs distance
calculation and performs an optimally superposed RMSD:

```python
import numpy as np

from mojomdanalysis.analysis.rms import rmsd
from mojomdanalysis.lib.distances import distance_array

reference = np.array([[0, 0, 0], [9, 0, 0]], dtype=np.float32)
configuration = np.array([[1, 0, 0]], dtype=np.float32)
box = np.array([10, 10, 10, 90, 90, 90], dtype=np.float32)

print(distance_array(reference, configuration, box=box))
# [[1.]
#  [2.]]

shifted = reference.astype(np.float64) + [4.0, -2.0, 1.0]
print(rmsd(reference, shifted, superposition=True))
# 0.0
```

Run it after `pixi run build`, either from a Pixi shell or through
`pixi run python your_script.py`.

## Performance

These are real best-of-five wall-clock measurements produced by
`pixi run bench`. Both implementations receive the same already-contiguous
arrays, and both are warmed before timing.

Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86_64. Python 3.13.14,
MDAnalysis 2.10.0, Mojo 1.1.0.dev2026081105.

| kernel | Mojo | MDAnalysis | relative |
| --- | ---: | ---: | ---: |
| `distance_array` (2,500 x 2,500) | 8.62 ms | 29.21 ms | 3.39x faster |
| `distance_array`, ortho PBC (2,500 x 2,500) | 9.82 ms | 181.04 ms | 18.44x faster |
| `self_distance_array` (5,000) | 10.75 ms | 61.16 ms | 5.69x faster |
| `calc_bonds` (2,000,000) | 1.51 ms | 33.05 ms | 21.83x faster |
| `calc_angles` (1,000,000) | 3.55 ms | 43.90 ms | 12.36x faster |
| `calc_dihedrals` (1,000,000) | 6.21 ms | 70.75 ms | 11.38x faster |
| `minimize_vectors`, ortho PBC (2,000,000) | 15.66 ms | 260.89 ms | 16.66x faster |
| centered `rmsd` (2,000,000) | 23.05 ms | 194.28 ms | 8.43x faster |
| superposed `rmsd` (1,000,000) | 13.16 ms | 72.05 ms | 5.48x faster |
| `contact_matrix`, cutoff 5 (5,000) | 6.42 ms | 39.74 ms | 6.19x faster |

Plain distance, condensed distance, bonds, and dense contacts use SIMD with a
scalar remainder. Large all-pairs rows and batched bond, angle, and dihedral
chunks run in parallel, while smaller calls stay serial to avoid thread-launch
overhead.

No GPU path is included. The two initial sub-5x benchmark targets, plain
distance arrays and bonds, are low-arithmetic-intensity kernels dominated by
memory traffic. The higher-intensity angle and dihedral kernels were already
more than 11x faster than upstream and were deliberately left alone. A GPU path
is therefore not justified by this benchmark profile; CPU remains the only
execution device.

## How it works

All kernels and C exports live in one compilation unit,
`src/kernels.mojo`. Python allocates inputs, outputs, and the small RMSD
workspace. `ctypes` makes one shared-library call per complete operation, with
buffers crossing the ABI as integer addresses. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric
`@export("name") ... abi("C")` functions. The Mojo side does not allocate or
retain memory.

Python validates shapes, dtypes, writable outputs, alignment, and contiguity
before entering native code. It keeps every NumPy owner alive for the complete
synchronous call. Geometry coordinates are converted to contiguous, aligned
row-major `(n, 3)` `float32` arrays, matching MDAnalysis's coordinate
conversion, and distance/angle results are `float64`. RMSD operates in
`float64`. Orthorhombic minimum images use component-wise wrapping. Triclinic
boxes are converted to a lower triangular lattice and nearby lattice images
are checked for the shortest Cartesian vector.

Rotational superposition uses Horn's symmetric quaternion formulation. A
four-by-four Jacobi eigensolve obtains the optimal rotation eigenvalue, giving
the same weighted least-squares RMSD as MDAnalysis's QCP implementation without
constructing a rotation matrix.

## License

MIT
