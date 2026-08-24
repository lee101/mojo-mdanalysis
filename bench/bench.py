from __future__ import annotations

import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import MDAnalysis
from MDAnalysis.analysis.distances import contact_matrix as mda_contacts
from MDAnalysis.analysis.rms import rmsd as mda_rmsd
from MDAnalysis.lib import distances as mda_distances

import mojomdanalysis as mojo


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def coordinates(n, seed):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        rng.uniform(-50.0, 50.0, size=(n, 3)), dtype=np.float32
    )


CASES = []


def case(name):
    def register(builder):
        CASES.append((name, builder))
        return builder

    return register


@case("distance_array (2,500 x 2,500)")
def _distance_array():
    a, b = coordinates(2_500, 1), coordinates(2_500, 2)
    return (
        lambda: mojo.distance_array(a, b),
        lambda: mda_distances.distance_array(a, b),
    )


@case("distance_array ortho PBC (2,500 x 2,500)")
def _distance_array_pbc():
    a, b = coordinates(2_500, 3), coordinates(2_500, 4)
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    return (
        lambda: mojo.distance_array(a, b, box=box),
        lambda: mda_distances.distance_array(a, b, box=box),
    )


@case("self_distance_array (5,000)")
def _self_distance_array():
    a = coordinates(5_000, 5)
    return (
        lambda: mojo.self_distance_array(a),
        lambda: mda_distances.self_distance_array(a),
    )


@case("calc_bonds (2,000,000)")
def _bonds():
    a, b = coordinates(2_000_000, 6), coordinates(2_000_000, 7)
    return (
        lambda: mojo.calc_bonds(a, b),
        lambda: mda_distances.calc_bonds(a, b),
    )


@case("calc_angles (1,000,000)")
def _angles():
    a = coordinates(1_000_000, 8)
    b = coordinates(1_000_000, 9)
    c = coordinates(1_000_000, 10)
    return (
        lambda: mojo.calc_angles(a, b, c),
        lambda: mda_distances.calc_angles(a, b, c),
    )


@case("calc_dihedrals (1,000,000)")
def _dihedrals():
    a = coordinates(1_000_000, 11)
    b = coordinates(1_000_000, 12)
    c = coordinates(1_000_000, 13)
    d = coordinates(1_000_000, 14)
    return (
        lambda: mojo.calc_dihedrals(a, b, c, d),
        lambda: mda_distances.calc_dihedrals(a, b, c, d),
    )


@case("minimize_vectors ortho PBC (2,000,000)")
def _minimize():
    vectors = coordinates(2_000_000, 15)
    box = np.array([37.0, 41.0, 43.0, 90.0, 90.0, 90.0])
    return (
        lambda: mojo.minimize_vectors(vectors, box),
        lambda: mda_distances.minimize_vectors(vectors, box),
    )


@case("rmsd centered (2,000,000)")
def _rmsd_centered():
    a = coordinates(2_000_000, 16).astype(np.float64)
    b = coordinates(2_000_000, 17).astype(np.float64)
    return (
        lambda: mojo.rmsd(a, b, center=True),
        lambda: mda_rmsd(a, b, center=True),
    )


@case("rmsd superposition (1,000,000)")
def _rmsd_superposition():
    a = coordinates(1_000_000, 18).astype(np.float64)
    b = coordinates(1_000_000, 19).astype(np.float64)
    return (
        lambda: mojo.rmsd(a, b, superposition=True),
        lambda: mda_rmsd(a, b, superposition=True),
    )


@case("contact_matrix cutoff=5 (5,000)")
def _contacts():
    a = coordinates(5_000, 20)
    return (
        lambda: mojo.contact_matrix(a, cutoff=5.0),
        lambda: mda_contacts(a, cutoff=5.0),
    )


def cpu_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print(
        f"Python {platform.python_version()}; MDAnalysis {MDAnalysis.__version__}; "
        "Mojo 1.1.0.dev2026081105"
    )
    print()
    print("| kernel | Mojo | MDAnalysis | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, builder in CASES:
        ours, theirs = builder()
        ours()
        theirs()
        mojo_seconds = best_time(ours)
        upstream_seconds = best_time(theirs)
        ratio = upstream_seconds / mojo_seconds
        label = "faster" if ratio >= 1.0 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
