from std.algorithm import parallelize
from std.math import acos, atan2, floor, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime PARALLEL_PAIR_THRESHOLD = 262144
comptime PARALLEL_ELEMENT_THRESHOLD = 65536
comptime PARALLEL_CHUNK_SIZE = 4096


def nearest_integer(value: Float64) -> Float64:
    if value < 0.0:
        return -floor(-value + 0.5)
    return floor(value + 0.5)


def mic(
    x_in: Float64, y_in: Float64, z_in: Float64, box: F64Ptr, mode: Int
) -> Tuple[Float64, Float64, Float64]:
    var x = x_in
    var y = y_in
    var z = z_in
    if mode == 0:
        return (x, y, z)
    if mode == 1:
        x -= nearest_integer(x / box[0]) * box[0]
        y -= nearest_integer(y / box[4]) * box[4]
        z -= nearest_integer(z / box[8]) * box[8]
        return (x, y, z)

    var shift = nearest_integer(z / box[8])
    x -= shift * box[6]
    y -= shift * box[7]
    z -= shift * box[8]
    shift = nearest_integer(y / box[4])
    x -= shift * box[3]
    y -= shift * box[4]
    x -= nearest_integer(x / box[0]) * box[0]

    var best_x = x
    var best_y = y
    var best_z = z
    var best_d2 = best_x * best_x + best_y * best_y + best_z * best_z
    for ii in range(-1, 2):
        for jj in range(-1, 2):
            for kk in range(-1, 2):
                var cx = (
                    x
                    + Float64(ii) * box[0]
                    + Float64(jj) * box[3]
                    + Float64(kk) * box[6]
                )
                var cy = y + Float64(jj) * box[4] + Float64(kk) * box[7]
                var cz = z + Float64(kk) * box[8]
                var d2 = cx * cx + cy * cy + cz * cz
                if d2 < best_d2:
                    best_d2 = d2
                    best_x = cx
                    best_y = cy
                    best_z = cz
    return (best_x, best_y, best_z)


def delta_f32(
    a: F32Ptr, ai: Int, b: F32Ptr, bi: Int, box: F64Ptr, mode: Int
) -> Tuple[Float64, Float64, Float64]:
    return mic(
        Float64(a[3 * ai]) - Float64(b[3 * bi]),
        Float64(a[3 * ai + 1]) - Float64(b[3 * bi + 1]),
        Float64(a[3 * ai + 2]) - Float64(b[3 * bi + 2]),
        box,
        mode,
    )


def distance_array(
    reference: F32Ptr,
    configuration: F32Ptr,
    dst: F64Ptr,
    n: Int,
    m: Int,
    box: F64Ptr,
    mode: Int,
):
    @parameter
    def row(i: Int):
        if mode == 0:
            comptime W = simdwidthof[DType.float64]()
            var rx = Float64(reference[3 * i])
            var ry = Float64(reference[3 * i + 1])
            var rz = Float64(reference[3 * i + 2])
            var j = 0
            while j + W <= m:
                var x = (
                    SIMD[DType.float64, W](rx)
                    - (configuration + 3 * j)
                    .strided_load[width=W](3)
                    .cast[DType.float64]()
                )
                var y = (
                    SIMD[DType.float64, W](ry)
                    - (configuration + 3 * j + 1)
                    .strided_load[width=W](3)
                    .cast[DType.float64]()
                )
                var z = (
                    SIMD[DType.float64, W](rz)
                    - (configuration + 3 * j + 2)
                    .strided_load[width=W](3)
                    .cast[DType.float64]()
                )
                dst.store(i * m + j, sqrt(x * x + y * y + z * z))
                j += W
            while j < m:
                var x = rx - Float64(configuration[3 * j])
                var y = ry - Float64(configuration[3 * j + 1])
                var z = rz - Float64(configuration[3 * j + 2])
                dst[i * m + j] = sqrt(x * x + y * y + z * z)
                j += 1
            return
        for j in range(m):
            var x, y, z = delta_f32(reference, i, configuration, j, box, mode)
            dst[i * m + j] = sqrt(x * x + y * y + z * z)

    if n * m >= PARALLEL_PAIR_THRESHOLD:
        parallelize[row](n)
    else:
        for i in range(n):
            row(i)


def self_distance_array(
    coordinates: F32Ptr, dst: F64Ptr, n: Int, box: F64Ptr, mode: Int
):
    @parameter
    def row(i: Int):
        var k = i * (2 * n - i - 1) // 2
        var j = i + 1
        if mode == 0:
            comptime W = simdwidthof[DType.float64]()
            var rx = Float64(coordinates[3 * i])
            var ry = Float64(coordinates[3 * i + 1])
            var rz = Float64(coordinates[3 * i + 2])
            while j + W <= n:
                var x = (
                    SIMD[DType.float64, W](rx)
                    - (coordinates + 3 * j)
                    .strided_load[width=W](3)
                    .cast[DType.float64]()
                )
                var y = (
                    SIMD[DType.float64, W](ry)
                    - (coordinates + 3 * j + 1)
                    .strided_load[width=W](3)
                    .cast[DType.float64]()
                )
                var z = (
                    SIMD[DType.float64, W](rz)
                    - (coordinates + 3 * j + 2)
                    .strided_load[width=W](3)
                    .cast[DType.float64]()
                )
                dst.store(k, sqrt(x * x + y * y + z * z))
                j += W
                k += W
            while j < n:
                var x = rx - Float64(coordinates[3 * j])
                var y = ry - Float64(coordinates[3 * j + 1])
                var z = rz - Float64(coordinates[3 * j + 2])
                dst[k] = sqrt(x * x + y * y + z * z)
                j += 1
                k += 1
            return
        while j < n:
            var x, y, z = delta_f32(coordinates, i, coordinates, j, box, mode)
            dst[k] = sqrt(x * x + y * y + z * z)
            j += 1
            k += 1

    var count = n * (n - 1) // 2
    if count >= PARALLEL_PAIR_THRESHOLD:
        parallelize[row](n - 1)
    else:
        for i in range(n - 1):
            row(i)


def bonds(
    a: F32Ptr, b: F32Ptr, dst: F64Ptr, n: Int, box: F64Ptr, mode: Int
):
    for i in range(n):
        var x, y, z = delta_f32(a, i, b, i, box, mode)
        dst[i] = sqrt(x * x + y * y + z * z)


def angles(
    a: F32Ptr,
    b: F32Ptr,
    c: F32Ptr,
    dst: F64Ptr,
    n: Int,
    box: F64Ptr,
    mode: Int,
):
    @parameter
    def element(i: Int):
        var ux, uy, uz = delta_f32(a, i, b, i, box, mode)
        var vx, vy, vz = delta_f32(c, i, b, i, box, mode)
        var uu = ux * ux + uy * uy + uz * uz
        var vv = vx * vx + vy * vy + vz * vz
        if uu == 0.0 or vv == 0.0:
            dst[i] = 0.0
        else:
            var cosine = (ux * vx + uy * vy + uz * vz) / sqrt(uu * vv)
            if cosine > 1.0:
                cosine = 1.0
            elif cosine < -1.0:
                cosine = -1.0
            dst[i] = acos(cosine)

    @parameter
    def chunk(chunk_index: Int):
        var begin = chunk_index * PARALLEL_CHUNK_SIZE
        var end = begin + PARALLEL_CHUNK_SIZE
        if end > n:
            end = n
        for i in range(begin, end):
            element(i)

    if n >= PARALLEL_ELEMENT_THRESHOLD:
        parallelize[chunk](
            (n + PARALLEL_CHUNK_SIZE - 1) // PARALLEL_CHUNK_SIZE
        )
    else:
        for i in range(n):
            element(i)


def dihedrals(
    a: F32Ptr,
    b: F32Ptr,
    c: F32Ptr,
    d: F32Ptr,
    dst: F64Ptr,
    n: Int,
    box: F64Ptr,
    mode: Int,
):
    @parameter
    def element(i: Int):
        var ux, uy, uz = delta_f32(b, i, a, i, box, mode)
        var vx, vy, vz = delta_f32(c, i, b, i, box, mode)
        var wx, wy, wz = delta_f32(d, i, c, i, box, mode)
        var n1x = uy * vz - uz * vy
        var n1y = uz * vx - ux * vz
        var n1z = ux * vy - uy * vx
        var n2x = vy * wz - vz * wy
        var n2y = vz * wx - vx * wz
        var n2z = vx * wy - vy * wx
        var n1sq = n1x * n1x + n1y * n1y + n1z * n1z
        var n2sq = n2x * n2x + n2y * n2y + n2z * n2z
        var vsq = vx * vx + vy * vy + vz * vz
        if n1sq == 0.0 or n2sq == 0.0 or vsq == 0.0:
            dst[i] = sqrt(-1.0)
        else:
            var cx = n1y * n2z - n1z * n2y
            var cy = n1z * n2x - n1x * n2z
            var cz = n1x * n2y - n1y * n2x
            var sine = (cx * vx + cy * vy + cz * vz) / sqrt(vsq)
            var cosine = n1x * n2x + n1y * n2y + n1z * n2z
            dst[i] = atan2(sine, cosine)

    @parameter
    def chunk(chunk_index: Int):
        var begin = chunk_index * PARALLEL_CHUNK_SIZE
        var end = begin + PARALLEL_CHUNK_SIZE
        if end > n:
            end = n
        for i in range(begin, end):
            element(i)

    if n >= PARALLEL_ELEMENT_THRESHOLD:
        parallelize[chunk](
            (n + PARALLEL_CHUNK_SIZE - 1) // PARALLEL_CHUNK_SIZE
        )
    else:
        for i in range(n):
            element(i)


def minimize_f32(vectors: F32Ptr, dst: F32Ptr, n: Int, box: F64Ptr, mode: Int):
    for i in range(n):
        var x, y, z = mic(
            Float64(vectors[3 * i]),
            Float64(vectors[3 * i + 1]),
            Float64(vectors[3 * i + 2]),
            box,
            mode,
        )
        dst[3 * i] = Float32(x)
        dst[3 * i + 1] = Float32(y)
        dst[3 * i + 2] = Float32(z)


def minimize_f64(vectors: F64Ptr, dst: F64Ptr, n: Int, box: F64Ptr, mode: Int):
    for i in range(n):
        var x, y, z = mic(
            vectors[3 * i], vectors[3 * i + 1], vectors[3 * i + 2], box, mode
        )
        dst[3 * i] = x
        dst[3 * i + 1] = y
        dst[3 * i + 2] = z


def contact_matrix(
    coordinates: F32Ptr,
    dst: U8Ptr,
    n: Int,
    cutoff: Float64,
    box: F64Ptr,
    mode: Int,
):
    if cutoff < 0.0:
        for i in range(n * n):
            dst[i] = 0
        return
    var cutoff2 = cutoff * cutoff

    @parameter
    def full_plain_row(i: Int):
        comptime W = simdwidthof[DType.float64]()
        var rx = Float64(coordinates[3 * i])
        var ry = Float64(coordinates[3 * i + 1])
        var rz = Float64(coordinates[3 * i + 2])
        var limit = SIMD[DType.float64, W](cutoff2)
        var j = 0
        while j + W <= n:
            var x = (
                SIMD[DType.float64, W](rx)
                - (coordinates + 3 * j)
                .strided_load[width=W](3)
                .cast[DType.float64]()
            )
            var y = (
                SIMD[DType.float64, W](ry)
                - (coordinates + 3 * j + 1)
                .strided_load[width=W](3)
                .cast[DType.float64]()
            )
            var z = (
                SIMD[DType.float64, W](rz)
                - (coordinates + 3 * j + 2)
                .strided_load[width=W](3)
                .cast[DType.float64]()
            )
            dst.store(
                i * n + j,
                (x * x + y * y + z * z).le(limit).cast[DType.uint8](),
            )
            j += W
        while j < n:
            var x = rx - Float64(coordinates[3 * j])
            var y = ry - Float64(coordinates[3 * j + 1])
            var z = rz - Float64(coordinates[3 * j + 2])
            dst[i * n + j] = UInt8(x * x + y * y + z * z <= cutoff2)
            j += 1

    @parameter
    def clear_row(i: Int):
        comptime W = simdwidthof[DType.uint8]()
        var zeros = SIMD[DType.uint8, W](0)
        var j = 0
        while j + W <= n:
            dst.store(i * n + j, zeros)
            j += W
        while j < n:
            dst[i * n + j] = 0
            j += 1

    @parameter
    def upper_row(i: Int):
        for j in range(i, n):
            var x, y, z = delta_f32(coordinates, i, coordinates, j, box, mode)
            if x * x + y * y + z * z <= cutoff2:
                dst[i * n + j] = 1
                dst[j * n + i] = 1

    if n * n >= PARALLEL_PAIR_THRESHOLD:
        if mode == 0:
            parallelize[full_plain_row](n)
        else:
            parallelize[clear_row](n)
            parallelize[upper_row](n)
    else:
        for i in range(n):
            clear_row(i)
        for i in range(n):
            upper_row(i)


def jacobi_largest(a: F64Ptr) -> Float64:
    for sweep in range(24):
        for p in range(4):
            for q in range(p + 1, 4):
                var apq = a[p * 4 + q]
                if abs(apq) <= 1.0e-15:
                    continue
                var app = a[p * 4 + p]
                var aqq = a[q * 4 + q]
                var tau = (aqq - app) / (2.0 * apq)
                var sign = 1.0
                if tau < 0.0:
                    sign = -1.0
                var t = sign / (abs(tau) + sqrt(1.0 + tau * tau))
                var cosine = 1.0 / sqrt(1.0 + t * t)
                var sine = t * cosine
                for k in range(4):
                    if k != p and k != q:
                        var akp = a[k * 4 + p]
                        var akq = a[k * 4 + q]
                        var new_kp = cosine * akp - sine * akq
                        var new_kq = sine * akp + cosine * akq
                        a[k * 4 + p] = new_kp
                        a[p * 4 + k] = new_kp
                        a[k * 4 + q] = new_kq
                        a[q * 4 + k] = new_kq
                a[p * 4 + p] = (
                    cosine * cosine * app
                    - 2.0 * sine * cosine * apq
                    + sine * sine * aqq
                )
                a[q * 4 + q] = (
                    sine * sine * app
                    + 2.0 * sine * cosine * apq
                    + cosine * cosine * aqq
                )
                a[p * 4 + q] = 0.0
                a[q * 4 + p] = 0.0
    var largest = a[0]
    for i in range(1, 4):
        if a[i * 4 + i] > largest:
            largest = a[i * 4 + i]
    return largest


def rmsd(
    a: F64Ptr,
    b: F64Ptr,
    weights: F64Ptr,
    work: F64Ptr,
    n: Int,
    weighted: Int,
    centered: Int,
    superposition: Int,
) -> Float64:
    if n <= 0:
        return sqrt(-1.0)
    var weight_sum = Float64(n)
    if weighted != 0:
        weight_sum = 0.0
        for i in range(n):
            weight_sum += weights[i]
    var amx = 0.0
    var amy = 0.0
    var amz = 0.0
    var bmx = 0.0
    var bmy = 0.0
    var bmz = 0.0
    if centered != 0 or superposition != 0:
        for i in range(n):
            var w = 1.0
            if weighted != 0:
                w = weights[i]
            amx += w * a[3 * i]
            amy += w * a[3 * i + 1]
            amz += w * a[3 * i + 2]
            bmx += w * b[3 * i]
            bmy += w * b[3 * i + 1]
            bmz += w * b[3 * i + 2]
        amx /= weight_sum
        amy /= weight_sum
        amz /= weight_sum
        bmx /= weight_sum
        bmy /= weight_sum
        bmz /= weight_sum

    var normalization = 1.0
    if weighted != 0:
        normalization = Float64(n) / weight_sum

    if superposition == 0:
        var total = 0.0
        for i in range(n):
            var dx = (a[3 * i] - amx) - (b[3 * i] - bmx)
            var dy = (a[3 * i + 1] - amy) - (b[3 * i + 1] - bmy)
            var dz = (a[3 * i + 2] - amz) - (b[3 * i + 2] - bmz)
            var w = normalization
            if weighted != 0:
                w *= weights[i]
            total += w * (dx * dx + dy * dy + dz * dz)
        return sqrt(total / Float64(n))

    for i in range(16):
        work[i] = 0.0
    var sxx = 0.0
    var sxy = 0.0
    var sxz = 0.0
    var syx = 0.0
    var syy = 0.0
    var syz = 0.0
    var szx = 0.0
    var szy = 0.0
    var szz = 0.0
    var energy = 0.0
    for i in range(n):
        var ax = a[3 * i] - amx
        var ay = a[3 * i + 1] - amy
        var az = a[3 * i + 2] - amz
        var bx = b[3 * i] - bmx
        var by = b[3 * i + 1] - bmy
        var bz = b[3 * i + 2] - bmz
        var w = normalization
        if weighted != 0:
            w *= weights[i]
        energy += w * (
            ax * ax + ay * ay + az * az + bx * bx + by * by + bz * bz
        )
        sxx += w * ax * bx
        sxy += w * ax * by
        sxz += w * ax * bz
        syx += w * ay * bx
        syy += w * ay * by
        syz += w * ay * bz
        szx += w * az * bx
        szy += w * az * by
        szz += w * az * bz

    work[0] = sxx + syy + szz
    work[1] = syz - szy
    work[2] = szx - sxz
    work[3] = sxy - syx
    work[4] = work[1]
    work[5] = sxx - syy - szz
    work[6] = sxy + syx
    work[7] = szx + sxz
    work[8] = work[2]
    work[9] = work[6]
    work[10] = -sxx + syy - szz
    work[11] = syz + szy
    work[12] = work[3]
    work[13] = work[7]
    work[14] = work[11]
    work[15] = -sxx - syy + szz
    var value = (energy - 2.0 * jacobi_largest(work)) / Float64(n)
    var cancellation_scale = energy / Float64(n)
    if cancellation_scale < 1.0:
        cancellation_scale = 1.0
    if abs(value) < 1.0e-12 * cancellation_scale:
        value = 0.0
    return sqrt(value)


@export("mda_distance_array")
def mda_distance_array(
    a: Int, b: Int, dst: Int, n: Int, m: Int, box: Int, mode: Int
) abi("C"):
    distance_array(
        F32Ptr(unsafe_from_address=a),
        F32Ptr(unsafe_from_address=b),
        F64Ptr(unsafe_from_address=dst),
        n,
        m,
        F64Ptr(unsafe_from_address=box),
        mode,
    )


@export("mda_self_distance_array")
def mda_self_distance_array(
    a: Int, dst: Int, n: Int, box: Int, mode: Int
) abi("C"):
    self_distance_array(
        F32Ptr(unsafe_from_address=a),
        F64Ptr(unsafe_from_address=dst),
        n,
        F64Ptr(unsafe_from_address=box),
        mode,
    )


@export("mda_calc_bonds")
def mda_calc_bonds(
    a: Int, b: Int, dst: Int, n: Int, box: Int, mode: Int
) abi("C"):
    bonds(
        F32Ptr(unsafe_from_address=a),
        F32Ptr(unsafe_from_address=b),
        F64Ptr(unsafe_from_address=dst),
        n,
        F64Ptr(unsafe_from_address=box),
        mode,
    )


@export("mda_calc_angles")
def mda_calc_angles(
    a: Int, b: Int, c: Int, dst: Int, n: Int, box: Int, mode: Int
) abi("C"):
    angles(
        F32Ptr(unsafe_from_address=a),
        F32Ptr(unsafe_from_address=b),
        F32Ptr(unsafe_from_address=c),
        F64Ptr(unsafe_from_address=dst),
        n,
        F64Ptr(unsafe_from_address=box),
        mode,
    )


@export("mda_calc_dihedrals")
def mda_calc_dihedrals(
    a: Int, b: Int, c: Int, d: Int, dst: Int, n: Int, box: Int, mode: Int
) abi("C"):
    dihedrals(
        F32Ptr(unsafe_from_address=a),
        F32Ptr(unsafe_from_address=b),
        F32Ptr(unsafe_from_address=c),
        F32Ptr(unsafe_from_address=d),
        F64Ptr(unsafe_from_address=dst),
        n,
        F64Ptr(unsafe_from_address=box),
        mode,
    )


@export("mda_minimize_vectors_f32")
def mda_minimize_vectors_f32(
    vectors: Int, dst: Int, n: Int, box: Int, mode: Int
) abi("C"):
    minimize_f32(
        F32Ptr(unsafe_from_address=vectors),
        F32Ptr(unsafe_from_address=dst),
        n,
        F64Ptr(unsafe_from_address=box),
        mode,
    )


@export("mda_minimize_vectors_f64")
def mda_minimize_vectors_f64(
    vectors: Int, dst: Int, n: Int, box: Int, mode: Int
) abi("C"):
    minimize_f64(
        F64Ptr(unsafe_from_address=vectors),
        F64Ptr(unsafe_from_address=dst),
        n,
        F64Ptr(unsafe_from_address=box),
        mode,
    )


@export("mda_contact_matrix")
def mda_contact_matrix(
    coordinates: Int, dst: Int, n: Int, cutoff: Float64, box: Int, mode: Int
) abi("C"):
    contact_matrix(
        F32Ptr(unsafe_from_address=coordinates),
        U8Ptr(unsafe_from_address=dst),
        n,
        cutoff,
        F64Ptr(unsafe_from_address=box),
        mode,
    )


@export("mda_rmsd")
def mda_rmsd(
    a: Int,
    b: Int,
    weights: Int,
    work: Int,
    n: Int,
    weighted: Int,
    centered: Int,
    superposition: Int,
) abi("C") -> Float64:
    return rmsd(
        F64Ptr(unsafe_from_address=a),
        F64Ptr(unsafe_from_address=b),
        F64Ptr(unsafe_from_address=weights),
        F64Ptr(unsafe_from_address=work),
        n,
        weighted,
        centered,
        superposition,
    )
