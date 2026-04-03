cimport cython
from cython cimport Py_ssize_t
from libc.math cimport (
    isnan,
    sqrt,
)
from libc.stdlib cimport (
    free,
    malloc,
)
from libc.string cimport memcpy

import numpy as np

cimport numpy as cnp
from numpy cimport (
    NPY_FLOAT64,
    NPY_INT8,
    NPY_INT16,
    NPY_INT32,
    NPY_INT64,
    NPY_OBJECT,
    NPY_UINT64,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

cnp.import_array()

cdef:
    float64_t FP_ERR = 1e-13
    float64_t NaN = <float64_t>np.nan

def nancorr_CramersV_fast(const int64_t[:, :] mat, bint bias_correction=True, minp=None):
    cdef:
        Py_ssize_t i, xi, yi, N, K
        int64_t minpv
        float64_t[:, ::1] result
        uint8_t[:, :] mask
        int64_t nobs = 0
        float64_t chi2, phi2, cramers_v, total
        int64_t r, k, min_dim
        int64_t max_categories = 100

        int64_t[:, :] contingency
        int64_t[:] row_sums, col_sums
        int64_t max_r, max_k
        int64_t vx, vy

    N, K = (<object>mat).shape

    if minp is None:
        minpv = 1
    else:
        minpv = <int64_t>minp

    result = np.empty((K, K), dtype=np.float64)
    mask = np.isfinite(mat).view(np.uint8)

    max_r = 0
    max_k = 0
    for xi in range(K):
        unique_vals = set()
        for i in range(N):
            if mask[i, xi]:
                unique_vals.add(mat[i, xi])
        max_r = max(max_r, len(unique_vals))
        max_k = max(max_k, len(unique_vals))

    max_categories = max(max_r, max_k)

    contingency = np.zeros((max_categories, max_categories), dtype=np.int64)
    row_sums = np.zeros(max_categories, dtype=np.int64)
    col_sums = np.zeros(max_categories, dtype=np.int64)

    for xi in range(K):
        for yi in range(xi + 1):
            if xi == yi:
                result[xi, xi] = 1.0
                continue

            for i in range(max_categories):
                row_sums[i] = 0
                col_sums[i] = 0
                for j in range(max_categories):
                    contingency[i, j] = 0

            nobs = 0
            for i in range(N):
                if mask[i, xi] and mask[i, yi]:
                    vx = mat[i, xi]
                    vy = mat[i, yi]
                    if vx >= 0 and vx < max_categories and vy >= 0 and vy < max_categories:
                        contingency[vx, vy] += 1
                        row_sums[vx] += 1
                        col_sums[vy] += 1
                        nobs += 1

            if nobs < minpv:
                result[xi, yi] = result[yi, xi] = np.nan
                continue

            chi2 = 0.0
            total = float(nobs)
            r = 0
            k = 0

            for i in range(max_categories):
                if row_sums[i] > 0:
                    r += 1
                if col_sums[i] > 0:
                    k += 1

            for i in range(max_categories):
                if row_sums[i] == 0:
                    continue
                for j in range(max_categories):
                    if col_sums[j] == 0:
                        continue
                    observed = contingency[i, j]
                    if observed > 0:
                        expected = (row_sums[i] * col_sums[j]) / total
                        chi2 += (observed - expected) ** 2 / expected

            min_dim = min(r, k) - 1

            if min_dim <= 0:
                result[xi, yi] = result[yi, xi] = np.nan
                continue

            phi2 = chi2 / total
            cramers_v = sqrt(phi2 / min_dim)

            if bias_correction:
                correction = (r - 1) * (k - 1) / max(1.0, total - 1)
                cramers_v = max(0.0, min(1.0, cramers_v - correction))

            if cramers_v < 0.0:
                cramers_v = 0.0
            elif cramers_v > 1.0:
                cramers_v = 1.0

            result[xi, yi] = result[yi, xi] = cramers_v

    return result.base