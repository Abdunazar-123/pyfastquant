#pragma once
// -----------------------------------------------------------------------
// Cross-platform SIMD math utilities.
//
// GCC, Clang, and MSVC all expose the same Intel intrinsics through
// <immintrin.h> once AVX2 is enabled at compile time:
//   - GCC / Clang:  -mavx2 -mfma
//   - MSVC:         /arch:AVX2
// Both toolchains define the __AVX2__ macro when that flag is active,
// so a single header works everywhere -- CMakeLists.txt is responsible
// for picking the right flag per compiler (see the top-level CMakeLists).
//
// If AVX2 is not enabled (e.g. building for an older CPU or ARM), the
// PFQ_HAS_AVX2 macro is 0 and callers should use the scalar fallbacks
// (norm_cdf / norm_pdf) directly -- the library still compiles and
// produces correct results, just without the vectorized speedup.
// -----------------------------------------------------------------------

#include <cstddef>
#include <cmath>

#if defined(__AVX2__)
    #include <immintrin.h>
    #define PFQ_HAS_AVX2 1
#else
    #define PFQ_HAS_AVX2 0
#endif

namespace PyFastQuant {
namespace Engine {

constexpr double SQRT_2       = 1.4142135623730951;
constexpr double INV_SQRT_2PI = 0.3989422804014327;

// ---------------------------------------------------------------------
// Scalar standard normal CDF / PDF. norm_cdf uses erfc, which is
// numerically stable and accurate to machine precision -- this is the
// reference implementation everything else is validated against.
// ---------------------------------------------------------------------
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / SQRT_2);
}

inline double norm_pdf(double x) {
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

#if PFQ_HAS_AVX2

// ---------------------------------------------------------------------
// Vectorized normal CDF for 4 doubles at once (AVX2, 256-bit lanes).
// Uses the Abramowitz & Stegun 7.1.26 rational approximation to erf,
// applied to y = |x| / sqrt(2), then converted to CDF(x) = 0.5*(1 +
// erf(x/sqrt(2))). Max absolute error ~1.5e-7, which is far smaller
// than the Monte Carlo statistical error for any practical path count,
// so it never becomes the accuracy bottleneck.
//
// Note: AVX2 has no native exp() instruction. The polynomial and all
// t, t^2..t^5 powers are fully vectorized; only the final exp(-y^2) is
// computed lane-by-lane via std::exp. This is still a net win because
// it's one scalar exp() per 4 elements, not four independent full CDF
// evaluations.
// ---------------------------------------------------------------------
inline __m256d norm_cdf_avx2(__m256d vx) {
    const __m256d vone  = _mm256_set1_pd(1.0);
    const __m256d vzero = _mm256_setzero_pd();
    const __m256d vsign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFULL));

    const __m256d va1 = _mm256_set1_pd(0.254829592);
    const __m256d va2 = _mm256_set1_pd(-0.284496736);
    const __m256d va3 = _mm256_set1_pd(1.421413741);
    const __m256d va4 = _mm256_set1_pd(-1.453152027);
    const __m256d va5 = _mm256_set1_pd(1.061405429);
    const __m256d vp  = _mm256_set1_pd(0.3275911);
    const __m256d vinv_sqrt2 = _mm256_set1_pd(0.7071067811865476);

    __m256d vy = _mm256_and_pd(_mm256_mul_pd(vx, vinv_sqrt2), vsign_mask); // |x|/sqrt(2)

    __m256d vt  = _mm256_div_pd(vone, _mm256_add_pd(vone, _mm256_mul_pd(vp, vy)));
    __m256d vt2 = _mm256_mul_pd(vt,  vt);
    __m256d vt3 = _mm256_mul_pd(vt2, vt);
    __m256d vt4 = _mm256_mul_pd(vt3, vt);
    __m256d vt5 = _mm256_mul_pd(vt4, vt);

    __m256d vpoly = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(va1, vt), _mm256_mul_pd(va2, vt2)),
        _mm256_add_pd(_mm256_mul_pd(va3, vt3),
            _mm256_add_pd(_mm256_mul_pd(va4, vt4), _mm256_mul_pd(va5, vt5)))
    );

    alignas(32) double y2_arr[4];
    _mm256_store_pd(y2_arr, _mm256_mul_pd(vy, vy));
    alignas(32) double exp_arr[4];
    for (int i = 0; i < 4; ++i) exp_arr[i] = std::exp(-y2_arr[i]);
    __m256d vexp = _mm256_load_pd(exp_arr);

    __m256d verf_abs = _mm256_sub_pd(vone, _mm256_mul_pd(vpoly, vexp)); // erf(|x|/sqrt2)

    __m256d vcond_neg   = _mm256_cmp_pd(vx, vzero, _CMP_LT_OQ);
    __m256d verf_signed = _mm256_blendv_pd(verf_abs, _mm256_sub_pd(vzero, verf_abs), vcond_neg);

    return _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_add_pd(vone, verf_signed));
}

// ---------------------------------------------------------------------
// Batched norm_cdf over an array. Processes 4 elements at a time with
// AVX2, then a scalar tail for n % 4 != 0. `in` and `out` may alias
// each other but must both be at least `n` doubles.
// ---------------------------------------------------------------------
inline void norm_cdf_array(const double* in, double* out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(&in[i]);
        __m256d vresult = norm_cdf_avx2(vx);
        _mm256_storeu_pd(&out[i], vresult);
    }
    for (; i < n; ++i) {
        out[i] = norm_cdf(in[i]);
    }
}

// ---------------------------------------------------------------------
// Vectorized reduction: computes sum(data[0..n)) and sum(data[i]^2)
// simultaneously. This is the hot loop for aggregating Monte Carlo
// payoffs into a price + standard error, and is called once per
// pricing request on an array that can be 1M+ elements long, so the
// 4x throughput here directly cuts wall-clock time.
// `data` must be 32-byte aligned (guaranteed by AlignedAllocator).
// ---------------------------------------------------------------------
inline void sum_and_sum_sq_avx2(const double* data, std::size_t n,
                                 double& out_sum, double& out_sum_sq) {
    __m256d vsum    = _mm256_setzero_pd();
    __m256d vsum_sq = _mm256_setzero_pd();

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_load_pd(&data[i]);
        vsum    = _mm256_add_pd(vsum, v);
        vsum_sq = _mm256_fmadd_pd(v, v, vsum_sq); // v*v + vsum_sq in one instruction
    }

    alignas(32) double sum_lanes[4], sum_sq_lanes[4];
    _mm256_store_pd(sum_lanes, vsum);
    _mm256_store_pd(sum_sq_lanes, vsum_sq);

    double sum = sum_lanes[0] + sum_lanes[1] + sum_lanes[2] + sum_lanes[3];
    double sum_sq = sum_sq_lanes[0] + sum_sq_lanes[1] + sum_sq_lanes[2] + sum_sq_lanes[3];

    for (; i < n; ++i) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }

    out_sum = sum;
    out_sum_sq = sum_sq;
}

#else // !PFQ_HAS_AVX2 -- portable scalar fallbacks with identical signatures

inline void norm_cdf_array(const double* in, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = norm_cdf(in[i]);
}

inline void sum_and_sum_sq_avx2(const double* data, std::size_t n,
                                 double& out_sum, double& out_sum_sq) {
    double sum = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }
    out_sum = sum;
    out_sum_sq = sum_sq;
}

#endif // PFQ_HAS_AVX2

} // namespace Engine
} // namespace PyFastQuant
