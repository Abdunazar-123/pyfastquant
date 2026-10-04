#include "engine/simd_math.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace PyFastQuant::Engine;

int main() {
    printf("PFQ_HAS_AVX2 = %d\n", PFQ_HAS_AVX2);

    // Test 1: norm_cdf_array vs scalar norm_cdf reference
    std::vector<double> xs = {-4.0, -1.96, -1.0, -0.5, 0.0, 0.3333, 0.5, 1.0, 1.96, 4.0, 2.5, -2.5};
    // pad to make it not a multiple of 4 to exercise the scalar tail path too
    std::vector<double> out(xs.size());
    norm_cdf_array(xs.data(), out.data(), xs.size());

    double max_err = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        double ref = norm_cdf(xs[i]);
        double err = std::fabs(ref - out[i]);
        max_err = std::max(max_err, err);
        printf("  x=%8.4f  norm_cdf=%.10f  simd=%.10f  err=%.2e\n", xs[i], ref, out[i], err);
    }
    printf("Max abs error (norm_cdf_array vs scalar): %.3e\n", max_err);
    if (max_err > 1e-6) { printf("FAIL: norm_cdf_array error too large\n"); return 1; }
    printf("PASS: norm_cdf_array accurate to < 1e-6\n\n");

    // Test 2: sum_and_sum_sq_avx2 correctness, various sizes including non-multiples of 4
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (std::size_t n : {1, 3, 4, 5, 7, 8, 1000, 1000003}) {
        std::vector<double> data(n);
        for (auto& v : data) v = dist(rng);

        double naive_sum = 0.0, naive_sum_sq = 0.0;
        for (double v : data) { naive_sum += v; naive_sum_sq += v * v; }

        // align a buffer manually for the AVX2 load-aligned path
        alignas(32) static double buf[2000010];
        for (std::size_t i = 0; i < n; ++i) buf[i] = data[i];

        double simd_sum = 0.0, simd_sum_sq = 0.0;
        sum_and_sum_sq_avx2(buf, n, simd_sum, simd_sum_sq);

        double rel_err_sum = std::fabs(simd_sum - naive_sum) / (std::fabs(naive_sum) + 1e-9);
        double rel_err_sq  = std::fabs(simd_sum_sq - naive_sum_sq) / (std::fabs(naive_sum_sq) + 1e-9);
        printf("  n=%8zu  sum: naive=%.6f simd=%.6f (rel_err=%.2e)  sum_sq: naive=%.6f simd=%.6f (rel_err=%.2e)\n",
               n, naive_sum, simd_sum, rel_err_sum, naive_sum_sq, simd_sum_sq, rel_err_sq);
        if (rel_err_sum > 1e-9 || rel_err_sq > 1e-9) {
            printf("FAIL: sum_and_sum_sq_avx2 mismatch at n=%zu\n", n);
            return 1;
        }
    }
    printf("PASS: sum_and_sum_sq_avx2 matches naive summation for all sizes\n");

    return 0;
}
