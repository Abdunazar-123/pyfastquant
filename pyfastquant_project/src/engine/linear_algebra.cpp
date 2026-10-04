#include "engine/linear_algebra.hpp"
#include "engine/simd_math.hpp" // for PFQ_HAS_AVX2 / immintrin.h
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace PyFastQuant {
namespace Engine {

void LinearAlgebraEngine::matrix_multiply_simd(
    const double* A, const double* B, double* C,
    std::size_t rowsA, std::size_t colsA, std::size_t colsB
) {
    std::memset(C, 0, rowsA * colsB * sizeof(double));

#if PFQ_HAS_AVX2
    // i-k-j loop order: for each fixed (i,k), A[i][k] is a scalar
    // broadcast across a whole AVX2 register, and we sweep across a
    // row of B / C four columns at a time with a fused multiply-add.
    // This is both more cache-friendly (B and C are walked row-wise,
    // matching row-major layout) and more vectorizable than the naive
    // i-j-k order, which only vectorizes a dot product reduction.
    for (std::size_t i = 0; i < rowsA; ++i) {
        double* C_row = &C[i * colsB];
        for (std::size_t k = 0; k < colsA; ++k) {
            __m256d va = _mm256_set1_pd(A[i * colsA + k]);
            const double* B_row = &B[k * colsB];

            std::size_t j = 0;
            for (; j + 4 <= colsB; j += 4) {
                __m256d vb = _mm256_loadu_pd(&B_row[j]);
                __m256d vc = _mm256_loadu_pd(&C_row[j]);
                vc = _mm256_fmadd_pd(va, vb, vc);
                _mm256_storeu_pd(&C_row[j], vc);
            }
            double a_ik = A[i * colsA + k];
            for (; j < colsB; ++j) {
                C_row[j] += a_ik * B_row[j];
            }
        }
    }
#else
    for (std::size_t i = 0; i < rowsA; ++i) {
        for (std::size_t k = 0; k < colsA; ++k) {
            double a_ik = A[i * colsA + k];
            for (std::size_t j = 0; j < colsB; ++j) {
                C[i * colsB + j] += a_ik * B[k * colsB + j];
            }
        }
    }
#endif
}

void LinearAlgebraEngine::cholesky_decomposition(double* matrix, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = matrix[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= matrix[i * n + k] * matrix[j * n + k];
            }
            if (i == j) {
                if (sum <= 0) throw std::runtime_error("Matrix is not positive definite");
                matrix[i * n + j] = std::sqrt(sum);
            } else {
                matrix[i * n + j] = sum / matrix[j * n + j];
            }
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            matrix[i * n + j] = 0.0;
        }
    }
}

} // namespace Engine
} // namespace PyFastQuant
