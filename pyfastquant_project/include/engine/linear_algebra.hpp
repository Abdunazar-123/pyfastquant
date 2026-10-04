#pragma once
#include <cstddef>

namespace PyFastQuant {
namespace Engine {

// Used for multi-asset Monte Carlo (correlated GBM paths via Cholesky
// of the covariance matrix) -- a natural extension of the single-asset
// engine, and matrix_multiply_simd is the building block for applying
// that correlation structure to a batch of random draws efficiently.
class LinearAlgebraEngine {
public:
    // C (rowsA x colsB) = A (rowsA x colsA) * B (colsA x colsB), row-major.
    static void matrix_multiply_simd(
        const double* A, const double* B, double* C,
        std::size_t rowsA, std::size_t colsA, std::size_t colsB
    );

    // In-place lower-triangular Cholesky factor of an n x n
    // symmetric positive-definite matrix (row-major).
    static void cholesky_decomposition(double* matrix, std::size_t n);
};

} // namespace Engine
} // namespace PyFastQuant
