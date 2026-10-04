// Standalone sanity test for the C++ engine, independent of the Python
// bindings. Useful for verifying the engine builds and behaves
// correctly on a new machine before dealing with pybind11/CMake at all.
//
// Build (Linux/macOS):
//   g++ -std=c++20 -O3 -mavx2 -mfma -fopenmp -I../include \
//       ../src/engine/monte_carlo.cpp ../src/engine/black_scholes.cpp \
//       ../src/engine/linear_algebra.cpp native_test.cpp -o native_test
//
// Build (Windows, Developer Command Prompt):
//   cl /std:c++20 /O2 /arch:AVX2 /openmp /I..\include ^
//      ..\src\engine\monte_carlo.cpp ..\src\engine\black_scholes.cpp ^
//      ..\src\engine\linear_algebra.cpp native_test.cpp /Fe:native_test.exe

#include "engine/monte_carlo.hpp"
#include "engine/black_scholes.hpp"
#include "engine/linear_algebra.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace PyFastQuant::Engine;

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) { printf("  FAIL: %s\n", msg); ++failures; }
    else       { printf("  ok:   %s\n", msg); }
}

int main() {
    printf("=== 1. Black-Scholes sanity check ===\n");
    // Known textbook case: S0=100, K=100, T=1, r=0.05, sigma=0.2, call
    // Reference analytic price ~10.4506
    auto bs = BlackScholesEngine::price_option(100.0, 100.0, 1.0, 0.05, 0.2, true);
    printf("  BS call price = %.4f (expected ~10.4506)\n", bs.price);
    check(std::fabs(bs.price - 10.4506) < 0.01, "BS price matches known textbook value");
    check(bs.delta > 0.0 && bs.delta < 1.0, "call delta in (0,1)");

    printf("\n=== 2. Monte Carlo convergence to Black-Scholes ===\n");
    double S0 = 100.0, K = 105.0, T = 1.0, r = 0.05, sigma = 0.2;
    auto bs2 = BlackScholesEngine::price_option(S0, K, T, r, sigma, true);
    printf("  Analytic BS price: %.6f\n", bs2.price);

    for (std::size_t n : {std::size_t(1000), std::size_t(100000), std::size_t(2000000)}) {
        MonteCarloEngine engine(n, 50, 0);
        auto res = engine.price_european_option(S0, K, T, r, sigma, true, false);
        double err = std::fabs(res.price - bs2.price);
        printf("  n=%8zu  MC price=%.6f  std_error=%.6f  |MC-BS|=%.6f  time=%.2fms\n",
               n, res.price, res.std_error, err, res.computation_time_ms);
    }
    // With 2M paths the MC price should be within a handful of std errors of BS
    {
        MonteCarloEngine engine(2000000, 50, 0);
        auto res = engine.price_european_option(S0, K, T, r, sigma, true, false);
        double z = std::fabs(res.price - bs2.price) / res.std_error;
        printf("  z-score at n=2,000,000: %.2f (should typically be < ~4)\n", z);
        check(z < 6.0, "MC price within a reasonable number of std errors of BS at large n");
    }

    printf("\n=== 3. Antithetic variates: variance reduction check ===\n");
    {
        std::size_t n = 200000;
        MonteCarloEngine engine(n, 50, 0);
        auto standard   = engine.price_european_option(S0, K, T, r, sigma, true, false);
        auto antithetic = engine.price_european_option(S0, K, T, r, sigma, true, true);
        printf("  standard   (n=%zu): price=%.6f  std_error=%.6f\n", n, standard.price, standard.std_error);
        printf("  antithetic (n=%zu): price=%.6f  std_error=%.6f\n", n, antithetic.price, antithetic.std_error);
        double reduction_pct = 100.0 * (1.0 - antithetic.std_error / standard.std_error);
        printf("  std_error reduction: %.1f%%\n", reduction_pct);
        check(antithetic.std_error < standard.std_error, "antithetic std_error is lower than standard for same n");
        check(std::fabs(antithetic.price - bs2.price) < 0.5, "antithetic price still close to BS analytic price");
    }

    printf("\n=== 4. Put-call parity ===\n");
    {
        MonteCarloEngine engine(500000, 50, 0);
        auto call = engine.price_european_option(S0, K, T, r, sigma, true, true);
        auto put  = engine.price_european_option(S0, K, T, r, sigma, false, true);
        double lhs = call.price - put.price;
        double rhs = S0 - K * std::exp(-r * T);
        printf("  C - P = %.6f   S0 - K*e^(-rT) = %.6f   diff = %.6f\n", lhs, rhs, lhs - rhs);
        check(std::fabs(lhs - rhs) < 0.3, "put-call parity holds within MC noise");
    }

    printf("\n=== 5. Linear algebra: matrix multiply vs naive reference ===\n");
    {
        std::size_t rowsA = 5, colsA = 7, colsB = 6;
        std::vector<double> A(rowsA * colsA), B(colsA * colsB), C(rowsA * colsB), C_naive(rowsA * colsB, 0.0);
        for (std::size_t i = 0; i < A.size(); ++i) A[i] = std::sin(double(i) * 0.37) * 3.0;
        for (std::size_t i = 0; i < B.size(); ++i) B[i] = std::cos(double(i) * 0.53) * 2.0;

        LinearAlgebraEngine::matrix_multiply_simd(A.data(), B.data(), C.data(), rowsA, colsA, colsB);

        for (std::size_t i = 0; i < rowsA; ++i)
            for (std::size_t k = 0; k < colsA; ++k)
                for (std::size_t j = 0; j < colsB; ++j)
                    C_naive[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];

        double max_err = 0.0;
        for (std::size_t i = 0; i < C.size(); ++i) max_err = std::max(max_err, std::fabs(C[i] - C_naive[i]));
        printf("  max abs error vs naive triple loop: %.3e\n", max_err);
        check(max_err < 1e-9, "SIMD matrix multiply matches naive reference");
    }

    printf("\n=== 6. Cholesky decomposition ===\n");
    {
        // Simple 3x3 SPD correlation-like matrix
        std::vector<double> M = {
            4, 2, 2,
            2, 5, 3,
            2, 3, 6
        };
        std::vector<double> M_orig = M;
        LinearAlgebraEngine::cholesky_decomposition(M.data(), 3);
        // reconstruct L * L^T and compare to original
        std::vector<double> recon(9, 0.0);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    recon[i * 3 + j] += M[i * 3 + k] * M[j * 3 + k];
        double max_err = 0.0;
        for (int i = 0; i < 9; ++i) max_err = std::max(max_err, std::fabs(recon[i] - M_orig[i]));
        printf("  max abs error of L*L^T vs original: %.3e\n", max_err);
        check(max_err < 1e-9, "Cholesky L*L^T reconstructs original matrix");
    }

    printf("\n=====================================\n");
    if (failures == 0) {
        printf("ALL CHECKS PASSED\n");
    } else {
        printf("%d CHECK(S) FAILED\n", failures);
    }
    printf("=====================================\n");
    return failures == 0 ? 0 : 1;
}
