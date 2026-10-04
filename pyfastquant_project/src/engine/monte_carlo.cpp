#include "engine/monte_carlo.hpp"
#include "engine/simd_math.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace PyFastQuant {
namespace Engine {

MonteCarloEngine::MonteCarloResult MonteCarloEngine::price_european_option(
    double S0, double K, double T, double r, double sigma,
    bool is_call, bool use_antithetic, double /*confidence_level*/
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    const double dt       = T / static_cast<double>(m_num_time_steps);
    const double drift    = (r - 0.5 * sigma * sigma) * dt;
    const double diffusion = sigma * std::sqrt(dt);
    const double discount = std::exp(-r * T);

    // Number of *independent samples* actually used to estimate the
    // mean/variance: in standard mode this is one per path; in
    // antithetic mode it's one per (Z, -Z) pair, so half as many
    // samples but each built from two negatively-correlated paths.
    const std::size_t n_samples = use_antithetic ? (m_num_paths / 2) : m_num_paths;
    if (n_samples == 0) {
        MonteCarloResult empty{0, 0, 0, 0, 0, use_antithetic};
        return empty;
    }

    std::vector<double, AlignedAllocator<double>> samples(n_samples);

    #pragma omp parallel num_threads(static_cast<int>(m_threads))
    {
        #ifdef _OPENMP
            std::size_t thread_id = static_cast<std::size_t>(omp_get_thread_num());
        #else
            std::size_t thread_id = 0;
        #endif

        std::mt19937_64 rng(12345ULL + thread_id * 1000003ULL);
        std::normal_distribution<double> norm_dist(0.0, 1.0);

        if (!use_antithetic) {
            #pragma omp for
            for (std::size_t i = 0; i < n_samples; ++i) {
                double S = S0;
                for (std::size_t j = 0; j < m_num_time_steps; ++j) {
                    double z = norm_dist(rng);
                    S *= std::exp(drift + diffusion * z);
                }
                samples[i] = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
            }
        } else {
            #pragma omp for
            for (std::size_t i = 0; i < n_samples; ++i) {
                double S_plus  = S0;
                double S_minus = S0;
                for (std::size_t j = 0; j < m_num_time_steps; ++j) {
                    double z = norm_dist(rng);
                    // Mirror path: negate the *same* random draw at every
                    // step, not just at the end -- this keeps the
                    // construction correct for any future path-dependent
                    // payoff (Asian, barrier), not only terminal payoffs.
                    S_plus  *= std::exp(drift + diffusion * z);
                    S_minus *= std::exp(drift - diffusion * z);
                }
                double payoff_plus  = is_call ? std::max(S_plus  - K, 0.0) : std::max(K - S_plus,  0.0);
                double payoff_minus = is_call ? std::max(S_minus - K, 0.0) : std::max(K - S_minus, 0.0);
                samples[i] = 0.5 * (payoff_plus + payoff_minus);
            }
        }
    }

    double sum = 0.0, sum_sq = 0.0;
    sum_and_sum_sq_avx2(samples.data(), n_samples, sum, sum_sq);

    const double mean = sum / static_cast<double>(n_samples);
    const double price = discount * mean;

    const double variance  = std::max(0.0, (sum_sq / static_cast<double>(n_samples)) - mean * mean);
    const double std_error = discount * std::sqrt(variance / static_cast<double>(n_samples));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    MonteCarloResult result;
    result.price = price;
    result.std_error = std_error;
    result.confidence_interval_low  = price - 1.96 * std_error;
    result.confidence_interval_high = price + 1.96 * std_error;
    result.computation_time_ms = static_cast<double>(duration.count()) / 1000.0;
    result.used_antithetic = use_antithetic;

    return result;
}

std::vector<MonteCarloEngine::MonteCarloResult>
MonteCarloEngine::price_european_option_batch(
    const std::vector<double>& S0,
    const std::vector<double>& K,
    const std::vector<double>& T,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<bool>& is_call,
    bool use_antithetic
) {
    std::size_t n = S0.size();
    std::vector<MonteCarloResult> results(n);

    for (std::size_t i = 0; i < n; ++i) {
        results[i] = price_european_option(S0[i], K[i], T[i], r[i], sigma[i], is_call[i], use_antithetic);
    }

    return results;
}

} // namespace Engine
} // namespace PyFastQuant
