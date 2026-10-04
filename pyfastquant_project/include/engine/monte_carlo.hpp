#pragma once

#include <vector>
#include <cstddef>
#include <thread>

#include "engine/memory_align.hpp"

namespace PyFastQuant {
namespace Engine {

class MonteCarloEngine {
public:
    std::size_t m_num_paths;
    std::size_t m_num_time_steps;
    std::size_t m_threads;

    struct alignas(64) MonteCarloResult {
        double price;
        double std_error;
        double confidence_interval_low;
        double confidence_interval_high;
        double computation_time_ms;
        bool   used_antithetic;
    };

    explicit MonteCarloEngine(
        std::size_t num_paths = 1000000,
        std::size_t num_time_steps = 100,
        std::size_t threads = 0
    ) : m_num_paths(num_paths),
        m_num_time_steps(num_time_steps),
        m_threads(threads == 0 ? std::thread::hardware_concurrency() : threads) {
        if (m_threads == 0) m_threads = 1; // hardware_concurrency() can return 0
    }

    // use_antithetic: if true, draws num_paths/2 independent normal
    // variates Z and pairs each path with its mirror image (-Z) at
    // every time step. This is variance reduction, not extra
    // simulation -- the number of *independent* random draws is
    // halved, but the resulting estimator has strictly lower variance
    // for any payoff that is monotonic (or close to it) in the driving
    // Brownian path, which European calls/puts are. See the
    // accompanying write-up for the derivation of why the covariance
    // term Cov(payoff(Z), payoff(-Z)) <= 0 in that case.
    MonteCarloResult price_european_option(
        double S0, double K, double T, double r, double sigma,
        bool is_call, bool use_antithetic = false,
        double confidence_level = 0.95
    );

    std::vector<MonteCarloResult> price_european_option_batch(
        const std::vector<double>& S0,
        const std::vector<double>& K,
        const std::vector<double>& T,
        const std::vector<double>& r,
        const std::vector<double>& sigma,
        const std::vector<bool>& is_call,
        bool use_antithetic = false
    );
};

} // namespace Engine
} // namespace PyFastQuant
