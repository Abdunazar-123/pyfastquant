#pragma once
#include <vector>
#include <cstddef>

namespace PyFastQuant {
namespace Engine {

struct BlackScholesResult {
    double price;
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
};

// Analytic Black-Scholes pricer. This is the ground truth used to
// validate the Monte Carlo engine: for a European option, MC price
// should converge to this value as num_paths grows, with error
// shrinking at the theoretical O(1/sqrt(n)) rate (see benchmarks/).
class BlackScholesEngine {
public:
    static BlackScholesResult price_option(
        double S0, double K, double T, double r, double sigma, bool is_call
    );

    static std::vector<BlackScholesResult> price_option_batch(
        const double* S0, const double* K, const double* T,
        const double* r, const double* sigma, const bool* is_call,
        std::size_t n
    );
};

} // namespace Engine
} // namespace PyFastQuant
