#include "engine/black_scholes.hpp"
#include "engine/simd_math.hpp"
#include <cmath>

namespace PyFastQuant {
namespace Engine {

BlackScholesResult BlackScholesEngine::price_option(
    double S0, double K, double T, double r, double sigma, bool is_call
) {
    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;

    double Nd1 = norm_cdf(d1);
    double Nd2 = norm_cdf(d2);
    double Nminusd1 = norm_cdf(-d1);
    double Nminusd2 = norm_cdf(-d2);
    double pdf_d1 = norm_pdf(d1);

    BlackScholesResult result{};

    if (is_call) {
        result.price = S0 * Nd1 - K * std::exp(-r * T) * Nd2;
        result.delta = Nd1;
        result.theta = -(S0 * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * std::exp(-r * T) * Nd2;
        result.rho   = K * T * std::exp(-r * T) * Nd2;
    } else {
        result.price = K * std::exp(-r * T) * Nminusd2 - S0 * Nminusd1;
        result.delta = Nd1 - 1.0;
        result.theta = -(S0 * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * std::exp(-r * T) * Nminusd2;
        result.rho   = -K * T * std::exp(-r * T) * Nminusd2;
    }

    result.gamma = pdf_d1 / (S0 * sigma * sqrt_T);
    result.vega  = S0 * pdf_d1 * sqrt_T;

    return result;
}

std::vector<BlackScholesResult> BlackScholesEngine::price_option_batch(
    const double* S0, const double* K, const double* T,
    const double* r, const double* sigma, const bool* is_call,
    std::size_t n
) {
    std::vector<BlackScholesResult> results(n);
    for (std::size_t i = 0; i < n; ++i) {
        results[i] = price_option(S0[i], K[i], T[i], r[i], sigma[i], is_call[i]);
    }
    return results;
}

} // namespace Engine
} // namespace PyFastQuant
