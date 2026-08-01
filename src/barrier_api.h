#pragma once
#include <cstdint>

#ifdef _WIN32
#define BARRIER_EXPORT __declspec(dllexport)
#else
#define BARRIER_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// rng_type: 0 = Mersenne Twister, 1 = randomized/scrambled Sobol.
// brownian_bridge: 0 = discrete monitoring, nonzero = continuous-monitoring bridge weighting.
BARRIER_EXPORT double up_and_out_call_analytic(
    double spot, double strike, double barrier, double maturity,
    double rate, double dividend_yield, double volatility);

BARRIER_EXPORT int up_and_out_call_mc_cpu(
    double spot, double strike, double barrier, double maturity,
    double rate, double dividend_yield, double volatility,
    std::uint64_t n_paths, int n_steps, int rng_type,
    std::uint64_t seed, int brownian_bridge,
    double* price, double* standard_error, double* elapsed_ms);

BARRIER_EXPORT int up_and_out_call_mc_cuda(
    double spot, double strike, double barrier, double maturity,
    double rate, double dividend_yield, double volatility,
    std::uint64_t n_paths, int n_steps, int rng_type,
    std::uint64_t seed, int brownian_bridge,
    double* price, double* standard_error, double* elapsed_ms);

}
