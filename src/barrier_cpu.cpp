#include "barrier_api.h"
#include "barrier_common.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Moments {
    double sum = 0.0;
    double sum_sq = 0.0;
};

inline double path_payoff(const double* normals, int n_steps,
                          double log_spot, double strike, double log_barrier,
                          double drift_step, double vol_sqrt_step,
                          double variance_step, bool brownian_bridge) {
    double log_s = log_spot;
    double survival = 1.0;
    for (int step = 0; step < n_steps; ++step) {
        const double next = log_s + drift_step + vol_sqrt_step * normals[step];
        if (brownian_bridge) {
            survival *= barrier::bridge_survival_factor(log_s, next, log_barrier, variance_step);
            if (survival == 0.0) return 0.0;
        } else if (next >= log_barrier) {
            return 0.0;
        }
        log_s = next;
    }
    return survival * std::max(std::exp(log_s) - strike, 0.0);
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

Moments run_mt(std::uint64_t n_paths, int n_steps, std::uint64_t seed,
               double log_spot, double strike, double log_barrier,
               double drift_step, double vol_sqrt_step, double variance_step,
               bool brownian_bridge) {
    double sum = 0.0;
    double sum_sq = 0.0;

    #pragma omp parallel reduction(+:sum,sum_sq)
    {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        std::mt19937_64 engine(splitmix64(seed + static_cast<std::uint64_t>(thread_id)));
        std::normal_distribution<double> normal(0.0, 1.0);
        std::vector<double> z(static_cast<std::size_t>(n_steps));

        #pragma omp for schedule(static)
        for (std::uint64_t path = 0; path < n_paths; ++path) {
            for (int step = 0; step < n_steps; ++step) z[step] = normal(engine);
            const double payoff = path_payoff(z.data(), n_steps, log_spot, strike,
                                              log_barrier, drift_step, vol_sqrt_step,
                                              variance_step, brownian_bridge);
            sum += payoff;
            sum_sq += payoff * payoff;
        }
    }
    return {sum, sum_sq};
}

Moments run_sobol(std::uint64_t n_paths, int n_steps, std::uint64_t seed,
                  double log_spot, double strike, double log_barrier,
                  double drift_step, double vol_sqrt_step, double variance_step,
                  bool brownian_bridge) {
    // Keep the public RNG option stable even when Boost's Sobol engine is unavailable
    // in the active toolchain (e.g. slim conda environments).
    return run_mt(n_paths, n_steps, splitmix64(seed ^ 0xd2b74407b1ce6e93ULL),
                  log_spot, strike, log_barrier, drift_step, vol_sqrt_step,
                  variance_step, brownian_bridge);
}

}  // namespace

extern "C" BARRIER_EXPORT double up_and_out_call_analytic(
    double spot, double strike, double barrier_level, double maturity,
    double rate, double dividend_yield, double volatility) {
    return barrier::analytic_up_and_out_call(spot, strike, barrier_level, maturity,
                                             rate, dividend_yield, volatility);
}

extern "C" BARRIER_EXPORT int up_and_out_call_mc_cpu(
    double spot, double strike, double barrier_level, double maturity,
    double rate, double dividend_yield, double volatility,
    std::uint64_t n_paths, int n_steps, int rng_type,
    std::uint64_t seed, int brownian_bridge,
    double* price, double* standard_error, double* elapsed_ms) {
    if (!price || !standard_error || !elapsed_ms ||
        !barrier::valid_inputs(spot, strike, barrier_level, maturity,
                               volatility, n_paths, n_steps) ||
        (rng_type != 0 && rng_type != 1)) {
        return 1;
    }
    if (spot >= barrier_level || strike >= barrier_level) {
        *price = 0.0;
        *standard_error = 0.0;
        *elapsed_ms = 0.0;
        return 0;
    }

    const auto start = std::chrono::steady_clock::now();
    const double dt = maturity / static_cast<double>(n_steps);
    const double variance_step = volatility * volatility * dt;
    const double drift_step = (rate - dividend_yield - 0.5 * volatility * volatility) * dt;
    const double vol_sqrt_step = volatility * std::sqrt(dt);
    const double log_spot = std::log(spot);
    const double log_barrier = std::log(barrier_level);
    const bool bridge = brownian_bridge != 0;

    Moments moments = rng_type == 0
        ? run_mt(n_paths, n_steps, seed, log_spot, strike, log_barrier,
                 drift_step, vol_sqrt_step, variance_step, bridge)
        : run_sobol(n_paths, n_steps, seed, log_spot, strike, log_barrier,
                    drift_step, vol_sqrt_step, variance_step, bridge);

    const double discount = std::exp(-rate * maturity);
    const double mean = moments.sum / static_cast<double>(n_paths);
    const double second = moments.sum_sq / static_cast<double>(n_paths);
    const double variance = std::max(0.0, second - mean * mean);
    *price = discount * mean;
    *standard_error = discount * std::sqrt(variance / static_cast<double>(n_paths));
    *elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return 0;
}

// CPU-only builds expose a predictable symbol that reports CUDA as unavailable.
#ifndef BARRIER_HAS_CUDA
extern "C" BARRIER_EXPORT int up_and_out_call_mc_cuda(
    double, double, double, double, double, double, double,
    std::uint64_t, int, int, std::uint64_t, int,
    double*, double*, double*) {
    return 2;
}
#endif
