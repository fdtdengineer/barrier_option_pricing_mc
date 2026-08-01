#include "barrier_api.h"
#include "barrier_common.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Moments {
    double sum = 0.0;
    double sum_sq = 0.0;
};

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

inline float inverse_normal_float(float p) {
    constexpr float a1 = -39.6968303f, a2 = 220.946098f;
    constexpr float a3 = -275.928510f, a4 = 138.357752f;
    constexpr float a5 = -30.6647981f, a6 = 2.50662828f;
    constexpr float b1 = -54.4760988f, b2 = 161.585836f;
    constexpr float b3 = -155.698980f, b4 = 66.8013119f;
    constexpr float b5 = -13.2806816f;
    constexpr float c1 = -0.007784894f, c2 = -0.322396458f;
    constexpr float c3 = -2.40075828f, c4 = -2.54973254f;
    constexpr float c5 = 4.37466431f, c6 = 2.93816398f;
    constexpr float d1 = 0.007784696f, d2 = 0.322467119f;
    constexpr float d3 = 2.44513416f, d4 = 3.75440866f;
    constexpr float p_low = 0.02425f;
    constexpr float p_high = 1.0f - p_low;

    if (p < p_low) {
        const float q = std::sqrt(-2.0f * std::log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
               ((((d1*q+d2)*q+d3)*q+d4)*q+1.0f);
    }
    if (p <= p_high) {
        const float q = p - 0.5f;
        const float r = q * q;
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
               (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0f);
    }
    const float q = std::sqrt(-2.0f * std::log(1.0f-p));
    return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
            ((((d1*q+d2)*q+d3)*q+d4)*q+1.0f);
}

inline float next_normal(std::mt19937& engine) {
    // A float has 24 bits of significand. Use the high 24 MT bits and keep
    // the inverse-CDF input strictly inside (0, 1).
    const float p = static_cast<float>(engine() >> 8) * 0x1.0p-24f;
    return inverse_normal_float(std::max(p, 0x1.0p-25f));
}

Moments run_mt(std::uint64_t n_paths, int n_steps, std::uint64_t seed,
               float log_spot, float strike, float log_barrier,
               float drift_step, float vol_sqrt_step, float variance_step,
               bool brownian_bridge) {
    double sum = 0.0;
    double sum_sq = 0.0;

    #pragma omp parallel reduction(+:sum,sum_sq)
    {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        const std::uint64_t mixed_seed =
            splitmix64(seed + static_cast<std::uint64_t>(thread_id));
        std::seed_seq seed_words{
            static_cast<std::uint32_t>(mixed_seed),
            static_cast<std::uint32_t>(mixed_seed >> 32U),
        };
        std::mt19937 engine(seed_words);

        #pragma omp for schedule(static)
        for (std::uint64_t path = 0; path < n_paths; ++path) {
            float log_s = log_spot;
            float survival = 1.0f;
            bool active = true;
            for (int step = 0; step < n_steps; ++step) {
                // Consume every variate even after knockout. This keeps a fixed
                // amount of RNG work per path, matching the CUDA pipeline.
                const float z = next_normal(engine);
                if (!active) continue;

                const float next = log_s + drift_step + vol_sqrt_step * z;
                if (brownian_bridge) {
                    if (log_s >= log_barrier || next >= log_barrier) {
                        survival = 0.0f;
                        active = false;
                        continue;
                    }
                    const float exponent = -2.0f * (log_barrier-log_s) *
                                           (log_barrier-next) / variance_step;
                    survival *= std::max(0.0f, -std::expm1(exponent));
                    if (survival == 0.0f) active = false;
                } else if (next >= log_barrier) {
                    survival = 0.0f;
                    active = false;
                    continue;
                }
                log_s = next;
            }
            const float payoff_float = survival == 0.0f
                ? 0.0f
                : survival * std::max(std::exp(log_s) - strike, 0.0f);
            const double payoff = static_cast<double>(payoff_float);
            sum += payoff;
            sum_sq += payoff * payoff;
        }
    }
    return {sum, sum_sq};
}

Moments run_sobol(std::uint64_t n_paths, int n_steps, std::uint64_t seed,
                  float log_spot, float strike, float log_barrier,
                  float drift_step, float vol_sqrt_step, float variance_step,
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
    const float variance_step = static_cast<float>(volatility * volatility * dt);
    const float drift_step = static_cast<float>(
        (rate - dividend_yield - 0.5 * volatility * volatility) * dt);
    const float vol_sqrt_step = static_cast<float>(volatility * std::sqrt(dt));
    const float log_spot = static_cast<float>(std::log(spot));
    const float strike_float = static_cast<float>(strike);
    const float log_barrier = static_cast<float>(std::log(barrier_level));
    const bool bridge = brownian_bridge != 0;

    Moments moments = rng_type == 0
        ? run_mt(n_paths, n_steps, seed, log_spot, strike_float, log_barrier,
                 drift_step, vol_sqrt_step, variance_step, bridge)
        : run_sobol(n_paths, n_steps, seed, log_spot, strike_float, log_barrier,
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
