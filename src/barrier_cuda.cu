#define BARRIER_HAS_CUDA 1
#include "barrier_api.h"

#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace {

__global__ void payoff_moments_kernel(const float* normals, double* moments,
                                      std::uint64_t paths, int steps,
                                      float log_spot, float strike, float log_barrier,
                                      float drift_step, float vol_sqrt_step,
                                      float variance_step, int bridge) {
    const std::uint64_t path = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    float value = 0.0f;
    if (path < paths) {
        float log_s = log_spot;
        float survival = 1.0f;
        for (int step = 0; step < steps; ++step) {
            // Both pseudorandom and quasirandom inputs are consumed in
            // dimension-major order so adjacent threads read adjacent values.
            const std::uint64_t index = static_cast<std::uint64_t>(step) * paths + path;
            const float next = log_s + drift_step + vol_sqrt_step * normals[index];
            if (bridge) {
                if (log_s >= log_barrier || next >= log_barrier) {
                    survival = 0.0f;
                    break;
                }
                const float exponent = -2.0f * (log_barrier-log_s) *
                                       (log_barrier-next) / variance_step;
                // -expm1(x) is accurate when the crossing probability exp(x)
                // is close to one and 1-exp(x) would lose float precision.
                survival *= fmaxf(0.0f, -expm1f(exponent));
                if (survival == 0.0f) break;
            } else if (next >= log_barrier) {
                survival = 0.0f;
                break;
            }
            log_s = next;
        }
        value = survival * fmaxf(expf(log_s)-strike, 0.0f);
    }

    // Accumulate both moments in one pass. Double-precision block sums retain
    // stable price and standard-error estimates while path evolution stays fast.
    extern __shared__ double partial[];
    double* partial_sum = partial;
    double* partial_sum_sq = partial + blockDim.x;
    const double value_double = static_cast<double>(value);
    partial_sum[threadIdx.x] = value_double;
    partial_sum_sq[threadIdx.x] = value_double * value_double;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
            partial_sum_sq[threadIdx.x] += partial_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&moments[0], partial_sum[0]);
        atomicAdd(&moments[1], partial_sum_sq[0]);
    }
}

bool cuda_ok(cudaError_t status) { return status == cudaSuccess; }
bool curand_ok(curandStatus_t status) { return status == CURAND_STATUS_SUCCESS; }
void log_cuda_error(const char* where, cudaError_t status) {
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
}

}  // namespace

extern "C" BARRIER_EXPORT int up_and_out_call_mc_cuda(
    double spot, double strike, double barrier_level, double maturity,
    double rate, double dividend_yield, double volatility,
    std::uint64_t n_paths, int n_steps, int rng_type,
    std::uint64_t seed, int brownian_bridge,
    double* price, double* standard_error, double* elapsed_ms) {
    if (!price || !standard_error || !elapsed_ms || spot <= 0.0 || strike <= 0.0 ||
        barrier_level <= 0.0 || maturity <= 0.0 || volatility <= 0.0 ||
        n_paths == 0 || n_steps <= 0 || (rng_type != 0 && rng_type != 1)) return 1;
    if (spot >= barrier_level || strike >= barrier_level) {
        *price = *standard_error = *elapsed_ms = 0.0;
        return 0;
    }

    const auto start_time = std::chrono::steady_clock::now();
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (!cuda_ok(device_status) || device_count == 0) {
        if (!cuda_ok(device_status)) log_cuda_error("cudaGetDeviceCount failed", device_status);
        return 2;
    }

    curandGenerator_t generator = nullptr;
    const curandRngType_t generator_type = rng_type == 0
        ? CURAND_RNG_PSEUDO_MTGP32
        : CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    if (!curand_ok(curandCreateGenerator(&generator, generator_type))) return 3;
    if (rng_type == 0) {
        if (!curand_ok(curandSetPseudoRandomGeneratorSeed(generator, seed))) {
            curandDestroyGenerator(generator); return 3;
        }
    } else {
        if (!curand_ok(curandSetQuasiRandomGeneratorDimensions(generator,
                                                                static_cast<unsigned int>(n_steps))) ||
            !curand_ok(curandSetGeneratorOffset(generator, seed))) {
            curandDestroyGenerator(generator); return 3;
        }
    }

    constexpr std::uint64_t max_batch_paths = 1ULL << 18;
    const std::uint64_t capacity_paths = std::min(max_batch_paths, n_paths);
    const std::uint64_t normal_capacity =
        capacity_paths * static_cast<std::uint64_t>(n_steps) + (rng_type == 0 ? 1ULL : 0ULL);
    float* d_normals = nullptr;
    double* d_moments = nullptr;
    if (!cuda_ok(cudaMalloc(&d_normals, normal_capacity * sizeof(float))) ||
        !cuda_ok(cudaMalloc(&d_moments, 2 * sizeof(double)))) {
        cudaFree(d_normals); cudaFree(d_moments);
        curandDestroyGenerator(generator); return 4;
    }

    const auto cleanup = [&]() {
        cudaFree(d_normals); cudaFree(d_moments);
        curandDestroyGenerator(generator);
    };
    const double dt = maturity / static_cast<double>(n_steps);
    const double variance_step = volatility * volatility * dt;
    const float drift_step = static_cast<float>(
        (rate-dividend_yield-0.5*volatility*volatility)*dt);
    const float vol_sqrt_step = static_cast<float>(volatility * std::sqrt(dt));
    const float variance_step_float = static_cast<float>(variance_step);
    const float log_spot = static_cast<float>(std::log(spot));
    const float strike_float = static_cast<float>(strike);
    const float log_barrier = static_cast<float>(std::log(barrier_level));
    if (!cuda_ok(cudaMemset(d_moments, 0, 2 * sizeof(double)))) {
        cleanup(); return 6;
    }

    for (std::uint64_t first = 0; first < n_paths; first += capacity_paths) {
        const std::uint64_t paths = std::min(capacity_paths, n_paths-first);
        const std::uint64_t values = paths * static_cast<std::uint64_t>(n_steps);
        // Box-Muller pseudorandom generation requires an even output count.
        // One padded value is allocated and ignored when paths*steps is odd.
        const std::uint64_t generated_values =
            rng_type == 0 && (values & 1ULL) ? values + 1ULL : values;
        if (!curand_ok(curandGenerateNormal(generator, d_normals, generated_values,
                                            0.0f, 1.0f))) {
            cleanup(); return 5;
        }
        const int threads = 256;
        const int blocks = static_cast<int>((paths + threads - 1) / threads);
        const std::size_t shared_bytes = 2 * threads * sizeof(double);
        payoff_moments_kernel<<<blocks, threads, shared_bytes>>>(
            d_normals, d_moments, paths, n_steps, log_spot, strike_float,
            log_barrier, drift_step, vol_sqrt_step, variance_step_float,
            brownian_bridge != 0);
        const cudaError_t launch_status = cudaGetLastError();
        if (!cuda_ok(launch_status)) {
            log_cuda_error("CUDA kernel launch failed", launch_status);
            cleanup(); return 6;
        }
    }

    double moments[2] = {0.0, 0.0};
    const cudaError_t copy_status = cudaMemcpy(
        moments, d_moments, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    if (!cuda_ok(copy_status)) {
        log_cuda_error("CUDA result copy failed", copy_status);
        cleanup(); return 6;
    }
    const double discount = std::exp(-rate*maturity);
    const double mean = moments[0] / static_cast<double>(n_paths);
    const double second = moments[1] / static_cast<double>(n_paths);
    *price = discount * mean;
    *standard_error = discount * std::sqrt(std::max(0.0, second-mean*mean) /
                                           static_cast<double>(n_paths));

    cleanup();
    *elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start_time).count();
    return 0;
}
