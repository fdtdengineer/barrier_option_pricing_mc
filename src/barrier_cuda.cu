#define BARRIER_HAS_CUDA 1
#include "barrier_api.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace {

__device__ double inv_norm(double p) {
    const double a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02, a4 = 1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01, a6 = 2.506628277459239e+00;
    const double b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02, b4 = 6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;
    const double c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00, c4 = -2.549732539343734e+00;
    const double c5 = 4.374664141464968e+00, c6 = 2.938163982698783e+00;
    const double d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01;
    const double d3 = 2.445134137142996e+00, d4 = 3.754408661907416e+00;
    const double p_low = 0.02425, p_high = 1.0 - p_low;
    p = fmin(fmax(p, 1.0e-15), 1.0 - 1.0e-15);
    if (p < p_low) {
        const double q = sqrt(-2.0 * log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
               ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
    }
    if (p <= p_high) {
        const double q = p - 0.5, r = q*q;
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
               (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
    }
    const double q = sqrt(-2.0 * log(1.0-p));
    return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
            ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
}

__global__ void payoff_kernel(const double* uniforms, double* payoff, double* payoff_sq,
                              std::uint64_t paths, int steps, int dimension_major,
                              double log_spot, double strike, double log_barrier,
                              double drift_step, double vol_sqrt_step,
                              double variance_step, int bridge) {
    const std::uint64_t path = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
    if (path >= paths) return;
    double log_s = log_spot;
    double survival = 1.0;
    for (int step = 0; step < steps; ++step) {
        const std::uint64_t index = dimension_major
            ? static_cast<std::uint64_t>(step) * paths + path
            : path * static_cast<std::uint64_t>(steps) + step;
        const double z = inv_norm(uniforms[index]);
        const double next = log_s + drift_step + vol_sqrt_step * z;
        if (bridge) {
            if (log_s >= log_barrier || next >= log_barrier) {
                survival = 0.0;
                break;
            }
            const double hit = exp(-2.0 * (log_barrier-log_s) *
                                   (log_barrier-next) / variance_step);
            survival *= fmax(0.0, 1.0-hit);
            if (survival == 0.0) break;
        } else if (next >= log_barrier) {
            survival = 0.0;
            break;
        }
        log_s = next;
    }
    const double value = survival * fmax(exp(log_s)-strike, 0.0);
    payoff[path] = value;
    payoff_sq[path] = value * value;
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
    double *d_uniforms = nullptr, *d_payoff = nullptr, *d_payoff_sq = nullptr;
    if (!cuda_ok(cudaMalloc(&d_uniforms, capacity_paths * static_cast<std::uint64_t>(n_steps) * sizeof(double))) ||
        !cuda_ok(cudaMalloc(&d_payoff, capacity_paths * sizeof(double))) ||
        !cuda_ok(cudaMalloc(&d_payoff_sq, capacity_paths * sizeof(double)))) {
        cudaFree(d_uniforms); cudaFree(d_payoff); cudaFree(d_payoff_sq);
        curandDestroyGenerator(generator); return 4;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    const double dt = maturity / static_cast<double>(n_steps);
    const double variance_step = volatility * volatility * dt;
    const double drift_step = (rate-dividend_yield-0.5*volatility*volatility)*dt;
    const double vol_sqrt_step = volatility * std::sqrt(dt);
    const double log_spot = std::log(spot), log_barrier = std::log(barrier_level);
    double sum = 0.0, sum_sq = 0.0;

    for (std::uint64_t first = 0; first < n_paths; first += capacity_paths) {
        const std::uint64_t paths = std::min(capacity_paths, n_paths-first);
        const std::uint64_t values = paths * static_cast<std::uint64_t>(n_steps);
        if (!curand_ok(curandGenerateUniformDouble(generator, d_uniforms, values))) {
            cudaFree(d_uniforms); cudaFree(d_payoff); cudaFree(d_payoff_sq);
            curandDestroyGenerator(generator); return 5;
        }
        const int threads = 256;
        const int blocks = static_cast<int>((paths + threads - 1) / threads);
        payoff_kernel<<<blocks, threads>>>(d_uniforms, d_payoff, d_payoff_sq,
            paths, n_steps, rng_type == 1, log_spot, strike, log_barrier,
            drift_step, vol_sqrt_step, variance_step, brownian_bridge != 0);
        const cudaError_t launch_status = cudaGetLastError();
        if (!cuda_ok(launch_status)) {
            log_cuda_error("CUDA kernel launch failed", launch_status);
            cudaFree(d_uniforms); cudaFree(d_payoff); cudaFree(d_payoff_sq);
            curandDestroyGenerator(generator); return 6;
        }
        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (!cuda_ok(sync_status)) {
            log_cuda_error("CUDA kernel execution failed", sync_status);
            cudaFree(d_uniforms); cudaFree(d_payoff); cudaFree(d_payoff_sq);
            curandDestroyGenerator(generator); return 6;
        }
        thrust::device_ptr<double> p(d_payoff), p2(d_payoff_sq);
        sum += thrust::reduce(p, p + paths, 0.0, thrust::plus<double>());
        sum_sq += thrust::reduce(p2, p2 + paths, 0.0, thrust::plus<double>());
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);
    const double discount = std::exp(-rate*maturity);
    const double mean = sum / static_cast<double>(n_paths);
    const double second = sum_sq / static_cast<double>(n_paths);
    *price = discount * mean;
    *standard_error = discount * std::sqrt(std::max(0.0, second-mean*mean) /
                                           static_cast<double>(n_paths));
    *elapsed_ms = milliseconds;

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_uniforms); cudaFree(d_payoff); cudaFree(d_payoff_sq);
    curandDestroyGenerator(generator);
    return 0;
}
