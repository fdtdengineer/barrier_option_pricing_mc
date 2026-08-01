# CUDA large-path performance analysis

## Summary

When the number of Monte Carlo paths becomes large, the CUDA implementation is
only moderately faster than the OpenMP CPU implementation. The main bottleneck
is not kernel launch overhead, reduction, or insufficient grid size. It is the
per-path payoff kernel: one CUDA thread handles one path, but executes all 64
time steps sequentially using several expensive double-precision operations.

The GPU is already sufficiently occupied at around 100,000 paths. Beyond that
point, adding paths primarily adds proportional work, so the elapsed time is
expected to grow approximately linearly.

## Test environment

- GPU: NVIDIA RTX A2000 12 GB, compute capability 8.6
- CPU: Intel Xeon W-2295, 18 cores / 36 logical CPUs
- Default benchmark parameters:
  - 64 time steps
  - Brownian bridge enabled
  - path counts from `10^2` to `10^6`
- Source under investigation: `src/barrier_cuda.cu`
- Investigation date: 2026-08-01

The source code was not changed during this investigation. Temporary diagnostic
binaries and a Release build were created outside the repository.

## Reproduced end-to-end timings

For 1,000,000 paths, one representative run produced:

| Backend / RNG | Elapsed time |
|---|---:|
| CPU MT | 240.76 ms |
| CPU Sobol mode | 211.96 ms |
| CUDA MT | 167.32 ms |
| CUDA Sobol | 116.39 ms |

CUDA is faster, but the observed speedup is only approximately 1.3x to 2.1x.
Timings vary between runs because of GPU warm-up, clock state, and system load;
the relative bottleneck breakdown below was consistent.

## Stage-level breakdown

A temporary diagnostic executable invoked the same payoff kernel and measured
cuRAND generation, the payoff kernel, and the two reductions separately for
1,000,000 paths and 64 steps. Representative warmed-up results were:

| RNG | Brownian bridge | RNG generation | Payoff kernel | Reduction | Total |
|---|---:|---:|---:|---:|---:|
| MT | enabled | 6.10 ms | 142.10 ms | 2.65 ms | 150.85 ms |
| Sobol | enabled | 2.22 ms | 122.51 ms | 1.05 ms | 125.78 ms |
| MT | disabled | 6.06 ms | 110.82 ms | 1.84 ms | 118.72 ms |
| Sobol | disabled | 2.26 ms | 80.64 ms | 1.37 ms | 84.27 ms |

The payoff kernel accounts for roughly 95--97% of the measured GPU work.
Optimizing synchronization or reduction alone therefore has limited upside in
the current implementation.

## Root causes

### 1. The time-step loop is serial within each thread

The implementation maps one path to one CUDA thread, which is a reasonable
path-level decomposition. Inside each thread, however, all time steps are
executed serially because the next log-price depends on the previous one:

```cpp
for (int step = 0; step < steps; ++step) {
    // log_s from the preceding step is required
}
```

Increasing the number of paths improves utilization only until the GPU is
saturated. It cannot parallelize the dependent time evolution within a path.

Relevant code: `src/barrier_cuda.cu`, `payoff_kernel`, around lines 45--78.

### 2. Expensive double-precision functions are evaluated for every step

Each path step performs an inverse-normal approximation. Depending on the input
branch, this includes polynomial evaluation, division, `log`, and `sqrt` in
double precision. With Brownian bridge weighting enabled, every surviving step
also evaluates a double-precision `exp`.

Disabling Brownian bridge reduced the measured payoff-kernel time by about
22--34%, confirming that the per-step bridge exponential is significant.
The RTX A2000 is not a high-throughput double-precision compute GPU, making this
workload especially unfavorable.

Relevant code:

- `inv_norm`: `src/barrier_cuda.cu`, around lines 14--43
- bridge `exp`: `src/barrier_cuda.cu`, around lines 60--68
- terminal payoff `exp`: `src/barrier_cuda.cu`, around line 75

### 3. MT uses a warp-unfriendly path-major layout

For MT, the payoff kernel indexes the random-number buffer as:

```cpp
index = path * steps + step;
```

At a fixed step, adjacent threads in a warp read values separated by 64 doubles,
or 512 bytes. These loads are not coalesced. Sobol uses the dimension-major
layout:

```cpp
index = step * paths + path;
```

Adjacent threads then read adjacent doubles. This is substantially friendlier
to the GPU memory system. The measured MT payoff kernel was slower than the
Sobol payoff kernel even though both execute the same path-update arithmetic;
the different memory layout is the leading explanation, with differing path
divergence also potentially contributing.

Relevant code: `src/barrier_cuda.cu`, around lines 55--58.

### 4. Random values are materialized in global memory

The implementation first generates all uniforms into a device buffer, then
launches a separate kernel that reads them and converts each uniform through
`inv_norm`.

For 1,000,000 paths and 64 steps, this represents approximately 512 MB of
double-precision random values written and subsequently read. The allocation is
batched at 262,144 paths, corresponding to an approximately 128 MiB uniform
buffer per full batch.

Although cuRAND generation itself only took a few milliseconds after warm-up,
materializing the buffer adds global-memory traffic and forces the payoff kernel
to perform the inverse-normal transformation separately.

Relevant code: `src/barrier_cuda.cu`, around lines 126--171.

### 5. Branch divergence is present

Threads may terminate their path loops at different steps after hitting the
barrier. The inverse-normal approximation also has three branches. Threads in
the same warp can therefore execute different control-flow paths, leaving some
lanes idle. This was not isolated as the primary bottleneck, but compounds the
serial per-thread workload.

## Items investigated that were not primary causes

### Kernel launch and reduction overhead

The implementation synchronizes before executing two sequential Thrust
reductions for each batch. This could be streamlined, but the reductions were
only about 1--3 ms in the 1,000,000-path diagnostic. It is not the reason for
the observed large-path scaling.

### Batch boundary

The maximum batch size is `1 << 18`, or 262,144 paths. Measurements immediately
below and above that boundary did not show a twofold time jump. The extra
partial batch has a fixed cost, but batching is not the dominant large-path
bottleneck.

### Block size and register spilling

The payoff kernel uses 256 threads per block. Compiling with PTX assembler
diagnostics reported:

- 35 registers per thread
- no local stack frame
- no spill loads or stores

There is no evidence that pathological register spilling is limiting the
kernel. Other block sizes may still yield incremental improvements, but they do
not address the main arithmetic and memory costs.

### Missing CMake Release build type

The current build cache has an empty `CMAKE_BUILD_TYPE`, and its CUDA compile
command does not explicitly contain `-O3`. A temporary Release build added
`-O3 -DNDEBUG`, but large-path runtime was effectively unchanged. This is not a
primary explanation for the observed performance.

## Timing caveat

During the initial diagnosis, CUDA `elapsed_ms` excluded generator creation,
allocation, and cleanup, while CPU timing included its RNG and OpenMP setup.
This was subsequently corrected: both backends now report `steady_clock` wall
time covering backend initialization, simulation, result transfer, and resource
cleanup after input validation. This makes the small- and large-path runtime
plots directly comparable.

## Suggested optimization order

No optimizations were implemented as part of this investigation. Based on the
measurements, the most useful experiments are:

1. Make MT accesses dimension-major or otherwise guarantee coalesced warp loads.
2. Avoid storing uniforms followed by a custom `inv_norm`; benchmark direct
   normal generation or an on-thread RNG design.
3. Evaluate single or mixed precision while checking pricing bias and standard
   error against the current double-precision implementation.
4. Reduce the cost of Brownian bridge weighting, especially the per-step
   double-precision exponential.
5. Fuse the two payoff-moment reductions or use a single pair-valued reduction.
6. Remove unnecessary synchronization and overlap batches only after the payoff
   kernel bottleneck has been reduced.
7. Tune block size and validate occupancy as a secondary optimization.

The first four items target the measured dominant cost. Reduction and launch
changes are worthwhile cleanup, but should not be expected to produce a large
speedup on their own.

## Implemented optimizations and results

The recommendations above were implemented on 2026-08-01 without changing the
GBM evolution, barrier conditions, Brownian bridge formula, discounting, or
standard-error formula.

Implemented changes:

1. cuRAND now generates standard-normal floats directly, removing the custom
   per-value inverse-normal calculation from the payoff kernel.
2. Both MT and Sobol buffers are consumed in dimension-major order, giving
   adjacent threads coalesced reads at each time step.
3. Normal storage, path state, and Brownian bridge calculations use single
   precision. Block-level payoff sums and squared-payoff sums remain double
   precision.
4. Brownian bridge survival uses `-expm1f(exponent)` instead of
   `1-expf(exponent)` to avoid cancellation when an endpoint is very close to
   the barrier.
5. Each block reduces payoff and squared payoff together and atomically adds
   two double-precision moments. The two payoff arrays, two Thrust reductions,
   and per-batch device synchronization were removed.
6. Odd MT output sizes are padded by one normal value to satisfy cuRAND's
   Box--Muller output-length requirement; the padding value is ignored.
7. Single-configuration CMake builds now default to `Release` when no build
   type is supplied.
8. CUDA timing now includes generator creation, allocation, result transfer,
   and cleanup, matching the CPU wall-time scope.

### Performance comparison

The old Release library and optimized library were measured in the same test
session using external wall-clock timing. The following values are medians of
five runs at 1,000,000 paths and 64 steps on the RTX A2000:

| RNG | Brownian bridge | Before | After | Speedup |
|---|---:|---:|---:|---:|
| MT | enabled | 113.31 ms | 8.80 ms | 12.9x |
| Sobol | enabled | 120.17 ms | 9.44 ms | 12.7x |
| MT | disabled | 85.01 ms | 9.09 ms | 9.4x |
| Sobol | disabled | 93.25 ms | 8.82 ms | 10.6x |

GPU timings at this scale fluctuate with clock state and system load, so these
figures should be treated as representative rather than absolute. The speedup
was large enough that the conclusion was not sensitive to that variation.

### Pricing validation

The optimized CUDA implementation was checked against the analytic continuous
barrier price using 1,000,000 paths for both RNG modes. The validation covered:

- the default parameters;
- a near barrier at 105;
- a far barrier at 180;
- volatility 0.1 and 0.5;
- 32, 64, and 128 time steps.

All ten CUDA results were within six reported Monte Carlo standard errors of the
analytic price. For the near-barrier case, whose analytic value was about
0.0089483, the absolute differences were approximately `5.1e-5` for MT and
`2.0e-4` for Sobol.

Automated CUDA tests additionally cover both RNG modes and an odd
`1001 paths * 33 steps` MT request. A discrete-monitoring CUDA result is also
checked against the independent CPU estimate using their combined standard
error. Tests skip cleanly on machines without an available CUDA runtime.

## CPU-equivalent optimizations

The CPU backend was optimized using the same principles where they apply
without CUDA-specific mechanisms:

1. Normal generation and path evolution are fused, removing the per-thread
   normal vector and its extra write/read pass.
2. Path state and Brownian bridge arithmetic use floats, including the stable
   `-expm1f(exponent)` form. Final payoff moments remain double precision.
3. CPU MT uses 32-bit `std::mt19937`, which is closer to CUDA MTGP32, and maps
   its high 24 output bits directly through a float inverse-normal CDF.
4. Every normal is still consumed after path knockout. RNG work remains fixed
   at `paths * steps`, matching the CUDA pipeline.
5. The OpenMP static path partition and double-precision reduction are retained.

CPU `sobol` remains the documented MT-based fallback rather than a true Sobol
engine; it benefits from the same optimized path and MT code. At 1,000,000
paths and 64 steps, the CPU changes produced approximately 1.23x--1.39x speedup,
depending on RNG label and Brownian bridge mode. Both CPU RNG modes were also
validated against the analytic price across the same parameter cases as CUDA.

## Extended 10^7 and 10^8 path run

On 2026-08-01, `scripts/run_comparison.py` was extended from `10^2`--`10^6`
paths to `10^2`--`10^8` paths in powers of ten. The price, normalized squared
error, and runtime SVG plots were regenerated for all four series with 64 time
steps and Brownian bridge enabled.

The newly added measurements were:

| Series | Paths | Price | Standard error | Wall time | Normalized squared error |
|---|---:|---:|---:|---:|---:|
| CPU MT | 10,000,000 | 3.20092082 | 1.854e-3 | 941.80 ms | 3.2607e-7 |
| CPU MT | 100,000,000 | 3.20283390 | 5.864e-4 | 8,537.23 ms | 6.9157e-10 |
| CPU Sobol fallback | 10,000,000 | 3.20656625 | 1.855e-3 | 1,173.79 ms | 1.4200e-6 |
| CPU Sobol fallback | 100,000,000 | 3.20348751 | 5.865e-4 | 9,041.24 ms | 5.3073e-8 |
| CUDA MT | 10,000,000 | 3.20210510 | 1.854e-3 | 58.50 ms | 4.0504e-8 |
| CUDA MT | 100,000,000 | 3.20309965 | 5.864e-4 | 518.66 ms | 1.1940e-8 |
| CUDA Sobol | 10,000,000 | 3.20285158 | 1.855e-3 | 30.92 ms | 1.0123e-9 |
| CUDA Sobol | 100,000,000 | 3.20301074 | 5.863e-4 | 314.05 ms | 6.6444e-9 |

The analytic price was `3.20274968`. At 100,000,000 paths, every estimate was
within approximately 1.3 reported standard errors of the analytic value. The
standard error decreased from about `5.86e-3` at 1,000,000 paths to `5.86e-4`
at 100,000,000 paths, matching the expected inverse-square-root path scaling.

For the added large-path points, the observed CUDA-to-CPU runtime ratios were:

| RNG label | 10,000,000 paths | 100,000,000 paths |
|---|---:|---:|
| MT | 16.1x | 16.5x |
| Sobol label | 38.0x | 28.8x |

The Sobol row is only a comparison of the plotted runtime series: CUDA uses
scrambled Sobol64, whereas CPU `sobol` is still the MT-based fallback.

### Why CUDA elapsed time still increases at 10^7 and 10^8 paths

The increase is not evidence that CUDA throughput degrades at these path
counts. It is primarily the expected result of increasing the total workload
after the GPU has already reached useful occupancy.

Each path is assigned to one CUDA thread, but its 64 time steps remain serial
because every log-price update depends on the preceding step. Once there are
enough paths to saturate the GPU, additional paths cannot make those dependent
steps more parallel. They add another proportional amount of normal generation,
path evolution, Brownian bridge arithmetic, and moment accumulation.

The implementation also limits one batch to 262,144 paths. Consequently,
10,000,000 paths require about 39 batches and 100,000,000 paths require about
382 batches. Batching bounds peak memory usage, but each batch must still run
its cuRAND generation and payoff kernel. The batches currently execute in
sequence, so their total time accumulates approximately linearly.

Normalizing the measurements by path confirms that large-path throughput is
stable rather than declining:

| RNG | Paths | Wall time | Time per path | Approx. throughput |
|---|---:|---:|---:|---:|
| CUDA MT | 10,000,000 | 58.50 ms | 5.85 ns | 171 million paths/s |
| CUDA MT | 100,000,000 | 518.66 ms | 5.19 ns | 193 million paths/s |
| CUDA Sobol | 10,000,000 | 30.92 ms | 3.09 ns | 323 million paths/s |
| CUDA Sobol | 100,000,000 | 314.05 ms | 3.14 ns | 318 million paths/s |

Thus, the tenfold path increase produces roughly a nine- to tenfold elapsed-time
increase while useful throughput stays flat or improves slightly. At smaller
path counts, fixed costs such as CUDA context/API work, generator setup,
allocation, and cleanup are a larger fraction of wall time, so extrapolating
from the small-path curve can make the large-path elapsed time look like a
slowdown even though the GPU is simply operating in its steady-state regime.

Possible further improvements would target steady-state throughput rather than
occupancy: overlapping batch generation and payoff work with multiple streams,
increasing the batch size when memory permits, or reducing the remaining
per-step arithmetic. None of these changes the fundamental linear scaling in
the total number of simulated path steps.
