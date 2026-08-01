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

The reported CUDA `elapsed_ms` begins after generator creation, generator
configuration, and `cudaMalloc`, and stops before resource destruction. It is
therefore GPU-work timing rather than complete API-call wall time. Including
allocation and generator lifecycle overhead would make short CUDA runs appear
somewhat slower. This caveat does not materially affect the large-path
bottleneck, which is dominated by the payoff kernel.

Relevant code: `src/barrier_cuda.cu`, around lines 109--137 and 174--186.

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
