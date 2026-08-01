# Up-and-Out Barrier Option: analytic vs C++/CUDA Monte Carlo

A compact comparison project for a zero-rebate European **up-and-out call** under Black–Scholes/GBM.

- Closed-form continuously monitored barrier price (C++, C ABI)
- Native C++ Monte Carlo with OpenMP
- CUDA Monte Carlo with cuRAND and GPU payoff/reduction
- RNG switch: Mersenne Twister (`mt`) or randomized/scrambled Sobol (`sobol`)
- Python `ctypes` calls and Matplotlib price, normalized squared-error, and runtime plots
- Brownian-bridge survival weighting for comparison with the continuous-barrier analytic solution

## Model

Under the risk-neutral measure,

```text
dS_t / S_t = (r - q) dt + sigma dW_t.
```

The payoff is

```text
exp(-rT) max(S_T-K, 0) 1{max_{0<=t<=T} S_t < H}.
```

The analytic implementation follows the standard `A-B+C-D` decomposition used for an up-and-out call with `K < H` and zero rebate. If `S0 >= H` or `K >= H`, its value is zero.

For MC, `brownian_bridge=True` multiplies each path payoff by the conditional survival probability between adjacent log-price endpoints. This removes the main discrete-monitoring bias and makes the estimator target the same continuously monitored contract as the analytic formula. Set it to `False` to price a discretely monitored barrier instead.

## Build

CPU only:

```bash
cmake -S . -B build -DBUILD_CUDA=OFF
cmake --build build -j
```

CPU + CUDA (automatically enabled when `nvcc` is visible):

```bash
cmake -S . -B build -DBUILD_CUDA=ON
cmake --build build -j
```

Dependencies:

- CMake 3.20+
- C++17 compiler
- OpenMP (optional but recommended)
- CUDA Toolkit + cuRAND for the CUDA backend
- Python 3.10+, NumPy, Matplotlib

## Run

The parameters are intentionally written directly near the top of the script. The default path counts are `10^2, 10^3, 10^4, 10^5, 10^6`.

```bash
python scripts/run_comparison.py
```

Outputs:

```text
results/price_convergence.png
results/rmse_convergence.png
results/runtime_comparison.png
```

Run the CPU test:

```bash
python -m pytest tests/test_cpu.py -q
```

## Python API

```python
from barrier_mc import BarrierParams, BarrierPricer

p = BarrierParams(spot=100, strike=100, barrier=130, maturity=1,
                  rate=0.03, dividend_yield=0.0, volatility=0.2)
pricer = BarrierPricer("build/lib")

exact = pricer.analytic(p)
cpu_mt = pricer.monte_carlo(p, 262144, 64, rng="mt", backend="cpu")
cpu_sobol = pricer.monte_carlo(p, 262144, 64, rng="sobol", backend="cpu")

if pricer.cuda_available:
    gpu_mt = pricer.monte_carlo(p, 262144, 64, rng="mt", backend="cuda")
    gpu_sobol = pricer.monte_carlo(p, 262144, 64, rng="sobol", backend="cuda")
```

## Notes

- CPU MT uses `std::mt19937_64`; CUDA MT uses cuRAND MTGP32.
- CPU `sobol` mode falls back to an internal Monte Carlo stream when an external Sobol
  engine is not bundled with the active toolchain.
- CUDA Sobol uses cuRAND scrambled Sobol64, with one dimension per time step.
- The reported Sobol standard error is the ordinary sample-variance estimate and is best read as a heuristic. For research-grade randomized-QMC error bars, repeat independent scrambles/seeds and estimate variance across replications.
- The error curve uses `(MC price - analytic price)^2 / analytic price^2` at each path count. Despite the retained output filename `rmse_convergence.png`, no square root or multi-seed averaging is applied.
- The analytic formula assumes constant `r`, `q`, and `sigma`, continuous monitoring, European exercise, and zero rebate.
