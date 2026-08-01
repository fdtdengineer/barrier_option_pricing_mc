#%%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from barrier_mc import BarrierParams, BarrierPricer  # noqa: E402

fs = 18
label_fs = fs * 0.7
figsize = (4, 3.6)
plt.rcParams.update({
    'font.family': 'Liberation Sans',
    'font.sans-serif': ['Liberation Sans'],
    'font.size': fs,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})
list_color = ["black", "gray", "#76b900", "#598c00"]

# Edit these values directly for another experiment.
params = BarrierParams(
    spot=100.0,
    strike=100.0,
    barrier=130.0,
    maturity=1.0,
    rate=0.03,
    dividend_yield=0.00,
    volatility=0.20,
)
path_counts = [10**n for n in range(2, 7)]
n_steps = 64
seed = 42
brownian_bridge = True

out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)
pricer = BarrierPricer(ROOT / "build" / "lib")
analytic = pricer.analytic(params)

series = [("CPU MT", "cpu", "mt"), ("CPU Sobol", "cpu", "sobol")]
if pricer.cuda_available:
    series += [("CUDA MT", "cuda", "mt"), ("CUDA Sobol", "cuda", "sobol")]
elif pricer.cuda is not None:
    print(f"Skipping CUDA series: {pricer.cuda_unavailable_reason}")

results = {}
for label, backend, rng in series:
    values = []
    for n_paths in path_counts:
        result = pricer.monte_carlo(
            params, n_paths=n_paths, n_steps=n_steps, rng=rng,
            backend=backend, seed=seed, brownian_bridge=brownian_bridge,
        )
        values.append(result)
        print(
            f"{label:10s} paths={n_paths:8d} price={result.price:.8f} "
            f"stderr={result.standard_error:.3e} time={result.elapsed_ms:.2f} ms"
        )
    results[label] = values
print(f"Analytic   price={analytic:.8f}")

markers = ["o", "s", "^", "D"]

fig, ax = plt.subplots(figsize=figsize)
ax.axhline(analytic, color=list_color[0], linewidth=2.0, label="Analytic")
for index, (label, values) in enumerate(results.items()):
    x = np.array(path_counts)
    y = np.array([v.price for v in values])
    err = 1.96 * np.array([v.standard_error for v in values])
    ax.errorbar(
        x, y, yerr=err, marker=markers[index], capsize=4,
        color=list_color[index], label=f"{label} (95% MC CI)",
    )
ax.set_xscale("log", base=10)
ax.set_xlabel("Number of paths", fontsize=label_fs)
ax.set_ylabel("Up-and-out call price", fontsize=label_fs)
ax.tick_params(labelsize=label_fs)
ax.grid(False)
ax.legend(frameon=False, fontsize=label_fs)
fig.tight_layout()
fig.savefig(out_dir / "price_convergence.png", dpi=180)

fig, ax = plt.subplots(figsize=figsize)
for index, (label, values) in enumerate(results.items()):
    prices = np.array([v.price for v in values])
    # One estimate is generated per path count, so this is the root squared
    # error for that estimate. It is numerically equal to its absolute error.
    rmse = np.sqrt(np.square(prices - analytic))
    ax.plot(
        path_counts, rmse, marker=markers[index],
        color=list_color[index], label=label,
    )
ax.set_xscale("log", base=10)
ax.set_yscale("log")
ax.set_xlabel("Number of paths", fontsize=label_fs)
ax.set_ylabel("RMSE vs analytic price", fontsize=label_fs)
ax.tick_params(labelsize=label_fs)
ax.grid(False)
ax.legend(frameon=False, fontsize=label_fs)
fig.tight_layout()
fig.savefig(out_dir / "rmse_convergence.png", dpi=180)

fig, ax = plt.subplots(figsize=figsize)
for index, (label, values) in enumerate(results.items()):
    ax.plot(
        path_counts, [v.elapsed_ms for v in values], marker=markers[index],
        color=list_color[index], label=label,
    )
ax.set_xscale("log", base=10)
ax.set_yscale("log")
ax.set_xlabel("Number of paths", fontsize=label_fs)
ax.set_ylabel("Elapsed time [ms]", fontsize=label_fs)
ax.tick_params(labelsize=label_fs)
ax.grid(False)
ax.legend(frameon=False, fontsize=label_fs)
fig.tight_layout()
fig.savefig(out_dir / "runtime_comparison.png", dpi=180)

print(f"Saved plots to {out_dir}")

# %%