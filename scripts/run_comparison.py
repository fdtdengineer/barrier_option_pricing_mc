from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from barrier_mc import BarrierParams, BarrierPricer  # noqa: E402

fs = 18
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
path_counts = [2**12, 2**14, 2**16, 2**18]
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

fig, ax = plt.subplots(figsize=(10, 7))
ax.axhline(analytic, color=list_color[0], linewidth=2.0, label="Analytic")
markers = ["o", "s", "^", "D"]
for index, (label, values) in enumerate(results.items()):
    x = np.array(path_counts)
    y = np.array([v.price for v in values])
    err = 1.96 * np.array([v.standard_error for v in values])
    ax.errorbar(
        x, y, yerr=err, marker=markers[index], capsize=4,
        color=list_color[index], label=f"{label} (95% MC CI)",
    )
ax.set_xscale("log", base=2)
ax.set_xlabel("Number of paths")
ax.set_ylabel("Up-and-out call price")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out_dir / "price_convergence.png", dpi=180)

fig, ax = plt.subplots(figsize=(10, 7))
for index, (label, values) in enumerate(results.items()):
    ax.plot(
        path_counts, [v.elapsed_ms for v in values], marker=markers[index],
        color=list_color[index], label=label,
    )
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Number of paths")
ax.set_ylabel("Elapsed time [ms]")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out_dir / "runtime_comparison.png", dpi=180)

print(f"Saved plots to {out_dir}")
