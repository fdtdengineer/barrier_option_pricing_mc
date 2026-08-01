#%%
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from barrier_mc import BarrierParams, BarrierPricer


def test_bridge_mc_is_close_to_analytic() -> None:
    pricer = BarrierPricer(ROOT / "build" / "lib")
    params = BarrierParams()
    analytic = pricer.analytic(params)
    result = pricer.monte_carlo(
        params, n_paths=131072, n_steps=32, rng="sobol",
        backend="cpu", seed=17, brownian_bridge=True,
    )
    assert abs(result.price - analytic) < max(5.0 * result.standard_error, 0.02)


def test_knocked_out_at_inception() -> None:
    pricer = BarrierPricer(ROOT / "build" / "lib")
    params = BarrierParams(spot=130.0, barrier=130.0)
    assert pricer.analytic(params) == 0.0
