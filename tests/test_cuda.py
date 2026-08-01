from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from barrier_mc import BarrierParams, BarrierPricer  # noqa: E402


@pytest.fixture(scope="module")
def cuda_pricer() -> BarrierPricer:
    pricer = BarrierPricer(ROOT / "build" / "lib")
    if not pricer.cuda_available:
        pytest.skip(pricer.cuda_unavailable_reason or "CUDA is unavailable")
    return pricer


@pytest.mark.parametrize("rng", ["mt", "sobol"])
def test_cuda_bridge_price_is_close_to_analytic(
    cuda_pricer: BarrierPricer, rng: str,
) -> None:
    params = BarrierParams()
    analytic = cuda_pricer.analytic(params)
    result = cuda_pricer.monte_carlo(
        params,
        n_paths=262_144,
        n_steps=64,
        rng=rng,
        backend="cuda",
        seed=42,
        brownian_bridge=True,
    )

    assert result.backend == "cuda"
    assert math.isfinite(result.price)
    assert math.isfinite(result.standard_error)
    assert abs(result.price - analytic) < max(6.0 * result.standard_error, 0.02)


def test_cuda_mt_accepts_odd_path_step_product(cuda_pricer: BarrierPricer) -> None:
    result = cuda_pricer.monte_carlo(
        BarrierParams(),
        n_paths=1_001,
        n_steps=33,
        rng="mt",
        backend="cuda",
        seed=7,
        brownian_bridge=True,
    )

    assert result.backend == "cuda"
    assert math.isfinite(result.price)
    assert result.price >= 0.0
    assert math.isfinite(result.standard_error)
    assert result.standard_error >= 0.0


def test_cuda_discrete_price_agrees_with_cpu(cuda_pricer: BarrierPricer) -> None:
    params = BarrierParams()
    cpu = cuda_pricer.monte_carlo(
        params,
        n_paths=262_144,
        n_steps=64,
        rng="mt",
        backend="cpu",
        seed=42,
        brownian_bridge=False,
    )
    cuda = cuda_pricer.monte_carlo(
        params,
        n_paths=262_144,
        n_steps=64,
        rng="mt",
        backend="cuda",
        seed=42,
        brownian_bridge=False,
    )

    combined_standard_error = math.hypot(cpu.standard_error, cuda.standard_error)
    assert abs(cuda.price - cpu.price) < max(6.0 * combined_standard_error, 0.02)
