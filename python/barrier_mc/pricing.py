from __future__ import annotations

import ctypes
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Rng = Literal["mt", "sobol"]
Backend = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class BarrierParams:
    spot: float = 100.0
    strike: float = 100.0
    barrier: float = 130.0
    maturity: float = 1.0
    rate: float = 0.03
    dividend_yield: float = 0.00
    volatility: float = 0.20


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    elapsed_ms: float
    backend: Backend
    rng: Rng
    n_paths: int
    n_steps: int
    brownian_bridge: bool


class BarrierPricer:
    def __init__(self, library_dir: str | os.PathLike[str] | None = None) -> None:
        self.library_dir = Path(library_dir) if library_dir else self._default_library_dir()
        self.cpu = self._load("barrier_cpu", required=True)
        self.cuda = self._load("barrier_cuda", required=False)
        self._configure(self.cpu)
        if self.cuda is not None:
            self._configure(self.cuda)

    @staticmethod
    def _default_library_dir() -> Path:
        env = os.environ.get("BARRIER_MC_LIB_DIR")
        if env:
            return Path(env)
        return Path(__file__).resolve().parents[2] / "build" / "lib"

    @staticmethod
    def _library_filename(name: str) -> str:
        system = platform.system()
        if system == "Windows":
            return f"{name}.dll"
        if system == "Darwin":
            return f"lib{name}.dylib"
        return f"lib{name}.so"

    def _load(self, name: str, required: bool) -> ctypes.CDLL | None:
        path = self.library_dir / self._library_filename(name)
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Missing {path}. Build first with: cmake -S . -B build && cmake --build build -j"
                )
            return None
        return ctypes.CDLL(str(path))

    @staticmethod
    def _configure(lib: ctypes.CDLL) -> None:
        doubles7 = [ctypes.c_double] * 7
        if hasattr(lib, "up_and_out_call_analytic"):
            lib.up_and_out_call_analytic.argtypes = doubles7
            lib.up_and_out_call_analytic.restype = ctypes.c_double
        signature = doubles7 + [
            ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        if hasattr(lib, "up_and_out_call_mc_cpu"):
            lib.up_and_out_call_mc_cpu.argtypes = signature
            lib.up_and_out_call_mc_cpu.restype = ctypes.c_int
        if hasattr(lib, "up_and_out_call_mc_cuda"):
            lib.up_and_out_call_mc_cuda.argtypes = signature
            lib.up_and_out_call_mc_cuda.restype = ctypes.c_int

    def analytic(self, p: BarrierParams) -> float:
        value = self.cpu.up_and_out_call_analytic(
            p.spot, p.strike, p.barrier, p.maturity,
            p.rate, p.dividend_yield, p.volatility,
        )
        if not math.isfinite(value):
            raise ValueError("Invalid barrier option inputs")
        return float(value)

    @property
    def cuda_available(self) -> bool:
        return self.cuda is not None

    def monte_carlo(
        self,
        p: BarrierParams,
        n_paths: int,
        n_steps: int = 64,
        rng: Rng = "mt",
        backend: Backend = "cpu",
        seed: int = 42,
        brownian_bridge: bool = True,
    ) -> MonteCarloResult:
        if rng not in ("mt", "sobol"):
            raise ValueError(f"Unknown RNG: {rng}")
        if backend not in ("cpu", "cuda"):
            raise ValueError(f"Unknown backend: {backend}")
        lib = self.cpu if backend == "cpu" else self.cuda
        if lib is None:
            raise RuntimeError("CUDA library is not built. Configure with a CUDA toolkit and rebuild.")
        fn = lib.up_and_out_call_mc_cpu if backend == "cpu" else lib.up_and_out_call_mc_cuda
        price = ctypes.c_double()
        stderr = ctypes.c_double()
        elapsed = ctypes.c_double()
        status = fn(
            p.spot, p.strike, p.barrier, p.maturity,
            p.rate, p.dividend_yield, p.volatility,
            n_paths, n_steps, 0 if rng == "mt" else 1, seed,
            int(brownian_bridge), ctypes.byref(price), ctypes.byref(stderr),
            ctypes.byref(elapsed),
        )
        if status != 0:
            messages = {
                1: "invalid input",
                2: "CUDA device/backend unavailable",
                3: "cuRAND initialization failed",
                4: "CUDA allocation failed",
                5: "cuRAND generation failed",
                6: "CUDA kernel launch failed",
            }
            raise RuntimeError(f"{backend} Monte Carlo failed: {messages.get(status, f'error {status}')}")
        return MonteCarloResult(
            price=price.value,
            standard_error=stderr.value,
            elapsed_ms=elapsed.value,
            backend=backend,
            rng=rng,
            n_paths=n_paths,
            n_steps=n_steps,
            brownian_bridge=brownian_bridge,
        )
