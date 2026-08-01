from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from barrier_mc import BarrierPricer  # noqa: E402


def main() -> int:
    print(f"CONDA_PREFIX={os.environ.get('CONDA_PREFIX')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"torch.version.cuda={torch.version.cuda}")
        print(f"torch.cuda.is_available={torch.cuda.is_available()}")
        print(f"torch.cuda.device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch_error={exc!r}")

    pricer = BarrierPricer(ROOT / "build" / "lib")
    print(f"barrier_mc.cuda_available={pricer.cuda_available}")
    print(f"barrier_mc.cuda_unavailable_reason={pricer.cuda_unavailable_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
