"""Try ways of getting numba's CUDA support working, each in its own process (a crash only ends one).

Run through hpc/lidar/check_numba_variants.sh on a GPU node. Each variant creates a numba CUDA
context and launches a tiny kernel; the table at the end says which survived.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

KERNEL = """
import numpy as np
from numba import cuda
@cuda.jit
def add(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]
a = np.arange(8, dtype=np.float32); b = np.ones(8, dtype=np.float32); out = np.zeros(8, dtype=np.float32)
add[1, 8](a, b, out)
print("kernel result", out.tolist())
"""

VARIANTS: List[Tuple[str, str, Dict[str, str]]] = [
    ("A numba first, no torch context", KERNEL, {}),
    ("B torch context first, then numba (the evaluation's order)",
     "import torch; torch.zeros(1).cuda()\n" + KERNEL, {}),
    ("C nvidia cuda-python binding, numba first",
     KERNEL, {"NUMBA_CUDA_USE_NVIDIA_BINDING": "1"}),
    ("D nvidia cuda-python binding, torch context first",
     "import torch; torch.zeros(1).cuda()\n" + KERNEL, {"NUMBA_CUDA_USE_NVIDIA_BINDING": "1"}),
    ("E per-thread default stream off, numba first",
     KERNEL, {"NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM": "0", "NUMBA_CUDA_LOG_LEVEL": "WARNING"}),
]


def run(code: str, extra: Dict[str, str]) -> Tuple[int, str]:
    """Run ``code`` in a fresh interpreter with ``extra`` environment; (exit code, last output)."""
    env = dict(os.environ, **extra, PYTHONFAULTHANDLER="1")
    done = subprocess.run(
        [sys.executable, "-u", "-c", code],
        env=env, capture_output=True, text=True, timeout=240, check=False,
    )
    text = (done.stdout + done.stderr).strip().splitlines()
    return done.returncode, " | ".join(text[-2:]) if text else ""


def main() -> None:
    """Run every variant and print a summary table."""
    print("numba CUDA variants (exit 0 = works, 139 = segmentation fault)")
    results: List[Tuple[str, int, str]] = []
    for name, code, extra in VARIANTS:
        status, tail = run(code, extra)
        results.append((name, status, tail))
        print(f"{'OK  ' if status == 0 else 'FAIL'} exit {status:4d}  {name}")
        if status != 0:
            print(f"        {tail[:200]}")
    best: Optional[str] = next((n for n, s, _ in results if s == 0), None)
    print("first working variant:", best)


if __name__ == "__main__":
    main()
