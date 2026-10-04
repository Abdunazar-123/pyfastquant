"""Helper utilities for introspection and quick benchmarking."""

import sys
import platform
import time
from typing import Dict, Any
import numpy as np

from . import _core


def get_hardware_info() -> Dict[str, Any]:
    return {
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "CPU Cores": _core.get_hardware_concurrency(),
        "Python Version": sys.version.split()[0],
        "Cache Line Size": _core.get_cache_line_size(),
        "AVX2 Support": _core.has_avx2(),
        "NumPy Version": np.__version__,
        "PyFastQuant Version": _core.__version__,
    }


def benchmark_engine(
    num_runs: int = 5,
    num_paths: int = 1_000_000,
    num_time_steps: int = 100,
    use_antithetic: bool = False,
) -> Dict[str, Any]:
    """Quick benchmark of a single pricing call, repeated num_runs times.

    For a full standard-vs-antithetic-vs-NumPy-vs-pure-Python comparison
    with plots, see benchmarks/run_benchmarks.py instead -- this
    function is meant for a fast sanity check from a Python shell.
    """
    from .pricer import OptionPricer

    pricer = OptionPricer(num_paths=num_paths, num_time_steps=num_time_steps)

    times = []
    prices = []
    std_errors = []

    print(f"Benchmark: {num_paths:,} paths, {num_time_steps} steps, "
          f"{pricer.threads} threads, antithetic={use_antithetic}")

    for i in range(num_runs):
        start = time.perf_counter()
        result = pricer.price_european(100.0, 105.0, 1.0, 0.05, 0.2, True, use_antithetic)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        prices.append(result.price)
        std_errors.append(result.std_error)
        print(f"  Run {i+1}: {elapsed:.2f} ms, price=${result.price:.4f}, "
              f"std_error=${result.std_error:.4f}")

    return {
        "num_runs": num_runs,
        "num_paths": num_paths,
        "threads": pricer.threads,
        "use_antithetic": use_antithetic,
        "avg_time_ms": float(np.mean(times)),
        "min_time_ms": float(np.min(times)),
        "max_time_ms": float(np.max(times)),
        "avg_price": float(np.mean(prices)),
        "avg_std_error": float(np.mean(std_errors)),
        "paths_per_second": num_paths / (np.mean(times) / 1000),
    }
