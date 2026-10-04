#!/usr/bin/env python3
"""
PyFastQuant benchmark suite.

Run after building the package (`pip install -e .` from the project
root). Produces, under benchmarks/results/:

  speed_comparison.png     pure Python vs NumPy vs C++ engine, by path count
  speed_comparison.csv     the raw numbers behind that plot
  convergence.png          |MC price - Black-Scholes price| vs num_paths,
                            log-log, with a fitted slope and the
                            theoretical O(1/sqrt(n)) reference line
  thread_scaling.png       runtime and speedup vs thread count
  variance_reduction.png   standard error: standard MC vs antithetic MC

If pyfastquant hasn't been built yet, the script still runs the pure
Python / NumPy sections and the theoretical convergence reference line,
so you can sanity-check the benchmarking logic itself before the C++
extension is compiled.
"""

import math
import time
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Fixed option parameters used throughout, so every section is
# comparing apples to apples.
S0, K, T, R, SIGMA = 100.0, 105.0, 1.0, 0.05, 0.2
NUM_STEPS = 50


# -------------------------------------------------------------------
# Reference Black-Scholes price (pure Python, no C++ dependency --
# this is the ground truth every Monte Carlo method is compared to).
# -------------------------------------------------------------------
def norm_cdf_ref(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call_ref(S0, K, T, r, sigma) -> float:
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm_cdf_ref(d1) - K * math.exp(-r * T) * norm_cdf_ref(d2)


BS_PRICE = black_scholes_call_ref(S0, K, T, R, SIGMA)


# -------------------------------------------------------------------
# Method 1: pure Python, nested loops. This is the naive baseline
# every "why is my code slow" story starts from.
# -------------------------------------------------------------------
def mc_pure_python(num_paths: int, num_steps: int = NUM_STEPS, seed: int = 42) -> float:
    import random
    rng = random.Random(seed)
    dt = T / num_steps
    drift = (R - 0.5 * SIGMA ** 2) * dt
    diffusion = SIGMA * math.sqrt(dt)
    total = 0.0
    for _ in range(num_paths):
        S = S0
        for _ in range(num_steps):
            z = rng.gauss(0.0, 1.0)
            S *= math.exp(drift + diffusion * z)
        total += max(S - K, 0.0)
    return math.exp(-R * T) * (total / num_paths)


# -------------------------------------------------------------------
# Method 2: NumPy, fully vectorized (no Python-level loop over paths
# or steps). This is the ceiling for "pure Python ecosystem" speed.
# -------------------------------------------------------------------
def mc_numpy(num_paths: int, num_steps: int = NUM_STEPS, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    dt = T / num_steps
    drift = (R - 0.5 * SIGMA ** 2) * dt
    diffusion = SIGMA * math.sqrt(dt)
    Z = rng.standard_normal((num_paths, num_steps))
    log_returns = drift + diffusion * Z
    log_S_T = np.log(S0) + log_returns.sum(axis=1)
    S_T = np.exp(log_S_T)
    payoffs = np.maximum(S_T - K, 0.0)
    return math.exp(-R * T) * payoffs.mean()


# -------------------------------------------------------------------
# Method 3: the C++/AVX2/OpenMP engine, via pyfastquant. Optional
# import so the rest of the script still runs before you've built it.
# -------------------------------------------------------------------
try:
    from pyfastquant import OptionPricer, get_hardware_info
    HAVE_PYFASTQUANT = True
except ImportError:
    HAVE_PYFASTQUANT = False
    print(
        "NOTE: pyfastquant C++ extension is not built/importable here.\n"
        "      Run `pip install -e .` from the project root first, then\n"
        "      re-run this script for the full C++ / thread-scaling /\n"
        "      variance-reduction comparisons. Continuing with the pure\n"
        "      Python and NumPy sections only.\n"
    )


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


# =====================================================================
# Section 1: speed comparison across path counts
# =====================================================================
def section_speed_comparison():
    print("=" * 70)
    print("SECTION 1: Pure Python vs NumPy vs C++ engine")
    print("=" * 70)

    # Pure Python is O(num_paths * num_steps) with real Python-level
    # overhead per iteration, so it gets painfully slow fast -- cap it
    # lower than the other two methods or this section takes forever.
    python_path_counts = [1_000, 10_000, 50_000]
    fast_path_counts = [1_000, 10_000, 100_000, 1_000_000]

    rows = []

    for n in fast_path_counts:
        row = {"num_paths": n}

        if n in python_path_counts:
            price, t_ms = timed(mc_pure_python, n)
            row["python_ms"] = t_ms
            row["python_price"] = price
            print(f"  n={n:>9,}  pure Python : {t_ms:9.2f} ms  price=${price:.4f}")
        else:
            row["python_ms"] = None
            row["python_price"] = None

        price, t_ms = timed(mc_numpy, n)
        row["numpy_ms"] = t_ms
        row["numpy_price"] = price
        print(f"  n={n:>9,}  NumPy       : {t_ms:9.2f} ms  price=${price:.4f}")

        if HAVE_PYFASTQUANT:
            pricer = OptionPricer(num_paths=n, num_time_steps=NUM_STEPS)
            result, t_ms = timed(pricer.price_european, S0, K, T, R, SIGMA, True, False)
            row["cpp_ms"] = result.computation_time_ms
            row["cpp_price"] = result.price
            print(f"  n={n:>9,}  C++ engine  : {result.computation_time_ms:9.2f} ms  "
                  f"price=${result.price:.4f}  "
                  f"({row['numpy_ms']/result.computation_time_ms:.1f}x vs NumPy)")
        else:
            row["cpp_ms"] = None
            row["cpp_price"] = None

        rows.append(row)
        print()

    # CSV
    with open(RESULTS_DIR / "speed_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plot: log-log runtime vs path count
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ns = [r["num_paths"] for r in rows]

    py_ns = [r["num_paths"] for r in rows if r["python_ms"] is not None]
    py_ms = [r["python_ms"] for r in rows if r["python_ms"] is not None]
    if py_ns:
        ax.plot(py_ns, py_ms, "o-", label="Pure Python", color="#d62728")

    ax.plot(ns, [r["numpy_ms"] for r in rows], "s-", label="NumPy (vectorized)", color="#ff7f0e")

    if HAVE_PYFASTQUANT:
        ax.plot(ns, [r["cpp_ms"] for r in rows], "^-", label="C++ / AVX2 / OpenMP", color="#2ca02c")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Monte Carlo pricing runtime: Python vs NumPy vs C++ engine")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "speed_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'speed_comparison.png'}\n")


# =====================================================================
# Section 2: convergence to the Black-Scholes analytic price
# =====================================================================
def section_convergence():
    print("=" * 70)
    print("SECTION 2: Convergence to Black-Scholes (theoretical O(1/sqrt(n)))")
    print("=" * 70)
    print(f"Analytic Black-Scholes price: {BS_PRICE:.6f}\n")
    print("Note: a *single* Monte Carlo run's error is a noisy random draw --")
    print("it does not reliably trace 1/sqrt(n) on its own. What actually")
    print("follows that rate is the *expected* error, so each point below is")
    print("averaged (RMSE) over several independent seeds at that path count.\n")

    path_counts = [1_000, 5_000, 20_000, 100_000, 500_000, 2_000_000]
    n_trials = 15
    errors_numpy = []

    for n in path_counts:
        trial_errors = []
        for trial in range(n_trials):
            price = mc_numpy(n, seed=1000 + trial)
            trial_errors.append(price - BS_PRICE)
        rmse = float(np.sqrt(np.mean(np.square(trial_errors))))
        errors_numpy.append(rmse)
        print(f"  n={n:>9,}  RMSE over {n_trials} seeds = {rmse:.6f}")

    errors_cpp = None
    if HAVE_PYFASTQUANT:
        errors_cpp = []
        print()
        for n in path_counts:
            trial_errors = []
            last_std_error = None
            for trial in range(n_trials):
                pricer = OptionPricer(num_paths=n, num_time_steps=NUM_STEPS)
                result = pricer.price_european(S0, K, T, R, SIGMA, True, False)
                trial_errors.append(result.price - BS_PRICE)
                last_std_error = result.std_error
            rmse = float(np.sqrt(np.mean(np.square(trial_errors))))
            errors_cpp.append(rmse)
            print(f"  n={n:>9,}  C++ RMSE over {n_trials} seeds = {rmse:.6f}  "
                  f"(std_error from one run: {last_std_error:.6f})")

    # Fit slope in log-log space to check it's close to the theoretical -0.5
    log_n = np.log(path_counts)
    log_err = np.log(errors_numpy)
    slope, intercept = np.polyfit(log_n, log_err, 1)
    print(f"\nFitted convergence rate (NumPy): error ~ n^{slope:.3f} "
          f"(theory predicts -0.5)")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(path_counts, errors_numpy, "o-", label=f"NumPy MC RMSE ({n_trials} seeds)", color="#ff7f0e")
    if errors_cpp:
        ax.plot(path_counts, errors_cpp, "^-", label=f"C++ MC RMSE ({n_trials} seeds)", color="#2ca02c")

    # theoretical O(1/sqrt(n)) reference line, anchored to match at the
    # first data point so the *slope* is what's being compared visually
    ref = errors_numpy[0] * np.sqrt(path_counts[0] / np.array(path_counts, dtype=float))
    ax.plot(path_counts, ref, "k--", label=r"Theoretical $O(1/\sqrt{n})$", alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths (n)")
    ax.set_ylabel("RMSE vs Black-Scholes price")
    ax.set_title("Monte Carlo convergence rate vs analytic Black-Scholes price")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'convergence.png'}\n")


# =====================================================================
# Section 3: thread scaling (requires pyfastquant + multiple cores)
# =====================================================================
def section_thread_scaling():
    print("=" * 70)
    print("SECTION 3: Thread scaling (proves GIL release is doing real work)")
    print("=" * 70)

    if not HAVE_PYFASTQUANT:
        print("Skipped: pyfastquant not built.\n")
        return

    import os
    max_threads = os.cpu_count() or 1
    if max_threads < 2:
        print(f"Skipped: this machine only reports {max_threads} CPU core(s), "
              f"so thread scaling can't be demonstrated here. Re-run this "
              f"script on a multi-core machine.\n")
        return

    thread_counts = sorted(set(
        [t for t in [1, 2, 4, 8, max_threads] if t <= max_threads]
    ))
    n = 4_000_000
    times = []

    for t in thread_counts:
        pricer = OptionPricer(num_paths=n, num_time_steps=NUM_STEPS, threads=t)
        result = pricer.price_european(S0, K, T, R, SIGMA, True, False)
        times.append(result.computation_time_ms)
        speedup = times[0] / result.computation_time_ms
        print(f"  threads={t:>3}  time={result.computation_time_ms:9.2f} ms  "
              f"speedup vs 1 thread: {speedup:.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.plot(thread_counts, times, "o-", color="#1f77b4")
    ax1.set_xlabel("Threads")
    ax1.set_ylabel("Runtime (ms)")
    ax1.set_title(f"Runtime vs thread count (n={n:,} paths)")
    ax1.grid(True, alpha=0.3)

    speedups = [times[0] / t for t in times]
    ax2.plot(thread_counts, speedups, "o-", color="#2ca02c", label="Measured speedup")
    ax2.plot(thread_counts, thread_counts, "k--", alpha=0.5, label="Ideal linear speedup")
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Speedup vs 1 thread")
    ax2.set_title("Speedup vs thread count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "thread_scaling.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'thread_scaling.png'}\n")


# =====================================================================
# Section 4: variance reduction from antithetic variates
# =====================================================================
def section_variance_reduction():
    print("=" * 70)
    print("SECTION 4: Antithetic variates -- variance reduction")
    print("=" * 70)

    if not HAVE_PYFASTQUANT:
        print("Skipped: pyfastquant not built.\n")
        return

    path_counts = [10_000, 50_000, 200_000, 1_000_000]
    std_errors_standard = []
    std_errors_antithetic = []

    for n in path_counts:
        pricer = OptionPricer(num_paths=n, num_time_steps=NUM_STEPS)
        r_std = pricer.price_european(S0, K, T, R, SIGMA, True, False)
        r_anti = pricer.price_european(S0, K, T, R, SIGMA, True, True)
        std_errors_standard.append(r_std.std_error)
        std_errors_antithetic.append(r_anti.std_error)
        reduction = 100.0 * (1.0 - r_anti.std_error / r_std.std_error)
        print(f"  n={n:>9,}  standard std_error={r_std.std_error:.6f}  "
              f"antithetic std_error={r_anti.std_error:.6f}  "
              f"reduction={reduction:5.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(path_counts, std_errors_standard, "o-", label="Standard MC", color="#d62728")
    ax.plot(path_counts, std_errors_antithetic, "^-", label="Antithetic variates", color="#2ca02c")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Standard error of price estimate")
    ax.set_title("Variance reduction: standard MC vs antithetic variates")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "variance_reduction.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'variance_reduction.png'}\n")


if __name__ == "__main__":
    if HAVE_PYFASTQUANT:
        info = get_hardware_info()
        print("Hardware / build info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()

    section_speed_comparison()
    section_convergence()
    section_thread_scaling()
    section_variance_reduction()

    print("=" * 70)
    print(f"All results saved under: {RESULTS_DIR}")
    print("=" * 70)
