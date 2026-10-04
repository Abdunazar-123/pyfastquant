# PyFastQuant

A Monte Carlo option pricing engine with a three-layer architecture:

```
Layer 3: Python API (pyfastquant/pricer.py)     <- ergonomic, type-hinted, validated
Layer 2: pybind11 bridge (pybind_wrapper.cpp)    <- zero-copy NumPy, GIL release
Layer 1: C++ engine (monte_carlo.cpp)            <- AVX2 SIMD, OpenMP threading, cache-aligned memory
```

Includes an analytic Black-Scholes engine (used to validate the Monte Carlo
engine), antithetic-variates variance reduction with a proof of why it works,
and a small linear algebra module (SIMD matrix multiply, Cholesky
decomposition) for future multi-asset extensions.

**See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for the mathematical
derivation of the variance reduction technique and the empirical validation
— that document is the core of this project, not just the code.**

## Quick start

### 1. Prerequisites

- Python 3.8+
- CMake 3.15+
- A C++20 compiler with AVX2 support:
  - **Windows**: Visual Studio Build Tools with the "Desktop development
    with C++" workload (provides MSVC + the Windows SDK). Run commands
    below from the **Developer Command Prompt for VS**, not a plain CMD.
  - **Linux**: GCC 10+ or Clang 12+ (`sudo apt install build-essential`)
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)
- Optional: OpenMP (bundled with GCC/MSVC; on macOS install via
  `brew install libomp` if you want multi-threading — the library still
  builds and runs correctly without it, just single-threaded)

### 2. Build

```bash
pip install -e .
```

This runs CMake under the hood (see `setup.py` / `CMakeLists.txt`), compiles
the C++ engine, and installs the Python package in editable mode.

### 3. Verify the build

```bash
python test_run.py
```

For a deeper sanity check independent of the Python layer entirely (useful
if something goes wrong and you want to isolate whether the issue is in the
C++ engine or the pybind11 bridge):

```bash
# Linux/macOS
g++ -std=c++20 -O3 -mavx2 -mfma -fopenmp -Iinclude src/engine/*.cpp tests/native_test.cpp -o native_test && ./native_test

# Windows (Developer Command Prompt)
cl /std:c++20 /O2 /arch:AVX2 /openmp /Iinclude src\engine\*.cpp tests\native_test.cpp /Fe:native_test.exe
native_test.exe
```

This test was used to verify every numerical claim in `RESEARCH_NOTES.md`
before it was written — it checks the Monte Carlo engine against the
analytic Black-Scholes price, confirms antithetic variates measurably lower
the standard error, checks put-call parity, and validates the SIMD matrix
multiply and Cholesky decomposition against naive reference implementations.

### 4. Usage

```python
from pyfastquant import OptionPricer, BlackScholesPricer

pricer = OptionPricer(num_paths=1_000_000, num_time_steps=100)
result = pricer.price_european(S0=100, K=105, T=1.0, r=0.05, sigma=0.2, is_call=True)
print(result)

# Variance-reduced pricing at the same path count:
result_antithetic = pricer.price_european(
    S0=100, K=105, T=1.0, r=0.05, sigma=0.2, is_call=True, use_antithetic=True
)
print(f"Standard std_error:   {result.std_error:.6f}")
print(f"Antithetic std_error: {result_antithetic.std_error:.6f}")

# Analytic validation
bs = BlackScholesPricer()
print(bs.price_european(S0=100, K=105, T=1.0, r=0.05, sigma=0.2, is_call=True))
```

### 5. Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

Produces, under `benchmarks/results/`:
- `speed_comparison.png` — pure Python vs NumPy vs C++ engine, by path count
- `convergence.png` — Monte Carlo RMSE vs Black-Scholes, log-log, against the
  theoretical `O(1/sqrt(n))` reference line
- `thread_scaling.png` — runtime and speedup vs thread count (multi-core
  machines only — proves GIL release is doing real work, not just present
  in the code)
- `variance_reduction.png` — standard error, standard MC vs antithetic

### 6. Tests

```bash
pytest tests/test_pricer.py -v
```

## Project structure

```
pyfastquant_project/
├── CMakeLists.txt              cross-platform build config (MSVC + GCC/Clang)
├── setup.py / pyproject.toml   Python packaging, drives the CMake build
├── include/engine/
│   ├── memory_align.hpp        cache-aligned allocator (Windows + POSIX)
│   ├── simd_math.hpp           AVX2 norm_cdf, AVX2 sum/sum_sq reduction
│   ├── monte_carlo.hpp         engine interface (standard + antithetic)
│   ├── black_scholes.hpp       analytic pricer interface
│   └── linear_algebra.hpp      SIMD matmul, Cholesky interface
├── src/engine/                 implementations
├── src/bindings/
│   └── pybind_wrapper.cpp      Python <-> C++ bridge (zero-copy, GIL release)
├── pyfastquant/                Python package (pricer, pricing, utils)
├── benchmarks/
│   └── run_benchmarks.py       speed / convergence / scaling / variance plots
├── tests/
│   ├── native_test.cpp         C++-only correctness suite (no pybind11 needed)
│   └── test_pricer.py          Python-level pytest suite
├── RESEARCH_NOTES.md           the math: derivation + empirical validation
└── verify_simd.cpp             standalone SIMD correctness check
```

## What's mathematically verified here (not just claimed)

- Monte Carlo price converges to the analytic Black-Scholes price as
  `n -> infinity`, at the theoretical `O(1/sqrt(n))` rate (fitted slope:
  `-0.487` against a theoretical `-0.5`).
- Antithetic variates measurably reduce standard error (21.2% reduction
  observed at n=200,000 for a standard call contract), and Section 3.4 of
  `RESEARCH_NOTES.md` proves this reduction is *guaranteed* for monotonic
  payoffs (calls/puts), not just empirically likely.
- Put-call parity (`C - P = S0 - K*e^{-rT}`) holds within Monte Carlo noise.
- The AVX2 SIMD summation and matrix multiply are bit-for-bit / near-exact
  matches against naive scalar reference implementations.

## Next steps (not yet implemented)

- Quasi-Monte Carlo (Sobol/Halton sequences) for `O(1/n)` convergence
- Longstaff-Schwartz least-squares Monte Carlo for American-style options
- Multi-asset correlated GBM using the existing Cholesky decomposition
