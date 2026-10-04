"""PyFastQuant - high-performance quantitative finance library.

C++ Monte Carlo engine (AVX2 SIMD + OpenMP threading) exposed to
Python through pybind11, with an analytic Black-Scholes engine for
validation and a small linear algebra module (SIMD matrix multiply,
Cholesky decomposition) for future multi-asset / correlated-path work.
"""

try:
    from . import _core
except ImportError as e:
    raise ImportError(
        "PyFastQuant's native extension was not found. Build it with:\n"
        "    pip install -e .\n"
        "This requires CMake and a C++ compiler with C++20 + AVX2 support "
        "(MSVC via 'Desktop development with C++', or GCC/Clang)."
    ) from e

__version__ = _core.__version__

from .pricer import OptionPricer, BlackScholesPricer
from .pricing import PriceResult
from .utils import get_hardware_info, benchmark_engine

__all__ = [
    "OptionPricer",
    "BlackScholesPricer",
    "PriceResult",
    "get_hardware_info",
    "benchmark_engine",
]
