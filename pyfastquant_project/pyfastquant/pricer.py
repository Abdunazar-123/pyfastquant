"""User-facing pricer API."""

from typing import List
import numpy as np

from . import _core
from .pricing import PriceResult


class OptionPricer:
    """Monte Carlo European option pricer, backed by a C++/AVX2/OpenMP engine."""

    def __init__(
        self,
        num_paths: int = 1_000_000,
        num_time_steps: int = 100,
        threads: int = 0,
    ):
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive, got {num_paths}")
        if num_time_steps <= 0:
            raise ValueError(f"num_time_steps must be positive, got {num_time_steps}")

        self._engine = _core.MonteCarloEngine(
            num_paths=num_paths,
            num_time_steps=num_time_steps,
            threads=threads,
        )
        self._num_paths = num_paths
        self._num_time_steps = num_time_steps
        self._threads = threads if threads > 0 else _core.get_hardware_concurrency()

    @property
    def num_paths(self) -> int:
        return self._num_paths

    @property
    def num_time_steps(self) -> int:
        return self._num_time_steps

    @property
    def threads(self) -> int:
        return self._threads

    def price_european(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool = True,
        use_antithetic: bool = False,
    ) -> PriceResult:
        """Price a single European option.

        Set use_antithetic=True for variance-reduced pricing: internally
        this draws num_paths/2 independent normal variates and pairs
        each with its mirror image, which lowers the standard error for
        monotonic payoffs (calls/puts) at the same path count.
        """
        result = self._engine.price_european_option(
            S0, K, T, r, sigma, is_call, use_antithetic
        )
        return PriceResult(
            price=result["price"],
            std_error=result["std_error"],
            confidence_interval=(
                result["confidence_interval_low"],
                result["confidence_interval_high"],
            ),
            computation_time_ms=result["computation_time_ms"],
            used_antithetic=result["used_antithetic"],
        )

    def price_european_batch(
        self,
        S0: List[float],
        K: List[float],
        T: List[float],
        r: List[float],
        sigma: List[float],
        is_call: List[bool],
        use_antithetic: bool = False,
    ) -> List[PriceResult]:
        S0_arr = np.asarray(S0, dtype=np.float64)
        K_arr = np.asarray(K, dtype=np.float64)
        T_arr = np.asarray(T, dtype=np.float64)
        r_arr = np.asarray(r, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        is_call_arr = np.asarray(is_call, dtype=bool)

        results = self._engine.price_european_option_batch(
            S0_arr, K_arr, T_arr, r_arr, sigma_arr, is_call_arr, use_antithetic
        )

        return [
            PriceResult(
                price=r["price"],
                std_error=r["std_error"],
                confidence_interval=(
                    r["confidence_interval_low"],
                    r["confidence_interval_high"],
                ),
                computation_time_ms=r["computation_time_ms"],
                used_antithetic=r["used_antithetic"],
            )
            for r in results
        ]

    def __repr__(self) -> str:
        return (
            f"OptionPricer(num_paths={self.num_paths:,}, "
            f"num_time_steps={self.num_time_steps}, "
            f"threads={self.threads})"
        )


class BlackScholesPricer:
    """Analytic Black-Scholes pricer -- the validation baseline for OptionPricer."""

    def price_european(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool = True,
    ) -> PriceResult:
        result = _core.BlackScholesEngine.price_option(S0, K, T, r, sigma, is_call)
        return PriceResult(
            price=result["price"],
            greeks={
                "delta": result["delta"],
                "gamma": result["gamma"],
                "theta": result["theta"],
                "vega": result["vega"],
                "rho": result["rho"],
            },
        )

    def __repr__(self) -> str:
        return "BlackScholesPricer()"
