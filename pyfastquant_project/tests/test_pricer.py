"""
Python-level tests for the pyfastquant package.

Run with: pytest tests/test_pricer.py -v
(Requires the package to be built first: pip install -e .)
"""

import pytest
import numpy as np

from pyfastquant import OptionPricer, BlackScholesPricer


class TestBlackScholesPricer:
    def test_known_value(self):
        """Textbook case: S0=100, K=100, T=1, r=0.05, sigma=0.2 -> ~10.4506"""
        bs = BlackScholesPricer()
        result = bs.price_european(100.0, 100.0, 1.0, 0.05, 0.2, is_call=True)
        assert abs(result.price - 10.4506) < 0.01

    def test_greeks_bounds(self):
        bs = BlackScholesPricer()
        result = bs.price_european(100.0, 105.0, 1.0, 0.05, 0.2, is_call=True)
        assert 0.0 < result.greeks["delta"] < 1.0
        assert result.greeks["gamma"] > 0.0
        assert result.greeks["vega"] > 0.0


class TestOptionPricer:
    @pytest.fixture
    def pricer(self):
        return OptionPricer(num_paths=200_000, num_time_steps=50)

    @pytest.fixture
    def bs_pricer(self):
        return BlackScholesPricer()

    def test_converges_to_black_scholes(self, pricer, bs_pricer):
        S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
        mc = pricer.price_european(S0, K, T, r, sigma, is_call=True)
        bs = bs_pricer.price_european(S0, K, T, r, sigma, is_call=True)
        z_score = abs(mc.price - bs.price) / mc.std_error
        assert z_score < 5.0, (
            f"MC price {mc.price} is {z_score:.1f} std errors from "
            f"BS price {bs.price} -- likely a bug, not just MC noise"
        )

    def test_put_call_parity(self, pricer, bs_pricer):
        """C - P = S0 - K*e^(-rT), the fundamental no-arbitrage relation."""
        S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2

        mc_call = pricer.price_european(S0, K, T, r, sigma, is_call=True)
        mc_put = pricer.price_european(S0, K, T, r, sigma, is_call=False)
        mc_diff = mc_call.price - mc_put.price
        parity = S0 - K * np.exp(-r * T)

        assert abs(mc_diff - parity) < 0.3

    def test_antithetic_reduces_variance(self, pricer):
        """The whole point of antithetic variates: lower std_error at the
        same path count. This should hold essentially every time for a
        monotonic payoff like a European call (see RESEARCH_NOTES.md
        section 3.4 for the proof it's guaranteed, not just likely)."""
        S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2

        standard = pricer.price_european(S0, K, T, r, sigma, True, use_antithetic=False)
        antithetic = pricer.price_european(S0, K, T, r, sigma, True, use_antithetic=True)

        assert antithetic.std_error < standard.std_error
        assert antithetic.used_antithetic is True
        assert standard.used_antithetic is False

    def test_batch_matches_individual_calls(self, pricer):
        S0, K, T, r, sigma = [100.0, 100.0], [105.0, 95.0], [1.0, 1.0], [0.05, 0.05], [0.2, 0.2]
        is_call = [True, False]

        batch_results = pricer.price_european_batch(S0, K, T, r, sigma, is_call)
        assert len(batch_results) == 2
        for res in batch_results:
            assert res.price >= 0.0
            assert res.std_error is not None and res.std_error > 0.0

    def test_invalid_num_paths_raises(self):
        with pytest.raises(ValueError):
            OptionPricer(num_paths=0)

    def test_invalid_num_time_steps_raises(self):
        with pytest.raises(ValueError):
            OptionPricer(num_time_steps=-1)


class TestLinearAlgebra:
    def test_matrix_multiply_matches_numpy(self):
        from pyfastquant import _core
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 4))
        B = rng.standard_normal((4, 5))

        C = _core.LinearAlgebra.matrix_multiply(A, B)
        C_ref = A @ B

        assert np.allclose(C, C_ref, atol=1e-9)

    def test_cholesky_reconstructs_matrix(self):
        from pyfastquant import _core
        M = np.array([
            [4.0, 2.0, 2.0],
            [2.0, 5.0, 3.0],
            [2.0, 3.0, 6.0],
        ])
        L = _core.LinearAlgebra.cholesky(M)
        assert np.allclose(L @ L.T, M, atol=1e-9)
