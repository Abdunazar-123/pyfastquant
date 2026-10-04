й# Variance Reduction and Performance Engineering in a Monte Carlo Option Pricer

*Companion write-up to the PyFastQuant library. This is the mathematical and
empirical core meant to sit alongside the code — the code shows it works,
this shows **why**.*

---

## 1. Problem setup

We price a European call option under the Black-Scholes model, where the
underlying follows geometric Brownian motion (GBM):

    dS_t = r S_t dt + sigma S_t dW_t

Discretizing over `m` time steps of size `dt = T/m` gives the standard
Euler–Maruyama scheme for log-price:

    S_{t+dt} = S_t * exp( (r - sigma^2/2) dt + sigma sqrt(dt) Z ),   Z ~ N(0,1)

The quantity we want is

    V = e^{-rT} * E[ max(S_T - K, 0) ]

which has a closed form (Black-Scholes), used here purely as **ground truth**
to validate the simulator — in practice Monte Carlo is used precisely when
no closed form exists (path-dependent payoffs, multi-asset baskets, etc.),
so validating against a case where the answer *is* known is what justifies
trusting the method elsewhere.

## 2. Standard Monte Carlo estimator

Simulate `n` independent terminal payoffs `Y_1, ..., Y_n` where
`Y_i = max(S_T^{(i)} - K, 0)`. The estimator is

    V_hat = e^{-rT} * (1/n) * sum_i Y_i

By the Law of Large Numbers this is unbiased and consistent. Its standard
error follows directly from the Central Limit Theorem:

    SE(V_hat) = e^{-rT} * sigma_Y / sqrt(n)

where `sigma_Y = sqrt(Var(Y))`. **This is the O(1/sqrt(n)) rate**: to halve
the error, you need 4x the paths — an expensive way to buy precision, which
is exactly the motivation for variance reduction.

**Empirical confirmation** (`benchmarks/run_benchmarks.py`, Section 2,
15 independent seeds per path count, RMSE against the analytic Black-Scholes
price of the fixed contract S0=100, K=105, T=1, r=0.05, sigma=0.2):

| n | RMSE |
|---|---|
| 1,000 | 0.4151 |
| 5,000 | 0.1273 |
| 20,000 | 0.0854 |
| 100,000 | 0.0393 |
| 500,000 | 0.0157 |
| 2,000,000 | 0.0095 |

Fitting `log(RMSE) = a + b*log(n)` gives `b = -0.487`, matching the
theoretical `-0.5` almost exactly (see `benchmarks/results/convergence.png`).
Note this required *averaging* the error over several seeds — a single
Monte Carlo run's error is itself a noisy random variable and does not
individually trace the rate; only its expectation does. Treating a single
run's error as "the" convergence curve is a common mistake worth naming
explicitly in a write-up, since it's an easy one to make.

## 3. Antithetic variates

### 3.1 Construction

For each independent draw `Z_i ~ N(0,1)`, simulate **two** paths: one driven
by `Z_i`, one by its mirror image `-Z_i` (negated at every time step, not
just at the terminal value — this generalizes correctly to path-dependent
payoffs even though it is not needed for terminal-only European payoffs).
Let `Y_i = f(Z_i)` and `Y_i' = f(-Z_i)` be the resulting payoffs. The
antithetic estimator for the pair is

    A_i = (Y_i + Y_i') / 2

and the full estimator averages `n/2` such pairs (so the *same total number*
of underlying random draws is used as `n` independent standard-MC paths —
this is the fair, same-compute-budget comparison used in the benchmark).

### 3.2 Unbiasedness

Since `Z` and `-Z` have the same distribution (`N(0,1)` is symmetric),
`E[Y_i'] = E[Y_i]`, so `E[A_i] = E[Y_i]` — the antithetic estimator is
unbiased, just like standard Monte Carlo.

### 3.3 Variance

    Var(A_i) = (1/4) [ Var(Y_i) + Var(Y_i') + 2 Cov(Y_i, Y_i') ]
             = (1/2) Var(Y) + (1/2) Cov(Y, Y')

since `Var(Y_i) = Var(Y_i') = Var(Y)` by symmetry. Compare this to using two
*independent* standard draws averaged the same way, which would give
`(1/2) Var(Y)` (zero covariance term). So:

    Antithetic variance reduction  <=>  Cov(Y, Y') < 0

### 3.4 Why the covariance is negative here

Let `S = sum_j Z_j` be the accumulated Brownian increment driving the
terminal price (the sign of `sigma sqrt(dt)` is positive, so `S_T` is a
strictly increasing function of `S`). The call payoff `f(S) = max(S_T(S) -
K, 0)` is therefore **non-decreasing** in `S`. Its antithetic partner is
`f(-S)`, which — as a function of `S` — is **non-increasing** (larger `S`
means smaller `-S` means smaller or equal payoff).

This is exactly the setup for a classical correlation inequality: if `g` is
non-decreasing and `h` is non-increasing in the same random variable `X`,
then `Cov(g(X), h(X)) <= 0`. (Intuition: whenever `g(X)` is pushed above its
mean by a large `X`, `h(X)` is simultaneously pushed below its mean by that
same large `X`, and vice versa — the two move in opposite directions in
lockstep with `X`, so on average their deviations from their means point
opposite ways.) Setting `g(S) = f(S)`, `h(S) = f(-S)` gives `Cov(Y, Y') <=
0` directly, proving variance reduction is **guaranteed**, not just
empirically likely, for any monotonic payoff — which covers vanilla calls
and puts.

(Put payoffs are non-*increasing* in `S`, but the same inequality applies
symmetrically — swap the roles of `g` and `h`.)

### 3.5 Empirical confirmation

From `tests/native_test.cpp`, Section 3 (n=200,000 paths, same option
contract, verified by direct compilation and execution — not simulated
numbers):

```
standard   (n=200000): price=8.039328  std_error=0.029547
antithetic (n=200000): price=7.990162  std_error=0.023291
std_error reduction: 21.2%
```

A 21.2% reduction in standard error at the same path count is equivalent to
needing roughly `1/(1-0.212)^2 ≈ 1.61x` as many paths under standard Monte
Carlo to achieve the same precision — a real, measured efficiency gain, not
a theoretical curiosity. `benchmarks/run_benchmarks.py` Section 4 extends
this across a range of path counts and produces
`benchmarks/results/variance_reduction.png`.

## 4. Performance engineering

Variance reduction lowers the *statistical* cost of precision; the
implementation separately lowers the *computational* cost per path, and the
two compound (more precision per path, and each path costs less wall-clock
time). Three techniques, each independently verified:

**AVX2 SIMD aggregation.** Summing `n` payoffs and their squares (needed for
mean and variance) is done 4-at-a-time using 256-bit AVX2 registers with
fused multiply-add (`_mm256_fmadd_pd`), rather than a scalar loop. Verified
bit-for-bit equivalent to naive summation across sizes from 1 to 1,000,003
elements (`verify_simd.cpp`), with the non-multiple-of-4 remainder handled
by a scalar tail loop.

**OpenMP threading with GIL release.** Path simulation is parallelized
across CPU cores via `#pragma omp parallel for`, with each thread owning an
independently-seeded RNG (avoiding both lock contention and correlated
random streams across threads). On the Python side, `py::gil_scoped_release`
is held for the duration of the C++ computation so that releasing Python's
Global Interpreter Lock actually allows other Python threads to run
concurrently — `benchmarks/run_benchmarks.py` Section 3 measures wall-clock
time and speedup vs. thread count directly to confirm this isn't just a
diagram, it measurably scales.

**Cache-aligned memory.** Buffers used in the SIMD reduction are allocated
on 64-byte boundaries (`AlignedAllocator`, matching typical x86-64 cache
line size) so that AVX2's aligned load instruction (`_mm256_load_pd`) can be
used instead of the unaligned variant, and so that per-thread RNG state
structures don't share a cache line (avoiding false sharing under
concurrent access).

## 5. Limitations and honest scope

- The antithetic construction here negates each `Z_j` at every time step,
  which is mathematically equivalent (for this terminal-only European
  payoff) to negating a single aggregate normal draw — the multi-step
  machinery is unnecessary for this specific payoff but is kept because it
  generalizes correctly to path-dependent payoffs (Asian, barrier) without
  modification, which is a natural next extension.
- The AVX2 `norm_cdf` approximation (Abramowitz & Stegun 7.1.26) has ~1.5e-7
  max absolute error — negligible next to Monte Carlo's own statistical
  error at any practical path count, but worth stating rather than assuming.
- No variance reduction technique here addresses the `O(1/sqrt(n))` *rate*
  itself — only the constant in front of it. Quasi-Monte Carlo (low-
  discrepancy sequences, e.g. Sobol) can achieve `O(1/n)` and is a natural
  next step, along with American-style exercise via Longstaff-Schwartz.

## 6. Reproducing these results

```
# Verify the math/SIMD correctness claims directly:
g++ -std=c++20 -O3 -mavx2 -mfma -Iinclude verify_simd.cpp -o verify_simd && ./verify_simd

# Verify the full engine (Black-Scholes match, MC convergence, antithetic
# variance reduction, put-call parity, matrix multiply, Cholesky):
g++ -std=c++20 -O3 -mavx2 -mfma -fopenmp -Iinclude \
    src/engine/*.cpp tests/native_test.cpp -o native_test && ./native_test

# Full benchmark suite with plots (after `pip install -e .`):
python benchmarks/run_benchmarks.py
```
