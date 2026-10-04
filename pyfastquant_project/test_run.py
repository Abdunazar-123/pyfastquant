"""Quick smoke test -- run after `pip install -e .` to confirm the build works."""

from pyfastquant import OptionPricer, BlackScholesPricer, get_hardware_info

print("=" * 60)
print("PyFastQuant smoke test")
print("=" * 60)

print("\n[Hardware / build info]")
for k, v in get_hardware_info().items():
    print(f"  {k}: {v}")

print("\n[1] Monte Carlo pricer...")
pricer = OptionPricer(num_paths=100_000, num_time_steps=100)
print(f"    {pricer}")

result = pricer.price_european(S0=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2, is_call=True)
print(f"\n{result}")

print("\n[2] Antithetic variates (same path count, should have lower std_error)...")
result_anti = pricer.price_european(
    S0=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2, is_call=True, use_antithetic=True
)
print(f"\n{result_anti}")
assert result_anti.std_error < result.std_error, "antithetic should reduce std_error"
print("\n  OK: antithetic std_error is lower than standard.")

print("\n[3] Black-Scholes analytic pricer...")
bs = BlackScholesPricer()
bs_result = bs.price_european(S0=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2, is_call=True)
print(f"\n{bs_result}")

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)
