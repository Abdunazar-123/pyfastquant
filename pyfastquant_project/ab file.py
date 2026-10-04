import pyfastquant as pfq

# Параметры опциона
S0 = 100.0   # Текущая цена актива
K = 100.0    # Страйк
T = 1.0      # Срок (в годах)
r = 0.05     # Риск-фри ставка
sigma = 0.2  # Волатильность

# Расчет через C++ Monte Carlo
pricer = pfq.MonteCarloPricer(
    num_paths=500_000,
    num_time_steps=252,
    use_antithetic=True
)
price, std_err = pricer.price_european_option(S0, K, T, r, sigma, is_call=True)

print(f"Monte Carlo Price: {price:.4f} +/- {std_err:.4f}")