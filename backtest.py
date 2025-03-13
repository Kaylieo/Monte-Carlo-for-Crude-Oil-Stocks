import pandas as pd
import numpy as np
import scipy.stats as st
from monte_carlo import run_ms_egarch_simulation_logic  # Use your hybrid MSGARCH model
import sqlite3
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load and Prepare Historical Data
# ------------------------------
historical_data = pd.read_csv("BP.csv", parse_dates=["date"])
historical_data.sort_values("date", inplace=True)

# ------------------------------
# 2. Define Backtest Parameters
# ------------------------------
train_end_date = pd.Timestamp("2018-12-31")
test_data = historical_data[historical_data["date"] > train_end_date].copy()

num_simulations = 5000
forecast_horizon = 5       # Multi-day forecast horizon (e.g., 5 days)
confidence_level = 0.95
var_probability = 1 - confidence_level  # 0.05 for 5% VaR

step = 30  # Only run the simulation every 30 days to reduce computation

# ------------------------------
# 3. Initialize Counters and Storage
# ------------------------------
exceptions = 0
total_forecasts = 0

forecasted_vars = []
actual_prices = []
dates = []
hit_series = []

# ------------------------------
# 4. Rolling Window Backtest (Multi-Day)
# ------------------------------
for i in range(0, len(test_data) - forecast_horizon, step):
    # Forecast 'forecast_horizon' days ahead from index i
    test_row = test_data.iloc[i]
    test_date = test_row["date"]
    
    # The actual price we compare to is forecast_horizon days ahead
    comparison_index = i + forecast_horizon
    actual_row = test_data.iloc[comparison_index]
    actual_price = actual_row["close"]
    actual_date = actual_row["date"]

    # Training data: all historical data before test_date
    train_data = historical_data[historical_data["date"] < test_date].copy()
    if train_data.empty:
        continue

    # Try running the MSGARCH simulation
    try:
        sim_results = run_ms_egarch_simulation_logic(
            train_data, 
            ticker="BP", 
            num_days=forecast_horizon, 
            num_simulations=num_simulations
        )
        # Extract final simulated prices
        final_prices = sim_results["final_prices"]
        if final_prices.ndim > 1:
            final_prices = final_prices[-1, :]
    except ValueError as e:
        if "incorrect number of forecast days" in str(e):
            print("Warning: MSGARCH simulation did not return the full horizon. Falling back to 1-day forecast.")
            # Fallback: Run simulation with 1-day forecast
            sim_results = run_ms_egarch_simulation_logic(
                train_data, 
                ticker="BP", 
                num_days=1, 
                num_simulations=num_simulations
            )
            final_prices = sim_results["final_prices"]
            if final_prices.ndim > 1:
                final_prices = final_prices[-1, :]
            # Replicate the 1-day simulated final prices to match the desired forecast_horizon
            final_prices = np.tile(final_prices, forecast_horizon)
        else:
            raise

    var_forecast = np.percentile(final_prices, var_probability * 100)

    forecasted_vars.append(var_forecast)
    actual_prices.append(actual_price)
    dates.append(actual_date)

    hit = 1 if actual_price < var_forecast else 0
    hit_series.append(hit)
    exceptions += hit
    total_forecasts += 1

# ------------------------------
# 5. Calculate Exception Rate and Kupiec Test
# ------------------------------
exception_rate = exceptions / total_forecasts
print(f"Total forecasts: {total_forecasts}")
print(f"Exceptions (VaR breaches): {exceptions}")
print(f"Exception Rate: {exception_rate * 100:.2f}%")

n = total_forecasts
N = exceptions
p = var_probability
observed_rate = N / n if N > 0 else 1e-10

LR_pof = -2 * np.log(
    ((1 - p) ** (n - N) * (p) ** N) /
    ((1 - observed_rate) ** (n - N) * (observed_rate) ** N)
)
kupiec_pvalue = 1 - st.chi2.cdf(LR_pof, df=1)
print(f"Kupiec Test Statistic: {LR_pof:.2f}")
print(f"Kupiec Test p-value: {kupiec_pvalue:.4f}")

# ------------------------------
# 6. Christoffersen Conditional Coverage Test
# ------------------------------
def christoffersen_test(hits):
    hits = np.array(hits)
    T = len(hits)
    hits_lag = np.concatenate(([0], hits[:-1]))
    
    n00 = np.sum((hits_lag == 0) & (hits == 0))
    n01 = np.sum((hits_lag == 0) & (hits == 1))
    n10 = np.sum((hits_lag == 1) & (hits == 0))
    n11 = np.sum((hits_lag == 1) & (hits == 1))
    
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = np.sum(hits) / T
    
    L_indep = ((1 - pi) ** (T - np.sum(hits))) * (pi ** np.sum(hits))
    L_cc = ((1 - pi0) ** n00 * pi0 ** n01) * ((1 - pi1) ** n10 * pi1 ** n11)
    
    LR_cc = -2 * np.log(L_indep / L_cc) if L_cc > 0 else np.nan
    p_value = 1 - st.chi2.cdf(LR_cc, df=2)
    return LR_cc, p_value

LR_cc, cond_pvalue = christoffersen_test(hit_series)
print(f"Christoffersen Test Statistic: {LR_cc:.2f}")
print(f"Christoffersen Test p-value: {cond_pvalue:.4f}")

# ------------------------------
# 7. Visualization
# ------------------------------
backtest_results = pd.DataFrame({
    "date": dates,
    "actual_price": actual_prices,
    "forecasted_VaR": forecasted_vars,
    "hit": hit_series
})
print(backtest_results.head())
backtest_results.to_csv("backtest_results.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(backtest_results["date"], backtest_results["actual_price"], label="Actual Price", marker="o")
plt.plot(backtest_results["date"], backtest_results["forecasted_VaR"], label="Forecasted VaR", marker="x")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{forecast_horizon}-Day Ahead VaR Backtest")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 2))
plt.stem(backtest_results["date"], backtest_results["hit"], use_line_collection=True)
plt.xlabel("Date")
plt.ylabel("Hit (1=breach)")
plt.title(f"{forecast_horizon}-Day Ahead VaR Breaches")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()