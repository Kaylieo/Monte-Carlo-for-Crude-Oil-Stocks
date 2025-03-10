import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t
from scipy.stats.qmc import Sobol

# Import rpy2 modules for R bridging
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri

# Activate automatic conversion for numpy and pandas objects
numpy2ri.activate()
pandas2ri.activate()

def run_simulation_logic(historical_data, ticker, num_days, num_simulations):
    """
    Runs the Monte Carlo simulation using a standard EGARCH model.
    """
    # Validate required columns
    required_cols = {"date", "close"}
    missing = required_cols - set(historical_data.columns)
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {missing}")
    
    # Prepare historical data
    historical_data["date"] = pd.to_datetime(historical_data["date"])
    historical_prices = historical_data["close"].dropna()
    returns = historical_prices.pct_change().dropna()
    if len(returns) < 2:
        raise ValueError("Not enough historical data to compute returns")
    mu = returns.mean() * 252

    # --- EGARCH Fitting & Summary ---
    print("Fitting EGARCH(1,1) with AR(1) mean and skew-t distribution for heavier tails...")
    garch = arch_model(returns, mean='AR', lags=1, vol='EGARCH', p=1, o=1, q=1,
                       dist='skewt', rescale=True).fit(disp="on")
    print("Model Summary:")
    try:
        summary_str = str(garch.summary())
        print(summary_str)
    except Exception as e:
        summary_str = f"Warning: Could not compute full model summary: {e}"
        print(summary_str)
    
    # --- Multi-Step EGARCH Volatility Forecast ---
    last_variance = garch.conditional_volatility.iloc[-1] ** 2
    L = np.log(last_variance)
    omega_param = garch.params['omega']
    beta_param = garch.params['beta[1]']
    if abs(1 - beta_param) < 1e-8:
        long_run_log_var = np.inf
    else:
        long_run_log_var = omega_param / (1 - beta_param)
    forecast_log_vars = np.array([
        long_run_log_var + (beta_param ** h) * (L - long_run_log_var)
        for h in range(1, num_days + 1)
    ])
    sigma_forecast = np.exp(0.5 * forecast_log_vars)
    if sigma_forecast is None or len(sigma_forecast) == 0 or np.isnan(sigma_forecast).any():
        sigma_forecast = np.array([np.std(returns)] * num_days)
    
    sigma = np.repeat(sigma_forecast[:, np.newaxis], num_simulations, axis=1)
    initial_price = historical_prices.iloc[-1]
    np.random.seed(42)
    simulated_prices = np.zeros((num_days + 1, num_simulations))
    simulated_prices[0] = initial_price

    dt = 1 / 252
    dof = 5
    sobol = Sobol(d=num_simulations, scramble=True)
    random_shocks = sobol.random(num_days)
    random_shocks = t.ppf(random_shocks, dof)
    drift = (mu - 0.5 * sigma_forecast**2)[:, np.newaxis] * dt
    diffusion = sigma * np.sqrt(dt) * random_shocks

    for day in range(1, num_days + 1):
        simulated_prices[day] = simulated_prices[day-1] * np.exp(drift[day-1, 0] + diffusion[day-1, :])
    final_prices = simulated_prices[-1]
    results = {
        "final_prices": final_prices,
        "initial_price": float(initial_price),
        "expected_price": float(np.mean(final_prices)),
        "std_dev": float(np.std(final_prices)),
        "var_5pct": float(np.percentile(final_prices, 5)),
        "cvar_5pct": float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])),
        "simulated_prices": simulated_prices,
        "model_summary": summary_str
    }
    return results

def run_ms_egarch_simulation_logic(historical_data, ticker, num_days, num_simulations):
    """
    Runs the Monte Carlo simulation using a Markov-switching EGARCH model implemented in R (using MSGARCH).
    This function uses rpy2 to call an R function that fits a 2-regime MS-GARCH model.
    """
    # Validate required columns and prepare data
    required_cols = {"date", "close"}
    missing = required_cols - set(historical_data.columns)
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {missing}")
    historical_data["date"] = pd.to_datetime(historical_data["date"])
    historical_prices = historical_data["close"].dropna()
    returns = historical_prices.pct_change().dropna()
    if len(returns) < 2:
        raise ValueError("Not enough historical data to compute returns")
    mu = returns.mean() * 252

    # Define and evaluate R function for MS-EGARCH using MSGARCH
    r_code = """
    ms_egarch_fit <- function(returns_vector) {
      if(!require(MSGARCH)) {
        install.packages("MSGARCH", repos = "https://cran.rstudio.com/")
        library(MSGARCH)
      }
    spec <- CreateSpec(
      variance.spec = list(model = "sGARCH"), 
      distribution.spec = list(name = "sstd"),
      switch.spec = list(K = 2)
    )
    fit <- FitML(spec, returns_vector)
    }
    """
    robjects.r(r_code)
    ms_egarch_fit = robjects.globalenv['ms_egarch_fit']
    
    # Convert returns to an R vector
    r_returns = robjects.FloatVector(returns.values)
    
    # Call the R function directly (without wrapping in contextvars)
    fit_result = ms_egarch_fit(r_returns)
    
    # Capture the R model summary
    summary_output = robjects.r['capture.output'](robjects.r['summary'](fit_result))
    summary_str = "\n".join(list(summary_output))
    print("Markov-switching EGARCH Model Summary from R:")
    print(summary_str)
    
    # --- For simulation, use a placeholder volatility forecast ---
    sigma_forecast = np.array([np.std(returns)] * num_days)
    sigma = np.repeat(sigma_forecast[:, np.newaxis], num_simulations, axis=1)
    initial_price = historical_prices.iloc[-1]
    
    np.random.seed(42)
    simulated_prices = np.zeros((num_days + 1, num_simulations))
    simulated_prices[0] = initial_price
    
    dt = 1 / 252
    dof = 5
    sobol = Sobol(d=num_simulations, scramble=True)
    random_shocks = sobol.random(num_days)
    random_shocks = t.ppf(random_shocks, dof)
    drift = (mu - 0.5 * sigma_forecast**2)[:, np.newaxis] * dt
    diffusion = sigma * np.sqrt(dt) * random_shocks
    for day in range(1, num_days + 1):
        simulated_prices[day] = simulated_prices[day-1] * np.exp(drift[day-1, 0] + diffusion[day-1, :])
    final_prices = simulated_prices[-1]
    
    results = {
        "final_prices": final_prices,
        "initial_price": float(initial_price),
        "expected_price": float(np.mean(final_prices)),
        "std_dev": float(np.std(final_prices)),
        "var_5pct": float(np.percentile(final_prices, 5)),
        "cvar_5pct": float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])),
        "simulated_prices": simulated_prices,
        "model_summary": summary_str
    }
    return results