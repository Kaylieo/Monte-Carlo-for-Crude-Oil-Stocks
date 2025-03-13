import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t
from scipy.stats.qmc import Sobol

# rpy2 modules for R bridging
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri

# Activate automatic conversion for numpy and pandas objects
numpy2ri.activate()
pandas2ri.activate()

from typing import Tuple

def _validate_and_prepare_data(historical_data: pd.DataFrame, ticker: str) -> Tuple[pd.Series, pd.Series]:
    """
    Helper function to ensure the DataFrame has the required columns
    and prepare the 'date' and 'close' series for analysis.
    
    :param historical_data: DataFrame containing 'date' and 'close' columns.
    :param ticker: Ticker symbol for error reporting.
    :return: (historical_prices, returns) where:
             historical_prices = Series of close prices
             returns = Series of daily returns (pct_change) after dropping NaNs
    :raises ValueError: If required columns are missing or insufficient data is found.
    """
    required_cols = {"date", "close"}
    missing = required_cols - set(historical_data.columns)
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {missing}")
    
    historical_data["date"] = pd.to_datetime(historical_data["date"])
    historical_prices = historical_data["close"].dropna()
    returns = historical_prices.pct_change().dropna()
    
    if len(returns) < 2:
        raise ValueError("Not enough historical data to compute returns.")
    
    return historical_prices, returns

def run_ms_egarch_simulation_logic(historical_data: pd.DataFrame,
                                   ticker: str,
                                   num_days: int,
                                   num_simulations: int) -> dict:
    """
    If num_days == 1:
      Runs the original single-day MSGARCH code unchanged.
    If num_days > 1:
      Iteratively performs single-day simulations day by day,
      chaining them to produce a multi-day path, without re-fitting.
    """

    import rpy2.robjects as robjects
    import numpy as np

    # Validate data
    historical_prices, returns = _validate_and_prepare_data(historical_data, ticker)

    # If user only wants 1 day, run your original code exactly:
    if num_days == 1:
        # ------------------- Original Single-Day Code -------------------
        # (unchanged, exactly as in your snippet)
        mu = 0
        r_code = """
        ms_egarch_fit <- function(returns_vector) {
          if(!require(MSGARCH)) {
            install.packages("MSGARCH", repos = "https://cran.rstudio.com/")
            library(MSGARCH)
          }
          # Hybrid approach:
          # Regime 1: nearly normal tails; Regime 2: very heavy tails (nu = 2).
          # Transition matrix: From regime 1: 98% stay, 2% switch;
                               From regime 2: 20% switch to normal, 80% stay.
          spec <- CreateSpec(
            variance.spec = list(model = "sGARCH"),
            distribution.spec = list(
              name = "sstd",
              fixed.parameters = list(nu = c(1e6, 2))
            ),
            switch.spec = list(
              K = 2,
              start.values = list(P = matrix(c(0.98, 0.02, 0.20, 0.80), 2, 2))
            )
          )
          fit <- FitML(spec, returns_vector)
          return(fit)
        }
        """
        robjects.r(r_code)
        ms_egarch_fit = robjects.globalenv['ms_egarch_fit']

        r_returns = robjects.FloatVector(returns.values)
        fit_result = ms_egarch_fit(r_returns)
        summary_output = robjects.r['capture.output'](robjects.r['summary'](fit_result))
        summary_str = "\n".join(list(summary_output))
        print(f"Markov-switching EGARCH (Hybrid) Model Summary for {ticker}:")
        print(summary_str)

        Simulate = robjects.r("MSGARCH:::simulate.MSGARCH_SPEC")
        sim_forecast = Simulate(fit_result, **{'n.ahead': 1, 'nsim': num_simulations})
        sim_forecast_list = list(sim_forecast)
        vol_array = np.array(sim_forecast_list[2])  # shape (1, nsim, 2)
        vol_avg = np.mean(vol_array, axis=2)        # shape (1, nsim)
        sigma_forecast = np.mean(vol_avg, axis=1)   # shape (1,)

        hist_daily_vol = np.std(returns)
        sim_daily_vol = sigma_forecast * np.sqrt(1/252)
        scaling_factor = hist_daily_vol / sim_daily_vol
        sigma_forecast = sigma_forecast * scaling_factor

        sigma = np.repeat(sigma_forecast[:, np.newaxis], num_simulations, axis=1)
        initial_price = historical_prices.iloc[-1]

        import numpy as np
        np.random.seed(42)
        simulated_prices = np.zeros((2, num_simulations))  # day 0 + day 1
        simulated_prices[0] = initial_price

        dt = 1 / 252
        dof = 5
        from scipy.stats import t
        from scipy.stats.qmc import Sobol
        sobol_engine = Sobol(d=num_simulations, scramble=True)
        random_shocks_uniform = sobol_engine.random(1)
        random_shocks = t.ppf(random_shocks_uniform, dof)

        drift = (-0.5 * sigma_forecast**2)[:, np.newaxis] * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks
        simulated_prices[1] = simulated_prices[0] * np.exp(drift[0, 0] + diffusion[0])

        final_prices = simulated_prices[-1]
        return {
            "final_prices": final_prices,
            "initial_price": float(initial_price),
            "expected_price": float(np.mean(final_prices)),
            "std_dev": float(np.std(final_prices)),
            "var_5pct": float(np.percentile(final_prices, 5)),
            "cvar_5pct": float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])),
            "simulated_prices": simulated_prices,
            "model_summary": summary_str
        }

    else:
        # ------------------- Iterative Multi-Day Approach -------------------
        # We'll chain single-day forecasts day by day, reusing the same fit_result
        # (so we do NOT re-fit each day). This minimal approach ensures we never
        # request n.ahead > 1 from MSGARCH, thus avoiding "incorrect number of forecast days."
        import numpy as np
        import pandas as pd
        from scipy.stats import t
        from scipy.stats.qmc import Sobol

        # Fit the model once
        mu = 0
        r_code = """
        ms_egarch_fit <- function(returns_vector) {
          if(!require(MSGARCH)) {
            install.packages("MSGARCH", repos = "https://cran.rstudio.com/")
            library(MSGARCH)
          }
          # Hybrid approach:
          spec <- CreateSpec(
            variance.spec = list(model = "sGARCH"),
            distribution.spec = list(
              name = "sstd",
              fixed.parameters = list(nu = c(1e6, 2))
            ),
            switch.spec = list(
              K = 2,
              start.values = list(P = matrix(c(0.98, 0.02, 0.20, 0.80), 2, 2))
            )
          )
          fit <- FitML(spec, returns_vector)
          return(fit)
        }
        """
        robjects.r(r_code)
        ms_egarch_fit = robjects.globalenv['ms_egarch_fit']

        r_returns = robjects.FloatVector(returns.values)
        fit_result = ms_egarch_fit(r_returns)
        summary_output = robjects.r['capture.output'](robjects.r['summary'](fit_result))
        summary_str = "\n".join(list(summary_output))
        print(f"Markov-switching EGARCH (Hybrid) Model Summary for {ticker}:")
        print(summary_str)

        # We'll store the entire path in simulated_prices: shape (num_days+1, num_simulations)
        initial_price = historical_prices.iloc[-1]
        simulated_prices = np.zeros((num_days + 1, num_simulations))
        simulated_prices[0] = initial_price

        # We'll do single-day simulation each iteration, from the same fit_result
        # but each day starts from the last simulated price.
        # We'll keep the same scale factor for the entire horizon, using the original returns.
        hist_daily_vol = np.std(returns)

        np.random.seed(42)
        dt = 1 / 252
        dof = 5

        # We'll define a small function that runs the single-day forecast from the existing fit_result
        def single_day_forecast(last_price_array: np.ndarray):
            # single-day forecast with 'MSGARCH:::simulate.MSGARCH_SPEC'
            Simulate = robjects.r("MSGARCH:::simulate.MSGARCH_SPEC")
            sim_forecast = Simulate(fit_result, **{'n.ahead': 1, 'nsim': num_simulations})
            sim_forecast_list = list(sim_forecast)

            vol_array = np.array(sim_forecast_list[2])  # shape (1, nsim, 2)
            vol_avg = np.mean(vol_array, axis=2)        # shape (1, nsim)
            sigma_forecast = np.mean(vol_avg, axis=1)   # shape (1,)

            # scale
            sim_daily_vol = sigma_forecast * np.sqrt(1/252)
            scaling_factor = hist_daily_vol / sim_daily_vol
            sigma_forecast = sigma_forecast * scaling_factor  # shape (1,)

            sigma = np.repeat(sigma_forecast[:, np.newaxis], num_simulations, axis=1)

            # Generate random shocks
            sobol_engine = Sobol(d=num_simulations, scramble=True)
            random_shocks_uniform = sobol_engine.random(1)
            random_shocks = t.ppf(random_shocks_uniform, dof)

            drift = (-0.5 * sigma_forecast**2)[:, np.newaxis] * dt
            diffusion = sigma * np.sqrt(dt) * random_shocks

            # compute new prices for each simulation path
            new_prices = last_price_array * np.exp(drift[0, 0] + diffusion[0])
            return new_prices

        # Iteratively forecast each day
        for day in range(1, num_days + 1):
            last_price_array = simulated_prices[day - 1]
            new_prices = single_day_forecast(last_price_array)
            simulated_prices[day] = new_prices

        final_prices = simulated_prices[-1]

        return {
            "final_prices": final_prices,
            "initial_price": float(initial_price),
            "expected_price": float(np.mean(final_prices)),
            "std_dev": float(np.std(final_prices)),
            "var_5pct": float(np.percentile(final_prices, 5)),
            "cvar_5pct": float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])),
            "simulated_prices": simulated_prices,
            "model_summary": summary_str
        }