# -------------------------
# IMPORTS
# -------------------------
# Core data science
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t
from scipy.stats.qmc import Sobol

# R integration
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter

# Type hints
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class SimulationResults:
    """Container for Monte Carlo simulation results."""
    final_prices: np.ndarray
    initial_price: float
    expected_price: float
    std_dev: float
    var_5pct: float
    cvar_5pct: float
    simulated_prices: np.ndarray
    model_summary: str

class MonteCarloSimulator:
    """A class for running Monte Carlo simulations using Markov-switching EGARCH models."""
    
    def __init__(self, ticker: str):
        """Initialize the simulator.
        
        Args:
            ticker: The stock ticker symbol.
        """
        self.ticker = ticker
        self.historical_prices: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        self.fit_result = None
        self.hist_daily_vol: Optional[float] = None
        self._setup_r_environment()

    def _setup_r_environment(self) -> None:
        """Set up the R environment and define the MS-EGARCH fitting function."""
        r_code = """
ms_egarch_fit <- function(returns_vector) {
  if(!require(MSGARCH)) {
    install.packages("MSGARCH", repos="https://cran.rstudio.com/")
    library(MSGARCH)
  }
  spec <- CreateSpec(
    variance.spec = list(model="sGARCH"),
    distribution.spec = list(name="sstd", fixed.parameters=list(nu=c(1e6,2))),
    switch.spec = list(K=2, start.values=list(P=matrix(c(0.98,0.02,0.20,0.80),2,2)))
  )
  fit <- FitML(spec, returns_vector)
  return(fit)
}"""
        robjects.r(r_code)
        self.ms_egarch_fit = robjects.globalenv['ms_egarch_fit']

    def _validate_and_prepare_data(self, historical_data: pd.DataFrame) -> None:
        """Validate and prepare the input data.
        
        Args:
            historical_data: DataFrame containing 'date' and 'close' columns.
            
        Raises:
            ValueError: If required columns are missing or insufficient data is found.
        """
        required_cols = {"date", "close"}
        missing = required_cols - set(historical_data.columns)
        if missing:
            raise ValueError(f"Missing required columns for {self.ticker}: {missing}")
        
        historical_data["date"] = pd.to_datetime(historical_data["date"])
        self.historical_prices = historical_data["close"].dropna()
        self.returns = self.historical_prices.pct_change().dropna()
        
        if len(self.returns) < 2:
            raise ValueError("Not enough historical data to compute returns.")
        
        self.hist_daily_vol = np.std(self.returns)

    def fit_model(self, historical_data: pd.DataFrame) -> str:
        """Fit the MS-EGARCH model to the historical data.
        
        Args:
            historical_data: DataFrame containing 'date' and 'close' columns.
            
        Returns:
            str: Model summary.
        """
        self._validate_and_prepare_data(historical_data)
        
        with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
            r_returns = robjects.FloatVector(self.returns.values)
            self.fit_result = self.ms_egarch_fit(r_returns)
            
        summary_output = robjects.r['capture.output'](robjects.r['summary'](self.fit_result))
        summary_str = "\n".join(list(summary_output))
        print(f"Markov-switching EGARCH (Hybrid) Model Summary for {self.ticker}:")
        print(summary_str)
        return summary_str

    def run_simulation(self,
                      num_days: int,
                      num_simulations: int) -> SimulationResults:
        """Run the Monte Carlo simulation for the specified number of days and simulations.
        
        Args:
            num_days: Number of days to simulate forward.
            num_simulations: Number of simulation paths to generate.
            
        Returns:
            SimulationResults object containing simulation statistics and paths.
            
        Raises:
            ValueError: If model hasn't been fit or parameters are invalid.
        """
        if self.fit_result is None:
            raise ValueError("Model must be fit before running simulations")
            
        if num_days < 1:
            raise ValueError("num_days must be at least 1")
        if num_simulations < 1:
            raise ValueError("num_simulations must be at least 1")

        # Run single-day or multi-day simulation based on num_days
        if num_days == 1:
            return self._run_single_day_simulation(num_simulations)
        else:
            return self._run_multi_day_simulation(num_days, num_simulations)
            
    def _run_single_day_simulation(self, num_simulations: int) -> SimulationResults:
        """Run a single-day Monte Carlo simulation.
        
        Args:
            num_simulations: Number of simulation paths.
            
        Returns:
            SimulationResults for a single day simulation.
        """
        Simulate = robjects.r("MSGARCH:::simulate.MSGARCH_SPEC")
        sim_forecast = Simulate(self.fit_result, **{'n.ahead': 1, 'nsim': num_simulations})
        sim_forecast_list = list(sim_forecast)
        
        # Extract simulation results
        returns_array = np.array(sim_forecast_list[0])  # shape (1, nsim)
        initial_price = self.historical_prices.iloc[-1]
        final_prices = initial_price * (1 + returns_array[0])
        
        # Calculate statistics
        return SimulationResults(
            final_prices=final_prices,
            initial_price=float(initial_price),
            expected_price=float(np.mean(final_prices)),
            std_dev=float(np.std(final_prices)),
            var_5pct=float(np.percentile(final_prices, 5)),
            cvar_5pct=float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])),
            simulated_prices=np.vstack([np.full(num_simulations, initial_price), final_prices]),
            model_summary=""
        )
        
    def _run_multi_day_simulation(self, num_days: int, num_simulations: int) -> SimulationResults:
        """Run a multi-day Monte Carlo simulation.
        
        Args:
            num_days: Number of days to simulate.
            num_simulations: Number of simulation paths.
            
        Returns:
            SimulationResults for the multi-day simulation.
        """
        initial_price = self.historical_prices.iloc[-1]
        simulated_prices = np.zeros((num_days + 1, num_simulations))
        simulated_prices[0] = initial_price

        dt = 1 / 252  # Trading days in a year
        dof = 5       # Degrees of freedom for t-distribution
        np.random.seed(42)

        for day in range(1, num_days + 1):
            last_price_array = simulated_prices[day - 1]
            new_prices = self._forecast_next_day(last_price_array, num_simulations, dt, dof)
            simulated_prices[day] = new_prices

        final_prices = simulated_prices[-1]
        return SimulationResults(
            final_prices=final_prices,
            initial_price=float(initial_price),
            expected_price=float(np.mean(final_prices)),
            std_dev=float(np.std(final_prices)),
            var_5pct=float(np.percentile(final_prices, 5)),
            cvar_5pct=float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])),
            simulated_prices=simulated_prices,
            model_summary=""
        )

    def _forecast_next_day(self, last_price_array: np.ndarray, num_simulations: int,
                          dt: float, dof: int) -> np.ndarray:
        """Generate price forecasts for the next day.
        
        Args:
            last_price_array: Array of prices from the previous day.
            num_simulations: Number of simulation paths.
            dt: Time step (1/252 for daily).
            dof: Degrees of freedom for t-distribution.
            
        Returns:
            Array of forecasted prices for the next day.
        """
        Simulate = robjects.r("MSGARCH:::simulate.MSGARCH_SPEC")
        sim_forecast = Simulate(self.fit_result, **{'n.ahead': 1, 'nsim': num_simulations})
        sim_forecast_list = list(sim_forecast)
        
        # Calculate volatility forecast
        vol_array = np.array(sim_forecast_list[2])  # shape (1, nsim, 2)
        vol_avg = np.mean(vol_array, axis=2)        # shape (1, nsim)
        sigma_forecast = np.mean(vol_avg, axis=1)     # shape (1,)
        
        # Scale volatility
        sim_daily_vol = sigma_forecast * np.sqrt(1/252)
        scaling_factor = self.hist_daily_vol / sim_daily_vol
        sigma_forecast = sigma_forecast * scaling_factor
        sigma = np.repeat(sigma_forecast[:, np.newaxis], num_simulations, axis=1)
        
        # Generate random shocks using Sobol sequence
        sobol_engine = Sobol(d=num_simulations, scramble=True)
        random_shocks_uniform = sobol_engine.random(1)
        random_shocks = t.ppf(random_shocks_uniform, dof)
        
        # Calculate price movements
        drift = (-0.5 * sigma_forecast**2)[:, np.newaxis] * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks
        
        # Generate new prices
        return last_price_array * np.exp(drift[0, 0] + diffusion[0])
