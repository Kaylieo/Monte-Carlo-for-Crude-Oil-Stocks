import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from monte_carlo import MonteCarloSimulator


def calculate_mape(simulated_prices: np.ndarray, real_prices: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between simulated prices and actual prices.
    
    :param simulated_prices: Array of simulated prices from the model.
    :param real_prices: Array of actual historical prices for comparison.
    :return: MAPE value as a percentage.
    """
    mape = mean_absolute_percentage_error(real_prices, simulated_prices) * 100
    accuracy = 100 - mape
    return accuracy


if __name__ == "__main__":
    # Example usage
    ticker = "XOM"
    num_days = 30
    num_simulations = 1000

    # Create sample historical data
    np.random.seed(0)
    dates = pd.date_range(start="2024-12-01", periods=100)
    close_prices = pd.Series(np.cumsum(np.random.normal(0, 1, 100)) + 100, index=dates)
    historical_data = pd.DataFrame({
        'date': dates,
        'close': close_prices
    })
    
    # Initialize and run simulator
    simulator = MonteCarloSimulator(ticker)
    simulator.fit_model(historical_data)
    results = simulator.run_simulation(num_days, num_simulations)
    
    # Calculate accuracy using the mean path
    mean_path = results.simulated_prices.mean(axis=1)
    actual_prices = close_prices[-num_days-1:].values
    accuracy = calculate_mape(mean_path, actual_prices)
    
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"Expected Price after {num_days} days: ${results.expected_price:.2f}")
    print(f"95% VaR: ${results.var_5pct:.2f}")
    print(f"95% CVaR: ${results.cvar_5pct:.2f}")
