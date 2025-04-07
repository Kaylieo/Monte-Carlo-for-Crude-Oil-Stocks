import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from monte_carlo import run_ms_egarch_simulation_logic


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

    # Load historical data (replace this with your actual historical data retrieval logic)
    np.random.seed(0)
    dates = pd.date_range(start="2024-12-01", periods=100)
    close_prices = np.cumsum(np.random.normal(0, 1, 100)) + 100
    historical_data = pd.DataFrame({"date": dates, "close": close_prices})

    # Split data into training and testing sets
    training_data = historical_data.iloc[:-num_days]
    testing_data = historical_data.iloc[-num_days:]

    # Run your simulation logic
    simulation_results = run_ms_egarch_simulation_logic(training_data, ticker, num_days, num_simulations)
    simulated_prices = simulation_results["simulated_prices"].mean(axis=1)[1:]  # Mean price across simulations for each day

    # Actual prices for testing period
    real_prices = testing_data["close"].values

    # Calculate accuracy
    accuracy = calculate_mape(simulated_prices, real_prices)
    print(f"Simulation Accuracy: {accuracy:.2f}%")
