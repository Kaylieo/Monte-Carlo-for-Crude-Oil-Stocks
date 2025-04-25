"""Pytest suite for Monte Carlo simulation.

This module contains unit tests for validating data preparation and simulation logic in monte_carlo.py, including error cases and integration with the MS-EGARCH model.
"""

import pytest
import pandas as pd
import numpy as np
from monte_carlo import _validate_and_prepare_data, run_ms_egarch_simulation_logic

def test_validate_and_prepare_data_valid():
    # Create a simple valid DataFrame with the required columns.
    data = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "close": [100, 102, 101]
    }
    df = pd.DataFrame(data)
    
    historical_prices, returns = _validate_and_prepare_data(df, "TEST")
    
    # Check that historical_prices and returns are Pandas Series
    assert isinstance(historical_prices, pd.Series)
    assert isinstance(returns, pd.Series)
    
    # historical_prices should have length 3, returns should have length 2 (since returns = pct_change dropna)
    assert len(historical_prices) == 3
    assert len(returns) == 2
    
    # Check that the first return is approximately (102/100 - 1) = 0.02
    np.testing.assert_almost_equal(returns.iloc[0], 0.02, decimal=2)

def test_validate_and_prepare_data_missing_column():
    # Create a DataFrame missing the 'close' column.
    data = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "open": [100, 102, 101]
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        _validate_and_prepare_data(df, "TEST")

def test_validate_and_prepare_data_insufficient_data():
    # Create a DataFrame with only one row (insufficient to compute returns).
    data = {
        "date": ["2020-01-01"],
        "close": [100]
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="Not enough historical data"):
        _validate_and_prepare_data(df, "TEST")

# Mark this test as an integration test since it relies on R and the MSGARCH package.
# @pytest.mark.skip(reason="Integration test: requires R and MSGARCH installed")
def test_run_ms_egarch_simulation_logic_single_day():
    # Create a DataFrame with valid historical data.
    data = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        "close": [100, 102, 101, 103]
    }
    df = pd.DataFrame(data)
    
    # Run the simulation for a single day with a small number of simulations.
    result = run_ms_egarch_simulation_logic(df, "TEST", num_days=1, num_simulations=100)
    
    # Expected keys in the result dictionary.
    expected_keys = {
        "final_prices", "initial_price", "expected_price",
        "std_dev", "var_5pct", "cvar_5pct", "simulated_prices", "model_summary"
    }
    assert expected_keys.issubset(result.keys())
    
    # Check that final_prices is a NumPy array of the correct length.
    assert isinstance(result["final_prices"], np.ndarray)
    assert result["final_prices"].shape[0] == 100