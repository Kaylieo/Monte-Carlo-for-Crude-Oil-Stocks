# test_monte_carlo.py
import pytest
import pandas as pd
from unittest.mock import patch
import numpy as np

# Import your function from Monte_Carlo_UI.py
# e.g.:
# from Monte_Carlo_UI import run_simulation_logic

def make_dummy_data(rows=5, include_close=True, include_date=True):
    """
    Create a simple DataFrame for testing.
    By default includes 'date' and 'close' columns with dummy values.
    """
    columns = {}
    if include_date:
        # Use a date range for the 'date' column
        columns["date"] = pd.date_range("2022-01-01", periods=rows)
    if include_close:
        columns["close"] = np.linspace(100, 110, rows)  # dummy ascending prices
    # Add any other columns if needed
    return pd.DataFrame(columns)

def test_run_simulation_logic_missing_close():
    """
    If 'close' column is missing, we expect an error/stop.
    """
    # Create data without 'close'
    df = make_dummy_data(include_close=False)
    with pytest.raises(Exception) as excinfo:
        # If you're using st.stop(), it raises a Streamlit StopException,
        # which is a subclass of BaseException. So you might catch BaseException or StopException.
        # Or if you replaced st.stop() with raise SystemExit, you'd do pytest.raises(SystemExit).
        from monte_carlo import run_simulation_logic
        run_simulation_logic(df, "XOM", 30, 1000)
    # Optionally check the error message
    # assert "missing for XOM" in str(excinfo.value)

def test_run_simulation_logic_missing_date():
    """
    If 'date' column is missing, we expect an error/stop.
    """
    df = make_dummy_data(include_date=False)
    with pytest.raises(Exception):
        from monte_carlo import run_simulation_logic
        run_simulation_logic(df, "XOM", 30, 1000)

def test_run_simulation_logic_empty_data():
    """
    If DataFrame is empty, we expect an error/stop.
    """
    df = pd.DataFrame(columns=["date", "close"])  # columns exist but no rows
    with pytest.raises(Exception):
        from monte_carlo import run_simulation_logic
        run_simulation_logic(df, "XOM", 30, 1000)

def test_run_simulation_logic_insufficient_data():
    """
    If there's only 1 row, we can't compute returns => error/stop.
    """
    df = make_dummy_data(rows=1)  # only 1 row => no returns
    with pytest.raises(Exception):
        from monte_carlo import run_simulation_logic
        run_simulation_logic(df, "XOM", 30, 1000)

def test_run_simulation_logic_success():
    """
    Normal case: multiple rows, date & close present => should succeed
    """
    df = make_dummy_data(rows=200)
    from monte_carlo import run_simulation_logic
    result = run_simulation_logic(df, "XOM", 30, 1000)
    assert "final_prices" in result
    assert len(result["final_prices"]) == 1000
    assert "expected_price" in result
    # etc. for additional checks