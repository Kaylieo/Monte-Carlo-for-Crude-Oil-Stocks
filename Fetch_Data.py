"""Module for fetching and caching stock data.

This module fetches historical price data for specified crude oil stocks using yfinance and stores it in an SQLite database. It provides functions to ensure database tables exist and manage data retrieval.
"""

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# List of crude oil stocks
CRUDE_OIL_STOCKS = ["XOM", "CVX", "OXY", "BP", "COP", "EOG", "MPC", "VLO", "PSX", "HES"]

def ensure_table_exists(conn, ticker):
    """
    Ensure that a table for the given ticker exists in the SQLite database.
    The table contains 'date' (as a TEXT PRIMARY KEY) and 'close' (as REAL).
    """
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {ticker} (
            date TEXT PRIMARY KEY,
            close REAL
        )
    """
    conn.execute(create_table_query)
    conn.commit()

def fetch_stock_data(ticker, start_date="2000-01-01"):
    """
    Fetch historical stock data from Yahoo Finance for a given ticker, process the data,
    and update the SQLite database.

    Steps:
    1. Download data using yfinance (with auto_adjust=False to get unadjusted prices).
    2. Flatten multi-index columns and convert them to lower-case.
    3. Reset the index to create a 'date' column.
    4. Identify the unadjusted close column (e.g., 'close_xom') and rename it to 'close'.
    5. Remove rows that already exist in the database and insert new data.

    Returns:
        bool: True if data was successfully updated; False otherwise.
    """
    print(f"🔍 Fetching data for {ticker} from {start_date} to present...")

    with sqlite3.connect("stock_data.db") as conn:
        ensure_table_exists(conn, ticker)

        # Download stock data
        data = yf.download(ticker, start=start_date, auto_adjust=False)
        if data.empty:
            print(f"⚠️ No data returned for {ticker}. Skipping...")
            return False

        print(f"✅ Downloaded columns for {ticker}: {list(data.columns)}")

        # Flatten multi-index columns and convert to lower-case
        data.columns = ['_'.join(map(str, col)).strip() for col in data.columns.to_flat_index()]
        data.columns = [col.lower().strip() for col in data.columns]
        print(f"Flattened columns for {ticker}: {list(data.columns)}")

        # Reset index to create a 'date' column
        data.reset_index(inplace=True)
        print("Columns after reset_index:", list(data.columns))

        # Ensure there's a 'date' column; if not, rename the first column to 'date'
        if "date" not in data.columns:
            data.rename(columns={data.columns[0]: "date"}, inplace=True)
        if "date" not in data.columns:
            print(f"❌ Could not properly set the 'date' column for {ticker}. Columns: {list(data.columns)}")
            return False

        # Convert 'date' to string for SQLite
        data["date"] = data["date"].astype(str)

        # Identify the unadjusted close column (e.g., 'close_xom')
        close_col = next((col for col in data.columns if col.startswith("close_")), None)
        if not close_col:
            print(f"⚠️ Could not find the unadjusted close column for {ticker}. Columns: {list(data.columns)}")
            return False

        # Rename the identified column to 'close' and convert it to float
        data.rename(columns={close_col: "close"}, inplace=True)
        data["close"] = data["close"].astype(float)

        # Remove duplicates by checking for existing dates in the database
        existing_dates = pd.read_sql(f'SELECT date FROM "{ticker}"', conn)["date"].tolist()
        new_data = data[~data["date"].isin(existing_dates)]

        if not new_data.empty:
            new_data[["date", "close"]].to_sql(ticker, conn, if_exists="append", index=False)
            print(f"✅ {ticker} updated with {len(new_data)} new rows.")
        else:
            print(f"⚠️ No new rows to add for {ticker}.")

        # Display final row count
        final_count = pd.read_sql(f'SELECT COUNT(*) AS cnt FROM "{ticker}"', conn)["cnt"].iloc[0]
        print(f"📈 {ticker} now has {final_count} total rows in the database.")

    return True

if __name__ == "__main__":
    print("📥 Fetching extended data for all crude oil stocks...")
    for stock in CRUDE_OIL_STOCKS:
        fetch_stock_data(stock, start_date="2000-01-01")
    print("✅ All data is now up to date in SQLite.")