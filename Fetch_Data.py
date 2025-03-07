import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

crude_oil_stocks = ["XOM", "CVX", "OXY", "BP", "COP", "EOG", "MPC", "VLO", "PSX", "HES"]

def ensure_table_exists(conn, ticker):
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {ticker} (
            date TEXT PRIMARY KEY,
            close REAL
        )
    """)
    conn.commit()

def fetch_stock_data(ticker, start_date="2015-01-01"):
    """
    Fetches historical stock data from a fixed start date and updates the SQLite database.
    1) Ensures we flatten multi-level columns from yfinance.
    2) Properly resets the row index to create a 'date' column.
    3) Identifies and renames the unadjusted 'Close_<TICKER>' column to 'close'.
    4) Inserts only 'date' and 'close' into the SQLite table.
    """
    print(f"🔍 Fetching data for {ticker} from {start_date} to present...")

    # Connect to SQLite and ensure table exists
    conn = sqlite3.connect("stock_data.db")
    ensure_table_exists(conn, ticker)

    # 1) Fetch all columns so we can confirm 'Close_<ticker>' is present
    #    auto_adjust=False -> ensures 'Close' is unadjusted
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=None,
        auto_adjust=False
    )

    if data.empty:
        print(f"⚠️ No data returned for {ticker}. Skipping...")
        conn.close()
        return False

    # Print raw columns for debugging
    print(f"✅ Downloaded columns for {ticker}:", list(data.columns))

    # 2) Flatten multi-index columns -> e.g. ('Close', 'XOM') => 'Close_XOM'
    #    Then make them lowercase for consistency
    data.columns = [
        "_".join([str(c) for c in col]).strip()  # e.g. "Close_XOM"
        for col in data.columns.to_flat_index()
    ]
    data.columns = data.columns.str.lower().str.strip()
    print(f"Flattened columns for {ticker}:", list(data.columns))

    # 3) Reset the row index to create a normal "date" column
    data.reset_index(inplace=True)
    print("Columns after reset_index:", list(data.columns))

    # yfinance usually calls the row index 'Date', but let's rename it robustly:
    # If we find 'Date' or 'index', rename it to 'date'; otherwise rename the first column.
    if "Date".lower() in data.columns:
        data.rename(columns={"date": "date"}, inplace=True)  # might be no-op, but let's keep it consistent
    elif "index" in data.columns:
        data.rename(columns={"index": "date"}, inplace=True)
    elif data.columns[0] != "date":
        # fallback: rename the first column to 'date'
        data.rename(columns={data.columns[0]: "date"}, inplace=True)

    # Double-check that we do have a 'date' column now
    if "date" not in data.columns:
        print(f"❌ Could not properly rename the date column for {ticker}. Columns are:", list(data.columns))
        conn.close()
        return False

    # Convert 'date' to string for SQLite
    data["date"] = data["date"].astype(str)

    # 4) Identify the unadjusted close column: 'close_<ticker>'
    #    If you prefer adjusted close, search for 'adj close_<ticker>' instead
    close_col = None
    for col in data.columns:
        # e.g. "close_xom", "close_cvx", etc.
        if col.startswith("close_"):
            close_col = col
            break
    if not close_col:
        print(f"⚠️ Could not find 'close_<ticker>' in flattened data for {ticker}. Columns:", list(data.columns))
        conn.close()
        return False

    # Rename it to 'close'
    data.rename(columns={close_col: "close"}, inplace=True)

    # Convert 'close' to float
    data["close"] = data["close"].astype(float)

    # 5) Remove duplicates by comparing dates in DB
    existing_dates = pd.read_sql(f'SELECT date FROM "{ticker}"', conn)["date"].tolist()
    new_data = data[~data["date"].isin(existing_dates)]

    if not new_data.empty:
        # Insert only 'date' and 'close' into the table
        to_insert = new_data[["date", "close"]].copy()
        to_insert.to_sql(ticker, conn, if_exists="append", index=False)
        print(f"✅ {ticker} updated with {len(to_insert)} new rows.")
    else:
        print(f"⚠️ No new rows to add for {ticker}.")

    # 6) Print final row count
    final_count = pd.read_sql(f'SELECT COUNT(*) AS cnt FROM "{ticker}"', conn)["cnt"].iloc[0]
    print(f"📈 {ticker} now has {final_count} total rows in the database.")

    conn.close()
    return True

if __name__ == "__main__":
    print("📥 Fetching extended data for all crude oil stocks...")
    for stock in crude_oil_stocks:
        fetch_stock_data(stock, start_date="2020-01-01")  # Adjust as needed
    print("✅ All data is now up to date in SQLite.")