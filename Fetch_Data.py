import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

crude_oil_stocks = ["XOM", "CVX", "OXY", "BP", "COP", "EOG", "MPC", "VLO", "PSX", "HES"]

def fetch_stock_data(ticker, period="6mo"):
    """Fetches historical stock data and updates SQLite only if needed."""
    print(f"🔍 Checking latest data for {ticker}...")

    conn = sqlite3.connect("stock_data.db")
    cursor = conn.cursor()

    # ✅ Check the latest stored date
    cursor.execute(f"SELECT MAX(date) FROM {ticker}")
    last_date = cursor.fetchone()[0]

    # If table doesn't exist or data is too old, fetch new data
    needs_update = False
    if last_date is None:
        print(f"⚠️ No existing data for {ticker}. Fetching full data...")
        needs_update = True
    else:
        last_date = datetime.strptime(last_date, "%Y-%m-%d")
        if last_date < datetime.today() - timedelta(days=1):
            print(f"🔄 Updating {ticker} with latest prices...")
            needs_update = True
        else:
            print(f"✅ {ticker} data is already up to date!")

    if not needs_update:
        conn.close()
        return False  # No update needed

    try:
        # ✅ Fetch latest stock data
        data = yf.download(ticker, period=period)[["Close"]]

        if data.empty:
            print(f"⚠️ No new data found for {ticker}. Skipping...")
            return False

        # ✅ Ensure "Date" is properly stored as a column
        data.reset_index(inplace=True)
        data.rename(columns={"Date": "date", "Close": "close"}, inplace=True)

        # ✅ Convert Date column to string format for SQLite compatibility
        data["date"] = data["date"].astype(str)

        # ✅ If table exists, append only new data instead of replacing
        if last_date is not None:
            existing_data = pd.read_sql(f'SELECT * FROM "{ticker}"', conn)
            data = data[~data["date"].isin(existing_data["date"])]  # Remove duplicates

        if not data.empty:
            data.to_sql(ticker, conn, if_exists="append", index=False)  # Append new data
            print(f"✅ {ticker} data updated in SQLite.")

        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return False

if __name__ == "__main__":
    print("📥 Fetching data for all crude oil stocks...")
    for stock in crude_oil_stocks:
        fetch_stock_data(stock)
    print("✅ All data is now up to date in SQLite.")
    