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

def fetch_stock_data(ticker, period="6mo"):
    """Fetches historical stock data and updates SQLite only if needed."""
    print(f"🔍 Checking latest data for {ticker}...")

    conn = sqlite3.connect("stock_data.db")
    ensure_table_exists(conn, ticker)  # ✅ Ensure table exists
    cursor = conn.cursor()

    # ✅ Check the latest stored date
    cursor.execute(f"SELECT MAX(date) FROM {ticker}")
    last_date = cursor.fetchone()[0]

    needs_update = False
    if last_date is None:
        print(f"⚠️ No existing data for {ticker}. Fetching full data...")
        needs_update = True
    else:
        try:
            last_date = datetime.strptime(last_date, "%Y-%m-%d")
            if last_date < datetime.today() - timedelta(days=1):
                print(f"🔄 Updating {ticker} with latest prices...")
                needs_update = True
            else:
                print(f"✅ {ticker} data is already up to date!")
        except ValueError:
            print(f"⚠️ Error parsing last_date for {ticker}. Fetching fresh data...")
            needs_update = True

    if not needs_update:
        conn.close()
        return False  # No update needed

    try:
        # ✅ Fetch latest stock data (handles auto_adjust notice)
        data = yf.download(ticker, period=period)[["Close"]]

        if data.empty:
            print(f"⚠️ No new data found for {ticker}. Skipping...")
            conn.close()
            return False

        # ✅ Flatten MultiIndex if it exists
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # ✅ Ensure "Date" is properly stored as a column
        data.reset_index(inplace=True)
        data.rename(columns={"Date": "date", "Close": "close", f"Close_{ticker}": "close"}, inplace=True)

        # ✅ Print columns for debugging
        print("DataFrame columns before insertion:", data.columns.tolist())

        # ✅ Ensure column names are formatted correctly
        data.columns = data.columns.str.lower().str.strip()
        data.rename(columns={"date": "date", "close": "close"}, inplace=True)

        # ✅ Convert Date column to string format for SQLite compatibility
        data["date"] = data["date"].astype(str)
        data["close"] = data["close"].astype(float)  # Ensures proper float values

        # ✅ If table exists, append only new data
        existing_dates = pd.read_sql(f'SELECT date FROM "{ticker}"', conn)["date"].tolist()
        new_data = data[~data["date"].isin(existing_dates)]  # Remove duplicates

        if not new_data.empty:
            new_data[['date', 'close']].to_sql(ticker, conn, if_exists="append", index=False)
            print(f"✅ {ticker} data updated in SQLite.")
        else:
            print(f"⚠️ No new rows to add for {ticker}.")

        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        conn.close()
        return False

if __name__ == "__main__":
    print("📥 Fetching data for all crude oil stocks...")
    for stock in crude_oil_stocks:
        fetch_stock_data(stock)
    print("✅ All data is now up to date in SQLite.")