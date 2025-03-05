import yfinance as yf
import pandas as pd
import sqlite3

crude_oil_stocks = ["XOM", "CVX", "OXY", "BP", "COP", "EOG", "MPC", "VLO", "PSX", "HES"]

def fetch_stock_data(ticker, period="6mo"):
    """Fetch historical stock data and store it in SQLite."""
    print(f"Fetching data for {ticker}...")

    try:
        # ✅ Download stock data
        data = yf.download(ticker, period=period)[["Close"]]

        if data.empty:
            print(f"⚠️ No data found for {ticker}. Skipping...")
            return False
        
        # ✅ Ensure "Date" is properly stored as a column
        data.reset_index(inplace=True)

        # ✅ Flatten Multi-Index Columns if necessary
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # ✅ Rename columns properly
        data.rename(columns={"Date": "date", "Close": "close"}, inplace=True)

        # ✅ Convert Date column to string format to avoid SQLite issues
        data["date"] = data["date"].astype(str)

        # ✅ Store in SQLite
        conn = sqlite3.connect("stock_data.db")
        data.to_sql(ticker, conn, if_exists="replace", index=False)  # Save without index
        conn.close()

        print(f"✅ Data for {ticker} saved to SQLite database.")
        return True
    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {e}")
        return False

if __name__ == "__main__":
    print("📥 Fetching data for all crude oil stocks...")
    for stock in crude_oil_stocks:
        fetch_stock_data(stock)
    print("✅ All data has been stored in SQLite.")