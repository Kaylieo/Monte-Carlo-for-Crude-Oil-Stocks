import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import subprocess

# Run the Streamlit App -> streamlit run Monte_Carlo_UI.py
# Locate MonteCarlo folder -> cd MonteCarlo

# Set wide layout
st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

# Initialize session state for dark mode
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Toggle Dark Mode
st.session_state["dark_mode"] = st.toggle("Toggle Dark Mode", value=st.session_state["dark_mode"])

# Apply Theme Colors
if st.session_state["dark_mode"]:
    background_color = "#000000"
    text_color = "white"
    button_color = "#ff0000"
    slider_color = "#ff0000"
else:
    background_color = "#ffffff"
    text_color = "black"
    button_color = "#ff0000"
    slider_color = "#ff0000"

# Apply Custom CSS
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}
        div.stButton > button, .stDownloadButton > button {{
            background-color: {button_color}; color: white; border-radius: 10px; padding: 10px;
            border: none; font-weight: bold;
        }}
        div.stButton > button:hover, .stDownloadButton > button:hover {{ background-color: #cc0000; }}
        .stMarkdown, .stText, .stSelectbox, .stSlider label, .stSlider div, label {{
            color: {text_color} !important; font-weight: bold;
        }}
        .stSlider > div[role="slider"] {{
            background-color: {slider_color} !important;
        }}
        /* Ensure Toggle Dark Mode Text Adjusts Dynamically */
        [data-testid="stWidgetLabel"] {{
            color: {text_color} !important;
            font-weight: bold;
        }}
        /* Center the title */
        h1 {{
            text-align: center;
        }}
        /* Center the success message */
        div[data-testid="stNotification"] {{
            display: flex;
            justify-content: center;
        }}
        /* Center the Run Simulation button */
        div.stButton {{
            display: flex;
            justify-content: center;
        }}
        /* Center Historical Prices and Make it Wider */
        .stDataFrame {{
            margin: auto;
            width: 80% !important;  /* Adjust width as needed */
        }}
        h2 {{
            text-align: center;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Monte Carlo Simulation for Crude Oil Stocks")  # UI title centered

# Dropdown for stock selection
crude_oil_stocks = ["XOM", "CVX", "OXY", "BP", "COP", "EOG", "MPC", "VLO", "PSX", "HES"]
ticker = st.selectbox("Select Crude Oil Stock:", crude_oil_stocks)

# Sliders for simulation parameters
num_simulations = st.slider("Number of Simulations:", min_value=1000, max_value=10000, step=1000, value=5000)
num_days = st.slider("Time Horizon (Days):", min_value=10, max_value=180, step=10, value=30)

# Function to fetch data if missing
def fetch_data_if_missing(ticker):
    """Checks if stock data exists, and if not, runs Fetch_Data.py to download it."""
    conn = sqlite3.connect("stock_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (ticker,))
    table_exists = cursor.fetchone() is not None
    conn.close()

    if not table_exists:
        st.warning(f"⚠️ No data found for {ticker}. Fetching data automatically...")
        try:
            subprocess.run(["python", "Fetch_Data.py"], check=True)
            conn = sqlite3.connect("stock_data.db")
            historical_data = pd.read_sql(f'SELECT * FROM "{ticker}"', conn)
            conn.close()
            return historical_data
        except Exception as e:
            st.error(f"❌ Failed to fetch data automatically. Error: {e}")
            return None
    
    conn = sqlite3.connect("stock_data.db")
    try:
        historical_data = pd.read_sql(f'SELECT * FROM "{ticker}"', conn)
    except Exception:
        st.error(f"⚠️ Error retrieving data for {ticker}.")
        historical_data = None
    conn.close()
    return historical_data

# Fetch data
historical_data = fetch_data_if_missing(ticker)

if historical_data is None or historical_data.empty:
    st.error(f"⚠️ Data for {ticker} is unavailable.")
    st.stop()
else:
    st.success(f"✅ Data for {ticker} loaded successfully.")

# Initialize simulated_prices to prevent reference issues
simulated_prices = None  

if st.button("Run Simulation"):
    historical_data.columns = historical_data.columns.str.lower()

    if "close" not in historical_data.columns:
        st.error(f"⚠️ The expected 'close' price column is missing for {ticker}. Check your database structure.")
        st.stop()

    if "date" in historical_data.columns:
        historical_data["date"] = pd.to_datetime(historical_data["date"])
    else:
        st.error(f"⚠️ The table for {ticker} does not contain a 'date' column.")
        st.stop()

    historical_prices = historical_data["close"]
    returns = historical_prices.pct_change().dropna()

    if len(returns) < 2:
        st.error("⚠️ Not enough historical data to run a meaningful simulation. Try another stock.")
        st.stop()

    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    initial_price = historical_prices.iloc[-1]

    # Monte Carlo Simulation
    np.random.seed(42)
    simulated_prices = np.zeros((num_days, num_simulations))
    simulated_prices[0] = initial_price

    dt = 1 / 252
    for t in range(1, num_days):
        random_shocks = np.random.normal(0, 1, num_simulations)
        simulated_prices[t] = simulated_prices[t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks
        )

    final_prices = simulated_prices[-1]

    # Plot Simulation
    fig = go.Figure()
    for i in range(num_simulations):
        fig.add_trace(go.Scatter(x=list(range(num_days)), y=simulated_prices[:, i], mode='lines', line=dict(width=0.5), opacity=0.5, showlegend=False))

    fig.add_trace(go.Scatter(x=list(range(num_days)), y=simulated_prices.mean(axis=1), mode='lines', line=dict(color='red', width=2), name='Mean Path'))

    fig.update_layout(
    title={
        'text': f"Monte Carlo Simulation of {ticker} Stock Prices ({num_simulations} Simulations)",
        'x': 0.5,  
        'xanchor': 'center',  
        'yanchor': 'top'
    },
    xaxis_title="Days",
    yaxis_title="Stock Price ($)",
    template="plotly_dark" if st.session_state["dark_mode"] else "plotly_white",
    showlegend=True
    )

    st.plotly_chart(fig)

    st.markdown("<hr style='border: 1px solid #ff0000;'>", unsafe_allow_html=True)

    # Display Key Stats
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h3>📉 <strong>Initial Price ({ticker}):</strong> ${initial_price:.2f}</h3>
            <h3>📈 <strong>Expected Price After {num_days} Days:</strong> ${np.mean(final_prices):.2f}</h3>
            <h3>📊 <strong>Standard Deviation of Final Prices:</strong> ${np.std(final_prices):.2f}</h3>
            <h3>⚠️ <strong>5% Value at Risk (VaR):</strong> ${np.percentile(final_prices, 5):.2f}</h3>
            <h3>🔻 <strong>Conditional VaR (Expected Shortfall):</strong> ${np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]):.2f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display Historical Data with Better Styling
st.markdown("<hr style='border: 1px solid #ff0000;'>", unsafe_allow_html=True)  # Add a divider

# Center and Style the "Historical Prices" Header
st.markdown(
    f"<h2 style='text-align: center; color: {text_color};'>Historical Prices</h2>",
    unsafe_allow_html=True
)

# Apply Better Table Styling for Readability
styled_df = historical_data[["date", "close"]].tail(10).style.set_table_styles(
    [
        {"selector": "thead th", "props": [("background-color", "#ff0000"), ("color", "white"), ("font-size", "16px"), ("text-align", "center")]},
        {"selector": "tbody td", "props": [("background-color", "#f9f9f9"), ("text-align", "center"), ("font-size", "14px")]}
    ]
)

st.dataframe(styled_df)

# Provide CSV Download Button (only if simulation was run)
if simulated_prices is not None:
    csv_filename = f"MonteCarlo_{ticker}.csv"
    # Format DataFrame for Better Readability
    csv_data = pd.DataFrame(simulated_prices)
    csv_data.index = range(1, num_days + 1)  # Number rows as days
    csv_data.index.name = "Day"
    csv_data.columns = [f"Simulation {i+1}" for i in range(num_simulations)]  # Add column labels
    csv_data = csv_data.round(2)  # Reduce decimal places for clarity

    # Reset index so "Day" is a column in the CSV
    csv_data = csv_data.reset_index()

    # Better Centering for Download Button
    col1, col2, col3 = st.columns([3, 1, 3])  # Equal side margins for perfect centering
    with col2:
        st.download_button(
            label="Download Simulated Data",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name=csv_filename,
            mime="text/csv"
        )