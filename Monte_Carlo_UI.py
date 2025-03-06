import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import subprocess
from arch import arch_model
import scipy.stats as stats

# Run the Streamlit App -> streamlit run Monte_Carlo_UI.py
# Locate MonteCarlo folder -> cd MonteCarlo

# Set wide layout
st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

# Initialize session state for dark mode
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Toggle Dark Mode
st.session_state["dark_mode"] = st.toggle("Toggle Dark Mode", value=st.session_state["dark_mode"])

# Apply Background Overlay Based on Theme
light_mode_image = "https://i.imgur.com/fwgvLyX.jpeg"
dark_mode_image = "https://i.imgur.com/WvFsAcX.jpeg"
background_image = dark_mode_image if st.session_state["dark_mode"] else light_mode_image

st.markdown(
    f"""
    <style>
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('{background_image}');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            opacity: 0.2;
            z-index: -1;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Monte Carlo Simulation for Crude Oil Stocks")

crude_oil_stocks = ["XOM", "CVX", "OXY", "BP", "COP", "EOG", "MPC", "VLO", "PSX", "HES"]
ticker = st.selectbox("Select Crude Oil Stock:", crude_oil_stocks)
num_simulations = st.slider("Number of Simulations:", min_value=1000, max_value=10000, step=1000, value=5000)
num_days = st.slider("Time Horizon (Days):", min_value=10, max_value=180, step=10, value=30)

def fetch_latest_stock_data(ticker):
    try:
        subprocess.run(["python", "Fetch_Data.py"], check=True)
        conn = sqlite3.connect("stock_data.db")
        historical_data = pd.read_sql(f'SELECT * FROM "{ticker}"', conn)
        conn.close()
        return historical_data
    except Exception as e:
        st.error(f"❌ Error fetching the latest stock data: {e}")
        return None

historical_data = fetch_latest_stock_data(ticker)
if historical_data is None or historical_data.empty:
    st.error(f"⚠️ Data for {ticker} is unavailable.")
    st.stop()
else:
    st.success(f"✅ Data for {ticker} loaded successfully.")

simulated_prices = None  
if st.button("Run Simulation"):
    historical_data.columns = historical_data.columns.str.lower()
    if "close" not in historical_data.columns:
        st.error(f"⚠️ The expected 'close' price column is missing for {ticker}.")
        st.stop()
    if "date" in historical_data.columns:
        historical_data["date"] = pd.to_datetime(historical_data["date"])
    
    historical_prices = historical_data["close"]
    returns = historical_prices.pct_change().dropna()
    if len(returns) < 2:
        st.error("⚠️ Not enough historical data to run a meaningful simulation.")
        st.stop()
    
    # Fit GARCH Model
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_results = garch_model.fit(disp='off')
    sigma_t = garch_results.conditional_volatility
    
    # Monte Carlo Simulation with Extreme Event Adjustments
    mu = returns.mean() * 252
    dt = 1 / 252
    
    lambda_jump = 0.1  # Expected number of jumps per year
    jump_magnitude = np.random.normal(-0.02, 0.05, num_simulations)
    nu = 5  # Degrees of freedom for t-distribution
    
    # Regime-Switching Model
    bull_mu, bear_mu = 0.08, -0.15
    bull_sigma, bear_sigma = 0.15, 0.45
    p_bull_to_bear, p_bear_to_bull = 0.2, 0.3
    market_regime = np.random.choice(["bull", "bear"], p=[0.8, 0.2])
    
    np.random.seed(42)
    simulated_prices = np.zeros((num_days, num_simulations))
    simulated_prices[0] = historical_prices.iloc[-1]
    
    for t in range(1, num_days):
        jump_occurred = np.random.poisson(lambda_jump, num_simulations)
        random_shocks = stats.t.rvs(df=nu, size=num_simulations)
        
        if market_regime == "bull":
            mu, sigma = bull_mu, bull_sigma
            market_regime = np.random.choice(["bull", "bear"], p=[1 - p_bull_to_bear, p_bull_to_bear])
        else:
            mu, sigma = bear_mu, bear_sigma
            market_regime = np.random.choice(["bull", "bear"], p=[p_bear_to_bull, 1 - p_bear_to_bull])

        simulated_prices[t] = simulated_prices[t-1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks + jump_occurred * jump_magnitude
        )

    final_prices = simulated_prices[-1]
    fig = go.Figure()
    for i in range(num_simulations):
        fig.add_trace(go.Scatter(x=list(range(num_days)), y=simulated_prices[:, i], mode='lines', opacity=0.5, showlegend=False))
    fig.add_trace(go.Scatter(x=list(range(num_days)), y=simulated_prices.mean(axis=1), mode='lines', line=dict(color='red', width=2), name='Mean Path'))
    st.plotly_chart(fig)
    
    csv_filename = f"MonteCarlo_{ticker}.csv"
    csv_data = pd.DataFrame(simulated_prices)
    csv_data.index = range(1, num_days + 1)
    csv_data.index.name = "Day"
    csv_data.columns = [f"Simulation {i+1}" for i in range(num_simulations)]
    csv_data = csv_data.round(2)
    csv_data = csv_data.reset_index()
    
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.download_button("Download Simulated Data", data=csv_data.to_csv(index=False).encode("utf-8"), file_name=csv_filename, mime="text/csv")
