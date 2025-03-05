import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

# Run the Streamlit App -> streamlit run Monte_Carlo_UI.py
# Locate MonteCarlo folder -> cd MonteCarlo

# Custom Streamlit Theme (Black, White, Red)
st.markdown(
    """
    <style>
        .stApp { background-color: #000000; color: white; }
        div.stButton > button {
            background-color: #ff0000; color: white; border-radius: 10px; padding: 10px;
            border: none; font-weight: bold;
        }
        div.stButton > button:hover { background-color: #cc0000; }
        h1, h2, h3, label, .stSelectbox label, .stSlider label { color: white !important; font-weight: bold; }
        .stMarkdown, .stText { color: white; font-size: 16px; }
        .stSlider > div[role="slider"] { background-color: #ff0000 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI for User Input
st.title("Monte Carlo Simulation for Crude Oil Stocks")

# Dropdown for stock selection
crude_oil_stocks = ["XOM", "CVX", "OXY", "BP", "COP", "EOG", "MPC", "VLO", "PSX", "HES"]
ticker = st.selectbox("Select Crude Oil Stock:", crude_oil_stocks)

# Sliders for simulation parameters
num_simulations = st.slider("Number of Simulations:", min_value=1000, max_value=10000, step=1000, value=5000)
num_days = st.slider("Time Horizon (Days):", min_value=10, max_value=180, step=10, value=30)

if st.button("Run Simulation"):
    # Connect to SQLite Database
    conn = sqlite3.connect("stock_data.db")
    
    try:
        query = f'SELECT * FROM "{ticker}"'
        historical_data = pd.read_sql(query, conn)
    except Exception:
        st.error(f"⚠️ No data found for {ticker}. Run `fetch_data.py` first to fetch stock data.")
        conn.close()
        st.stop()
    
    conn.close()
    historical_data.columns = historical_data.columns.str.lower()
    
    if "close" not in historical_data.columns:
        st.error(f"⚠️ The expected 'close' price column is missing for {ticker}. Check your database structure.")
        st.stop()
    
    if "date" in historical_data.columns:
        historical_data["date"] = pd.to_datetime(historical_data["date"])
        historical_data.set_index("date", inplace=True)
    else:
        st.error(f"⚠️ The table for {ticker} does not contain a 'date' column. Please check your database.")
        st.stop()
    
    historical_prices = historical_data["close"]
    returns = historical_prices.pct_change().dropna()
    
    if len(returns) < 2:
        st.error("⚠️ Not enough historical data to run a meaningful simulation. Try another stock.")
        st.stop()
    
    mu = returns.mean() * 252  # Annualized return
    sigma = returns.std() * np.sqrt(252)  # Annualized volatility
    initial_price = historical_prices.iloc[-1]
    
    # Generate Monte Carlo Simulated Stock Prices
    np.random.seed(42)
    simulated_prices = np.zeros((num_days, num_simulations))
    simulated_prices[0] = initial_price
    
    dt = 1 / 252  # Daily time step
    for t in range(1, num_days):
        random_shocks = np.random.normal(0, 1, num_simulations)
        simulated_prices[t] = simulated_prices[t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks
        )

    # Create Multi-Colored Line Graph (Non-Animated)
    fig = go.Figure()
    
    for i in range(num_simulations):
        fig.add_trace(go.Scatter(
            x=list(range(num_days)),
            y=simulated_prices[:, i],
            mode='lines',
            line=dict(width=0.5),
            opacity=0.5,
            showlegend=False
        ))
    
    # ➖ Add Mean Path in Red
    fig.add_trace(go.Scatter(
        x=list(range(num_days)),
        y=simulated_prices.mean(axis=1),
        mode='lines',
        line=dict(color='red', width=2),
        name='Mean Path'
    ))

    fig.update_layout(
        title=f"Monte Carlo Simulation of {ticker} Stock Prices ({num_simulations} Simulations)",
        xaxis_title="Days",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        showlegend=True
    )

    st.plotly_chart(fig)
    
    final_prices = simulated_prices[-1]
    expected_price = np.mean(final_prices)
    std_dev = np.std(final_prices)
    var_95 = np.percentile(final_prices, 5)
    cvar_95 = np.mean(final_prices[final_prices <= var_95])
    
    st.write(f"📉 **Initial Price ({ticker}):** ${initial_price:.2f}")
    st.write(f"📈 **Expected Price After {num_days} Days:** ${expected_price:.2f}")
    st.write(f"📊 **Standard Deviation of Final Prices:** ${std_dev:.2f}")
    st.write(f"⚠️ **5% Value at Risk (VaR):** ${var_95:.2f}")
    st.write(f"🔻 **Conditional VaR (Expected Shortfall):** ${cvar_95:.2f}")