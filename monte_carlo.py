import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import subprocess
from arch import arch_model
from scipy.stats import t
from scipy.stats.qmc import Sobol

st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

# Dark mode toggle
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
st.session_state["dark_mode"] = st.toggle("Toggle Dark Mode", value=st.session_state["dark_mode"])

# Colors (no background images)
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

# Custom CSS
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
        [data-testid="stWidgetLabel"] {{
            color: {text_color} !important;
            font-weight: bold;
        }}
        h1 {{
            text-align: center;
        }}
        div[data-testid="stNotification"] {{
            display: flex;
            justify-content: center;
        }}
        div.stButton {{
            display: flex;
            justify-content: center;
        }}
        .stDataFrame {{
            margin: auto;
            width: 80% !important;
        }}
        h2 {{
            text-align: center;
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
    """Fetch latest data from Yahoo via Fetch_Data.py, then read from SQLite."""
    try:
        subprocess.run(["python", "Fetch_Data.py"], check=True)
        conn = sqlite3.connect("stock_data.db")
        df = pd.read_sql(f'SELECT * FROM "{ticker}"', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"❌ Error fetching the latest stock data: {e}")
        return None

def run_simulation_logic(historical_data, ticker, num_days, num_simulations):
    """
    Runs the Monte Carlo simulation and returns a dictionary with results.
    If required columns or data are missing, it calls st.error() and stops the app.
    """
    # Validate required columns
    if "close" not in historical_data.columns:
        st.error(f"⚠️ The expected 'close' price column is missing for {ticker}.")
        st.stop()
    if "date" not in historical_data.columns:
        st.error(f"⚠️ The table for {ticker} does not contain a 'date' column.")
        st.stop()
    
    # Prepare historical data
    historical_data["date"] = pd.to_datetime(historical_data["date"])
    historical_prices = historical_data["close"].dropna()

    returns = historical_prices.pct_change().dropna()
    if len(returns) < 2:
        st.error("⚠️ Not enough historical data to run a meaningful simulation.")
        st.stop()

    # Annualized mean return (simple estimate)
    mu = returns.mean() * 252

    st.write("Fitting EGARCH(1,1) with AR(1) mean and skew-t distribution for heavier tails...")
    garch = arch_model(
        returns,
        mean='AR',
        lags=1,
        vol='EGARCH',
        p=1,
        o=1,
        q=1,
        dist='skewt',
        rescale=True
    ).fit(disp="on")

    st.write("Model Summary:")
    try:
        st.write(garch.summary())
    except Exception as e:
        st.warning("Warning: Could not compute full model summary due to: " + str(e))

    # -----------------------------
    # MANUALLY COMPUTE MULTI-STEP EGARCH VOLATILITY FORECAST
    # -----------------------------
    last_variance = garch.conditional_volatility.iloc[-1] ** 2
    L = np.log(last_variance)
    omega_param = garch.params['omega']
    beta_param = garch.params['beta[1]']
    long_run_log_var = omega_param / (1 - beta_param)
    forecast_log_vars = np.array([
        long_run_log_var + (beta_param ** h) * (L - long_run_log_var)
        for h in range(1, num_days + 1)
    ])
    sigma_forecast = np.exp(0.5 * forecast_log_vars)

    if sigma_forecast is None or len(sigma_forecast) == 0:
        st.warning("⚠️ Final fallback to historical volatility.")
        sigma_forecast = np.array([np.std(returns)] * num_days)
        sigma_forecast = np.nan_to_num(sigma_forecast, nan=np.std(returns))

    sigma = np.repeat(sigma_forecast[:, np.newaxis], num_simulations, axis=1)
    initial_price = historical_prices.iloc[-1]

    np.random.seed(42)
    simulated_prices = np.zeros((num_days + 1, num_simulations))
    simulated_prices[0] = initial_price

    dt = 1 / 252  # daily fraction of a trading year
    dof = 5       # degrees of freedom for t-distribution

    sobol = Sobol(d=num_simulations, scramble=True)
    random_shocks = sobol.random(num_days)
    random_shocks = t.ppf(random_shocks, dof)

    drift = (mu - 0.5 * sigma_forecast**2)[:, np.newaxis] * dt
    diffusion = sigma * np.sqrt(dt) * random_shocks

    for day in range(1, num_days + 1):
        simulated_prices[day] = (
            simulated_prices[day - 1] *
            np.exp(drift[day - 1, 0] + diffusion[day - 1, :])
        )

    final_prices = simulated_prices[-1]

    results = {
        "final_prices": final_prices,
        "initial_price": float(initial_price),
        "expected_price": float(np.mean(final_prices)),
        "std_dev": float(np.std(final_prices)),
        "var_5pct": float(np.percentile(final_prices, 5)),
        "cvar_5pct": float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])),
        "simulated_prices": simulated_prices
    }
    return results

# Get historical data from SQLite (via Fetch_Data.py)
historical_data = fetch_latest_stock_data(ticker)
if historical_data is None or historical_data.empty:
    st.error(f"⚠️ Data for {ticker} is unavailable.")
    st.stop()
else:
    st.success(f"✅ Data for {ticker} loaded successfully.")

if st.button("Run Simulation"):
    historical_data.columns = historical_data.columns.str.lower()
    sim_results = run_simulation_logic(historical_data, ticker, num_days, num_simulations)
    
    final_prices = sim_results["final_prices"]
    initial_price = sim_results["initial_price"]

    # Plot results
    x_values = list(range(num_days + 1))
    fig = go.Figure()
    for i in range(num_simulations):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=sim_results["simulated_prices"][:, i],
                mode='lines',
                line=dict(width=0.5),
                opacity=0.5,
                showlegend=False
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=sim_results["simulated_prices"].mean(axis=1),
            mode='lines',
            line=dict(color='red', width=2),
            name='Mean Path'
        )
    )
    fig.update_layout(
        title={
            'text': f"Monte Carlo Simulation of {ticker} ({num_simulations} Simulations)",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Days",
        yaxis_title="Stock Price ($)",
        template="plotly_dark" if st.session_state["dark_mode"] else "plotly_white"
    )
    st.plotly_chart(fig)

    # Prepare CSV for download
    csv_filename = f"MonteCarlo_{ticker}.csv"
    csv_data = pd.DataFrame(sim_results["simulated_prices"])
    csv_data.index = range(1, num_days + 2)
    csv_data.index.name = "Day"
    csv_data.columns = [f"Simulation {i+1}" for i in range(num_simulations)]
    csv_data = csv_data.round(2).reset_index()
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.download_button(
            label="Download Simulated Data",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name=csv_filename,
            mime="text/csv"
        )
    st.markdown("<hr style='border: 1px solid #ff0000;'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h3>📉 <strong>Initial Price ({ticker}):</strong> ${initial_price:.2f}</h3>
            <h3>📈 <strong>Expected Price After {num_days} Days:</strong> ${sim_results["expected_price"]:.2f}</h3>
            <h3>📊 <strong>Std. Dev. of Final Prices:</strong> ${sim_results["std_dev"]:.2f}</h3>
            <h3>⚠️ <strong>5% VaR:</strong> ${sim_results["var_5pct"]:.2f}</h3>
            <h3>🔻 <strong>Conditional VaR (Expected Shortfall):</strong> ${sim_results["cvar_5pct"]:.2f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr style='border: 1px solid #ff0000;'>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center; color: {text_color};'>Historical Prices</h2>", unsafe_allow_html=True)

# Format historical data table with two decimals for the 'close' column
styled_df = historical_data[["date", "close"]].tail(10).copy()
styled_df["close"] = styled_df["close"].apply(lambda x: f"{x:.2f}")
st.markdown(
    """
    <style>
        .stDataFrame {
            margin-left: auto;
            margin-right: auto;
            width: 60% !important;
        }
        .dataframe tbody tr th {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.dataframe(styled_df, use_container_width=True, hide_index=True)