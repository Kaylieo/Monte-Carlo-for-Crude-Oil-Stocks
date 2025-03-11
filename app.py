import os
os.environ["RPY2_NO_CONTEXTVAR"] = "1"

import streamlit as st
import pandas as pd
import sqlite3
import subprocess
import plotly.graph_objects as go
import numpy as np
from monte_carlo import run_simulation_logic, run_ms_egarch_simulation_logic
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri

# Activate automatic conversion for numpy and pandas objects
numpy2ri.activate()
pandas2ri.activate()

# --- Define a cached R function loader for the Markov-switching EGARCH model ---
@st.cache_resource
def get_ms_egarch_fit():
    """
    Loads the R function 'ms_egarch_fit' using MSGARCH.
    This function is cached to run only once in a single-threaded context.
    """
    r_code = """
    ms_egarch_fit <- function(returns_vector) {
      if(!require(MSGARCH)) {
        install.packages("MSGARCH", repos = "https://cran.rstudio.com/")
        library(MSGARCH)
      }
      spec <- CreateSpec(model = "MS-GARCH", distribution = "sstd", K = 2)
      fit <- FitML(spec, returns_vector)
      return(fit)
    }
    """
    robjects.r(r_code)
    return robjects.globalenv['ms_egarch_fit']

st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

# Dark mode toggle
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
st.session_state["dark_mode"] = st.toggle("Toggle Dark Mode", value=st.session_state["dark_mode"])

# Colors: define text_color, background_color, etc.
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
        div.stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: #cc0000;
        }}
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
        
        /* Ensure the radio button labels use the same text color in dark mode */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label span,
        [data-testid="stRadio"] label div,
        [data-testid="stRadio"] label p {{
            color: {text_color} !important;
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

# Let the user select the model type:
model_type = st.radio("Select Model Type:", ("Standard EGARCH", "Markov-switching EGARCH (MSGARCH via R)"))

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

historical_data = fetch_latest_stock_data(ticker)
if historical_data is None or historical_data.empty:
    st.error(f"⚠️ Data for {ticker} is unavailable.")
    st.stop()
else:
    st.success(f"✅ Data for {ticker} loaded successfully.")

if st.button("Run Simulation"):
    historical_data.columns = historical_data.columns.str.lower()

    # Use the selected model type
    if model_type == "Standard EGARCH":
        try:
            sim_results = run_simulation_logic(historical_data, ticker, num_days, num_simulations)
        except ValueError as ve:
            st.error(f"⚠️ {ve}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.stop()
    else:
        try:
            sim_results = run_ms_egarch_simulation_logic(historical_data, ticker, num_days, num_simulations)
        except ValueError as ve:
            st.error(f"⚠️ {ve}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.stop()

    st.info("EGARCH model summary has been printed to the terminal.")

    final_prices = sim_results["final_prices"]
    simulated_prices = sim_results["simulated_prices"]
    initial_price = sim_results["initial_price"]

    # Plot simulated paths with explicit best, worst, and mean paths
    x_values = list(range(num_days + 1))
    fig = go.Figure()
    for i in range(num_simulations):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=simulated_prices[:, i],
                mode='lines',
                line=dict(width=0.5),
                opacity=0.5,
                showlegend=False
            )
        )
    best_idx = int(np.argmax(final_prices))
    worst_idx = int(np.argmin(final_prices))
    best_path = simulated_prices[:, best_idx]
    worst_path = simulated_prices[:, worst_idx]
    mean_path = simulated_prices.mean(axis=1)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=mean_path,
            mode='lines',
            line=dict(color='red', width=3),
            name='Mean Path'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=best_path,
            mode='lines',
            line=dict(color='green', width=3),
            name='Best Path'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=worst_path,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Worst Path'
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

    # Plot Final Price Distribution Histogram
    st.markdown("<h3 style='text-align:center'>Distribution of Final Prices</h3>", unsafe_allow_html=True)
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=final_prices,
            nbinsx=50,
            marker_color='#ff0000',
            opacity=0.7
        )
    )
    fig_hist.update_layout(
        xaxis_title="Final Price",
        yaxis_title="Frequency",
        template="plotly_dark" if st.session_state["dark_mode"] else "plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Download button
    csv_filename = f"MonteCarlo_{ticker}.csv"
    csv_data = pd.DataFrame(simulated_prices)
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