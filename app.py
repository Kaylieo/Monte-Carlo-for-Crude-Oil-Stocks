# -------------------------
# IMPORTS
# -------------------------
import os
os.environ["RPY2_NO_CONTEXTVAR"] = "1"

import streamlit as st
import pandas as pd
import sqlite3
import subprocess
import numpy as np
import plotly.graph_objects as go
from monte_carlo import MonteCarloSimulator
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri

# Activate automatic conversion for numpy and pandas objects
numpy2ri.activate()
pandas2ri.activate()

st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

# -------------------------
# COLOR PALETTE
# -------------------------
primary_color = "#73648A"      # Primary color for buttons
secondary_color = "#0F0326"    # Dark mode background
neutral_color = "#EBE8FC"      # Light mode background

# -------------------------
# DARK MODE BUTTON
# -------------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

button_label = "Light Mode" if st.session_state["dark_mode"] else "Dark Mode"

if st.button(button_label):
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]
    st.rerun()  # Forces immediate rerun so the label changes right away

# -------------------------
# SET COLORS BASED ON DARK MODE
# -------------------------
if st.session_state["dark_mode"]:
    background_color = secondary_color
    text_color = neutral_color
    button_color = primary_color
    slider_color = primary_color
    hr_border_color = primary_color
    button_hover_color = primary_color
else:
    background_color = neutral_color
    text_color = secondary_color
    button_color = primary_color
    slider_color = primary_color
    hr_border_color = primary_color
    button_hover_color = primary_color

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}

        /* Buttons (Run Simulation, Download, etc.) */
        div.stButton > button, .stDownloadButton > button {{
            background-color: {button_color}; 
            color: white; 
            border-radius: 10px;
            padding: 10px; 
            border: none; 
            font-weight: bold;
        }}
        /* Hover color for active buttons */
        div.stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: #5C4E72 !important; /* Slightly darker purple */
            color: white !important;
        }}
        /* Focus & Active states - override red */
        div.stButton > button:focus,
        div.stButton > button:active,
        div.stButton > button:focus:active,
        div.stButton > button:focus:not(:disabled) {{
            background-color: #5C4E72 !important;
            color: white !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        /* Disabled button color override */
        div.stButton > button[disabled],
        div.stButton > button:disabled,
        div.stButton > button[aria-disabled="true"],
        div.stButton > button[disabled]:hover,
        div.stButton > button:disabled:hover,
        div.stButton > button[aria-disabled="true"]:hover {{
            background-color: {button_color} !important;
            color: white !important;
            opacity: 0.6 !important;
            cursor: not-allowed !important;
            border: none !important;
            box-shadow: none !important;
        }}

        /* Text elements */
        .stMarkdown,
        .stText,
        .stSelectbox,
        .stSlider label,
        .stSlider div,
        label,
        [data-testid="stWidgetLabel"] {{
            color: {text_color} !important;
            font-weight: bold;
        }}

        /* --- Slider Styling (Only Change Selected Track + Handle) --- */

        /* 1. The selected track portion (from min to handle) */
        [data-baseweb="slider"] > div:nth-child(2) [role="progressbar"] {{
            background-color: {slider_color} !important;
        }}

        /* 2. The handle (knob) */
        [data-baseweb="slider"] [role="slider"] {{
            background-color: {slider_color} !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            width: 15px !important;
            height: 15px !important;
            border-radius: 50% !important;
        }}

        /* Optional: Handle hover/focus states */
        [data-baseweb="slider"] [role="slider"]:hover,
        [data-baseweb="slider"] [role="slider"]:focus {{
            background-color: {slider_color} !important;
            border: none !important;
            box-shadow: 0 0 0 2px {slider_color}40 !important;
            outline: none !important;
        }}

        /* Dropdown (select box) border styling */
        [data-testid="stSelectbox"] div[role="listbox"] {{
            border: 2px solid {primary_color} !important;
        }}

        /* Header alignment */
        h1, h2 {{
            text-align: center;
        }}

        /* Center notifications & buttons */
        div[data-testid="stNotification"], div.stButton {{
            display: flex;
            justify-content: center;
        }}

        /* DataFrame styling */
        .stDataFrame {{
            margin: auto;
            width: 80% !important;
        }}

        /* Radio button text color fix */
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

# -------------------------
# APP CONTENT
# -------------------------
st.title("Monte Carlo Simulation (Markov-switching EGARCH)")

# Crude oil stocks
crude_oil_stocks = ["XOM", "CVX", "OXY", "BP", "COP", "EOG", "MPC", "VLO", "PSX", "HES"]
ticker = st.selectbox("Select Crude Oil Stock:", crude_oil_stocks)

num_simulations = st.slider("Number of Simulations:", min_value=1000, max_value=10000, step=1000, value=5000)
num_days = st.slider("Time Horizon (Days):", min_value=10, max_value=180, step=10, value=30)

# Info message about Markov-switching
st.markdown(
    """
    <div style="text-align: center;">
        <div class="stInfo">
            A Markov-switching EGARCH model summary will be printed to the terminal once you run the simulation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

def fetch_latest_stock_data(ticker):
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
    st.markdown(
        f"""
        <div style="text-align: center;">
            <div class="stSuccess">
                Data for {ticker} loaded successfully.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def run_simulation():
    """Run Monte Carlo simulation using the MonteCarloSimulator class."""
    try:
        simulator = MonteCarloSimulator(ticker)
        simulator.fit_model(historical_data)
        return simulator.run_simulation(num_days, num_simulations)
    except ValueError as ve:
        st.error(f"⚠️ {ve}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

if st.button("Run Simulation"):
    historical_data.columns = historical_data.columns.str.lower()
    sim_results = run_simulation()

    final_prices = sim_results.final_prices
    simulated_prices = sim_results.simulated_prices
    initial_price = sim_results.initial_price

    x_values = list(range(num_days + 1))
    fig = go.Figure()

    # Plot all simulation paths
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
    # Best, worst, mean
    best_idx = int(np.argmax(final_prices))
    worst_idx = int(np.argmin(final_prices))
    best_path = simulated_prices[:, best_idx]
    worst_path = simulated_prices[:, worst_idx]
    mean_path = simulated_prices.mean(axis=1)

    fig.add_trace(go.Scatter(x=x_values, y=mean_path, mode='lines',
                             line=dict(color=primary_color, width=3), name='Mean Path'))
    fig.add_trace(go.Scatter(x=x_values, y=best_path, mode='lines',
                             line=dict(color='green', width=3), name='Best Path'))
    fig.add_trace(go.Scatter(x=x_values, y=worst_path, mode='lines',
                             line=dict(color='blue', width=3), name='Worst Path'))
    fig.update_layout(
        title={'text': f"Monte Carlo Simulation of {ticker} ({num_simulations} Simulations)", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Days",
        yaxis_title="Stock Price ($)",
        template="plotly_dark" if st.session_state["dark_mode"] else "plotly_white"
    )
    st.plotly_chart(fig)

    # Distribution of final prices
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=final_prices,
            nbinsx=50,
            marker_color=primary_color,
            opacity=0.7
        )
    )
    fig_hist.update_layout(
        title={'text': f"Distribution of Final Prices for {ticker}", 'x': 0.5, 'xanchor': 'center'},
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

    st.markdown(f"<hr style='border: 1px solid {hr_border_color};'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h3>📉 <strong>Initial Price ({ticker}):</strong> ${initial_price:.2f}</h3>
            <h3>📈 <strong>Expected Price After {num_days} Days:</strong> ${sim_results.expected_price:.2f}</h3>
            <h3>📊 <strong>Std. Dev. of Final Prices:</strong> ${sim_results.std_dev:.2f}</h3>
            <h3>⚠️ <strong>5% VaR:</strong> ${sim_results.var_5pct:.2f}</h3>
            <h3>🔻 <strong>Conditional VaR (Expected Shortfall):</strong> ${sim_results.cvar_5pct:.2f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(f"<hr style='border: 1px solid {hr_border_color};'>", unsafe_allow_html=True)
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