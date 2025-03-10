# Monte Carlo Simulation for Crude Oil Stocks

A **Streamlit-based** Monte Carlo simulation for predicting crude oil stock prices using **historical data** and **stochastic modeling**.

---

## Features

✅ **Stock Selection** - Choose from major crude oil stocks (XOM, CVX, BP, etc.)  
✅ **Monte Carlo Simulations** - Run up to **10,000** simulations  
✅ **Time Horizon Customization** - Forecast stock prices up to **180 days**  
✅ **Interactive UI** - Built with **Streamlit**  
✅ **Visualizations** - Multi-colored **Plotly** line charts  
✅ **Risk Metrics** - Calculates **Expected Price, VaR, and CVaR**  

---

## 📂 Project Structure

MonteCarlo/

│── stock_data.db         **# SQLite database storing historical stock prices**

│── Monte_Carlo_UI.py     **# Streamlit app for user interaction**

│── Fetch_Data.py         **# Fetches stock data and stores in SQLite**

│── requirements.txt      **# Dependencies for the project**

│── README.md             **# Project documentation**

│── .gitignore            **# Files to be ignored in version control**

---

## 🛠 Installation & Setup
```bash
1️⃣ Clone the Repository:
git clone https://github.com/Kaylieo/Monte-Carlo-for-Crude-Oil-Stocks.git
cd Monte-Carlo-for-Crude-Oil-Stocks

2️⃣ Create Virtual Environment:
conda env create -f environment.yml
conda activate MonteCarloEnv # Mac/Linux

3️⃣ Install Dependencies:
pip install -r requirements.txt

4️⃣ Run the Streamlit App:
streamlit run Monte_Carlo_UI.py
```
---

## 🖥️ Usage

1️⃣ **Select a crude oil stock** from the dropdown

2️⃣ **Adjust** the number of simulations and time horizon

3️⃣ **Click “Run Simulation”** to generate Monte Carlo paths

4️⃣ **View** the interactive chart and risk metrics

---

## 📌 Key Calculations

**1️⃣ Multi-Step EGARCH Volatility Forecast**

We fit an AR(1)-EGARCH(1,1) model (with skew-t innovations) to historical returns:

This yields a time-varying volatility (\sigma_t) for each forecast day via the EGARCH recursion.

The model captures leverage effects and heavier tails (skew-t) better than a basic Gaussian assumption.

**2️⃣ Monte Carlo Price Updates**

Each day’s stock price is updated using a stochastic process:

[
S_t = S_{t-1} \times \exp \Bigl((\mu - \tfrac{1}{2}\sigma_t^2),\Delta t ;+; \sigma_t \sqrt{\Delta t}\cdot Z_t\Bigr)
]

Where:

	•	( S_t ) = Simulated stock price at time ( t )

	•	( \mu ) = Annualized mean of historical returns (drift)

	•	( \sigma ) = EGARCH(1,1)-based volatility forecast for day ( t )

	•	( \Delta t ) = 1 trading day ((\tfrac{1}{252}))

	•	( Z_t ) = Random draw from a skew-t distribution

**3️⃣ Expected Price at Final Time (T)**

The expected stock price at the end of the simulation horizon is:

[
E[S_T] = \frac{1}{N} \sum_{i=1}^{N} S_{T}^{(i)}
]


Where:

	•	( E[S_T] ) = Expected final price

	•	( S_{T}^{(i)} ) = Final stock price from the ( i^{th} ) simulation

	•	( N ) = Number of simulations

**4️⃣ Value at Risk (VaR) - 5%**

Value at Risk represents the worst expected loss over a given time horizon at a 5% confidence level:

[
VaR = \text{5th percentile of simulated final prices}
]

This means there is a 5% chance that the stock price will fall below this value at the end of the simulation.

**5️⃣ Conditional Value at Risk (CVaR) - Expected Shortfall**

Conditional VaR (Expected Shortfall) estimates the average loss if the price falls below VaR:

[
CVaR = \text{Mean of all values below VaR}
]

This provides a more accurate measure of tail risk compared to standard VaR.

---

## 🏗 Future Improvements

✅ Optimize performance for larger datasets

✅ Add real-time stock data fetching from an API

✅ Implement options pricing using Monte Carlo

---

## 🤝 Contributing

1️⃣ Fork the repository

2️⃣ Create a new branch (git checkout -b feature-branch)

3️⃣ Commit changes (git commit -m "Added feature XYZ")

4️⃣ Push to GitHub (git push origin feature-branch)

5️⃣ Submit a Pull Request

---

## 📜 License

This project is open-source under the MIT License.

---

## 📬 Contact

📧 Email: [Kaylieoneal@yahoo.com]

📍 GitHub: Kaylieo
