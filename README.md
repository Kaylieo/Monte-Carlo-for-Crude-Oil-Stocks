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
│── stock_data.db         # SQLite database storing historical stock prices
│── Monte_Carlo_UI.py     # Streamlit app for user interaction
│── Fetch_Data.py         # Fetches stock data and stores in SQLite
│── requirements.txt      # Dependencies for the project
│── README.md             # Project documentation
│── .gitignore            # Files to be ignored in version control

---

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Kaylieo/Monte-Carlo-for-Crude-Oil-Stocks.git
cd Monte-Carlo-for-Crude-Oil-Stocks

**### 2️⃣ Create Virtual Enviornment**
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

**### 3️⃣ Install Dependencies**
pip install -r requirements.txt

**### 4️⃣ Run the Streamlit App**
streamlit run Monte_Carlo_UI.py

---

**## 🖥️ Usage**
1️⃣ Select a crude oil stock from the dropdown
2️⃣ Adjust the number of simulations and time horizon
3️⃣ Click “Run Simulation” to generate Monte Carlo paths
4️⃣ View the interactive chart and risk metrics

---

**## 📊 Example Simulation**
<img width="856" alt="Screenshot 2025-03-05 at 9 37 07 AM" src="https://github.com/user-attachments/assets/a66af7df-9d6c-4e4d-99bf-19702a0e8a8b" />

---

**## 📌 Key Calculations**
**1️⃣ Monte Carlo Path Generation:**

The stock price at each time step is simulated using the Geometric Brownian Motion (GBM) formula:


S_t = S_{t-1} \times e^{(\mu - 0.5\sigma^2)dt + \sigma \sqrt{dt} \cdot Z}


Where:
	•	 S_t  = Simulated stock price at time  t 
	•	 \mu  = Mean of historical daily returns
	•	 \sigma  = Standard deviation (volatility) of historical returns
	•	 dt  = Time step (1 trading day, typically  \frac{1}{252} )
	•	 Z  = Random standard normal variable
**2️⃣ Expected Price at Final Time (T)**

The expected stock price at the end of the simulation is computed as:


E[S_T] = \frac{1}{N} \sum_{i=1}^{N} S_{T}^{(i)}


Where:
	•	 E[S_T]  = Expected final price
	•	 S_{T}^{(i)}  = Final stock price from the  i^{th}  simulation
	•	 N  = Number of simulations
**3️⃣ Value at Risk (VaR) - 5%**

Value at Risk represents the worst expected loss over a given time horizon at a 5% confidence level:


VaR = \text{5th percentile of simulated final prices}


This means there is a 5% chance that the stock price will fall below this value at the end of the simulation.
**4️⃣ Conditional Value at Risk (CVaR) - Expected Shortfall**

Conditional VaR (Expected Shortfall) estimates the average loss if the price falls below VaR:


CVaR = \text{Mean of all values below VaR}


This provides a more accurate measure of tail risk compared to standard VaR.

---

**## 🏗 Future Improvements**
✅ Optimize performance for larger datasets
✅ Add real-time stock data fetching from an API
✅ Implement options pricing using Monte Carlo

---

**## 🤝 Contributing**
1️⃣	Fork the repository
2️⃣	Create a new branch (git checkout -b feature-branch)
3️⃣	Commit changes (git commit -m "Added feature XYZ")
4️⃣	Push to GitHub (git push origin feature-branch)
5️⃣	Submit a Pull Request

---

**## 📜 License**
This project is open-source under the MIT License.

---

**## 📬 Contact**
📧 Email: [Kaylieoneal@yahoo.com]
📍 GitHub: Kaylieo
