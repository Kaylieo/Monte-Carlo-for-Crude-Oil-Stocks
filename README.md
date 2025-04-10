# Monte Carlo Simulation for Crude Oil Stocks with Markov-Switching EGARCH

A sophisticated financial modeling application that combines **Markov-switching EGARCH** (MS-EGARCH) with Monte Carlo simulation to forecast crude oil stock prices. Built with Python, R, and Streamlit, featuring an elegant dark/light mode UI.

## 🔬 Technical Overview

This project implements a hybrid Python-R architecture for advanced financial modeling:

- **MS-EGARCH Model**: Captures regime-switching volatility dynamics using a 2-state Markov chain
- **Skewed Student-t Distribution**: Models heavy-tailed returns and asymmetric volatility responses
- **Sobol Sequences**: Low-discrepancy sequences for more efficient Monte Carlo sampling
- **Python-R Bridge**: Leverages `rpy2` for seamless integration with R's MSGARCH package

## ✨ Key Features

### Advanced Modeling
- 🔄 **Markov-Switching EGARCH**: Regime-switching volatility model
- 📊 **Skewed Student-t Innovations**: Better tail risk modeling
- 🎲 **Sobol Sequence Sampling**: Improved Monte Carlo efficiency
- 📈 **Multi-Day Path Generation**: Iterative simulation without refitting

### Risk Analytics
- 📉 **Value at Risk (VaR)**: 5% confidence level risk metrics
- 💹 **Conditional VaR**: Expected shortfall calculations
- 📊 **Volatility Forecasting**: EGARCH-based predictions
- 📈 **Return Distribution Analysis**: Full distribution metrics

### User Interface
- 🌓 **Dark/Light Mode**: Elegant theme switching
- 📱 **Responsive Design**: Custom CSS with modern aesthetics
- 📊 **Interactive Plots**: Real-time Plotly visualizations
- 💾 **Data Export**: Download simulation results as CSV

### Stock Coverage
- 🛢️ **Major Oil Companies**: XOM, CVX, BP, COP
- ⚡ **Refiners**: VLO, MPC, PSX
- 🔍 **E&P Companies**: EOG, OXY, HES

## 🔧 Technical Architecture

```
MonteCarlo/
├── app.py                 # Streamlit UI with dark/light mode
├── monte_carlo.py         # Core MS-EGARCH simulation engine
├── fetch_data.py         # Stock data acquisition module
├── backtest.py          # Model validation framework
├── diagnostics.py       # Model diagnostic tools
├── test_monte_carlo.py  # Unit tests
├── environment.yml      # Conda environment specification
└── install_older_packages.R  # R package dependencies
```

## 📐 Mathematical Framework

### 1. MS-EGARCH Specification
The model switches between two volatility regimes using a Markov chain:

```math
r_t = μ_t + σ_{t,S_t}ε_t
```
where S_t ∈ {1,2} is the regime state and ε_t follows a skewed Student-t distribution.

### 2. Price Path Generation
Stock prices are simulated using:
```math
S_t = S_{t-1} exp((μ - σ_t²/2)Δt + σ_t√Δt Z_t)
```
where Z_t ~ skew-t(ν,λ) and σ_t is the MS-EGARCH volatility.

## 🛠 Installation

### Prerequisites
- Python 3.12.9
- R 4.3.3
- Conda package manager

### Setup Steps
1. Clone and setup environment:
```bash
git clone https://github.com/Kaylieo/Monte-Carlo-for-Crude-Oil-Stocks.git
cd Monte-Carlo-for-Crude-Oil-Stocks
conda env create -f environment.yml
conda activate MonteCarloEnv
```

2. Install R dependencies:
```bash
Rscript install_older_packages.R
```

3. Launch application:
```bash
streamlit run app.py
```

## 🎯 Usage

1. Select a crude oil stock (e.g., XOM, CVX)
2. Configure simulation parameters:
   - Number of simulations (up to 10,000)
   - Time horizon (up to 180 days)
3. Run simulation and analyze:
   - Interactive price path visualization
   - Distribution of final prices
   - Risk metrics (VaR, CVaR)
   - Download simulation data

## 🔬 Model Validation

The MS-EGARCH model is validated through:
- Out-of-sample backtesting
- Regime classification accuracy
- Volatility forecast evaluation
- Residual diagnostics

## 📚 Dependencies

### Python Packages
- streamlit>=1.24.0
- pandas>=1.5.0
- numpy>=1.23.0
- plotly>=5.13.0
- rpy2=3.4.5
- arch=7.2.0
- scipy>=1.9.0

### R Packages
- MSGARCH
- rugarch
- parallel

## 🚀 Future Improvements

- **GPU Acceleration**: Implement CUDA support for faster Monte Carlo simulations
- **Additional Asset Classes**: Extend modeling to natural gas and renewable energy stocks
- **Advanced Regime Detection**: Implement machine learning-based regime classification
- **Real-time Data Integration**: Add live market data feeds for continuous model updates
- **Enhanced Visualization**: Add 3D surface plots for volatility term structure
- **API Development**: Create RESTful API for programmatic access to simulations

## 📬 Contact

- 📧 Email: [Kaylieoneal@yahoo.com]
- 📍 GitHub: [Kaylieo](https://github.com/Kaylieo)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
