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
├── backtest.py           # Model validation framework
├── diagnostics.py        # Model diagnostic tools
├── test_monte_carlo.py   # Unit tests
├── conda-lock.yml        # Locked environment for exact reproducibility
├── requirements.txt      # Pip dependencies
├── install_older_packages.R  # Manual R package installation
└── screenshots/          # UI preview images for README
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
S_t = S_{t-1} \exp\left((μ - \frac{σ_t²}{2})Δt + σ_t\sqrt{Δt} Z_t\right)
```
where Z_t ~ skew-t(ν,λ) and σ_t is the MS-EGARCH volatility.

## 🛠️ Installation (Exact Reproducibility)

### Prerequisites
- Python 3.12.9
- R 4.3.3
- Conda package manager

### Setup Instructions

1. **Fork and Clone the Repository**
   - Go to [this repository](https://github.com/Kaylieo/Monte-Carlo-for-Crude-Oil-Stocks)
   - Click **Fork** in the upper-right corner
   - Then, clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/Monte-Carlo-for-Crude-Oil-Stocks.git
cd Monte-Carlo-for-Crude-Oil-Stocks
```

2. **Create and activate the environment using the lock file**
```bash
# For macOS Apple Silicon (M1/M2/M3):
conda-lock install --name msgarch_env --file conda-osx-arm64.lock

# Or for other platforms (replace accordingly):
# conda-lock install --name msgarch_env --file conda-linux-64.lock
# conda-lock install --name msgarch_env --file conda-win-64.lock

conda activate msgarch_env
```

3. **Install pip dependencies**
```bash
pip install -r requirements.txt
```

4. **Install MSGARCH using the provided script**
```bash
Rscript install_older_packages.R
```

✅ This installs MSGARCH v2.50

5. **Launch the app**
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

## 📸 Preview

| Interface | Simulation Graph | Histogram | Summary + Historical Prices |
|----------|------------------|-----------|-----------------------------|
| ![](screenshots/Screenshot%202025-04-23%20at%205.27.30%E2%80%AFPM.png) | ![](screenshots/Screenshot%202025-04-23%20at%205.27.55%E2%80%AFPM.png) | ![](screenshots/Screenshot%202025-04-23%20at%205.29.39%E2%80%AFPM.png) | ![](screenshots/Screenshot%202025-04-23%20at%205.30.22%E2%80%AFPM.png) |

## 🔬 Model Validation

The MS-EGARCH model is validated through:
- Out-of-sample backtesting
- Regime classification accuracy
- Volatility forecast evaluation
- Residual diagnostics

## 📚 Dependencies

### Python Packages
See `requirements.txt` for pip-based packages.

### R Packages
Installed via Conda (except MSGARCH):
- r-base=4.3.3
- r-mass=7.3_60.0.1
- r-matrix=1.6_5
- r-expm=1.0_0
- r-codetools

Manually install:
- MSGARCH==2.50 (via `install_older_packages.R`)

> 🔒 Dependency management uses platform-specific `conda-lock` files. Always use `conda-lock install` to ensure all pip and R dependencies (like Tornado ≥6.5.0 and MSGARCH) are included.

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
