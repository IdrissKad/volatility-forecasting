# Volatility Forecasting & Delta-Hedging Strategies

A comprehensive Python framework for volatility forecasting using GARCH models, realized volatility, and implied volatility, with backtesting capabilities for delta-hedged strategies.

## Features

### Volatility Models
- **GARCH Family**: Standard GARCH(1,1), EGARCH, GJR-GARCH
- **Realized Volatility**: Simple, EWMA, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
- **Implied Volatility**: Black-Scholes IV extraction, volatility surface, smile analysis

### Trading Strategies
- **Delta Hedging**: Dynamic delta-neutral portfolio management
- **Gamma Hedging**: Second-order risk management
- **Vega Hedging**: Volatility risk neutralization
- **Volatility Trading**: Directional and relative value strategies

### Backtesting Framework
- Comprehensive performance metrics (Sharpe, Sortino, Calmar ratios)
- Transaction costs and slippage modeling
- Multi-strategy comparison
- Risk analytics and drawdown analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/volatility-forecasting.git
cd volatility-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.data import DataLoader
from src.models import GARCHModel, RealizedVolatility
from src.strategies import DeltaHedgingStrategy
from src.backtesting import VolatilityBacktester

# Load data
loader = DataLoader()
data = loader.fetch_stock_data("SPY", "2023-01-01", "2024-01-01")

# Calculate realized volatility
rv_calc = RealizedVolatility()
data['RV'] = rv_calc.simple_realized_vol(data['Log_Returns'])

# Fit GARCH model
garch = GARCHModel(p=1, q=1)
garch.fit(data['Log_Returns'].dropna())
forecast = garch.forecast(horizon=5)

# Run delta hedging simulation
hedger = DeltaHedgingStrategy()
results = hedger.simulate_delta_hedging(
    data['Close'],
    strike=400,
    expiry=0.25,
    volatility=0.20
)
```

## Project Structure

```
volatility-forecasting/
│
├── src/
│   ├── data/              # Data loading and processing
│   ├── models/             # Volatility models (GARCH, RV, IV)
│   ├── strategies/         # Trading strategies
│   ├── backtesting/        # Backtesting engine
│   └── utils/              # Visualization and utilities
│
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
├── data/                   # Data storage
│   ├── raw/               # Raw market data
│   └── processed/         # Processed features
│
├── results/                # Output directory
│   ├── reports/           # Performance reports
│   └── plots/             # Visualizations
│
├── config/                 # Configuration files
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Models

### GARCH Models
```python
from src.models import GARCHModel, EGARCHModel

# Standard GARCH(1,1)
garch = GARCHModel(p=1, q=1, dist='normal')
garch.fit(returns)

# EGARCH for asymmetric effects
egarch = EGARCHModel(p=1, o=1, q=1)
egarch.fit(returns)
```

### Realized Volatility
```python
from src.models import RealizedVolatility

rv = RealizedVolatility()
simple_rv = rv.simple_realized_vol(returns, window=20)
parkinson = rv.parkinson_vol(high, low, window=20)
garman_klass = rv.garman_klass_vol(open, high, low, close, window=20)
```

### Implied Volatility
```python
from src.models import ImpliedVolatility

iv = ImpliedVolatility()
implied_vol = iv.implied_vol_newton_raphson(
    market_price=10,
    S=100, K=105, T=0.25, r=0.05
)
```

## Backtesting

```python
from src.backtesting import VolatilityBacktester, BacktestConfig

config = BacktestConfig(
    initial_capital=100000,
    position_size=0.1,
    transaction_cost=0.001
)

backtester = VolatilityBacktester(config)
results = backtester.backtest_volatility_forecast(
    price_data,
    volatility_forecasts,
    strategy_function
)
```

## Performance Metrics

The framework calculates comprehensive performance metrics:
- **Returns**: Total, annualized, risk-adjusted
- **Risk**: Volatility, VaR, CVaR, max drawdown
- **Ratios**: Sharpe, Sortino, Calmar, Information
- **Trading**: Win rate, profit factor, average trade
- **Costs**: Transaction costs, slippage impact

## Visualization

```python
from src.utils import VolatilityVisualizer

viz = VolatilityVisualizer()

# Volatility comparison
viz.plot_volatility_comparison(data, models=['GARCH', 'EWMA'])

# Volatility surface
viz.plot_volatility_surface(strikes, expiries, implied_vols)

# Backtest results
viz.plot_backtest_results(results)
```

## Running the Example

```bash
python example_usage.py
```

This will:
1. Download SPY data
2. Calculate various volatility measures
3. Fit GARCH models
4. Extract implied volatility
5. Backtest trading strategies
6. Run delta-hedging simulations
7. Generate comprehensive reports

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- arch (GARCH models)
- yfinance (data fetching)
- plotly (visualization)
- See `requirements.txt` for full list

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:
```
@software{volatility_forecasting,
  title={Volatility Forecasting and Delta-Hedging Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/volatility-forecasting}
}
```