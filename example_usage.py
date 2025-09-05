"""
Volatility Forecasting and Delta-Hedging Strategy Example
=========================================================

This example demonstrates:
1. Loading financial data
2. Computing realized volatility measures
3. Fitting GARCH models for volatility forecasting
4. Extracting implied volatility from options
5. Backtesting volatility trading strategies
6. Implementing delta-hedging strategies
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data import DataLoader
from src.models import GARCHModel, EGARCHModel, RealizedVolatility, ImpliedVolatility
from src.strategies import DeltaHedgingStrategy
from src.backtesting import (
    VolatilityBacktester, 
    BacktestConfig,
    volatility_trading_strategy,
    mean_reversion_strategy
)
from src.utils import VolatilityVisualizer


def main():
    print("=" * 60)
    print("Volatility Forecasting and Trading System")
    print("=" * 60)
    
    # 1. Data Loading
    print("\n1. Loading Market Data...")
    loader = DataLoader()
    
    ticker = "SPY"
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    
    # Fetch daily price data
    daily_data = loader.fetch_stock_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    )
    
    print(f"   Loaded {len(daily_data)} days of data for {ticker}")
    print(f"   Date range: {daily_data.index[0]} to {daily_data.index[-1]}")
    
    # 2. Realized Volatility Calculation
    print("\n2. Computing Realized Volatility Measures...")
    rv_calculator = RealizedVolatility()
    
    # Calculate different realized volatility measures
    daily_data['Simple_RV'] = rv_calculator.simple_realized_vol(
        daily_data['Log_Returns'], window=20
    )
    
    daily_data['EWMA_RV'] = rv_calculator.exponentially_weighted_vol(
        daily_data['Log_Returns'], alpha=0.94
    )
    
    daily_data['Parkinson_RV'] = rv_calculator.parkinson_vol(
        daily_data['High'], daily_data['Low'], window=20
    )
    
    daily_data['GarmanKlass_RV'] = rv_calculator.garman_klass_vol(
        daily_data['Open'], daily_data['High'], 
        daily_data['Low'], daily_data['Close'], window=20
    )
    
    print("   Calculated: Simple, EWMA, Parkinson, Garman-Klass volatilities")
    
    # 3. GARCH Model Fitting
    print("\n3. Fitting GARCH Models...")
    
    # Standard GARCH(1,1)
    garch_model = GARCHModel(p=1, q=1, dist='normal')
    returns = daily_data['Log_Returns'].dropna()
    
    # Rolling GARCH forecast
    garch_forecast = garch_model.rolling_forecast(
        returns, 
        window_size=252,
        forecast_horizon=1
    )
    
    print(f"   GARCH(1,1) model fitted with {len(garch_forecast)} forecasts")
    
    # EGARCH model for asymmetric effects
    egarch_model = EGARCHModel(p=1, o=1, q=1)
    egarch_forecast = egarch_model.rolling_forecast(
        returns,
        window_size=252,
        forecast_horizon=1
    )
    
    print(f"   EGARCH(1,1,1) model fitted with {len(egarch_forecast)} forecasts")
    
    # 4. Options and Implied Volatility
    print("\n4. Extracting Implied Volatility...")
    
    try:
        # Fetch options data
        calls, puts = loader.fetch_options_data(ticker)
        
        if not calls.empty:
            iv_calculator = ImpliedVolatility()
            
            # Extract implied volatility
            spot_price = daily_data['Close'].iloc[-1]
            time_to_expiry = 30/365  # 30 days
            
            calls_with_iv = iv_calculator.extract_iv_from_options_chain(
                calls, spot_price, time_to_expiry
            )
            
            # Calculate VIX-like measure
            vix_approx = iv_calculator.vix_approximation(
                calls_with_iv, spot_price, time_to_expiry
            )
            
            print(f"   Implied volatility extracted from {len(calls_with_iv)} options")
            print(f"   VIX approximation: {vix_approx:.2f}")
        else:
            print("   No options data available for implied volatility")
            
    except Exception as e:
        print(f"   Could not fetch options data: {e}")
    
    # 5. Backtesting Volatility Strategies
    print("\n5. Backtesting Volatility Trading Strategies...")
    
    # Prepare forecast data
    volatility_forecasts = {
        'GARCH': garch_forecast['forecast_vol'],
        'EGARCH': egarch_forecast['forecast_vol'],
        'EWMA': daily_data['EWMA_RV']
    }
    
    # Initialize backtester
    config = BacktestConfig(
        initial_capital=100000,
        position_size=0.1,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    backtester = VolatilityBacktester(config)
    
    # Run backtest with volatility trading strategy
    backtest_results = backtester.backtest_volatility_forecast(
        daily_data,
        volatility_forecasts,
        volatility_trading_strategy,
        lookback_window=20
    )
    
    print("\n   Backtest Results Summary:")
    comparison = backtest_results['comparison']
    
    for model in comparison.index:
        print(f"\n   {model}:")
        print(f"     Total Return: {comparison.loc[model, 'total_return']:.2%}")
        print(f"     Sharpe Ratio: {comparison.loc[model, 'sharpe_ratio']:.2f}")
        print(f"     Max Drawdown: {comparison.loc[model, 'max_drawdown']:.2%}")
        print(f"     Win Rate: {comparison.loc[model, 'win_rate']:.2%}")
    
    # 6. Delta Hedging Strategy
    print("\n6. Testing Delta-Hedging Strategy...")
    
    hedger = DeltaHedgingStrategy(
        risk_free_rate=0.05,
        transaction_cost=0.001
    )
    
    # Simulate delta hedging
    strike = daily_data['Close'].mean()
    expiry = 30/252  # 30 days in years
    
    # Use a subset of data for delta hedging
    hedge_prices = daily_data['Close'].iloc[-30:]
    
    hedge_results = hedger.simulate_delta_hedging(
        hedge_prices,
        strike=strike,
        expiry=expiry,
        volatility=0.20,  # Use 20% implied vol
        option_type='call'
    )
    
    print(f"\n   Delta Hedging Results:")
    print(f"     Hedge P&L: ${hedge_results['hedge_pnl']:.2f}")
    print(f"     Option P&L: ${hedge_results['option_pnl']:.2f}")
    print(f"     Total P&L: ${hedge_results['total_pnl']:.2f}")
    print(f"     Transaction Costs: ${hedge_results['transaction_costs']:.2f}")
    print(f"     Hedge Effectiveness: {hedge_results['hedge_effectiveness']:.2%}")
    
    # 7. Visualization
    print("\n7. Generating Visualizations...")
    
    viz = VolatilityVisualizer()
    
    # Combine forecasts with realized volatility
    vol_comparison = pd.DataFrame({
        'Realized_Vol': daily_data['Simple_RV'],
        'GARCH': garch_forecast['forecast_vol'],
        'EGARCH': egarch_forecast['forecast_vol']
    }).dropna()
    
    # Create comparison plot
    fig1 = viz.plot_volatility_comparison(
        vol_comparison,
        models=['GARCH', 'EGARCH'],
        title='Volatility Forecasts vs Realized',
        save_path='results/plots/volatility_comparison.html'
    )
    
    # Plot backtest results
    fig2 = viz.plot_backtest_results(
        backtest_results,
        title='Strategy Backtest Results',
        save_path='results/plots/backtest_results.html'
    )
    
    # Plot delta hedging performance
    fig3 = viz.plot_delta_hedge_performance(
        hedge_results,
        title='Delta Hedging Performance',
        save_path='results/plots/delta_hedge.html'
    )
    
    # Plot metrics comparison
    fig4 = viz.plot_metrics_comparison(
        comparison,
        metrics=['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'],
        title='Strategy Performance Metrics',
        save_path='results/plots/metrics_comparison.html'
    )
    
    print("   Visualizations saved to results/plots/")
    
    # 8. Generate Report
    print("\n8. Generating Final Report...")
    
    viz.create_report(
        backtest_results,
        save_path='results/reports/volatility_report.html'
    )
    
    print("   Report saved to results/reports/volatility_report.html")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return {
        'data': daily_data,
        'forecasts': volatility_forecasts,
        'backtest_results': backtest_results,
        'hedge_results': hedge_results
    }


if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Run the example
    results = main()
    
    print("\nExample completed successfully!")
    print("Check the 'results' folder for outputs and visualizations.")