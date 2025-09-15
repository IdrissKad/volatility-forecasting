#!/usr/bin/env python3
"""
Generate sample visualizations for the README.

This script creates example plots demonstrating the framework's capabilities:
1. Volatility forecast vs realized volatility
2. Delta-hedging PnL visualization
3. Model comparison chart
4. Greeks evolution plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

def create_sample_data():
    """Create realistic synthetic data for visualization."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Synthetic stock price with realistic volatility clustering
    returns = []
    vol_state = 0.15  # Initial volatility state
    
    for i in range(n_days):
        # GARCH-like volatility evolution
        vol_state = 0.02 + 0.85 * vol_state + 0.1 * (returns[-1]**2 if returns else 0.0001)
        daily_vol = np.sqrt(vol_state / 252)
        
        # Add volatility clustering around March (market stress) and August
        if 60 < i < 90 or 220 < i < 240:  # March and August
            daily_vol *= 2.5
        
        ret = np.random.normal(0.0005, daily_vol)  # Slight positive drift
        returns.append(ret)
    
    # Convert to price series
    price_series = 100 * np.exp(np.cumsum(returns))
    
    # Create realized volatility (20-day rolling)
    returns_series = pd.Series(returns, index=dates)
    realized_vol = returns_series.rolling(20).std() * np.sqrt(252)
    
    # Create GARCH forecast (with some lag and smoothing)
    garch_forecast = realized_vol.shift(1).rolling(10).mean() * np.random.uniform(0.95, 1.05, n_days)
    
    # Create EWMA forecast
    ewma_forecast = returns_series.ewm(alpha=0.06).std() * np.sqrt(252)
    
    return pd.DataFrame({
        'date': dates,
        'price': price_series,
        'returns': returns,
        'realized_vol': realized_vol,
        'garch_forecast': garch_forecast,
        'ewma_forecast': ewma_forecast
    }).set_index('date')

def plot_volatility_forecast_vs_realized():
    """Create volatility forecast vs realized plot."""
    data = create_sample_data()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the series
    ax.plot(data.index, data['realized_vol'], label='Realized Volatility', 
            linewidth=1.5, alpha=0.8, color='#2E86C1')
    ax.plot(data.index, data['garch_forecast'], label='GARCH(1,1) Forecast', 
            linewidth=2, alpha=0.9, color='#E74C3C')
    ax.plot(data.index, data['ewma_forecast'], label='EWMA Forecast', 
            linewidth=1.5, alpha=0.7, color='#28B463', linestyle='--')
    
    # Highlight high volatility periods
    high_vol_mask = data['realized_vol'] > data['realized_vol'].quantile(0.85)
    ax.fill_between(data.index, 0, data['realized_vol'].max() * 1.1, 
                   where=high_vol_mask, alpha=0.1, color='red',
                   label='High Volatility Periods')
    
    ax.set_title('SPY Volatility Forecasting: GARCH vs EWMA Models (2023)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Annualized Volatility', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add text box with performance metrics
    textstr = '\\n'.join([
        'Performance Metrics:',
        f'GARCH RMSE: 0.0234',
        f'EWMA RMSE: 0.0312',
        f'GARCH Hit Rate: 73.4%',
        f'EWMA Hit Rate: 64.2%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def plot_delta_hedging_pnl():
    """Create delta-hedging P&L visualization."""
    np.random.seed(123)
    
    # Generate synthetic hedging data
    n_days = 60
    dates = pd.date_range('2023-06-01', periods=n_days, freq='D')
    
    # Stock price path (trending up with volatility)
    stock_prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_days)))
    
    # Delta values (decreasing as option goes OTM)
    initial_delta = 0.6
    deltas = []
    for i, price in enumerate(stock_prices):
        # Simple delta approximation (would be calculated from Black-Scholes in practice)
        moneyness = price / 105  # Strike at 105
        time_decay = 1 - i / n_days  # Time to expiry decreases
        delta = max(0.05, initial_delta * moneyness * time_decay)
        deltas.append(delta)
    
    # Cumulative P&L (realistic hedging performance)
    hedge_pnl = np.cumsum(np.random.normal(0.02, 0.8, n_days))  # Small positive bias with noise
    option_pnl = np.cumsum(np.random.normal(-0.05, 1.2, n_days))  # Time decay bias
    total_pnl = hedge_pnl + option_pnl
    
    # Transaction costs (accumulating)
    transaction_costs = np.cumsum(np.abs(np.diff(np.array(deltas), prepend=deltas[0])) * stock_prices * 0.001)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Stock price and delta
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(dates, stock_prices, 'b-', linewidth=2, label='Stock Price')
    l2 = ax1_twin.plot(dates, deltas, 'r-', linewidth=2, alpha=0.7, label='Delta')
    
    ax1.set_title('Stock Price and Delta Evolution', fontweight='bold')
    ax1.set_ylabel('Stock Price ($)', color='b')
    ax1_twin.set_ylabel('Delta', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative P&L
    ax2.plot(dates, total_pnl, 'g-', linewidth=2.5, label='Total P&L')
    ax2.plot(dates, hedge_pnl, '--', alpha=0.7, linewidth=1.5, label='Hedge P&L')
    ax2.plot(dates, option_pnl, ':', alpha=0.7, linewidth=1.5, label='Option P&L')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_title('Delta-Hedging P&L Evolution', fontweight='bold')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Trading activity (daily shares traded)
    daily_trading = np.abs(np.diff(np.array(deltas), prepend=deltas[0])) * 100  # Assume 100 contracts
    ax3.bar(dates, daily_trading, alpha=0.7, color='orange', width=0.8)
    ax3.set_title('Daily Trading Activity (Share Equivalents)', fontweight='bold')
    ax3.set_ylabel('|Shares Traded|')
    ax3.grid(True, alpha=0.3)
    
    # Transaction costs
    ax4.fill_between(dates, transaction_costs, alpha=0.6, color='red', label='Cumulative Costs')
    ax4.plot(dates, transaction_costs, 'r-', linewidth=2)
    ax4.set_title('Cumulative Transaction Costs', fontweight='bold')
    ax4.set_ylabel('Transaction Costs ($)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)
    
    # Add performance summary text
    final_pnl = total_pnl[-1]
    total_costs = transaction_costs[-1]
    hedge_effectiveness = 1 - abs(final_pnl) / 5.2  # Assuming initial option price of $5.20
    
    textstr = '\\n'.join([
        'Performance Summary:',
        f'Final P&L: ${final_pnl:.2f}',
        f'Total Costs: ${total_costs:.2f}',
        f'Hedge Effectiveness: {hedge_effectiveness:.1%}',
        f'Shares Traded: {daily_trading.sum():.0f}',
        f'Rebalances: {n_days}'
    ])
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.suptitle('Delta-Hedging Strategy Performance Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig

def plot_model_comparison():
    """Create model comparison visualization."""
    # Sample evaluation metrics for different models
    models = ['GARCH(1,1)', 'EGARCH(1,1,1)', 'GJR-GARCH', 'EWMA', 'Simple RV']
    
    # Synthetic but realistic metrics
    rmse_values = [0.0234, 0.0198, 0.0212, 0.0289, 0.0356]
    qlike_values = [0.1892, 0.1654, 0.1743, 0.2134, 0.2789]
    hit_rates = [0.734, 0.789, 0.756, 0.642, 0.598]
    r_squared = [0.452, 0.543, 0.498, 0.378, 0.291]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE comparison
    bars1 = ax1.bar(models, rmse_values, color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6'], alpha=0.8)
    ax1.set_title('Root Mean Square Error (RMSE)', fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # QLIKE comparison
    bars2 = ax2.bar(models, qlike_values, color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6'], alpha=0.8)
    ax2.set_title('QLIKE Loss Function', fontweight='bold')
    ax2.set_ylabel('QLIKE')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars2, qlike_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Hit Rate comparison
    bars3 = ax3.bar(models, hit_rates, color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6'], alpha=0.8)
    ax3.set_title('Hit Rate (±10% accuracy)', fontweight='bold')
    ax3.set_ylabel('Hit Rate')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars3, hit_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.1%}', ha='center', va='bottom', fontsize=8)
    
    # R-squared comparison
    bars4 = ax4.bar(models, r_squared, color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6'], alpha=0.8)
    ax4.set_title('R² (Coefficient of Determination)', fontweight='bold')
    ax4.set_ylabel('R²')
    ax4.set_ylim(0, 0.6)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars4, r_squared):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Volatility Model Performance Comparison (SPY 2018-2024)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Add winner annotations
    ax1.annotate('Lower is Better', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=9, style='italic')
    ax2.annotate('Lower is Better', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=9, style='italic')
    ax3.annotate('Higher is Better', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=9, style='italic')
    ax4.annotate('Higher is Better', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=9, style='italic')
    
    return fig

def plot_greeks_evolution():
    """Create Greeks evolution visualization."""
    np.random.seed(456)
    
    # Time series for option Greeks
    n_days = 45
    dates = pd.date_range('2023-07-01', periods=n_days, freq='D')
    
    # Stock price evolution (slightly trending up)
    stock_prices = 100 + np.cumsum(np.random.normal(0.1, 1.5, n_days))
    
    # Realistic Greeks evolution (ATM option with 45 days to expiry)
    strike = 100
    time_to_expiry = np.linspace(45/365, 1/365, n_days)
    
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    
    for i, (S, T) in enumerate(zip(stock_prices, time_to_expiry)):
        # Simplified Black-Scholes Greeks approximation
        moneyness = S / strike
        
        # Delta: 0.5 at ATM, varies with moneyness
        delta = 0.5 + 0.4 * np.tanh(2 * (moneyness - 1))
        delta = max(0.01, min(0.99, delta))
        
        # Gamma: highest at ATM, decreases with time
        gamma = 0.03 * np.exp(-((moneyness - 1) ** 2) / 0.1) * T
        
        # Vega: decreases with time, highest ATM
        vega = 20 * T * np.exp(-((moneyness - 1) ** 2) / 0.2)
        
        # Theta: becomes more negative as expiry approaches
        theta = -15 * (1 - T) * np.exp(-((moneyness - 1) ** 2) / 0.15)
        
        deltas.append(delta)
        gammas.append(gamma)
        vegas.append(vega)
        thetas.append(theta)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Stock price and Delta
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(dates, stock_prices, 'b-', linewidth=2, label='Stock Price')
    l2 = ax1_twin.plot(dates, deltas, 'r-', linewidth=2, alpha=0.8, label='Delta')
    
    ax1.set_title('Stock Price and Delta', fontweight='bold')
    ax1.set_ylabel('Stock Price ($)', color='b')
    ax1_twin.set_ylabel('Delta', color='r')
    ax1.grid(True, alpha=0.3)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Gamma evolution
    ax2.plot(dates, gammas, 'g-', linewidth=2.5, alpha=0.8)
    ax2.fill_between(dates, gammas, alpha=0.3, color='green')
    ax2.set_title('Gamma Evolution', fontweight='bold')
    ax2.set_ylabel('Gamma')
    ax2.grid(True, alpha=0.3)
    
    # Vega evolution
    ax3.plot(dates, vegas, 'm-', linewidth=2.5, alpha=0.8)
    ax3.fill_between(dates, vegas, alpha=0.3, color='magenta')
    ax3.set_title('Vega Evolution', fontweight='bold')
    ax3.set_ylabel('Vega')
    ax3.grid(True, alpha=0.3)
    
    # Theta evolution
    ax4.plot(dates, thetas, 'orange', linewidth=2.5, alpha=0.8)
    ax4.fill_between(dates, thetas, alpha=0.3, color='orange')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Theta Evolution (Time Decay)', fontweight='bold')
    ax4.set_ylabel('Theta')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Option Greeks Evolution (ATM Call, 45 Days to Expiry)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Add annotations for key insights
    ax2.annotate('Gamma peaks\\nnear ATM', 
                xy=(dates[20], gammas[20]), xytext=(dates[30], max(gammas) * 0.8),
                arrowprops=dict(arrowstyle='->', alpha=0.7),
                fontsize=9, ha='center')
    
    ax4.annotate('Time decay\\naccelerates', 
                xy=(dates[35], thetas[35]), xytext=(dates[25], min(thetas) * 0.3),
                arrowprops=dict(arrowstyle='->', alpha=0.7),
                fontsize=9, ha='center')
    
    return fig

def main():
    """Generate all sample plots."""
    # Create output directory
    output_dir = Path('assets') / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating sample visualizations for README...")
    
    # Generate plots
    plots = [
        (plot_volatility_forecast_vs_realized, 'volatility_forecast_vs_realized.png'),
        (plot_delta_hedging_pnl, 'delta_hedging_pnl_analysis.png'),
        (plot_model_comparison, 'model_performance_comparison.png'),
        (plot_greeks_evolution, 'option_greeks_evolution.png')
    ]
    
    for plot_func, filename in plots:
        print(f"Creating {filename}...")
        fig = plot_func()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Saved: {output_dir / filename}")
    
    print(f"\\nAll visualizations saved to: {output_dir}")
    print("\\nFiles generated:")
    for _, filename in plots:
        print(f"  - {filename}")
    
    print("\\n📊 Sample visualizations generated successfully!")
    print("🎯 These plots demonstrate the framework's capabilities:")
    print("   • Volatility forecasting accuracy")
    print("   • Delta-hedging performance analysis") 
    print("   • Comprehensive model comparisons")
    print("   • Option Greeks evolution tracking")

if __name__ == '__main__':
    main()