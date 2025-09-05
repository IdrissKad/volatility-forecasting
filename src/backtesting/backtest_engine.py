import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    initial_capital: float = 1000000
    position_size: float = 0.1
    rebalance_frequency: str = 'daily'
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    use_leverage: bool = False
    max_leverage: float = 2.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class VolatilityBacktester:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config if config else BacktestConfig()
        self.results = {}
        self.trades = []
        
    def backtest_volatility_forecast(
        self,
        price_data: pd.DataFrame,
        volatility_forecasts: Dict[str, pd.Series],
        strategy_func: Callable,
        lookback_window: int = 20
    ) -> Dict:
        results = {}
        
        for model_name, forecast in volatility_forecasts.items():
            model_results = self._run_single_backtest(
                price_data,
                forecast,
                strategy_func,
                lookback_window
            )
            results[model_name] = model_results
        
        comparison = self._compare_results(results)
        
        return {
            'individual_results': results,
            'comparison': comparison
        }
    
    def _run_single_backtest(
        self,
        price_data: pd.DataFrame,
        volatility_forecast: pd.Series,
        strategy_func: Callable,
        lookback_window: int
    ) -> Dict:
        aligned_data = pd.DataFrame({
            'price': price_data['Close'],
            'returns': price_data['Returns'],
            'forecast_vol': volatility_forecast
        }).dropna()
        
        capital = self.config.initial_capital
        position = 0
        trades = []
        portfolio_value = []
        positions_history = []
        
        for i in range(lookback_window, len(aligned_data)):
            current_date = aligned_data.index[i]
            current_price = aligned_data['price'].iloc[i]
            current_vol = aligned_data['forecast_vol'].iloc[i]
            historical_vol = aligned_data['returns'].iloc[i-lookback_window:i].std() * np.sqrt(252)
            
            signal = strategy_func(
                current_vol,
                historical_vol,
                current_price,
                position
            )
            
            if signal != 0:
                trade_value = capital * self.config.position_size * signal
                shares = trade_value / current_price
                
                transaction_cost = abs(trade_value) * self.config.transaction_cost
                slippage_cost = abs(trade_value) * self.config.slippage
                
                position += shares
                capital -= trade_value + transaction_cost + slippage_cost
                
                trades.append({
                    'date': current_date,
                    'action': 'buy' if signal > 0 else 'sell',
                    'price': current_price,
                    'shares': shares,
                    'value': trade_value,
                    'transaction_cost': transaction_cost,
                    'slippage': slippage_cost,
                    'forecast_vol': current_vol,
                    'realized_vol': historical_vol
                })
            
            portfolio_val = capital + position * current_price
            portfolio_value.append({
                'date': current_date,
                'portfolio_value': portfolio_val,
                'cash': capital,
                'position_value': position * current_price,
                'position_shares': position
            })
            
            positions_history.append({
                'date': current_date,
                'position': position,
                'forecast_vol': current_vol,
                'realized_vol': historical_vol
            })
        
        portfolio_df = pd.DataFrame(portfolio_value).set_index('date')
        trades_df = pd.DataFrame(trades)
        positions_df = pd.DataFrame(positions_history).set_index('date')
        
        metrics = self._calculate_metrics(portfolio_df, trades_df)
        
        return {
            'portfolio': portfolio_df,
            'trades': trades_df,
            'positions': positions_df,
            'metrics': metrics
        }
    
    def _calculate_metrics(
        self,
        portfolio_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> Dict:
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / 
                       portfolio_df['portfolio_value'].iloc[0] - 1)
        
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        rolling_max = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        winning_trades = trades_df[trades_df['value'] > 0] if not trades_df.empty else pd.DataFrame()
        losing_trades = trades_df[trades_df['value'] < 0] if not trades_df.empty else pd.DataFrame()
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        avg_win = winning_trades['value'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['value'].mean()) if len(losing_trades) > 0 else 0
        
        profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) \
                       if len(losing_trades) > 0 and avg_loss > 0 else np.inf
        
        total_trades = len(trades_df)
        total_transaction_costs = trades_df['transaction_cost'].sum() if not trades_df.empty else 0
        total_slippage = trades_df['slippage'].sum() if not trades_df.empty else 0
        
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        sortino_ratio = np.sqrt(252) * returns.mean() / returns[returns < 0].std() \
                       if len(returns[returns < 0]) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(portfolio_df)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_transaction_costs': total_transaction_costs,
            'total_slippage': total_slippage,
            'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1],
            'volatility': returns.std() * np.sqrt(252)
        }
    
    def _compare_results(self, results: Dict) -> pd.DataFrame:
        comparison_data = []
        
        for model_name, model_results in results.items():
            metrics = model_results['metrics']
            metrics['model'] = model_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data).set_index('model')
        
        comparison_df['rank_return'] = comparison_df['total_return'].rank(ascending=False)
        comparison_df['rank_sharpe'] = comparison_df['sharpe_ratio'].rank(ascending=False)
        comparison_df['rank_drawdown'] = comparison_df['max_drawdown'].rank(ascending=False)
        
        return comparison_df
    
    def backtest_delta_hedge(
        self,
        spot_prices: pd.Series,
        volatility_forecast: pd.Series,
        strike: float,
        expiry: float,
        option_type: str = 'call'
    ) -> Dict:
        from ..strategies.delta_hedging import DeltaHedgingStrategy
        
        hedger = DeltaHedgingStrategy(
            risk_free_rate=0.05,
            transaction_cost=self.config.transaction_cost
        )
        
        hedge_results = []
        
        rolling_window = 60
        
        for i in range(rolling_window, len(spot_prices) - 20):
            window_prices = spot_prices.iloc[i:i+20]
            forecast_vol = volatility_forecast.iloc[i] if i < len(volatility_forecast) else 0.2
            
            result = hedger.simulate_delta_hedging(
                window_prices,
                strike=strike,
                expiry=expiry/252,
                volatility=forecast_vol,
                option_type=option_type
            )
            
            hedge_results.append({
                'date': spot_prices.index[i+20],
                'hedge_pnl': result['hedge_pnl'],
                'option_pnl': result['option_pnl'],
                'total_pnl': result['total_pnl'],
                'transaction_costs': result['transaction_costs'],
                'hedge_effectiveness': result['hedge_effectiveness']
            })
        
        results_df = pd.DataFrame(hedge_results).set_index('date')
        
        metrics = {
            'avg_hedge_effectiveness': results_df['hedge_effectiveness'].mean(),
            'total_pnl': results_df['total_pnl'].sum(),
            'avg_transaction_costs': results_df['transaction_costs'].mean(),
            'pnl_volatility': results_df['total_pnl'].std(),
            'sharpe_ratio': results_df['total_pnl'].mean() / results_df['total_pnl'].std() \
                          if results_df['total_pnl'].std() > 0 else 0
        }
        
        return {
            'results': results_df,
            'metrics': metrics
        }


def volatility_trading_strategy(
    forecast_vol: float,
    realized_vol: float,
    current_price: float,
    current_position: float
) -> float:
    vol_spread = forecast_vol - realized_vol
    threshold = 0.02
    
    if vol_spread > threshold:
        return 1
    elif vol_spread < -threshold:
        return -1
    else:
        return 0


def mean_reversion_strategy(
    forecast_vol: float,
    realized_vol: float,
    current_price: float,
    current_position: float
) -> float:
    vol_ratio = forecast_vol / realized_vol if realized_vol > 0 else 1
    
    if vol_ratio > 1.2:
        return -1
    elif vol_ratio < 0.8:
        return 1
    else:
        return 0


def momentum_strategy(
    forecast_vol: float,
    realized_vol: float,
    current_price: float,
    current_position: float
) -> float:
    vol_change = (forecast_vol - realized_vol) / realized_vol if realized_vol > 0 else 0
    
    if vol_change > 0.1:
        return 1
    elif vol_change < -0.1:
        return -1
    else:
        return 0