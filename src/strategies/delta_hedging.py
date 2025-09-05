import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DeltaHedgingStrategy:
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        transaction_cost: float = 0.001,
        rebalance_frequency: str = 'daily'
    ):
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        self.positions_history = []
        self.pnl_history = []
        
    def black_scholes_delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def black_scholes_gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def black_scholes_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    def black_scholes_theta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (term1 + term2) / 365
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365
        
        return theta
    
    def simulate_delta_hedging(
        self,
        spot_prices: pd.Series,
        strike: float,
        expiry: float,
        volatility: float,
        option_type: str = 'call',
        initial_option_price: Optional[float] = None
    ) -> Dict:
        n_periods = len(spot_prices)
        dt = expiry / n_periods
        
        hedge_positions = []
        hedge_costs = []
        cumulative_pnl = []
        transaction_costs_total = 0
        
        time_to_expiry = expiry
        prev_delta = 0
        cumulative_hedge_pnl = 0
        
        if initial_option_price is None:
            initial_option_price = self._black_scholes_price(
                spot_prices.iloc[0], strike, expiry, self.risk_free_rate, 
                volatility, option_type
            )
        
        for i, (date, S) in enumerate(spot_prices.items()):
            time_to_expiry = expiry - i * dt
            
            if time_to_expiry <= 0:
                break
            
            delta = self.black_scholes_delta(
                S, strike, time_to_expiry, self.risk_free_rate, 
                volatility, option_type
            )
            
            gamma = self.black_scholes_gamma(
                S, strike, time_to_expiry, self.risk_free_rate, volatility
            )
            
            vega = self.black_scholes_vega(
                S, strike, time_to_expiry, self.risk_free_rate, volatility
            )
            
            theta = self.black_scholes_theta(
                S, strike, time_to_expiry, self.risk_free_rate, 
                volatility, option_type
            )
            
            shares_to_trade = delta - prev_delta
            
            hedge_cost = -shares_to_trade * S
            transaction_cost = abs(shares_to_trade * S * self.transaction_cost)
            transaction_costs_total += transaction_cost
            
            if i > 0:
                stock_pnl = prev_delta * (S - spot_prices.iloc[i-1])
                cumulative_hedge_pnl += stock_pnl
            
            hedge_positions.append({
                'date': date,
                'spot': S,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'shares_held': delta,
                'shares_traded': shares_to_trade,
                'hedge_cost': hedge_cost,
                'transaction_cost': transaction_cost,
                'cumulative_pnl': cumulative_hedge_pnl - transaction_costs_total
            })
            
            prev_delta = delta
        
        final_spot = spot_prices.iloc[-1]
        if option_type == 'call':
            option_payoff = max(final_spot - strike, 0)
        else:
            option_payoff = max(strike - final_spot, 0)
        
        final_stock_value = prev_delta * final_spot
        
        total_hedge_cost = sum([pos['hedge_cost'] for pos in hedge_positions])
        
        hedge_pnl = cumulative_hedge_pnl - transaction_costs_total
        option_pnl = initial_option_price - option_payoff
        total_pnl = hedge_pnl + option_pnl
        
        results = {
            'positions': pd.DataFrame(hedge_positions),
            'initial_option_price': initial_option_price,
            'option_payoff': option_payoff,
            'final_stock_value': final_stock_value,
            'total_hedge_cost': total_hedge_cost,
            'transaction_costs': transaction_costs_total,
            'hedge_pnl': hedge_pnl,
            'option_pnl': option_pnl,
            'total_pnl': total_pnl,
            'hedge_effectiveness': 1 - abs(total_pnl / initial_option_price) if initial_option_price != 0 else 0
        }
        
        return results
    
    def gamma_hedging(
        self,
        spot_prices: pd.Series,
        strike_hedge: float,
        strike_gamma: float,
        expiry: float,
        volatility: float,
        option_type: str = 'call'
    ) -> Dict:
        n_periods = len(spot_prices)
        dt = expiry / n_periods
        
        positions = []
        time_to_expiry = expiry
        
        for i, (date, S) in enumerate(spot_prices.items()):
            time_to_expiry = expiry - i * dt
            
            if time_to_expiry <= 0:
                break
            
            delta_main = self.black_scholes_delta(
                S, strike_hedge, time_to_expiry, self.risk_free_rate,
                volatility, option_type
            )
            gamma_main = self.black_scholes_gamma(
                S, strike_hedge, time_to_expiry, self.risk_free_rate, volatility
            )
            
            delta_gamma_hedge = self.black_scholes_delta(
                S, strike_gamma, time_to_expiry, self.risk_free_rate,
                volatility, option_type
            )
            gamma_gamma_hedge = self.black_scholes_gamma(
                S, strike_gamma, time_to_expiry, self.risk_free_rate, volatility
            )
            
            if abs(gamma_gamma_hedge) > 1e-6:
                gamma_hedge_ratio = -gamma_main / gamma_gamma_hedge
            else:
                gamma_hedge_ratio = 0
            
            total_delta = delta_main + gamma_hedge_ratio * delta_gamma_hedge
            
            positions.append({
                'date': date,
                'spot': S,
                'delta_main': delta_main,
                'gamma_main': gamma_main,
                'gamma_hedge_ratio': gamma_hedge_ratio,
                'stock_position': -total_delta,
                'gamma_neutral': abs(gamma_main + gamma_hedge_ratio * gamma_gamma_hedge) < 1e-6
            })
        
        return {
            'positions': pd.DataFrame(positions),
            'strategy': 'gamma_neutral_hedging'
        }
    
    def vega_hedging(
        self,
        spot_prices: pd.Series,
        strikes: List[float],
        expiries: List[float],
        volatilities: List[float],
        target_vega: float = 0
    ) -> Dict:
        positions = []
        
        for date, S in spot_prices.items():
            vegas = []
            deltas = []
            
            for strike, expiry, vol in zip(strikes, expiries, volatilities):
                vega = self.black_scholes_vega(
                    S, strike, expiry, self.risk_free_rate, vol
                )
                delta = self.black_scholes_delta(
                    S, strike, expiry, self.risk_free_rate, vol, 'call'
                )
                
                vegas.append(vega)
                deltas.append(delta)
            
            A = np.array(vegas).reshape(-1, 1)
            b = np.array([target_vega])
            
            weights, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            portfolio_vega = np.sum(weights.flatten() * np.array(vegas))
            portfolio_delta = np.sum(weights.flatten() * np.array(deltas))
            
            positions.append({
                'date': date,
                'spot': S,
                'weights': weights.flatten(),
                'portfolio_vega': portfolio_vega,
                'portfolio_delta': portfolio_delta,
                'stock_hedge': -portfolio_delta
            })
        
        return {
            'positions': pd.DataFrame(positions),
            'strategy': 'vega_neutral_hedging'
        }
    
    def _black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def analyze_hedge_performance(
        self,
        results: Dict
    ) -> Dict:
        positions = results['positions']
        
        metrics = {
            'total_pnl': results['total_pnl'],
            'hedge_pnl': results['hedge_pnl'],
            'option_pnl': results['option_pnl'],
            'transaction_costs': results['transaction_costs'],
            'hedge_effectiveness': results['hedge_effectiveness'],
            'max_delta': positions['delta'].max(),
            'min_delta': positions['delta'].min(),
            'avg_delta': positions['delta'].mean(),
            'max_gamma': positions['gamma'].max(),
            'avg_gamma': positions['gamma'].mean(),
            'total_shares_traded': positions['shares_traded'].abs().sum(),
            'n_rebalances': len(positions),
            'sharpe_ratio': self._calculate_sharpe_ratio(positions['cumulative_pnl'])
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(
        self,
        pnl_series: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        if len(pnl_series) < 2:
            return 0
        
        returns = pnl_series.diff().dropna()
        
        if returns.std() == 0:
            return 0
        
        sharpe = np.sqrt(periods_per_year) * returns.mean() / returns.std()
        
        return sharpe