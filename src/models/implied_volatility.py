import numpy as np
import pandas as pd
from scipy import interpolate, optimize
from scipy.stats import norm
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ImpliedVolatility:
    def __init__(self):
        self.risk_free_rate = 0.05
        
    def black_scholes_price(
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
    
    def vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)
    
    def implied_vol_newton_raphson(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        initial_guess: float = 0.2,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        sigma = initial_guess
        
        for i in range(max_iterations):
            model_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            vega_val = self.vega(S, K, T, r, sigma)
            
            if abs(vega_val) < 1e-10:
                return None
            
            price_diff = market_price - model_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            sigma = sigma + price_diff / vega_val
            
            if sigma <= 0:
                sigma = 0.001
            elif sigma > 5:
                sigma = 5
        
        return None
    
    def implied_vol_bisection(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        lower_bound: float = 0.001,
        upper_bound: float = 5.0,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        def objective(sigma):
            return self.black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        
        try:
            result = optimize.brentq(objective, lower_bound, upper_bound, xtol=tolerance)
            return result
        except ValueError:
            return None
    
    def extract_iv_from_options_chain(
        self,
        options_df: pd.DataFrame,
        spot_price: float,
        time_to_expiry: float,
        risk_free_rate: Optional[float] = None
    ) -> pd.DataFrame:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        options_df = options_df.copy()
        
        options_df['impliedVolatility_calculated'] = np.nan
        
        for idx, row in options_df.iterrows():
            if 'lastPrice' in row and row['lastPrice'] > 0:
                iv = self.implied_vol_newton_raphson(
                    market_price=row['lastPrice'],
                    S=spot_price,
                    K=row['strike'],
                    T=time_to_expiry,
                    r=risk_free_rate,
                    option_type=row.get('type', 'call')
                )
                options_df.at[idx, 'impliedVolatility_calculated'] = iv
        
        options_df['moneyness'] = options_df['strike'] / spot_price
        
        return options_df
    
    def volatility_surface(
        self,
        options_data: pd.DataFrame,
        spot_price: float,
        expiry_dates: List[float],
        method: str = 'cubic'
    ) -> Dict:
        surface_data = []
        
        for expiry in expiry_dates:
            expiry_options = options_data[options_data['expiry'] == expiry]
            
            valid_data = expiry_options.dropna(subset=['impliedVolatility_calculated'])
            
            if len(valid_data) > 3:
                strikes = valid_data['strike'].values
                ivs = valid_data['impliedVolatility_calculated'].values
                
                f = interpolate.interp1d(
                    strikes, 
                    ivs, 
                    kind=method, 
                    fill_value='extrapolate',
                    bounds_error=False
                )
                
                strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
                iv_grid = f(strike_grid)
                
                for strike, iv in zip(strike_grid, iv_grid):
                    surface_data.append({
                        'strike': strike,
                        'expiry': expiry,
                        'implied_vol': iv,
                        'moneyness': strike / spot_price
                    })
        
        return pd.DataFrame(surface_data)
    
    def volatility_smile(
        self,
        options_df: pd.DataFrame,
        expiry: float,
        spot_price: float
    ) -> pd.DataFrame:
        expiry_options = options_df[options_df['expiry'] == expiry].copy()
        
        expiry_options = expiry_options.sort_values('strike')
        
        smile_data = expiry_options[['strike', 'impliedVolatility_calculated']].copy()
        smile_data['moneyness'] = smile_data['strike'] / spot_price
        
        return smile_data.dropna()
    
    def term_structure(
        self,
        options_df: pd.DataFrame,
        strike_type: str = 'atm'
    ) -> pd.DataFrame:
        term_data = []
        
        for expiry in options_df['expiry'].unique():
            expiry_options = options_df[options_df['expiry'] == expiry]
            
            if strike_type == 'atm':
                atm_options = expiry_options[
                    abs(expiry_options['moneyness'] - 1.0) < 0.05
                ]
                if not atm_options.empty:
                    avg_iv = atm_options['impliedVolatility_calculated'].mean()
                    term_data.append({
                        'expiry': expiry,
                        'implied_vol': avg_iv
                    })
        
        return pd.DataFrame(term_data).sort_values('expiry')
    
    def vix_approximation(
        self,
        options_df: pd.DataFrame,
        spot_price: float,
        time_to_expiry: float = 30/365,
        risk_free_rate: Optional[float] = None
    ) -> float:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        F = spot_price * np.exp(risk_free_rate * time_to_expiry)
        
        K0 = options_df.iloc[(options_df['strike'] - F).abs().argsort()[:1]]['strike'].values[0]
        
        valid_options = options_df[
            (options_df['strike'] > 0) & 
            (options_df['lastPrice'] > 0)
        ].copy()
        
        valid_options = valid_options.sort_values('strike')
        
        variance_sum = 0
        
        for i, row in valid_options.iterrows():
            K = row['strike']
            
            if i == 0:
                delta_K = valid_options.iloc[1]['strike'] - K
            elif i == len(valid_options) - 1:
                delta_K = K - valid_options.iloc[-2]['strike']
            else:
                delta_K = (valid_options.iloc[i+1]['strike'] - valid_options.iloc[i-1]['strike']) / 2
            
            if K < K0:
                Q = row['lastPrice'] if row['type'] == 'put' else row['lastPrice']
            elif K > K0:
                Q = row['lastPrice'] if row['type'] == 'call' else row['lastPrice']
            else:
                Q = (row['lastPrice'] + row['lastPrice']) / 2
            
            variance_sum += (delta_K / (K ** 2)) * np.exp(risk_free_rate * time_to_expiry) * Q
        
        variance = (2 / time_to_expiry) * variance_sum - (1 / time_to_expiry) * ((F / K0 - 1) ** 2)
        
        vix = 100 * np.sqrt(variance)
        
        return vix