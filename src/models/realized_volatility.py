import numpy as np
import pandas as pd
from typing import Optional, Dict, Union
from scipy import stats


class RealizedVolatility:
    def __init__(self, annualization_factor: int = 252):
        self.annualization_factor = annualization_factor
    
    def simple_realized_vol(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        return returns.rolling(window=window).std() * np.sqrt(self.annualization_factor)
    
    def exponentially_weighted_vol(
        self,
        returns: pd.Series,
        alpha: float = 0.94
    ) -> pd.Series:
        return returns.ewm(alpha=1-alpha, adjust=False).std() * np.sqrt(self.annualization_factor)
    
    def parkinson_vol(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 20
    ) -> pd.Series:
        hl_ratio = np.log(high / low)
        factor = 1 / (4 * np.log(2))
        
        return np.sqrt(
            (hl_ratio ** 2).rolling(window=window).mean() * factor * self.annualization_factor
        )
    
    def garman_klass_vol(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        hl_ratio = np.log(high / low)
        co_ratio = np.log(close / open_price)
        
        hl_component = 0.5 * (hl_ratio ** 2)
        co_component = (2 * np.log(2) - 1) * (co_ratio ** 2)
        
        gk_var = (hl_component - co_component).rolling(window=window).mean()
        
        return np.sqrt(gk_var * self.annualization_factor)
    
    def rogers_satchell_vol(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        hc_ratio = np.log(high / close)
        ho_ratio = np.log(high / open_price)
        lc_ratio = np.log(low / close)
        lo_ratio = np.log(low / open_price)
        
        rs_component = hc_ratio * ho_ratio + lc_ratio * lo_ratio
        
        return np.sqrt(
            rs_component.rolling(window=window).mean() * self.annualization_factor
        )
    
    def yang_zhang_vol(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        k: Optional[float] = None
    ) -> pd.Series:
        if k is None:
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        overnight_var = ((np.log(open_price / close.shift(1))) ** 2).rolling(window=window).mean()
        
        open_to_close_var = ((np.log(close / open_price)) ** 2).rolling(window=window).mean()
        
        rs_vol = self.rogers_satchell_vol(open_price, high, low, close, window)
        rs_var = rs_vol ** 2 / self.annualization_factor
        
        yz_var = overnight_var + k * open_to_close_var + (1 - k) * rs_var
        
        return np.sqrt(yz_var * self.annualization_factor)
    
    def intraday_vol(
        self,
        prices: pd.Series,
        frequency: str = '5min',
        method: str = 'simple'
    ) -> float:
        returns = prices.pct_change().dropna()
        
        if method == 'simple':
            n_periods = self._get_periods_per_day(frequency)
            return returns.std() * np.sqrt(n_periods * self.annualization_factor)
        
        elif method == 'realized_variance':
            squared_returns = returns ** 2
            realized_var = squared_returns.sum()
            n_periods = self._get_periods_per_day(frequency)
            return np.sqrt(realized_var * n_periods * self.annualization_factor)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def bipower_variation(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        abs_returns = np.abs(returns)
        mu_1 = np.sqrt(2 / np.pi)
        
        bv = (np.pi / 2) * (abs_returns * abs_returns.shift(1)).rolling(window=window).sum()
        
        return np.sqrt(bv * self.annualization_factor / window)
    
    def jump_robust_vol(
        self,
        returns: pd.Series,
        window: int = 20,
        alpha: float = 0.05
    ) -> Dict[str, pd.Series]:
        rv = (returns ** 2).rolling(window=window).sum()
        
        bv = self.bipower_variation(returns, window) ** 2 * window / self.annualization_factor
        
        tri_power = self._tri_power_quarticity(returns, window)
        
        z_stat = np.sqrt(window) * (rv - bv) / np.sqrt(tri_power)
        
        critical_value = stats.norm.ppf(1 - alpha)
        jump_indicator = z_stat > critical_value
        
        continuous_vol = np.sqrt(np.where(jump_indicator, bv, rv) * self.annualization_factor / window)
        jump_vol = np.sqrt(np.maximum(rv - bv, 0) * self.annualization_factor / window)
        
        return {
            'continuous_vol': continuous_vol,
            'jump_vol': jump_vol,
            'jump_indicator': jump_indicator
        }
    
    def _tri_power_quarticity(
        self,
        returns: pd.Series,
        window: int
    ) -> pd.Series:
        abs_returns = np.abs(returns)
        mu_43 = 2 ** (2/3) * stats.gamma(7/6) / stats.gamma(1/2)
        
        tpq = window * mu_43 ** (-3) * (
            (abs_returns ** (4/3)) * 
            (abs_returns.shift(1) ** (4/3)) * 
            (abs_returns.shift(2) ** (4/3))
        ).rolling(window=window).sum()
        
        return tpq
    
    def _get_periods_per_day(self, frequency: str) -> int:
        freq_map = {
            '1min': 390,
            '5min': 78,
            '10min': 39,
            '15min': 26,
            '30min': 13,
            '1h': 6.5,
            '60min': 6.5
        }
        
        return freq_map.get(frequency, 78)