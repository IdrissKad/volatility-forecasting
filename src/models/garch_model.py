import numpy as np
import pandas as pd
from arch import arch_model
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class GARCHModel:
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = 'normal',
        mean_model: str = 'Constant'
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.mean_model = mean_model
        self.model = None
        self.fitted_model = None
        self.params = None
        
    def fit(
        self,
        returns: np.ndarray,
        update_freq: int = 1,
        disp: str = 'off'
    ) -> None:
        returns_scaled = returns * 100
        
        self.model = arch_model(
            returns_scaled,
            mean=self.mean_model,
            vol='GARCH',
            p=self.p,
            q=self.q,
            dist=self.dist
        )
        
        self.fitted_model = self.model.fit(
            update_freq=update_freq,
            disp=disp
        )
        
        self.params = self.fitted_model.params
        
    def forecast(
        self,
        horizon: int = 1,
        method: str = 'analytic',
        reindex: bool = True
    ) -> Dict[str, np.ndarray]:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.fitted_model.forecast(
            horizon=horizon,
            method=method,
            reindex=reindex
        )
        
        variance_forecast = forecast.variance.values[-1, :] / 10000
        volatility_forecast = np.sqrt(variance_forecast)
        
        return {
            'variance': variance_forecast,
            'volatility': volatility_forecast
        }
    
    def rolling_forecast(
        self,
        returns: pd.Series,
        window_size: int = 252,
        forecast_horizon: int = 1
    ) -> pd.DataFrame:
        n = len(returns)
        forecasts = []
        
        for i in range(window_size, n):
            train_data = returns.iloc[i-window_size:i].values
            
            try:
                self.fit(train_data, disp='off')
                forecast = self.forecast(horizon=forecast_horizon)
                
                forecasts.append({
                    'date': returns.index[i],
                    'forecast_vol': forecast['volatility'][0],
                    'forecast_var': forecast['variance'][0]
                })
            except:
                forecasts.append({
                    'date': returns.index[i],
                    'forecast_vol': np.nan,
                    'forecast_var': np.nan
                })
        
        return pd.DataFrame(forecasts).set_index('date')
    
    def get_model_info(self) -> Dict:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood,
            'params': self.params.to_dict(),
            'p': self.p,
            'q': self.q,
            'distribution': self.dist
        }
    
    def conditional_volatility(self) -> pd.Series:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.conditional_volatility / 100


class EGARCHModel(GARCHModel):
    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = 'normal'
    ):
        super().__init__(p=p, q=q, dist=dist)
        self.o = o
    
    def fit(
        self,
        returns: np.ndarray,
        update_freq: int = 1,
        disp: str = 'off'
    ) -> None:
        returns_scaled = returns * 100
        
        self.model = arch_model(
            returns_scaled,
            mean='Constant',
            vol='EGARCH',
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist
        )
        
        self.fitted_model = self.model.fit(
            update_freq=update_freq,
            disp=disp
        )
        
        self.params = self.fitted_model.params


class GJRGARCHModel(GARCHModel):
    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = 'normal'
    ):
        super().__init__(p=p, q=q, dist=dist)
        self.o = o
    
    def fit(
        self,
        returns: np.ndarray,
        update_freq: int = 1,
        disp: str = 'off'
    ) -> None:
        returns_scaled = returns * 100
        
        self.model = arch_model(
            returns_scaled,
            mean='Constant',
            vol='GARCH',
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist
        )
        
        self.fitted_model = self.model.fit(
            update_freq=update_freq,
            disp=disp
        )
        
        self.params = self.fitted_model.params