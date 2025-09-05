import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.data_cache = {}
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        
        if cache_key in self.data_cache:
            logger.info(f"Returning cached data for {ticker}")
            return self.data_cache[cache_key]
        
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        self.data_cache[cache_key] = df
        
        return df
    
    def fetch_options_data(
        self,
        ticker: str,
        expiry_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        stock = yf.Ticker(ticker)
        
        if expiry_date is None:
            expiry_dates = stock.options
            if not expiry_dates:
                raise ValueError(f"No options data available for {ticker}")
            expiry_date = expiry_dates[0]
        
        logger.info(f"Fetching options data for {ticker} with expiry {expiry_date}")
        
        opt_chain = stock.option_chain(expiry_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        calls['type'] = 'call'
        puts['type'] = 'put'
        
        return calls, puts
    
    def fetch_intraday_data(
        self,
        ticker: str,
        period: str = "1d",
        interval: str = "5m"
    ) -> pd.DataFrame:
        logger.info(f"Fetching intraday data for {ticker}")
        
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No intraday data found for ticker {ticker}")
        
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        return df
    
    def prepare_volatility_data(
        self,
        df: pd.DataFrame,
        lookback_window: int = 20
    ) -> pd.DataFrame:
        df = df.copy()
        
        df['Realized_Vol'] = df['Log_Returns'].rolling(
            window=lookback_window
        ).std() * np.sqrt(252)
        
        df['High_Low_Vol'] = (
            np.log(df['High'] / df['Low']) / (4 * np.log(2))
        ).rolling(window=lookback_window).mean() * np.sqrt(252)
        
        df['Parkinson_Vol'] = np.sqrt(
            252 / lookback_window * 
            (np.log(df['High'] / df['Low']) ** 2).rolling(window=lookback_window).sum()
        )
        
        df['Garman_Klass_Vol'] = np.sqrt(
            252 / lookback_window * (
                0.5 * (np.log(df['High'] / df['Low']) ** 2).rolling(window=lookback_window).sum() -
                (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2).rolling(window=lookback_window).sum()
            )
        )
        
        return df