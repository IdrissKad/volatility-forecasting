import pytest
import numpy as np
import pandas as pd
from src.models.realized_volatility import RealizedVolatility


class TestRealizedVolatility:
    @pytest.fixture
    def rv_calculator(self):
        return RealizedVolatility(annualization_factor=252)
    
    @pytest.fixture
    def sample_returns(self):
        """Generate synthetic returns with known volatility"""
        np.random.seed(42)
        n_days = 100
        true_vol = 0.20  # 20% annual volatility
        daily_vol = true_vol / np.sqrt(252)
        returns = np.random.normal(0, daily_vol, n_days)
        return pd.Series(returns, index=pd.date_range('2023-01-01', periods=n_days))
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Generate synthetic OHLC data with known properties"""
        np.random.seed(42)
        n_days = 100
        base_price = 100.0
        
        # Generate random walks for each series
        returns = np.random.normal(0, 0.01, n_days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create realistic OHLC from prices
        opens = prices * np.random.uniform(0.995, 1.005, n_days)
        highs = np.maximum(opens, prices) * np.random.uniform(1.0, 1.02, n_days)
        lows = np.minimum(opens, prices) * np.random.uniform(0.98, 1.0, n_days)
        closes = prices
        
        dates = pd.date_range('2023-01-01', periods=n_days)
        
        return {
            'open': pd.Series(opens, index=dates),
            'high': pd.Series(highs, index=dates),
            'low': pd.Series(lows, index=dates),
            'close': pd.Series(closes, index=dates)
        }

    def test_simple_realized_vol_properties(self, rv_calculator, sample_returns):
        """Test basic properties of simple realized volatility"""
        vol = rv_calculator.simple_realized_vol(sample_returns, window=20)
        
        # Check that volatility is positive
        assert (vol.dropna() > 0).all()
        
        # Check that first 19 values are NaN (window-1)
        assert pd.isna(vol.iloc[:19]).all()
        
        # Check that we have non-NaN values after window period
        assert not pd.isna(vol.iloc[19:]).any()
        
        # Check reasonable magnitude (should be between 0.05 and 1.0 for typical equity data)
        vol_values = vol.dropna()
        assert (vol_values >= 0.05).all() and (vol_values <= 1.0).all()

    def test_simple_realized_vol_known_case(self, rv_calculator):
        """Test simple realized volatility with a known case"""
        # Create returns with known standard deviation
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
        
        expected_std = returns.std()
        expected_vol = expected_std * np.sqrt(252)
        
        vol = rv_calculator.simple_realized_vol(returns, window=5)
        
        # Should only have one non-NaN value at the end
        assert pd.isna(vol.iloc[:-1]).all()
        np.testing.assert_almost_equal(vol.iloc[-1], expected_vol, decimal=6)

    def test_exponentially_weighted_vol(self, rv_calculator, sample_returns):
        """Test exponentially weighted volatility"""
        vol = rv_calculator.exponentially_weighted_vol(sample_returns, alpha=0.94)
        
        # Check that volatility is positive
        assert (vol.dropna() > 0).all()
        
        # EWMA should produce fewer NaN values than rolling window
        assert vol.dropna().shape[0] >= sample_returns.dropna().shape[0] - 1

    def test_parkinson_vol(self, rv_calculator, sample_ohlc_data):
        """Test Parkinson volatility estimator"""
        vol = rv_calculator.parkinson_vol(
            sample_ohlc_data['high'], 
            sample_ohlc_data['low'], 
            window=20
        )
        
        # Check basic properties
        assert (vol.dropna() > 0).all()
        assert pd.isna(vol.iloc[:19]).all()  # First window-1 values should be NaN
        
        # Parkinson should be less noisy than close-to-close for same data
        # (this is a statistical property, not always true for single realization)
        vol_values = vol.dropna()
        assert len(vol_values) > 0

    def test_parkinson_vol_extreme_case(self, rv_calculator):
        """Test Parkinson volatility with extreme high-low ratios"""
        # Create data where high/low ratio is constant
        n = 25
        high = pd.Series([110.0] * n)  # 10% above base
        low = pd.Series([90.0] * n)   # 10% below base
        
        vol = rv_calculator.parkinson_vol(high, low, window=20)
        
        # All non-NaN values should be identical since H/L ratio is constant
        vol_values = vol.dropna()
        assert len(np.unique(vol_values.round(6))) == 1

    def test_garman_klass_vol(self, rv_calculator, sample_ohlc_data):
        """Test Garman-Klass volatility estimator"""
        vol = rv_calculator.garman_klass_vol(
            sample_ohlc_data['open'],
            sample_ohlc_data['high'], 
            sample_ohlc_data['low'],
            sample_ohlc_data['close'], 
            window=20
        )
        
        # Check basic properties
        assert (vol.dropna() > 0).all()
        assert pd.isna(vol.iloc[:19]).all()
        
        # Should have reasonable values
        vol_values = vol.dropna()
        assert (vol_values >= 0.01).all() and (vol_values <= 2.0).all()

    def test_rogers_satchell_vol(self, rv_calculator, sample_ohlc_data):
        """Test Rogers-Satchell volatility estimator"""
        vol = rv_calculator.rogers_satchell_vol(
            sample_ohlc_data['open'],
            sample_ohlc_data['high'], 
            sample_ohlc_data['low'],
            sample_ohlc_data['close'], 
            window=20
        )
        
        # Check basic properties
        assert (vol.dropna() > 0).all()
        assert pd.isna(vol.iloc[:19]).all()

    def test_yang_zhang_vol(self, rv_calculator, sample_ohlc_data):
        """Test Yang-Zhang volatility estimator"""
        vol = rv_calculator.yang_zhang_vol(
            sample_ohlc_data['open'],
            sample_ohlc_data['high'], 
            sample_ohlc_data['low'],
            sample_ohlc_data['close'], 
            window=20
        )
        
        # Check basic properties
        assert (vol.dropna() > 0).all()
        assert pd.isna(vol.iloc[:19]).all()  # Should have NaN for first observations
        
        # Yang-Zhang should handle overnight gaps
        vol_values = vol.dropna()
        assert len(vol_values) > 0

    def test_yang_zhang_with_custom_k(self, rv_calculator, sample_ohlc_data):
        """Test Yang-Zhang volatility with custom k parameter"""
        vol1 = rv_calculator.yang_zhang_vol(
            sample_ohlc_data['open'],
            sample_ohlc_data['high'], 
            sample_ohlc_data['low'],
            sample_ohlc_data['close'], 
            window=20,
            k=0.5
        )
        
        vol2 = rv_calculator.yang_zhang_vol(
            sample_ohlc_data['open'],
            sample_ohlc_data['high'], 
            sample_ohlc_data['low'],
            sample_ohlc_data['close'], 
            window=20,
            k=0.1
        )
        
        # Different k values should produce different results
        assert not vol1.equals(vol2)

    def test_bipower_variation(self, rv_calculator, sample_returns):
        """Test bipower variation estimator"""
        bv = rv_calculator.bipower_variation(sample_returns, window=20)
        
        # Check basic properties
        assert (bv.dropna() > 0).all()
        assert pd.isna(bv.iloc[:19]).all()

    def test_jump_robust_vol(self, rv_calculator, sample_returns):
        """Test jump-robust volatility decomposition"""
        result = rv_calculator.jump_robust_vol(sample_returns, window=20, alpha=0.05)
        
        # Check that all components are returned
        assert 'continuous_vol' in result
        assert 'jump_vol' in result
        assert 'jump_indicator' in result
        
        # Check basic properties
        continuous_vol = result['continuous_vol']
        jump_vol = result['jump_vol']
        jump_indicator = result['jump_indicator']
        
        assert (continuous_vol.dropna() >= 0).all()
        assert (jump_vol.dropna() >= 0).all()
        assert jump_indicator.dtype == bool

    def test_intraday_vol_simple(self, rv_calculator):
        """Test intraday volatility calculation with simple method"""
        # Create high-frequency price data
        np.random.seed(42)
        n_points = 78  # 5-minute intervals in a trading day
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n_points))))
        
        vol = rv_calculator.intraday_vol(prices, frequency='5min', method='simple')
        
        assert vol > 0
        assert isinstance(vol, float)

    def test_intraday_vol_realized_variance(self, rv_calculator):
        """Test intraday volatility with realized variance method"""
        np.random.seed(42)
        n_points = 78
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n_points))))
        
        vol = rv_calculator.intraday_vol(prices, frequency='5min', method='realized_variance')
        
        assert vol > 0
        assert isinstance(vol, float)

    def test_intraday_vol_invalid_method(self, rv_calculator):
        """Test intraday volatility with invalid method raises error"""
        prices = pd.Series([100, 101, 102, 103])
        
        with pytest.raises(ValueError, match="Unknown method"):
            rv_calculator.intraday_vol(prices, method='invalid')

    def test_get_periods_per_day(self, rv_calculator):
        """Test frequency mapping for periods per day"""
        assert rv_calculator._get_periods_per_day('1min') == 390
        assert rv_calculator._get_periods_per_day('5min') == 78
        assert rv_calculator._get_periods_per_day('1h') == 6.5
        assert rv_calculator._get_periods_per_day('unknown') == 78  # default

    def test_annualization_factor(self):
        """Test different annualization factors"""
        rv_252 = RealizedVolatility(annualization_factor=252)
        rv_365 = RealizedVolatility(annualization_factor=365)
        
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
        
        vol_252 = rv_252.simple_realized_vol(returns, window=5).iloc[-1]
        vol_365 = rv_365.simple_realized_vol(returns, window=5).iloc[-1]
        
        # 365-day annualization should be higher than 252-day
        assert vol_365 > vol_252
        
        # Ratio should be approximately sqrt(365/252)
        expected_ratio = np.sqrt(365/252)
        actual_ratio = vol_365 / vol_252
        np.testing.assert_almost_equal(actual_ratio, expected_ratio, decimal=10)

    def test_consistency_across_estimators(self, rv_calculator, sample_ohlc_data):
        """Test that different estimators produce reasonable relative results"""
        # Calculate different volatility measures
        parkinson = rv_calculator.parkinson_vol(
            sample_ohlc_data['high'], sample_ohlc_data['low'], window=20
        ).dropna()
        
        gk = rv_calculator.garman_klass_vol(
            sample_ohlc_data['open'], sample_ohlc_data['high'],
            sample_ohlc_data['low'], sample_ohlc_data['close'], window=20
        ).dropna()
        
        rs = rv_calculator.rogers_satchell_vol(
            sample_ohlc_data['open'], sample_ohlc_data['high'],
            sample_ohlc_data['low'], sample_ohlc_data['close'], window=20
        ).dropna()
        
        # All should be positive
        assert (parkinson > 0).all()
        assert (gk > 0).all()
        assert (rs > 0).all()
        
        # All should be in reasonable range
        for vol_series in [parkinson, gk, rs]:
            assert (vol_series > 0.01).all() and (vol_series < 2.0).all()

    def test_edge_cases_empty_data(self, rv_calculator):
        """Test handling of edge cases with empty data"""
        empty_series = pd.Series([], dtype=float)
        
        vol = rv_calculator.simple_realized_vol(empty_series, window=5)
        assert len(vol) == 0

    def test_edge_cases_single_value(self, rv_calculator):
        """Test handling of single value data"""
        single_value = pd.Series([0.01])
        
        vol = rv_calculator.simple_realized_vol(single_value, window=5)
        assert pd.isna(vol.iloc[0])

    def test_window_larger_than_data(self, rv_calculator):
        """Test behavior when window is larger than available data"""
        short_returns = pd.Series([0.01, -0.02, 0.015])
        
        vol = rv_calculator.simple_realized_vol(short_returns, window=10)
        assert pd.isna(vol).all()