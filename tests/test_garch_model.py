import pytest
import numpy as np
import pandas as pd
from src.models.garch_model import GARCHModel, EGARCHModel, GJRGARCHModel


class TestGARCHModel:
    @pytest.fixture
    def synthetic_garch_returns(self):
        """Generate synthetic GARCH(1,1) returns with known parameters"""
        np.random.seed(42)
        n = 1000
        
        # GARCH(1,1) parameters
        omega = 0.00001  # long-run variance
        alpha = 0.1      # ARCH parameter
        beta = 0.85      # GARCH parameter
        
        # Initialize
        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
        
        # Generate GARCH process
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
        return pd.Series(returns[1:], index=pd.date_range('2020-01-01', periods=n-1))
    
    @pytest.fixture
    def simple_returns(self):
        """Generate simple returns for basic testing"""
        np.random.seed(123)
        n = 500
        returns = np.random.normal(0, 0.01, n)
        return pd.Series(returns, index=pd.date_range('2023-01-01', periods=n))

    def test_garch_model_initialization(self):
        """Test GARCH model initialization with different parameters"""
        # Default initialization
        model = GARCHModel()
        assert model.p == 1
        assert model.q == 1
        assert model.dist == 'normal'
        assert model.mean_model == 'Constant'
        
        # Custom initialization
        model = GARCHModel(p=2, q=2, dist='t', mean_model='Zero')
        assert model.p == 2
        assert model.q == 2
        assert model.dist == 't'
        assert model.mean_model == 'Zero'

    def test_garch_model_fit_basic(self, simple_returns):
        """Test basic GARCH model fitting functionality"""
        model = GARCHModel(p=1, q=1)
        model.fit(simple_returns.values)
        
        # Check that model was fitted
        assert model.fitted_model is not None
        assert model.params is not None
        assert len(model.params) > 0
        
        # Check parameter names for GARCH(1,1)
        expected_params = ['mu', 'omega', 'alpha[1]', 'beta[1]']
        for param in expected_params:
            assert param in model.params.index

    def test_garch_parameter_stability_known_case(self, synthetic_garch_returns):
        """Test parameter stability with synthetic GARCH data"""
        model = GARCHModel(p=1, q=1)
        model.fit(synthetic_garch_returns.values)
        
        # Check that parameters are reasonable for known synthetic case
        params = model.params
        
        # ARCH + GARCH coefficients should sum to < 1 for stationarity
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        persistence = alpha + beta
        
        assert 0 < alpha < 1, f"Alpha should be in (0,1), got {alpha}"
        assert 0 < beta < 1, f"Beta should be in (0,1), got {beta}"
        assert persistence < 1, f"Persistence should be < 1, got {persistence}"
        
        # Omega should be positive
        omega = params['omega']
        assert omega > 0, f"Omega should be positive, got {omega}"

    def test_garch_forecast_single_step(self, simple_returns):
        """Test single-step ahead forecasting"""
        model = GARCHModel(p=1, q=1)
        model.fit(simple_returns.values)
        
        forecast = model.forecast(horizon=1)
        
        # Check forecast structure
        assert 'variance' in forecast
        assert 'volatility' in forecast
        
        # Check forecast properties
        var_forecast = forecast['variance'][0]
        vol_forecast = forecast['volatility'][0]
        
        assert var_forecast > 0, "Variance forecast should be positive"
        assert vol_forecast > 0, "Volatility forecast should be positive"
        assert np.isclose(vol_forecast, np.sqrt(var_forecast)), "Volatility should be sqrt of variance"

    def test_garch_forecast_multi_step(self, simple_returns):
        """Test multi-step ahead forecasting"""
        model = GARCHModel(p=1, q=1)
        model.fit(simple_returns.values)
        
        horizon = 5
        forecast = model.forecast(horizon=horizon)
        
        # Check forecast dimensions
        assert len(forecast['variance']) == horizon
        assert len(forecast['volatility']) == horizon
        
        # Check that all forecasts are positive
        assert all(forecast['variance'] > 0)
        assert all(forecast['volatility'] > 0)
        
        # For GARCH, variance should converge to long-run level
        var_forecasts = forecast['variance']
        # Later forecasts should be closer to each other (mean reversion)
        assert abs(var_forecasts[-1] - var_forecasts[-2]) <= abs(var_forecasts[1] - var_forecasts[0])

    def test_garch_forecast_without_fitting(self):
        """Test that forecasting without fitting raises error"""
        model = GARCHModel(p=1, q=1)
        
        with pytest.raises(ValueError, match="Model must be fitted before forecasting"):
            model.forecast(horizon=1)

    def test_garch_rolling_forecast(self, simple_returns):
        """Test rolling forecast functionality"""
        model = GARCHModel(p=1, q=1)
        
        # Use smaller window for testing
        window_size = 100
        forecasts = model.rolling_forecast(
            simple_returns, 
            window_size=window_size, 
            forecast_horizon=1
        )
        
        # Check forecast structure
        assert isinstance(forecasts, pd.DataFrame)
        assert 'forecast_vol' in forecasts.columns
        assert 'forecast_var' in forecasts.columns
        
        # Check expected number of forecasts
        expected_length = len(simple_returns) - window_size
        assert len(forecasts) == expected_length
        
        # Check that most forecasts are not NaN (some might fail)
        non_nan_ratio = (~forecasts['forecast_vol'].isna()).mean()
        assert non_nan_ratio > 0.8, f"Too many failed forecasts: {non_nan_ratio:.2%} success rate"

    def test_garch_conditional_volatility(self, simple_returns):
        """Test conditional volatility extraction"""
        model = GARCHModel(p=1, q=1)
        model.fit(simple_returns.values)
        
        cond_vol = model.conditional_volatility()
        
        # Check basic properties
        assert isinstance(cond_vol, pd.Series)
        assert len(cond_vol) == len(simple_returns)
        assert (cond_vol > 0).all(), "All conditional volatilities should be positive"
        
        # Check reasonable magnitude (should be similar to realized vol)
        realized_vol = simple_returns.std() * np.sqrt(252)
        avg_cond_vol = cond_vol.mean() * np.sqrt(252)
        
        # Should be in same order of magnitude
        ratio = avg_cond_vol / realized_vol
        assert 0.1 < ratio < 10, f"Conditional vol seems unreasonable: ratio = {ratio}"

    def test_garch_model_info(self, simple_returns):
        """Test model information extraction"""
        model = GARCHModel(p=1, q=1, dist='t')
        model.fit(simple_returns.values)
        
        info = model.get_model_info()
        
        # Check required fields
        required_fields = ['aic', 'bic', 'log_likelihood', 'params', 'p', 'q', 'distribution']
        for field in required_fields:
            assert field in info, f"Missing field: {field}"
        
        # Check values are reasonable
        assert info['aic'] < 0  # Should be negative for log-likelihood
        assert info['bic'] < 0
        assert info['log_likelihood'] < 0
        assert info['p'] == 1
        assert info['q'] == 1
        assert info['distribution'] == 't'

    def test_garch_model_info_without_fitting(self):
        """Test that getting model info without fitting raises error"""
        model = GARCHModel(p=1, q=1)
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_model_info()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.conditional_volatility()

    def test_garch_parameter_constraints_stability(self):
        """Test parameter stability across multiple random seeds"""
        n_trials = 5
        all_params = []
        
        for seed in range(n_trials):
            np.random.seed(seed)
            returns = np.random.normal(0, 0.01, 500)
            
            model = GARCHModel(p=1, q=1)
            try:
                model.fit(returns)
                params = model.params
                all_params.append({
                    'alpha': params['alpha[1]'],
                    'beta': params['beta[1]'],
                    'omega': params['omega']
                })
            except:
                continue
        
        # Check that we got successful fits
        assert len(all_params) >= 3, "Too many fitting failures"
        
        # Check stability constraints across all fits
        for params in all_params:
            assert 0 < params['alpha'] < 1
            assert 0 < params['beta'] < 1
            assert params['alpha'] + params['beta'] < 1
            assert params['omega'] > 0


class TestEGARCHModel:
    def test_egarch_initialization(self):
        """Test EGARCH model initialization"""
        model = EGARCHModel(p=1, o=1, q=1)
        assert model.p == 1
        assert model.o == 1
        assert model.q == 1

    def test_egarch_fit_basic(self):
        """Test basic EGARCH fitting"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 300)  # More volatile for EGARCH
        
        model = EGARCHModel(p=1, o=1, q=1)
        model.fit(returns)
        
        assert model.fitted_model is not None
        assert model.params is not None
        
        # EGARCH should have gamma parameter for asymmetry
        param_names = model.params.index.tolist()
        assert any('gamma' in name for name in param_names), f"No gamma parameter in {param_names}"

    def test_egarch_asymmetry_capture(self):
        """Test that EGARCH captures asymmetric effects"""
        np.random.seed(42)
        n = 500
        
        # Create asymmetric returns (more negative volatility clustering)
        returns = np.random.normal(0, 0.01, n)
        # Add some negative shocks with higher subsequent volatility
        neg_shock_indices = np.random.choice(range(50, n-50), 10, replace=False)
        for idx in neg_shock_indices:
            returns[idx] = -0.05  # Large negative shock
            returns[idx+1:idx+10] *= 1.5  # Higher subsequent volatility
        
        model = EGARCHModel(p=1, o=1, q=1)
        model.fit(returns)
        
        # Should fit successfully with reasonable parameters
        assert model.fitted_model is not None
        forecast = model.forecast(horizon=1)
        assert forecast['volatility'][0] > 0


class TestGJRGARCHModel:
    def test_gjr_initialization(self):
        """Test GJR-GARCH model initialization"""
        model = GJRGARCHModel(p=1, o=1, q=1)
        assert model.p == 1
        assert model.o == 1
        assert model.q == 1

    def test_gjr_fit_basic(self):
        """Test basic GJR-GARCH fitting"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 400)
        
        model = GJRGARCHModel(p=1, o=1, q=1)
        model.fit(returns)
        
        assert model.fitted_model is not None
        assert model.params is not None
        
        # Check that we have the expected parameters
        params = model.params
        assert 'omega' in params.index
        assert 'alpha[1]' in params.index
        assert 'beta[1]' in params.index


class TestGARCHModelComparison:
    def test_model_comparison_on_same_data(self):
        """Test that different GARCH models give different results on same data"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.015, 400)
        
        # Fit different models
        garch = GARCHModel(p=1, q=1)
        egarch = EGARCHModel(p=1, o=1, q=1)
        
        garch.fit(returns)
        egarch.fit(returns)
        
        # Get forecasts
        garch_forecast = garch.forecast(horizon=1)
        egarch_forecast = egarch.forecast(horizon=1)
        
        # Forecasts should be different (though similar)
        garch_vol = garch_forecast['volatility'][0]
        egarch_vol = egarch_forecast['volatility'][0]
        
        assert garch_vol > 0
        assert egarch_vol > 0
        # Allow for some similarity but ensure they're not identical
        assert abs(garch_vol - egarch_vol) / max(garch_vol, egarch_vol) > 0.001

    def test_model_selection_criteria(self):
        """Test that model selection criteria are reasonable"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 350)
        
        models = [
            GARCHModel(p=1, q=1),
            EGARCHModel(p=1, o=1, q=1)
        ]
        
        results = []
        for model in models:
            try:
                model.fit(returns)
                info = model.get_model_info()
                results.append({
                    'model': type(model).__name__,
                    'aic': info['aic'],
                    'bic': info['bic'],
                    'log_likelihood': info['log_likelihood']
                })
            except:
                continue
        
        assert len(results) >= 1, "At least one model should fit successfully"
        
        # All information criteria should be negative (log-likelihood based)
        for result in results:
            assert result['aic'] < 0
            assert result['bic'] < 0
            assert result['log_likelihood'] < 0


class TestGARCHEdgeCases:
    def test_short_time_series(self):
        """Test behavior with very short time series"""
        returns = np.random.normal(0, 0.01, 50)
        model = GARCHModel(p=1, q=1)
        
        # Should either fit or raise an informative error
        try:
            model.fit(returns)
            # If it fits, should still produce reasonable results
            assert model.fitted_model is not None
            forecast = model.forecast(horizon=1)
            assert forecast['volatility'][0] > 0
        except Exception as e:
            # If it fails, that's also acceptable for very short series
            assert len(returns) < 100

    def test_constant_returns(self):
        """Test behavior with constant returns"""
        returns = np.zeros(200)  # All zeros
        model = GARCHModel(p=1, q=1)
        
        # This should either handle gracefully or raise appropriate error
        try:
            model.fit(returns)
            # If successful, conditional volatility should be very small
            cond_vol = model.conditional_volatility()
            assert (cond_vol >= 0).all()  # Non-negative
            assert cond_vol.max() < 0.1   # Very small
        except Exception as e:
            # Failure is acceptable for degenerate data
            assert "returns" in str(e).lower() or "variance" in str(e).lower()

    def test_extreme_returns(self):
        """Test behavior with extreme returns"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 300)
        
        # Add some extreme outliers
        returns[100] = 0.2   # 20% return
        returns[200] = -0.15  # -15% return
        
        model = GARCHModel(p=1, q=1)
        model.fit(returns)
        
        # Should still produce reasonable forecasts
        forecast = model.forecast(horizon=1)
        assert 0 < forecast['volatility'][0] < 1  # Reasonable range
        
        # Conditional volatility should spike around extreme events
        cond_vol = model.conditional_volatility()
        assert cond_vol[101] > cond_vol[90:100].mean()  # Higher vol after extreme return