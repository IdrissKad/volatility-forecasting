import pytest
import numpy as np
import pandas as pd
from src.strategies.delta_hedging import DeltaHedgingStrategy


class TestDeltaHedgingStrategy:
    @pytest.fixture
    def hedger(self):
        return DeltaHedgingStrategy(
            risk_free_rate=0.05,
            transaction_cost=0.001,
            rebalance_frequency='daily'
        )
    
    @pytest.fixture
    def synthetic_stock_path(self):
        """Generate synthetic stock price path using GBM"""
        np.random.seed(42)
        n_days = 30
        S0 = 100.0
        mu = 0.10  # drift
        sigma = 0.20  # volatility
        dt = 1/252  # daily steps
        
        # Generate geometric Brownian motion
        returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt,
            sigma * np.sqrt(dt),
            n_days
        )
        prices = S0 * np.exp(np.cumsum(returns))
        
        dates = pd.date_range('2024-01-01', periods=n_days)
        return pd.Series([S0] + list(prices), index=[dates[0] - pd.Timedelta(days=1)] + list(dates))

    @pytest.fixture
    def simple_stock_path(self):
        """Simple stock path for basic testing"""
        prices = [100, 102, 101, 103, 105, 104, 106]
        dates = pd.date_range('2024-01-01', periods=len(prices))
        return pd.Series(prices, index=dates)

    def test_hedger_initialization(self):
        """Test DeltaHedgingStrategy initialization"""
        hedger = DeltaHedgingStrategy(
            risk_free_rate=0.03,
            transaction_cost=0.0005,
            rebalance_frequency='hourly'
        )
        
        assert hedger.risk_free_rate == 0.03
        assert hedger.transaction_cost == 0.0005
        assert hedger.rebalance_frequency == 'hourly'
        assert hedger.positions_history == []
        assert hedger.pnl_history == []

    def test_black_scholes_delta_call(self, hedger):
        """Test Black-Scholes delta calculation for calls"""
        # ATM call should have delta around 0.5
        delta = hedger.black_scholes_delta(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call')
        assert 0.45 < delta < 0.55
        
        # ITM call should have higher delta
        itm_delta = hedger.black_scholes_delta(S=110, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call')
        assert itm_delta > delta
        
        # OTM call should have lower delta
        otm_delta = hedger.black_scholes_delta(S=90, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call')
        assert otm_delta < delta

    def test_black_scholes_delta_put(self, hedger):
        """Test Black-Scholes delta calculation for puts"""
        # ATM put should have delta around -0.5
        delta = hedger.black_scholes_delta(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='put')
        assert -0.55 < delta < -0.45
        
        # Put delta should be negative
        assert delta < 0

    def test_black_scholes_gamma(self, hedger):
        """Test Black-Scholes gamma calculation"""
        gamma = hedger.black_scholes_gamma(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        
        # Gamma should be positive
        assert gamma > 0
        
        # ATM options should have highest gamma
        itm_gamma = hedger.black_scholes_gamma(S=110, K=100, T=0.25, r=0.05, sigma=0.2)
        otm_gamma = hedger.black_scholes_gamma(S=90, K=100, T=0.25, r=0.05, sigma=0.2)
        
        assert gamma > itm_gamma
        assert gamma > otm_gamma

    def test_black_scholes_vega(self, hedger):
        """Test Black-Scholes vega calculation"""
        vega = hedger.black_scholes_vega(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        
        # Vega should be positive
        assert vega > 0
        
        # Longer time to expiry should have higher vega
        long_vega = hedger.black_scholes_vega(S=100, K=100, T=0.5, r=0.05, sigma=0.2)
        assert long_vega > vega

    def test_black_scholes_theta_call(self, hedger):
        """Test Black-Scholes theta calculation for calls"""
        theta = hedger.black_scholes_theta(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call')
        
        # Theta should be negative for long options (time decay)
        assert theta < 0

    def test_black_scholes_theta_put(self, hedger):
        """Test Black-Scholes theta calculation for puts"""
        theta = hedger.black_scholes_theta(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='put')
        
        # Theta should be negative for long options
        assert theta < 0

    def test_black_scholes_price_call_put_parity(self, hedger):
        """Test Black-Scholes pricing satisfies call-put parity"""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
        
        call_price = hedger._black_scholes_price(S, K, T, r, sigma, 'call')
        put_price = hedger._black_scholes_price(S, K, T, r, sigma, 'put')
        
        # Call - Put = S - K*exp(-rT) (put-call parity)
        expected_diff = S - K * np.exp(-r * T)
        actual_diff = call_price - put_price
        
        np.testing.assert_almost_equal(actual_diff, expected_diff, decimal=6)

    def test_delta_hedging_simulation_basic(self, hedger, simple_stock_path):
        """Test basic delta hedging simulation"""
        results = hedger.simulate_delta_hedging(
            spot_prices=simple_stock_path,
            strike=100,
            expiry=0.1,  # ~25 days
            volatility=0.2,
            option_type='call'
        )
        
        # Check result structure
        required_keys = ['positions', 'initial_option_price', 'option_payoff', 
                        'hedge_pnl', 'option_pnl', 'total_pnl', 'hedge_effectiveness']
        for key in required_keys:
            assert key in results
        
        # Check positions DataFrame
        positions = results['positions']
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) > 0
        
        required_cols = ['date', 'spot', 'delta', 'gamma', 'shares_traded', 'hedge_cost']
        for col in required_cols:
            assert col in positions.columns
        
        # Check Greeks are reasonable
        assert (positions['delta'] >= -1).all() and (positions['delta'] <= 1).all()
        assert (positions['gamma'] >= 0).all()
        
        # Initial option price should be positive
        assert results['initial_option_price'] > 0

    def test_delta_hedging_call_itm_payoff(self, hedger):
        """Test delta hedging with ITM call at expiration"""
        # Create path where call finishes ITM
        prices = [100, 105, 110, 115, 120]
        dates = pd.date_range('2024-01-01', periods=len(prices))
        stock_path = pd.Series(prices, index=dates)
        
        strike = 100
        results = hedger.simulate_delta_hedging(
            spot_prices=stock_path,
            strike=strike,
            expiry=0.02,  # ~5 days
            volatility=0.2,
            option_type='call'
        )
        
        # Final price is 120, strike is 100, so payoff should be 20
        expected_payoff = max(120 - 100, 0)
        assert abs(results['option_payoff'] - expected_payoff) < 1e-6

    def test_delta_hedging_put_itm_payoff(self, hedger):
        """Test delta hedging with ITM put at expiration"""
        # Create path where put finishes ITM
        prices = [100, 95, 90, 85, 80]
        dates = pd.date_range('2024-01-01', periods=len(prices))
        stock_path = pd.Series(prices, index=dates)
        
        strike = 100
        results = hedger.simulate_delta_hedging(
            spot_prices=stock_path,
            strike=strike,
            expiry=0.02,
            volatility=0.2,
            option_type='put'
        )
        
        # Final price is 80, strike is 100, so payoff should be 20
        expected_payoff = max(100 - 80, 0)
        assert abs(results['option_payoff'] - expected_payoff) < 1e-6

    def test_delta_hedging_otm_expiry(self, hedger):
        """Test delta hedging with OTM option expiring worthless"""
        # Create path where call expires OTM
        prices = [100, 98, 96, 94, 92]
        dates = pd.date_range('2024-01-01', periods=len(prices))
        stock_path = pd.Series(prices, index=dates)
        
        results = hedger.simulate_delta_hedging(
            spot_prices=stock_path,
            strike=100,
            expiry=0.02,
            volatility=0.2,
            option_type='call'
        )
        
        # Should expire worthless
        assert results['option_payoff'] == 0

    def test_delta_hedging_transaction_costs(self, hedger, simple_stock_path):
        """Test that transaction costs are properly calculated"""
        # Test with high transaction costs
        high_cost_hedger = DeltaHedgingStrategy(transaction_cost=0.01)  # 1% cost
        
        results_high = high_cost_hedger.simulate_delta_hedging(
            spot_prices=simple_stock_path,
            strike=100,
            expiry=0.1,
            volatility=0.2,
            option_type='call'
        )
        
        # Test with low transaction costs
        low_cost_hedger = DeltaHedgingStrategy(transaction_cost=0.0001)  # 0.01% cost
        
        results_low = low_cost_hedger.simulate_delta_hedging(
            spot_prices=simple_stock_path,
            strike=100,
            expiry=0.1,
            volatility=0.2,
            option_type='call'
        )
        
        # High cost should result in higher transaction costs
        assert results_high['transaction_costs'] > results_low['transaction_costs']
        
        # Both should have positive transaction costs
        assert results_high['transaction_costs'] > 0
        assert results_low['transaction_costs'] > 0

    def test_delta_hedging_perfect_hedge_theory(self, hedger):
        """Test theoretical perfect hedge with continuous rebalancing"""
        # Create very fine-grained price path
        np.random.seed(42)
        n_steps = 100
        S0 = 100
        T = 0.1
        r = 0.05
        sigma = 0.2
        
        dt = T / n_steps
        returns = np.random.normal((r - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
        prices = S0 * np.exp(np.cumsum([0] + list(returns)))
        
        dates = pd.date_range('2024-01-01', periods=n_steps+1)
        stock_path = pd.Series(prices, index=dates)
        
        # Use very low transaction costs for near-perfect hedge
        perfect_hedger = DeltaHedgingStrategy(transaction_cost=0.0001)
        
        results = perfect_hedger.simulate_delta_hedging(
            spot_prices=stock_path,
            strike=100,
            expiry=T,
            volatility=sigma,
            option_type='call'
        )
        
        # With fine rebalancing and low costs, hedge should be quite effective
        assert results['hedge_effectiveness'] > 0.8  # At least 80% effective

    def test_gamma_hedging_basic(self, hedger, simple_stock_path):
        """Test basic gamma hedging functionality"""
        results = hedger.gamma_hedging(
            spot_prices=simple_stock_path,
            strike_hedge=100,
            strike_gamma=105,
            expiry=0.1,
            volatility=0.2,
            option_type='call'
        )
        
        # Check result structure
        assert 'positions' in results
        assert 'strategy' in results
        assert results['strategy'] == 'gamma_neutral_hedging'
        
        positions = results['positions']
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) > 0
        
        required_cols = ['date', 'spot', 'delta_main', 'gamma_main', 'gamma_hedge_ratio']
        for col in required_cols:
            assert col in positions.columns

    def test_vega_hedging_basic(self, hedger, simple_stock_path):
        """Test basic vega hedging functionality"""
        results = hedger.vega_hedging(
            spot_prices=simple_stock_path,
            strikes=[95, 100, 105],
            expiries=[0.08, 0.1, 0.12],
            volatilities=[0.18, 0.2, 0.22],
            target_vega=0
        )
        
        # Check result structure
        assert 'positions' in results
        assert 'strategy' in results
        assert results['strategy'] == 'vega_neutral_hedging'
        
        positions = results['positions']
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) > 0
        
        required_cols = ['date', 'spot', 'weights', 'portfolio_vega', 'portfolio_delta']
        for col in required_cols:
            assert col in positions.columns

    def test_hedge_performance_analysis(self, hedger, simple_stock_path):
        """Test hedge performance analysis"""
        results = hedger.simulate_delta_hedging(
            spot_prices=simple_stock_path,
            strike=100,
            expiry=0.1,
            volatility=0.2,
            option_type='call'
        )
        
        metrics = hedger.analyze_hedge_performance(results)
        
        # Check required metrics
        required_metrics = ['total_pnl', 'hedge_pnl', 'option_pnl', 'transaction_costs',
                           'hedge_effectiveness', 'max_delta', 'min_delta', 'avg_delta',
                           'total_shares_traded', 'n_rebalances']
        
        for metric in required_metrics:
            assert metric in metrics
        
        # Check reasonable values
        assert isinstance(metrics['n_rebalances'], (int, np.integer))
        assert metrics['n_rebalances'] > 0
        assert metrics['total_shares_traded'] >= 0
        assert -1 <= metrics['max_delta'] <= 1
        assert -1 <= metrics['min_delta'] <= 1

    def test_sharpe_ratio_calculation(self, hedger):
        """Test Sharpe ratio calculation"""
        # Test with trending PnL
        pnl_series = pd.Series([0, 1, 2, 3, 4, 5])
        sharpe = hedger._calculate_sharpe_ratio(pnl_series)
        assert sharpe > 0  # Positive trend should give positive Sharpe
        
        # Test with flat PnL
        flat_pnl = pd.Series([1, 1, 1, 1, 1])
        flat_sharpe = hedger._calculate_sharpe_ratio(flat_pnl)
        assert flat_sharpe == 0  # No variation should give zero Sharpe
        
        # Test with single value
        single_pnl = pd.Series([1])
        single_sharpe = hedger._calculate_sharpe_ratio(single_pnl)
        assert single_sharpe == 0

    def test_greeks_consistency(self, hedger):
        """Test consistency between Greeks calculations"""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
        
        # Calculate Greeks
        delta = hedger.black_scholes_delta(S, K, T, r, sigma, 'call')
        gamma = hedger.black_scholes_gamma(S, K, T, r, sigma)
        vega = hedger.black_scholes_vega(S, K, T, r, sigma)
        theta = hedger.black_scholes_theta(S, K, T, r, sigma, 'call')
        
        # All should be finite
        assert np.isfinite(delta)
        assert np.isfinite(gamma)
        assert np.isfinite(vega)
        assert np.isfinite(theta)
        
        # Call delta should be between 0 and 1
        assert 0 <= delta <= 1
        
        # Gamma and vega should be positive
        assert gamma >= 0
        assert vega >= 0
        
        # Theta should be negative for long call
        assert theta <= 0

    def test_edge_cases_zero_time_to_expiry(self, hedger):
        """Test edge case with zero time to expiry"""
        # At expiry, delta should be 0 or 1 for calls
        spot_prices = [90, 100, 110]
        strike = 100
        
        for S in spot_prices:
            delta = hedger.black_scholes_delta(S, strike, T=1e-6, r=0.05, sigma=0.2, option_type='call')
            
            if S > strike:
                assert abs(delta - 1.0) < 0.01  # Should be close to 1 for ITM
            else:
                assert abs(delta) < 0.01  # Should be close to 0 for OTM

    def test_edge_cases_very_high_volatility(self, hedger):
        """Test edge case with very high volatility"""
        # Very high volatility should still produce reasonable Greeks
        delta = hedger.black_scholes_delta(S=100, K=100, T=0.25, r=0.05, sigma=2.0, option_type='call')
        gamma = hedger.black_scholes_gamma(S=100, K=100, T=0.25, r=0.05, sigma=2.0)
        
        # Should still be in valid ranges
        assert 0 <= delta <= 1
        assert gamma >= 0
        assert np.isfinite(delta)
        assert np.isfinite(gamma)

    def test_synthetic_perfect_hedge_zero_vol(self, hedger):
        """Test perfect hedge case with zero volatility (deterministic path)"""
        # Create deterministic path
        prices = [100, 101, 102, 103, 104, 105]
        dates = pd.date_range('2024-01-01', periods=len(prices))
        stock_path = pd.Series(prices, index=dates)
        
        # Use zero transaction costs for perfect hedge
        perfect_hedger = DeltaHedgingStrategy(transaction_cost=0.0)
        
        results = perfect_hedger.simulate_delta_hedging(
            spot_prices=stock_path,
            strike=100,
            expiry=0.1,
            volatility=0.01,  # Very low vol
            option_type='call'
        )
        
        # With deterministic path and no costs, hedge should be very effective
        assert results['hedge_effectiveness'] > 0.9
        assert results['transaction_costs'] == 0

    def test_delta_hedging_different_frequencies(self, hedger, simple_stock_path):
        """Test delta hedging with different rebalancing frequencies"""
        # This test mainly checks that different frequencies don't break the simulation
        frequencies = ['daily', 'hourly', 'continuous']
        
        for freq in frequencies:
            hedger.rebalance_frequency = freq
            results = hedger.simulate_delta_hedging(
                spot_prices=simple_stock_path,
                strike=100,
                expiry=0.1,
                volatility=0.2,
                option_type='call'
            )
            
            # Should complete successfully regardless of frequency setting
            assert 'total_pnl' in results
            assert np.isfinite(results['total_pnl'])

    def test_hedging_with_custom_initial_price(self, hedger, simple_stock_path):
        """Test delta hedging with custom initial option price"""
        custom_price = 5.0
        
        results = hedger.simulate_delta_hedging(
            spot_prices=simple_stock_path,
            strike=100,
            expiry=0.1,
            volatility=0.2,
            option_type='call',
            initial_option_price=custom_price
        )
        
        # Should use the custom price
        assert abs(results['initial_option_price'] - custom_price) < 1e-6

    def test_portfolio_greeks_aggregation(self, hedger):
        """Test that portfolio Greeks aggregate correctly"""
        S, r, T, sigma = 100, 0.05, 0.25, 0.2
        
        # Individual option Greeks
        delta1 = hedger.black_scholes_delta(S, 95, T, r, sigma, 'call')
        delta2 = hedger.black_scholes_delta(S, 105, T, r, sigma, 'call')
        
        gamma1 = hedger.black_scholes_gamma(S, 95, T, r, sigma)
        gamma2 = hedger.black_scholes_gamma(S, 105, T, r, sigma)
        
        # Portfolio of +1 call @ 95, -1 call @ 105 (call spread)
        portfolio_delta = delta1 - delta2
        portfolio_gamma = gamma1 - gamma2
        
        # Portfolio delta should be between individual deltas
        assert min(delta1, delta2) <= abs(portfolio_delta) <= max(delta1, delta2)
        
        # Greeks should be finite
        assert np.isfinite(portfolio_delta)
        assert np.isfinite(portfolio_gamma)