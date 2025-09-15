import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class VolatilityForecastEvaluator:
    """
    Comprehensive evaluation framework for volatility forecasting models.
    
    Supports walk-forward, expanding window evaluation with comprehensive metrics
    including MSE, RMSE, QLIKE, and Diebold-Mariano tests for model comparison.
    """
    
    def __init__(self, annualization_factor: int = 252):
        self.annualization_factor = annualization_factor
        
    def walk_forward_evaluation(
        self,
        returns: pd.Series,
        models: Dict[str, Any],
        initial_window: int = 252,
        step_size: int = 1,
        forecast_horizon: int = 1,
        refit_frequency: int = 22,
        min_observations: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform walk-forward evaluation of volatility models.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
        models : dict
            Dictionary of model instances with fit() and forecast() methods
        initial_window : int
            Initial training window size
        step_size : int
            Step size for walk-forward
        forecast_horizon : int
            Forecast horizon in periods
        refit_frequency : int
            How often to refit models (in periods)
        min_observations : int
            Minimum observations required for fitting
            
        Returns:
        --------
        dict : Dictionary with model names as keys and evaluation DataFrames as values
        """
        n = len(returns)
        results = {model_name: [] for model_name in models.keys()}
        
        # Calculate realized volatility for comparison
        realized_vol = self._calculate_realized_volatility(returns, forecast_horizon)
        
        start_idx = initial_window
        refit_counter = 0
        fitted_models = {}
        
        for i in range(start_idx, n - forecast_horizon + 1, step_size):
            # Get training data
            if i < len(realized_vol):
                train_end = i
                train_start = max(0, train_end - initial_window)
                train_data = returns.iloc[train_start:train_end]
                
                # Check if we have enough data
                if len(train_data) < min_observations:
                    continue
                
                # Refit models if needed
                if refit_counter % refit_frequency == 0:
                    fitted_models = self._fit_models(models, train_data)
                
                # Generate forecasts
                forecast_date = returns.index[i]
                actual_vol = realized_vol.iloc[i] if i < len(realized_vol) else np.nan
                
                for model_name, fitted_model in fitted_models.items():
                    try:
                        if fitted_model is not None:
                            forecast = self._get_model_forecast(fitted_model, forecast_horizon)
                            forecast_vol = forecast.get('volatility', [np.nan])[0]
                        else:
                            forecast_vol = np.nan
                            
                        results[model_name].append({
                            'date': forecast_date,
                            'actual_vol': actual_vol,
                            'forecast_vol': forecast_vol,
                            'forecast_var': forecast_vol ** 2 if not np.isnan(forecast_vol) else np.nan,
                            'actual_var': actual_vol ** 2 if not np.isnan(actual_vol) else np.nan,
                            'train_start': train_data.index[0],
                            'train_end': train_data.index[-1],
                            'n_train_obs': len(train_data)
                        })
                        
                    except Exception as e:
                        results[model_name].append({
                            'date': forecast_date,
                            'actual_vol': actual_vol,
                            'forecast_vol': np.nan,
                            'forecast_var': np.nan,
                            'actual_var': actual_vol ** 2 if not np.isnan(actual_vol) else np.nan,
                            'train_start': train_data.index[0] if len(train_data) > 0 else pd.NaT,
                            'train_end': train_data.index[-1] if len(train_data) > 0 else pd.NaT,
                            'n_train_obs': len(train_data),
                            'error': str(e)
                        })
                
                refit_counter += 1
        
        # Convert to DataFrames
        return {name: pd.DataFrame(data).set_index('date') for name, data in results.items()}
    
    def expanding_window_evaluation(
        self,
        returns: pd.Series,
        models: Dict[str, Any],
        initial_window: int = 252,
        step_size: int = 1,
        forecast_horizon: int = 1,
        refit_frequency: int = 22
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform expanding window evaluation of volatility models.
        
        Similar to walk-forward but uses expanding window instead of rolling window.
        """
        n = len(returns)
        results = {model_name: [] for model_name in models.keys()}
        
        realized_vol = self._calculate_realized_volatility(returns, forecast_horizon)
        
        start_idx = initial_window
        refit_counter = 0
        fitted_models = {}
        
        for i in range(start_idx, n - forecast_horizon + 1, step_size):
            if i < len(realized_vol):
                # Expanding window: always start from beginning
                train_data = returns.iloc[:i]
                
                # Refit models if needed
                if refit_counter % refit_frequency == 0:
                    fitted_models = self._fit_models(models, train_data)
                
                # Generate forecasts
                forecast_date = returns.index[i]
                actual_vol = realized_vol.iloc[i] if i < len(realized_vol) else np.nan
                
                for model_name, fitted_model in fitted_models.items():
                    try:
                        if fitted_model is not None:
                            forecast = self._get_model_forecast(fitted_model, forecast_horizon)
                            forecast_vol = forecast.get('volatility', [np.nan])[0]
                        else:
                            forecast_vol = np.nan
                            
                        results[model_name].append({
                            'date': forecast_date,
                            'actual_vol': actual_vol,
                            'forecast_vol': forecast_vol,
                            'forecast_var': forecast_vol ** 2 if not np.isnan(forecast_vol) else np.nan,
                            'actual_var': actual_vol ** 2 if not np.isnan(actual_vol) else np.nan,
                            'train_start': train_data.index[0],
                            'train_end': train_data.index[-1],
                            'n_train_obs': len(train_data)
                        })
                        
                    except Exception as e:
                        results[model_name].append({
                            'date': forecast_date,
                            'actual_vol': actual_vol,
                            'forecast_vol': np.nan,
                            'forecast_var': np.nan,
                            'actual_var': actual_vol ** 2 if not np.isnan(actual_vol) else np.nan,
                            'train_start': train_data.index[0],
                            'train_end': train_data.index[-1],
                            'n_train_obs': len(train_data),
                            'error': str(e)
                        })
                
                refit_counter += 1
        
        return {name: pd.DataFrame(data).set_index('date') for name, data in results.items()}
    
    def calculate_forecast_metrics(
        self,
        evaluation_results: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive forecast evaluation metrics.
        
        Parameters:
        -----------
        evaluation_results : dict
            Results from walk_forward_evaluation or expanding_window_evaluation
            
        Returns:
        --------
        dict : Nested dictionary with model names and metrics
        """
        metrics = {}
        
        for model_name, results_df in evaluation_results.items():
            # Filter out NaN values
            valid_data = results_df.dropna(subset=['actual_vol', 'forecast_vol'])
            
            if len(valid_data) == 0:
                metrics[model_name] = {
                    'n_forecasts': 0,
                    'mse': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'qlike': np.nan,
                    'r2': np.nan,
                    'hit_rate': np.nan,
                    'bias': np.nan,
                    'directional_accuracy': np.nan
                }
                continue
            
            actual = valid_data['actual_vol'].values
            forecast = valid_data['forecast_vol'].values
            actual_var = valid_data['actual_var'].values
            forecast_var = valid_data['forecast_var'].values
            
            # Basic metrics
            mse = np.mean((actual - forecast) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual - forecast))
            bias = np.mean(forecast - actual)
            
            # R-squared
            ss_res = np.sum((actual - forecast) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            # QLIKE (Quasi-Likelihood) - preferred for volatility forecasting
            qlike = np.mean(actual_var / forecast_var - np.log(actual_var / forecast_var) - 1)
            
            # Hit rate (percentage of forecasts within 10% of actual)
            hit_rate = np.mean(np.abs(forecast - actual) / actual <= 0.1)
            
            # Directional accuracy
            if len(valid_data) > 1:
                actual_changes = np.diff(actual) > 0
                forecast_changes = np.diff(forecast) > 0
                directional_accuracy = np.mean(actual_changes == forecast_changes)
            else:
                directional_accuracy = np.nan
            
            metrics[model_name] = {
                'n_forecasts': len(valid_data),
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'qlike': qlike,
                'r2': r2,
                'hit_rate': hit_rate,
                'bias': bias,
                'directional_accuracy': directional_accuracy
            }
        
        return metrics
    
    def diebold_mariano_test(
        self,
        evaluation_results: Dict[str, pd.DataFrame],
        model1: str,
        model2: str,
        loss_function: str = 'mse'
    ) -> Dict[str, float]:
        """
        Perform Diebold-Mariano test for forecast comparison.
        
        Parameters:
        -----------
        evaluation_results : dict
            Results from evaluation
        model1 : str
            Name of first model
        model2 : str
            Name of second model
        loss_function : str
            Loss function to use ('mse', 'mae', 'qlike')
            
        Returns:
        --------
        dict : Test statistics and p-value
        """
        if model1 not in evaluation_results or model2 not in evaluation_results:
            raise ValueError(f"Models {model1} or {model2} not found in evaluation results")
        
        df1 = evaluation_results[model1].dropna(subset=['actual_vol', 'forecast_vol'])
        df2 = evaluation_results[model2].dropna(subset=['actual_vol', 'forecast_vol'])
        
        # Align data by date
        common_dates = df1.index.intersection(df2.index)
        
        if len(common_dates) < 10:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'n_observations': len(common_dates),
                'interpretation': 'Insufficient common observations'
            }
        
        df1_aligned = df1.loc[common_dates]
        df2_aligned = df2.loc[common_dates]
        
        actual = df1_aligned['actual_vol'].values
        forecast1 = df1_aligned['forecast_vol'].values
        forecast2 = df2_aligned['forecast_vol'].values
        
        # Calculate loss differential
        if loss_function == 'mse':
            loss1 = (actual - forecast1) ** 2
            loss2 = (actual - forecast2) ** 2
        elif loss_function == 'mae':
            loss1 = np.abs(actual - forecast1)
            loss2 = np.abs(actual - forecast2)
        elif loss_function == 'qlike':
            actual_var = df1_aligned['actual_var'].values
            forecast1_var = df1_aligned['forecast_var'].values
            forecast2_var = df2_aligned['forecast_var'].values
            loss1 = actual_var / forecast1_var - np.log(actual_var / forecast1_var) - 1
            loss2 = actual_var / forecast2_var - np.log(actual_var / forecast2_var) - 1
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        loss_diff = loss1 - loss2
        
        # DM test statistic
        d_bar = np.mean(loss_diff)
        n = len(loss_diff)
        
        # Calculate variance with potential autocorrelation adjustment
        gamma0 = np.var(loss_diff, ddof=1)
        
        # Simple version without autocorrelation adjustment
        dm_stat = d_bar / np.sqrt(gamma0 / n)
        
        # Two-sided test
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        # Interpretation
        if p_value < 0.01:
            significance = "highly significant (1%)"
        elif p_value < 0.05:
            significance = "significant (5%)"
        elif p_value < 0.10:
            significance = "marginally significant (10%)"
        else:
            significance = "not significant"
        
        if dm_stat > 0:
            interpretation = f"{model2} outperforms {model1} ({significance})"
        else:
            interpretation = f"{model1} outperforms {model2} ({significance})"
        
        return {
            'statistic': dm_stat,
            'p_value': p_value,
            'n_observations': n,
            'interpretation': interpretation,
            'mean_loss_diff': d_bar
        }
    
    def model_confidence_set(
        self,
        evaluation_results: Dict[str, pd.DataFrame],
        confidence_level: float = 0.10
    ) -> Dict[str, Any]:
        """
        Perform Model Confidence Set (MCS) procedure to identify best models.
        
        Parameters:
        -----------
        evaluation_results : dict
            Results from evaluation
        confidence_level : float
            Confidence level for the test (default 10%)
            
        Returns:
        --------
        dict : MCS results including best models and p-values
        """
        model_names = list(evaluation_results.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            return {'best_models': model_names, 'p_values': {}, 'eliminated_order': []}
        
        # Calculate pairwise DM tests
        dm_results = {}
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    try:
                        dm_result = self.diebold_mariano_test(
                            evaluation_results, model1, model2, 'qlike'
                        )
                        dm_results[(model1, model2)] = dm_result
                    except:
                        continue
        
        # Simple MCS approximation - identify models that are not significantly worse
        remaining_models = set(model_names)
        eliminated_order = []
        p_values = {}
        
        # Calculate average performance for ranking
        avg_performance = {}
        for model_name in model_names:
            df = evaluation_results[model_name].dropna(subset=['actual_vol', 'forecast_vol'])
            if len(df) > 0:
                avg_qlike = np.mean(df['actual_var'] / df['forecast_var'] - 
                                  np.log(df['actual_var'] / df['forecast_var']) - 1)
                avg_performance[model_name] = avg_qlike
            else:
                avg_performance[model_name] = np.inf
        
        # Sort by performance
        sorted_models = sorted(avg_performance.keys(), key=lambda x: avg_performance[x])
        
        # Simple elimination: remove models significantly worse than the best
        best_model = sorted_models[0]
        
        for model in sorted_models[1:]:
            if (best_model, model) in dm_results:
                dm_result = dm_results[(best_model, model)]
            elif (model, best_model) in dm_results:
                dm_result = dm_results[(model, best_model)]
                dm_result['statistic'] *= -1  # Flip sign
            else:
                continue
            
            p_values[model] = dm_result['p_value']
            
            if dm_result['p_value'] < confidence_level:
                eliminated_order.append(model)
                remaining_models.discard(model)
        
        return {
            'best_models': list(remaining_models),
            'eliminated_models': eliminated_order,
            'p_values': p_values,
            'confidence_level': confidence_level,
            'avg_performance': avg_performance
        }
    
    def _calculate_realized_volatility(
        self,
        returns: pd.Series,
        horizon: int = 1
    ) -> pd.Series:
        """Calculate realized volatility for the given horizon."""
        if horizon == 1:
            # Simple squared returns for one-step-ahead
            realized_vol = np.sqrt(returns ** 2 * self.annualization_factor)
        else:
            # Rolling sum of squared returns for multi-step
            rolling_var = (returns ** 2).rolling(window=horizon, min_periods=1).sum()
            realized_vol = np.sqrt(rolling_var * self.annualization_factor / horizon)
        
        return realized_vol
    
    def _fit_models(
        self,
        models: Dict[str, Any],
        train_data: pd.Series
    ) -> Dict[str, Any]:
        """Fit all models on training data."""
        fitted_models = {}
        
        for name, model in models.items():
            try:
                # Create a copy of the model to avoid modifying original
                model_copy = type(model)(**model.__dict__)
                model_copy.fit(train_data.values)
                fitted_models[name] = model_copy
            except Exception as e:
                print(f"Warning: Failed to fit model {name}: {str(e)}")
                fitted_models[name] = None
        
        return fitted_models
    
    def _get_model_forecast(
        self,
        model: Any,
        horizon: int
    ) -> Dict[str, np.ndarray]:
        """Get forecast from a fitted model."""
        try:
            if hasattr(model, 'forecast'):
                return model.forecast(horizon=horizon)
            else:
                # Fallback for models without forecast method
                return {'volatility': np.array([np.nan])}
        except Exception as e:
            return {'volatility': np.array([np.nan])}
    
    def summary_report(
        self,
        evaluation_results: Dict[str, pd.DataFrame],
        include_dm_tests: bool = True
    ) -> str:
        """
        Generate comprehensive summary report of evaluation results.
        
        Parameters:
        -----------
        evaluation_results : dict
            Results from evaluation
        include_dm_tests : bool
            Whether to include Diebold-Mariano test results
            
        Returns:
        --------
        str : Formatted summary report
        """
        metrics = self.calculate_forecast_metrics(evaluation_results)
        
        report = ["=" * 80]
        report.append("VOLATILITY FORECAST EVALUATION SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Model performance summary
        report.append("MODEL PERFORMANCE METRICS")
        report.append("-" * 40)
        
        # Create performance table
        model_names = list(metrics.keys())
        if model_names:
            # Header
            header = f"{'Model':<15} {'N':<6} {'RMSE':<8} {'QLIKE':<8} {'R²':<8} {'Hit Rate':<8} {'Dir Acc':<8}"
            report.append(header)
            report.append("-" * len(header))
            
            # Sort by RMSE
            sorted_models = sorted(model_names, 
                                 key=lambda x: metrics[x]['rmse'] if not np.isnan(metrics[x]['rmse']) else float('inf'))
            
            for model in sorted_models:
                m = metrics[model]
                row = f"{model:<15} {m['n_forecasts']:<6} {m['rmse']:<8.4f} {m['qlike']:<8.4f} {m['r2']:<8.4f} {m['hit_rate']:<8.4f} {m['directional_accuracy']:<8.4f}"
                report.append(row)
        
        report.append("")
        
        # Diebold-Mariano tests if requested
        if include_dm_tests and len(model_names) > 1:
            report.append("DIEBOLD-MARIANO PAIRWISE TESTS (QLIKE Loss)")
            report.append("-" * 50)
            
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    try:
                        dm_result = self.diebold_mariano_test(
                            evaluation_results, model1, model2, 'qlike'
                        )
                        report.append(f"{model1} vs {model2}:")
                        report.append(f"  Statistic: {dm_result['statistic']:.4f}")
                        report.append(f"  P-value: {dm_result['p_value']:.4f}")
                        report.append(f"  {dm_result['interpretation']}")
                        report.append("")
                    except Exception as e:
                        report.append(f"{model1} vs {model2}: Test failed ({str(e)})")
                        report.append("")
        
        # Model Confidence Set
        try:
            mcs_result = self.model_confidence_set(evaluation_results)
            report.append("MODEL CONFIDENCE SET (90% confidence level)")
            report.append("-" * 45)
            report.append(f"Best models: {', '.join(mcs_result['best_models'])}")
            if mcs_result['eliminated_models']:
                report.append(f"Eliminated models: {', '.join(mcs_result['eliminated_models'])}")
            report.append("")
        except Exception as e:
            report.append(f"Model Confidence Set: Failed ({str(e)})")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)