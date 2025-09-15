#!/usr/bin/env python3
"""
Command-line interface for volatility forecasting and delta-hedging backtests.

Usage:
    python -m src.cli backtest --ticker SPY --start 2018-01-01
    python -m src.cli forecast --ticker AAPL --models GARCH,EGARCH
    python -m src.cli hedge --ticker SPY --strike 400 --expiry 0.25
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yaml
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader
from models.garch_model import GARCHModel, EGARCHModel, GJRGARCHModel
from models.realized_volatility import RealizedVolatility
from strategies.delta_hedging import DeltaHedgingStrategy
from evaluation.forecast_evaluation import VolatilityForecastEvaluator
from backtesting.backtest_engine import VolatilityBacktester
from utils.visualization import VolatilityVisualizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'default_config.yaml'
    
    if not Path(config_path).exists():
        # Return default configuration
        return {
            'data': {
                'default_start': '2018-01-01',
                'default_end': None,
                'cache_dir': 'data/cache'
            },
            'models': {
                'garch': {'p': 1, 'q': 1, 'dist': 'normal'},
                'egarch': {'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'}
            },
            'evaluation': {
                'initial_window': 252,
                'refit_frequency': 22,
                'min_observations': 100
            },
            'hedging': {
                'risk_free_rate': 0.05,
                'transaction_cost': 0.001,
                'rebalance_frequency': 'daily'
            },
            'output': {
                'results_dir': 'results',
                'save_plots': True,
                'plot_format': 'png'
            }
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def backtest_volatility_models(args, config: dict, logger: logging.Logger):
    """Run volatility forecasting backtest."""
    logger.info(f"Starting volatility backtest for {args.ticker}")
    
    # Load data
    logger.info(f"Loading data for {args.ticker} from {args.start} to {args.end}")
    try:
        import yfinance as yf
        data = yf.download(args.ticker, start=args.start, end=args.end, progress=False)
        if data.empty:
            raise ValueError(f"No data found for {args.ticker}")
        
        data['Returns'] = data['Adj Close'].pct_change()
        data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        data = data.dropna()
        
        logger.info(f"Loaded {len(data)} observations")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Initialize models
    models = {}
    model_list = args.models.split(',') if args.models else ['GARCH', 'EGARCH']
    
    for model_name in model_list:
        model_name = model_name.strip().upper()
        if model_name == 'GARCH':
            models['GARCH(1,1)'] = GARCHModel(p=1, q=1, dist='normal')
        elif model_name == 'EGARCH':
            models['EGARCH(1,1,1)'] = EGARCHModel(p=1, o=1, q=1, dist='normal')
        elif model_name == 'GJR':
            models['GJR-GARCH(1,1,1)'] = GJRGARCHModel(p=1, o=1, q=1, dist='normal')
        else:
            logger.warning(f"Unknown model: {model_name}")
    
    logger.info(f"Initialized models: {list(models.keys())}")
    
    # Run evaluation
    evaluator = VolatilityForecastEvaluator(annualization_factor=252)
    
    logger.info("Running walk-forward evaluation...")
    try:
        evaluation_results = evaluator.walk_forward_evaluation(
            returns=data['Log_Returns'],
            models=models,
            initial_window=config['evaluation']['initial_window'],
            step_size=1,
            forecast_horizon=1,
            refit_frequency=config['evaluation']['refit_frequency'],
            min_observations=config['evaluation']['min_observations']
        )
        
        # Calculate metrics
        metrics = evaluator.calculate_forecast_metrics(evaluation_results)
        
        logger.info("Evaluation completed successfully")
        
        # Display results
        print("\\n" + "=" * 60)
        print("VOLATILITY FORECASTING BACKTEST RESULTS")
        print("=" * 60)
        
        metrics_df = pd.DataFrame(metrics).T
        key_metrics = ['n_forecasts', 'rmse', 'qlike', 'r2', 'hit_rate']
        print(metrics_df[key_metrics].round(6))
        
        # Find best model
        best_model = metrics_df['qlike'].idxmin()
        print(f"\\nBest model (lowest QLIKE): {best_model}")
        print(f"QLIKE: {metrics_df.loc[best_model, 'qlike']:.6f}")
        print(f"RMSE: {metrics_df.loc[best_model, 'rmse']:.6f}")
        
        # Generate summary report
        if args.detailed:
            summary_report = evaluator.summary_report(evaluation_results, include_dm_tests=True)
            print("\\n" + summary_report)
        
        # Save results
        output_dir = Path(config['output']['results_dir']) / f"{args.ticker}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results
        for model_name, results_df in evaluation_results.items():
            filename = output_dir / f"forecast_{model_name.replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')}.csv"
            results_df.to_csv(filename)
        
        # Save metrics
        metrics_df.to_csv(output_dir / 'forecast_metrics.csv')
        
        # Save summary report
        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write(evaluator.summary_report(evaluation_results, include_dm_tests=True))
        
        logger.info(f"Results saved to: {output_dir}")
        print(f"\\nResults saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0


def forecast_volatility(args, config: dict, logger: logging.Logger):
    """Generate volatility forecasts."""
    logger.info(f"Generating volatility forecasts for {args.ticker}")
    
    # Load data
    try:
        import yfinance as yf
        data = yf.download(args.ticker, start=args.start, end=args.end, progress=False)
        data['Returns'] = data['Adj Close'].pct_change()
        data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        data = data.dropna()
        
        logger.info(f"Loaded {len(data)} observations")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Initialize and fit model
    model_name = args.model.upper()
    
    if model_name == 'GARCH':
        model = GARCHModel(p=1, q=1, dist='normal')
    elif model_name == 'EGARCH':
        model = EGARCHModel(p=1, o=1, q=1, dist='normal')
    elif model_name == 'GJR':
        model = GJRGARCHModel(p=1, o=1, q=1, dist='normal')
    else:
        logger.error(f"Unknown model: {model_name}")
        return 1
    
    try:
        logger.info(f"Fitting {model_name} model...")
        model.fit(data['Log_Returns'].values)
        
        # Generate forecasts
        horizon = args.horizon
        forecast = model.forecast(horizon=horizon)
        
        print(f"\\n{model_name} Volatility Forecasts for {args.ticker}:")
        print("=" * 50)
        
        for h in range(horizon):
            vol_forecast = forecast['volatility'][h]
            print(f"{h+1}-step ahead: {vol_forecast:.4f} ({vol_forecast:.2%} annualized)")
        
        # Current conditional volatility
        current_vol = model.conditional_volatility().iloc[-1] * np.sqrt(252)
        print(f"\\nCurrent conditional volatility: {current_vol:.2%}")
        
        # Model information
        if args.detailed:
            info = model.get_model_info()
            print(f"\\nModel Information:")
            print(f"AIC: {info['aic']:.2f}")
            print(f"BIC: {info['bic']:.2f}")
            print(f"Log-likelihood: {info['log_likelihood']:.2f}")
            
            print(f"\\nParameters:")
            for param, value in info['params'].items():
                print(f"  {param}: {value:.6f}")
        
        # Save forecast
        output_dir = Path(config['output']['results_dir']) / 'forecasts'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        forecast_df = pd.DataFrame({
            'horizon': range(1, horizon + 1),
            'volatility_forecast': forecast['volatility'],
            'variance_forecast': forecast['variance']
        })
        
        filename = output_dir / f"{args.ticker}_{model_name}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        forecast_df.to_csv(filename, index=False)
        
        logger.info(f"Forecast saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        return 1
    
    return 0


def hedge_simulation(args, config: dict, logger: logging.Logger):
    """Run delta-hedging simulation."""
    logger.info(f"Running delta-hedging simulation for {args.ticker}")
    
    # Load data
    try:
        import yfinance as yf
        data = yf.download(args.ticker, start=args.start, end=args.end, progress=False)
        data = data.dropna()
        
        logger.info(f"Loaded {len(data)} observations")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Initialize hedging strategy
    hedger = DeltaHedgingStrategy(
        risk_free_rate=config['hedging']['risk_free_rate'],
        transaction_cost=args.transaction_cost or config['hedging']['transaction_cost'],
        rebalance_frequency=config['hedging']['rebalance_frequency']
    )
    
    try:
        # Run simulation
        logger.info(f"Running delta-hedging simulation...")
        logger.info(f"  Strike: {args.strike}")
        logger.info(f"  Expiry: {args.expiry} years")
        logger.info(f"  Volatility: {args.volatility:.2%}")
        logger.info(f"  Transaction cost: {hedger.transaction_cost:.4f}")
        
        results = hedger.simulate_delta_hedging(
            spot_prices=data['Adj Close'],
            strike=args.strike,
            expiry=args.expiry,
            volatility=args.volatility,
            option_type=args.option_type or 'call'
        )
        
        # Analyze performance
        performance = hedger.analyze_hedge_performance(results)
        
        print(f"\\nDelta-Hedging Simulation Results for {args.ticker}:")
        print("=" * 55)
        print(f"Initial option price: ${results['initial_option_price']:.3f}")
        print(f"Final option payoff: ${results['option_payoff']:.3f}")
        print(f"Total P&L: ${results['total_pnl']:.3f}")
        print(f"Hedge P&L: ${results['hedge_pnl']:.3f}")
        print(f"Option P&L: ${results['option_pnl']:.3f}")
        print(f"Transaction costs: ${results['transaction_costs']:.3f}")
        print(f"Hedge effectiveness: {results['hedge_effectiveness']:.2%}")
        
        print(f"\\nPerformance Metrics:")
        print(f"Total shares traded: {performance['total_shares_traded']:.0f}")
        print(f"Number of rebalances: {performance['n_rebalances']}")
        print(f"Average delta: {performance['avg_delta']:.4f}")
        print(f"Max delta: {performance['max_delta']:.4f}")
        print(f"Min delta: {performance['min_delta']:.4f}")
        print(f"Sharpe ratio: {performance['sharpe_ratio']:.4f}")
        
        # Save results
        output_dir = Path(config['output']['results_dir']) / 'hedging'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        positions_df = results['positions']
        filename = output_dir / f"{args.ticker}_hedge_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        positions_df.to_csv(filename)
        
        # Save summary
        summary = {
            'ticker': args.ticker,
            'strike': args.strike,
            'expiry': args.expiry,
            'volatility': args.volatility,
            'option_type': args.option_type or 'call',
            **{f'result_{k}': v for k, v in results.items() if isinstance(v, (int, float))},
            **{f'perf_{k}': v for k, v in performance.items() if isinstance(v, (int, float))}
        }
        
        summary_df = pd.DataFrame([summary])
        summary_filename = output_dir / f"{args.ticker}_hedge_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_filename, index=False)
        
        logger.info(f"Results saved to: {output_dir}")
        print(f"\\nResults saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Hedging simulation failed: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Volatility Forecasting and Delta-Hedging CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s backtest --ticker SPY --start 2018-01-01
  %(prog)s forecast --ticker AAPL --model GARCH --horizon 5
  %(prog)s hedge --ticker SPY --strike 400 --expiry 0.25 --volatility 0.20
        """
    )
    
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run volatility model backtest')
    backtest_parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    backtest_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--models', type=str, help='Comma-separated model names (GARCH,EGARCH,GJR)')
    backtest_parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate volatility forecasts')
    forecast_parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    forecast_parser.add_argument('--model', type=str, default='GARCH', 
                               choices=['GARCH', 'EGARCH', 'GJR'], help='Model type')
    forecast_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    forecast_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    forecast_parser.add_argument('--horizon', type=int, default=1, help='Forecast horizon')
    forecast_parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    
    # Hedge command
    hedge_parser = subparsers.add_parser('hedge', help='Run delta-hedging simulation')
    hedge_parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    hedge_parser.add_argument('--strike', type=float, required=True, help='Option strike price')
    hedge_parser.add_argument('--expiry', type=float, required=True, help='Time to expiry (years)')
    hedge_parser.add_argument('--volatility', type=float, required=True, help='Volatility assumption')
    hedge_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    hedge_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    hedge_parser.add_argument('--option-type', type=str, default='call', 
                            choices=['call', 'put'], help='Option type')
    hedge_parser.add_argument('--transaction-cost', type=float, help='Transaction cost rate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.debug(f"Loaded configuration: {config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Set default dates if not provided
    if not hasattr(args, 'start') or args.start is None:
        args.start = config['data']['default_start']
    if not hasattr(args, 'end') or args.end is None:
        args.end = config['data'].get('default_end', datetime.now().strftime('%Y-%m-%d'))
    
    # Execute command
    try:
        if args.command == 'backtest':
            return backtest_volatility_models(args, config, logger)
        elif args.command == 'forecast':
            return forecast_volatility(args, config, logger)
        elif args.command == 'hedge':
            return hedge_simulation(args, config, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())