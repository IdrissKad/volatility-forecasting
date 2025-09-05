import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class VolatilityVisualizer:
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = px.colors.qualitative.Plotly
        
    def plot_volatility_comparison(
        self,
        data: pd.DataFrame,
        models: List[str],
        realized_col: str = 'Realized_Vol',
        title: str = 'Volatility Comparison',
        save_path: Optional[str] = None
    ) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Volatility Forecasts', 'Forecast Errors'),
            row_heights=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[realized_col],
                name='Realized',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        for i, model in enumerate(models):
            if model in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[model],
                        name=model,
                        line=dict(color=self.colors[i % len(self.colors)])
                    ),
                    row=1, col=1
                )
                
                error = data[model] - data[realized_col]
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=error,
                        name=f'{model} Error',
                        line=dict(color=self.colors[i % len(self.colors)])
                    ),
                    row=2, col=1
                )
        
        fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray")
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volatility", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_volatility_surface(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        implied_vols: np.ndarray,
        title: str = 'Implied Volatility Surface',
        save_path: Optional[str] = None
    ) -> go.Figure:
        fig = go.Figure(data=[
            go.Surface(
                x=strikes,
                y=expiries,
                z=implied_vols,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Implied Vol')
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Time to Expiry',
                zaxis_title='Implied Volatility',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_volatility_smile(
        self,
        strikes: pd.Series,
        implied_vols: pd.Series,
        spot_price: float,
        title: str = 'Volatility Smile',
        save_path: Optional[str] = None
    ) -> go.Figure:
        moneyness = strikes / spot_price
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=moneyness,
                y=implied_vols,
                mode='lines+markers',
                name='Implied Vol',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            )
        )
        
        fig.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="red",
            annotation_text="ATM"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Implied Volatility',
            height=500,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_delta_hedge_performance(
        self,
        results: Dict,
        title: str = 'Delta Hedging Performance',
        save_path: Optional[str] = None
    ) -> go.Figure:
        positions = results['positions']
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Spot Price', 'Delta', 'P&L', 'Greeks'),
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        fig.add_trace(
            go.Scatter(
                x=positions['date'],
                y=positions['spot'],
                name='Spot',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=positions['date'],
                y=positions['delta'],
                name='Delta',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=positions['date'],
                y=positions['cumulative_pnl'],
                name='Cumulative P&L',
                line=dict(color='red'),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=positions['date'],
                y=positions['gamma'],
                name='Gamma',
                line=dict(color='purple')
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=positions['date'],
                y=positions['vega'],
                name='Vega',
                line=dict(color='orange'),
                yaxis='y2'
            ),
            row=4, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Delta", row=2, col=1)
        fig.update_yaxes(title_text="P&L", row=3, col=1)
        fig.update_yaxes(title_text="Greeks", row=4, col=1)
        
        fig.update_layout(
            title=title,
            height=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_backtest_results(
        self,
        results: Dict,
        title: str = 'Backtest Results',
        save_path: Optional[str] = None
    ) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Value', 'Returns Distribution',
                'Drawdown', 'Position History',
                'Volatility Forecast vs Realized', 'Trade Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        for model_name, model_results in results['individual_results'].items():
            portfolio = model_results['portfolio']
            positions = model_results['positions']
            trades = model_results['trades']
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=portfolio['portfolio_value'],
                    name=model_name,
                    mode='lines'
                ),
                row=1, col=1
            )
            
            returns = portfolio['portfolio_value'].pct_change().dropna()
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name=model_name,
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=2
            )
            
            rolling_max = portfolio['portfolio_value'].cummax()
            drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=drawdown,
                    name=model_name,
                    mode='lines',
                    fill='tozeroy'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=positions.index,
                    y=positions['position'],
                    name=model_name,
                    mode='lines'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=positions.index,
                    y=positions['forecast_vol'],
                    name=f'{model_name} Forecast',
                    mode='lines'
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=positions.index,
                    y=positions['realized_vol'],
                    name='Realized',
                    mode='lines',
                    line=dict(dash='dash')
                ),
                row=3, col=1
            )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_yaxes(title_text="Position", row=2, col=2)
        fig.update_yaxes(title_text="Volatility", row=3, col=1)
        
        fig.update_layout(
            title=title,
            height=1200,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_metrics_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['total_return', 'sharpe_ratio', 'max_drawdown'],
        title: str = 'Strategy Metrics Comparison',
        save_path: Optional[str] = None
    ) -> go.Figure:
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            specs=[[{"type": "bar"}] * len(metrics)]
        )
        
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    x=comparison_df.index,
                    y=comparison_df[metric],
                    name=metric,
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_report(
        self,
        results: Dict,
        save_path: str = 'volatility_report.html'
    ) -> None:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        html_content = f"""
        <html>
        <head>
            <title>Volatility Forecasting Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Volatility Forecasting and Trading Report</h1>
            
            <h2>Performance Metrics</h2>
            {results['comparison'].to_html()}
            
            <h2>Key Findings</h2>
            <ul>
                <li>Best performing model: {results['comparison']['total_return'].idxmax()}</li>
                <li>Highest Sharpe ratio: {results['comparison']['sharpe_ratio'].idxmax()}</li>
                <li>Lowest drawdown: {results['comparison']['max_drawdown'].idxmax()}</li>
            </ul>
            
            <h2>Detailed Results</h2>
            <p>See interactive charts below for detailed analysis.</p>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)