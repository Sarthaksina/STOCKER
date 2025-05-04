"""
Portfolio Visualization Module for STOCKER Pro

This module provides visualization capabilities for portfolio analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

from stocker.cloud.portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioVisualizer:
    """
    Portfolio visualization functionality
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize portfolio visualizer
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
        self.use_interactive = True  # Default to interactive visualizations
        
    def toggle_interactive(self, use_interactive: bool = True) -> None:
        """
        Toggle between interactive (Plotly) and static (Matplotlib) visualizations
        
        Args:
            use_interactive: Whether to use interactive visualizations
        """
        self.use_interactive = use_interactive
        logger.info(f"Set interactive visualizations to: {use_interactive}")
    
    def plot_portfolio_composition(self, 
                                  weights: Dict[str, float],
                                  save_path: Optional[str] = None) -> Any:
        """
        Plot portfolio composition as a pie chart
        
        Args:
            weights: Dictionary mapping asset names to weights
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure or Plotly figure
        """
        if self.use_interactive:
            # Create interactive Plotly pie chart
            fig = px.pie(
                names=list(weights.keys()),
                values=list(weights.values()),
                title="Portfolio Composition",
                hover_data=[f"{w*100:.2f}%" for w in weights.values()],
                labels={'names': 'Asset', 'values': 'Weight'},
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                title_font_size=20,
                legend_title_font_size=16,
                legend_font_size=14
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Saved interactive portfolio composition chart to {save_path}")
                
            return fig
        else:
            # Create static Matplotlib pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(
                weights.values(), 
                labels=weights.keys(), 
                autopct='%1.1f%%',
                startangle=90,
                shadow=False
            )
            ax.axis('equal')
            plt.title('Portfolio Composition', fontsize=16)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Saved portfolio composition chart to {save_path}")
                
            return fig
    
    def plot_performance_comparison(self,
                                   portfolio_values: pd.Series,
                                   benchmark_values: Optional[pd.Series] = None,
                                   save_path: Optional[str] = None) -> Any:
        """
        Plot portfolio performance compared to benchmark
        
        Args:
            portfolio_values: Series of portfolio values over time
            benchmark_values: Optional series of benchmark values over time
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure or Plotly figure
        """
        if self.use_interactive:
            # Create interactive Plotly line chart
            fig = go.Figure()
            
            # Normalize values to start at 100
            port_norm = 100 * portfolio_values / portfolio_values.iloc[0]
            
            fig.add_trace(go.Scatter(
                x=port_norm.index,
                y=port_norm,
                mode='lines',
                name='Portfolio',
                line=dict(width=2, color='blue')
            ))
            
            if benchmark_values is not None:
                bench_norm = 100 * benchmark_values / benchmark_values.iloc[0]
                fig.add_trace(go.Scatter(
                    x=bench_norm.index,
                    y=bench_norm,
                    mode='lines',
                    name='Benchmark',
                    line=dict(width=2, color='red', dash='dash')
                ))
            
            fig.update_layout(
                title='Portfolio Performance Comparison (Base = 100)',
                xaxis_title='Date',
                yaxis_title='Value',
                legend_title='Assets',
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Add range slider
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Saved interactive performance comparison chart to {save_path}")
                
            return fig
        else:
            # Create static Matplotlib line chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Normalize values to start at 100
            port_norm = 100 * portfolio_values / portfolio_values.iloc[0]
            port_norm.plot(ax=ax, label='Portfolio', linewidth=2)
            
            if benchmark_values is not None:
                bench_norm = 100 * benchmark_values / benchmark_values.iloc[0]
                bench_norm.plot(ax=ax, label='Benchmark', linewidth=2, linestyle='--')
            
            ax.set_title('Portfolio Performance Comparison (Base = 100)', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Saved performance comparison chart to {save_path}")
                
            return fig

    # Additional interactive visualization methods would be added here
    def plot_backtest_results(self, 
                             backtest_results: Dict[str, Any],
                             save_path: Optional[str] = None,
                             interactive: Optional[bool] = None) -> Any:
        """
        Plot backtest results
        
        Args:
            backtest_results: Dictionary with backtest results
            save_path: Optional path to save the plot
            interactive: Whether to use interactive Plotly (overrides default)
            
        Returns:
            Matplotlib figure or Plotly figure
        """
        use_plotly = self.use_plotly if interactive is None else interactive
        
        if use_plotly:
            return self._plot_backtest_results_plotly(backtest_results, save_path)
        else:
            return self._plot_backtest_results_mpl(backtest_results, save_path)
    
    def _plot_backtest_results_plotly(self,
                                     backtest_results: Dict[str, Any],
                                     save_path: Optional[str] = None) -> go.Figure:
        """Plot backtest results using Plotly"""
        portfolio_values = backtest_results['portfolio_values']
        benchmark_values = backtest_results.get('benchmark_values')
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add portfolio value line
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                name="Portfolio Value",
                line=dict(color='blue', width=2)
            ),
            secondary_y=False,
        )
        
        # Add benchmark if available
        if benchmark_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_values.index,
                    y=benchmark_values.values,
                    name="Benchmark Value",
                    line=dict(color='gray', width=2, dash='dash')
                ),
                secondary_y=False,
            )
        
        # Add drawdown on secondary axis
        if 'drawdowns' in backtest_results:
            drawdowns = backtest_results['drawdowns']
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns.values * 100,  # Convert to percentage
                    name="Drawdown",
                    line=dict(color='red', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)'
                ),
                secondary_y=True,
            )
        
        # Add annotations for key metrics
        annotations = []
        metrics = [
            ('Final Value', f"${backtest_results['final_value']:.2f}"),
            ('Total Return', f"{backtest_results['total_return']*100:.2f}%"),
            ('Annual Return', f"{backtest_results['annual_return']*100:.2f}%"),
            ('Sharpe Ratio', f"{backtest_results['sharpe_ratio']:.2f}"),
            ('Max Drawdown', f"{backtest_results['max_drawdown']*100:.2f}%")
        ]
        
        # Set up the layout
        fig.update_layout(
            title="Portfolio Backtest Results",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            yaxis2_title="Drawdown (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            height=600,
            width=1000,
            margin=dict(l=50, r=50, t=80, b=50),
            template="plotly_white"
        )
        
        # Add a table with metrics
        table_data = [
            ['Metric', 'Value'],
        ]
        for metric, value in metrics:
            table_data.append([metric, value])
        
        table = go.Figure(data=[go.Table(
            header=dict(
                values=table_data[0],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[[row[0] for row in table_data[1:]], 
                         [row[1] for row in table_data[1:]]],
                fill_color='lavender',
                align='left'
            )
        )])
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            table.write_html(save_path.replace('.html', '_metrics.html'))
        
        return fig
    
    def _plot_backtest_results_mpl(self,
                                  backtest_results: Dict[str, Any],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot backtest results using Matplotlib (for static reports)"""
        # ... existing matplotlib implementation ...
        # ... existing code ...
        
        return fig
    
    def plot_efficient_frontier(self,
                               optimization_results: Dict[str, Any],
                               save_path: Optional[str] = None,
                               interactive: Optional[bool] = None) -> Any:
        """
        Plot efficient frontier
        
        Args:
            optimization_results: Dictionary with optimization results
            save_path: Optional path to save the plot
            interactive: Whether to use interactive Plotly (overrides default)
            
        Returns:
            Matplotlib figure or Plotly figure
        """
        use_plotly = self.use_plotly if interactive is None else interactive
        
        if use_plotly:
            return self._plot_efficient_frontier_plotly(optimization_results, save_path)
        else:
            return self._plot_efficient_frontier_mpl(optimization_results, save_path)
    
    def _plot_efficient_frontier_plotly(self,
                                       optimization_results: Dict[str, Any],
                                       save_path: Optional[str] = None) -> go.Figure:
        """Plot efficient frontier using Plotly"""
        frontier_returns = optimization_results['frontier_returns']
        frontier_risks = optimization_results['frontier_risks']
        optimal_return = optimization_results.get('optimal_return')
        optimal_risk = optimization_results.get('optimal_risk')
        
        # Create scatter plot for efficient frontier
        fig = go.Figure()
        
        # Add efficient frontier line
        fig.add_trace(
            go.Scatter(
                x=frontier_risks,
                y=frontier_returns,
                mode='lines+markers',
                name='Efficient Frontier',
                marker=dict(
                    size=6,
                    color=np.arange(len(frontier_risks)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Portfolio Index")
                ),
                hovertemplate='Risk: %{x:.4f}<br>Return: %{y:.4f}'
            )
        )
        
        # Add optimal portfolio if available
        if optimal_return is not None and optimal_risk is not None:
            fig.add_trace(
                go.Scatter(
                    x=[optimal_risk],
                    y=[optimal_return],
                    mode='markers',
                    name='Optimal Portfolio',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='star'
                    ),
                    hovertemplate='Risk: %{x:.4f}<br>Return: %{y:.4f}'
                )
            )
        
        # Add individual assets if available
        if 'asset_returns' in optimization_results and 'asset_risks' in optimization_results:
            asset_returns = optimization_results['asset_returns']
            asset_risks = optimization_results['asset_risks']
            asset_names = optimization_results.get('asset_names', [f"Asset {i}" for i in range(len(asset_returns))])
            
            fig.add_trace(
                go.Scatter(
                    x=asset_risks,
                    y=asset_returns,
                    mode='markers+text',
                    name='Individual Assets',
                    marker=dict(
                        color='blue',
                        size=10,
                        line=dict(
                            color='black',
                            width=1
                        )
                    ),
                    text=asset_names,
                    textposition="top center",
                    hovertemplate='%{text}<br>Risk: %{x:.4f}<br>Return: %{y:.4f}'
                )
            )
        
        # Set up the layout
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            width=800,
            template="plotly_white"
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _plot_efficient_frontier_mpl(self,
                                    optimization_results: Dict[str, Any],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot efficient frontier using Matplotlib (for static reports)"""
        # ... existing matplotlib implementation ...
        # ... existing code ...
        
        return fig
    
    def plot_risk_metrics(self,
                         risk_metrics: Dict[str, Any],
                         save_path: Optional[str] = None,
                         interactive: Optional[bool] = None) -> Any:
        """
        Plot risk metrics
        
        Args:
            risk_metrics: Dictionary with risk metrics
            save_path: Optional path to save the plot
            interactive: Whether to use interactive Plotly (overrides default)
            
        Returns:
            Matplotlib figure or Plotly figure
        """
        use_plotly = self.use_plotly if interactive is None else interactive
        
        if use_plotly:
            return self._plot_risk_metrics_plotly(risk_metrics, save_path)
        else:
            return self._plot_risk_metrics_mpl(risk_metrics, save_path)
    
    def _plot_risk_metrics_plotly(self,
                                 risk_metrics: Dict[str, Any],
                                 save_path: Optional[str] = None) -> go.Figure:
        """Plot risk metrics using Plotly"""
        # Create a subplot with 2 rows and 2 columns
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=("Return Distribution", "Drawdown Analysis", 
                           "Risk Contribution", "Correlation Matrix")
        )
        
        # 1. Return Distribution (top left)
        if 'returns' in risk_metrics:
            returns = risk_metrics['returns']
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name="Returns",
                    nbinsx=30,
                    marker_color='blue',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Add VaR and CVaR lines
            if 'var_95' in risk_metrics and 'cvar_95' in risk_metrics:
                var_95 = risk_metrics['var_95']
                cvar_95 = risk_metrics['cvar_95']
                
                fig.add_vline(
                    x=-var_95,
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="95% VaR",
                    annotation_position="top right",
                    row=1, col=1
                )
                
                fig.add_vline(
                    x=-cvar_95,
                    line_width=2,
                    line_dash="dot",
                    line_color="darkred",
                    annotation_text="95% CVaR",
                    annotation_position="top right",
                    row=1, col=1
                )
        
        # 2. Drawdown Analysis (top right)
        if 'drawdowns' in risk_metrics:
            drawdowns = risk_metrics['drawdowns']
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns.values * 100,  # Convert to percentage
                    name="Drawdowns",
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(color='red', width=1)
                ),
                row=1, col=2
            )
        
        # 3. Risk Contribution (bottom left)
        if 'risk_contribution' in risk_metrics:
            risk_contrib = risk_metrics['risk_contribution']
            labels = list(risk_contrib.keys())
            values = list(risk_contrib.values())
            
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    name="Risk Contribution",
                    hole=0.3,
                    textinfo='label+percent',
                    marker=dict(
                        colors=px.colors.qualitative.Plotly
                    )
                ),
                row=2, col=1
            )
        
        # 4. Correlation Matrix (bottom right)
        if 'correlation_matrix' in risk_metrics:
            corr_matrix = risk_metrics['correlation_matrix']
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    name="Correlation"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Portfolio Risk Analysis",
            showlegend=False,
            template="plotly_white"
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _plot_risk_metrics_mpl(self,
                              risk_metrics: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot risk metrics using Matplotlib (for static reports)"""
        # ... existing matplotlib implementation ...
        # ... existing code ...
        
        return fig