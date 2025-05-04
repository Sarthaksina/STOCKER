"""
Portfolio Reporting Module for STOCKER Pro

This module provides functionality for generating comprehensive portfolio reports.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import jinja2
import pdfkit
import plotly.io as pio
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from stocker.cloud.portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioReporter:
    """
    Portfolio reporting functionality for generating PDF and HTML reports
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize portfolio reporter
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
        self.template_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'templates')
        self.output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')
        
        # Create directories if they don't exist
        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Create default template if it doesn't exist
        self._ensure_default_template_exists()
    
    def _ensure_default_template_exists(self) -> None:
        """Ensure the default report template exists"""
        default_template_path = os.path.join(self.template_dir, 'portfolio_report.html')
        
        if not os.path.exists(default_template_path):
            default_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{{ report_title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .section { margin-bottom: 30px; }
                    .metrics-table { width: 100%; border-collapse: collapse; }
                    .metrics-table th, .metrics-table td { 
                        border: 1px solid #ddd; padding: 8px; text-align: left; 
                    }
                    .metrics-table th { background-color: #f2f2f2; }
                    .chart-container { margin: 20px 0; text-align: center; }
                    .footer { text-align: center; margin-top: 50px; font-size: 0.8em; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ report_title }}</h1>
                    <p>Generated on {{ generation_date }}</p>
                </div>
                
                <div class="section">
                    <h2>Portfolio Summary</h2>
                    <p>{{ portfolio_summary }}</p>
                    
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        {% for metric, value in summary_metrics.items() %}
                        <tr>
                            <td>{{ metric }}</td>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Portfolio Composition</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{{ composition_chart }}" alt="Portfolio Composition">
                    </div>
                    
                    <h3>Holdings</h3>
                    <table class="metrics-table">
                        <tr>
                            <th>Asset</th>
                            <th>Weight</th>
                            <th>Value</th>
                            <th>Return</th>
                        </tr>
                        {% for asset in holdings %}
                        <tr>
                            <td>{{ asset.name }}</td>
                            <td>{{ asset.weight }}%</td>
                            <td>${{ asset.value }}</td>
                            <td>{{ asset.return }}%</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Performance Analysis</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{{ performance_chart }}" alt="Performance Chart">
                    </div>
                    
                    <h3>Performance Metrics</h3>
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Benchmark</th>
                        </tr>
                        {% for metric in performance_metrics %}
                        <tr>
                            <td>{{ metric.name }}</td>
                            <td>{{ metric.value }}</td>
                            <td>{{ metric.benchmark }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Risk Analysis</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{{ risk_chart }}" alt="Risk Chart">
                    </div>
                    
                    <h3>Risk Metrics</h3>
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        {% for metric in risk_metrics %}
                        <tr>
                            <td>{{ metric.name }}</td>
                            <td>{{ metric.value }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                {% if factor_analysis %}
                <div class="section">
                    <h2>Factor Analysis</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{{ factor_chart }}" alt="Factor Exposures">
                    </div>
                    
                    <h3>Factor Exposures</h3>
                    <table class="metrics-table">
                        <tr>
                            <th>Factor</th>
                            <th>Exposure</th>
                            <th>Significance</th>
                        </tr>
                        {% for factor in factor_exposures %}
                        <tr>
                            <td>{{ factor.name }}</td>
                            <td>{{ factor.exposure }}</td>
                            <td>{{ factor.significance }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
                
                <div class="footer">
                    <p>Generated by STOCKER Pro on {{ generation_date }}</p>
                </div>
            </body>
            </html>
            """
            
            with open(default_template_path, 'w') as f:
                f.write(default_template)
            
            logger.info(f"Created default report template at {default_template_path}")
    
    def _figure_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_str
    
    def _plotly_to_base64(self, fig) -> str:
        """Convert a plotly figure to base64 string"""
        img_bytes = pio.to_image(fig, format='png', width=800, height=500)
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        return img_str
    
    def generate_report(self,
                       portfolio_data: Dict[str, Any],
                       template_name: str = 'portfolio_report.html',
                       output_format: str = 'html',
                       output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive portfolio report
        
        Args:
            portfolio_data: Dictionary with portfolio data
            template_name: Name of the template to use
            output_format: Output format ('html' or 'pdf')
            output_path: Optional output path
            
        Returns:
            Path to the generated report
        """
        # Prepare template data
        template_data = {
            'report_title': portfolio_data.get('title', 'Portfolio Report'),
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio_summary': portfolio_data.get('summary', 'Portfolio analysis report.'),
            'summary_metrics': portfolio_data.get('summary_metrics', {}),
            'holdings': portfolio_data.get('holdings', []),
            'performance_metrics': portfolio_data.get('performance_metrics', []),
            'risk_metrics': portfolio_data.get('risk_metrics', []),
            'factor_analysis': 'factor_exposures' in portfolio_data,
            'factor_exposures': portfolio_data.get('factor_exposures', [])
        }
        
        # Convert charts to base64
        if 'composition_chart' in portfolio_data:
            if hasattr(portfolio_data['composition_chart'], 'savefig'):  # Matplotlib
                template_data['composition_chart'] = self._fig_to_base64(portfolio_data['composition_chart'])
            else:  # Plotly
                template_data['composition_chart'] = self._plotly_to_base64(portfolio_data['composition_chart'])
        
        if 'performance_chart' in portfolio_data:
            if hasattr(portfolio_data['performance_chart'], 'savefig'):  # Matplotlib
                template_data['performance_chart'] = self._fig_to_base64(portfolio_data['performance_chart'])
            else:  # Plotly
                template_data['performance_chart'] = self._plotly_to_base64(portfolio_data['performance_chart'])
        
        if 'risk_chart' in portfolio_data:
            if hasattr(portfolio_data['risk_chart'], 'savefig'):  # Matplotlib
                template_data['risk_chart'] = self._fig_to_base64(portfolio_data['risk_chart'])
            else:  # Plotly
                template_data['risk_chart'] = self._plotly_to_base64(portfolio_data['risk_chart'])
        
        if 'factor_chart' in portfolio_data:
            if hasattr(portfolio_data['factor_chart'], 'savefig'):  # Matplotlib
                template_data['factor_chart'] = self._fig_to_base64(portfolio_data['factor_chart'])
            else:  # Plotly
                template_data['factor_chart'] = self._plotly_to_base64(portfolio_data['factor_chart'])
        
        # Render template
        template = self.jinja_env.get_template(template_name)
        html_content = template.render(**template_data)
        
        # Save report
        if output_format == 'html':
            if output_path is None:
                output_path = os.path.join(
                    self.output_dir, 
                    f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                )
                
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Generated HTML report at {output_path}")
            return output_path
            
        elif output_format == 'pdf':
            if output_path is None:
                output_path = os.path.join(
                    self.output_dir, 
                    f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
                
            # Convert HTML to PDF
            try:
                pdfkit.from_string(
                    html_content,
                    output_path,
                    options={
                        'page-size': 'Letter',
                        'margin-top': '0.75in',
                        'margin-right': '0.75in',
                        'margin-bottom': '0.75in',
                        'margin-left': '0.75in',
                        'encoding': 'UTF-8',
                        'no-outline': None,
                        'quiet': ''
                    }
                )
                logger.info(f"Generated PDF report at {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")
                # Fallback to HTML if PDF generation fails
                html_path = output_path.replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"Generated HTML report (PDF failed) at {html_path}")
                return html_path
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _figure_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_str
    
    def _plotly_to_base64(self, fig) -> str:
        """Convert a plotly figure to base64 string"""
        img_bytes = pio.to_image(fig, format='png', width=800, height=500)
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        return img_str
    
    def create_performance_report(self,
                                 portfolio_returns: pd.Series,
                                 benchmark_returns: Optional[pd.Series] = None,
                                 holdings: Optional[pd.DataFrame] = None,
                                 risk_metrics: Optional[Dict[str, Any]] = None,
                                 factor_analysis: Optional[Dict[str, Any]] = None,
                                 output_path: Optional[str] = None) -> str:
        """
        Create a comprehensive performance report
        
        Args:
            portfolio_returns: Series of portfolio returns
            benchmark_returns: Optional series of benchmark returns
            holdings: Optional DataFrame with holdings information
            risk_metrics: Optional dictionary with risk metrics
            factor_analysis: Optional dictionary with factor analysis results
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Calculate performance metrics
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility
        
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Prepare benchmark comparison if available
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            benchmark_ann_return = (1 + benchmark_returns.mean()) ** 252 - 1
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)
            benchmark_sharpe = benchmark_ann_return / benchmark_vol
            
            # Calculate beta and alpha
            cov_matrix = np.cov(portfolio_returns, benchmark_returns)
            beta = cov_matrix[0, 1] / np.var(benchmark_returns)
            alpha = annualized_return - beta * benchmark_ann_return
            
            benchmark_metrics = {
                'benchmark_return': benchmark_ann_return,
                'benchmark_volatility': benchmark_vol,
                'benchmark_sharpe': benchmark_sharpe,
                'alpha': alpha,
                'beta': beta,
                'tracking_error': np.std(portfolio_returns - benchmark_returns) * np.sqrt(252),
                'information_ratio': alpha / (np.std(portfolio_returns - benchmark_returns) * np.sqrt(252))
            }
        
        # Create performance chart (Plotly)
        performance_fig = go.Figure()
        
        performance_fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        if benchmark_returns is not None:
            performance_fig.add_trace(go.Scatter(
                x=benchmark_cum_returns.index,
                y=benchmark_cum_returns.values * 100,
                mode='lines',
                name='Benchmark',
                line=dict(color='red', width=2)
            ))
        
        performance_fig.update_layout(
            title='Cumulative Performance',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create holdings chart if available
        composition_fig = None
        if holdings is not None:
            composition_fig = go.Figure(data=[go.Pie(
                labels=holdings['symbol'],
                values=holdings['weight'],
                hole=.3,
                textinfo='label+percent'
            )])
            
            composition_fig.update_layout(
                title='Portfolio Composition',
                template='plotly_white'
            )
        
        # Create risk chart
        risk_fig = None
        if risk_metrics is not None:
            # Create a radar chart for risk metrics
            risk_fig = go.Figure()
            
            # Select key risk metrics
            key_metrics = {
                'Volatility': risk_metrics.get('volatility', volatility),
                'VaR (95%)': risk_metrics.get('var_95', 0),
                'CVaR (95%)': risk_metrics.get('cvar_95', 0),
                'Max Drawdown': risk_metrics.get('max_drawdown', max_drawdown),
                'Beta': benchmark_metrics.get('beta', 1) if benchmark_metrics else 1
            }
            
            # Normalize metrics for radar chart
            max_values = {
                'Volatility': 0.3,
                'VaR (95%)': 0.1,
                'CVaR (95%)': 0.15,
                'Max Drawdown': 0.5,
                'Beta': 2.0
            }
            
            # Calculate normalized values (lower is better for risk metrics)
            normalized_metrics = {}
            for metric, value in key_metrics.items():
                # Invert the scale so lower risk = higher score on radar
                if metric == 'Beta':
                    # For beta, 1.0 is neutral, so calculate distance from 1.0
                    normalized_metrics[metric] = 1 - min(abs(value - 1) / 1, 1)
                else:
                    # For other metrics, lower is better
                    normalized_metrics[metric] = 1 - min(abs(value) / max_values[metric], 1)
            
            # Create radar chart
            risk_fig = go.Figure()
            
            risk_fig.add_trace(go.Scatterpolar(
                r=list(normalized_metrics.values()),
                theta=list(normalized_metrics.keys()),
                fill='toself',
                name='Risk Profile'
            ))
            
            risk_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title='Risk Profile',
                template='plotly_white'
            )
        
        # Create factor chart if available
        factor_fig = None
        if factor_analysis is not None and 'factor_exposures' in factor_analysis:
            factor_exposures = factor_analysis['factor_exposures']
            
            # Create bar chart for factor exposures
            factor_fig = go.Figure()
            
            # Sort factors by absolute exposure
            sorted_factors = sorted(
                factor_exposures.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            factors = [f[0] for f in sorted_factors]
            exposures = [f[1] for f in sorted_factors]
            
            # Add color based on exposure direction
            colors = ['green' if e > 0 else 'red' for e in exposures]
            
            factor_fig.add_trace(go.Bar(
                x=factors,
                y=exposures,
                marker_color=colors,
                name='Factor Exposure'
            ))
            
            factor_fig.update_layout(
                title='Factor Exposures',
                xaxis_title='Factor',
                yaxis_title='Exposure',
                template='plotly_white'
            )
        
        # Prepare performance metrics for the report
        performance_metrics_list = [
            {'name': 'Annualized Return', 'value': f"{annualized_return:.2%}", 'benchmark': f"{benchmark_metrics.get('benchmark_return', 0):.2%}" if benchmark_metrics else 'N/A'},
            {'name': 'Volatility', 'value': f"{volatility:.2%}", 'benchmark': f"{benchmark_metrics.get('benchmark_volatility', 0):.2%}" if benchmark_metrics else 'N/A'},
            {'name': 'Sharpe Ratio', 'value': f"{sharpe_ratio:.2f}", 'benchmark': f"{benchmark_metrics.get('benchmark_sharpe', 0):.2f}" if benchmark_metrics else 'N/A'},
            {'name': 'Max Drawdown', 'value': f"{max_drawdown:.2%}", 'benchmark': 'N/A'},
        ]
        
        if benchmark_metrics:
            performance_metrics_list.extend([
                {'name': 'Alpha', 'value': f"{benchmark_metrics['alpha']:.2%}", 'benchmark': 'N/A'},
                {'name': 'Beta', 'value': f"{benchmark_metrics['beta']:.2f}", 'benchmark': 'N/A'},
                {'name': 'Information Ratio', 'value': f"{benchmark_metrics['information_ratio']:.2f}", 'benchmark': 'N/A'},
                {'name': 'Tracking Error', 'value': f"{benchmark_metrics['tracking_error']:.2%}", 'benchmark': 'N/A'},
            ])
        
        # Prepare risk metrics for the report
        risk_metrics_list = []
        if risk_metrics:
            for metric, value in risk_metrics.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.2%}" if abs(value) < 1 else f"{value:.2f}"
                    risk_metrics_list.append({'name': metric, 'value': formatted_value})
        else:
            risk_metrics_list = [
                {'name': 'Volatility', 'value': f"{volatility:.2%}"},
                {'name': 'Max Drawdown', 'value': f"{max_drawdown:.2%}"},
                {'name': 'VaR (95%)', 'value': f"{portfolio_returns.quantile(0.05) * np.sqrt(252):.2%}"},
            ]
        
        # Prepare factor exposures for the report
        factor_exposures_list = []
        if factor_analysis and 'factor_exposures' in factor_analysis:
            for factor, exposure in factor_analysis['factor_exposures'].items():
                significance = factor_analysis.get('factor_pvalues', {}).get(factor, 1.0)
                factor_exposures_list.append({
                    'name': factor,
                    'exposure': f"{exposure:.2f}",
                    'significance': 'Significant' if significance < 0.05 else 'Not Significant'
                })
        
        # Prepare holdings for the report
        holdings_list = []
        if holdings is not None:
            for _, row in holdings.iterrows():
                holdings_list.append({
                    'name': row.get('symbol', 'Unknown'),
                    'weight': f"{row.get('weight', 0) * 100:.2f}",
                    'value': f"{row.get('value', 0):.2f}",
                    'return': f"{row.get('return', 0) * 100:.2f}"
                })
        
        # Prepare report data
        report_data = {
            'title': 'Portfolio Performance Report',
            'summary': f"Performance analysis for the period {portfolio_returns.index[0].strftime('%Y-%m-%d')} to {portfolio_returns.index[-1].strftime('%Y-%m-%d')}.",
            'summary_metrics': {
                'Annualized Return': f"{annualized_return:.2%}",
                'Volatility': f"{volatility:.2%}",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Max Drawdown': f"{max_drawdown:.2%}"
            },
            'holdings': holdings_list,
            'performance_metrics': performance_metrics_list,
            'risk_metrics': risk_metrics_list,
            'performance_chart': performance_fig,
            'risk_chart': risk_fig
        }
        
        if composition_fig:
            report_data['composition_chart'] = composition_fig
            
        if factor_fig and factor_exposures_list:
            report_data['factor_chart'] = factor_fig
            report_data['factor_exposures'] = factor_exposures_list
        
        # Generate the report
        return self.generate_report(
            portfolio_data=report_data,
            output_format='html' if not output_path or output_path.endswith('.html') else 'pdf',
            output_path=output_path
        )