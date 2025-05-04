"""
Portfolio Factor Analysis Module for STOCKER Pro

This module provides factor analysis capabilities to understand portfolio exposures.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px

from src.features.portfolio.portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

class FactorAnalyzer:
    """
    Factor analysis for understanding portfolio exposures
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize factor analyzer
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
        self.use_interactive = True  # Default to interactive visualizations
        
    def analyze_factor_exposure(self,
                               portfolio_returns: pd.Series,
                               factor_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze portfolio exposure to factors using regression
        
        Args:
            portfolio_returns: Series of portfolio returns
            factor_returns: DataFrame of factor returns
            
        Returns:
            Dictionary with factor analysis results
        """
        # Align data
        aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:
            logger.warning(f"Insufficient data for factor analysis: {len(aligned_data)} observations")
            
        # Extract aligned series
        y = aligned_data.iloc[:, 0]
        X = aligned_data.iloc[:, 1:]
        
        # Add constant for regression
        X = sm.add_constant(X)
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Extract results
        factor_exposures = model.params.iloc[1:].to_dict()
        factor_pvalues = model.pvalues.iloc[1:].to_dict()
        factor_tvalues = model.tvalues.iloc[1:].to_dict()
        
        # Calculate factor contribution to risk
        factor_cov = X.iloc[:, 1:].cov()
        factor_betas = model.params.iloc[1:]
        
        # Factor contribution to variance
        factor_variance_contribution = pd.Series(
            {factor: (factor_betas[factor] ** 2) * factor_cov.loc[factor, factor] 
             for factor in factor_betas.index},
            index=factor_betas.index
        )
        
        # Normalize to get percentage contribution
        total_explained_variance = factor_variance_contribution.sum()
        factor_variance_pct = factor_variance_contribution / total_explained_variance
        
        # Calculate tracking error
        tracking_error = np.sqrt(model.mse_resid * 252)  # Annualized
        
        # Calculate R-squared
        r_squared = model.rsquared
        adjusted_r_squared = model.rsquared_adj
        
        return {
            'factor_exposures': factor_exposures,
            'factor_pvalues': factor_pvalues,
            'factor_tvalues': factor_tvalues,
            'factor_variance_contribution': factor_variance_contribution.to_dict(),
            'factor_variance_pct': factor_variance_pct.to_dict(),
            'tracking_error': tracking_error,
            'r_squared': r_squared,
            'adjusted_r_squared': adjusted_r_squared,
            'regression_summary': model.summary().as_text()
        }
    
    def analyze_style_exposures(self,
                               portfolio_returns: pd.Series,
                               style_factors: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze portfolio exposure to style factors (size, value, momentum, etc.)
        
        Args:
            portfolio_returns: Series of portfolio returns
            style_factors: DataFrame of style factor returns
            
        Returns:
            Dictionary with style analysis results
        """
        # Run factor analysis
        style_analysis = self.analyze_factor_exposure(portfolio_returns, style_factors)
        
        # Categorize exposures
        significant_exposures = {}
        for factor, exposure in style_analysis['factor_exposures'].items():
            pvalue = style_analysis['factor_pvalues'][factor]
            if pvalue < 0.05:  # Statistically significant
                if exposure > 0:
                    direction = 'positive'
                else:
                    direction = 'negative'
                
                if abs(exposure) > 0.5:
                    strength = 'strong'
                elif abs(exposure) > 0.2:
                    strength = 'moderate'
                else:
                    strength = 'weak'
                
                significant_exposures[factor] = {
                    'exposure': exposure,
                    'pvalue': pvalue,
                    'direction': direction,
                    'strength': strength
                }
        
        style_analysis['significant_exposures'] = significant_exposures
        
        # Determine dominant style
        if significant_exposures:
            # Find factor with highest absolute exposure
            dominant_factor = max(
                significant_exposures.items(),
                key=lambda x: abs(x[1]['exposure'])
            )
            style_analysis['dominant_style'] = {
                'factor': dominant_factor[0],
                'exposure': dominant_factor[1]['exposure'],
                'direction': dominant_factor[1]['direction'],
                'strength': dominant_factor[1]['strength']
            }
        else:
            style_analysis['dominant_style'] = None
        
        return style_analysis
    
    def plot_factor_exposures(self,
                             factor_analysis: Dict[str, Any],
                             save_path: Optional[str] = None) -> Any:
        """
        Plot factor exposures
        
        Args:
            factor_analysis: Dictionary with factor analysis results
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure or Plotly figure
        """
        factor_exposures = factor_analysis['factor_exposures']
        factor_pvalues = factor_analysis['factor_pvalues']
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Factor': list(factor_exposures.keys()),
            'Exposure': list(factor_exposures.values()),
            'P-Value': list(factor_pvalues.values()),
            'Significant': [p < 0.05 for p in factor_pvalues.values()]
        })
        
        # Sort by absolute exposure
        df['AbsExposure'] = df['Exposure'].abs()
        df = df.sort_values('AbsExposure', ascending=False)
        
        if self.use_interactive:
            # Create interactive Plotly bar chart
            fig = px.bar(
                df,
                x='Factor',
                y='Exposure',
                color='Significant',
                color_discrete_map={True: '#1f77b4', False: '#d3d3d3'},
                title='Portfolio Factor Exposures',
                labels={'Exposure': 'Factor Exposure (Beta)', 'Factor': 'Factor Name'}
            )
            
            # Add significance markers
            for i, row in df.iterrows():
                sig_text = '***' if row['P-Value'] < 0.01 else ('**' if row['P-Value'] < 0.05 else '')
                fig.add_annotation(
                    x=row['Factor'],
                    y=row['Exposure'],
                    text=sig_text,
                    showarrow=False,
                    yshift=10
                )
            
            # Add hover information
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Exposure: %{y:.4f}<br>P-Value: %{customdata:.4f}',
                customdata=df['P-Value']
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title='Factor',
                yaxis_title='Exposure (Beta)',
                legend_title='Statistically Significant',
                template='plotly_white',
                height=600,
                width=900
            )
            
            # Add zero line
            fig.add_shape(
                type='line',
                x0=-0.5,
                x1=len(df)-0.5,
                y0=0,
                y1=0,
                line=dict(color='black', width=1, dash='dash')
            )
            
            if save_path:
                fig.write_html(save_path)
                
            return fig
        else:
            # Fallback to matplotlib for non-interactive environments
            # Implementation remains for compatibility
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bars with different colors based on significance
            colors = ['#1f77b4' if sig else '#d3d3d3' for sig in df['Significant']]
            bars = ax.bar(df['Factor'], df['Exposure'], color=colors)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Add labels and title
            ax.set_xlabel('Factor')
            ax.set_ylabel('Exposure (Beta)')
            ax.set_title('Portfolio Factor Exposures')
            
            # Add significance markers
            for i, (factor, exposure, pvalue) in enumerate(zip(df['Factor'], df['Exposure'], df['P-Value'])):
                sig_text = '***' if pvalue < 0.01 else ('**' if pvalue < 0.05 else '')
                ax.text(i, exposure + (0.02 if exposure >= 0 else -0.05), sig_text, ha='center')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#1f77b4', label='Significant (p<0.05)'),
                Patch(facecolor='#d3d3d3', label='Not Significant')
            ]
            ax.legend(handles=legend_elements, loc='best')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig