import os
import json
import boto3
import requests
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ThunderComputePortfolioOptimizer:
    """
    Portfolio optimization using ThunderCompute cloud infrastructure
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 s3_bucket: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_region: str = 'eu-north-1'):
        """
        Initialize the ThunderCompute portfolio optimizer
        
        Args:
            api_key: ThunderCompute API key (if None, load from env)
            s3_bucket: S3 bucket name for data storage (if None, load from env)
            aws_access_key_id: AWS access key ID (if None, load from env)
            aws_secret_access_key: AWS secret access key (if None, load from env)
            aws_region: AWS region
        """
        # Load credentials from environment variables if not provided
        self.api_key = api_key or os.environ.get('THUNDERCOMPUTE_API_KEY')
        self.s3_bucket = s3_bucket or os.environ.get('S3_BUCKET_NAME')
        self.aws_access_key_id = aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.aws_region = aws_region or os.environ.get('AWS_REGION', 'eu-north-1')
        
        # Validate credentials
        if not self.api_key:
            raise ValueError("Missing ThunderCompute API key. Please provide it as a parameter or set THUNDERCOMPUTE_API_KEY environment variable.")
        if not self.s3_bucket:
            raise ValueError("Missing S3 bucket name. Please provide it as a parameter or set S3_BUCKET_NAME environment variable.")
        if not self.aws_access_key_id:
            raise ValueError("Missing AWS access key ID. Please provide it as a parameter or set AWS_ACCESS_KEY_ID environment variable.")
        if not self.aws_secret_access_key:
            raise ValueError("Missing AWS secret access key. Please provide it as a parameter or set AWS_SECRET_ACCESS_KEY environment variable.")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )
        
        # ThunderCompute API base URL
        self.api_base_url = "https://api.thundercompute.com/v1"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _upload_data_to_s3(self, 
                          data: pd.DataFrame, 
                          file_name: str) -> str:
        """
        Upload data to S3 bucket
        
        Args:
            data: DataFrame to upload
            file_name: Name of the file in S3
            
        Returns:
            S3 URI of the uploaded file
        """
        # Convert DataFrame to CSV
        csv_buffer = data.to_csv(index=True).encode()
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=file_name,
            Body=csv_buffer
        )
        
        # Return S3 URI
        return f"s3://{self.s3_bucket}/{file_name}"
    
    def _download_data_from_s3(self, 
                              file_name: str) -> pd.DataFrame:
        """
        Download data from S3 bucket
        
        Args:
            file_name: Name of the file in S3
            
        Returns:
            Downloaded DataFrame
        """
        # Download from S3
        response = self.s3_client.get_object(
            Bucket=self.s3_bucket,
            Key=file_name
        )
        
        # Convert to DataFrame
        return pd.read_csv(response['Body'])
    
    def _submit_job(self, 
                   job_config: Dict[str, Any]) -> str:
        """
        Submit a job to ThunderCompute
        
        Args:
            job_config: Job configuration
            
        Returns:
            Job ID
        """
        # Submit job
        response = requests.post(
            f"{self.api_base_url}/jobs",
            headers=self.headers,
            json=job_config
        )
        
        # Check response
        if response.status_code == 200:
            return response.json().get('job_id', '')
        else:
            raise Exception(f"Failed to submit job: {response.text}")
    
    def _get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status from ThunderCompute
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status
        """
        # Get job status
        response = requests.get(
            f"{self.api_base_url}/jobs/{job_id}",
            headers=self.headers
        )
        
        # Check response
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get job status: {response.text}")
    
    def calculate_portfolio_metrics(self, 
                                  returns: pd.DataFrame, 
                                  weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio metrics (return, volatility, Sharpe ratio)
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Calculate portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _negative_sharpe_ratio(self, 
                             weights: np.ndarray, 
                             returns: pd.DataFrame) -> float:
        """
        Calculate negative Sharpe ratio (for minimization)
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Negative Sharpe ratio
        """
        portfolio_metrics = self.calculate_portfolio_metrics(returns, weights)
        return -portfolio_metrics['sharpe_ratio']
    
    def optimize_portfolio(self, 
                         price_data: pd.DataFrame, 
                         risk_free_rate: float = 0.02,
                         use_cloud: bool = False) -> Dict[str, Any]:
        """
        Optimize portfolio weights using Modern Portfolio Theory
        
        Args:
            price_data: DataFrame of asset prices
            risk_free_rate: Risk-free rate
            use_cloud: Whether to use cloud computing
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # If using cloud computing, submit job to ThunderCompute
        if use_cloud:
            # Upload data to S3
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            s3_path = self._upload_data_to_s3(
                price_data, 
                f"portfolio_optimization/{timestamp}/price_data.csv"
            )
            
            # Submit job
            job_config = {
                "job_type": "portfolio_optimization",
                "data_path": s3_path,
                "parameters": {
                    "risk_free_rate": risk_free_rate
                }
            }
            
            job_id = self._submit_job(job_config)
            
            # Wait for job to complete
            while True:
                job_status = self._get_job_status(job_id)
                if job_status['status'] == 'COMPLETED':
                    # Download results
                    result_path = job_status['result_path']
                    result_file = result_path.split('/')[-1]
                    results = self._download_data_from_s3(result_file)
                    
                    # Parse results
                    weights = results['weights'].values
                    metrics = {
                        'return': results['return'].iloc[0],
                        'volatility': results['volatility'].iloc[0],
                        'sharpe_ratio': results['sharpe_ratio'].iloc[0]
                    }
                    
                    return {
                        'weights': weights,
                        'metrics': metrics,
                        'assets': price_data.columns.tolist()
                    }
                
                elif job_status['status'] == 'FAILED':
                    raise Exception(f"Job failed: {job_status.get('error', 'Unknown error')}")
                
                # Sleep for 5 seconds
                time.sleep(5)
        
        # Otherwise, optimize locally
        else:
            num_assets = len(returns.columns)
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Constraints (weights sum to 1)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds (no short selling)
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Optimize
            result = minimize(
                self._negative_sharpe_ratio,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get optimized weights
            weights = result['x']
            
            # Calculate metrics
            metrics = self.calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'metrics': metrics,
                'assets': price_data.columns.tolist()
            }
    
    def generate_efficient_frontier(self, 
                                  price_data: pd.DataFrame, 
                                  num_portfolios: int = 1000) -> Dict[str, Any]:
        """
        Generate efficient frontier
        
        Args:
            price_data: DataFrame of asset prices
            num_portfolios: Number of portfolios to generate
            
        Returns:
            Dictionary with efficient frontier data
        """
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Number of assets
        num_assets = len(returns.columns)
        
        # Arrays to store results
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        # Generate random portfolios
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(returns, weights)
            
            # Store results
            results[0, i] = portfolio_metrics['volatility']
            results[1, i] = portfolio_metrics['return']
            results[2, i] = portfolio_metrics['sharpe_ratio']
        
        # Find portfolio with highest Sharpe ratio
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_allocation = weights_record[max_sharpe_idx]
        max_sharpe_metrics = self.calculate_portfolio_metrics(returns, max_sharpe_allocation)
        
        # Find portfolio with minimum volatility
        min_vol_idx = np.argmin(results[0])
        min_vol_allocation = weights_record[min_vol_idx]
        min_vol_metrics = self.calculate_portfolio_metrics(returns, min_vol_allocation)
        
        return {
            'efficient_frontier': {
                'volatility': results[0],
                'return': results[1],
                'sharpe_ratio': results[2]
            },
            'max_sharpe': {
                'weights': max_sharpe_allocation,
                'metrics': max_sharpe_metrics
            },
            'min_volatility': {
                'weights': min_vol_allocation,
                'metrics': min_vol_metrics
            },
            'assets': price_data.columns.tolist()
        }
    
    def plot_efficient_frontier(self, 
                              efficient_frontier_data: Dict[str, Any], 
                              save_path: Optional[str] = None) -> None:
        """
        Plot efficient frontier
        
        Args:
            efficient_frontier_data: Efficient frontier data from generate_efficient_frontier
            save_path: Path to save the plot (if None, display plot)
        """
        # Extract data
        volatility = efficient_frontier_data['efficient_frontier']['volatility']
        returns = efficient_frontier_data['efficient_frontier']['return']
        sharpe_ratio = efficient_frontier_data['efficient_frontier']['sharpe_ratio']
        
        max_sharpe_vol = efficient_frontier_data['max_sharpe']['metrics']['volatility']
        max_sharpe_ret = efficient_frontier_data['max_sharpe']['metrics']['return']
        
        min_vol_vol = efficient_frontier_data['min_volatility']['metrics']['volatility']
        min_vol_ret = efficient_frontier_data['min_volatility']['metrics']['return']
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.scatter(volatility, returns, c=sharpe_ratio, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Sharpe Ratio')
        
        # Plot max Sharpe ratio and min volatility portfolios
        plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='r', s=300, label='Max Sharpe Ratio')
        plt.scatter(min_vol_vol, min_vol_ret, marker='*', color='g', s=300, label='Min Volatility')
        
        # Add labels and title
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        
        # Save or display plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def backtest_portfolio(self, 
                         price_data: pd.DataFrame, 
                         weights: np.ndarray, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Backtest portfolio performance
        
        Args:
            price_data: DataFrame of asset prices
            weights: Array of asset weights
            start_date: Start date for backtest (if None, use first date in price_data)
            end_date: End date for backtest (if None, use last date in price_data)
            
        Returns:
            Dictionary with backtest results
        """
        # Filter data by date range
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility
        
        # Calculate drawdowns
        wealth_index = (1 + portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()
        
        return {
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
        }
    
    def plot_backtest_results(self, 
                            backtest_results: Dict[str, Any], 
                            benchmark_returns: Optional[pd.Series] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Plot backtest results
        
        Args:
            backtest_results: Backtest results from backtest_portfolio
            benchmark_returns: Optional benchmark returns for comparison
            save_path: Path to save the plot (if None, display plot)
        """
        # Extract data
        portfolio_returns = backtest_results['portfolio_returns']
        cumulative_returns = backtest_results['cumulative_returns']
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot cumulative returns
        ax1.plot(cumulative_returns, label='Portfolio')
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            # Calculate cumulative benchmark returns
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            ax1.plot(cumulative_benchmark, label='Benchmark')
        
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Portfolio Performance')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdowns
        wealth_index = (1 + portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        ax2.fill_between(drawdowns.index, drawdowns.values, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        # Add metrics as text
        metrics = backtest_results['metrics']
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annualized Return: {metrics['annualized_return']:.2%}\n"
            f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}"
        )
        
        plt.figtext(0.01, 0.01, metrics_text, fontsize=10, va='bottom')
        
        plt.tight_layout()
        
        # Save or display plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()