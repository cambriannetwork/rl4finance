"""
Calculate and visualize covariance and correlation between cryptocurrency prices.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.data import get_latest_price_file, load_price_data

def calculate_returns(prices, return_type='arithmetic', period='daily'):
    """Calculate returns from price data.
    
    Args:
        prices (pd.DataFrame): Price data with datetime index
        return_type (str): Type of return calculation ('arithmetic' or 'log')
        period (str): Return period ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
    
    Returns:
        pd.DataFrame: Returns data
    """
    # Check minimum data points required for each period
    min_points = {
        'daily': 2,
        'weekly': 8,
        'monthly': 30,
        'quarterly': 90,
        'yearly': 365
    }
    required = min_points.get(period, 2)
    
    if len(prices) < required:
        raise ValueError(
            f"Insufficient data points. Need at least {required} for {period} analysis, "
            f"but only have {len(prices)}"
        )
    
    # First resample data based on period
    period_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q',
        'yearly': 'Y'
    }
    
    # Resample to desired period (using last price of the period)
    if period != 'daily':
        prices = prices.resample(period_map[period]).last()
        
        # Check if we still have enough data after resampling
        if len(prices) < 2:
            raise ValueError(
                f"Insufficient data points after {period} resampling. "
                f"Need at least 2 points, but only have {len(prices)}"
            )
    
    # Calculate returns based on type
    if return_type == 'arithmetic':
        returns = prices.pct_change()
    else:  # log returns
        returns = np.log(prices / prices.shift(1))
    
    # Remove any rows with missing data
    returns = returns.dropna()
    
    # Check final data points
    print(f"\nFinal number of return periods: {len(returns)}")
    
    return returns

def plot_matrices(returns, save_path='covariance_correlation_matrices.png'):
    """Create and save covariance and correlation matrix heatmaps."""
    # Calculate matrices
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot covariance matrix
    sns.heatmap(cov_matrix, 
                ax=ax1,
                annot=True,  # Show values
                cmap='coolwarm',  # Red-blue colormap
                center=0,  # Center the colormap at 0
                fmt='.4f')  # Format annotations to 4 decimal places
    ax1.set_title('Covariance Matrix')
    
    # Plot correlation matrix
    sns.heatmap(corr_matrix,
                ax=ax2,
                annot=True,  # Show values
                cmap='coolwarm',  # Red-blue colormap
                center=0,  # Center the colormap at 0
                vmin=-1,  # Fix scale from -1 to 1
                vmax=1,
                fmt='.4f')  # Format annotations to 4 decimal places
    ax2.set_title('Correlation Matrix')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path)
    print(f"Matrices plot saved to {save_path}")
    
    # Display plot
    plt.show()
    
    return cov_matrix, corr_matrix

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate and visualize asset price covariance and correlation'
    )
    parser.add_argument(
        '--return-type',
        choices=['arithmetic', 'log'],
        default='arithmetic',
        help='Type of return calculation (default: arithmetic)'
    )
    parser.add_argument(
        '--period',
        choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
        default='daily',
        help='Return calculation period (default: daily)'
    )
    return parser.parse_args()

def main():
    try:
        # Print start time
        print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parse arguments
        args = parse_args()
        
        # Get latest price data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tools_dir = os.path.join(script_dir, '..', 'tools')
        price_file = get_latest_price_file(tools_dir)
        print(f"\nProcessing file: {os.path.abspath(price_file)}")
        print(f"File created: {datetime.fromtimestamp(os.path.getctime(price_file)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load and process data
        prices = load_price_data(price_file)
        print("\nAssets found:", ", ".join(prices.columns))
        
        # Calculate returns
        returns = calculate_returns(
            prices,
            return_type=args.return_type,
            period=args.period
        )
        print(f"\nCalculating {args.return_type} returns on {args.period} basis")
        
        # Plot matrices
        cov_matrix, corr_matrix = plot_matrices(returns)
        
        # Print matrices
        print("\nCovariance Matrix:")
        print(cov_matrix.round(4))
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(4))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
