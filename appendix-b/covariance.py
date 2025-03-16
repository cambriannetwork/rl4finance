"""
Calculate and visualize covariance and correlation between cryptocurrency prices.
"""
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from common.data import get_latest_price_file, load_price_data, calculate_returns, resample_prices

def plot_matrices(returns, period='daily', annualize=False, save_path='covariance_correlation_matrices.png'):
    """Create and save covariance and correlation matrix heatmaps.
    
    Args:
        returns: DataFrame of asset returns
        period: Return calculation period ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
        annualize: Whether to annualize the covariance values
        save_path: Path to save the plot
    """
    # Calculate matrices
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()
    
    # Annualize covariance if requested
    if annualize:
        # Annualization factors for different periods
        annualization_factors = {
            'daily': 365,      # Crypto markets trade every day
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        trading_periods = annualization_factors[period]
        cov_matrix = cov_matrix * trading_periods
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot covariance matrix
    sns.heatmap(cov_matrix, 
                ax=ax1,
                annot=True,  # Show values
                cmap='coolwarm',  # Red-blue colormap
                center=0,  # Center the colormap at 0
                fmt='.4f')  # Format annotations to 4 decimal places
    
    # Set title based on whether covariance is annualized
    if annualize:
        ax1.set_title(f'Annualized Covariance Matrix ({period.capitalize()})')
    else:
        ax1.set_title(f'{period.capitalize()} Covariance Matrix')
    
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
    parser.add_argument(
        '--annualize',
        action='store_true',
        help='Annualize covariance values'
    )
    return parser.parse_args()

def main():
    try:
        # Print start time
        print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parse arguments
        args = parse_args()
        
        # Get latest price data
        price_file = get_latest_price_file()
        print(f"\nProcessing file: {os.path.abspath(price_file)}")
        print(f"File created: {datetime.fromtimestamp(os.path.getctime(price_file)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load price data (keeping timestamps as milliseconds)
        prices = load_price_data(price_file)
        print("\nAssets found:", ", ".join(prices.columns))
        
        # Calculate returns with resampling if period is specified
        returns = calculate_returns(prices, return_type=args.return_type, period=args.period)
        print(f"\nCalculating {args.return_type} returns on {args.period} basis")
        print(f"Number of return periods: {len(returns)}")
        
        # Plot matrices
        cov_matrix, corr_matrix = plot_matrices(returns, period=args.period, annualize=args.annualize)
        
        # Print matrices
        if args.annualize:
            print(f"\nAnnualized Covariance Matrix ({args.period.capitalize()}):")
        else:
            print(f"\n{args.period.capitalize()} Covariance Matrix:")
        print(cov_matrix.round(4))
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(4))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
