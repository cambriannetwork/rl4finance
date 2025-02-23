"""
Calculate and visualize covariance and correlation between cryptocurrency prices.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def get_latest_price_file():
    """Find the latest token_prices CSV file in the tools directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.join(script_dir, '..', 'tools')
    pattern = os.path.join(tools_dir, 'token_prices_*.csv')

    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No token_prices CSV files found in tools directory")
    
    return sorted(files, key=os.path.getctime, reverse=True)[0]  # Sort by file creation time

def load_price_data(file_path):
    """Load and preprocess price data from CSV."""
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Pivot the data to get prices for each token in columns
    pivot_df = df.pivot(index='timestamp', columns='token', values='price')
    
    return pivot_df

def calculate_returns(prices, return_type='arithmetic', period='daily'):
    """Calculate returns from price data.
    
    Args:
        prices (pd.DataFrame): Price data with datetime index
        return_type (str): Type of return calculation ('arithmetic' or 'log')
        period (str): Return period ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
    
    Returns:
        pd.DataFrame: Returns data
    """
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
    
    # Calculate returns based on type
    if return_type == 'arithmetic':
        returns = prices.pct_change()
    else:  # log returns
        returns = np.log(prices / prices.shift(1))
    
    return returns.dropna()

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
        price_file = get_latest_price_file()
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
