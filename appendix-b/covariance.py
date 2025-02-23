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

from common.data import get_latest_price_file, load_price_data, calculate_returns

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
        returns = calculate_returns(prices, return_type=args.return_type)
        print(f"\nCalculating {args.return_type} returns")
        print(f"Number of return periods: {len(returns)}")
        
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
