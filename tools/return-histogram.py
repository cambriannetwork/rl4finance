#!/usr/bin/env python3
"""
Visualize cryptocurrency return distributions with histograms.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from common.data import get_latest_price_file, load_price_data, calculate_returns

def plot_return_histograms(returns, period='daily', return_type='arithmetic', save_path=None):
    """Plot return histograms for all tokens in a grid layout with color-coded bars.
    
    Args:
        returns: DataFrame with token returns
        period: Return calculation period ('daily', 'weekly', 'monthly', etc.)
        return_type: Type of return calculation ('arithmetic' or 'log')
        save_path: Path to save the plot (optional)
    """
    # Calculate grid dimensions based on number of tokens
    n_tokens = len(returns.columns)
    
    # Determine grid layout (try to make it square-ish)
    if n_tokens <= 3:
        n_rows, n_cols = 1, n_tokens
    else:
        n_rows = int(np.ceil(np.sqrt(n_tokens)))
        n_cols = int(np.ceil(n_tokens / n_rows))
    
    # Create a figure with subplots in a grid (without shared x-axis to allow individual ranges)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # If only one row or column, axes won't be a 2D array
    if n_tokens == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Plot histogram for each token
    for i, token in enumerate(returns.columns):
        if i >= len(axes_flat):
            break
            
        token_returns = returns[token].dropna()
        
        # Calculate statistics
        mean_return = token_returns.mean()
        median_return = token_returns.median()
        std_dev = token_returns.std()
        skewness = token_returns.skew()
        kurtosis = token_returns.kurtosis()
        
        # Calculate data range for this token
        min_val = token_returns.min()
        max_val = token_returns.max()
        
        # Add a small buffer (5% of range) to avoid cutting off edge bars
        data_range = max_val - min_val
        buffer = data_range * 0.05
        x_min = min_val - buffer
        x_max = max_val + buffer
        
        # Set total number of bins and calculate consistent bin width
        total_bins = 50
        total_range = x_max - x_min
        bin_width = total_range / total_bins
        
        # Split data into positive and negative returns with exact zero handling
        positive_returns = token_returns[token_returns > 0]  # Strictly positive
        zero_returns = token_returns[token_returns == 0]     # Exactly zero
        negative_returns = token_returns[token_returns < 0]  # Strictly negative
        
        # Create bin edges with consistent width
        # For negative returns: from min_val to 0
        if not negative_returns.empty:
            # Calculate number of bins needed for negative range
            n_neg_bins = int(np.ceil(abs(x_min) / bin_width))
            # Create bin edges from x_min to 0 with consistent width
            neg_bins = np.linspace(x_min, 0, n_neg_bins + 1)
            axes_flat[i].hist(negative_returns, bins=neg_bins, alpha=0.75, color='red', 
                             edgecolor='black', label='Negative Returns')
        
        # For positive returns: from 0 to max_val
        if not positive_returns.empty:
            # Calculate number of bins needed for positive range
            n_pos_bins = int(np.ceil(x_max / bin_width))
            # Create bin edges from 0 to x_max with consistent width
            pos_bins = np.linspace(0, x_max, n_pos_bins + 1)
            axes_flat[i].hist(positive_returns, bins=pos_bins, alpha=0.75, color='green', 
                             edgecolor='black', label='Positive Returns')
        
        # Handle zero returns separately (assign to positive for visual consistency)
        if not zero_returns.empty:
            # Plot zeros as a single green bar at exactly zero
            zero_bin = np.array([-1e-10, 1e-10])  # Tiny bin around zero
            axes_flat[i].hist(zero_returns, bins=zero_bin, alpha=0.75, color='green',
                             edgecolor='black')
        
        # No vertical lines for mean and median - values are shown in the stats box only
        
        # Add statistics to the plot
        stats_text = (f"Mean: {mean_return:.4f}\n"
                     f"Median: {median_return:.4f}\n"
                     f"Std Dev: {std_dev:.4f}\n"
                     f"Skewness: {skewness:.4f}\n"
                     f"Kurtosis: {kurtosis:.4f}")
        
        axes_flat[i].text(0.02, 0.95, stats_text, transform=axes_flat[i].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set title and labels
        axes_flat[i].set_title(f'{token} {period.capitalize()} {return_type} Returns', fontsize=14)
        axes_flat[i].set_ylabel('Frequency', fontsize=12)
        axes_flat[i].set_xlabel('Return', fontsize=12)  # Add x-label to all plots
        
        # Set x-axis limits to focus on the actual data range
        axes_flat[i].set_xlim(x_min, x_max)
        
        axes_flat[i].grid(True, alpha=0.3)
        axes_flat[i].legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    # Format dates for title
    start_date = returns.index[0]
    end_date = returns.index[-1]
    
    # Convert to datetime if they're not already
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date, unit='ms')
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date, unit='ms')
    
    # Add overall title - positioned higher to avoid overlap
    fig.suptitle(f'Return Distributions\n{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}', 
                fontsize=16, y=0.95)
    
    # Adjust layout with more space for title
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.2)
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show plot
    plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize cryptocurrency return distributions'
    )
    parser.add_argument(
        '--file',
        help='Path to price data file (default: latest file in data directory)'
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
        '--save',
        help='Path to save the plot (optional)'
    )
    return parser.parse_args()

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Get price data file
        if args.file:
            price_file = args.file
        else:
            price_file = get_latest_price_file()
        
        print(f"\nProcessing file: {os.path.abspath(price_file)}")
        print(f"File created: {datetime.fromtimestamp(os.path.getctime(price_file)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load price data
        prices = load_price_data(price_file)
        print(f"Assets found: {', '.join(prices.columns)}")
        
        # Calculate returns
        returns = calculate_returns(prices, return_type=args.return_type, period=args.period)
        print(f"\nCalculating {args.return_type} returns on {args.period} basis")
        print(f"Number of return periods: {len(returns)}")
        
        # Plot return histograms
        plot_return_histograms(returns, args.period, args.return_type, args.save)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
