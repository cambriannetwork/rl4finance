#!/usr/bin/env python3
"""
Visualize cryptocurrency return distributions with histograms.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# Set up Tufte-inspired style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Times', 'serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.axisbelow': True,
    'axes.linewidth': 0.8,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3
})

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
    
    # Create a figure with subplots in a grid with a reasonable initial size
    # that allows user resizing - adjusted for better text spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 11), facecolor='#f9f9f9')
    
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
            # Use more muted red color with thinner edges for negative returns - no label
            axes_flat[i].hist(negative_returns, bins=neg_bins, alpha=0.7, color='#d66563', 
                             edgecolor='#555555', linewidth=0.5)
        
        # For positive returns: from 0 to max_val
        if not positive_returns.empty:
            # Calculate number of bins needed for positive range
            n_pos_bins = int(np.ceil(x_max / bin_width))
            # Create bin edges from 0 to x_max with consistent width
            pos_bins = np.linspace(0, x_max, n_pos_bins + 1)
            # Use more muted green color with thinner edges for positive returns - no label
            axes_flat[i].hist(positive_returns, bins=pos_bins, alpha=0.7, color='#6cad68', 
                             edgecolor='#555555', linewidth=0.5)
        
        # Handle zero returns separately (assign to positive for visual consistency)
        if not zero_returns.empty:
            # Plot zeros as a single green bar at exactly zero
            zero_bin = np.array([-1e-10, 1e-10])  # Tiny bin around zero
            # Use same muted green for zero returns with thinner edges
            axes_flat[i].hist(zero_returns, bins=zero_bin, alpha=0.7, color='#6cad68',
                             edgecolor='#555555', linewidth=0.5)
        
        # No vertical lines for mean and median - values are shown in the stats box only
        
        # Add statistics directly to the plot in a more elegant way
        # Use direct labeling with minimal boxing for better data-ink ratio
        stats_text = (f"μ: {mean_return:.4f}   σ: {std_dev:.4f}\n"
                     f"median: {median_return:.4f}\n"
                     f"skew: {skewness:.2f}   kurt: {kurtosis:.2f}")
        
        # Move stats box to upper right corner
        axes_flat[i].text(0.97, 0.95, stats_text, transform=axes_flat[i].transAxes,
                    verticalalignment='top', horizontalalignment='right', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='#f9f9f9', 
                             edgecolor='#cccccc', alpha=0.9, pad=0.5))
        
        # Simplified title to take less space
        axes_flat[i].set_title(f'{token} Returns', fontsize=10)
        axes_flat[i].set_ylabel('Frequency', fontsize=10)
        axes_flat[i].set_xlabel('Return', fontsize=10)
        
        # Set x-axis limits to focus on the actual data range
        axes_flat[i].set_xlim(x_min, x_max)
        
        # Add subtle grid lines only on y-axis for reference without distraction
        axes_flat[i].grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.3)
        
        # Add a subtle zero line for reference
        axes_flat[i].axvline(x=0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Remove legend as requested - color coding is sufficient
        
        # Remove top and right spines for cleaner look
        axes_flat[i].spines['top'].set_visible(False)
        axes_flat[i].spines['right'].set_visible(False)
        axes_flat[i].spines['left'].set_linewidth(0.8)
        axes_flat[i].spines['bottom'].set_linewidth(0.8)
    
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
    
    # Even more spacing between subplots to avoid text overlap
    plt.subplots_adjust(top=0.82, hspace=0.6, wspace=0.5)
    
    # Add overall title with date range - moved higher to avoid overlap
    fig.suptitle(f'Return Distributions\n{start_date:%B %d, %Y} to {end_date:%B %d, %Y}', 
                fontsize=13, y=0.90)
    
    # Save plot with optimized settings for online presentation
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#f9f9f9')
        print(f"Plot saved to {save_path}")
    
    # Show plot and block until window is closed
    plt.show(block=True)

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
