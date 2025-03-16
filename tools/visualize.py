"""
Visualize cryptocurrency prices from log files.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from common.data import get_latest_price_file, load_price_data

def plot_prices(prices, save_path=None):
    """Plot price data for all tokens.
    
    Args:
        prices: DataFrame with token prices
        save_path: Path to save the plot (optional)
    """
    # Convert millisecond timestamps to datetime for plotting
    prices_datetime = prices.copy()
    prices_datetime.index = pd.to_datetime(prices_datetime.index, unit='ms')
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each token
    for token in prices.columns:
        plt.plot(prices_datetime.index, prices[token], label=token, linewidth=2)
        
        # Add min/max price information to legend
        min_price = prices[token].min()
        max_price = prices[token].max()
        plt.plot([], [], ' ', label=f'{token} min: ${min_price:.2f}, max: ${max_price:.2f}')
    
    # Format plot
    plt.title('Cryptocurrency Prices\n' + 
              f'{prices_datetime.index[0]:%Y-%m-%d} to {prices_datetime.index[-1]:%Y-%m-%d}',
              fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show plot
    plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize cryptocurrency prices from log files'
    )
    parser.add_argument(
        '--file',
        help='Path to price data file (default: latest file in data directory)'
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
        print(f"Date range: {pd.to_datetime(prices.index[0], unit='ms').strftime('%Y-%m-%d')} to {pd.to_datetime(prices.index[-1], unit='ms').strftime('%Y-%m-%d')}")
        print(f"Number of data points: {len(prices)}")
        
        # Plot prices
        plot_prices(prices, args.save)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
