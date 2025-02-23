"""
Calculate and visualize covariance between cryptocurrency prices.
"""
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_latest_price_file():
    """Find the latest token_prices CSV file in the tools directory."""
    pattern = os.path.join('tools', 'token_prices_*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No token_prices CSV files found in tools directory")
    return max(files)  # Latest file based on timestamp in filename

def load_price_data(file_path):
    """Load and preprocess price data from CSV."""
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Pivot the data to get prices for each token in columns
    pivot_df = df.pivot(index='timestamp', columns='token', values='price')
    
    return pivot_df

def calculate_returns(prices):
    """Calculate percentage returns from price data."""
    return prices.pct_change().dropna()

def plot_covariance(returns, save_path='covariance_heatmap.png'):
    """Create and save a covariance matrix heatmap."""
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, 
                annot=True,  # Show values
                cmap='coolwarm',  # Red-blue colormap
                center=0,  # Center the colormap at 0
                fmt='.4f')  # Format annotations to 4 decimal places
    
    plt.title('Asset Price Covariance Matrix')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path)
    print(f"Covariance heatmap saved to {save_path}")
    
    # Display plot
    plt.show()
    
    return cov_matrix

def main():
    try:
        # Get latest price data
        price_file = get_latest_price_file()
        print(f"Using price data from: {price_file}")
        
        # Load and process data
        prices = load_price_data(price_file)
        print("\nAssets found:", ", ".join(prices.columns))
        
        # Calculate returns
        returns = calculate_returns(prices)
        
        # Plot covariance matrix
        cov_matrix = plot_covariance(returns)
        
        # Print covariance matrix
        print("\nCovariance Matrix:")
        print(cov_matrix.round(4))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
