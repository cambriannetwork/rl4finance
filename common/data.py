"""
Common data loading and processing functions.
"""
import os
import glob
import pandas as pd
import numpy as np

def get_latest_price_file(tools_dir):
    """Find the latest token_prices CSV file in the tools directory.
    
    Args:
        tools_dir (str): Path to the tools directory
        
    Returns:
        str: Path to the latest price file
        
    Raises:
        FileNotFoundError: If no price files found
    """
    pattern = os.path.join(tools_dir, 'token_prices_*.csv')

    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No token_prices CSV files found in tools directory")
    
    return sorted(files, key=os.path.getctime, reverse=True)[0]  # Sort by file creation time

def load_price_data(file_path):
    """Load and preprocess price data from CSV.
    
    Args:
        file_path (str): Path to the price data CSV file
        
    Returns:
        pd.DataFrame: Price data with datetime index and token columns
    """
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Pivot the data to get prices for each token in columns
    pivot_df = df.pivot(index='timestamp', columns='token', values='price')
    
    return pivot_df

def calculate_returns(prices, return_type='arithmetic', period='daily'):
    """Calculate returns from price data.
    
    Args:
        prices (pd.DataFrame): Price data with datetime index and token columns
        return_type (str): Type of return calculation ('arithmetic' or 'log')
        period (str): Return calculation period ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
        
    Returns:
        pd.DataFrame: Returns data
    """
    
    # Resample data based on period
    period_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME',  # Month End frequency
        'quarterly': 'QE',  # Quarter End frequency
        'yearly': 'YE'  # Year End frequency
    }
    
    # Get the pandas frequency string for the period
    freq = period_map[period]
    
    # Resample to the specified period, taking the last price of each period
    prices = prices.resample(freq).last()
    
    # Calculate returns based on type
    if return_type == 'arithmetic':
        returns = prices.pct_change()
    else:  # log returns
        returns = np.log(prices / prices.shift(1))
    
    # Remove any rows with missing data
    returns = returns.dropna()
    
    return returns
