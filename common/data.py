"""
Common data loading and processing functions.
"""
import os
import glob
import pandas as pd
import numpy as np

def get_latest_price_file(project_root=None):
    """Find the latest token_prices CSV file in the data directory.
    
    Args:
        project_root (str, optional): Path to project root directory. 
                                    If None, will be inferred from current file location.
        
    Returns:
        str: Path to the latest price file
        
    Raises:
        FileNotFoundError: If no price files found
    """
    if project_root is None:
        # Get project root directory (parent of common directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    data_dir = os.path.join(project_root, 'data')
    pattern = os.path.join(data_dir, 'token_prices_*.csv')

    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No token_prices CSV files found in data directory")
    
    return sorted(files, key=os.path.getctime, reverse=True)[0]  # Sort by file creation time

def load_price_data(file_path):
    """Load price data from CSV, keeping timestamps as milliseconds.
    
    Args:
        file_path (str): Path to the price data CSV file
        
    Returns:
        pd.DataFrame: Price data with millisecond timestamp index and token columns
    """
    df = pd.read_csv(file_path)
    
    # Pivot the data to get prices for each token in columns
    # No timestamp conversion - keep as milliseconds
    pivot_df = df.pivot(index='timestamp', columns='token', values='price')
    
    return pivot_df

def resample_prices(prices, period):
    """Resample price data to the specified period.
    
    Args:
        prices (pd.DataFrame): Price data with millisecond timestamp index
        period (str): Resampling period ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
        
    Returns:
        pd.DataFrame: Resampled price data with datetime index
    """
    # Convert millisecond timestamps to datetime for resampling
    prices_datetime = prices.copy()
    prices_datetime.index = pd.to_datetime(prices_datetime.index, unit='ms')
    
    # Resample based on period
    period_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME',  # Month End frequency
        'quarterly': 'QE',  # Quarter End frequency
        'yearly': 'YE'  # Year End frequency
    }
    freq = period_map[period]
    
    # Resample to the specified period, taking the last price of each period
    resampled = prices_datetime.resample(freq).last()
    
    return resampled

def calculate_returns(prices, return_type='arithmetic', period=None):
    """Calculate returns from price data.
    
    Args:
        prices (pd.DataFrame): Price data with millisecond timestamp index or datetime index
        return_type (str): Type of return calculation ('arithmetic' or 'log')
        period (str, optional): Optional resampling period ('daily', 'weekly', etc.)
                               If None, no resampling is performed
        
    Returns:
        pd.DataFrame: Returns data
    """
    # Make a copy of the prices to avoid modifying the original
    prices_copy = prices.copy()
    
    # Check if the index is already datetime, if not convert it
    if not isinstance(prices_copy.index, pd.DatetimeIndex):
        prices_copy.index = pd.to_datetime(prices_copy.index, unit='ms')
    
    # Optional resampling based on period
    if period is not None:
        prices_copy = resample_prices(prices_copy, period)
    
    # Calculate returns based on type
    if return_type == 'arithmetic':
        returns = prices_copy.pct_change()
    else:  # log returns
        returns = np.log(prices_copy / prices_copy.shift(1))
    
    # Remove any rows with missing data
    returns = returns.dropna()
    
    return returns
