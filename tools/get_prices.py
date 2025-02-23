"""
Script to fetch cryptocurrency prices from CoinGecko Pro API.
"""
import os
import sys
import json
import time
import logging
import argparse
from typing import List, Dict, Optional
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
COINGECKO_API_URL = "https://pro-api.coingecko.com/api/v3"
DEFAULT_CURRENCY = "usd"
DEFAULT_DAYS = "5"
DEFAULT_INTERVAL = "daily"
DEFAULT_TOKENS = ["ETH", "BTC"]
VALID_INTERVALS = ["5m", "hourly", "daily"]
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Example usage message
USAGE_EXAMPLES = """
examples:
  # Get daily prices for ETH and BTC (default)
  python get_prices.py

  # Get daily prices for specific tokens
  python get_prices.py --tokens ETH WBTC

  # Get 5-minute data for the last day
  python get_prices.py --tokens ETH WBTC SOL --interval 5m --days 1

  # Get weekly data
  python get_prices.py --tokens ETH WBTC --period weekly
"""

class CoinGeckoAPI:
    def __init__(self):
        self.api_key = self._load_api_key()
        self.session = requests.Session()
        self.session.headers.update({
            'accept': 'application/json',
            'x-cg-pro-api-key': self.api_key
        })

    def _load_api_key(self) -> str:
        """Load CoinGecko API key from environment variables."""
        load_dotenv()
        api_key = os.getenv('COINGECKO_API_KEY')
        if not api_key:
            raise ValueError("COINGECKO_API_KEY not found in .env file")
        return api_key

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with retry mechanism."""
        for attempt in range(MAX_RETRIES):
            try:
                headers = {
                    'accept': 'application/json',
                    'x-cg-pro-api-key': self.api_key
                }
                response = self.session.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if hasattr(e.response, 'status_code'):
                    if e.response.status_code == 429:
                        logger.warning("Rate limit exceeded, waiting before retry...")
                    elif e.response.status_code in (401, 403):
                        logger.error(f"API key error: {e.response.text}")
                        raise ValueError("Invalid or unauthorized API key") from e
                
                if attempt == MAX_RETRIES - 1:
                    raise
                
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        raise Exception("Failed after max retries")

    def get_coin_id(self, symbol: str) -> str:
        """Get coin ID from symbol using /coins/list endpoint."""
        symbol = symbol.lower()
        
        # Known mappings for common tokens
        KNOWN_MAPPINGS = {
            'eth': 'ethereum',
            'wbtc': 'wrapped-bitcoin',
            'sol': 'solana',
            'grt': 'the-graph'
        }
        
        if symbol in KNOWN_MAPPINGS:
            return KNOWN_MAPPINGS[symbol]
        
        # For unknown symbols, fetch from API
        if not hasattr(self, '_symbol_to_id_cache'):
            url = f"{COINGECKO_API_URL}/coins/list"
            data = self._make_request(url)
            
            # Build cache mapping symbols to IDs
            self._symbol_to_id_cache = {}
            for coin in data:
                self._symbol_to_id_cache[coin['symbol'].lower()] = coin['id']
        
        if symbol not in self._symbol_to_id_cache:
            raise ValueError(f"No coin found for symbol: {symbol}")
        
        return self._symbol_to_id_cache[symbol]

    def fetch_historical_prices(
        self, 
        coin_id: str, 
        currency: str = DEFAULT_CURRENCY,
        days: str = DEFAULT_DAYS,
        interval: str = DEFAULT_INTERVAL
    ) -> List:
        """Fetch historical prices from CoinGecko Pro API."""
        url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": currency,
            "days": days,
            "interval": interval
        }
        
        data = self._make_request(url, params=params)
        return data["prices"]  # List of [timestamp, price]

def format_data(prices: List, symbol: str) -> pd.DataFrame:
    """Convert raw API response into a DataFrame with token column."""
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["token"] = symbol
    return df[["timestamp", "token", "price"]]

def get_prices(
    symbols: List[str],
    currency: str = DEFAULT_CURRENCY,
    days: str = DEFAULT_DAYS,
    interval: str = DEFAULT_INTERVAL
) -> pd.DataFrame:
    """Get historical prices for multiple tokens."""
    if not symbols:
        raise ValueError("No symbols provided")
    
    api = CoinGeckoAPI()
    all_data = []
    
    for symbol in tqdm(symbols, desc="Fetching prices"):
        try:
            logger.info(f"Processing {symbol}...")
            coin_id = api.get_coin_id(symbol)
            prices = api.fetch_historical_prices(coin_id, currency, days, interval)
            df = format_data(prices, symbol)
            all_data.append(df)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data was retrieved for any symbols")
    
    # Concatenate all data vertically
    result = pd.concat(all_data, ignore_index=True)
    return result.sort_values(['token', 'timestamp'])

def resample_prices(df: pd.DataFrame, period: str = 'daily') -> pd.DataFrame:
    """Resample price data to desired period.
    
    Args:
        df: DataFrame with timestamp, token, price columns
        period: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
        
    Returns:
        DataFrame: Resampled price data
    """
    # Print data coverage and quality info
    logger.info("\nData Quality Report:")
    
    # Print date ranges per asset
    logger.info("\nDate ranges per asset:")
    for token in df['token'].unique():
        token_data = df[df['token'] == token]
        logger.info(f"{token}: {token_data['timestamp'].min():%Y-%m-%d} to {token_data['timestamp'].max():%Y-%m-%d}")
    
    # Check data points per asset
    counts = df.groupby('token').size()
    if counts.nunique() > 1:
        logger.info("\nData points per asset:")
        for token, count in counts.items():
            logger.info(f"{token}: {count} points")
    
    # Resample data
    period_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME',  # Month End frequency
        'quarterly': 'QE',  # Quarter End frequency
        'yearly': 'YE'  # Year End frequency
    }
    freq = period_map.get(period, 'D')
    
    # Pivot, resample, and melt back
    pivot_df = df.pivot(index='timestamp', columns='token', values='price')
    resampled = pivot_df.resample(freq).last()
    
    # Check for missing data after resampling
    missing = resampled.isnull().sum()
    if missing.any():
        logger.warning("\nMissing data points after alignment:")
        for token, count in missing.items():
            if count > 0:
                logger.warning(f"{token}: {count} missing points")
    
    # Fill any missing values
    resampled = resampled.ffill()
    
    # Melt back to long format
    result = resampled.reset_index().melt(
        id_vars=['timestamp'],
        var_name='token',
        value_name='price'
    )
    
    logger.info(f"\nFinal number of periods: {len(resampled)}")
    
    return result.sort_values(['token', 'timestamp'])

def save_data(df: pd.DataFrame) -> None:
    """Save data to CSV file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'token_prices_{timestamp}.csv'
    df.to_csv(filename, index=False)
    logger.info(f"Data saved to {filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Fetch cryptocurrency prices from CoinGecko Pro API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_EXAMPLES
    )
    parser.add_argument('--tokens', nargs='+', required=False,
                      help='One or more token symbols (e.g., BTC ETH). Defaults to ETH and BTC if not specified.')
    parser.add_argument('--days', default=DEFAULT_DAYS,
                      help=f'Number of days of data to fetch (default: {DEFAULT_DAYS})')
    parser.add_argument('--interval', default=DEFAULT_INTERVAL,
                      choices=VALID_INTERVALS,
                      help=f'Data interval (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--currency', default=DEFAULT_CURRENCY,
                      help=f'Price currency (default: {DEFAULT_CURRENCY})')
    parser.add_argument('--period',
                      choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                      default='daily',
                      help='Resample period (default: daily)')
    
    args = parser.parse_args()
    
    try:
        symbols = [s.upper() for s in (args.tokens or DEFAULT_TOKENS)]
        df = get_prices(symbols, args.currency, args.days, args.interval)
        
        # Always show data quality info and resample if needed
        if args.period == 'daily':
            # Just show data quality info without resampling
            logger.info("\nData Quality Report:")
            logger.info("\nDate ranges per asset:")
            for token in df['token'].unique():
                token_data = df[df['token'] == token]
                logger.info(f"{token}: {token_data['timestamp'].min():%Y-%m-%d} to {token_data['timestamp'].max():%Y-%m-%d}")
            
            # Check data points per asset
            counts = df.groupby('token').size()
            if counts.nunique() > 1:
                logger.info("\nData points per asset:")
                for token, count in counts.items():
                    logger.info(f"{token}: {count} points")
            
            # Check for missing data
            pivot_df = df.pivot(index='timestamp', columns='token', values='price')
            missing = pivot_df.isnull().sum()
            if missing.any():
                logger.warning("\nMissing data points after alignment:")
                for token, count in missing.items():
                    if count > 0:
                        logger.warning(f"{token}: {count} missing points")
            
            logger.info(f"\nFinal number of periods: {len(pivot_df)}")
        else:
            # Resample and show data quality info
            print(f"\nResampling to {args.period} frequency...")
            df = resample_prices(df, args.period)
        
        logger.info("\nLast 5 entries of price data:")
        print(df.tail())
        
        save_data(df)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
