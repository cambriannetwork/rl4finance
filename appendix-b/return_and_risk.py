"""
Calculate portfolio return and risk using price data.

This script demonstrates:
- Portfolio weights creation
- Expected return calculation: r_p = X^T R
- Portfolio variance calculation: σ_p^2 = X^T V X
"""
import os
import numpy as np
from datetime import datetime

from common.data import get_latest_price_file, load_price_data

def create_portfolio_weights(tokens):
    """Create random portfolio weights that sum to 1.
    
    Args:
        tokens: List of token names
        
    Returns:
        np.array: Random weights that sum to 1
    """
    # Create random weights
    weights = np.random.random(len(tokens))
    # Normalize to sum to 1
    return weights / weights.sum()

def calculate_portfolio_metrics(returns, weights):
    """Calculate portfolio expected return and risk.
    
    Args:
        returns: DataFrame of asset returns
        weights: Array of portfolio weights
        
    Returns:
        tuple: (expected_return, portfolio_risk, annualized_return, annualized_risk)
    """
    # Calculate expected returns for each asset (daily)
    expected_returns = returns.mean()
    
    # Expected portfolio return: r_p = X^T R
    portfolio_return = np.dot(weights, expected_returns)
    
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    
    # Portfolio variance: σ_p^2 = X^T V X
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Portfolio risk (standard deviation)
    portfolio_risk = np.sqrt(portfolio_variance)
    
    # Annualize metrics (assuming daily returns)
    trading_days = 365  # Crypto markets trade every day
    annualized_return = (1 + portfolio_return) ** trading_days - 1
    annualized_risk = portfolio_risk * np.sqrt(trading_days)
    
    return portfolio_return, portfolio_risk, annualized_return, annualized_risk

def main():
    print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get latest price data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.join(script_dir, '..', 'tools')
    price_file = get_latest_price_file(tools_dir)
    print(f"\nProcessing file: {os.path.abspath(price_file)}")
    print(f"File created: {datetime.fromtimestamp(os.path.getctime(price_file)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and process data
    prices = load_price_data(price_file)
    print("\nAssets found:", ", ".join(prices.columns))
    
    # Calculate returns (using arithmetic returns)
    # Fill any non-leading NA values before pct_change
    returns = prices.ffill().pct_change().dropna()
    print(f"\nNumber of return periods: {len(returns)}")
    
    # Create portfolio weights
    weights = create_portfolio_weights(returns.columns)
    
    # Calculate portfolio metrics
    daily_return, daily_risk, annual_return, annual_risk = calculate_portfolio_metrics(returns, weights)
    
    # Print results
    print("\nPortfolio Composition:")
    print("-" * 40)
    for token, weight in zip(returns.columns, weights):
        print(f"{token:6s}: {weight:7.2%}")
    
    print("\nDaily Metrics:")
    print("-" * 40)
    print(f"Expected Return: {daily_return:7.2%}")
    print(f"Risk:           {daily_risk:7.2%}")
    print(f"Return/Risk:    {daily_return/daily_risk:7.2f}")
    
    print("\nAnnualized Metrics:")
    print("-" * 40)
    print(f"Expected Return: {annual_return:7.2%}")
    print(f"Risk:           {annual_risk:7.2%}")
    print(f"Return/Risk:    {annual_return/annual_risk:7.2f}")

if __name__ == "__main__":
    main()
