"""
Calculate and visualize portfolio risk-return characteristics with hover information.
"""
import os
import argparse
import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
from datetime import datetime

from common.data import get_latest_price_file, load_price_data, calculate_returns

def create_random_portfolios(returns, n_portfolios=5000, allow_short=False):
    """Generate random portfolio weights and calculate their metrics.
    
    Args:
        returns: DataFrame of asset returns
        n_portfolios: Number of random portfolios to generate
        allow_short: Whether to allow short selling
        
    Returns:
        tuple: (weights, returns, volatilities, sharpe_ratios)
    """
    n_assets = len(returns.columns)
    
    if allow_short:
        # Generate weights that sum to 1 but can be negative
        weights = np.random.randn(n_portfolios, n_assets)
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
    else:
        # Generate random weights using Dirichlet distribution (non-negative, sum to 1)
        weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
    
    # Calculate portfolio metrics
    portfolio_returns = weights @ returns.mean()
    
    # Calculate portfolio volatilities using matrix multiplication
    cov_matrix = returns.cov()
    portfolio_volatility = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))
    
    # Calculate Sharpe ratios (assuming 0 risk-free rate for simplicity)
    sharpe_ratios = portfolio_returns / portfolio_volatility
    
    return weights, portfolio_returns, portfolio_volatility, sharpe_ratios

def calculate_gmvp(returns, allow_short=False):
    """Calculate Global Minimum Variance Portfolio.
    
    Args:
        returns: DataFrame of asset returns
        allow_short: Whether to allow short selling
        
    Returns:
        tuple: (weights, return, volatility, sharpe_ratio)
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov()
    
    if allow_short:
        # Analytical solution when short selling is allowed
        inverse_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n_assets)
        w_gmvp = inverse_cov @ ones / (ones.T @ inverse_cov @ ones)
    else:
        # Numerical solution with no-short-selling constraint
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        )
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = sco.minimize(
            lambda w: w.T @ cov_matrix @ w,  # Minimize portfolio variance
            n_assets * [1. / n_assets],  # Start with equal weights
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        w_gmvp = result.x
    
    # Calculate GMVP metrics
    return_gmvp = w_gmvp @ returns.mean()
    vol_gmvp = np.sqrt(w_gmvp.T @ cov_matrix @ w_gmvp)
    sharpe_gmvp = return_gmvp / vol_gmvp
    
    return w_gmvp, return_gmvp, vol_gmvp, sharpe_gmvp

def plot_portfolios(returns, random_results, gmvp_results, save=False, period='daily'):
    """Create and optionally save portfolio visualization with hover information.
    
    Args:
        returns: DataFrame of asset returns
        random_results: (weights, returns, vols, sharpe) for random portfolios
        gmvp_results: (weights, return, vol, sharpe) for GMVP
        save: Whether to save plots to files
    """
    # Unpack results
    rand_weights, rand_rets, rand_vols, rand_sharpe = random_results
    gmvp_w, gmvp_ret, gmvp_vol, gmvp_sharpe = gmvp_results
    
    # Create main plot
    plt.figure(figsize=(10, 6))
    
    # Plot random portfolios
    scatter = plt.scatter(rand_vols, rand_rets,
                         c=rand_sharpe,
                         cmap='viridis',
                         marker='o',
                         alpha=0.3,
                         label='Random Portfolios')
    
    # Plot individual assets
    asset_returns = returns.mean()
    asset_vols = returns.std()
    plt.scatter(asset_vols, asset_returns,
               color='black',
               marker='D',
               s=100,
               label='Individual Assets')
    
    # Plot GMVP
    plt.scatter([gmvp_vol], [gmvp_ret],
               color='red',
               marker='*',
               s=200,
               label='Global Minimum Variance Portfolio')
    
    # Set axis limits based on data
    x_min = min(min(rand_vols), min(asset_vols))
    x_max = max(max(rand_vols), max(asset_vols))
    y_min = min(min(rand_rets), min(asset_returns))
    y_max = max(max(rand_rets), max(asset_returns))
    plt.xlim(x_min * 0.9, x_max * 1.1)
    plt.ylim(y_min * 0.9, y_max * 1.1)
    
    # Customize plot
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.title('Portfolio Analysis')
    plt.legend()
    plt.grid()
    
    # Add hover annotations using matplotlib's event handling
    annot = plt.annotate("", xy=(0,0), xytext=(10,10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind, collection, data_type):
        pos = collection.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        
        if data_type == "random":
            idx = ind["ind"][0]
            weights_str = '\n'.join([f"{asset}: {w:6.2%}" for asset, w in zip(returns.columns, rand_weights[idx])])
            text = (f"Return: {rand_rets[idx]:6.2%}\n"
                   f"Risk: {rand_vols[idx]:6.2%}\n"
                   f"Sharpe: {rand_sharpe[idx]:6.2f}\n"
                   f"\nWeights:\n{weights_str}")
        elif data_type == "asset":
            idx = ind["ind"][0]
            asset = returns.columns[idx]
            text = (f"{asset}\n"
                   f"Return: {asset_returns[idx]:6.2%}\n"
                   f"Risk: {asset_vols[idx]:6.2%}")
        else:  # GMVP
            weights_str = '\n'.join([f"{asset}: {w:6.2%}" for asset, w in zip(returns.columns, gmvp_w)])
            text = (f"Global Minimum Variance Portfolio\n"
                   f"Return: {gmvp_ret:6.2%}\n"
                   f"Risk: {gmvp_vol:6.2%}\n"
                   f"Sharpe: {gmvp_sharpe:6.2f}\n"
                   f"\nWeights:\n{weights_str}")
        
        annot.set_text(text)

    def hover(event):
        if event.inaxes != plt.gca():
            return
        
        cont_random, ind_random = scatter.contains(event)
        cont_asset, ind_asset = plt.gca().collections[1].contains(event)
        cont_gmvp, ind_gmvp = plt.gca().collections[2].contains(event)
        
        if cont_random:
            update_annot(ind_random, scatter, "random")
            annot.set_visible(True)
        elif cont_asset:
            update_annot(ind_asset, plt.gca().collections[1], "asset")
            annot.set_visible(True)
        elif cont_gmvp:
            update_annot(ind_gmvp, plt.gca().collections[2], "gmvp")
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        plt.draw()

    plt.gcf().canvas.mpl_connect("motion_notify_event", hover)
    
    if save:
        plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate and visualize portfolio characteristics'
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
        '--portfolios',
        type=int,
        default=5000,
        help='Number of random portfolios to simulate (default: 5000)'
    )
    parser.add_argument(
        '--short',
        action='store_true',
        help='Allow short selling in portfolio optimization'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to files instead of displaying interactively'
    )
    return parser.parse_args()

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Print start time
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
        
        # Calculate returns
        returns = calculate_returns(prices, return_type=args.return_type, period=args.period)
        print(f"\nCalculating {args.return_type} returns on {args.period} basis")
        print(f"Number of return periods: {len(returns)}")
        
        # Generate random portfolios
        print(f"\nGenerating {args.portfolios} random portfolios...")
        random_results = create_random_portfolios(returns, args.portfolios, args.short)
        
        # Calculate GMVP
        print("\nCalculating Global Minimum Variance Portfolio...")
        gmvp_results = calculate_gmvp(returns, args.short)
        
        # Print GMVP details
        print("\nGlobal Minimum Variance Portfolio:")
        print("-" * 40)
        for asset, weight in zip(returns.columns, gmvp_results[0]):
            print(f"{asset:6s}: {weight:7.2%}")
        print(f"\nExpected Return: {gmvp_results[1]:7.2%}")
        print(f"Risk:           {gmvp_results[2]:7.2%}")
        print(f"Sharpe Ratio:   {gmvp_results[3]:7.2f}")
        
        # Plot results
        print("\nGenerating visualization...")
        plot_portfolios(returns, random_results, gmvp_results, args.save_plots, args.period)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
