import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

# Generate synthetic asset returns (mean returns and covariance matrix)
n_assets = 5
np.random.seed(42)
mean_returns = np.random.uniform(0.05, 0.15, n_assets)  # Mean returns (5-15%)
cov_matrix = np.random.rand(n_assets, n_assets)
cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  # Make it symmetric
cov_matrix += n_assets * np.eye(n_assets)  # Ensure positive definiteness

# Generate random portfolios
n_portfolios = 5000
weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
portfolio_returns = weights @ mean_returns
portfolio_volatility = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))

# Compute Efficient Frontier
def min_variance_weights(target_return):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda w: w @ mean_returns - target_return}  # Target return
    )
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling
    result = sco.minimize(lambda w: w.T @ cov_matrix @ w, num_assets * [1. / num_assets], 
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

eff_returns = np.linspace(min(portfolio_returns), max(portfolio_returns), 100)
eff_vols = [np.sqrt(min_variance_weights(r).T @ cov_matrix @ min_variance_weights(r)) for r in eff_returns]

# Compute Global Minimum Variance Portfolio (GMVP)
inverse_cov = np.linalg.inv(cov_matrix)
ones = np.ones(n_assets)
w_gmvp = inverse_cov @ ones / (ones.T @ inverse_cov @ ones)
return_gmvp = w_gmvp @ mean_returns
vol_gmvp = np.sqrt(w_gmvp.T @ cov_matrix @ w_gmvp)

# Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatility, portfolio_returns, c=portfolio_returns / portfolio_volatility, cmap='viridis', marker='o', alpha=0.3, label='Random Portfolios')
plt.plot(eff_vols, eff_returns, 'r-', linewidth=2, label='Efficient Frontier')
plt.scatter(vol_gmvp, return_gmvp, color='red', marker='*', s=200, label='Global Minimum Variance Portfolio (GMVP)')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier and Global Minimum Variance Portfolio')
plt.legend()
plt.grid()
plt.savefig('efficient_frontier.png')
plt.show()

# Plot GMVP Only
plt.figure(figsize=(6, 4))
plt.scatter(vol_gmvp, return_gmvp, color='red', marker='*', s=200, label='Global Minimum Variance Portfolio (GMVP)')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('Global Minimum Variance Portfolio')
plt.legend()
plt.grid()
plt.savefig('gmvp_plot.png')
plt.show()
