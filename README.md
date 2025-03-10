# RL4Finance

> **WARNING:** This repository was "vibe-coded" and is intended for educational purposes only. It serves as an experimental implementation of concepts from the book [Reinforcement Learning for Finance](https://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf).

A Python project for reinforcement learning in finance, portfolio optimization, and cryptocurrency price data analysis.

## Project Structure

```
RL4Finance/
├── chapters/           # Reinforcement learning models
│   └── 7/              # Chapter 7 models
│       └── asset_alloc_discrete.py  # Discrete asset allocation model
├── appendix-b/          # Portfolio optimization scripts
│   ├── frontier.py     # Portfolio optimization with efficient frontier
│   ├── covariance.py   # Asset returns covariance/correlation analysis
│   └── return_and_risk.py  # Return and risk analysis
├── tools/              # Utility scripts
│   └── get_prices.py   # Crypto price data fetcher
├── common/             # Shared functionality
│   └── data.py         # Data loading and processing functions
├── data/               # Price data storage (created by setup)
├── setup.py            # Package installation configuration
└── requirements.txt    # Project dependencies
```

Note: The `data/` directory is automatically created during setup and is used to store price data files. This directory is excluded from version control.

## Features

### Reinforcement Learning Models (chapters/)

#### Asset Allocation Discrete Model (chapters/7/asset_alloc_discrete.py)

This script implements a discrete asset allocation model using reinforcement learning techniques:
- Markov Decision Process (MDP) formulation for asset allocation
- Backward induction for optimal Q-value function approximation
- Deep Neural Network (DNN) function approximation
- Monte Carlo sampling for expectation calculations
- Comparison with analytical solution

Features:
- Configurable parameters:
  - Time steps
  - Expectation samples
  - Solve iterations
  - State samples
  - Verbosity level
  - Risky return mean (μ)
  - Risky return standard deviation (σ)
  - Riskless return rate (r)
  - Risk aversion parameter (a)
- Progress reporting with different verbosity levels
- Timing information for performance analysis
- Analytical solution comparison

Usage:
```bash
# Basic usage with default parameters
python chapters/7/asset_alloc_discrete.py

# With custom parameters
python chapters/7/asset_alloc_discrete.py --time-steps 3 --verbose 1 --mu 0.15 --sigma 0.25 --rate 0.05 --risk-aversion 2.0

# Detailed output with timing information
python chapters/7/asset_alloc_discrete.py --verbose 2 --state-samples 500
```

### Portfolio Optimization (appendix-b/)

The `frontier.py` script demonstrates portfolio optimization techniques:
- Efficient Frontier calculation with interactive visualization
- Global Minimum Variance Portfolio (GMVP)
- Random portfolio generation with configurable size
- Risk-return visualization with hover information
- Support for short selling
- Flexible return period calculation (daily, weekly, monthly, quarterly, yearly)

Features:
- Interactive plot with hover details showing:
  - Portfolio weights
  - Expected return
  - Risk (standard deviation)
  - Sharpe ratio
- Configurable options:
  - Return type (arithmetic/log)
  - Time period (daily/weekly/monthly/quarterly/yearly)
  - Number of random portfolios
  - Short selling allowance
  - Plot saving

### Covariance Analysis (appendix-b/covariance.py)

Analyzes relationships between asset returns:
- Calculates and visualizes covariance and correlation matrices
- Supports different return types:
  - Arithmetic returns: (Pt - Pt-1)/Pt-1
  - Log returns: ln(Pt/Pt-1)
- Flexible time periods:
  - Daily (default)
  - Weekly
  - Monthly
  - Quarterly
  - Yearly
- Automatically uses latest price data from tools/
- Generates side-by-side visualization of:
  - Covariance matrix (raw co-movement)
  - Correlation matrix (normalized -1 to 1 scale)

### Cryptocurrency Price Data (tools/)

The `get_prices.py` script fetches historical cryptocurrency price data using the CoinGecko Pro API:
- Default tokens: ETH and BTC
- Customizable time intervals (5m, hourly, daily)
- Flexible resampling periods (daily, weekly, monthly, quarterly, yearly)
- Data quality reporting:
  - Date ranges per asset
  - Data points per asset
  - Missing data points after alignment
  - Final number of periods
- CSV output with timestamp, token, and price columns

## Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:0xsamgreen/RL4Finance.git
   cd RL4Finance
   ```

2. Set up the conda environment:
   ```bash
   # Make sure you have conda installed (Miniconda or Anaconda)
   ./create_conda_env.sh
   
   # Activate the environment
   conda activate rl4finance
   ```

3. Configure CoinGecko API:
   - Create a `.env` file in the root directory
   - Add your CoinGecko Pro API key:
     ```
     COINGECKO_API_KEY=your_api_key_here
     ```


## Usage

### Portfolio Optimization

#### Return and Risk Analysis
Basic usage (daily arithmetic returns):
```bash
python appendix-b/return_and_risk.py
```

With options:
```bash
# Calculate log returns on weekly basis
python appendix-b/return_and_risk.py --return-type log --period weekly

# Monthly arithmetic returns
python appendix-b/return_and_risk.py --period monthly

# Other period options
python appendix-b/return_and_risk.py --period daily    # Default
python appendix-b/return_and_risk.py --period quarterly
python appendix-b/return_and_risk.py --period yearly
```

The script calculates:
1. Portfolio composition (random weights)
2. Period-specific metrics (return, risk, Sharpe ratio)
3. Annualized metrics for comparison

#### Efficient Frontier

Run the script to see portfolio optimization in action:
```bash
# Basic usage (daily arithmetic returns)
python appendix-b/frontier.py

# Monthly returns with 1000 portfolios and short selling
python appendix-b/frontier.py --period monthly --portfolios 1000 --short

# Save plots instead of displaying
python appendix-b/frontier.py --save-plots

# Other options
python appendix-b/frontier.py --return-type log  # Use log returns
python appendix-b/frontier.py --period weekly    # Weekly returns
python appendix-b/frontier.py --portfolios 5000  # More random portfolios
```

The script will:
1. Load the latest price data
2. Calculate returns based on specified options
3. Generate random portfolios
4. Find the Global Minimum Variance Portfolio
5. Create interactive visualization with hover information

### Covariance Analysis

Basic usage (daily arithmetic returns):
```bash
python appendix-b/covariance.py
```

With options:
```bash
# Calculate log returns on weekly basis
python appendix-b/covariance.py --return-type log --period weekly

# Monthly arithmetic returns
python appendix-b/covariance.py --period monthly

# Other period options
python appendix-b/covariance.py --period daily    # Default
python appendix-b/covariance.py --period quarterly
python appendix-b/covariance.py --period yearly
```

The script will:
1. Find the latest price data file
2. Calculate returns based on specified options
3. Generate covariance and correlation matrices
4. Create and save visualizations
5. Display numerical results

### Price Data Collection

Basic usage (fetches ETH and BTC prices):
```bash
python tools/get_prices.py
```

Custom tokens and intervals:
```bash
# Get 5-minute data for specific tokens
python tools/get_prices.py --tokens ETH WBTC SOL --interval 5m --days 1

# Get daily data for the last 90 days
python tools/get_prices.py --tokens ETH WBTC --days 90
```

## Requirements

- Python 3.8+
- Conda (Miniconda or Anaconda)
- See environment.yml for package dependencies
