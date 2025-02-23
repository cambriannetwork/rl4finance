# RL4Finance

A Python project for portfolio optimization and cryptocurrency price data analysis.

## Project Structure

```
RL4Finance/
├── appendix-b/
│   ├── main.py         # Portfolio optimization with efficient frontier
│   ├── main2.py        # Extended version with plot saving
│   └── covariance.py   # Asset returns covariance/correlation analysis
├── tools/
│   └── get_prices.py   # Crypto price data fetcher
├── common/             # Shared functionality
│   └── data.py         # Data loading and processing functions
├── setup.py            # Package installation configuration
└── requirements.txt    # Project dependencies
```

## Features

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

2. Run the setup script:
   ```bash
   ./setup.sh
   ```

3. Install package in editable mode with dependencies:
   ```bash
   pip install -e .
   ```

4. Configure CoinGecko API:
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
- See requirements.txt for package dependencies
