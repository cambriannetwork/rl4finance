# RL4Finance

> **WARNING:** This repository is intended for educational purposes only. It serves as an experimental implementation of concepts from the book [Reinforcement Learning for Finance](https://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf).

A Python project for reinforcement learning in finance, portfolio optimization, and cryptocurrency price data analysis.

## Project Structure

```
RL4Finance/
├── chapters/           # Reinforcement learning models
│   └── 7/              # Chapter 7 models
│       ├── asset_alloc.py      # Implementation using rl_lib
│       └── asset_alloc_toy.py  # Toy model with backward induction
├── rl_lib/             # Modular reinforcement learning library
│   ├── distribution/   # Probability distributions
│   ├── mdp/            # Markov Decision Processes
│   ├── function_approx/# Function approximation
│   ├── utils/          # Utility functions
│   ├── adp/            # Approximate Dynamic Programming
│   ├── logging/        # JSON-formatted logging functionality
│   └── tests/          # Unit tests
├── appendix-b/          # Portfolio optimization scripts
│   ├── frontier.py     # Portfolio optimization with efficient frontier
│   ├── covariance.py   # Asset returns covariance/correlation analysis
│   └── return_and_risk.py  # Return and risk analysis
├── tools/              # Utility scripts
│   ├── get_prices.py   # Crypto price data fetcher
│   ├── visualize.py    # Price visualization tool
│   └── return-histogram.py # Return distribution visualization
├── common/             # Shared functionality
│   └── data.py         # Data loading and processing functions
├── data/               # Price data storage (created by setup)
├── logs/               # JSON log files (created at runtime)
├── setup.py            # Package installation configuration
└── requirements.txt    # Project dependencies
```

Note: The `data/` directory is automatically created during setup and is used to store price data files. This directory is excluded from version control.

## Features

### Reinforcement Learning Library (rl_lib/)

The `rl_lib` package is a modular reinforcement learning library that provides components for building and solving reinforcement learning problems. It includes:

- **Distribution Module**: Classes for probability distributions (Gaussian, Choose, Constant, etc.)
- **MDP Module**: Classes for Markov Decision Processes, states, and policies
- **Function Approximation Module**: Neural network function approximation with Adam optimization
- **Utils Module**: Utility functions for iteration, convergence, and accumulation
- **ADP Module**: Approximate Dynamic Programming algorithms including backward induction
- **Logging Module**: JSON-formatted logging with automatic timestamped log files

Features:
- Modular design with clear separation of concerns
- Type hints for better code readability and IDE support
- Comprehensive docstrings for all classes and methods
- Unit tests to verify functionality

Installation:
```bash
# Install the rl_lib package in development mode
cd rl_lib
pip install -e .
```

### Reinforcement Learning Models (chapters/)

#### Asset Allocation Models (chapters/7/)

##### Main Implementation (asset_alloc.py)

This implementation uses the modular `rl_lib` package to solve the asset allocation problem:

- Uses the Distribution classes from rl_lib.distribution
- Leverages the MDP framework from rl_lib.mdp
- Employs function approximation from rl_lib.function_approx
- Applies backward induction algorithms from rl_lib.adp

Features:
- Modular implementation that demonstrates the use of the rl_lib package
- Identical results to the standalone implementation
- Cleaner code with better separation of concerns
- Matches the analytical solution for optimal asset allocation

Usage:
```bash
# Run the implementation using rl_lib
python chapters/7/asset_alloc.py

# Enable debug logging with info level
python chapters/7/asset_alloc.py --debug --log-level info

# Use a custom log file name
python chapters/7/asset_alloc.py --debug --log-file my_custom_log.json
```

The output shows:
1. Backward induction results for each time step
2. Optimal risky asset allocation and value
3. Optimal weights for the function approximation
4. Analytical solution for comparison

##### Toy Model Implementation (chapters/7/asset_alloc_toy.py)

A simplified implementation of the asset allocation problem using backward induction:

- Uses a grid-based approach for wealth representation
- Implements Monte Carlo sampling for return distributions
- Visualizes optimal policy vs. wealth for each time step
- Demonstrates core concepts without the full rl_lib framework

Features:
- Simple, self-contained implementation for educational purposes
- Discrete set of possible actions (risky-asset allocations)
- CARA utility function with risk aversion parameter
- Visualization of optimal policy across time steps

Usage:
```bash
# Run the toy implementation
python chapters/7/asset_alloc_toy.py
```

#### Logging Features

The asset allocation model includes comprehensive JSON-formatted logging:

- Logs are stored in the `logs/` directory with timestamped filenames
- Debug mode can be enabled with the `--debug` flag
- Log levels can be set with `--log-level` (debug, info, warning, error)
- Custom log file names can be specified with `--log-file`
- Logs include:
  - Model configuration parameters
  - Backward induction progress
  - Weight summaries at each time step
  - Optimal allocations and values
  - Analytical solution details
  - Comparison between numerical and analytical results

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

### Cryptocurrency Price Data and Visualization (tools/)

#### Price Data Collection (get_prices.py)

The `get_prices.py` script fetches historical cryptocurrency price data using the CoinGecko Pro API:
- Default tokens: ETH and BTC
- Support for additional tokens: WBTC, SOL, GRT, ADA, BNB
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


#### Price Visualization (visualize.py)

The `visualize.py` script provides visualization tools for cryptocurrency price data:

- Plots price data for all tokens in a single chart
- Automatically converts millisecond timestamps to datetime for plotting
- Displays min/max price information in the legend
- Supports saving plots to file
- Works with the latest price data by default

Features:
- Clean, informative visualization with proper date formatting
- Automatic detection of date ranges from the data
- Grid lines and formatted axes for better readability
- Detailed information about the data being visualized

Usage:
```bash
# Basic usage (uses latest price file)
python tools/visualize.py

# Use a specific price file
python tools/visualize.py --file data/prices_2025-03-14.csv

# Save the plot to a file
python tools/visualize.py --save prices_chart.png
```

#### Return Distribution Analysis (return-histogram.py)

The `return-histogram.py` script visualizes the distribution of cryptocurrency returns:

- Creates histograms of returns for all tokens in a grid layout
- Color-codes returns (red for negative, green for positive)
- Displays detailed statistics for each token:
  - Mean
  - Median
  - Standard deviation
  - Skewness
  - Kurtosis
- Supports different return periods and types
- Optimizes bin sizes for consistent visualization
- Automatically focuses on the actual data range

Features:
- Grid layout for easy comparison between tokens
- Color-coded histograms for intuitive interpretation
- Statistical summary for quantitative analysis
- Consistent bin widths for accurate visual comparison
- Customizable options for different analysis needs

Usage:
```bash
# Basic usage (daily arithmetic returns)
python tools/return-histogram.py

# Weekly returns
python tools/return-histogram.py --period weekly

# Log returns
python tools/return-histogram.py --return-type log

# Save the plot to a file
python tools/return-histogram.py --save returns_histogram.png
```

## Usage

### Visualization

```bash
# Visualize price data
python tools/visualize.py

# Visualize return distributions
python tools/return-histogram.py

# Save visualizations to files
python tools/visualize.py --save my_price_chart.png
python tools/return-histogram.py --save my_returns_histogram.png
```

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
