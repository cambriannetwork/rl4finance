# RL4Finance

A Python project for portfolio optimization and cryptocurrency price data analysis.

## Project Structure

```
RL4Finance/
├── appendix-b/
│   ├── main.py         # Portfolio optimization with efficient frontier
│   └── main2.py        # Extended version with plot saving
├── tools/
│   └── get_prices.py   # Crypto price data fetcher
├── setup.sh            # Setup script
└── requirements.txt    # Project dependencies
```

## Features

### Portfolio Optimization (appendix-b/)

The `main.py` and `main2.py` scripts demonstrate portfolio optimization techniques:
- Efficient Frontier calculation
- Global Minimum Variance Portfolio (GMVP)
- Random portfolio generation
- Risk-return visualization

Key differences between the scripts:
- `main.py`: Displays plots interactively
- `main2.py`: Saves plots to files (efficient_frontier.png and gmvp_plot.png)

Both scripts use synthetic data to demonstrate the concepts of:
- Mean returns calculation
- Covariance matrix computation
- Portfolio weight optimization
- Risk-return trade-off visualization

### Cryptocurrency Price Data (tools/)

The `get_prices.py` script fetches historical cryptocurrency price data using the CoinGecko Pro API:
- Default tokens: ETH and BTC
- Customizable time intervals (5m, hourly, daily)
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

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure CoinGecko API:
   - Create a `.env` file in the root directory
   - Add your CoinGecko Pro API key:
     ```
     COINGECKO_API_KEY=your_api_key_here
     ```

## Usage

### Portfolio Optimization

Run either script to see portfolio optimization in action:
```bash
python appendix-b/main.py
# or
python appendix-b/main2.py
```

The scripts will:
1. Generate synthetic asset data
2. Calculate efficient frontier
3. Find the Global Minimum Variance Portfolio
4. Display/save visualization plots

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
