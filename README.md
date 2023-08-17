# Stock Analysis and Trading Strategy Generator

This Python program is designed to analyze stock data and generate trading strategies based on trend analysis and filtering techniques. It utilizes historical data, computes trends, applies filters, and identifies buy/sell actions.

## Features

- Fetches OHLC (Open, High, Low, Close) data for a given stock symbol
- Computes linear trend lines and detrends the close prices
- Normalizes and filters the detrended data using a Hanning filter
- Calculates the filtered velocity and its z-score
- Generates buy and sell actions based on the z-score threshold
- Plots the close price, detrended data, filtered data, and z-score of filtered velocity
- Visualizes buy and sell actions on the plots

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `pandas_market_calendars`
- `scipy`
- `argparse`
- `alpaca_trade_api`

## Usage

1. Install the required dependencies: `pip install -r requirements.txt`
2. Set up API keys for the Alpaca API by setting environment variables: `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`.
3. Run the program with the desired command-line arguments. For example:

```
python main.py -s AAPL -n 504 -w 31 -sd 0.01 -t Day
```
- `-s/--symbol`: Stock symbol to analyze
- `-n/--ndays`: Number of trading days to fetch data for
- `-w/--window`: Window size for the Hanning filter
- `-sd/--std_dev`: Number of standard deviations for the z-score threshold
- `-t/--timeframe`: Timeframe for the OHLC data (Day or Minute)

## Examples

- Fetch and analyze 504 trading days of AAPL stock data with a Hanning filter window of 31 and a z-score threshold of 0.01:

```
python main.py -s AAPL -n 504 -w 31 -sd 0.01 -t Day
```

![Example_plot_display](images/Figure_1.png)

Here is AAPL at a later time:

![Example_plot_display](images/Figure_2.png)

AAPL  last action was Sell on 2023-07-20 (  28 days ago, or   20 trading-days ago) at a price of  193.130 last price  174.010 percent change     9.900

## License

This project is licensed under the [MIT License](LICENSE).
