"""Module for processing and analyzing stock market data."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from datetime import timedelta, datetime
from alpaca_trade_api.rest import TimeFrame
from matplotlib.offsetbox import AnchoredText
from crypto_helper import DataConfig, DataFetcher, ActionComputer, DataPlotter

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def compute_trend_and_filter(num_samples, data_percent, window_size):
    """Compute trend and filter for given data."""
    gradient_val = (data_percent.iloc[-1] - data_percent.iloc[0]) / (
        num_samples - 1
    )
    intercept_val = data_percent.iloc[0]
    remove_trend = data_percent - (
        gradient_val * np.arange(num_samples) + intercept_val
    )
    computed_filter = signal.windows.hann(window_size)
    filter_result = signal.convolve(
        remove_trend, computed_filter, mode="same"
    ) / sum(computed_filter)
    data_filter_val = filter_result + (
        gradient_val * np.arange(num_samples) + intercept_val
    )
    return data_filter_val, gradient_val, intercept_val


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch OHLC data.")
parser.add_argument("-s", "--symbol", help="Stock symbol (Default: BTC/USD)", default="BTC/USD")
parser.add_argument(
        "-n", "--ndays", help="Number of trading days (Default: 504)", type=int, default=504
)
parser.add_argument(
    "-w",
    "--window",
    help="Define window size for the Hanning filter (Default:0, computed from number of samples)",
    type=int,
)
parser.add_argument(
    "-sd",
    "--std_dev",
    help="Number of standard deviations for the dashed lines (Default: 0.01)",
    type=float,
    default=0.01,
)
parser.add_argument(
    "-t",
    "--timeframe",
    help="Timeframe for the OHLC data (Default: Day)",
    choices=["Day", "Minute"],
    default="Day",
)
parser.add_argument(
    "-ns",
    "--num_samples",
    help="Maximum number of samples in analysis (Default: 5000)",
    type=int,
    default=5000,
)
parser.add_argument(
    "-c",
    "--convert",
    help="Convert prices to standard deviations (0 or 1, Default: 0)",
    type=int,
    choices=[0, 1],
    default=0,
)
parser.add_argument(
    "-p",
    "--plot_switch",
    help="Switch to turn on (1) or off (0) plotting (Default: 1)",
    type=int,
    choices=[0, 1],
    default=1,
)
args = parser.parse_args()

# Define the time frame
TIMEFRAME = TimeFrame.Minute if args.timeframe == "Minute" else TimeFrame.Day

# Get today's date with current UTC time
END_DATE = datetime.utcnow()

# Calculate the start date, which is ndays before the end date
START_DATE = END_DATE - timedelta(days=args.ndays)

# Convert symbol to upper case
SYMBOL = args.symbol.upper()

# Create a DataConfig instance with all necessary parameters
DATA_CONFIG = DataConfig(
    SYMBOL,
    TIMEFRAME,
    START_DATE,
    END_DATE,
    args.ndays,
    args.timeframe,
    args.num_samples,
)

# Pass the DataConfig instance to the DataFetcher constructor
DATA_FETCHER = DataFetcher(DATA_CONFIG)

CURRENT_PRICE, data = DATA_FETCHER.fetch_crypto_data()

data = data[-args.num_samples:]
num_samples = data.shape[0]

# Convert prices to standard deviations if std_dev argument is set to 1
if args.convert == 1:
    # Copy the close column to a close_orig column
    data["close_orig"] = data["close"].copy()
    mean_price = data["close"].mean()
    std_dev_price = data["close"].std()
    data["close"] = (data["close"] - mean_price) / std_dev_price
    CURRENT_PRICE = (CURRENT_PRICE - mean_price) / std_dev_price

# Create linear trend line
x = np.linspace(0, len(data.index) - 1, len(data.index))
y = data["close"]
y_trend = np.linspace(y.iloc[0], y.iloc[-1], len(data.index))
trend_line = np.poly1d(np.polyfit(x, y_trend, 1))

# Subtract trend line from close prices
data["close_detrend"] = data["close"] - trend_line(x)

# Normalize the detrended data
data["close_detrend_norm"] = data["close_detrend"] / max(
    abs(data["close_detrend"])
)

if args.window is None or args.window == 0:
    if args.timeframe == "Minute":
        args.window = round(data.shape[0] // 12.43)
    else:
        args.window = round(data.shape[0] // 9.88)
    if (args.window % 2) == 0:
        args.window += 1

window_size = args.window
data_filter_val, gradient_val, intercept_val = compute_trend_and_filter(
    len(data["close_detrend_norm"]), data["close_detrend_norm"], window_size
)

# Use only filtered data for the next steps
data["close_detrend_norm_filt"] = data_filter_val

# Calculate the filtered velocity (first derivative)
filtered_velocity = np.gradient(data["close_detrend_norm_filt"], x[1] - x[0])

# Calculate the z-score of the filtered velocity
filtered_velocity_zscore = (
    filtered_velocity - np.mean(filtered_velocity)
) / np.std(filtered_velocity)

# Add z-score velocity to a new column in the dataframe
data["zscore_velocity"] = filtered_velocity_zscore

# Add 'Color' column to the dataframe, setting color based on z-score velocity
data["Color"] = np.where(
    data["zscore_velocity"] >= args.std_dev,
    "Red",
    np.where(data["zscore_velocity"] <= -args.std_dev, "Green", ""),
)

# Add 'Action' column to the dataframe, setting all rows to an empty string initially
data["Action"] = ""

LAST_RED = None
LAST_GREEN = None

first = int(data.index[0] + 1)
last = int(data.index[-1])

for i in range(first, last):
    # Update LAST_RED or LAST_GREEN
    if data.loc[i - 1, "Color"] == "Red":
        LAST_RED = i - 1
    elif data.loc[i - 1, "Color"] == "Green":
        LAST_GREEN = i - 1

    # Set the 'Sell' action at the point of the last change after the last 'Red'
    if (
        data.loc[i - 1, "zscore_velocity"] > 0
        and data.loc[i, "zscore_velocity"] < 0
        and LAST_RED is not None
    ):
        data.loc[i, "Action"] = "Sell"
        LAST_RED = None  # Reset LAST_RED

    # Set the 'Buy' action at the point of the last change after the last 'Green'
    if (
        data.loc[i - 1, "zscore_velocity"] < 0
        and data.loc[i, "zscore_velocity"] > 0
        and LAST_GREEN is not None
    ):
        data.loc[i, "Action"] = "Buy"
        LAST_GREEN = None  # Reset LAST_GREEN

# Set 'DateTime' as the index
data.set_index("DateTime", inplace=True)

# Compute last action and print important information
ActionComputer.compute_actions(
    SYMBOL, data, END_DATE, args.timeframe, args.convert, window_size, num_samples
)

if args.plot_switch == 1:
    data["close_detrend_norm_filt_adj"] = data[
        "close_detrend_norm_filt"
    ] * max(abs(data["close_detrend"])) + trend_line(x)

    # Create subplots with custom layout
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    fig.set_size_inches(12, 10)

    # Add information text box
    info_text = f"Filter window: {window_size}"
    anchored_text = AnchoredText(
        info_text, loc="lower left", prop=dict(size=8), frameon=False
    )
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax3.add_artist(anchored_text)

    color_dict = {"Red": "green", "Green": "red"}

    DataPlotter.plot_close_price(
        data, SYMBOL, ax1, color_dict, args.timeframe
    )
    DataPlotter.plot_detrended_data(data, args, ax2, args.timeframe)
    DataPlotter.plot_z_score_velocity(data, args, ax3)
    DataPlotter.plot_buy_sell_markers(data, ax1, ax2, args.timeframe)

    plt.tight_layout()
    plt.show()
