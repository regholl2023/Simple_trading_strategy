"""
Main script to fetch OHLC data, apply trend and filtering, and visualize the results.
"""

import os
import argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from scipy import signal
from alpaca_trade_api.rest import TimeFrame
from matplotlib.offsetbox import AnchoredText

from helper import (
    fetch_data,
    compute_actions,
    plot_close_price,
    plot_detrended_data,
    plot_z_score_velocity,
    plot_buy_sell_markers,
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def compute_trend_and_filter(num_samples, data_percent, window_size):
    """
    Compute the trend line and apply the filter to the given data.

    :param num_samples: Number of samples in the data
    :param data_percent: Percentage data
    :param window_size: Window size for filtering
    :return: Filtered data, gradient, intercept
    """
    gradient_val = (data_percent.iloc[-1] - data_percent.iloc[0]) / (
        num_samples - 1
    )
    intercept_val = data_percent.iloc[0]

    remove_trend = [
        data_percent.iloc[i] - ((gradient_val * i) + intercept_val)
        for i in range(num_samples)
    ]

    computed_filter = signal.windows.hann(window_size)
    filter_result = signal.convolve(
        remove_trend, computed_filter, mode="same"
    ) / sum(computed_filter)

    data_filter_result = [
        filter_result[i] + ((gradient_val * i) + intercept_val)
        for i in range(num_samples)
    ]

    return data_filter_result, gradient_val, intercept_val


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch OHLC data.")
parser.add_argument("-s", "--symbol", help="Stock symbol", required=True)
parser.add_argument(
    "-n", "--ndays", help="Number of trading days", type=int, default=504
)
parser.add_argument(
    "-w",
    "--window",
    help="Define window size for the Hanning filter",
    type=int,
)
parser.add_argument(
    "-sd",
    "--std_dev",
    help="Number of standard deviations for the dashed lines",
    type=float,
    default=0.01,
)
parser.add_argument(
    "-t",
    "--timeframe",
    help="Timeframe for the OHLC data",
    choices=["Day", "Minute"],
    default="Day",
)
parser.add_argument(
    "-ns", "--num_samples",
    help="Number of samples",
    type=int,
    default=5000,
)
args = parser.parse_args()

# Define the time frame
TIMEFRAME = TimeFrame.Minute if args.timeframe == "Minute" else TimeFrame.Day

# Get today's date
END_DATE = date.today()

# Get the NYSE trading calendar
NYSE = mcal.get_calendar("NYSE")

# Calculate the start date
START_DATE = NYSE.valid_days(
    start_date=END_DATE - timedelta(days=2 * args.ndays), end_date=END_DATE
)[-args.ndays]

# Convert symbol to upper case
SYMBOL = args.symbol.upper()

# Get API keys from environment variables
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

# Fetch OHLC data using the fetch_data function from helper.py
CURRENT_PRICE, DATA = fetch_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE, args.ndays, args.timeframe, args.num_samples)

# Create linear trend line
X_VALUES = np.linspace(0, len(DATA.index) - 1, len(DATA.index))
Y_VALUES = DATA["close"]
Y_TREND = np.linspace(Y_VALUES.iloc[0], Y_VALUES.iloc[-1], len(DATA.index))
TREND_LINE = np.poly1d(np.polyfit(X_VALUES, Y_TREND, 1))

# Subtract trend line from close prices
DATA["close_detrend"] = DATA["close"] - TREND_LINE(X_VALUES)

# Normalize the detrended data
DATA["close_detrend_norm"] = DATA["close_detrend"] / max(
    abs(DATA["close_detrend"])
)

if args.window is None or args.window == 0:
    if args.timeframe == 'Minute':
        args.window = round(DATA.shape[0] // 8.17)
    else:
        args.window = round(DATA.shape[0] // 14.40)
    if (args.window % 2) == 0:
        args.window += 1

WINDOW_SIZE = args.window
DATA_FILTER, GRADIENT, INTERCEPT = compute_trend_and_filter(
    len(DATA["close_detrend_norm"]), DATA["close_detrend_norm"], WINDOW_SIZE
)

# Use only filtered data for the next steps
DATA["close_detrend_norm_filt"] = DATA_FILTER

# Calculate the filtered velocity (first derivative)
FILTERED_VELOCITY = np.gradient(
    DATA["close_detrend_norm_filt"], X_VALUES[1] - X_VALUES[0]
)

# Calculate the z-score of the filtered velocity
FILTERED_VELOCITY_ZSCORE = (
    FILTERED_VELOCITY - np.mean(FILTERED_VELOCITY)
) / np.std(FILTERED_VELOCITY)

# Add z-score velocity to a new column in the dataframe
DATA["zscore_velocity"] = FILTERED_VELOCITY_ZSCORE

# Add 'Color' column to the dataframe, setting color based on z-score velocity
DATA["Color"] = np.where(
    DATA["zscore_velocity"] >= args.std_dev,
    "Red",
    np.where(DATA["zscore_velocity"] <= -args.std_dev, "Green", ""),
)

# Add 'Action' column to the dataframe, setting all rows to an empty string initially
DATA["Action"] = ""

LAST_RED = None
LAST_GREEN = None

first = int(DATA.index[0]+1)
last = int(DATA.index[-1])

for i in range(first, last):
    # Update LAST_RED or LAST_GREEN
    if DATA.loc[i - 1, "Color"] == "Red":
        LAST_RED = i - 1
    elif DATA.loc[i - 1, "Color"] == "Green":
        LAST_GREEN = i - 1

    # Set the 'Sell' action at the point of the last change after the last 'Red'
    if (
        DATA.loc[i - 1, "zscore_velocity"]
        > 0
        > DATA.loc[i, "zscore_velocity"]
        and LAST_RED is not None
    ):
        DATA.loc[i, "Action"] = "Sell"
        LAST_RED = None  # Reset LAST_RED

    # Set the 'Buy' action at the point of the last change after the last 'Green'
    if (
        DATA.loc[i - 1, "zscore_velocity"]
        < 0
        < DATA.loc[i, "zscore_velocity"]
        and LAST_GREEN is not None
    ):
        DATA.loc[i, "Action"] = "Buy"
        LAST_GREEN = None  # Reset LAST_GREEN

# Set 'DateTime' as the index
DATA.set_index("DateTime", inplace=True)

# Compute last action and print important information
compute_actions(SYMBOL, DATA, END_DATE, args.timeframe)

DATA["close_detrend_norm_filt_adj"] = DATA["close_detrend_norm_filt"] * max(
    abs(DATA["close_detrend"])
) + TREND_LINE(X_VALUES)

# Create subplots with custom layout
FIG, (AX1, AX2, AX3) = plt.subplots(
    3, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)
FIG.set_size_inches(12, 10)

# Add information text box
info_text = f"Filter window: {WINDOW_SIZE}"
anchored_text = AnchoredText(info_text, loc="lower left", prop=dict(size=8), frameon=False)
anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
AX3.add_artist(anchored_text)

COLOR_DICT = {"Red": "green", "Green": "red"}

plot_close_price(DATA, SYMBOL, AX1, COLOR_DICT, args.timeframe)
plot_detrended_data(DATA, args, AX2, args.timeframe)
plot_z_score_velocity(DATA, args, AX3)
plot_buy_sell_markers(DATA, AX1, AX2, args.timeframe)

plt.tight_layout()
plt.show()
