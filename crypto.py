"""Module for processing and analyzing stock market data."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from scipy import signal
from alpaca_trade_api.rest import TimeFrame
from matplotlib.offsetbox import AnchoredText
from datetime import date, timedelta, datetime
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
args = parser.parse_args()

# Define the time frame
TIMEFRAME = TimeFrame.Minute if args.timeframe == "Minute" else TimeFrame.Day

# Get today's date
END_DATE = date.today()
END_DATE = datetime.combine(END_DATE, datetime.min.time())

# Get the NYSE trading calendar
NYSE = mcal.get_calendar("NYSE")

# Calculate the start date
START_DATE = NYSE.valid_days(
    start_date=END_DATE - timedelta(days=2 * args.ndays), end_date=END_DATE
)[-args.ndays]

# Convert to Python datetime object if needed
START_DATE = pd.Timestamp(START_DATE).to_pydatetime()

# Convert symbol to upper case
SYMBOL = args.symbol.upper()

# Create a DataConfig instance with all necessary parameters
DATA_CONFIG = DataConfig(SYMBOL, TIMEFRAME, START_DATE, END_DATE, args.ndays)

# Pass the DataConfig instance to the DataFetcher constructor
DATA_FETCHER = DataFetcher(DATA_CONFIG)

CURRENT_PRICE, data = DATA_FETCHER.fetch_crypto_data()

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
    args.window = round(args.ndays // 14.40)
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

for i in range(
    1, len(data)
):  # start from the second row since we are checking with the previous row
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
ActionComputer.compute_actions(SYMBOL, data, END_DATE)

data["close_detrend_norm_filt_adj"] = data["close_detrend_norm_filt"] * max(
    abs(data["close_detrend"])
) + trend_line(x)

# Create subplots with custom layout
fig, (ax1, ax2, ax3) = plt.subplots(
    3, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)
fig.set_size_inches(12, 10)

# Add information text box
info_text = f"Filter window: {window_size}"
anchored_text = AnchoredText(info_text, loc="upper left", prop=dict(size=8), frameon=False)
anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(anchored_text)

color_dict = {"Red": "green", "Green": "red"}

DataPlotter.plot_close_price(data, SYMBOL, ax1, color_dict)
DataPlotter.plot_detrended_data(data, args, ax2)
DataPlotter.plot_z_score_velocity(data, args, ax3)
DataPlotter.plot_buy_sell_markers(data, ax1, ax2)

plt.tight_layout()
plt.show()
