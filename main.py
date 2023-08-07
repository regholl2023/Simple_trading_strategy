import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from scipy import signal
from datetime import date, timedelta
from alpaca_trade_api.rest import REST, TimeFrame

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


def compute_trend_and_filter(num_samples, data_percent, window):
    gradient = (data_percent.iloc[-1] - data_percent.iloc[0]) / (
        num_samples - 1
    )
    intercept = data_percent.iloc[0]

    remove_trend = np.zeros(num_samples)
    for i in range(num_samples):
        remove_trend[i] = data_percent.iloc[i] - ((gradient * i) + intercept)

    computed_filter = signal.windows.hann(window)
    filter_result = signal.convolve(
        remove_trend, computed_filter, mode="same"
    ) / sum(computed_filter)

    data_filter = np.zeros(num_samples)
    for i in range(num_samples):
        data_filter[i] = filter_result[i] + ((gradient * i) + intercept)

    return data_filter, gradient, intercept


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
    default=31,
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
timeframe = TimeFrame.Minute if args.timeframe == "Minute" else TimeFrame.Day

# Get today's date
end_date = date.today()

# Get the NYSE trading calendar
nyse = mcal.get_calendar("NYSE")

# Calculate the start date
start_date = nyse.valid_days(
    start_date=end_date - timedelta(days=2 * args.ndays), end_date=end_date
)[-args.ndays]

# Convert symbol to upper case
symbol = args.symbol.upper()

# Get API keys from environment variables
api_key = os.getenv("APCA_API_KEY_ID")
api_secret = os.getenv("APCA_API_SECRET_KEY")

# Fetch OHLC data using the fetch_data function from helper.py
current_price, data = fetch_data(symbol, timeframe, start_date, end_date, args.ndays)

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

window = args.window
data_filter, gradient, intercept = compute_trend_and_filter(
    len(data["close_detrend_norm"]), data["close_detrend_norm"], window
)

# Use only filtered data for the next steps
data["close_detrend_norm_filt"] = data_filter

# Calculate the filtered velocity (first derivative)
filtered_velocity = np.gradient(data["close_detrend_norm_filt"], x[1] - x[0])

# Calculate the z-score of the filtered velocity
filtered_velocity_zscore = (
    filtered_velocity - np.mean(filtered_velocity)
) / np.std(filtered_velocity)

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

last_red = None
last_green = None

for i in range(
    1, len(data)
):  # start from the second row since we are checking with the previous row
    # Update last_red or last_green
    if data.loc[i - 1, "Color"] == "Red":
        last_red = i - 1
    elif data.loc[i - 1, "Color"] == "Green":
        last_green = i - 1

    # Set the 'Sell' action at the point of the last change after the last 'Red'
    if (
        data.loc[i - 1, "zscore_velocity"] > 0
        and data.loc[i, "zscore_velocity"] < 0
        and last_red is not None
    ):
        data.loc[i, "Action"] = "Sell"
        last_red = None  # Reset last_red

    # Set the 'Buy' action at the point of the last change after the last 'Green'
    if (
        data.loc[i - 1, "zscore_velocity"] < 0
        and data.loc[i, "zscore_velocity"] > 0
        and last_green is not None
    ):
        data.loc[i, "Action"] = "Buy"
        last_green = None  # Reset last_green

# Set 'DateTime' as the index
data.set_index("DateTime", inplace=True)

# Compute last action and print important information
compute_actions(data, end_date)

data["close_detrend_norm_filt_adj"] = data["close_detrend_norm_filt"] * max(
    abs(data["close_detrend"])
) + trend_line(x)

# Create subplots with custom layout
fig, (ax1, ax2, ax3) = plt.subplots(
    3, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)
fig.set_size_inches(12, 10)

color_dict = {"Red": "green", "Green": "red"}

plot_close_price(data, symbol, ax1, color_dict)
plot_detrended_data(data, args, ax2)
plot_z_score_velocity(data, args, ax3)
plot_buy_sell_markers(data, ax1, ax2)

plt.tight_layout()
plt.show()
