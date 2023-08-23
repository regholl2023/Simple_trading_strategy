import os
import argparse
import numpy as np
import pandas as pd

from scipy import signal
from datetime import date, timedelta
from alpaca_trade_api.rest import TimeFrame
from multi_helper_class import DataFetcher, ActionComputer

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


class DataProcessor:
    @staticmethod
    def compute_trend_and_filter(num_samples, data_percent, window):
        gradient = (data_percent.iloc[-1] - data_percent.iloc[0]) / (
            num_samples - 1
        )
        intercept = data_percent.iloc[0]
        remove_trend = data_percent - (
            (gradient * np.arange(num_samples)) + intercept
        )
        computed_filter = signal.windows.hann(window)
        filter_result = signal.convolve(
            remove_trend, computed_filter, mode="same"
        ) / sum(computed_filter)
        data_filter = filter_result + (
            (gradient * np.arange(num_samples)) + intercept
        )
        return data_filter, gradient, intercept


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch OHLC data.")
parser.add_argument(
    "-l",
    "--list",
    help="Path to file containing stock symbols, one per line",
    required=True,
)
parser.add_argument(
    "-n", "--ndays", help="Number of trading days", type=int, default=504
)
parser.add_argument(
    "-w",
    "--window",
    help="Define window size for the Hanning filter",
    type=int,
    default=0,
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
    "-ns",
    "--num_samples",
    help="Number of samples",
    type=int,
    default=5000,
)
args = parser.parse_args()

# Read symbols from file
with open(args.list, "r") as file:
    symbols = [line.split()[0].upper() for line in file]

# Define the time frame
timeframe = TimeFrame.Minute if args.timeframe == "Minute" else TimeFrame.Day

data_fetcher = DataFetcher(
    symbols, timeframe, args.ndays, args.num_samples, args.timeframe
)
df, end_date = data_fetcher.fetch_historical_data_v2()

data = pd.DataFrame()

for symbol in symbols:
    if args.timeframe == "Day":
        symbol_data = df[df["symbol"] == symbol][-args.ndays :].copy()
    else:
        symbol_data = df[df["symbol"] == symbol][-args.num_samples :].copy()
    symbol_data["t"] = pd.to_datetime(symbol_data["t"])
    symbol_data = symbol_data.sort_values(by="t")
    symbol_data.reset_index(drop=True, inplace=True)
    temp_df = symbol_data.reset_index(drop=True)["c"].to_frame()
    data = pd.concat([data, temp_df], axis=0, ignore_index=True)
    current_price = data.iloc[-1].values

    # Create linear trend line
    x = np.linspace(0, len(temp_df.index) - 1, len(temp_df.index))
    y = temp_df["c"]
    y_trend = np.linspace(y.iloc[0], y.iloc[-1], len(temp_df.index))
    trend_line = np.poly1d(np.polyfit(x, y_trend, 1))

    # Subtract trend line from close prices
    symbol_data["close_detrend"] = temp_df["c"] - trend_line(x)

    # Normalize the detrended data
    symbol_data["close_detrend_norm"] = symbol_data["close_detrend"] / max(
        abs(symbol_data["close_detrend"])
    )

    if args.window is None or args.window == 0:
        if args.timeframe == "Minute":
            args.window = round(data.shape[0] // 8.17)
        else:
            args.window = round(data.shape[0] // 14.40)
        if (args.window % 2) == 0:
            args.window += 1

    window = args.window
    data_filter, gradient, intercept = DataProcessor.compute_trend_and_filter(
        len(symbol_data["close_detrend_norm"]),
        symbol_data["close_detrend_norm"],
        window,
    )

    # Use only filtered data for the next steps
    symbol_data["close_detrend_norm_filt"] = data_filter

    # Calculate the filtered velocity (first derivative)
    filtered_velocity = np.gradient(
        symbol_data["close_detrend_norm_filt"], x[1] - x[0]
    )

    # Calculate the z-score of the filtered velocity
    filtered_velocity_zscore = (
        filtered_velocity - np.mean(filtered_velocity)
    ) / np.std(filtered_velocity)

    # Add z-score velocity to a new column in the dataframe
    symbol_data["zscore_velocity"] = filtered_velocity_zscore

    # Add 'Color' column to the dataframe, setting color based on z-score velocity
    symbol_data["Color"] = np.where(
        symbol_data["zscore_velocity"] >= args.std_dev,
        "Red",
        np.where(
            symbol_data["zscore_velocity"] <= -args.std_dev, "Green", ""
        ),
    )

    # Add 'Action' column to the dataframe, setting all rows to an empty string initially
    symbol_data["Action"] = ""

    last_red = None
    last_green = None

    first = int(symbol_data.index[0] + 1)
    last = int(symbol_data.index[-1])

    for i in range(first, last):
        # Update last_red or last_green
        if symbol_data.loc[i - 1, "Color"] == "Red":
            last_red = i - 1
        elif symbol_data.loc[i - 1, "Color"] == "Green":
            last_green = i - 1

        # Set the 'Sell' action at the point of the last change after the last 'Red'
        if (
            symbol_data.iloc[i - 1]["zscore_velocity"] > 0
            and symbol_data.iloc[i]["zscore_velocity"] < 0
            and last_red is not None
        ):
            symbol_data.loc[symbol_data.index[i], "Action"] = "Sell"
            last_red = None  # Reset last_red

        # Set the 'Buy' action at the point of the last change after the last 'Green'
        if (
            symbol_data.iloc[i - 1]["zscore_velocity"] < 0
            and symbol_data.iloc[i]["zscore_velocity"] > 0
            and last_green is not None
        ):
            symbol_data.loc[symbol_data.index[i], "Action"] = "Buy"
            last_green = None  # Reset last_green

    # Set 't' as the index
    symbol_data.set_index("t", inplace=True)

    # Compute last action and print important information
    ActionComputer.compute_actions(
        symbol, symbol_data, end_date, args.timeframe
    )
