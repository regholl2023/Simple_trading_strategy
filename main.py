import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from scipy import signal
from datetime import date, timedelta
from alpaca_trade_api.rest import REST, TimeFrame

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

# Instantiate Alpaca API
api = REST(api_key, api_secret)

# Fetch OHLC data
df = api.get_bars(
    symbol,
    timeframe,
    start_date.isoformat(),
    end_date.isoformat(),
    adjustment="split",
).df
data = pd.DataFrame(df["close"])

# Get current_price
current_price = api.get_latest_trade(symbol).price

# Compare the last date in the data dataframe with the current date
last_date_in_data = data.index[-1].date()
end_date_tz = pd.Timestamp(end_date).tz_localize(data.index.tz)

if last_date_in_data == end_date_tz.date():
    # If they are the same and the current price is different, replace the stored price
    if current_price != data.iloc[-1, 0]:
        data.iloc[-1, 0] = current_price
else:
    # Append the date and current price to the data dataframe
    data = data.append(pd.DataFrame({'close': current_price}, index=[end_date_tz]))

# Store datetime information in a new column
data["DateTime"] = data.index

# Reset index and drop the old one
data.reset_index(drop=True, inplace=True)

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

# print(data)

data["close_detrend_norm_filt_adj"] = data["close_detrend_norm_filt"] * max(
    abs(data["close_detrend"])
) + trend_line(x)
color_dict = {"Red": "green", "Green": "red"}

# Create subplots with custom layout
fig, (ax1, ax2, ax3) = plt.subplots(
    3, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)
fig.set_size_inches(12, 10)

# Plot the close price
data["close"].plot(
    ax=ax1,
    grid=True,
    title=f'Close price for {symbol} from {data.index.min().strftime("%Y-%m-%d")} to {data.index.max().strftime("%Y-%m-%d")}, last close price: {data["close"].iloc[-1]}',
)

data["close_detrend_norm_filt_adj"].plot(
    ax=ax1, grid=True, color="black", label="Filtered Close Price"
)
previous_row = data.iloc[0]
for i, row in data.iloc[1:].iterrows():
    segment_color = color_dict.get(row["Color"], "black")
    ax1.plot(
        [previous_row.name, i],
        [
            previous_row["close_detrend_norm_filt_adj"],
            row["close_detrend_norm_filt_adj"],
        ],
        color=segment_color,
        alpha=0.12,
        linewidth=7.0,
    )
    previous_row = row
ax1.set_ylabel("Price")
ax1.grid(color="lightgrey")
ax1.legend()

# Plot the detrended, normalized close price and the filtered data
data["close_detrend_norm"].plot(
    ax=ax2,
    grid=True,
    title="Detrended and Normalized Close Price",
    label="Detrended and Normalized",
)
data["close_detrend_norm_filt"].plot(
    ax=ax2, grid=True, label="Low-pass Filtered"
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Normalized and Filtered Price")
ax2.grid(color="lightgrey")
ax2.legend()

# Plot the z-score of the filtered velocity
ax3.plot(data.index, filtered_velocity_zscore, color="blue")
ax3.axhline(0, color="black", linewidth=1)
ax3.axhline(args.std_dev, color="black", linewidth=1, linestyle="--")
ax3.axhline(-args.std_dev, color="black", linewidth=1, linestyle="--")

# Color z-score values above +std_dev in red and below -std_dev in green
ax3.fill_between(
    data.index,
    filtered_velocity_zscore,
    args.std_dev,
    where=filtered_velocity_zscore > args.std_dev,
    facecolor="red",
    alpha=0.3,
)
ax3.fill_between(
    data.index,
    filtered_velocity_zscore,
    -args.std_dev,
    where=filtered_velocity_zscore < -args.std_dev,
    facecolor="green",
    alpha=0.3,
)

ax3.set_xlabel("Date")
ax3.set_ylabel("Filtered Velocity (Z-score)")
ax3.set_title("Z-score of Filtered Velocity")
ax3.grid(color="lightgrey")

# Marker size
marker_size = 100

# Extract rows where 'Action' is 'Buy' or 'Sell'
buy_actions = data[data["Action"] == "Buy"]
sell_actions = data[data["Action"] == "Sell"]

# Add these points to the second subplot
ax2.scatter(
    buy_actions.index,
    buy_actions["close_detrend_norm"],
    color="green",
    marker="^",
    label="Buy",
    s=marker_size,
)
ax2.scatter(
    sell_actions.index,
    sell_actions["close_detrend_norm"],
    color="red",
    marker="v",
    label="Sell",
    s=marker_size,
)

# Marker size
marker_size = 100

# Offset percentages
buy_offset = 0.05  # Buy marker will be placed 5% below the current price
sell_offset = 0.05  # Sell marker will be placed 5% above the current price

# Extract rows where 'Action' is 'Buy' or 'Sell'
buy_actions = data[data["Action"] == "Buy"]
sell_actions = data[data["Action"] == "Sell"]

# Get the limits of the y-axis
y_min, y_max = ax1.get_ylim()

# Compute 5% of the range of the y-axis
offset = (y_max - y_min) * 0.05

# Add these points to the first subplot
for buy_date, buy_data in buy_actions.iterrows():
    ax1.scatter(
        buy_date,
        buy_data["close"],
        color="green",
        marker="^",
        label="Buy",
        s=marker_size,
    )
    ax1.text(
        buy_date,
        buy_data["close"] - offset,  # Adjust y position by offset
        f'Buy: {buy_data["close"]:.2f}',
        color="green",
        verticalalignment="top",
        horizontalalignment="center",
    )

for sell_date, sell_data in sell_actions.iterrows():
    ax1.scatter(
        sell_date,
        sell_data["close"],
        color="red",
        marker="v",
        label="Sell",
        s=marker_size,
    )
    ax1.text(
        sell_date,
        sell_data["close"] + offset,  # Adjust y position by offset
        f'Sell: {sell_data["close"]:.2f}',
        color="red",
        verticalalignment="bottom",
        horizontalalignment="center",
    )

# Update the legends for the subplots to include the 'Buy' and 'Sell' actions
ax2.legend()

plt.tight_layout()
plt.show()
