import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
from alpaca_trade_api.rest import REST

def fetch_data(symbol, timeframe, start_date, end_date, ndays):
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

    # Convert the index to datetime type
    data.index = pd.to_datetime(data.index)

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
        data = data.append(
            pd.DataFrame({"close": current_price}, index=[end_date_tz])
        )

    # Store datetime information in a new column
    data["DateTime"] = data.index

    # Reset index and drop the old one
    data.reset_index(drop=True, inplace=True)

    return current_price, data

def compute_actions(data, end_date):
    # Logic to compute actions (Buy/Sell) and print the most recent action
    # ... (rest of the code for computing actions and printing details)

    # Extract rows where 'Action' is 'Buy' or 'Sell'
    buy_actions = data[data["Action"] == "Buy"]
    sell_actions = data[data["Action"] == "Sell"]

    # Get the last buy and sell dates
    last_buy_date = buy_actions.index[-1] if not buy_actions.empty else None
    last_sell_date = sell_actions.index[-1] if not sell_actions.empty else None

    # Determine the most recent action and print it
    if last_buy_date and last_sell_date:
        if last_buy_date > last_sell_date:
            last_action = "Buy"
            last_action_date = last_buy_date
            last_action_price = buy_actions.loc[last_buy_date, "close"]
        else:
            last_action = "Sell"
            last_action_date = last_sell_date
            last_action_price = sell_actions.loc[last_sell_date, "close"]
    elif last_buy_date:
        last_action = "Buy"
        last_action_date = last_buy_date
        last_action_price = buy_actions.loc[last_buy_date, "close"]
    elif last_sell_date:
        last_action = "Sell"
        last_action_date = last_sell_date
        last_action_price = sell_actions.loc[last_sell_date, "close"]
    else:
        last_action = None

    if last_action:
        # Find the row number from the end where the last action date is
        rows_from_end = len(data) - data.index.get_loc(last_action_date)

        days_ago = (end_date - last_action_date.date()).days
        print(
                f'The last action was a {last_action} on {last_action_date.strftime("%Y-%m-%d")} ({days_ago} days ago, or {rows_from_end} trading-days ago) at a price of {last_action_price:.2f}, last price {data["close"].iloc[-1]:.3f}'
    )
    else:
        print("No Buy or Sell actions were recorded.")

    return

def plot_close_price(data, symbol, ax1, color_dict):
    # Plotting close price and filtered close price
    # ... (rest of the code for plotting close price)

    data["close"].plot(
        ax=ax1,
        grid=True,
        title=f'Close price for {symbol} from {data.index.min().strftime("%Y-%m-%d")} to {data.index.max().strftime("%Y-%m-%d")}, last price: {data["close"].iloc[-1]}',
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

def plot_detrended_data(data, args, ax2):
    # Plotting detrended, normalized close price and the filtered data
    # ... (rest of the code for plotting detrended data)

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

def plot_z_score_velocity(data, args, ax3):
    # Plotting z-score of the filtered velocity
    # ... (rest of the code for plotting z-score velocity)

    ax3.plot(data.index, data["zscore_velocity"], color="blue")
    ax3.axhline(0, color="black", linewidth=1)
    ax3.axhline(args.std_dev, color="black", linewidth=1, linestyle="--")
    ax3.axhline(-args.std_dev, color="black", linewidth=1, linestyle="--")

    # Color z-score values above +std_dev in red and below -std_dev in green
    ax3.fill_between(
        data.index,
        data["zscore_velocity"],
        args.std_dev,
        where=data["zscore_velocity"] > args.std_dev,
        facecolor="red",
        alpha=0.3,
    )
    ax3.fill_between(
        data.index,
        data["zscore_velocity"],
        -args.std_dev,
        where=data["zscore_velocity"] < -args.std_dev,
        facecolor="green",
        alpha=0.3,
    )

    ax3.set_xlabel("Date")
    ax3.set_ylabel("Filtered Velocity (Z-score)")
    ax3.set_title("Z-score of Filtered Velocity")
    ax3.grid(color="lightgrey")

def plot_buy_sell_markers(data, ax1, ax2):
    # Plotting buy and sell markers
    # ... (rest of the code for plotting buy and sell markers)

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
