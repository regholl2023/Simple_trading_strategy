"""
Module containing helper functions for fetching, processing, and plotting financial data.
"""

import os
import subprocess
from alpaca_trade_api.rest import REST
import pandas as pd

def fetch_data(symbol, timeframe, start_date, end_date, ndays, sample_rate, num_samples):
    """
    Fetches financial data for the given symbol and timeframe.
    """
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    api = REST(api_key, api_secret)
    data_frame = api.get_bars(
        symbol,
        timeframe,
        start_date.isoformat(),
        end_date.isoformat(),
        adjustment="split",
    ).df
    data = pd.DataFrame(data_frame["close"])
    data.index = pd.to_datetime(data.index)

    # for testing specific dates
    # current_price = data["close"].iloc[-1]

    # comment this section out for testing specific dates
    current_price = api.get_latest_trade(symbol).price
    last_date_in_data = data.index[-1].date()
    end_date_tz = pd.Timestamp(end_date).tz_localize(data.index.tz)
    if last_date_in_data == end_date_tz.date():
        if current_price != data.iloc[-1, 0]:
            data.iloc[-1, 0] = current_price
    else:
        data = data.append(pd.DataFrame({"close": current_price}, index=[end_date_tz]))

    data["DateTime"] = data.index
    data.reset_index(drop=True, inplace=True)
    if sample_rate == 'Minute':
        data = data[-num_samples:]
    else:
        data = data[-ndays:]
    return current_price, data

def compute_actions(symbol, data, end_date, timeframe):
    """
    Computes and prints the most recent action (Buy/Sell) for the given data.
    """
    buy_actions = data[data["Action"] == "Buy"]
    sell_actions = data[data["Action"] == "Sell"]
    last_buy_date = buy_actions.index[-1] if not buy_actions.empty else None
    last_sell_date = sell_actions.index[-1] if not sell_actions.empty else None
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
        rows_from_end = len(data) - data.index.get_loc(last_action_date) - 1
        last_price = data["close"].iloc[-1]
        percent_change = (last_price - last_action_price) / last_action_price * 100.0
        if timeframe == "Minute":
            print(
                f'{symbol:5s} last action was {last_action:4s} on '
                f'{last_action_date.strftime("%Y-%m-%d:%H:%M")} '
                f'({rows_from_end:5d} samples ago) at a '
                f'price of {last_action_price:8.3f} last price {last_price:8.3f} '
                f'percent change {percent_change:9.3f}'
            )
        else:
            df_with_row_number = data.reset_index()
            print(
                f'{symbol:5s} last action was {last_action:4s} on '
                f'{last_action_date.strftime("%Y-%m-%d")} '
                f'({rows_from_end:4d} trading-days ago) at a '
                f'price of {last_action_price:8.3f} last price {last_price:8.3f} '
                f'percent change {percent_change:9.3f}'
            )
    else:
        print("No Buy or Sell actions were recorded.")

def get_company_name(symbol_namespace):
    """Get company name by symbol."""
    symbol = symbol_namespace.symbol.upper()
    file_path = "tickers.txt"
    command = f"awk -F '|' '$1 == \"{symbol}\" {{print $2}}' {file_path}"
    result = subprocess.run(
        command, stdout=subprocess.PIPE, shell=True, text=True, check=True
    )
    return result.stdout.strip() or ""

def plot_close_price(data, symbol, ax1, color_dict, timeframe):
    """Plot close price."""

    x_values = (
        range(len(data))
        if timeframe == "Minute"
        else data.index.to_numpy()
    )
    y_values = data["close"].to_numpy()

    ax1.plot(x_values, y_values)

    ax1.set_title(
        f'Close price for {symbol} from {data.index.min().strftime("%Y-%m-%d")} '
        f'to {data.index.max().strftime("%Y-%m-%d")}, last price: {data["close"].iloc[-1]}'
    )

    y_values_filtered = data[
        "close_detrend_norm_filt_adj"
    ].to_numpy()
    ax1.plot(
        x_values,
        y_values_filtered,
        color="black",
        label="Filtered Close Price",
    )

    previous_row = data.iloc[0]
    if timeframe == "Minute":
        for i, row in enumerate(data.iloc[1:].iterrows()):
            index, row_data = row
            segment_color = color_dict.get(row_data["Color"], "black")
            ax1.plot(
                [i, i + 1],
                [
                    previous_row["close_detrend_norm_filt_adj"],
                    row_data["close_detrend_norm_filt_adj"],
                ],
                color=segment_color,
                alpha=0.075,
                linewidth=7.0,
            )
            previous_row = row_data
    else:
        for i, (index, row_data) in enumerate(data.iloc[1:].iterrows()):
            segment_color = color_dict.get(row_data["Color"], "black")
            ax1.plot(
                [data.index[i], data.index[i + 1]],
                [
                    previous_row["close_detrend_norm_filt_adj"],
                    row_data["close_detrend_norm_filt_adj"],
                ],
                color=segment_color,
                alpha=0.15,
                linewidth=7.0,
            )
            previous_row = row_data

    if timeframe == "Minute":
        ax1.set_xticks(x_values[:: len(data) // 10])
        ax1.set_xticklabels(
            data.index[:: len(data) // 10].strftime("%Y-%m-%d %H:%M"),
            rotation=45,
        )

    ax1.set_ylabel("Price")
    ax1.grid(color="lightgrey")
    ax1.legend()

def plot_detrended_data(data, symbol, ax2, timeframe):
    """Plot detrended data."""

    x_values = (
        range(len(data))
        if timeframe == "Minute"
        else data.index.to_numpy()
    )
    y_values_detrend_norm = data["close_detrend_norm"].to_numpy()
    y_values_detrend_norm_filt = data[
        "close_detrend_norm_filt"
    ].to_numpy()

    company_name = get_company_name(symbol)
    title = (
        f"Detrended and Normalized Close Price for {company_name}"
        if company_name
        else "Detrended and Normalized Close Price"
    )

    ax2.plot(
        x_values, y_values_detrend_norm, label="Detrended and Normalized"
    )
    ax2.plot(
        x_values, y_values_detrend_norm_filt, label="Low-pass Filtered"
    )

    if timeframe == "Minute":
        ax2.set_xticks(x_values[:: len(data) // 10])
        ax2.set_xticklabels(
            data.index[:: len(data) // 10].strftime("%Y-%m-%d"),
            rotation=45,
        )

    ax2.set_title(title)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Normalized and Filtered Price")
    ax2.grid(color="lightgrey")
    ax2.legend()

def plot_z_score_velocity(data, args, ax3):
    """Plot z-score velocity."""

    x_values = range(len(data)) if args.timeframe == 'Minute' else data.index.to_numpy()

    y_values = data["zscore_velocity"].to_numpy()

    ax3.plot(x_values, y_values, color="blue")
    ax3.axhline(0, color="black", linewidth=1)
    ax3.axhline(args.std_dev, color="black", linewidth=1, linestyle="--")
    ax3.axhline(-args.std_dev, color="black", linewidth=1, linestyle="--")
    ax3.fill_between(
        x_values,
        y_values,
        args.std_dev,
        where=y_values > args.std_dev,
        facecolor="red",
        alpha=0.3,
    )
    ax3.fill_between(
        x_values,
        y_values,
        -args.std_dev,
        where=y_values < -args.std_dev,
        facecolor="green",
        alpha=0.3,
    )

    if args.timeframe == 'Minute':
        ax3.set_xticks(x_values[::len(data)//10])
        ax3.set_xticklabels(data.index[::len(data)//10].strftime("%Y-%m-%d %H:%M"), rotation=45)

    ax3.set_xlabel("Date")
    ax3.set_ylabel("Filtered Velocity (Z-score)")
    ax3.set_title("Z-score of Filtered Velocity")
    ax3.grid(color="lightgrey")

def plot_buy_sell_markers(data, ax1, ax2, timeframe):
    """Plot buy/sell markers."""
    marker_size = 100
    buy_actions = data[data["Action"] == "Buy"]
    sell_actions = data[data["Action"] == "Sell"]

    x_values_buy = (
        [list(data.index).index(idx) for idx in buy_actions.index]
        if timeframe == "Minute"
        else buy_actions.index.to_numpy()
    )
    x_values_sell = (
        [list(data.index).index(idx) for idx in sell_actions.index]
        if timeframe == "Minute"
        else sell_actions.index.to_numpy()
    )

    ax2.scatter(
        x_values_buy,
        buy_actions["close_detrend_norm"],
        color="green",
        marker="^",
        label="Buy",
        s=marker_size,
    )
    ax2.scatter(
        x_values_sell,
        sell_actions["close_detrend_norm"],
        color="red",
        marker="v",
        label="Sell",
        s=marker_size,
    )
    y_min, y_max = ax1.get_ylim()
    offset = (y_max - y_min) * 0.05
    for buy_x, buy_data in zip(x_values_buy, buy_actions.itertuples()):
        ax1.scatter(
            buy_x,
            buy_data.close,
            color="green",
            marker="^",
            label="Buy",
            s=marker_size,
        )
        ax1.text(
            buy_x,
            buy_data.close - offset,
            f"Buy: {buy_data.close:.2f}",
            color="green",
            verticalalignment="top",
            horizontalalignment="center",
        )
    for sell_x, sell_data in zip(
        x_values_sell, sell_actions.itertuples()
    ):
        ax1.scatter(
            sell_x,
            sell_data.close,
            color="red",
            marker="v",
            label="Sell",
            s=marker_size,
        )
        ax1.text(
            sell_x,
            sell_data.close + offset,
            f"Sell: {sell_data.close:.2f}",
            color="red",
            verticalalignment="bottom",
            horizontalalignment="center",
        )
