"""Module for fetching and plotting stock market data."""

import os
import subprocess
import pandas as pd
from alpaca_trade_api.rest import REST


class DataConfig:
    """Class to hold data configuration."""

    def __init__(self, symbol, timeframe, start_date, end_date, ndays):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.ndays = ndays


class DataFetcher:
    """Class for fetching stock market data."""

    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")
        self.api = REST(self.api_key, self.api_secret)

    def fetch_data(self):
        """Fetch stock market data."""
        data_frame = self.api.get_bars(
            self.config.symbol,
            self.config.timeframe,
            self.config.start_date.isoformat(),
            self.config.end_date.isoformat(),
            adjustment="split",
        ).df
        data = pd.DataFrame(data_frame["close"])
        data.index = pd.to_datetime(data.index)
        current_price = self.api.get_latest_trade(self.config.symbol).price
        last_date_in_data = data.index[-1].date()
        end_date_tz = pd.Timestamp(self.config.end_date).tz_localize(
            data.index.tz
        )
        if last_date_in_data == end_date_tz.date():
            if current_price != data.iloc[-1, 0]:
                data.iloc[-1, 0] = current_price
        else:
            data = data.append(
                pd.DataFrame({"close": current_price}, index=[end_date_tz])
            )
        data["DateTime"] = data.index
        data.reset_index(drop=True, inplace=True)
        return current_price, data


class ActionComputer:
    """Class to compute buy/sell actions."""

    @staticmethod
    def compute_actions(symbol, data, end_date):
        """Compute buy/sell actions."""
        buy_actions = data[data["Action"] == "Buy"]
        sell_actions = data[data["Action"] == "Sell"]
        last_buy_date = (
            buy_actions.index[-1] if not buy_actions.empty else None
        )
        last_sell_date = (
            sell_actions.index[-1] if not sell_actions.empty else None
        )
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
            rows_from_end = (
                len(data) - data.index.get_loc(last_action_date) - 1
            )
            days_ago = (end_date - last_action_date.date()).days
            last_price = data["c"].iloc[-1]
            percent_change = ( ( last_action_price - last_price ) / last_action_price ) * 100.0
            print(
                f'{symbol:5s} last action was {last_action:4s} on {last_action_date.strftime("%Y-%m-%d")} ({days_ago:4d} days ago, or {rows_from_end:4d} trading-days ago) at a price of {last_action_price:8.3f} last price {last_price:8.3f} percent change {percent_change:9.3f}'
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


class DataPlotter:
    """Class to plot stock market data."""

    @staticmethod
    def plot_close_price(data, symbol, ax1, color_dict):
        """Plot close price."""
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

    @staticmethod
    def plot_detrended_data(data, symbol, ax2):
        """Plot detrended data."""
        company_name = get_company_name(symbol)
        title = (
            f"Detrended and Normalized Close Price for {company_name}"
            if company_name
            else "Detrended and Normalized Close Price"
        )
        data["close_detrend_norm"].plot(
            ax=ax2,
            grid=True,
            title=title,
            label="Detrended and Normalized",
        )
        data["close_detrend_norm_filt"].plot(
            ax=ax2, grid=True, label="Low-pass Filtered"
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Normalized and Filtered Price")
        ax2.grid(color="lightgrey")
        ax2.legend()

    @staticmethod
    def plot_z_score_velocity(data, args, ax3):
        """Plot z-score velocity."""
        ax3.plot(data.index, data["zscore_velocity"], color="blue")
        ax3.axhline(0, color="black", linewidth=1)
        ax3.axhline(args.std_dev, color="black", linewidth=1, linestyle="--")
        ax3.axhline(-args.std_dev, color="black", linewidth=1, linestyle="--")
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

    @staticmethod
    def plot_buy_sell_markers(data, ax1, ax2):
        """Plot buy/sell markers."""
        marker_size = 100
        buy_actions = data[data["Action"] == "Buy"]
        sell_actions = data[data["Action"] == "Sell"]
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
        y_min, y_max = ax1.get_ylim()
        offset = (y_max - y_min) * 0.05
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
                buy_data["close"] - offset,
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
                sell_data["close"] + offset,
                f'Sell: {sell_data["close"]:.2f}',
                color="red",
                verticalalignment="bottom",
                horizontalalignment="center",
            )
