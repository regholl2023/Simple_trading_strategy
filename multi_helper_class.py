import os
import requests
import subprocess
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from datetime import date, datetime
from dateutil.tz import tzlocal
from alpaca_trade_api.rest import REST


class DataFetcher:
    def __init__(self, symbols, timeframe, ndays, num_samples, sample_rate):
        self.symbols = symbols
        self.timeframe = timeframe
        self.ndays = ndays
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")
        self.api = REST(self.api_key, self.api_secret)
        self.ns = num_samples
        self.sample_rate = sample_rate

    def calculate_start_date(self):
        nyse = mcal.get_calendar("NYSE")
        end_date = pd.Timestamp.now().normalize()
        start_date = pd.Timestamp("2000-01-01")
        trading_days = nyse.valid_days(
            start_date=start_date, end_date=end_date
        )

        if len(trading_days) < self.ndays:
            raise ValueError(
                "The number of trading days requested is more than the available trading days."
            )

        start_date = trading_days[-self.ndays]
        return start_date

    def fetch_historical_data_v2(self):
        """Fetch historical data using Alpaca's multi-bar API v2 and handle pagination."""

        # Join symbols into a comma-separated string
        symbol_str = ",".join(self.symbols)

        # Set the base URL for the Alpaca API
        base_url = "https://data.alpaca.markets/v2/stocks"

        # Initialize an empty DataFrame to store the results
        data = pd.DataFrame()

        # Initialize the page_token
        page_token = None

        # Calculate the start and end date
        start_date = self.calculate_start_date().strftime("%Y-%m-%d")
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        while True:
            # Build the query parameters
            params = {
                "start": start_date,
                "end": end_date,
                "timeframe": self.timeframe,
                "limit": 10000,
                "adjustment": "split",
                "symbols": symbol_str,
                "feed": "sip",
            }
            if page_token is not None:
                params["page_token"] = page_token

            # Send the GET request to the Alpaca API
            url = f"{base_url}/bars"
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            }
            response = requests.get(url, headers=headers, params=params)

            # Raise an exception if the request was unsuccessful
            response.raise_for_status()

            # Load the response data into a JSON object
            response_json = response.json()

            # Append the data for each symbol to the DataFrame
            for symbol, bars in list(
                response_json["bars"].items()
            ):  # Create a list copy for iteration
                df = pd.DataFrame(bars)
                df = df[["c", "t"]]
                df["symbol"] = symbol
                current_price = self.api.get_latest_trade(symbol).price

                last_date_in_data_str = df["t"].iloc[-1]
                last_date_in_data = datetime.strptime(
                    last_date_in_data_str, "%Y-%m-%dT%H:%M:%SZ"
                )
                last_date = last_date_in_data.strftime("%Y-%m-%d")

                if last_date == end_date:
                    if current_price != df["c"].iloc[-1]:
                        df.loc[df.index[-1], "c"] = current_price
                else:
                    today = pd.Timestamp.today()
                    new_row = pd.DataFrame(
                        {
                            "c": [current_price],
                            "symbol": [symbol],
                            "t": [today],
                        }
                    )
                    df = df.append(new_row, ignore_index=True)

                if self.sample_rate == 'Day':
                    df = df[-self.ndays:]
                else:
                    df = df[-self.ns:]
                data = data.append(df)

            # If there's a next_page_token, update the page_token and continue the loop
            page_token = response_json.get("next_page_token")
            if page_token is None:
                break

        return data, end_date


class ActionComputer:
    @staticmethod
    def compute_actions(symbol, data, end_date, timeframe):
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
                last_action_price = buy_actions.loc[last_buy_date, "c"]
            else:
                last_action = "Sell"
                last_action_date = last_sell_date
                last_action_price = sell_actions.loc[last_sell_date, "c"]
        elif last_buy_date:
            last_action = "Buy"
            last_action_date = last_buy_date
            last_action_price = buy_actions.loc[last_buy_date, "c"]
        elif last_sell_date:
            last_action = "Sell"
            last_action_date = last_sell_date
            last_action_price = sell_actions.loc[last_sell_date, "c"]
        else:
            last_action = None

        if last_action:
            last_price = data["c"].iloc[-1]
            percent_change = (last_price - last_action_price) / last_action_price * 100.0
            if timeframe == 'Day':
                rows_from_end = (
                    len(data) - data.index.get_loc(last_action_date) - 1
                )
                if isinstance(last_action_date, str):
                    date_object = datetime.strptime(
                        last_action_date, "%Y-%m-%dT%H:%M:%SZ"
                    )
                else:
                    date_object = last_action_date.to_pydatetime()

                date_string = date_object.strftime("%Y-%m-%d")
                end_date_object = datetime.strptime(end_date, "%Y-%m-%d")
                date_object = datetime.strptime(date_string, "%Y-%m-%d")

                df_with_row_number = data.reset_index()
                print(
                    f'{symbol:5s} last action was {last_action:4s} on '
                    f'{last_action_date.strftime("%Y-%m-%d")} '
                    f'({rows_from_end:4d} trading-days ago) at a '
                    f'price of {last_action_price:8.3f} last price {last_price:8.3f} '
                    f'percent change {percent_change:9.3f}'
                )
            else:
                rows_from_end = len(data) - data.index.get_loc(last_action_date) - 1
                print(
                    f'{symbol:5s} last action was {last_action:4s} on '
                    f'{last_action_date.strftime("%Y-%m-%d:%H:%M")} '
                    f'({rows_from_end:5d} samples ago) at a '
                    f'price of {last_action_price:8.3f} last price {last_price:8.3f} '
                    f'percent change {percent_change:9.3f}'
                )
        else:
            print("No Buy or Sell actions were recorded.")
