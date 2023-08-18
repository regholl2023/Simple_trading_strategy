import pandas as pd

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# no keys required for crypto data
client = CryptoHistoricalDataClient()

request_params = CryptoBarsRequest(
                        symbol_or_symbols=["BTC/USD", "ETH/USD"],
                        timeframe=TimeFrame.Day,
                        start=datetime(2022, 7, 1),
                        end=datetime(2023, 8, 18)
                 )

df = client.get_crypto_bars(request_params).df

print (df)

# Filter rows with 'BTC/USD' symbol
btc_data = df.loc['BTC/USD']

# Extract 'close' prices
btc_close_prices = btc_data['close']

print ( btc_close_prices )
