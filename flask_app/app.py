import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from io import BytesIO
from scipy import signal
from datetime import date, timedelta
from alpaca_trade_api.rest import TimeFrame
from matplotlib.offsetbox import AnchoredText
from helper_class import DataConfig, DataFetcher, DataPlotter, ActionComputer
from flask import (
    Flask,
    session,
    redirect,
    url_for,
    request,
    send_file,
    render_template,
)

import matplotlib

matplotlib.use("Agg")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_value")

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


def process_stock_data(
    symbol, ndays, window, std_dev, timeframe, num_samples
):
    TIMEFRAME = TimeFrame.Minute if timeframe == "Minute" else TimeFrame.Day
    END_DATE = date.today()
    NYSE = mcal.get_calendar("NYSE")
    START_DATE = NYSE.valid_days(
        start_date=END_DATE - timedelta(days=2 * ndays), end_date=END_DATE
    )[-ndays]
    SYMBOL = symbol.upper()
    DATA_CONFIG = DataConfig(
        SYMBOL, TIMEFRAME, START_DATE, END_DATE, ndays, timeframe, num_samples
    )
    DATA_FETCHER = DataFetcher(DATA_CONFIG)
    CURRENT_PRICE, data = DATA_FETCHER.fetch_data()

    x = np.linspace(0, len(data.index) - 1, len(data.index))
    y = data["close"]
    y_trend = np.linspace(y.iloc[0], y.iloc[-1], len(data.index))
    trend_line = np.poly1d(np.polyfit(x, y_trend, 1))

    data["close_detrend"] = data["close"] - trend_line(x)
    data["close_detrend_norm"] = data["close_detrend"] / max(
        abs(data["close_detrend"])
    )

    if window is None or window == 0:
        if timeframe == "Minute":
            window = round(data.shape[0] // 8.17)
        else:
            window = round(data.shape[0] // 14.40)
        if (window % 2) == 0:
            window += 1

    window_size = window
    data_filter_val, gradient_val, intercept_val = compute_trend_and_filter(
        len(data["close_detrend_norm"]),
        data["close_detrend_norm"],
        window_size,
    )

    data["close_detrend_norm_filt"] = data_filter_val
    filtered_velocity = np.gradient(
        data["close_detrend_norm_filt"], x[1] - x[0]
    )
    filtered_velocity_zscore = (
        filtered_velocity - np.mean(filtered_velocity)
    ) / np.std(filtered_velocity)
    data["zscore_velocity"] = filtered_velocity_zscore
    data["Color"] = np.where(
        data["zscore_velocity"] >= std_dev,
        "Red",
        np.where(data["zscore_velocity"] <= -std_dev, "Green", ""),
    )
    data["Action"] = ""

    LAST_RED = None
    LAST_GREEN = None

    first = int(data.index[0] + 1)
    last = int(data.index[-1])

    for i in range(first, last):
        if data.loc[i - 1, "Color"] == "Red":
            LAST_RED = i - 1
        elif data.loc[i - 1, "Color"] == "Green":
            LAST_GREEN = i - 1

        if (
            data.loc[i - 1, "zscore_velocity"] > 0
            and data.loc[i, "zscore_velocity"] < 0
            and LAST_RED is not None
        ):
            data.loc[i, "Action"] = "Sell"
            LAST_RED = None

        if (
            data.loc[i - 1, "zscore_velocity"] < 0
            and data.loc[i, "zscore_velocity"] > 0
            and LAST_GREEN is not None
        ):
            data.loc[i, "Action"] = "Buy"
            LAST_GREEN = None

    data.set_index("DateTime", inplace=True)
    message = ActionComputer.compute_actions(
        SYMBOL, data, END_DATE, timeframe
    )
    data["close_detrend_norm_filt_adj"] = data[
        "close_detrend_norm_filt"
    ] * max(abs(data["close_detrend"])) + trend_line(x)

    return window_size, data, message


@app.route("/process_stock_data")
def process_and_visualize_stock_data(
    symbol, ndays, window, std_dev, timeframe, num_samples, data, message
):
    fig, (ax1, ax2, ax3) = create_plot(
        data, symbol, window, std_dev, timeframe
    )

    img_stream = BytesIO()
    fig.savefig(img_stream, format="png")
    img_stream.seek(0)
    img_data = base64.b64encode(img_stream.read()).decode()

    return render_template(
        "show_image.html",
        img_data=img_data,
        message=message,
        stock_symbol=symbol,
    )


def create_plot(data, symbol, window, std_dev, timeframe):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    fig.set_size_inches(12, 10)

    info_text = f"Filter window: {window}"
    anchored_text = AnchoredText(
        info_text, loc="lower left", prop=dict(size=10), frameon=False
    )
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax3.add_artist(anchored_text)

    color_dict = {"Red": "green", "Green": "red"}

    DataPlotter.plot_close_price(data, symbol, ax1, color_dict, timeframe)
    DataPlotter.plot_detrended_data(data, symbol, ax2, timeframe)
    DataPlotter.plot_z_score_velocity(data, std_dev, ax3, timeframe)
    DataPlotter.plot_buy_sell_markers(data, ax1, ax2, timeframe)

    plt.tight_layout()

    return fig, (ax1, ax2, ax3)


@app.route("/")
def form():
    # Retrieve session variables and use them as default values in the form
    symbol = session.get("symbol", "")
    ndays = session.get("ndays", 504)
    window = session.get("window", 0)
    std_dev = session.get("std_dev", 0.01)
    timeframe = session.get("timeframe", "Day")
    num_samples = session.get("num_samples", 5000)
    return render_template(
        "index.html",
        symbol=symbol,
        ndays=ndays,
        window=window,
        std_dev=std_dev,
        timeframe=timeframe,
        num_samples=num_samples,
    )


@app.route("/analyze", methods=["POST"])
def analyze_stock():
    # Save form inputs to the session
    session["symbol"] = request.form["symbol"].upper()
    session["ndays"] = int(request.form.get("ndays", 504))
    session["window"] = max(int(request.form.get("window", 0)), 0)
    session["std_dev"] = float(request.form.get("std_dev", 0.01))
    session["timeframe"] = request.form.get("timeframe", "Day")
    session["num_samples"] = int(request.form.get("num_samples", 5000))

    # Use the session variables instead of request.form for calculations
    symbol = session["symbol"]
    ndays = session["ndays"]
    window = session["window"]
    std_dev = session["std_dev"]
    timeframe = session["timeframe"]
    num_samples = session["num_samples"]

    window, data, message = process_stock_data(
        symbol, ndays, window, std_dev, timeframe, num_samples
    )
    return process_and_visualize_stock_data(
        symbol, ndays, window, std_dev, timeframe, num_samples, data, message
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
