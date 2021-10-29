import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import streamlit as st


def main():

    st.set_page_config(
        page_title="PyStockAnalysis",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://github.com/smaranjitghose/PyFinAnalytics/",
            "Report a bug": "https://github.com/smaranjitghose/PyFinAnalytics/issues",
            "About": "# This app serves is an MVP for interactive dashboards that can be used for financial data analysis instead of Microsoft Excel",
        },
    )

    st.title("PyStockAnalysis")
    st.subheader("Smaranjit Ghose and Siddhant Mahurkar")

    start_date = st.date_input(
        "Start Date",
        min_value=datetime(2000, 1, 1),
        max_value=datetime.today(),
        help="Enter the date from which you want to analyse the stock data",
    )
    end_date = st.date_input(
        "End Date",
        min_value=start_date,
        max_value=datetime.today(),
        help="Enter the date till which you want to analyse the stock data",
    )
    if start_date > end_date:
        st.warning(
            "You seem to have selected a start date greater than end date. Please reselect the dates"
        )
    ticker = st.text_input(
        "Ticker",
        help="Enter the symbol used to uniquely identify publicly traded shares of your company of interest on the stock market",
    )
    if st.button("Get Market Data"):
        try:
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            st.balloons()
            # Display the stock data
            st.dataframe(stock_data)
            # Basic closing price analysis
            basic_closing_price_analysis(stock_data, ticker)
            # Compute Daily Change Percentage
            st.markdown(" ")
            st.markdown("### Daily Change Percentage")
            stock_data = get_daiy_change(stock_data)
            st.dataframe(
                stock_data.style.set_na_rep("Not Available").highlight_null(
                    null_color="orange"
                )
            )
            # Plot Daily Change Percentage
            plot_daily_change(stock_data, ticker)
            # Compute Log Returns
            st.markdown("#### Log Returns")
            stock_data = get_log_returns(stock_data)
            st.dataframe(stock_data)
            # Compute Beta
            st.markdown("#### Beta")
            beta = get_beta(ticker, start_date, end_date)
            st.write(f"Beta for {ticker} is {round(beta,4)}")
            # Compute Volatility
            st.markdown("#### Volatility")
            volatility = get_volatility(stock_data)
            st.write(f"Volatility for {ticker} is {round(volatility,4)}")
            # Compute 10 Day Moving Average
            st.markdown(" ")
            st.markdown("#### 10 Day Moving Average:")
            stock_data = get_10_day_ma(stock_data)
            st.dataframe(stock_data)
            # Plot 10 Day Moving Average
            st.markdown(" ")
            st.markdown("#### 10 Day Moving Average Plot:")
            plot_10_day_ma(stock_data, ticker)
            # Compute 10 Day Exponential Moving Average
            st.markdown(" ")
            st.markdown("#### 10 Day Exponential Moving Average:")
            stock_data = get_10_day_ema(stock_data)
            st.dataframe(stock_data)
            # Plot 10 Day Exponential Moving Average
            st.markdown(" ")
            st.markdown("#### 10 Day Exponential Moving Average Plot:")
            plot_10_day_ema(stock_data, ticker)
            # Trend Analysis using Moving Average
            st.markdown(" ")
            st.markdown("#### Trend Analysis using Moving Averages:")
            trend_analysis_ma(stock_data, ticker)
            # MACD
            st.markdown(" ")
            st.markdown("#### MACD:")
            get_macd(stock_data, ticker)
            # Stochastic Oscillator
            st.markdown(" ")
            st.markdown("#### Stochastic Oscillator:")
            get_stochastic_oscillator(stock_data, ticker)
            # Final Text
            app_closing_notes()
        except Exception as e:
            st.error("Please enter a valid ticker symbol")


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    """
    stock_data = pdr.get_data_yahoo(ticker, start_date, end_date)
    return stock_data


def basic_closing_price_analysis(stock_data, ticker):
    # Display minimum and maximum closing prices in the selected date range
    max_closing_price = stock_data["Close"].max()
    max_date = stock_data[stock_data["Close"] == max_closing_price].index[0]
    st.markdown(f"### Maximum closing price: {round(max_closing_price,4)}")
    st.markdown(f"Maximum Closing Price was observed on {max_date}")
    min_closing_price = stock_data["Close"].min()
    min_date = stock_data[stock_data["Close"] == min_closing_price].index[0]
    st.markdown(f"### Minimum closing price was: {round(min_closing_price,4)}")
    st.markdown(f"Minimum Closing Price was observed on {min_date}")
    # Plot movement of closing price
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(stock_data["Close"])
    ax.set_title(f"{ticker} stock price movement")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)


def get_daiy_change(stock_data):
    stock_data["Daily Change"] = stock_data["Close"].pct_change() * 100
    return stock_data


def plot_daily_change(stock_data, ticker):
    # Plot the daily change percentage
    st.markdown(" ")
    st.markdown("### Daily Change Plot")
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.plot(stock_data["Daily Change"])
    ax.set_title(f"Daily Changes for {ticker}")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)
    # Plot Daily Change Percentage Histogram
    st.markdown(" ")
    st.markdown("#### Daily Change Plot (Histogram)")
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.hist(stock_data["Daily Change"], bins=100)
    ax.set_title(f"Daily Changes for {ticker}")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)


def get_beta(ticker, start_date, end_date):
    tickers = [ticker, "^GSPC"]
    stock_data = pdr.get_data_yahoo(
        tickers, start=start_date, end=end_date, interval="m"
    )
    stock_data = stock_data["Adj Close"]
    log_returns = np.log(stock_data / stock_data.shift())
    cov = log_returns.cov()
    var = log_returns["^GSPC"].var()
    beta = cov.loc[ticker, "^GSPC"] / var
    return beta


def get_log_returns(stock_data):
    stock_data["Log Returns"] = np.log(
        stock_data["Close"] / stock_data["Close"].shift(1)
    )
    return stock_data


def get_volatility(stock_data):
    return stock_data["Log Returns"].std() * np.sqrt(252)


def get_10_day_ma(stock_data):
    stock_data["MA10"] = stock_data["Close"].rolling(10).mean()
    return stock_data


def plot_10_day_ma(stock_data, ticker):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(stock_data["MA10"])
    ax.set_title(f"10 Day Moving Average for {ticker}")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)


def get_10_day_ema(stock_data):
    stock_data["EMA10"] = stock_data["Close"].ewm(span=10, adjust=False).mean()
    return stock_data


def plot_10_day_ema(stock_data, ticker):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(stock_data["EMA10"])
    ax.set_title(f"10 Day Exponential Moving Average for {ticker}")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)


def trend_analysis_ma(stock_data, ticker):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(stock_data["MA10"], label="MA10")
    ax.plot(stock_data["EMA10"], label="EMA10")
    ax.plot(stock_data["Close"], label="Close")
    ax.legend()
    ax.set_title(f"Trend Analysis using Moving Average for {ticker}")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)


def get_macd(stock_data, ticker):
    exp1 = stock_data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = stock_data["Close"].ewm(span=26, adjust=False).mean()
    stock_data["MACD"] = exp1 - exp2
    stock_data["Signal Line"] = stock_data["MACD"].ewm(span=9, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(stock_data["MACD"], label="MACD")
    ax.plot(stock_data["Signal Line"], label="Signal Line")
    ax.legend()
    ax.set_title(f"MACD and Signal Line for {ticker}")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)


def get_stochastic_oscillator(stock_data, ticker):
    high14 = stock_data["High"].rolling(14).max()
    low14 = stock_data["Low"].rolling(14).min()
    stock_data["%K"] = (stock_data["Close"] - low14) * 100 / (high14 - low14)
    stock_data["%D"] = stock_data["%K"].rolling(3).mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(stock_data["%K"], label="%K")
    ax.plot(stock_data["%D"], label="%D")
    ax.axhline(80, c="purple", alpha=0.5)
    ax.axhline(20, c="green", alpha=0.5)
    ax.legend()
    ax.set_title(f"Stochastic Oscillator for {ticker}")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)


def app_closing_notes():
    st.write("Made with 💖 in Python for all budding financial analysts")


if __name__ == "__main__":
    main()

