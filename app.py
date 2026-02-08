import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
STOCK_LIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ADANIGAS.NS",
    "ADANIENT.NS",
    "ITC.NS"
]


st.set_page_config(page_title="Stock Strategy Backtest", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("Trading Strategy Controls")

ticker = st.sidebar.selectbox(
    "Select Stock (NSE)",
    options=STOCK_LIST,
    index=0
)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

short_window = st.sidebar.slider("Short MA", 10, 50, 20)
long_window = st.sidebar.slider("Long MA", 50, 200, 50)

# ---------------- Data ----------------
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data found")
    st.stop()

# ---------------- Strategy ----------------
df["MA_Short"] = df["Close"].rolling(short_window).mean()
df["MA_Long"] = df["Close"].rolling(long_window).mean()

df["Signal"] = 0
df.loc[df["MA_Short"] > df["MA_Long"], "Signal"] = 1

df["Position"] = df["Signal"].diff()

# ---------------- Backtest ----------------
df["Daily Return"] = df["Close"].pct_change()
df["Strategy Return"] = df["Daily Return"] * df["Signal"].shift(1)

df["Cumulative Market Return"] = (1 + df["Daily Return"]).cumprod()
df["Cumulative Strategy Return"] = (1 + df["Strategy Return"]).cumprod()

# Metrics
total_return = df["Cumulative Strategy Return"].iloc[-1] - 1
volatility = df["Strategy Return"].std() * np.sqrt(252)
sharpe = (df["Strategy Return"].mean() / df["Strategy Return"].std()) * np.sqrt(252)

# ---------------- UI ----------------
st.title(f"{ticker} Trading Strategy & Backtest")

col1, col2, col3 = st.columns(3)
col1.metric("Total Strategy Return", f"{total_return:.2%}")
col2.metric("Annual Volatility", f"{volatility:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# ---------------- Price Chart ----------------
st.subheader("Price & Moving Averages")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df.index, df["Close"], label="Close Price")
ax.plot(df.index, df["MA_Short"], label=f"MA {short_window}")
ax.plot(df.index, df["MA_Long"], label=f"MA {long_window}")
ax.legend()
st.pyplot(fig)

# ---------------- Equity Curve ----------------
st.subheader("Cumulative Returns")

fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.plot(df.index, df["Cumulative Market Return"], label="Buy & Hold")
ax2.plot(df.index, df["Cumulative Strategy Return"], label="Strategy")
ax2.legend()
st.pyplot(fig2)

# ---------------- Data Preview ----------------
st.subheader("Data Preview")
st.dataframe(df.tail())

