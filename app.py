import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Strategy Dashboard", layout="wide")

st.title(" Stock Strategy & Backtest Dashboard")


STOCKS = {
"Reliance Industries": "RELIANCE.NS",
"TCS": "TCS.NS",
"Infosys": "INFY.NS",
"HDFC Bank": "HDFCBANK.NS",
"ICICI Bank": "ICICIBANK.NS",
"Adani Gas": "ATGL.NS",
"Adani Enterprises": "ADANIENT.NS",
"SBI": "SBIN.NS",
"ITC": "ITC.NS",
"Wipro": "WIPRO.NS",
"Bharti Airtel": "BHARTIARTL.NS",
"HUL": "HINDUNILVR.NS",
"Asian Paints": "ASIANPAINT.NS",
"Maruti": "MARUTI.NS",
"Titan": "TITAN.NS",
"Bajaj Finance": "BAJFINANCE.NS",
"HCL Tech": "HCLTECH.NS",
"Adani Ports": "ADANIPORTS.NS",
"Coal India": "COALINDIA.NS",
"L&T": "LT.NS",}


st.sidebar.header("Stock Selector")

selected_name = st.sidebar.selectbox(
    "Choose Stock",
    list(STOCKS.keys())
)

ticker = STOCKS[selected_name]

manual = st.sidebar.text_input(
    "Or Type Symbol Manually",
    value=ticker
)

if manual:
    ticker = manual

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))


@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data found for this symbol")
    st.stop()

st.subheader(f"Data for: {selected_name} ({ticker})")
st.dataframe(df.tail())


df["EMA50"] = df["Close"].ewm(span=50).mean()
df["EMA200"] = df["Close"].ewm(span=200).mean()

df["Signal"] = 0
df.loc[df["EMA50"] > df["EMA200"], "Signal"] = 1
df.loc[df["EMA50"] < df["EMA200"], "Signal"] = -1

df["Returns"] = df["Close"].pct_change()
df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]

df.dropna(inplace=True)

cum_market = (1 + df["Returns"]).cumprod()
cum_strategy = (1 + df["Strategy_Returns"]).cumprod()

total_return = round((cum_strategy.iloc[-1] - 1) * 100, 2)

win_rate = round(
len(df[df["Strategy_Returns"] > 0]) /
len(df[df["Strategy_Returns"] != 0]) * 100,
2
)

max_drawdown = round((cum_strategy / cum_strategy.cummax() - 1).min() * 100, 2)


st.subheader("Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Total Return %", f"{total_return}%")
c2.metric("Win Rate %", f"{win_rate}%")
c3.metric("Max Drawdown %", f"{max_drawdown}%")


st.subheader("Equity Curve")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cum_market, label="Market")
ax.plot(cum_strategy, label="Strategy")
ax.legend()
st.pyplot(fig)

st.subheader("Price + EMA")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df["Close"], label="Close")
ax2.plot(df["EMA50"], label="EMA50")
ax2.plot(df["EMA200"], label="EMA200")
ax2.legend()
st.pyplot(fig2)


st.subheader("Trade Data")
st.dataframe(df.tail(20))

st.download_button(
"Download CSV",
df.to_csv().encode("utf-8"),
"backtest.csv",
"text/csv"
)
