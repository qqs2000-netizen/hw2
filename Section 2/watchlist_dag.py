import os
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

WATCHLIST = ["NVDA", "TSLA", "BTC-USD"]
ALERT_THRESHOLD = 3.0  # daily swing in %
N_DAYS = 30

DATA_DIR = os.path.expanduser("~/airflow/data")
SUMMARY_CSV = os.path.join(DATA_DIR, "watchlist_summary.csv")


def fetch_data():
    """Step 1: download the last 30 trading days for each ticker."""
    import yfinance as yf
    os.makedirs(DATA_DIR, exist_ok=True)
    for ticker in WATCHLIST:
        df = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df = df[["Open", "High", "Low", "Close", "Volume"]].tail(N_DAYS)
        df.index = df.index.tz_localize(None).date
        df.index.name = "Date"
        df.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))
        print(f"{ticker}: {len(df)} rows, {df.index.min()} to {df.index.max()}")


def analyze_and_summarize():
    """Step 2: 30-day return and average daily swing, saved to watchlist_summary.csv."""
    rows = []
    for ticker in WATCHLIST:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=["Date"])
        df["Swing_%"] = (df["High"] - df["Low"]) / df["Open"] * 100
        first_close, last_close = df["Close"].iloc[0], df["Close"].iloc[-1]
        rows.append({
            "Ticker": ticker,
            "Start_Date": df["Date"].iloc[0].date(),
            "End_Date": df["Date"].iloc[-1].date(),
            "Initial_Close": round(first_close, 2),
            "Latest_Close": round(last_close, 2),
            "Return_30d_%": round((last_close - first_close) / first_close * 100, 2),
            "Avg_Daily_Swing_%": round(df["Swing_%"].mean(), 2),
            "Latest_Day_Swing_%": round(df["Swing_%"].iloc[-1], 2),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {SUMMARY_CSV}")
    print(summary.to_string(index=False))


def alert():
    """Step 3: flag tickers whose latest daily swing is above the threshold."""
    summary = pd.read_csv(SUMMARY_CSV)
    hits = summary[summary["Latest_Day_Swing_%"] >= ALERT_THRESHOLD]
    if hits.empty:
        print(f"All assets traded within normal volatility ranges (threshold {ALERT_THRESHOLD}%).")
    for _, r in hits.iterrows():
        print(f"ALERT: {r['Ticker']} experienced high volatility "
              f"({r['Latest_Day_Swing_%']:.1f}% swing) on {r['End_Date']}")


default_args = {
    "owner": "qqs2000",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "watchlist_volatility",
    default_args=default_args,
    description="HW2 Q2.2 stock watchlist volatility pipeline",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["hw2"],
) as dag:
    step1 = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    step2 = PythonOperator(task_id="analyze_and_summarize", python_callable=analyze_and_summarize)
    step3 = PythonOperator(task_id="alert", python_callable=alert)

    step1 >> step2 >> step3
