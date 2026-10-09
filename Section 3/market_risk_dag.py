import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD", "ETH-USD"]
BENCHMARK = "SPY"
OUT_DIR = os.path.expanduser("~/airflow/data/mini")
PRICES_CSV = os.path.join(OUT_DIR, "prices.csv")


def fetch_prices():
    """Download one year of daily closes and keep the days on which every asset has a price."""
    import yfinance as yf
    os.makedirs(OUT_DIR, exist_ok=True)
    closes = {}
    for t in TICKERS:
        df = yf.Ticker(t).history(period="1y", interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {t}")
        s = df["Close"]
        s.index = s.index.tz_localize(None).normalize()
        closes[t] = s
        print(f"{t}: {len(s)} rows")
    prices = pd.DataFrame(closes).dropna()  # crypto trades on weekends, stocks do not
    prices.index.name = "Date"
    prices.to_csv(PRICES_CSV)
    print(f"Aligned price table: {prices.shape[0]} days x {prices.shape[1]} assets")


def compute_signals():
    """Trend and momentum per asset: 20/50-day moving average crossover and 14-day RSI."""
    prices = pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True)
    rows = []
    for t in TICKERS:
        p = prices[t]
        sma20, sma50 = p.rolling(20).mean(), p.rolling(50).mean()
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        trend = "Bullish" if sma20.iloc[-1] > sma50.iloc[-1] else "Bearish"
        r = rsi.iloc[-1]
        momentum = "Overbought" if r > 70 else "Oversold" if r < 30 else "Neutral"
        rows.append({"Ticker": t, "Last_Close": round(p.iloc[-1], 2),
                     "SMA20": round(sma20.iloc[-1], 2), "SMA50": round(sma50.iloc[-1], 2),
                     "Trend": trend, "RSI14": round(r, 1), "Momentum": momentum})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "signals.csv"), index=False)
    print(out.to_string(index=False))


def compute_risk():
    """Risk per asset: return, annualized volatility, Sharpe, max drawdown, beta vs SPY; plus correlations."""
    prices = pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True)
    rets = prices.pct_change().dropna()
    bench = rets[BENCHMARK]
    rows = []
    for t in TICKERS:
        r = rets[t]
        wealth = (1 + r).cumprod()
        rows.append({
            "Ticker": t,
            "Return_1y_%": round((prices[t].iloc[-1] / prices[t].iloc[0] - 1) * 100, 2),
            "Ann_Vol_%": round(r.std() * np.sqrt(252) * 100, 2),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2),
            "Max_Drawdown_%": round((wealth / wealth.cummax() - 1).min() * 100, 2),
            "Beta_vs_SPY": round(r.cov(bench) / bench.var(), 2),
        })
    risk = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    risk.to_csv(os.path.join(OUT_DIR, "risk.csv"), index=False)
    rets.corr().round(2).to_csv(os.path.join(OUT_DIR, "correlation.csv"))
    print(risk.to_string(index=False))
    print("\nCorrelation of daily returns:")
    print(rets.corr().round(2).to_string())


def make_charts():
    """Normalized prices, rolling 30-day volatility, and a correlation heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prices = pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True)
    rets = prices.pct_change().dropna()

    ax = (prices / prices.iloc[0] * 100).plot(figsize=(10, 5), title="Growth of 100 (1 year)")
    ax.set_ylabel("Indexed value")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "normalized_prices.png"), dpi=120); plt.close()

    ax = (rets.rolling(30).std() * np.sqrt(252) * 100).plot(figsize=(10, 5),
                                                            title="Rolling 30-day annualized volatility")
    ax.set_ylabel("%")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "rolling_volatility.png"), dpi=120); plt.close()

    corr = rets.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title("Correlation of daily returns")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=120); plt.close()
    print("Saved 3 charts to", OUT_DIR)


def write_report():
    """Combine signals and risk into one report and print the key findings."""
    signals = pd.read_csv(os.path.join(OUT_DIR, "signals.csv"))
    risk = pd.read_csv(os.path.join(OUT_DIR, "risk.csv"))
    corr = pd.read_csv(os.path.join(OUT_DIR, "correlation.csv"), index_col=0)
    report = risk.merge(signals, on="Ticker")
    report.to_csv(os.path.join(OUT_DIR, "market_risk_report.csv"), index=False)

    best = risk.iloc[0]
    riskiest = risk.sort_values("Ann_Vol_%").iloc[-1]
    pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    hi_pair, lo_pair = pairs.idxmax(), pairs.idxmin()
    print("=== MARKET RISK REPORT ===")
    print(report.to_string(index=False))
    print(f"\nBest risk-adjusted asset: {best['Ticker']} (Sharpe {best['Sharpe']})")
    print(f"Most volatile asset: {riskiest['Ticker']} ({riskiest['Ann_Vol_%']}% annualized)")
    print(f"Most correlated pair: {hi_pair[0]} / {hi_pair[1]} ({pairs.max():.2f})")
    print(f"Least correlated pair: {lo_pair[0]} / {lo_pair[1]} ({pairs.min():.2f})")
    for _, r in report.iterrows():
        if r["Momentum"] != "Neutral":
            print(f"SIGNAL: {r['Ticker']} is {r['Momentum'].lower()} (RSI {r['RSI14']})")


default_args = {
    "owner": "qqs2000",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "market_risk_dashboard",
    default_args=default_args,
    description="HW2 mini-project: stock and crypto trend and risk analysis",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["hw2", "mini-project"],
) as dag:
    fetch = PythonOperator(task_id="fetch_prices", python_callable=fetch_prices)
    signals = PythonOperator(task_id="compute_signals", python_callable=compute_signals)
    risk = PythonOperator(task_id="compute_risk", python_callable=compute_risk)
    charts = PythonOperator(task_id="make_charts", python_callable=make_charts)
    report = PythonOperator(task_id="write_report", python_callable=write_report)

    fetch >> [signals, risk, charts]
    [signals, risk] >> report
