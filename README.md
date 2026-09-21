# Market Strategy Lab

A Streamlit backtesting app that compares rules-based strategies with investing the same starting amount in SPY.

## Strategies

1. **Sector momentum** — ranks the nine original Select Sector SPDR ETFs and holds the strongest group, equal-weighted.
2. **Point-in-time S&P 500 momentum** — ranks only the securities that belonged to the S&P 500 at that month-end and holds the strongest group, equal-weighted.
3. **SPY moving-average trend** — holds SPY when it is above its moving average and SHY otherwise.

Momentum defaults to the return from 12 months ago through one month ago. Signals are delayed before the next return is applied. The app supports transaction costs and displays growth, drawdowns, performance metrics, calendar returns, rolling outperformance, ending holdings, and the exact holdings for every month. The full monthly holdings history can be downloaded as a CSV.

## Deploy on Streamlit Community Cloud

Upload these files to a new GitHub repository:

```text
app.py
strategy_engine.py
sp500_membership_intervals.csv
requirements.txt
README.md
THIRD_PARTY_NOTICES.md
```

At Streamlit Community Cloud, create an app using:

```text
Main file path: app.py
```

No API key is required. Prices are downloaded from Yahoo Finance through `yfinance` and cached for six hours. Historical S&P membership is bundled with the app and verified through August 18, 2026; the app refuses later end dates until that file is updated.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Interpretation limits

- Stock eligibility is point-in-time: a security can be ranked only while it was an S&P 500 member. This removes the original future-constituent error, such as allowing Tesla to compete in 2012.
- Yahoo Finance is not a survivorship-free institutional database. It may omit a discontinued historical symbol even when the membership file contains it. Such a security cannot be ranked when Yahoo supplies no price history. For publication-grade research, replace Yahoo with a vendor that includes delisted securities and delisting returns.
- Yahoo adjusted prices account for splits and distributions, but the test does not model bid/ask spreads, market impact, taxes, account restrictions, or exact execution prices.
- The trading-cost input is applied to one-way turnover. Selling one complete portfolio and buying another generates approximately 200% turnover.
- The Sharpe ratio uses a zero-percent risk-free rate.
- Past performance does not predict future performance.
