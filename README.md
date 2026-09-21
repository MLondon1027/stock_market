# Market Strategy Lab

A Streamlit backtesting app that compares rules-based strategies with investing the same starting amount in SPY.

## Strategies

1. **Sector momentum** — ranks the nine original Select Sector SPDR ETFs and holds the strongest group, equal-weighted.
2. **Large-cap stock momentum** — ranks a fixed liquid large-cap stock universe and holds the strongest group, equal-weighted.
3. **SPY moving-average trend** — holds SPY when it is above its moving average and SHY otherwise.

Momentum defaults to the return from 12 months ago through one month ago. Signals are delayed before the next return is applied. The app supports transaction costs and displays growth, drawdowns, performance metrics, calendar returns, rolling outperformance, and ending holdings.

## Deploy on Streamlit Community Cloud

Upload these files to a new GitHub repository:

```text
app.py
strategy_engine.py
requirements.txt
README.md
```

At Streamlit Community Cloud, create an app using:

```text
Main file path: app.py
```

No API key is required. Prices are downloaded from Yahoo Finance through `yfinance` and cached for six hours.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Interpretation limits

- The stock-momentum universe is a fixed list of current large-cap stocks. Historical results have survivorship bias and are not a point-in-time S&P 500 simulation.
- Yahoo adjusted prices account for splits and distributions, but the test does not model bid/ask spreads, market impact, taxes, account restrictions, or exact execution prices.
- The trading-cost input is applied to one-way turnover. Selling one complete portfolio and buying another generates approximately 200% turnover.
- The Sharpe ratio uses a zero-percent risk-free rate.
- Past performance does not predict future performance.

