from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass
class BacktestResult:
    name: str
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    equity: pd.DataFrame
    drawdowns: pd.DataFrame
    metrics: pd.DataFrame
    annual_returns: pd.DataFrame
    rolling_results: pd.DataFrame
    current_holdings: pd.DataFrame
    holdings_history: pd.DataFrame
    rebalance_count: int
    holdings_frequency: str = "monthly"


def _clean_prices(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    cleaned = prices.copy().sort_index()
    cleaned.index = pd.to_datetime(cleaned.index).tz_localize(None)
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")]
    return cleaned.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def momentum_weights(
    prices: pd.DataFrame,
    top_n: int,
    lookback_months: int = 12,
    skip_months: int = 1,
    eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create month-end equal weights using only information then available.

    The default score is the return from 12 months ago through one month ago.
    Returned weights are dated at formation month-end. The backtest delays them
    by one trading day before applying returns.
    """
    prices = _clean_prices(prices)
    month_end = prices.resample("ME").last()
    numerator = month_end.shift(skip_months)
    denominator = month_end.shift(lookback_months)
    scores = numerator.div(denominator).sub(1)

    if eligibility is not None:
        eligible_at_formation = eligibility.reindex(
            index=month_end.index, columns=month_end.columns, fill_value=False
        ).fillna(False).astype(bool)
        scores = scores.where(eligible_at_formation)

    weights = pd.DataFrame(0.0, index=month_end.index, columns=month_end.columns)
    for date, row in scores.iterrows():
        eligible = row.replace([np.inf, -np.inf], np.nan).dropna()
        if eligible.empty:
            continue
        winners = eligible.nlargest(min(top_n, len(eligible))).index
        weights.loc[date, winners] = 1.0 / len(winners)
    return weights


def annual_momentum_weights(
    prices: pd.DataFrame,
    top_n: int,
    eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create year-end weights from the preceding calendar year's return.

    A weight formed after the final trading-day close is applied beginning with
    the next trading day and remains unchanged until the following year.
    """
    prices = _clean_prices(prices)
    year_end = prices.resample("YE").last()
    scores = year_end.pct_change(fill_method=None)

    if eligibility is not None:
        eligible_at_formation = eligibility.reindex(
            index=year_end.index, columns=year_end.columns, fill_value=False
        ).fillna(False).astype(bool)
        scores = scores.where(eligible_at_formation)

    weights = pd.DataFrame(0.0, index=year_end.index, columns=year_end.columns)
    for formation_date, row in scores.iterrows():
        eligible = row.replace([np.inf, -np.inf], np.nan).dropna()
        if eligible.empty:
            continue
        winners = eligible.nlargest(min(top_n, len(eligible))).index
        weights.loc[formation_date, winners] = 1.0 / len(winners)
    return weights


def point_in_time_membership_mask(
    membership: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Return whether each ticker was an index member on each supplied date.

    Membership intervals use inclusive start dates and exclusive end dates.
    Multiple intervals for the same ticker are supported.
    """
    required = {"ticker", "start_date", "end_date"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(f"Membership data are missing columns: {sorted(missing)}")

    index = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None)
    columns = list(dict.fromkeys(tickers))
    mask = pd.DataFrame(False, index=index, columns=columns, dtype=bool)
    if mask.empty:
        return mask

    rows = membership.copy()
    rows["start_date"] = pd.to_datetime(rows["start_date"], errors="coerce")
    rows["end_date"] = pd.to_datetime(rows["end_date"], errors="coerce")
    rows = rows.dropna(subset=["ticker", "start_date"])

    available = set(columns)
    for row in rows.itertuples(index=False):
        ticker = str(row.ticker)
        if ticker not in available:
            continue
        active = index >= row.start_date
        if pd.notna(row.end_date):
            active &= index < row.end_date
        mask.loc[active, ticker] = True
    return mask


def _daily_weights(monthly_weights: pd.DataFrame, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    # Map calendar month-end signals to the final observed trading date of each month.
    mapped = monthly_weights.copy()
    actual_month_ends = pd.Series(trading_index, index=trading_index).groupby(
        trading_index.to_period("M")
    ).max()
    mapped.index = [actual_month_ends.get(date.to_period("M"), pd.NaT) for date in mapped.index]
    mapped = mapped.loc[~pd.isna(mapped.index)]
    mapped = mapped.loc[~mapped.index.duplicated(keep="last")]
    return mapped.reindex(trading_index).ffill().fillna(0.0).shift(1).fillna(0.0)


def portfolio_returns_from_weights(
    prices: pd.DataFrame,
    formation_weights: pd.DataFrame,
    transaction_cost_bps: float,
) -> tuple[pd.Series, pd.DataFrame, int]:
    prices = _clean_prices(prices)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    weights = _daily_weights(formation_weights, prices.index)
    weights = weights.reindex(columns=prices.columns, fill_value=0.0)

    gross = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    costs = turnover * (transaction_cost_bps / 10_000.0)
    net = (gross - costs).rename("Strategy")
    rebalance_count = int((turnover > 1e-12).sum())
    return net, weights, rebalance_count


def run_momentum_strategy(
    asset_prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    top_n: int,
    lookback_months: int,
    skip_months: int,
    transaction_cost_bps: float,
    eligibility: pd.DataFrame | None = None,
    rebalance_frequency: str = "monthly",
) -> tuple[pd.Series, pd.Series, pd.DataFrame, int]:
    asset_prices = _clean_prices(asset_prices)
    if rebalance_frequency == "annual":
        weights = annual_momentum_weights(asset_prices, top_n, eligibility)
    elif rebalance_frequency == "monthly":
        weights = momentum_weights(
            asset_prices, top_n, lookback_months, skip_months, eligibility
        )
    else:
        raise ValueError("Rebalance frequency must be 'monthly' or 'annual'.")
    strategy, daily_weights, count = portfolio_returns_from_weights(
        asset_prices, weights, transaction_cost_bps
    )
    benchmark = _clean_prices(benchmark_prices).iloc[:, 0].pct_change(fill_method=None).fillna(0.0)
    common = strategy.index.intersection(benchmark.index)
    return strategy.loc[common], benchmark.loc[common].rename("Buy & Hold SPY"), daily_weights.loc[common], count


def run_trend_strategy(
    spy_prices: pd.Series,
    defensive_prices: pd.Series,
    moving_average_days: int,
    transaction_cost_bps: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, int]:
    prices = pd.concat(
        [spy_prices.rename("SPY"), defensive_prices.rename("SHY")], axis=1
    ).sort_index().ffill().dropna()
    prices = _clean_prices(prices)
    returns = prices.pct_change(fill_method=None).fillna(0.0)

    risk_on = prices["SPY"].gt(prices["SPY"].rolling(moving_average_days).mean())
    # A signal observed at today's close can first be traded for tomorrow's return.
    spy_weight = risk_on.astype(float).shift(1).fillna(0.0)
    weights = pd.DataFrame({"SPY": spy_weight, "SHY": 1.0 - spy_weight}, index=prices.index)
    gross = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    strategy = (gross - turnover * transaction_cost_bps / 10_000.0).rename("Strategy")
    benchmark = returns["SPY"].rename("Buy & Hold SPY")
    count = int((turnover > 1e-12).sum())
    return strategy, benchmark, weights, count


def _drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    return wealth.div(wealth.cummax()).sub(1.0)


def _metric_row(returns: pd.Series, starting_capital: float) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {key: np.nan for key in ["Ending Value", "Total Return", "CAGR", "Volatility", "Sharpe", "Max Drawdown", "Worst Year"]}
    wealth = (1.0 + returns).cumprod()
    elapsed_years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    total_return = wealth.iloc[-1] - 1.0
    cagr = wealth.iloc[-1] ** (1.0 / elapsed_years) - 1.0
    volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if returns.std(ddof=1) > 0 else np.nan
    annual = returns.resample("YE").apply(lambda x: (1.0 + x).prod() - 1.0)
    return {
        "Ending Value": starting_capital * wealth.iloc[-1],
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": _drawdown(returns).min(),
        "Worst Year": annual.min() if not annual.empty else np.nan,
    }


def _rolling_outperformance(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    monthly = pd.concat([strategy, benchmark], axis=1).resample("ME").apply(
        lambda x: (1.0 + x).prod() - 1.0
    )
    output: list[dict[str, float | str | int]] = []
    for years in (3, 5):
        months = years * 12
        rolling = monthly.add(1.0).rolling(months).apply(np.prod, raw=True).sub(1.0)
        valid = rolling.dropna()
        beat_rate = (
            valid.iloc[:, 0].gt(valid.iloc[:, 1]).mean() if not valid.empty else np.nan
        )
        output.append(
            {
                "Window": f"{years} years",
                "Strategy beat SPY": beat_rate,
                "Completed windows": len(valid),
            }
        )
    return pd.DataFrame(output)


def summarize_backtest(
    name: str,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    daily_weights: pd.DataFrame,
    starting_capital: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rebalance_count: int,
    ticker_names: dict[str, str] | None = None,
    holdings_frequency: str = "monthly",
) -> BacktestResult:
    combined = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    combined = combined.loc[(combined.index >= start_date) & (combined.index <= end_date)]
    if combined.empty:
        raise ValueError("No overlapping return data are available for the selected dates.")

    # Begin both portfolios at exactly the same point.
    combined.iloc[0] = 0.0
    equity = (1.0 + combined).cumprod() * starting_capital
    drawdowns = pd.concat(
        [_drawdown(combined.iloc[:, 0]), _drawdown(combined.iloc[:, 1])], axis=1
    )
    drawdowns.columns = combined.columns

    period_weights = daily_weights.reindex(combined.index).ffill().fillna(0.0)
    period_turnover = period_weights.diff().abs().sum(axis=1)
    if len(period_turnover):
        period_turnover.iloc[0] = period_weights.iloc[0].abs().sum()
    period_rebalance_count = int((period_turnover > 1e-12).sum())

    metric_rows = {
        "Strategy": _metric_row(combined.iloc[:, 0], starting_capital),
        "Buy & Hold SPY": _metric_row(combined.iloc[:, 1], starting_capital),
    }
    metrics = pd.DataFrame(metric_rows).T
    metrics["Trade/Rebalance Events"] = [period_rebalance_count, 1]

    annual = combined.resample("YE").apply(lambda x: (1.0 + x).prod() - 1.0)
    annual.index = annual.index.year
    annual.index.name = "Year"
    rolling = _rolling_outperformance(combined.iloc[:, 0], combined.iloc[:, 1])

    final_weights = period_weights.iloc[-1]
    final_weights = final_weights[final_weights > 1e-8].sort_values(ascending=False)
    holdings = final_weights.rename("Weight").reset_index().rename(columns={"index": "Ticker"})
    if ticker_names:
        holdings["Name"] = holdings["Ticker"].map(ticker_names).fillna(holdings["Ticker"])
        holdings = holdings[["Ticker", "Name", "Weight"]]

    if holdings_frequency == "annual":
        history_weights = period_weights.groupby(period_weights.index.year).first()
        history_weights.index = pd.to_datetime(
            history_weights.index.astype(str) + "-01-01"
        )
    elif holdings_frequency == "monthly":
        history_weights = period_weights.resample("ME").last()
    else:
        raise ValueError("Holdings frequency must be 'monthly' or 'annual'.")

    history_rows: list[dict[str, object]] = []
    previous_tickers: set[str] = set()
    for period, row in history_weights.iterrows():
        selected = row[row > 1e-8].sort_values(ascending=False)
        current_tickers = set(selected.index)
        for ticker, weight in selected.items():
            history_rows.append(
                {
                    "Month": period.to_period("M").to_timestamp(),
                    "Ticker": ticker,
                    "Name": ticker_names.get(ticker, ticker) if ticker_names else ticker,
                    "Weight": float(weight),
                    "Status": "Retained" if ticker in previous_tickers else "Added",
                }
            )
        previous_tickers = current_tickers
    holdings_history = pd.DataFrame(
        history_rows, columns=["Month", "Ticker", "Name", "Weight", "Status"]
    )

    return BacktestResult(
        name=name,
        strategy_returns=combined.iloc[:, 0],
        benchmark_returns=combined.iloc[:, 1],
        equity=equity,
        drawdowns=drawdowns,
        metrics=metrics,
        annual_returns=annual,
        rolling_results=rolling,
        current_holdings=holdings,
        holdings_history=holdings_history,
        rebalance_count=period_rebalance_count,
        holdings_frequency=holdings_frequency,
    )


def calculate_download_start(start_date: pd.Timestamp, lookback_months: int = 12) -> pd.Timestamp:
    return start_date - pd.DateOffset(months=lookback_months + 2, days=10)


def available_assets(prices: pd.DataFrame, minimum_observations: int = 252) -> list[str]:
    prices = _clean_prices(prices)
    return [column for column in prices if prices[column].count() >= minimum_observations]
