import numpy as np
import pandas as pd

from strategy_engine import (
    momentum_weights,
    point_in_time_membership_mask,
    run_momentum_strategy,
    run_trend_strategy,
    summarize_backtest,
)


def synthetic_prices(days=900):
    index = pd.bdate_range("2020-01-01", periods=days)
    x = np.arange(days)
    return pd.DataFrame(
        {
            "UP": 100 * np.exp(0.0008 * x),
            "MID": 100 * np.exp(0.0004 * x),
            "DOWN": 100 * np.exp(-0.0002 * x),
        },
        index=index,
    )


def test_momentum_selects_winner_without_lookahead():
    prices = synthetic_prices()
    weights = momentum_weights(prices, top_n=1, lookback_months=12, skip_months=1)
    invested = weights[weights.sum(axis=1) > 0]
    assert not invested.empty
    assert (invested["UP"] == 1.0).all()


def test_point_in_time_membership_blocks_future_constituent():
    dates = pd.to_datetime(["2019-12-31", "2020-12-31", "2021-01-31"])
    membership = pd.DataFrame(
        {
            "ticker": ["TSLA", "OLD"],
            "start_date": ["2020-12-21", "2010-01-01"],
            "end_date": [None, "2020-06-01"],
        }
    )
    mask = point_in_time_membership_mask(membership, dates, ["TSLA", "OLD"])
    assert not mask.loc[pd.Timestamp("2019-12-31"), "TSLA"]
    assert mask.loc[pd.Timestamp("2020-12-31"), "TSLA"]
    assert mask.loc[pd.Timestamp("2019-12-31"), "OLD"]
    assert not mask.loc[pd.Timestamp("2020-12-31"), "OLD"]


def test_point_in_time_membership_includes_former_constituent_only_when_active():
    membership = pd.read_csv("sp500_membership_intervals.csv")
    membership["ticker"] = membership["ticker"].str.replace(".", "-", regex=False)
    dates = pd.to_datetime(["2020-12-31", "2024-12-31"])
    mask = point_in_time_membership_mask(membership, dates, ["ATVI"])
    assert mask.loc[pd.Timestamp("2020-12-31"), "ATVI"]
    assert not mask.loc[pd.Timestamp("2024-12-31"), "ATVI"]


def test_momentum_respects_point_in_time_eligibility():
    prices = synthetic_prices()
    dates = prices.resample("ME").last().index
    eligibility = pd.DataFrame(True, index=dates, columns=prices.columns)
    eligibility["UP"] = False
    weights = momentum_weights(
        prices, top_n=1, lookback_months=12, skip_months=1,
        eligibility=eligibility,
    )
    invested = weights[weights.sum(axis=1) > 0]
    assert not invested.empty
    assert (invested["MID"] == 1.0).all()


def test_momentum_and_summary_run():
    prices = synthetic_prices()
    strategy, benchmark, weights, count = run_momentum_strategy(
        prices, prices["MID"], 1, 12, 1, 5
    )
    result = summarize_backtest(
        "Test", strategy, benchmark, weights, 10_000,
        pd.Timestamp("2021-03-01"), prices.index[-1], count,
    )
    assert result.metrics.loc["Strategy", "Ending Value"] > 0
    assert result.equity.index.is_monotonic_increasing
    assert count > 0
    assert not result.holdings_history.empty
    assert set(result.holdings_history.columns) == {"Month", "Ticker", "Name", "Weight", "Status"}
    monthly_weight = result.holdings_history.groupby("Month")["Weight"].sum()
    assert np.allclose(monthly_weight, 1.0)


def test_trend_strategy_weights_sum_to_one():
    prices = synthetic_prices()
    strategy, benchmark, weights, count = run_trend_strategy(
        prices["UP"], prices["MID"], 200, 5
    )
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert strategy.index.equals(benchmark.index)
    assert count > 0
