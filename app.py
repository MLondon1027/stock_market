from __future__ import annotations

from datetime import date
from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from strategy_engine import (
    available_assets,
    calculate_download_start,
    point_in_time_membership_mask,
    run_momentum_strategy,
    run_trend_strategy,
    summarize_backtest,
)


st.set_page_config(page_title="Market Strategy Lab", page_icon="📈", layout="wide")

SECTORS = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

MEMBERSHIP_FILE = Path(__file__).with_name("sp500_membership_intervals.csv")
MEMBERSHIP_DATA_AS_OF = date(2026, 8, 18)


@st.cache_data(show_spinner=False)
def load_sp500_membership() -> pd.DataFrame:
    membership = pd.read_csv(MEMBERSHIP_FILE)
    membership["start_date"] = pd.to_datetime(membership["start_date"], errors="coerce")
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce")
    membership = membership.dropna(subset=["ticker", "start_date"])
    # Yahoo uses dashes for share classes while the membership source uses dots.
    membership["ticker"] = membership["ticker"].str.replace(".", "-", regex=False)
    return membership.sort_values(["start_date", "ticker"]).reset_index(drop=True)


def relevant_member_tickers(
    membership: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> list[str]:
    overlaps = membership["start_date"].le(end) & (
        membership["end_date"].isna() | membership["end_date"].gt(start)
    )
    return sorted(membership.loc[overlaps, "ticker"].unique())


@st.cache_data(ttl=21_600, show_spinner=False)
def download_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    data = pd.DataFrame()
    for attempt in range(2):
        data = yf.download(
            list(tickers),
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=8,
            group_by="column",
            timeout=30,
        )
        if not data.empty:
            break
        if attempt == 0:
            time.sleep(3)

    if data.empty:
        raise ValueError("The market-data provider returned no prices. Retry in a minute.")
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Downloaded data did not contain adjusted closing prices.")
        close = data["Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.sort_index().dropna(how="all")


def money(value: float) -> str:
    return f"${value:,.0f}"


def percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def display_result(result) -> None:
    strategy = result.metrics.loc["Strategy"]
    benchmark = result.metrics.loc["Buy & Hold SPY"]
    advantage = strategy["Ending Value"] - benchmark["Ending Value"]

    cols = st.columns(5)
    cols[0].metric("Strategy ending value", money(strategy["Ending Value"]), money(advantage) + " vs. SPY")
    cols[1].metric("SPY ending value", money(benchmark["Ending Value"]))
    cols[2].metric("Strategy CAGR", percent(strategy["CAGR"]), percent(strategy["CAGR"] - benchmark["CAGR"]) + " vs. SPY")
    cols[3].metric("Maximum drawdown", percent(strategy["Max Drawdown"]), percent(strategy["Max Drawdown"] - benchmark["Max Drawdown"]) + " vs. SPY")
    cols[4].metric("Rebalance/trade events", f"{int(strategy['Trade/Rebalance Events']):,}")

    st.subheader("Growth of the starting investment")
    growth = result.equity.rename_axis("Date").reset_index().melt("Date", var_name="Portfolio", value_name="Value")
    fig = px.line(growth, x="Date", y="Value", color="Portfolio")
    fig.update_layout(yaxis_tickprefix="$", hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drawdowns")
    dd = result.drawdowns.rename_axis("Date").reset_index().melt("Date", var_name="Portfolio", value_name="Drawdown")
    dd_fig = px.area(dd, x="Date", y="Drawdown", color="Portfolio")
    dd_fig.update_layout(yaxis_tickformat=".0%", hovermode="x unified", legend_title_text="")
    st.plotly_chart(dd_fig, use_container_width=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Performance statistics")
        metrics = result.metrics.copy()
        metrics["Ending Value"] = metrics["Ending Value"].map(money)
        for column in ["Total Return", "CAGR", "Volatility", "Max Drawdown", "Worst Year"]:
            metrics[column] = metrics[column].map(percent)
        metrics["Sharpe"] = metrics["Sharpe"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
        metrics["Trade/Rebalance Events"] = metrics["Trade/Rebalance Events"].astype(int)
        st.dataframe(metrics.T, use_container_width=True)

        st.subheader("Rolling consistency")
        rolling = result.rolling_results.copy()
        rolling["Strategy beat SPY"] = rolling["Strategy beat SPY"].map(percent)
        st.dataframe(rolling, hide_index=True, use_container_width=True)
        st.caption("Each overlapping window compares compounded strategy return with SPY over the same dates.")

    with right:
        st.subheader("Calendar-year returns")
        annual = result.annual_returns.copy()
        annual.columns = ["Strategy", "Buy & Hold SPY"]
        st.dataframe(annual.style.format("{:.1%}"), use_container_width=True, height=390)

        st.subheader("Holdings at the end of the test")
        holdings = result.current_holdings.copy()
        holdings["Weight"] = holdings["Weight"].map(percent)
        st.dataframe(holdings, hide_index=True, use_container_width=True)

    annual_holdings = result.holdings_frequency == "annual"
    period_name = "year" if annual_holdings else "month"
    period_title = "Year" if annual_holdings else "Month"

    st.subheader(f"Holdings for every {period_name}")
    if result.holdings_history.empty:
        st.info(f"No {period_name}ly holdings were available for this period.")
    else:
        period_options = sorted(result.holdings_history["Month"].drop_duplicates(), reverse=True)
        selected_period = st.selectbox(
            f"Select a {period_name}",
            options=period_options,
            format_func=(
                (lambda value: pd.Timestamp(value).strftime("%Y"))
                if annual_holdings
                else (lambda value: pd.Timestamp(value).strftime("%B %Y"))
            ),
        )
        period_holdings = result.holdings_history.loc[
            result.holdings_history["Month"] == selected_period
        ].copy()
        period_holdings["Month"] = period_holdings["Month"].dt.strftime(
            "%Y" if annual_holdings else "%B %Y"
        )
        period_holdings = period_holdings.rename(columns={"Month": period_title})
        period_holdings["Weight"] = period_holdings["Weight"].map(percent)
        st.dataframe(period_holdings, hide_index=True, use_container_width=True)
        st.caption(
            f"Added means the ticker was not held in the preceding {period_name}; "
            f"Retained means it remained in the portfolio."
        )

        with st.expander(f"View the complete {period_name}-by-{period_name} holdings table"):
            complete_history = result.holdings_history.copy()
            complete_history["Month"] = complete_history["Month"].dt.strftime(
                "%Y" if annual_holdings else "%Y-%m"
            )
            complete_history = complete_history.rename(columns={"Month": period_title})
            complete_history["Weight"] = complete_history["Weight"].map(percent)
            st.dataframe(complete_history, hide_index=True, use_container_width=True, height=500)

        history_csv = result.holdings_history.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download {period_name}ly holdings history",
            history_csv,
            f"{period_name}ly_holdings_history.csv",
            "text/csv",
        )

    csv = result.equity.rename_axis("Date").reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download daily portfolio values", csv, "backtest_portfolio_values.csv", "text/csv")


st.title("Market Strategy Lab")
st.write("Test rules-based strategies against buying and holding SPY, beginning with the same investment.")

with st.sidebar:
    st.header("Backtest settings")
    strategy_name = st.selectbox(
        "Strategy",
        [
            "Sector momentum",
            "Point-in-time S&P 500 momentum",
            "SPY 200-day trend",
        ],
    )
    maximum_end_date = (
        MEMBERSHIP_DATA_AS_OF
        if strategy_name == "Point-in-time S&P 500 momentum"
        else date.today()
    )
    start_date = st.date_input(
        "Start date", value=date(2010, 1, 1), min_value=date(2003, 1, 1),
        max_value=maximum_end_date, key=f"start_date_{strategy_name}",
    )
    end_date = st.date_input(
        "End date", value=maximum_end_date, min_value=date(2003, 1, 2),
        max_value=maximum_end_date, key=f"end_date_{strategy_name}",
    )
    starting_capital = st.number_input("Starting investment", min_value=100.0, value=10_000.0, step=1_000.0)
    transaction_cost_bps = st.number_input(
        "Cost per one-way trade (basis points)", min_value=0.0, max_value=100.0, value=5.0, step=1.0,
        help="Five basis points equals 0.05%. A complete switch includes both sales and purchases.",
    )

    rebalance_frequency = "monthly"
    if "momentum" in strategy_name.lower():
        default_top = 3 if strategy_name == "Sector momentum" else 10
        maximum_top = len(SECTORS) if strategy_name == "Sector momentum" else 25
        top_n = st.slider("Number of holdings", 1, maximum_top, default_top)
        if strategy_name == "Point-in-time S&P 500 momentum":
            rebalance_choice = st.selectbox(
                "Rebalancing schedule",
                [
                    "Monthly",
                    "Annually using the previous calendar year",
                ],
                help=(
                    "Annual mode ranks the prior calendar year's performance, "
                    "buys for the first trading day, and holds through year-end."
                ),
            )
            rebalance_frequency = (
                "annual" if rebalance_choice.startswith("Annually") else "monthly"
            )

        if rebalance_frequency == "annual":
            lookback_months = 12
            skip_months = 0
            st.caption(
                "Annual mode uses the complete previous calendar year. "
                "There is no separate lookback or excluded-month setting."
            )
        else:
            lookback_months = st.slider("Momentum lookback (months)", 3, 18, 12)
            skip_months = st.slider("Most recent months excluded", 0, 3, 1)
    else:
        moving_average_days = st.slider("Moving-average length (trading days)", 50, 300, 200, 10)

    run = st.button("Run backtest", type="primary", use_container_width=True)

st.info(
    "Research backtest only. Results exclude taxes and may differ from executable prices. "
    "Past performance does not predict future results."
)

if run:
    if start_date >= end_date:
        st.error("The start date must be before the end date.")
        st.stop()

    price_coverage_note: str | None = None
    try:
        with st.spinner("Downloading prices and running the strategy…"):
            requested_start = pd.Timestamp(start_date)
            requested_end = pd.Timestamp(end_date)

            if strategy_name == "Sector momentum":
                download_start = calculate_download_start(requested_start, lookback_months)
                tickers = tuple(list(SECTORS) + ["SPY"])
                prices = download_prices(tickers, download_start.date(), end_date)
                assets = [ticker for ticker in SECTORS if ticker in prices.columns]
                strategy, benchmark, weights, count = run_momentum_strategy(
                    prices[assets], prices["SPY"], top_n, lookback_months, skip_months, transaction_cost_bps
                )
                result = summarize_backtest(
                    strategy_name, strategy, benchmark, weights, starting_capital,
                    requested_start, requested_end, count, SECTORS,
                )

            elif strategy_name == "Point-in-time S&P 500 momentum":
                if end_date > MEMBERSHIP_DATA_AS_OF:
                    raise ValueError(
                        f"Historical S&P membership is verified only through "
                        f"{MEMBERSHIP_DATA_AS_OF:%B %d, %Y}. Choose an earlier end date."
                    )
                download_start = calculate_download_start(requested_start, lookback_months)
                membership = load_sp500_membership()
                member_tickers = relevant_member_tickers(
                    membership, download_start, requested_end
                )
                tickers = tuple(member_tickers + ["SPY"])
                prices = download_prices(tickers, download_start.date(), end_date)
                assets = [ticker for ticker in available_assets(prices.drop(columns=["SPY"], errors="ignore"), 120)]
                if len(assets) < top_n:
                    raise ValueError(f"Only {len(assets)} stocks had usable data; reduce the number of holdings or retry.")
                formation_rule = "YE" if rebalance_frequency == "annual" else "ME"
                formation_dates = prices[assets].resample(formation_rule).last().index
                eligibility = point_in_time_membership_mask(
                    membership, formation_dates, assets
                )
                full_eligibility = point_in_time_membership_mask(
                    membership, formation_dates, member_tickers
                )
                expected_members = full_eligibility.sum(axis=1).replace(0, pd.NA)
                covered_members = eligibility.sum(axis=1)
                coverage = covered_members.div(expected_members).dropna()
                if not coverage.empty:
                    price_coverage_note = (
                        f"Yahoo usable-symbol coverage across monthly S&P rosters: "
                        f"median {coverage.median():.1%}; minimum {coverage.min():.1%}."
                    )
                strategy, benchmark, weights, count = run_momentum_strategy(
                    prices[assets], prices["SPY"], top_n, lookback_months,
                    skip_months, transaction_cost_bps, eligibility,
                    rebalance_frequency,
                )
                result_name = (
                    f"{strategy_name} — annual prior-year winners"
                    if rebalance_frequency == "annual"
                    else strategy_name
                )
                result = summarize_backtest(
                    result_name, strategy, benchmark, weights, starting_capital,
                    requested_start, requested_end, count,
                    holdings_frequency=rebalance_frequency,
                )

            else:
                download_start = requested_start - pd.Timedelta(days=moving_average_days * 2)
                prices = download_prices(("SPY", "SHY"), download_start.date(), end_date)
                strategy, benchmark, weights, count = run_trend_strategy(
                    prices["SPY"], prices["SHY"], moving_average_days, transaction_cost_bps
                )
                result = summarize_backtest(
                    strategy_name, strategy, benchmark, weights, starting_capital,
                    requested_start, requested_end, count,
                    {"SPY": "S&P 500 ETF", "SHY": "1–3 Year Treasury ETF"},
                )

        display_result(result)

        if strategy_name == "Point-in-time S&P 500 momentum":
            if rebalance_frequency == "annual":
                st.caption(
                    "Each portfolio uses the previous calendar year's return and the S&P 500 "
                    "membership in effect at that year-end. It is then held until the next "
                    "year's first-trading-day rebalance."
                )
            else:
                st.caption(
                    "Eligibility uses the S&P 500 membership in effect at each month-end. "
                    "Tesla, for example, cannot be selected before December 2020."
                )
            st.caption(
                "Yahoo may omit some discontinued historical symbols; these omissions are "
                "disclosed in the README."
            )
            if price_coverage_note:
                st.caption(price_coverage_note)
        elif strategy_name == "Sector momentum":
            st.caption("Sector test uses the nine original Select Sector SPDR ETFs to preserve a long common history.")

    except Exception as exc:
        st.error(f"Backtest could not run: {exc}")
        st.caption("Yahoo Finance occasionally throttles large downloads. Wait briefly and run the test again.")
else:
    st.subheader("Included strategies")
    st.markdown(
        """
        - **Sector momentum:** each month, holds the strongest sector ETFs based on prior performance.
        - **Point-in-time S&P 500 momentum:** choose monthly rebalancing or buy the prior calendar year's winners once per year.
        - **SPY trend:** holds SPY above its moving average and short-term Treasury bonds otherwise.

        The app delays every signal before applying the next return, includes trading costs, and uses adjusted prices.
        """
    )
