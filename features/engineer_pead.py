"""
PEAD-specific feature engineering: seasonal-random-walk SUE and forward
returns. Separate from features/engineer.py (the existing pipeline's
technical-indicator features), which is untouched.
"""
import numpy as np
import pandas as pd

MIN_PRIOR_SURPRISES = 8


def compute_sue_srw(income_df: pd.DataFrame) -> pd.DataFrame:
    """
    Seasonal random walk SUE:
        EPS_surprise_t = EPS_t - EPS_{t-4}
        SUE_t = EPS_surprise_t / std(EPS_surprise, trailing 8 quarters)

    PIT discipline: for the announcement at publish_date t, only prior
    surprises with publish_date STRICTLY LESS than t are eligible for the
    denominator -- position in the report_date-sorted frame is not enough
    on its own (a restated or out-of-order filing could make a positionally
    prior quarter not actually knowable yet; see test_sue_srw_pit_discipline).

    Requires exactly MIN_PRIOR_SURPRISES (8) prior surprises, never more --
    this keeps the denominator window consistent across events instead of
    expanding with however much history happens to be available. If fewer
    than 8 PIT-eligible prior surprises exist, sue_srw is NaN.

    Returns columns: ticker, publish_date, sue_srw, n_quarters_used,
    eps_t, eps_t4, surprise_t, surprise_std. n_quarters_used is always an
    integer count (0-8), even when sue_srw is NaN, so callers can filter on
    e.g. `n_quarters_used < 8` directly instead of against a NaN.
    """
    required = {"ticker", "report_date", "publish_date", "eps_basic"}
    missing = required - set(income_df.columns)
    if missing:
        raise ValueError(f"compute_sue_srw requires columns {required}, missing {missing}")

    df = income_df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["publish_date"] = pd.to_datetime(df["publish_date"])
    df = df.sort_values(["ticker", "report_date"]).reset_index(drop=True)

    rows = []
    for ticker, group in df.groupby("ticker", sort=False):
        group = group.reset_index(drop=True)
        eps = group["eps_basic"].to_numpy(dtype=float)
        publish = group["publish_date"].to_numpy()
        n = len(group)

        surprise = np.full(n, np.nan)
        for i in range(4, n):
            if not np.isnan(eps[i]) and not np.isnan(eps[i - 4]):
                surprise[i] = eps[i] - eps[i - 4]

        for i in range(n):
            if np.isnan(surprise[i]):
                continue

            candidates = []
            for j in range(i - 1, -1, -1):
                if len(candidates) == MIN_PRIOR_SURPRISES:
                    break
                if np.isnan(surprise[j]):
                    continue
                if publish[j] < publish[i]:
                    candidates.append(surprise[j])

            n_used = len(candidates)
            if n_used == MIN_PRIOR_SURPRISES:
                surprise_std = float(np.std(candidates, ddof=1))
                sue = surprise[i] / surprise_std if surprise_std > 0 else np.nan
            else:
                surprise_std = np.nan
                sue = np.nan

            rows.append({
                "ticker": ticker,
                "publish_date": pd.Timestamp(publish[i]),
                "sue_srw": sue,
                "n_quarters_used": n_used,
                "eps_t": eps[i],
                "eps_t4": eps[i - 4],
                "surprise_t": surprise[i],
                "surprise_std": surprise_std,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        # Equivalent to the "n_quarters_used must be exactly 8 for every
        # non-null SUE" invariant, checked directly against non-null sue_srw
        # rows rather than via n_quarters_used.dropna() -- n_quarters_used is
        # never NaN here (it's always a real 0-8 count, deliberately, so
        # callers can filter on it directly), so a dropna()-based check would
        # be a no-op instead of the intended windowing-correctness check.
        non_null = result[result["sue_srw"].notna()]
        assert (non_null["n_quarters_used"] == MIN_PRIOR_SURPRISES).all(), (
            "windowing logic is wrong: found a non-null SUE without exactly "
            f"{MIN_PRIOR_SURPRISES} prior quarters"
        )
    return result


def compute_forward_returns(
    prices_df: pd.DataFrame,
    events_df: pd.DataFrame,
    hold_days: int = 20,
    stop_loss_pct: float = 0.10,
) -> pd.DataFrame:
    """
    For each (ticker, publish_date) event:

    entry_date = the first trading day STRICTLY AFTER publish_date -- i.e.
        the smallest date in the price series greater than publish_date.
        "T+1 open following the earnings announcement" means the first
        trading day's open after the announcement date itself, not after
        the first trading day after the announcement date. This handles
        every case uniformly with no intermediate "effective announcement
        date" step: a weekday publish enters the next trading day; a
        Saturday or Sunday publish enters the following Monday; a Friday
        publish before a 3-day weekend enters the following Tuesday.
    entry_price = open on entry_date.

    exit_date/exit_price: close hold_days trading days after entry_date,
    UNLESS the close on any day in [entry_date, exit_date] falls more than
    stop_loss_pct below entry_price, in which case exit is that day's close
    (stopped=True, days_to_stop recorded).

    Gross returns only -- transaction costs are applied in the backtest
    layer (backtest/evaluate.py), same convention as the rest of the
    pipeline.

    Returns columns: ticker, publish_date, entry_date, entry_price,
    exit_date, exit_price, gross_return, stopped, hold_days_actual,
    days_to_stop.
    """
    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    events_df = events_df.copy()
    events_df["publish_date"] = pd.to_datetime(events_df["publish_date"])

    rows = []
    for ticker, ticker_events in events_df.groupby("ticker", sort=False):
        price_group = prices_df[prices_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        dates = price_group["date"].to_numpy()
        opens = price_group["open"].to_numpy(dtype=float)
        closes = price_group["close"].to_numpy(dtype=float)
        n_prices = len(price_group)

        for _, event in ticker_events.iterrows():
            publish_date = event["publish_date"]

            # side="right" gives the index of the first date strictly
            # greater than publish_date directly -- no separate "advance to
            # next trading day, then +1" step, which was adding an extra
            # day of lag for weekend/holiday publish dates.
            entry_idx = np.searchsorted(dates, np.datetime64(publish_date), side="right")
            if entry_idx >= n_prices:
                continue  # not enough trailing price data loaded for this event yet

            entry_date = dates[entry_idx]
            entry_price = opens[entry_idx]

            exit_idx_primary = min(entry_idx + hold_days, n_prices - 1)
            stop_threshold = entry_price * (1 - stop_loss_pct)

            stopped = False
            days_to_stop = np.nan
            exit_idx = exit_idx_primary
            for k in range(entry_idx, exit_idx_primary + 1):
                if closes[k] < stop_threshold:
                    stopped = True
                    exit_idx = k
                    days_to_stop = k - entry_idx
                    break

            exit_date = dates[exit_idx]
            exit_price = closes[exit_idx]
            gross_return = (exit_price - entry_price) / entry_price

            rows.append({
                "ticker": ticker,
                "publish_date": pd.Timestamp(publish_date),
                "entry_date": pd.Timestamp(entry_date),
                "entry_price": entry_price,
                "exit_date": pd.Timestamp(exit_date),
                "exit_price": exit_price,
                "gross_return": gross_return,
                "stopped": stopped,
                "hold_days_actual": exit_idx - entry_idx,
                "days_to_stop": days_to_stop,
            })

    return pd.DataFrame(rows)
