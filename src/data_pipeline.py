# src/data_pipeline.py

import numpy as np
import pandas as pd
import yfinance as yf


def sort_date_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date then ticker (panel-friendly ordering)."""
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def sort_ticker_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by ticker then date (safe for shift/rolling within each ticker)."""
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_ohlc(tickers, start, end=None):
    """
    Download daily OHLCV + Adj Close for one or more tickers from yfinance.
    Returns a tidy DataFrame: one row per (date, ticker).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        tidy = []
        for t in tickers:
            sub = df.xs(t, axis=1, level=1).copy()
            sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
            sub["ticker"] = t
            sub = sub.reset_index()
            tidy.append(sub)
        out = pd.concat(tidy, ignore_index=True)
    else:
        out = df.copy()
        out.columns = [c.lower().replace(" ", "_") for c in out.columns]
        out = out.reset_index()
        out["ticker"] = tickers[0]

    out = out.rename(columns={"Date": "date", "adjclose": "adj_close"})
    needed = {"date","ticker","open","high","low","close","adj_close","volume"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Missing columns from yfinance output: {missing}")

    return sort_date_ticker(out)


def clean_and_align(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Drop missing adj_close, dedupe, and keep only common dates across tickers."""
    df = raw_data.dropna(subset=["adj_close"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"])

    n_tickers = df["ticker"].nunique()
    counts = df.groupby("date")["ticker"].nunique()
    common_dates = counts[counts == n_tickers].index

    df = df[df["date"].isin(common_dates)].copy()
    return sort_date_ticker(df)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns (%) from adj_close per ticker; drop NaN returns."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")

    tmp = sort_ticker_date(out)
    tmp["ret"] = tmp.groupby("ticker")["adj_close"].transform(
        lambda s: 100.0 * np.log(s / s.shift(1))
    )

    tmp = tmp.dropna(subset=["ret"]).copy()
    return sort_date_ticker(tmp)


def add_targets(df: pd.DataFrame, horizons=(1, 5)) -> pd.DataFrame:
    """Add forward realized variance targets (rv1_var, rv5_var, ...); drop NaN targets."""
    out = df.copy()
    out["ret2"] = out["ret"] ** 2

    tmp = sort_ticker_date(out)

    for h in horizons:
        if h == 1:
            tmp[f"rv{h}_var"] = tmp.groupby("ticker")["ret2"].shift(-1)
        else:
            tmp[f"rv{h}_var"] = tmp.groupby("ticker")["ret2"].transform(
                lambda s: s.shift(-1).rolling(window=h, min_periods=h).sum()
            )

    target_cols = [f"rv{h}_var" for h in horizons]
    tmp = tmp.dropna(subset=target_cols).copy()
    return sort_date_ticker(tmp)


def add_regime_labels(df: pd.DataFrame, crisis_windows: dict) -> pd.DataFrame:
    """
    Add regime labels for evaluation slicing.
    If crisis_windows is empty {}, all rows remain 'calm'.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["regime"] = "calm"

    for name, (start, end) in crisis_windows.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        mask = (out["date"] >= start_dt) & (out["date"] <= end_dt)
        out.loc[mask, "regime"] = name

    return sort_date_ticker(out)


def make_dataset(
    tickers,
    start="2000-01-01",
    end=None,
    horizons=(1, 5),
    crisis_windows=None,
) -> pd.DataFrame:
    """
    Full pipeline: download -> clean/align -> returns -> targets -> regime labels.
    If crisis_windows is None: uses {} (no special regimes; all 'calm').
    """
    crisis_windows = {} if crisis_windows is None else crisis_windows

    raw = download_ohlc(tickers=tickers, start=start, end=end)
    df = clean_and_align(raw)
    df = add_returns(df)
    df = add_targets(df, horizons=horizons)
    df = add_regime_labels(df, crisis_windows=crisis_windows)

    target_cols = [f"rv{h}_var" for h in horizons]
    keep_cols = [
        "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
        "ret", "ret2", *target_cols, "regime"
    ]
    return sort_date_ticker(df[keep_cols].copy())


if __name__ == "__main__":
    # This block runs ONLY if you execute this file directly:
    # python src/data_pipeline.py
    # It will NOT run when you import the module in a notebook.
    CRISIS_WINDOWS = {
        "GFC_2007_2009": ("2007-07-01", "2009-06-30"),
        "COVID_2020": ("2020-02-15", "2020-05-31"),
    }
    df = make_dataset(["SPY", "JPM"], start="2000-01-01", crisis_windows=CRISIS_WINDOWS)
    print(df.head())

    # Print few rows from GFC 
    print(df[df["regime"] == "GFC_2007_2009"].head())

    # Print few rows from COVID 
    print(df[df["regime"] == "COVID_2020"].head())

    # Check if GFC and COVID rows exist and majority are calm
    print(df["regime"].value_counts())
