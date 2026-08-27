import numpy as np
import pandas as pd

def hv_forecast_var(d: pd.DataFrame, window=20) -> pd.Series:
    """1-day variance forecast using rolling variance of returns."""
    return d["ret"].rolling(window).var()

def hv_forecast_var_h(d: pd.DataFrame, window=20, horizon=5) -> pd.Series:
    """h-day variance forecast via scaling: h * rolling_var."""
    return horizon * hv_forecast_var(d, window=window)

def ewma_var_series(ret: pd.Series, lam=0.94) -> pd.Series:
    """
    EWMA conditional variance series sigma2_t.
    sigma2_t = lam*sigma2_{t-1} + (1-lam)*ret_{t-1}^2
    """
    r2 = ret**2
    sigma2 = np.empty(len(ret), dtype=float)
    sigma2[:] = np.nan
    sigma2[0] = np.nanvar(ret.values, ddof=1)

    for t in range(1, len(ret)):
        sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r2.iloc[t-1]

    return pd.Series(sigma2, index=ret.index)

def ewma_forecast_var_h(d: pd.DataFrame, lam=0.94, horizon=1) -> pd.Series:
    """h-day variance forecast via h * EWMA sigma2_t."""
    sigma2 = ewma_var_series(d["ret"], lam=lam)
    return horizon * sigma2

def ewma_var_from_ret(ret: pd.Series, lam=0.94) -> pd.Series:
    """
    EWMA variance series sigma2_t aligned at time t (uses ret_{t-1}^2 in update).
    ret is a Series for ONE ticker.
    """
    r2 = ret**2
    sigma2 = np.empty(len(ret), dtype=float)
    sigma2[:] = np.nan

    # Initialize with sample variance of the series
    sigma2[0] = np.nanvar(ret.values, ddof=1)

    for t in range(1, len(ret)):
        sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r2.iloc[t-1]

    return pd.Series(sigma2, index=ret.index)

def make_baseline_forecasts(df: pd.DataFrame, hv_window=20, ewma_lam=0.94) -> pd.DataFrame:
    """
    Add baseline variance forecasts to the panel dataframe (no pandas apply warnings):
      - hv1_var, hv5_var
      - ewma1_var, ewma5_var

    Notes:
    - HV uses rolling variance of returns.
    - EWMA uses RiskMetrics recursion.
    - 5-day forecasts are approximated as 5 * 1-day conditional variance.
    """
    out = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    # Rolling Historical Variance (1-day variance forecast)
    out["hv1_var"] = out.groupby("ticker")["ret"].transform(
        lambda s: s.rolling(hv_window).var()
    )
    out["hv5_var"] = 5.0 * out["hv1_var"]

    # EWMA conditional variance series (1-day variance forecast)
    out["ewma1_var"] = out.groupby("ticker")["ret"].transform(
        lambda s: ewma_var_from_ret(s, lam=ewma_lam)
    )
    out["ewma5_var"] = 5.0 * out["ewma1_var"]

    # Return in panel-friendly order
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out