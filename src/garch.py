import numpy as np
import pandas as pd


from arch import arch_model

def garch_rolling_forecast_1d_single(
    d: pd.DataFrame,
    dist: str = "t",
    mean: str = "zero",
    p: int = 1,
    q: int = 1,
    refit_every: int = 21,   # ~monthly
    min_train: int = 750,    # ~3 years of trading days
):
    """
    Rolling/refit GARCH(p,q) one-step-ahead variance forecast for a single ticker.

    Inputs
    ------
    d: DataFrame for ONE ticker containing columns ['date', 'ret'].
       Must be sorted by date ascending.

    Output
    ------
    pd.Series of garch1_var aligned to d.index at time t:
      forecast at index t uses data up to t-1 and predicts variance of return at t (or t+1 depending on alignment).
    Here we align it as: at row t, store 1-step ahead forecast produced after observing returns up to t-1.
    """
    d = d.sort_values("date").copy()
    r = d["ret"].astype(float).reset_index(drop=True)

    # We'll return a Series aligned to d's original index
    fcast = pd.Series(index=d.index, dtype=float)

    last_fit = None
    last_fit_end = None

    # We iterate over rows; at "t" we fit on returns up to t-1 and store forecast for t (one-step ahead).
    # This lines up nicely with your targets (rv1_var at time t corresponds to ret_{t+1}^2 in your pipeline),
    # because later you'll compare forecasts aligned at time t with target rv1_var at time t.
    #
    # In other words:
    # - At time t, your features use info up to t
    # - Your target is ret_{t+1}^2
    # So your forecast should also be "variance of ret_{t+1}" computed using data up to t.
    #
    # We'll implement that by fitting on r[:t+1] and forecasting next step at row t.
    #
    # To keep it simple and consistent, we’ll:
    # - Fit on r.iloc[:t+1] (includes ret_t)
    # - Store the 1-step ahead forecast aligned at time t
    #
    # This is the correct alignment for predicting rv1_var(t) = ret_{t+1}^2

    for t in range(len(d)):
        if t < min_train:
            continue

        # Refit periodically
        if (last_fit is None) or (last_fit_end is None) or ((t - last_fit_end) >= refit_every):
            train = r.iloc[: t + 1]  # includes ret_t; forecasting t+1
            am = arch_model(train, mean=mean, vol="GARCH", p=p, q=q, dist=dist, rescale=False)
            try:
                last_fit = am.fit(disp="off")
                last_fit_end = t
            except Exception:
                # If fit fails, keep previous fit (if any)
                pass

        if last_fit is None:
            continue

        try:
            # One-step ahead variance forecast (for next period)
            v = last_fit.forecast(horizon=1, reindex=False).variance.values[-1, 0]
            fcast.iloc[t] = float(v)
        except Exception:
            pass

    return fcast

def add_garch_forecasts(
    df_f: pd.DataFrame,
    dist: str = "t",
    mean: str = "zero",
    p: int = 1,
    q: int = 1,
    refit_every: int = 21,
    min_train: int = 750,
):
    """
    Add GARCH forecasts to a panel DataFrame without using groupby.apply (avoids pandas FutureWarning).

    Adds:
    - garch1_var: 1-day ahead variance forecast aligned at time t (predicts ret_{t+1} variance)
    - garch5_var: simple approximation 5 * garch1_var (later we can do true multi-step)

    Returns df sorted by ['date','ticker'].
    """
    out = df_f.copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    garch_series_list = []

    for tkr in out["ticker"].unique():
        d = out[out["ticker"] == tkr].copy()
        s = garch_rolling_forecast_1d_single(
            d,
            dist=dist,
            mean=mean,
            p=p,
            q=q,
            refit_every=refit_every,
            min_train=min_train,
        )
        # s is aligned to d.index; store it in the master frame index
        garch_series_list.append(s.rename(tkr))

    # Combine forecasts back into one Series aligned to out.index
    # Each s has indices belonging to 'out' rows for that ticker; concat will align by index.
    garch_all = pd.concat(garch_series_list, axis=0).sort_index()
    out["garch1_var"] = garch_all

    # Simple 5-day approximation
    out["garch5_var"] = 5.0 * out["garch1_var"]

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)

def add_crisis_shading(fig, crisis_windows, opacity=0.45):
    for name, (start, end) in crisis_windows.items():
        fig.add_vrect(
            x0=pd.to_datetime(start),
            x1=pd.to_datetime(end),
            fillcolor="pink",
            opacity=opacity,
            line_width=0,
            annotation_text=name,
            annotation_position="top left"
        )
    return fig

import plotly.graph_objects as go
def add_crisis_shading(fig, start, end, name, crisis_windows, opacity=0.2):
    if start is None and end is None:

        for name, (start, end) in crisis_windows.items():
            fig.add_vrect(
                x0=pd.to_datetime(start),
                x1=pd.to_datetime(end),
                fillcolor="gray",
                opacity=opacity,
                line_width=0,
                annotation_text=name,
                annotation_position="top left"
            )
    else:
        fig.add_vrect(
                x0=pd.to_datetime(start),
                x1=pd.to_datetime(end),
                fillcolor="gray",
                opacity=opacity,
                line_width=0,
                annotation_text=name,
                annotation_position="top left"
            )

    return fig

import plotly.graph_objects as go
def plot_vol_compare(
    df,
    ticker,
    crisis_windows,
    target_var="rv1_var",
    forecast_vars=("hv1_var", "ewma1_var", "garch1_var_scaled"),
    start=None,
    end=None,
    name=None,
    title=None
):
    """
    Plot sqrt(variance) so it's more interpretable.
    target_var and forecast_vars should all be in percent^2 units.
    """
    d = df[df["ticker"] == ticker].sort_values("date").copy()
    if start is not None:
        d = d[d["date"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["date"] <= pd.to_datetime(end)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["date"], y=np.sqrt(d[target_var]),
        mode="lines", name="Realized vol"
    ))

    for col in forecast_vars:
        fig.add_trace(go.Scatter(
            x=d["date"], y=np.sqrt(d[col]),
            mode="lines", name=col
        ))

    fig.update_layout(
        title=title or f"{ticker}: Forecast vs Realized Vol",
        xaxis_title="Date",
        yaxis_title="Vol (same units as returns)"
    )
    add_crisis_shading(fig, start, end, name, crisis_windows)
    fig.show()


    # Helper function to fit parameters at refit points

def fit_garch_params(ret_series, mean="zero", dist="t", p=1, q=1):
    """
    Fit GARCH(p,q) on a return series (in percent units to match your current pipeline).
    Returns (omega, alpha, beta) for GARCH(1,1). For p=q=1 only.
    """
    am = arch_model(ret_series, mean=mean, vol="GARCH", p=p, q=q, dist=dist, rescale=False)
    res = am.fit(disp="off")
    params = res.params

    # For GARCH(1,1): omega, alpha[1], beta[1]
    omega = float(params["omega"])
    alpha = float(params[[k for k in params.index if "alpha[1]" in k][0]])
    beta  = float(params[[k for k in params.index if "beta[1]"  in k][0]])

    return omega, alpha, beta


# Code to implement rolling refit and daily recursion for a single ticker
def garch_refit_and_recurse_1d(
    d: pd.DataFrame,
    refit_every=21,
    min_train=750,
    mean="zero",
    dist="t",
):
    """
    Correct GARCH(1,1) 1-step-ahead variance forecast:
    - Refit parameters every refit_every days on expanding window
    - Update conditional variance DAILY via recursion between refits
    - Output aligned at time t as forecast of Var(ret_{t+1}) in percent^2

    This eliminates the flat-stretch problem.
    """
    d = d.sort_values("date").copy()
    ret = d["ret"].astype(float).reset_index(drop=True)      # percent returns
    r2  = (ret ** 2)

    fcast = pd.Series(index=d.index, dtype=float)

    omega = alpha = beta = None
    sigma2_t = None

    # initialize sigma2_t with sample variance once we have min_train
    for t in range(len(d)):
        if t < min_train:
            continue

        # refit parameters on schedule (or first time)
        if (omega is None) or ((t - min_train) % refit_every == 0):
            train = ret.iloc[: t + 1]
            try:
                omega, alpha, beta = fit_garch_params(train, mean=mean, dist=dist)
            except Exception:
                # if fit fails, keep previous params (if any)
                pass

            # initialize sigma2_t at refit time using recent variance
            if sigma2_t is None:
                sigma2_t = float(np.var(train.values, ddof=1))

        if omega is None:
            continue

        # Forecast Var(ret_{t+1}) using recursion based on information up to t
        # sigma2_{t+1} = omega + alpha * r_t^2 + beta * sigma2_t
        # Store forecast aligned at time t:
        sigma2_next = omega + alpha * float(r2.iloc[t]) + beta * float(sigma2_t)
        fcast.iloc[t] = sigma2_next

        # Advance state for next step
        sigma2_t = sigma2_next

    return fcast

# Wrapper function
def add_garch_refit_recurse(df_f, refit_every=21, min_train=750, mean="zero", dist="t"):
    out = df_f.copy().sort_values(["ticker","date"]).reset_index(drop=True)

    pieces = []
    for tkr in out["ticker"].unique():
        d = out[out["ticker"] == tkr].copy()
        s = garch_refit_and_recurse_1d(
            d,
            refit_every=refit_every,
            min_train=min_train,
            mean=mean,
            dist=dist,
        )
        pieces.append(s.rename(tkr))

    garch_all = pd.concat(pieces, axis=0).sort_index()
    out["garch1_var"] = garch_all
    out["garch5_var"] = 5.0 * out["garch1_var"]

    return out.sort_values(["date","ticker"]).reset_index(drop=True)


def plot_error_vol(df, ticker, target="rv1_var",
                   models=("ewma1_var","garch1_var","hv1_var"),
                   start=None, end=None, title=None):
    d = df[df["ticker"] == ticker].sort_values("date").copy()
    if start: d = d[d["date"] >= pd.to_datetime(start)]
    if end:   d = d[d["date"] <= pd.to_datetime(end)]

    realized = np.sqrt(d[target].values)

    fig = go.Figure()
    for m in models:
        err = np.sqrt(d[m].values) - realized
        fig.add_trace(go.Scatter(x=d["date"], y=err, mode="lines", name=f"{m} error"))

    fig.add_hline(y=0, line_width=1)
    fig.update_layout(
        title=title or f"{ticker}: Vol Forecast Error (sqrt(var) space)",
        xaxis_title="Date",
        yaxis_title="Forecast vol − Realized vol"
    )
    fig.show()



def gjr_garch_rolling_forecast_1d_single(
    d,
    refit_every=21,
    min_train=750,
    mean="zero",
    dist="t"
):
    
    d = d.sort_values("date").copy() # ensure chronological order
    ret = d["ret"].astype(float).values #extract returns
    r2 = ret ** 2 #sqaured returns
    
    fcast = np.full(len(d), np.nan)
    
    sigma2_t = None
    t = min_train
    
    while t < len(d):
        
        train = ret[:t+1]
        
        try:
            am = arch_model(
                train,
                mean=mean,
                vol="GARCH",
                p=1,
                o=1,        # <- THIS activates GJR asymmetry
                q=1,
                dist=dist,
                rescale=False
            )
            
            res = am.fit(disp="off")
            
            p = res.params
            
            omega = float(p["omega"])
            alpha = float(p[[k for k in p.index if "alpha[1]" in k][0]])
            gamma = float(p[[k for k in p.index if "gamma[1]" in k][0]])
            beta  = float(p[[k for k in p.index if "beta[1]"  in k][0]])
        
        except Exception:
            t += refit_every
            continue
        
        if sigma2_t is None:
            sigma2_t = np.var(train, ddof=1)
        
        t_end = min(t + refit_every, len(d))
        
        for i in range(t, t_end):
            
            indicator = 1.0 if ret[i] < 0 else 0.0
            
            sigma2_next = (
                omega
                + alpha * r2[i]
                + gamma * r2[i] * indicator
                + beta * sigma2_t
            )
            
            fcast[i] = sigma2_next
            sigma2_t = sigma2_next
        
        t = t_end
    
    return pd.Series(fcast, index=d.index)


def add_gjr_garch_forecast(
    df_f,
    refit_every=21,
    min_train=750,
    mean="zero",
    dist="t"
):
    
    out = df_f.copy().sort_values(["ticker","date"]).reset_index(drop=True)
    
    forecasts = []
    
    for tkr in out["ticker"].unique():
        
        d = out[out["ticker"] == tkr].copy()
        
        s = gjr_garch_rolling_forecast_1d_single(
            d,
            refit_every=refit_every,
            min_train=min_train,
            mean=mean,
            dist=dist
        )
        
        forecasts.append(s.rename(tkr))
    
    gjr_all = pd.concat(forecasts, axis=0).sort_index()
    
    out["gjr1_var"] = gjr_all
    out["gjr5_var"] = 5.0 * out["gjr1_var"]
    
    return out.sort_values(["date","ticker"]).reset_index(drop=True)