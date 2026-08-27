import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def plot_rv_acf(df, ticker, max_lag=150):
    """
    Plot autocorrelation of daily realized variance rv1_var
    for one ticker.
    """
    d = df[df["ticker"] == ticker].sort_values("date").copy()
    x = d["rv1_var"].dropna()

    acf_vals = [x.autocorr(lag=lag) for lag in range(1, max_lag + 1)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, max_lag + 1)),
        y=acf_vals,
        name="ACF"
    ))

    # highlight important lags
    for lag in [1, 7, 21, 100]:
        if lag <= max_lag:
            fig.add_vline(x=lag, line_dash="dash", line_color="red")

    fig.update_layout(
        title=f"{ticker}: Autocorrelation of Daily Realized Variance",
        xaxis_title="Lag (days)",
        yaxis_title="Autocorrelation"
    )

    fig.show()


def make_har_features_single(d):
    """
    Create HAR-RV features for a single ticker.

    Uses lagged realized variance at 3 time scales:
    - rv_d : yesterday's realized variance
    - rv_w : average realized variance over last 5 days
    - rv_m : average realized variance over last 22 days
    """
    d = d.sort_values("date").copy()

    # 1-day lag
    d["rv_d"] = d["rv1_var"].shift(1)

    # 5-day average lag
    d["rv_w"] = d["rv1_var"].shift(1).rolling(window=5).mean()

    # 22-day average lag
    d["rv_m"] = d["rv1_var"].shift(1).rolling(window=22).mean()

    return d

def add_har_features(df):
    """
    Apply HAR feature construction separately to each ticker.
    """
    out = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    pieces = []
    for tkr in out["ticker"].unique():
        d = out[out["ticker"] == tkr].copy()
        d = make_har_features_single(d)
        pieces.append(d)

    out = pd.concat(pieces, axis=0).sort_values(["date", "ticker"]).reset_index(drop=True)
    return out

''' Reuseable rolling HAR forecaster for any target'''

def har_rolling_forecast_single(
    d,
    target_col="rv1_var",
    refit_every=21,
    min_train=252
):
    """
    Expanding-window HAR forecast for one ticker.

    Parameters
    ----------
    d : DataFrame for one ticker, must contain:
        ['date', 'rv_d', 'rv_w', 'rv_m', target_col]
    target_col : str
        Either 'rv1_var' or 'rv5_var'
    refit_every : int
        Refit frequency in trading days
    min_train : int
        Minimum number of valid training rows
    """
    d = d.sort_values("date").copy()

    feature_cols = ["rv_d", "rv_w", "rv_m"]
    fcast = np.full(len(d), np.nan)

    t = min_train

    while t < len(d):
        # training data up to t
        train = d.iloc[:t+1].dropna(subset=feature_cols + [target_col]).copy()

        if len(train) < min_train:
            t += refit_every
            continue

        X_train = train[feature_cols].values
        y_train = train[target_col].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        # forecast until next refit
        t_end = min(t + refit_every, len(d))

        for i in range(t, t_end):
            row = d.iloc[i]

            if row[feature_cols].isna().any():
                continue

            X_test = row[feature_cols].values.reshape(1, -1)
            fcast[i] = model.predict(X_test)[0]

        t = t_end

    return pd.Series(fcast, index=d.index)

''' Add HAR forecasts for both 1-day and 5-day horizons'''

def add_har_forecasts(
    df,
    refit_every=21,
    min_train=252
):
    """
    Add proper HAR forecasts for both 1-day and 5-day horizons.

    Adds:
    - har1_var : HAR fitted directly to rv1_var
    - har5_var : HAR fitted directly to rv5_var
    """
    out = add_har_features(df)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    har1_list = []
    har5_list = []

    for tkr in out["ticker"].unique():
        d = out[out["ticker"] == tkr].copy()

        s1 = har_rolling_forecast_single(
            d,
            target_col="rv1_var",
            refit_every=refit_every,
            min_train=min_train
        )

        s5 = har_rolling_forecast_single(
            d,
            target_col="rv5_var",
            refit_every=refit_every,
            min_train=min_train
        )

        har1_list.append(s1.rename(tkr))
        har5_list.append(s5.rename(tkr))

    har1_all = pd.concat(har1_list, axis=0).sort_index()
    har5_all = pd.concat(har5_list, axis=0).sort_index()

    out["har1_var"] = har1_all
    out["har5_var"] = har5_all

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)

def qlike_loss(y_true, y_pred, eps=1e-12):
    """
    Mean QLIKE loss for variance forecasts.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)

    return np.mean(np.log(y_pred) + y_true / y_pred)


def optimize_ensemble_weight(
    df,
    model_a_col,
    model_b_col,
    target_col,
    start="2005-01-01",
    end="2006-12-31",
    weight_grid=None
):
    """
    Optimize ensemble weight w in:

        ensemble = w * model_a + (1 - w) * model_b

    by minimizing QLIKE on a development window.
    """
    if weight_grid is None:
        weight_grid = np.linspace(0.0, 1.0, 101)

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    # keep only development window
    d = d[
        (d["date"] >= pd.to_datetime(start)) &
        (d["date"] <= pd.to_datetime(end))
    ].copy()

    # keep only rows where both models and target exist
    d = d.dropna(subset=[model_a_col, model_b_col, target_col]).copy()

    best_w = None
    best_loss = np.inf
    rows = []

    for w in weight_grid:
        ens = w * d[model_a_col].values + (1.0 - w) * d[model_b_col].values
        loss = qlike_loss(d[target_col].values, ens)

        rows.append({"w": w, "qlike": loss})

        if loss < best_loss:
            best_loss = loss
            best_w = w

    res_df = pd.DataFrame(rows)
    return best_w, best_loss, res_df



def add_weighted_ensemble(df, w1, w5):
    """
    Add optimized ensemble forecasts:
      ens1_var = w1 * gjr1_var + (1-w1) * har1_var
      ens5_var = w5 * gjr5_var + (1-w5) * har5_var
    """
    out = df.copy()

    out["ens1_var"] = w1 * out["gjr1_var"] + (1.0 - w1) * out["har1_var"]
    out["ens5_var"] = w5 * out["gjr5_var"] + (1.0 - w5) * out["har5_var"]

    return out

import plotly.express as px
def plot_weight_search_both(w1_table, w5_table):
    """
    Compare 1-day and 5-day ensemble weight searches on the same figure.
    """
    d1 = w1_table.copy()
    d1["horizon"] = "1-day"

    d5 = w5_table.copy()
    d5["horizon"] = "5-day"

    d = pd.concat([d1, d5], axis=0)

    fig = px.line(
        d, x="w", y="qlike", color="horizon", markers=True,
        title="Ensemble Weight Search: 1-Day vs 5-Day"
    )
    fig.update_layout(
        xaxis_title="Weight on GJR",
        yaxis_title="Development-window QLIKE"
    )
    fig.show()
