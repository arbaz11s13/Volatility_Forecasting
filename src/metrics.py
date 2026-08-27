import numpy as np
import pandas as pd

def qlike(y_true, y_pred, eps=1e-12):
    """
    QLIKE loss for variance forecasts (lower is better).
    Uses eps to avoid log/div-by-zero.
    """
    y_pred = np.maximum(y_pred, eps)
    y_true = np.maximum(y_true, eps)
    return np.log(y_pred) + (y_true / y_pred)

def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2



def score(df_f, model_cols, target_col, eval_start):
    d = df_f[df_f["date"] >= pd.to_datetime(eval_start)].copy()

    rows = []
    for m in model_cols:
        valid = d.dropna(subset=[m, target_col])
        rows.append({
            "model": m,
            "target": target_col,
            "n": len(valid),
            "QLIKE": qlike(valid[target_col].values, valid[m].values).mean(),
            "MSE": mse(valid[target_col].values, valid[m].values).mean(),
        })
    return pd.DataFrame(rows).sort_values("QLIKE")

def score_by_regime(df_f, model_cols, target_col, eval_start):
    d = df_f[df_f["date"] >= pd.to_datetime(eval_start)].copy()
    out = []
    for regime, g in d.groupby("regime"):
        for m in model_cols:
            valid = g.dropna(subset=[m, target_col])
            if len(valid) == 0:
                continue
            out.append({
                "regime": regime,
                "model": m,
                "target": target_col,
                "n": len(valid),
                "QLIKE": qlike(valid[target_col].values, valid[m].values).mean(),
                "MSE": mse(valid[target_col].values, valid[m].values).mean(),
            })
    return pd.DataFrame(out).sort_values(["target","regime","QLIKE"])

def score_by_ticker(df_f, model_cols, target_col, eval_start):
    d = df_f[df_f["date"] >= pd.to_datetime(eval_start)].copy()
    out = []
    for ticker, g in d.groupby("ticker"):
        for m in model_cols:
            valid = g.dropna(subset=[m, target_col])
            out.append({
                "ticker": ticker,
                "model": m,
                "target": target_col,
                "n": len(valid),
                "QLIKE": qlike(valid[target_col].values, valid[m].values).mean(),
                "MSE": mse(valid[target_col].values, valid[m].values).mean(),
            })
    return pd.DataFrame(out).sort_values(["ticker","target","QLIKE"])