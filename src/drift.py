
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")




def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))

def drift_detection_for_model(model_name, model_obj, df, start, end, features, patch,
                              target_col="price", rmse_threshold=None, consecutive_up=3):
    """
    Patch-by-patch eval over [start, end]. Returns preds_df, rmse_df, alert, msg, last_patch_end.
    Works for XGB (predict) and SARIMAX (get_forecast).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index,utc=True)
    df = df.sort_index()

    start = to_utc(start)
    end = to_utc(end)

    needed = [target_col] + list(features)
    eval_df = df.loc[start:end, needed].dropna(subset=needed)

    patch_starts = pd.date_range(start=start, end=end, freq=patch)

    preds_parts = []
    rmses = []

    prev_rmse = None
    increasing_streak = 0
    alert = False
    msg = ""
    last_patch_end = None

    for i, p_start in enumerate(patch_starts):
        p_end = (patch_starts[i + 1] - pd.Timedelta(hours=1)) if i < len(patch_starts) - 1 else end
        patch_df = eval_df.loc[p_start:p_end]
        if patch_df.empty:
            continue

        X = patch_df[features].reindex(columns=features, fill_value=0)
        y_true = patch_df[target_col]

        if model_name == "XGBoost":
            y_pred = model_obj.predict(X)
            y_pred = pd.Series(y_pred, index=patch_df.index, name="pred")

        elif model_name=="SARIMAX": 
            fc = model_obj.get_forecast(steps=len(patch_df), exog=X)
            y_pred = fc.predicted_mean
            y_pred.index = patch_df.index
            y_pred.name = "pred"

        r = rmse(y_true.values, y_pred.values)

        preds_parts.append(pd.DataFrame({
            "actual": y_true,
            "pred": y_pred,
            "patch_rmse": r,
            "patch_start": p_start,
            "patch_end": p_end
        }, index=patch_df.index))

        rmses.append({
            "patch_start": p_start,
            "patch_end": p_end,
            "rmse": r,
            "n_obs": int(len(patch_df))
        })

        if prev_rmse is not None and r > prev_rmse:
            increasing_streak += 1
        else:
            increasing_streak = 0

        prev_rmse = r
        last_patch_end = p_end

        threshold_hit = (rmse_threshold is not None and r > rmse_threshold)
        streak_hit = (increasing_streak >= consecutive_up)

        if threshold_hit or streak_hit:
            alert = True
            msg = (
                f"⚠️ Drift alert: RMSE={r:.2f} for {p_start} → {p_end}. "
                f"Increasing streak={increasing_streak}."
            )
            break

    preds_df = pd.concat(preds_parts).sort_index() if preds_parts else pd.DataFrame()
    rmse_df = pd.DataFrame(rmses)
    return preds_df, rmse_df, alert, msg, last_patch_end

