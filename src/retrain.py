import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import ARTIFACTS_DIR


def retrain_xgb(df_train, features, target="price"):
    """
    Retrains XGB using df_train/df_val and overwrites artifacts/xgb.
    """
    X_train = df_train[features]
    y_train = df_train[target]
    #X_val = df_val[features]
    #y_val = df_val[target]

    model_xgb = xgb.XGBRegressor(base_score=0.5,
                           n_estimators=1500,
                           learning_rate=0.01,
                           objective='reg:squarederror',
                           colsample_bytree=0.8,
                           subsample=0.8,
                           max_depth=5,
                           enable_categorical=True)

    model_xgb.fit(X_train, y_train, verbose=False)

    save_dir = ARTIFACTS_DIR / "xgb"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_xgb.save_model(save_dir / "model.json")
    (save_dir / "feature_cols.json").write_text(json.dumps(list(features)))

    meta = {
        "target": target,
        "best_iteration": int(getattr(model_xgb, "best_iteration", -1)),
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def retrain_sarimax(df_train, features, target="price",
                   order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)):
    """
    Retrains SARIMAX and overwrites artifacts/sarimax.
    Uses df_train only (typical). df_val can be used for eval, but not required.
    """
    y_train = pd.to_numeric(df_train[target], errors="coerce")
    X_train = df_train[features].apply(pd.to_numeric, errors="coerce")

    tmp = pd.concat([y_train.rename("y"), X_train], axis=1).dropna()
    y_train = tmp["y"]
    X_train = tmp.drop(columns=["y"])

    model_sarimax = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model_sarimax.fit(disp=False)

    save_dir = ARTIFACTS_DIR / "sarimax"
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(res, save_dir / "model.pkl")
    (save_dir / "feature_cols.json").write_text(json.dumps(list(features)))

    meta = {
        "target": target,
        "order": order,
        "seasonal_order": seasonal_order,
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2))