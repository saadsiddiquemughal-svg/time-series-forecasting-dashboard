import json
import joblib
import xgboost as xgb
from pathlib import Path
from src.config import ARTIFACTS_DIR
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None


def load_xgb():
    model = xgb.XGBRegressor()
    model.load_model(ARTIFACTS_DIR / "xgb" / "model.json")

    with open(ARTIFACTS_DIR / "xgb" / "feature_cols.json") as f:
        features = json.load(f)

    return model, features


def load_sarimax():
    model = joblib.load(ARTIFACTS_DIR / "sarimax" / "model.pkl")

    with open(ARTIFACTS_DIR / "sarimax" / "feature_cols.json") as f:
        features = json.load(f)

    return model, features
def load_lstm():
    if load_model is None:
        return None
    model = load_model(ARTIFACTS_DIR / "lstm" / "model.keras")

    scaler = joblib.load(ARTIFACTS_DIR / "lstm" / "scaler.pkl")


    with open(ARTIFACTS_DIR / "lstm" / "feature_cols.json") as f:
        features = json.load(f)

    return model, scaler, features