import json
import joblib
import xgboost as xgb
from pathlib import Path
from src.config import ARTIFACTS_DIR
from tensorflow.keras.models import load_model


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
    model = load_model(ARTIFACTS_DIR / "lstm" / "model.keras")

    scaler = joblib.load(ARTIFACTS_DIR / "lstm" / "scaler.pkl")


    with open(ARTIFACTS_DIR / "lstm" / "feature_cols.json") as f:
        features = json.load(f)

    return model, scaler, features