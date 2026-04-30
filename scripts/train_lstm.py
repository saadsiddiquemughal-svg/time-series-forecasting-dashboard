import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import src.load_df as load_df
import joblib
import json
from pathlib import Path
from src.config import ARTIFACTS_DIR
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model

train_data_start = pd.to_datetime( '2019-01-01 00:00:00+00:00',utc=True)
train_data_end = pd.to_datetime('2025-01-31 23:00:00+00:00',utc=True)


df = load_df.load_data()

df_train = load_df.train_df(df,train_data_start,train_data_end)

target = ['price']
scaler_price = ColumnTransformer(transformers=[('price_scaler',MinMaxScaler(feature_range=(-1,1)),target)],remainder='drop')
scaling_pipeline = Pipeline([('scaler',scaler_price)])

scaled_train = scaling_pipeline.fit_transform(df_train)

X_train, y_train = [], []
lookback = 24

for i in range(lookback, len(scaled_train)-lookback):
    X_train.append(scaled_train[i-lookback:i])
    y_train.append(scaled_train[i:i+lookback, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(lookback)  # predicting 24 hours (1 day) at once
])

model_lstm.compile(
    optimizer="adam",
    loss="mse"
)

# Train the model
model = model_lstm.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1
)

ART_DIR = Path("artifacts") / "lstm"
ART_DIR.mkdir(parents=True, exist_ok=True)

# Save model
model_lstm.save(ART_DIR / "model.keras")  # ✅ best practice

# Save scaler pipeline
joblib.dump(scaling_pipeline, ART_DIR / "scaler.pkl")
with open(ART_DIR/"feature_cols.json","w") as f:
    json.dump(df_train.columns.to_list(),f)

print("Saved to:", ART_DIR)