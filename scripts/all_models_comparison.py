import sys
from pathlib import Path

# ---- project root ---- ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import src.load_df as load_df
from src.model_loader import load_xgb, load_sarimax, load_lstm
from src.lstm_forecast import forecast_lstm_multi_horizon
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
def main():
    # ---------------- Settings ----------------
    H = 24
    lookback = 24

    # Forecast start time (your test start)
    start_date = pd.to_datetime("2025-02-01 00:00:00+00:00")

    # ---------------- Load data ----------------
    df = load_df.load_data()  # full df for LSTM history
    df_test = load_df.test_df(df, start_date, df.index.max())

    window = df_test.head(H)
    if len(window) < H:
        raise ValueError(f"Not enough test data for {H} steps. Got {len(window)} rows.")

    y_true = window["price"]

    # ---------------- Load models ----------------
    xgb_model, xgb_feats = load_xgb()
    sarimax_model, sarimax_feats = load_sarimax()
    lstm_model, lstm_scaler, _ = load_lstm()

    # ---------------- Predict: XGBoost ----------------
    xgb_pred = xgb_model.predict(window[xgb_feats])
    xgb_pred = pd.Series(xgb_pred, index=window.index, name="XGBoost")

    # ---------------- Predict: SARIMAX ----------------
    sarimax_fc = sarimax_model.get_forecast(steps=H, exog=window[sarimax_feats])
    sarimax_pred = sarimax_fc.predicted_mean
    sarimax_pred.index = window.index
    sarimax_pred.name = "SARIMAX"

    # ---------------- Predict: LSTM (needs history) ----------------
    start_time = window.index[0]
    hist_end = start_time - pd.Timedelta(hours=1)
    history_df = df.loc[:hist_end].tail(lookback)

    if len(history_df) < lookback:
        raise ValueError(
            f"Not enough history for LSTM. Need {lookback} rows before {start_time}, got {len(history_df)}"
        )

    lstm_pred_arr = forecast_lstm_multi_horizon(
        model=lstm_model,
        scaling_pipeline=lstm_scaler,
        history_df=history_df,
        horizon=H,
        lookback=lookback,
        target_col="price",
    )
    lstm_pred = pd.Series(lstm_pred_arr, index=window.index, name="LSTM")

    plot_df = pd.DataFrame(
        {
            "Actual": y_true,
            "XGBoost": xgb_pred,
            "SARIMAX": sarimax_pred,
            "LSTM": lstm_pred,
        },
        index=window.index,
    )
    def rmse(y, yhat):
        return np.sqrt(mean_squared_error(y, yhat))

    metrics = {
    "XGBoost": {
        "rmse": rmse(plot_df["Actual"], plot_df["XGBoost"]),
        "mae": mean_absolute_error(plot_df["Actual"], plot_df["XGBoost"])
    },
    "SARIMAX": {
        "rmse": rmse(plot_df["Actual"], plot_df["SARIMAX"]),
        "mae": mean_absolute_error(plot_df["Actual"], plot_df["SARIMAX"])
    },
    "LSTM": {
        "rmse": rmse(plot_df["Actual"], plot_df["LSTM"]),
        "mae": mean_absolute_error(plot_df["Actual"], plot_df["LSTM"])
    }   
}

    fig, ax = plt.subplots(figsize=(13, 6))

  
    ax.plot(plot_df.index, plot_df["Actual"],label="Actual",color="black",linewidth=3.0)
    ax.plot(plot_df.index, plot_df["XGBoost"], label="XGBoost", linewidth=1.8, alpha=0.85)
    ax.plot(plot_df.index, plot_df["SARIMAX"], label="SARIMAX", linewidth=1.8, alpha=0.75)
    ax.plot(plot_df.index, plot_df["LSTM"], label="LSTM", linewidth=1.8, alpha=0.65)

   

    ax.set_title("24-hour Forecast (01-Feb-2025)", fontsize=16, pad=12)
    ax.set_xlabel("Time (UTC)", fontsize=12)
    ax.set_ylabel("Electricity Price", fontsize=12)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.tick_params(axis="both", labelsize=12)

  
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        frameon=False,
        fontsize=10
    )
   
    fig.tight_layout()

    out_path = PROJECT_ROOT / "artifacts"/ "all_models.png"

    fig.savefig(out_path, dpi=250, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()