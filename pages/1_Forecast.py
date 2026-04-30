import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from src.config import DATA_DIR
from src.model_loader import load_xgb, load_sarimax ,load_lstm
import src.load_df as load_df
from src.lstm_forecast import forecast_lstm_multi_horizon
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="⚡ Forecasting",
    layout="wide"
)

# ---------------- CSS DECORATION ----------------
st.markdown("""
<style>

.block-container {
    padding-top: 1.2rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, rgba(28,126,214,.15), rgba(255,146,43,.15));
    border-radius: 16px;
    padding: 18px 22px;
    border: 1px solid rgba(220,220,220,0.7);
}

.card {
    border-radius: 14px;
    padding: 15px;
    border: 1px solid rgba(220,220,220,0.7);
    background: rgba(255,255,255,0.75);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <h2>⚡ Electricity Price Forecasting</h2>
    Generate short-term price forecasts and evaluate model performance.
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- METRIC FUNCTION ----------------
@st.cache_data
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------------- LOAD DATA ----------------
df = load_df.load_data()
start_date = pd.to_datetime('2025-02-01 00:00:00+00:00')
end_date = df.index.max()
df_test = load_df.test_df(df, start_date, end_date)

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("⚙ Forecast Settings")

model_name = st.sidebar.selectbox(
    "Model",
    ["XGBoost", "SARIMAX", "LSTM"]
)

horizon_label = st.sidebar.selectbox(
    f"Horizon (from {start_date.date()})",
    ["Next 24 hours", "Next 48 hours", "Next 1 week", "Next 1 month"]
)

horizon_map = {
    "Next 24 hours": 24,
    "Next 48 hours": 48,
    "Next 1 week": 24 * 7,
    "Next 1 month": 24 * 30
}
H = horizon_map[horizon_label]

run = st.sidebar.button("🚀 Run Forecast")

# ---------------- MAIN AREA ----------------
if run:

    window = df_test.head(H)

    if len(window) < H:
        st.error(f"Not enough test data available ({len(window)} rows).")
        st.stop()

    y_true = window["price"]

    # ----- MODEL PREDICTION -----
    if model_name == "XGBoost":
        model, feats = load_xgb()
        y_pred = model.predict(window[feats])
        y_pred = pd.Series(y_pred, index=window.index, name="pred")

    elif model_name=="SARIMAX":
        model, feats = load_sarimax()
        fc = model.get_forecast(steps=H, exog=window[feats])
        y_pred = fc.predicted_mean
        y_pred.index = window.index
        y_pred.name = "pred"
    else:

        # ----- LSTM PREDICTION -----
        model_lstm, scaling_pipeline, _ = load_lstm()
        lookback = 24
        anchor_time = window.index[0]  # forecast starts here
        hist_end = anchor_time - pd.Timedelta(hours=1)

        history_df = df.loc[:hist_end].tail(lookback)

        if len(history_df) < lookback:
            st.error(
                f"Not enough history for LSTM. Need {lookback} hours before {anchor_time}, "
                f"but got {len(history_df)}."
            )
            st.stop()

        preds = forecast_lstm_multi_horizon(
            model=model_lstm,
            scaling_pipeline=scaling_pipeline,
            history_df=history_df,
            horizon=H,
            lookback=lookback,
            target_col="price",
        )

        y_pred = pd.Series(preds, index=window.index, name="pred")
    score = rmse(y_true, y_pred)

    # ---------------- KPI ROW ----------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Model", model_name)
    k2.metric("Forecast Horizon", horizon_label)
    k3.metric("RMSE", f"{score:.2f}")

    st.write("")

    plot_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })

    # ---------------- CHART + TABLE ----------------
    c1, c2 = st.columns([2.2, 1])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Actual vs Forecast")
        st.line_chart(plot_df)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Values")
        st.dataframe(plot_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Select settings in the sidebar and run forecast.")