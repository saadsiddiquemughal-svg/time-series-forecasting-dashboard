import time
import streamlit as st
import pandas as pd

import src.load_df as load_df
from src.model_loader import load_xgb, load_sarimax
from src.retrain import retrain_xgb, retrain_sarimax
from src.drift import drift_detection_for_model  # rmse imported inside your drift module if needed

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Drift Monitoring", page_icon="🧭", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; max-width: 1300px;}
.hero{
  background: linear-gradient(120deg, rgba(46,196,182,.12), rgba(28,126,214,.12));
  border: 1px solid rgba(222,226,230,.95);
  border-radius: 18px;
  padding: 18px 22px;
}
.card{
  border: 1px solid rgba(222,226,230,.95);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,.75);
}
.small {color: rgba(33,37,41,.72); font-size: .95rem;}
hr{margin: 0.8rem 0 0.8rem 0; border: none; border-top: 1px solid rgba(222,226,230,.8);}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
  <div style="font-size:1.9rem; font-weight:800;">🧭 Drift Detection + Auto-Retrain</div>
  <div class="small" style="margin-top:6px;">
    Monitor model performance in rolling patches. Trigger retraining when RMSE degrades beyond threshold.
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- LOAD DATA ----------------
@st.cache_data
def get_df_all():
    df = load_df.load_data()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

df_all = get_df_all()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙ Monitoring")
    st.caption("Configure drift checks and retraining triggers.")
    st.divider()

    model_name = st.selectbox("Model", ["XGBoost", "SARIMAX"])
    patch = st.selectbox("Patch size", ["1D", "3D", "7D", "14D"], index=1)

    st.subheader("Alert Rules")
    rmse_threshold = st.number_input("RMSE threshold", value=50.0, step=1.0)
    consecutive_up = st.number_input("Consecutive increases", value=3, min_value=1, step=1)

    st.subheader("Run Limits")
    min_test_days_left = st.number_input("Stop if test left < days", value=30, min_value=1, step=1)
    max_cycles = st.number_input("Max retrain cycles", value=20, min_value=1, step=1)

    st.subheader("Monitoring Range")
    min_dt, max_dt = df_all.index.min(), df_all.index.max()
    start_date = st.date_input("Start date", value=pd.to_datetime("2025-02-01", utc=True))
    end_date = st.date_input("End date", value=max_dt.date())

    st.divider()
    run = st.button("▶ Start monitoring", use_container_width=True)

start_ts = pd.Timestamp(start_date, tz="UTC")
end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)

# ---------------- PLACEHOLDERS / LAYOUT ----------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Model", model_name)
kpi2.metric("Patch", patch)
kpi3.metric("RMSE threshold", f"{rmse_threshold:.1f}")
kpi4.metric("Consecutive ↑", int(consecutive_up))

st.write("")

left, right = st.columns([1.45, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Actual vs Predicted (live)")
    chart_pred = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Recent Patch Metrics")
    table_rmse = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📉 Patch RMSE (live)")
    chart_rmse = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Event Log (live)")
    log_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

status = st.empty()
progress = st.progress(0)

# Keep history across cycles
pred_history = []
rmse_history = []
event_log = []
retrain_count = 0

# ---------------- RUN LOOP ----------------
if run:
    # load model + features
    if model_name == "XGBoost":
        model_obj, features = load_xgb()
    else:
        model_obj, features = load_sarimax()

    start_date_test = start_ts
    end_date_test = end_ts

    # train range (you can adjust logic as you like)
    start_date_train = df_all.index.min()
    end_date_train = start_date_test - pd.Timedelta(hours=1)

    for cycle in range(int(max_cycles)):
        progress.progress(int((cycle / max_cycles) * 100))

        df_test = load_df.test_df(df_all, start_date_test, end_date_test)

        status.info(
            f"Cycle {cycle+1}/{int(max_cycles)} • "
            f"Test: {start_date_test} → {end_date_test} • "
            f"Retrains: {retrain_count}"
        )

        preds_df, rmse_df, alert, msg, p_end = drift_detection_for_model(
            model_name=model_name,
            model_obj=model_obj,
            df=df_test,
            start=start_date_test,
            end=end_date_test,
            features=features,
            patch=patch,
            target_col="price",
            rmse_threshold=rmse_threshold,
            consecutive_up=int(consecutive_up),
        )

        # history
        if not preds_df.empty:
            pred_history.append(preds_df)
        if not rmse_df.empty:
            rmse_history.append(rmse_df)

        pred_all = pd.concat(pred_history).sort_index() if pred_history else pd.DataFrame()
        rmse_all = (
            pd.concat(rmse_history)
              .drop_duplicates(subset=["patch_start", "patch_end"])
              .sort_values("patch_start")
        ) if rmse_history else pd.DataFrame()

        # charts
        if not pred_all.empty:
            N = min(len(pred_all), 24 * 14)  # last 14 days hourly
            chart_pred.line_chart(pred_all[["actual", "pred"]].tail(N))

        if not rmse_all.empty:
            chart_rmse.line_chart(rmse_all.set_index("patch_start")[["rmse"]])
            table_rmse.dataframe(rmse_all.tail(12), use_container_width=True, height=350)

        # live log rendering
        if alert:
            event_log.append({"cycle": cycle+1, "event": "ALERT", "message": msg, "patch_end": str(p_end)})
            log_box.warning(msg)
        else:
            event_log.append({"cycle": cycle+1, "event": "OK", "message": "No alert", "patch_end": str(p_end)})
            log_box.success("✅ No drift alert detected. Monitoring stopped.")
            break

        # retrain triggered
        status.warning("⚠ Retraining triggered…")
        retrain_count += 1

        end_date_train = p_end
        df_train = load_df.train_df(df_all, start_date_train, end_date_train)

        remaining = pd.Timestamp(end_date_test) - pd.Timestamp(end_date_train)
        if remaining < pd.Timedelta(days=int(min_test_days_left)):
            status.error("🛑 Not enough test data left after retraining. Stopping.")
            event_log.append({"cycle": cycle+1, "event": "STOP", "message": "Not enough test left", "patch_end": str(p_end)})
            break

        # retrain + reload
        if model_name == "XGBoost":
            retrain_xgb(df_train, features, target="price")
            model_obj, features = load_xgb()
        else:
            retrain_sarimax(df_train, features, target="price")
            model_obj, features = load_sarimax()

        # move test window forward
        start_date_test = end_date_train

        time.sleep(0.15)

    progress.progress(100)

    st.write("")
    st.subheader("📌 Summary Event Log")
    st.dataframe(pd.DataFrame(event_log), use_container_width=True)

else:
    st.info("👈 Configure settings and click **Start monitoring**.")