import streamlit as st
from datetime import datetime
from pathlib import Path

plot_path = Path("artifacts/all_models.png")

st.set_page_config(
    page_title="⚡ Electricity Price Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px;}
.hero{
  background: linear-gradient(120deg, rgba(28,126,214,.12), rgba(255,146,43,.12));
  border: 1px solid rgba(222,226,230,.95);
  border-radius: 18px;
  padding: 22px;
}
.card{
  border: 1px solid rgba(222,226,230,.95);
  border-radius: 16px;
  padding: 16px;
  background: rgba(255,255,255,.75);
}
.pill{
  display:inline-block; padding:6px 10px; margin-right:8px; margin-top:8px;
  border-radius:999px; border:1px solid rgba(222,226,230,.95);
  background: rgba(255,255,255,.6); font-size:.85rem;
}
.plotbox{
  border: 1px dashed rgba(134,142,150,.8);
  border-radius: 16px;
  padding: 22px;
  background: rgba(248,249,250,.7);
  text-align:center;
}
.small {color: rgba(33,37,41,.72); font-size: .95rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Sidebar (simple)
st.sidebar.title("⚡ Menu")
st.sidebar.caption("Forecasting + Drift Monitoring")
st.sidebar.divider()
st.sidebar.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Hero
st.markdown(
    """
<div class="hero">
  <div style="font-size:2rem; font-weight:800; line-height:1.1;">
    Electricity Price Forecasting & Drift Monitoring
  </div>
  <div class="small" style="margin-top:10px;">
    A lightweight dashboard to forecast day-ahead prices and detect distribution shifts that degrade model performance.
  </div>
  <div>
    <span class="pill">Models: XGBoost • SARIMAX • (LSTM optional)</span>
    <span class="pill">Monitoring: Drift • Alerts • Retraining triggers</span>
    <span class="pill">Focus: German market, high renewables</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# 3 mini cards: Goal / Why / How
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Goal")
    st.write("Forecast day-ahead electricity prices and keep forecasts reliable as market conditions change.")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 Why it matters")
    st.write("Prices are volatile due to renewables, demand swings, and fuel costs. Better forecasts support bidding, hedging, and grid operations.")
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🛠️ How we achieve it")
    st.write("Train forecasting models, track performance, and detect drift in key drivers (load, renewables, gas price) to trigger retraining.")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")



st.markdown('<div class="plotbox">', unsafe_allow_html=True)
st.markdown("### 📈 Model Forecast Comparison")

if plot_path.exists():
    st.image(str(plot_path), use_container_width=True)
else:
    st.warning("Plot not found. Run forecasting script to generate it.")

st.markdown("</div>", unsafe_allow_html=True)