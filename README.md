# Time Series Forecasting Dashboard

## Overview
Production-style forecasting platform for analyzing and predicting time series data using classical statistical models and deep learning approaches. The project integrates automated preprocessing, model training, evaluation, and interactive Streamlit-based visualization.

---

## Features
- Time series preprocessing pipeline
- Forecasting with SARIMAX, XGBoost, and LSTM
- Interactive Streamlit dashboard
- Model performance comparison
- Visualization of historical and predicted trends
- Modular project structure

---

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow/Keras
- Streamlit
- Matplotlib / Plotly

---

## Project Structure
```bash
.
├── artifacts/
├── assets/
├── data/
├── notebook/
├── pages/
├── scripts/
├── src/
├── app.py
├── requirements.txt
└── README.md
```

## Models Used
- SARIMAX
- XGBoost
- LSTM

---

## Results
Add your evaluation metrics here.

Example:

| Model | RMSE | MAE |
|---|---|---|
| LSTM  | 16.12 | 13.11 |
| SARIMAX | 10.48 | 9.4 |
| XGBoost | 6.25 | 5.25 |


---

## Dashboard Preview

### Main Dashboard
![Dashboard](assets/screenshots/Dashboard-%20Home.png)

### Forecast Visualization
![Forecast](assets/screenshots/Dashboard-Forecast.png)

### Drift Detection and Auto-Retraining 
![Drift Detection and Auto-Retrain](assets/screenshots/Dashboard-drift%20detection%20and%20retrain.png)
---

## Installation

```bash
git clone https://github.com/saadsiddiquemughal-svg/time-series-forecasting-dashboard.git
cd time-series-forecasting-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Improvements
- Docker deployment
- Automated retraining pipeline
- Cloud deployment
- Real-time forecasting integration