import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import src.load_df as load_df
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error 
import joblib
import json
from pathlib import Path
from src.config import ARTIFACTS_DIR




train_data_start = pd.to_datetime( '2019-01-01 00:00:00+00:00')
train_data_end = pd.to_datetime('2025-01-31 23:00:00+00:00')


df = load_df.load_data()
df_train = load_df.train_df(df,train_data_start,train_data_end)


target ='price'
exo_col = ["load_forecast_da", 'res_sum_da', 'gen_forecast_da',
       'dayofyear_sin1', 'dayofyear_cos1','hour_sin1', 'hour_cos1',
       'dayofweek_sin1', 'dayofweek_cos1', 'is_holiday', 'price_gas'
       ]


y_train = df_train[target]
exo_train = df_train[exo_col]


model_sarimax = SARIMAX(
    y_train,
    exog=exo_train,
    order=(1,0,1),
    seasonal_order=(1,0,1,24),
    enforce_stationarity=False,
    enforce_invertibility=False
)
print(f'training model for data period : {train_data_start} to {train_data_end}')
model = model_sarimax.fit()

save_dir = ARTIFACTS_DIR/"sarimax"
save_dir.mkdir(parents=True,exist_ok=True)

joblib.dump(model,save_dir/"model.pkl")
with open(save_dir/"feature_cols.json","w") as f:
    json.dump(exo_col,f)

print('model has been trained and saved')    