import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import src.load_df as load_df
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error 
import joblib
import json
from pathlib import Path
from src.config import ARTIFACTS_DIR


train_data_start = pd.to_datetime( '2019-01-01 00:00:00+00:00')
train_data_end = pd.to_datetime('2025-01-31 23:00:00+00:00')



df = load_df.load_data()
df_train = load_df.train_df(df,train_data_start,train_data_end)


eval_data_start = train_data_end +pd.Timedelta(hours=1)
eval_data_end = eval_data_start + pd.DateOffset(months=3)-pd.Timedelta(hours=1)
df_eval = load_df.test_df(df,eval_data_start,eval_data_end)



feature_col = ['load_forecast_da', 'res_sum_da', 'gen_forecast_da', 'dayofyear_sin1',
       'dayofyear_cos1', 'hour_sin1', 'hour_cos1', 'dayofweek_sin1',
       'dayofweek_cos1', 'is_holiday', 'price_gas', 'year_2020', 'year_2021', 'year_2022',
       'year_2023', 'year_2024', 'year_2025']
target ='price'

X_train = df_train[feature_col]
y_train = df_train[target]
x_val = df_eval[feature_col]
y_val = df_eval[target]

model_xgb = xgb.XGBRegressor(base_score=0.5,
                           n_estimators=1500,
                           learning_rate=0.01,
                           objective='reg:squarederror',
                           colsample_bytree=0.8,
                           subsample=0.8,
                           max_depth=5,
                           enable_categorical=True)
model_fitted = model_xgb.fit(X_train,y_train)

save_dir = ARTIFACTS_DIR/"xgb"
save_dir.mkdir(parents=True,exist_ok=True)
model_fitted.save_model(save_dir/"model.json")
with open(save_dir / "feature_cols.json", "w") as f:
    json.dump(list(feature_col), f)

print('model has been trained and saved')