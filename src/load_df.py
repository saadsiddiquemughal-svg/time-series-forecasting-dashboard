from src.config import DATA_DIR
from pathlib import Path
import pandas as pd


data_path = DATA_DIR/"clean_data.csv"


def load_data():
    df = pd.read_csv(data_path)
    df['period_start_utc']=pd.to_datetime(df['period_start_utc'],utc=True)
    df = df.set_index('period_start_utc').sort_index()
    df = pd.get_dummies(df, columns=['year'], drop_first=True)
    df["residual_load"] = (df["load_forecast_da"]- df["off_wind_da"]- df["on_wind_da"]- df["solar_da"])
    return df

def train_df(df,start_date,end_date):
    df_train = df.loc[start_date:end_date].copy() 
    return df_train
def test_df(df,start_date,end_date):
    df_test = df.loc[start_date:end_date].copy()
    return df_test
