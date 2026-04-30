import numpy as np
import pandas as pd

def forecast_lstm_multi_horizon(
    model,
    scaling_pipeline,
    history_df: pd.DataFrame,
    horizon: int,
    lookback: int = 24,
    target_col: str = "price",
):
    """
    model: your trained Keras model with Dense(lookback) output
    scaling_pipeline: your sklearn Pipeline(ColumnTransformer(MinMaxScaler))
    history_df: dataframe containing at least the last lookback rows of target_col
    horizon: how many future steps to predict (e.g. 24, 48, 168, 720)
    """

    if len(history_df) < lookback:
        raise ValueError(f"Need at least {lookback} rows in history_df, got {len(history_df)}")

    # Use only the target column, keep as DataFrame for pipeline
    hist = history_df[[target_col]].copy()

    # Scale whole history (using train-fitted scaler)
    hist_scaled = scaling_pipeline.transform(hist)  # shape (n,1)

    # We will iteratively predict in blocks of `lookback`
    preds_scaled_all = []

    # Start window: last lookback points
    window = hist_scaled[-lookback:].reshape(1, lookback, 1)

    steps_remaining = horizon
    while steps_remaining > 0:
        # Predict next block of 24 (or lookback) steps
        block_scaled = model.predict(window, verbose=0).reshape(-1)  # shape (lookback,)
        take = min(steps_remaining, lookback)

        preds_scaled_all.extend(block_scaled[:take].tolist())
        steps_remaining -= take

        # Slide window forward by appending the *full* predicted block
        # so the next call can generate the next block
        window_seq = window.reshape(lookback, 1)
        new_seq = np.vstack([window_seq, block_scaled.reshape(-1, 1)])  # (lookback+lookback,1)
        window = new_seq[-lookback:].reshape(1, lookback, 1)

    preds_scaled_all = np.array(preds_scaled_all).reshape(-1, 1)

    # Inverse scale back to original price
    price_scaler = (
        scaling_pipeline.named_steps["scaler"]
        .named_transformers_["price_scaler"]
    )
    preds = price_scaler.inverse_transform(preds_scaled_all).reshape(-1)

    return preds