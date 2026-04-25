import numpy as np
import pandas as pd
from datetime import timedelta
from config import US_HOLIDAYS, INDIA_HOLIDAYS, XGBOOST_AVAILABLE, TIREX_AVAILABLE

if XGBOOST_AVAILABLE:
    import xgboost as xgb
if TIREX_AVAILABLE:
    from tirex_utils import get_tirex_prediction


def _build_hourly_row(ts, temp_df):
    return {
        "Hour": ts.hour, "DayOfWeek": ts.dayofweek, "Month": ts.month,
        "DayOfYear": ts.dayofyear, "IsWeekend": int(ts.dayofweek >= 5),
        "IsHoliday": (int(ts.date() in US_HOLIDAYS) if US_HOLIDAYS else 0),
        "Lag_1h":       temp_df["Load"].iloc[-1],
        "Lag_24h":      temp_df["Load"].iloc[-24]  if len(temp_df)>=24  else temp_df["Load"].iloc[-1],
        "Lag_168h":     temp_df["Load"].iloc[-168] if len(temp_df)>=168 else temp_df["Load"].iloc[-1],
        "RollMean_3h":  temp_df["Load"].tail(3).mean(),
        "RollMean_24h": temp_df["Load"].tail(24).mean(),
        "RollStd_24h":  temp_df["Load"].tail(24).std(),
    }

def iterative_future_forecast_xgboost(df_features, model, scaler, hours_ahead=168):
    temp_df = df_features.copy()
    future_dates = pd.date_range(start=temp_df.index[-1]+timedelta(hours=1),
                                 periods=hours_ahead, freq='H')
    preds = []
    for ts in future_dates:
        X = pd.DataFrame([_build_hourly_row(ts, temp_df)])
        X_sc = scaler.transform(X)
        pred = model.predict(xgb.DMatrix(X_sc))[0]
        preds.append(pred); temp_df.loc[ts, "Load"] = pred
    return future_dates, np.array(preds)

def iterative_future_forecast_sklearn(df_features, model, scaler, hours_ahead=168):
    temp_df = df_features.copy()
    future_dates = pd.date_range(start=temp_df.index[-1]+timedelta(hours=1),
                                 periods=hours_ahead, freq='H')
    preds = []
    for ts in future_dates:
        X = pd.DataFrame([_build_hourly_row(ts, temp_df)])
        X_sc = scaler.transform(X)
        pred = float(model.predict(X_sc)[0])
        preds.append(pred); temp_df.loc[ts, "Load"] = pred
    return future_dates, np.array(preds)

def iterative_future_forecast_lstm_us(df_clean, model, scaler_lstm, lookback=24, hours_ahead=168):
    scaled = scaler_lstm.transform(df_clean[['Load']].values)
    seq = scaled[-lookback:].reshape(1, lookback, 1)
    future_dates = pd.date_range(start=df_clean.index[-1]+timedelta(hours=1),
                                 periods=hours_ahead, freq='H')
    preds = []
    for _ in range(hours_ahead):
        p = model.predict(seq, verbose=0)[0,0]
        preds.append(p)
        seq = np.append(seq[:,1:,:], [[[p]]], axis=1)
    return future_dates, scaler_lstm.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

def _build_daily_row(ts, temp_df):
    load_history = temp_df["Load"]
    row = {
        "Year":      ts.year,
        "DayOfWeek": ts.dayofweek,
        "Month":     ts.month,
        "DayOfYear": ts.dayofyear,
        "IsWeekend": int(ts.dayofweek >= 5),
        "Season":    ts.month % 12 // 3,
        "TimeIndex": len(temp_df),
        "IsIndianHoliday": (int(ts.date() in INDIA_HOLIDAYS) if INDIA_HOLIDAYS else 0),
        
        # Cyclic
        "Month_Sin": np.sin(2 * np.pi * ts.month / 12),
        "Month_Cos": np.cos(2 * np.pi * ts.month / 12),
        "Day_Sin":   np.sin(2 * np.pi * ts.dayofweek / 7),
        "Day_Cos":   np.cos(2 * np.pi * ts.dayofweek / 7),
        
        # Lags
        "Lag_1d":  load_history.iloc[-1],
        "Lag_2d":  load_history.iloc[-2]  if len(load_history)>=2  else load_history.iloc[-1],
        "Lag_7d":  load_history.iloc[-7]  if len(load_history)>=7  else load_history.iloc[-1],
        "Lag_14d": load_history.iloc[-14] if len(load_history)>=14 else load_history.iloc[-1],
        "Lag_30d": load_history.iloc[-30] if len(load_history)>=30 else load_history.iloc[-1],
        
        # Rolling
        "RollMean_3d": load_history.tail(3).mean(),
        "RollMean_7d": load_history.tail(7).mean(),
        "RollStd_7d":  load_history.tail(7).std(),
        "RollMax_7d":  load_history.tail(7).max(),
        "RollMin_7d":  load_history.tail(7).min(),
        
        # EWM
        "EWM_7d": load_history.ewm(span=7, adjust=False).mean().iloc[-1],
        
        # Diff Features (Momentum)
        "Load_Diff_1d": load_history.iloc[-1] - (load_history.iloc[-2] if len(load_history)>=2 else load_history.iloc[-1]),
        "Load_Diff_7d": (load_history.iloc[-7] if len(load_history)>=7 else load_history.iloc[-1]) - \
                        (load_history.iloc[-8] if len(load_history)>=8 else load_history.iloc[-1]),

        # Interactions
        "Vol_Interaction": load_history.tail(7).std() / (load_history.tail(7).mean() + 1e-5),
        "Trend_6d": load_history.iloc[-1] - (load_history.iloc[-7] if len(load_history)>=7 else load_history.iloc[-1])
    }
    return row

def iterative_future_forecast_xgboost_daily(df_features, model, scaler, days_ahead=30):
    temp_df = df_features.copy()
    future_dates = pd.date_range(start=temp_df.index[-1]+timedelta(days=1),
                                 periods=days_ahead, freq='D')
    preds = []
    for ts in future_dates:
        X = pd.DataFrame([_build_daily_row(ts, temp_df)])
        X_sc = scaler.transform(X)
        pred = model.predict(xgb.DMatrix(X_sc))[0]
        preds.append(pred); temp_df.loc[ts, "Load"] = pred
    return future_dates, np.array(preds)

def iterative_future_forecast_sklearn_daily(df_features, model, scaler, days_ahead=30):
    temp_df = df_features.copy()
    future_dates = pd.date_range(start=temp_df.index[-1]+timedelta(days=1),
                                 periods=days_ahead, freq='D')
    preds = []
    for ts in future_dates:
        X = pd.DataFrame([_build_daily_row(ts, temp_df)])
        X_sc = scaler.transform(X)
        pred = float(model.predict(X_sc)[0])
        preds.append(pred); temp_df.loc[ts, "Load"] = pred
    return future_dates, np.array(preds)

def iterative_future_forecast_lstm_daily(df_clean, model, scaler_lstm, lookback=30, days_ahead=30):
    scaled = scaler_lstm.transform(df_clean[['Load']].values)
    seq = scaled[-lookback:].reshape(1, lookback, 1)
    future_dates = pd.date_range(start=df_clean.index[-1]+timedelta(days=1),
                                 periods=days_ahead, freq='D')
    preds = []
    for _ in range(days_ahead):
        p = model.predict(seq, verbose=0)[0,0]
        preds.append(p)
        seq = np.append(seq[:,1:,:], [[[p]]], axis=1)
    return future_dates, scaler_lstm.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

def iterative_future_forecast_cnn_lstm_daily(df_st, df_nat, model, scaler_st, scaler_nat, lookback=30, days_ahead=30):
    scaled_st = scaler_st.transform(df_st[['Load']].values)
    scaled_nat = scaler_nat.transform(df_nat[['Load']].values)
    
    seq_st = scaled_st[-lookback:].reshape(1, lookback, 1)
    seq_nat = scaled_nat[-lookback:].reshape(1, lookback, 1)
    
    future_dates = pd.date_range(start=df_st.index[-1]+timedelta(days=1),
                                 periods=days_ahead, freq='D')
    preds = []
    
    for _ in range(days_ahead):
        p = model.predict([seq_st, seq_nat], verbose=0)[0,0]
        preds.append(p)
        # Update sequences (simulating future national as last value or ideally we should have nat forecast)
        seq_st = np.append(seq_st[:,1:,:], [[[p]]], axis=1)
        # For simplicity, we repeat its last value or we could pass a pre-computed nat_future
        last_nat = seq_nat[:,-1:,:] 
        seq_nat = np.append(seq_nat[:,1:,:], last_nat, axis=1)
        
    return future_dates, scaler_st.inverse_transform(np.array(preds).reshape(-1,1)).flatten()



def future_forecast_tirex(df, steps=30):
    """Generate future forecast using TiRex."""
    preds = get_tirex_prediction(df['Load'], steps)
    
    freq = 'D' if len(df) < 5000 else 'H' # Simple heuristic based on data grain
    future_dates = pd.date_range(start=df.index[-1] + (timedelta(days=1) if freq=='D' else timedelta(hours=1)),
                                 periods=len(preds), freq=freq)
    return future_dates, preds

def iterative_future_forecast_dlinear_us(df_clean, model, scaler_lstm, lookback=24, hours_ahead=168):
    # DLinear US (Hourly) - logic is identical to LSTM iterative
    scaled = scaler_lstm.transform(df_clean[['Load']].values)
    seq = scaled[-lookback:].reshape(1, lookback, 1)
    future_dates = pd.date_range(start=df_clean.index[-1]+timedelta(hours=1),
                                 periods=hours_ahead, freq='H')
    preds = []
    for _ in range(hours_ahead):
        p = model.predict(seq, verbose=0)[0,0]
        preds.append(p)
        seq = np.append(seq[:,1:,:], [[[p]]], axis=1)
    return future_dates, scaler_lstm.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

def iterative_future_forecast_dlinear_daily(df_clean, model, scaler_lstm, lookback=30, days_ahead=30):
    # DLinear India (Daily)
    scaled = scaler_lstm.transform(df_clean[['Load']].values)
    seq = scaled[-lookback:].reshape(1, lookback, 1)
    future_dates = pd.date_range(start=df_clean.index[-1]+timedelta(days=1),
                                 periods=days_ahead, freq='D')
    preds = []
    for _ in range(days_ahead):
        p = model.predict(seq, verbose=0)[0,0]
        preds.append(p)
        seq = np.append(seq[:,1:,:], [[[p]]], axis=1)
    return future_dates, scaler_lstm.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
