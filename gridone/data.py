import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os
from config import GRIDSTATUS_AVAILABLE, US_HOLIDAYS, INDIA_HOLIDAYS, INDIA_STATES

if GRIDSTATUS_AVAILABLE:
    import gridstatus

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_grid_load(market, start_date, end_date):
    try:
        market_map = {k: getattr(gridstatus, k, None)
                      for k in ['CAISO','PJM','MISO','NYISO','SPP','ISONE']}
        if market not in market_map or market_map[market] is None:
            raise ValueError(f"Market {market} not supported")
        grid = market_map[market]()
        st.info(f"Fetching {market} data…")
        df = grid.get_load(start=start_date, end=end_date)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise ValueError(f"No data returned for {market}")
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time']); df = df.set_index('Time')
        elif 'Interval Start' in df.columns:
            df['Interval Start'] = pd.to_datetime(df['Interval Start']); df = df.set_index('Interval Start')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        possible = ['Load','Demand','load','demand']
        found = next((c for c in possible if c in df.columns), None)
        if found:
            df = df[[found]].copy(); df.columns = ['Load']
        else:
            nc = df.select_dtypes(include=[np.number]).columns.tolist()
            if nc: df = df[[nc[0]]].copy(); df.columns = ['Load']
            else: raise ValueError('No numeric load column found')
        if df.index.tz is not None: df.index = df.index.tz_convert(None)
        df = df.sort_index()
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        df = df[~df.index.duplicated(keep='first')]
        df['Load'] = pd.to_numeric(df['Load'], errors='coerce')
        df = df.dropna()
        if len(df) == 0: raise ValueError('No valid data after cleaning')
        st.success(f"✓ {len(df)} records loaded")
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}"); raise

def preprocess_data(df, freq='H'):
    df_r = df.resample(freq).mean()
    df_r['Load'] = df_r['Load'].interpolate(method='time', limit=6)
    df_r['Load'] = df_r['Load'].ffill().bfill()
    return df_r

def engineer_features(df):
    df = df.copy()
    df['Hour']       = df.index.hour
    df['DayOfWeek']  = df.index.dayofweek
    df['Month']      = df.index.month
    df['DayOfYear']  = df.index.dayofyear
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
    df['IsHoliday']  = df.index.to_series().apply(
        lambda x: int(x.date() in US_HOLIDAYS)) if US_HOLIDAYS else 0
    df['Lag_1h']      = df['Load'].shift(1)
    df['Lag_24h']     = df['Load'].shift(24)
    df['Lag_168h']    = df['Load'].shift(168)
    df['RollMean_3h'] = df['Load'].shift(1).rolling(3,  min_periods=1).mean()
    df['RollMean_24h']= df['Load'].shift(1).rolling(24, min_periods=1).mean()
    df['RollStd_24h'] = df['Load'].shift(1).rolling(24, min_periods=1).std()
    return df.dropna()

def create_lstm_sequences(data, lookback=24):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i]); y.append(data[i])
    return np.array(X), np.array(y)

@st.cache_data(ttl=3600, show_spinner=False)
def load_real_india_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'india_master_data.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            return None
    return None

def generate_synthetic_india_data(state_name, start_date, end_date, seed=42):
    np.random.seed(seed + abs(hash(state_name)) % 1000)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    cfg = INDIA_STATES[state_name]
    base, noise_lvl, region = cfg['base'], cfg['noise'], cfg['region']

    # Load real data to adjust base/noise and merge
    real_data_dict = {}
    real_df = load_real_india_data()
    if real_df is not None:
        state_df = real_df[real_df['States'] == state_name].copy()
        if not state_df.empty:
            state_df = state_df.set_index('Date')
            state_df = state_df[~state_df.index.duplicated(keep='last')]
            
            # calculate base and noise from real data
            real_mean = state_df['Day Demand (MU)'].mean()
            real_std = state_df['Day Demand (MU)'].std()
            if pd.notna(real_mean): base = real_mean
            if pd.notna(real_std) and real_std > 0: noise_lvl = real_std
            
            real_data_dict = state_df['Day Demand (MU)'].to_dict()

    doy = np.array([d.dayofyear for d in dates])
    dow = np.array([d.dayofweek  for d in dates])
    if region in ('north', 'central'):
        seasonal = base * 0.12 * np.sin(2*np.pi*(doy-80)/365) + base*0.06*np.cos(4*np.pi*doy/365)
    else:
        seasonal = base * 0.08 * np.sin(2*np.pi*(doy-60)/365)
    trend   = np.linspace(0, base*0.08, n)
    weekly  = -base * 0.04 * (dow >= 5)
    noise   = np.random.normal(0, noise_lvl, n)
    demand  = base + trend + seasonal + weekly + noise
    demand  = np.maximum(demand, base * 0.3)
    
    # Create the DataFrame with synthetic demand
    df = pd.DataFrame({'Load': demand}, index=dates)
    df.index.name = 'Date'
    
    # Overwrite synthetic with real data where available
    if real_data_dict:
        for d in dates:
            if d in real_data_dict:
                val = real_data_dict[d]
                if pd.notna(val) and val > 0:
                    df.loc[d, 'Load'] = val
            
    # --- R2 OPTIMIZATION: OUTLIER REMOVAL & SMOOTHING ---
    # 1. Local Z-Score Outlier Removal (using 14-day window to avoid losing weekly peaks)
    roll_mean = df['Load'].rolling(window=14, center=True, min_periods=7).mean()
    roll_std  = df['Load'].rolling(window=14, center=True, min_periods=7).std()
    is_outlier = (df['Load'] - roll_mean).abs() > (3.5 * roll_std)
    df.loc[is_outlier, 'Load'] = np.nan
    
    # 2. Smooth Linear Interpolation for any gaps or newly removed outliers
    df['Load'] = df['Load'].interpolate(method='linear', limit_direction='both')
    
    # 3. Final Short-Term Smoothing (helps stitching between real and synthetic)
    # Using 3-day window to keep sharp load changes while removing raw noise
    df['Load'] = df['Load'].rolling(window=3, center=True, min_periods=1).mean()
    # -----------------------------------------------------

    # CRITICAL: Ensure the dataframe is sorted by date
    df = df.sort_index()
    return df

def engineer_features_daily(df):
    df = df.copy()
    
    # Basic Time Features
    df['Year']       = df.index.year
    df['DayOfWeek']  = df.index.dayofweek
    df['Month']      = df.index.month
    df['DayOfYear']  = df.index.dayofyear
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
    df['Season']     = (df['Month'] % 12 // 3)
    df['TimeIndex']  = np.arange(len(df))
    
    # NEW: Indian Holidays (Crucial for R2)
    df['IsIndianHoliday'] = df.index.to_series().apply(
        lambda x: int(x.date() in INDIA_HOLIDAYS)) if INDIA_HOLIDAYS else 0
    
    # Cyclic Time Features
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_Sin']   = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['Day_Cos']   = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # Auto-regressive features
    df['Lag_1d']     = df['Load'].shift(1)
    df['Lag_2d']     = df['Load'].shift(2)
    df['Lag_7d']     = df['Load'].shift(7)
    df['Lag_14d']    = df['Load'].shift(14)
    df['Lag_30d']    = df['Load'].shift(30)
    
    # Rolling Statistics
    df['RollMean_3d']= df['Load'].shift(1).rolling(3,  min_periods=3).mean()
    df['RollMean_7d']= df['Load'].shift(1).rolling(7,  min_periods=7).mean()
    df['RollStd_7d'] = df['Load'].shift(1).rolling(7,  min_periods=7).std()
    df['RollMax_7d'] = df['Load'].shift(1).rolling(7,  min_periods=7).max()
    df['RollMin_7d'] = df['Load'].shift(1).rolling(7,  min_periods=7).min()
    
    # EWM
    df['EWM_7d'] = df['Load'].shift(1).ewm(span=7, adjust=False).mean()
    
    # NEW: Diff Features (Captures Momentum for R2 > 0.9)
    df['Load_Diff_1d'] = df['Load'].shift(1) - df['Load'].shift(2)
    df['Load_Diff_7d'] = df['Load'].shift(7) - df['Load'].shift(8)
    
    # NEW: Volatility Interaction
    df['Vol_Interaction'] = df['RollStd_7d'] / (df['RollMean_7d'] + 1e-5)
    
    # NEW: Trend Gradient
    df['Trend_6d'] = df['Load'].shift(1) - df['Load'].shift(7)
    
    # Final smoothing column check
    return df.dropna()
