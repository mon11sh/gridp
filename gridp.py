#!/usr/bin/env python3
"""Power Grid Load Forecasting & Anomaly Detection - MULTI-MODEL WITH FUTURE FORECASTING

Includes: XGBoost, LSTM, and Prophet with future forecasting capability
"""

# ============================================================================
# SSL FIX - MUST BE FIRST
# ============================================================================
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit
import streamlit as st

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# LSTM (TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# GridStatus
try:
    import gridstatus
    GRIDSTATUS_AVAILABLE = True
except Exception:
    GRIDSTATUS_AVAILABLE = False

# Holidays
try:
    import holidays
    US_HOLIDAYS = holidays.UnitedStates()
except Exception:
    US_HOLIDAYS = None


# ============================================================================
# DATA COLLECTION & PREPROCESSING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_grid_load(market: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical load data from GridStatus API."""
    try:
        market_map = {
            'CAISO': getattr(gridstatus, 'CAISO', None),
            'PJM': getattr(gridstatus, 'PJM', None),
            'MISO': getattr(gridstatus, 'MISO', None),
            'NYISO': getattr(gridstatus, 'NYISO', None),
            'SPP': getattr(gridstatus, 'SPP', None),
            'ISONE': getattr(gridstatus, 'ISONE', None)
        }

        if market not in market_map or market_map[market] is None:
            raise ValueError(f"Market {market} not supported")

        grid = market_map[market]()
        st.info(f"Fetching {market} data from {start_date.date()} to {end_date.date()}...")
        df = grid.get_load(start=start_date, end=end_date)

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise ValueError(f"No data returned for {market}")

        # Normalize index
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.set_index('Time')
        elif 'Interval Start' in df.columns:
            df['Interval Start'] = pd.to_datetime(df['Interval Start'])
            df = df.set_index('Interval Start')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Select load column
        possible = ['Load', 'Demand', 'load', 'demand']
        found = None
        for col in possible:
            if col in df.columns:
                found = col
                break
        if found is not None:
            df = df[[found]].copy()
            df.columns = ['Load']
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                df = df[[numeric_cols[0]]].copy()
                df.columns = ['Load']
            else:
                raise ValueError('No numeric load column found')

        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df = df.sort_index()
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        df = df[~df.index.duplicated(keep='first')]
        df['Load'] = pd.to_numeric(df['Load'], errors='coerce')
        df = df.dropna()

        if len(df) == 0:
            raise ValueError('No valid data after cleaning')

        st.success(f"✓ Final dataset: {len(df)} records")
        return df

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        raise


def preprocess_data(df: pd.DataFrame, freq='H') -> pd.DataFrame:
    """Clean and resample data."""
    df_resampled = df.resample(freq).mean()
    df_resampled['Load'] = df_resampled['Load'].interpolate(method='time', limit=6)
    df_resampled['Load'] = df_resampled['Load'].fillna(method='ffill').fillna(method='bfill')
    return df_resampled


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based and lag features for XGBoost."""
    df = df.copy()
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    if US_HOLIDAYS:
        df['IsHoliday'] = df.index.to_series().apply(lambda x: int(x.date() in US_HOLIDAYS))
    else:
        df['IsHoliday'] = 0

    df['Lag_1h'] = df['Load'].shift(1)
    df['Lag_24h'] = df['Load'].shift(24)
    df['Lag_168h'] = df['Load'].shift(168)
    df['RollMean_3h'] = df['Load'].shift(1).rolling(window=3, min_periods=1).mean()
    df['RollMean_24h'] = df['Load'].shift(1).rolling(window=24, min_periods=1).mean()
    df['RollStd_24h'] = df['Load'].shift(1).rolling(window=24, min_periods=1).std()

    df = df.dropna()
    return df


def create_lstm_sequences(data, lookback=24):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ============================================================================
# FUTURE FORECASTING FUNCTIONS
# ============================================================================

def iterative_future_forecast_xgboost(df_features, model, scaler, hours_ahead=24*7):
    """XGBoost iterative future forecasting."""
    temp_df = df_features.copy()
    future_dates = pd.date_range(start=temp_df.index[-1] + timedelta(hours=1), periods=hours_ahead, freq='H')
    preds = []

    for ts in future_dates:
        row = {
            "Hour": ts.hour,
            "DayOfWeek": ts.dayofweek,
            "Month": ts.month,
            "DayOfYear": ts.dayofyear,
            "IsWeekend": 1 if ts.dayofweek >= 5 else 0,
            "IsHoliday": 1 if (US_HOLIDAYS and ts.date() in US_HOLIDAYS) else 0,
            "Lag_1h": temp_df["Load"].iloc[-1],
            "Lag_24h": temp_df["Load"].iloc[-24] if len(temp_df) >= 24 else temp_df["Load"].iloc[-1],
            "Lag_168h": temp_df["Load"].iloc[-168] if len(temp_df) >= 168 else temp_df["Load"].iloc[-1],
            "RollMean_3h": temp_df["Load"].tail(3).mean(),
            "RollMean_24h": temp_df["Load"].tail(24).mean(),
            "RollStd_24h": temp_df["Load"].tail(24).std(),
        }
        X = pd.DataFrame([row])
        X_scaled = scaler.transform(X)
        dmat = xgb.DMatrix(X_scaled)
        pred = model.predict(dmat)[0]
        preds.append(pred)
        temp_df.loc[ts, "Load"] = pred

    return future_dates, np.array(preds)


def iterative_future_forecast_lstm(df_clean, model, scaler_lstm, lookback=24, hours_ahead=24*7):
    """LSTM iterative future forecasting."""
    scaled_data = scaler_lstm.transform(df_clean[['Load']].values)
    current_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    
    future_dates = pd.date_range(start=df_clean.index[-1] + timedelta(hours=1), periods=hours_ahead, freq='H')
    preds = []

    for _ in range(hours_ahead):
        pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        preds.append(pred_scaled)
        
        # Update sequence
        current_sequence = np.append(current_sequence[:, 1:, :], [[[pred_scaled]]], axis=1)

    preds_unscaled = scaler_lstm.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return future_dates, preds_unscaled


def future_forecast_prophet(model, hours_ahead=24*7):
    """Prophet future forecasting."""
    future = model.make_future_dataframe(periods=hours_ahead, freq='H')
    forecast = model.predict(future)
    
    # Get only future predictions
    future_forecast = forecast.tail(hours_ahead)
    future_dates = pd.to_datetime(future_forecast['ds'])
    future_preds = future_forecast['yhat'].values
    
    return future_dates, future_preds


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost regression model."""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(params, dtrain, num_boost_round=500, early_stopping_rounds=50, evals=evals, verbose_eval=False)
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50):
    """Train LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    return model, history


def train_prophet_model(df_train):
    """Train Prophet model."""
    prophet_df = df_train.reset_index()
    prophet_df.columns = ['ds', 'y']

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )

    with st.spinner("Training Prophet..."):
        model.fit(prophet_df)

    return model


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'R²': r2}


def detect_anomalies(y_true, y_pred, threshold=3):
    """Detect anomalies based on residuals."""
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    z_scores = np.abs((residuals - mean_residual) / std_residual)
    anomaly_mask = z_scores > threshold
    return anomaly_mask, residuals


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_full_timeline(train_df, test_df, train_pred, test_pred, anomalies=None):
    """Plot complete timeline."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Load'], mode='lines', name='Historical (Training)',
                            line=dict(color='lightblue', width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=train_df.index, y=train_pred, mode='lines', name='Training Fit',
                            line=dict(color='green', width=1, dash='dot'), opacity=0.5))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Load'], mode='lines', name='Actual (Test)',
                            line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_pred, mode='lines', name='Forecast (Test)',
                            line=dict(color='red', width=2, dash='dash')))

    if anomalies is not None and anomalies.any():
        fig.add_trace(go.Scatter(x=test_df.index[anomalies], y=test_df['Load'].values[anomalies],
                                mode='markers', name='Anomalies', marker=dict(color='orange', size=10, symbol='x')))

    fig.update_layout(title='Complete Timeline: Historical Data + Forecast', xaxis_title='Date',
                     yaxis_title='Load (MW)', hovermode='x unified', height=600,
                     legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def plot_future_overlay(train_df, test_df, future_predictions_dict):
    """Plot historical + test + future forecasts for all models."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Load'], mode='lines', name='Historical',
                            line=dict(color='lightblue', width=1)))

    # Test actual
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Load'], mode='lines', name='Test Actual',
                            line=dict(color='blue', width=2)))

    # Future forecasts
    colors = {'XGBoost': 'red', 'LSTM': 'green', 'Prophet': 'purple'}
    for model_name, (dates, preds) in future_predictions_dict.items():
        if dates is not None and preds is not None:
            fig.add_trace(go.Scatter(x=dates, y=preds, mode='lines', name=f'{model_name} Future',
                                    line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash')))

    fig.update_layout(title='Historical + Test + Future Forecasts (All Models)', xaxis_title='Date',
                     yaxis_title='Load (MW)', hovermode='x unified', height=650)
    return fig


def plot_model_comparison(test_df, predictions_dict):
    """Plot all models on one chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Load'], mode='lines',
                            name='Actual', line=dict(color='black', width=3)))

    colors = {'XGBoost': 'red', 'LSTM': 'blue', 'Prophet': 'green'}
    for model_name, preds in predictions_dict.items():
        if preds is not None:
            fig.add_trace(go.Scatter(x=test_df.index, y=preds, mode='lines',
                                    name=model_name, line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash')))

    fig.update_layout(title='Model Comparison: All Forecasts vs Actual', xaxis_title='Date',
                     yaxis_title='Load (MW)', hovermode='x unified', height=600)
    return fig


def plot_metrics_comparison(metrics_dict):
    """Bar chart comparing metrics across models."""
    models = list(metrics_dict.keys())
    metrics = ['MAE', 'RMSE', 'MAPE (%)', 'R²']

    fig = make_subplots(rows=2, cols=2, subplot_titles=metrics)

    for idx, metric in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1

        values = [metrics_dict[m][metric] for m in models]

        fig.add_trace(go.Bar(x=models, y=values, name=metric,
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']),
                     row=row, col=col)

        fig.update_yaxes(title_text=metric, row=row, col=col)

    fig.update_layout(height=600, showlegend=False, title_text="Metrics Comparison Across Models")
    return fig


def plot_load_history(df: pd.DataFrame):
    """Plot historical load data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Load'], mode='lines', name='Load', line=dict(color='royalblue', width=1)))
    fig.update_layout(title='Historical Load Data', xaxis_title='Date', yaxis_title='Load (MW)', hovermode='x unified', height=400)
    return fig


def plot_residuals(residuals, dates):
    """Plot residual analysis."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Residuals Over Time', 'Residual Distribution'))
    fig.add_trace(go.Scatter(x=dates, y=residuals, mode='lines', name='Residuals', line=dict(color='purple')), row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, name='Distribution', marker=dict(color='lightblue')), row=2, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Residual Value', row=2, col=1)
    fig.update_yaxes(title_text='Residual (MW)', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    fig.update_layout(height=600, showlegend=False)
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit application."""

    st.set_page_config(page_title='Grid Load Forecasting - Multi-Model', page_icon='⚡', layout='wide')
    st.title('⚡ Power Grid Load Forecasting - Multi-Model with Future Forecasting')
    st.markdown('Compare **XGBoost**, **LSTM**, and **Prophet** models with future forecasting capability.')

    # Sidebar configuration
    with st.sidebar:
        st.header('⚙️ Configuration')
        market = st.selectbox('Select Market', options=['CAISO', 'PJM', 'MISO', 'NYISO', 'SPP', 'ISONE'])

        st.subheader('Historical Data Range')
        default_end = datetime.now().date()
        default_start = default_end - timedelta(days=120)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', value=default_start, min_value=datetime(2018, 1, 1).date(), max_value=datetime.now().date())
        with col2:
            end_date = st.date_input('End Date', value=default_end, min_value=datetime(2018, 1, 1).date(), max_value=datetime.now().date())

        days_diff = (end_date - start_date).days
        st.caption(f'📅 {days_diff} days of historical data')

        if days_diff < 30:
            st.warning('⚠️ Select at least 30 days')

        st.subheader('Model Selection')
        use_xgboost = st.checkbox('XGBoost', value=XGBOOST_AVAILABLE, disabled=not XGBOOST_AVAILABLE)
        use_lstm = st.checkbox('LSTM', value=LSTM_AVAILABLE, disabled=not LSTM_AVAILABLE)
        use_prophet = st.checkbox('Prophet', value=PROPHET_AVAILABLE, disabled=not PROPHET_AVAILABLE)

        st.subheader('Parameters')
        max_test = max(30, days_diff // 2)
        test_size = st.slider('Test Set Size (days)', min_value=7, max_value=min(60, max_test), value=14)

        # Future forecasting
        forecast_future = st.checkbox('Forecast Future Dates', value=False)
        if forecast_future:
            future_days = st.slider('Days to Forecast Ahead', 1, 30, 7)
        else:
            future_days = 0

        st.divider()
        run_button = st.button('🚀 Run Forecast', type='primary', use_container_width=True)

    # Display library status
    with st.expander("📦 Library Availability"):
        st.write(f"- **XGBoost**: {'✅ Available' if XGBOOST_AVAILABLE else '❌ Not installed'}")
        st.write(f"- **LSTM (TensorFlow)**: {'✅ Available' if LSTM_AVAILABLE else '❌ Not installed'}")
        st.write(f"- **Prophet**: {'✅ Available' if PROPHET_AVAILABLE else '❌ Not installed'}")
        st.write(f"- **GridStatus**: {'✅ Available' if GRIDSTATUS_AVAILABLE else '❌ Not installed'}")

    if not GRIDSTATUS_AVAILABLE:
        st.error('GridStatus library required. Install: pip install gridstatus')
        return

    if not run_button:
        st.info('👈 Configure parameters and click **Run Forecast**')
        with st.expander('ℹ️ About Models'):
            st.markdown('''
            ### XGBoost
            - Gradient boosting with engineered features
            - Iterative future forecasting using predicted lags

            ### LSTM
            - Recurrent neural network for sequences
            - Iterative forecasting with sequence rolling

            ### Prophet
            - Facebook's time series forecaster
            - Native future forecasting with seasonality
            ''')
        return

    # Fetch data
    with st.spinner('⏳ Fetching data...'):
        try:
            df = fetch_grid_load(market, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()))
            st.success(f'✅ Loaded {len(df)} records from {df.index.min().date()} to {df.index.max().date()}')
        except Exception as e:
            st.error(f'Failed to fetch data: {str(e)}')
            return

    # Show raw data
    with st.expander('📊 Raw Data Sample'):
        st.dataframe(df.head(24), use_container_width=True)
        st.plotly_chart(plot_load_history(df), use_container_width=True)

    # Preprocessing
    with st.spinner('🔧 Preprocessing...'):
        df_clean = preprocess_data(df)
        min_needed = test_size * 24 + 500

        if len(df_clean) < min_needed:
            st.error(f'Not enough data. Need {min_needed} hours, got {len(df_clean)}')
            return

        # Train/test split
        test_samples = test_size * 24
        train_df = df_clean.iloc[:-test_samples]
        test_df = df_clean.iloc[-test_samples:]

    st.success(f'✅ Train: {len(train_df)} | Test: {len(test_df)}')

    # Storage for results
    predictions_dict = {}
    metrics_dict = {}
    future_predictions_dict = {}

    # Global variables for models (needed for future forecasting)
    xgb_model, scaler, df_features = None, None, None
    lstm_model, scaler_lstm = None, None
    prophet_model = None

    # ======================== XGBOOST ========================
    if use_xgboost and XGBOOST_AVAILABLE:
        with st.spinner('🤖 Training XGBoost...'):
            df_features = engineer_features(df_clean)
            test_samples_feat = test_size * 24
            train_feat = df_features.iloc[:-test_samples_feat]
            test_feat = df_features.iloc[-test_samples_feat:]

            feature_cols = [c for c in df_features.columns if c != 'Load']
            X_train = train_feat[feature_cols]
            y_train = train_feat['Load']
            X_test = test_feat[feature_cols]
            y_test = test_feat['Load']

            val_size = max(1, int(len(X_train) * 0.15))
            X_tr = X_train.iloc[:-val_size]
            y_tr = y_train.iloc[:-val_size]
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            xgb_model = train_xgboost_model(X_tr_scaled, y_tr.values, X_val_scaled, y_val.values)
            dtest = xgb.DMatrix(X_test_scaled)
            xgb_pred = xgb_model.predict(dtest)

            predictions_dict['XGBoost'] = xgb_pred
            metrics_dict['XGBoost'] = calculate_metrics(y_test.values, xgb_pred)

            # Future forecasting
            if forecast_future and future_days > 0:
                future_dates, future_pred = iterative_future_forecast_xgboost(df_features, xgb_model, scaler, hours_ahead=future_days*24)
                future_predictions_dict['XGBoost'] = (future_dates, future_pred)

        st.success('✅ XGBoost trained')

    # ======================== LSTM ========================
    if use_lstm and LSTM_AVAILABLE:
        with st.spinner('🧠 Training LSTM...'):
            scaler_lstm = MinMaxScaler()
            scaled_data = scaler_lstm.fit_transform(train_df[['Load']].values)

            lookback = 24
            X_lstm, y_lstm = create_lstm_sequences(scaled_data, lookback)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

            val_split = int(len(X_lstm) * 0.85)
            X_lstm_train, X_lstm_val = X_lstm[:val_split], X_lstm[val_split:]
            y_lstm_train, y_lstm_val = y_lstm[:val_split], y_lstm[val_split:]

            lstm_model, history = train_lstm_model(X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val, epochs=50)

            # Predict on test
            test_scaled = scaler_lstm.transform(test_df[['Load']].values)
            combined = np.vstack([scaled_data[-lookback:], test_scaled])

            lstm_test_preds = []
            for i in range(lookback, len(combined)):
                seq = combined[i-lookback:i].reshape(1, lookback, 1)
                pred = lstm_model.predict(seq, verbose=0)
                lstm_test_preds.append(pred[0, 0])

            lstm_pred = scaler_lstm.inverse_transform(np.array(lstm_test_preds).reshape(-1, 1)).flatten()
            lstm_pred = lstm_pred[:len(test_df)]

            predictions_dict['LSTM'] = lstm_pred
            metrics_dict['LSTM'] = calculate_metrics(test_df['Load'].values, lstm_pred)

            # Future forecasting
            if forecast_future and future_days > 0:
                future_dates, future_pred = iterative_future_forecast_lstm(df_clean, lstm_model, scaler_lstm, lookback, hours_ahead=future_days*24)
                future_predictions_dict['LSTM'] = (future_dates, future_pred)

        st.success('✅ LSTM trained')

    # ======================== PROPHET ========================
    if use_prophet and PROPHET_AVAILABLE:
        with st.spinner('📈 Training Prophet...'):
            prophet_model = train_prophet_model(train_df)

            # Test predictions
            future_test = pd.DataFrame({'ds': test_df.index})
            forecast = prophet_model.predict(future_test)
            prophet_pred = forecast['yhat'].values

            predictions_dict['Prophet'] = prophet_pred
            metrics_dict['Prophet'] = calculate_metrics(test_df['Load'].values, prophet_pred)

            # Future forecasting
            if forecast_future and future_days > 0:
                future_dates, future_pred = future_forecast_prophet(prophet_model, hours_ahead=future_days*24)
                future_predictions_dict['Prophet'] = (future_dates, future_pred)

        st.success('✅ Prophet trained')

    # ======================== RESULTS ========================
    st.header('📈 Results')

    # Metrics comparison
    if metrics_dict:
        st.subheader('📊 Model Performance Metrics')
        metrics_df = pd.DataFrame(metrics_dict).T
        st.dataframe(metrics_df.style.highlight_min(axis=0, subset=['MAE', 'RMSE', 'MAPE (%)'])
                                      .highlight_max(axis=0, subset=['R²']), use_container_width=True)

        st.plotly_chart(plot_metrics_comparison(metrics_dict), use_container_width=True)

    # Model comparison plot
    if predictions_dict:
        st.subheader('🔮 Model Comparison: Test Period')
        comparison_fig = plot_model_comparison(test_df, predictions_dict)
        st.plotly_chart(comparison_fig, use_container_width=True)

    # Future forecasts
    if forecast_future and future_predictions_dict:
        st.subheader('🚀 Future Forecasts (Beyond Historical Data)')
        future_overlay = plot_future_overlay(train_df, test_df, future_predictions_dict)
        st.plotly_chart(future_overlay, use_container_width=True)

        # Download future forecasts
        future_df_export = pd.DataFrame()
        for model_name, (dates, preds) in future_predictions_dict.items():
            if future_df_export.empty:
                future_df_export['Date'] = dates
            future_df_export[f'{model_name}_Forecast'] = preds

        csv_future = future_df_export.to_csv(index=False)
        st.download_button('Download Future Forecasts (CSV)', data=csv_future,
                          file_name=f"{market}_future_forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
                          mime='text/csv')

    # Individual model details
    if use_xgboost and 'XGBoost' in predictions_dict:
        with st.expander("📊 XGBoost Detailed Results"):
            anomaly_mask, residuals = detect_anomalies(test_df['Load'].values, predictions_dict['XGBoost'])
            
            X_train_full_scaled = scaler.transform(X_train)
            dtrain_full = xgb.DMatrix(X_train_full_scaled)
            train_pred = xgb_model.predict(dtrain_full)
            
            fig = plot_full_timeline(train_feat, test_feat, train_pred, predictions_dict['XGBoost'], anomaly_mask)
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(plot_residuals(residuals, test_df.index), use_container_width=True)

    if use_lstm and 'LSTM' in predictions_dict:
        with st.expander("🧠 LSTM Detailed Results"):
            lstm_residuals = test_df['Load'].values - predictions_dict['LSTM']
            st.plotly_chart(plot_residuals(lstm_residuals, test_df.index), use_container_width=True)

    if use_prophet and 'Prophet' in predictions_dict:
        with st.expander("📈 Prophet Detailed Results"):
            prophet_residuals = test_df['Load'].values - predictions_dict['Prophet']
            st.plotly_chart(plot_residuals(prophet_residuals, test_df.index), use_container_width=True)

    # Download test results
    st.subheader('💾 Export Test Results')
    results_df = test_df[['Load']].copy()
    results_df.columns = ['Actual']

    for model_name, preds in predictions_dict.items():
        if preds is not None:
            results_df[f'{model_name}_Predicted'] = preds
            results_df[f'{model_name}_Residual'] = results_df['Actual'] - preds

    csv = results_df.to_csv(index=True)
    st.download_button('Download Test Results (CSV)', data=csv,
                      file_name=f"{market}_multimodel_test_{datetime.now().strftime('%Y%m%d')}.csv",
                      mime='text/csv')


if __name__ == '__main__':
    main()
