import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from config import XGBOOST_AVAILABLE, LSTM_AVAILABLE, SARIMA_AVAILABLE, TIREX_AVAILABLE

if TIREX_AVAILABLE:
    # TiRex is zero-shot, so training is optional, but we import utilities if needed
    pass

if XGBOOST_AVAILABLE:
    import xgboost as xgb
if LSTM_AVAILABLE:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate, TimeDistributed, Reshape, AveragePooling1D, Subtract, Add
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow.keras.backend as K
if SARIMA_AVAILABLE:
    from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_xgboost_model(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,  # Slightly faster but more stable for daily data
        'max_depth': 5,          # Reduced complexity to avoid overfitting
        'min_child_weight': 2,   # Higher min_child_weight for better generalization
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,            # Added gamma for regularization
        'seed': 42
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    model  = xgb.train(params, dtrain, num_boost_round=1000,
                       early_stopping_rounds=100,
                       evals=[(dtrain,'train'),(dval,'val')],
                       verbose_eval=False)
    return model

def train_decision_tree_model(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=150, max_depth=12,
                                  min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', r2_score])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=32, callbacks=[es], verbose=0)
    return model

def train_cnn_lstm_model(X_train_st, X_train_nat, y_train, X_val_st, X_val_nat, y_val, epochs=100):
    """
    Hybrid CNN-LSTM model with dual inputs (State & National context)
    Optimized for higher R-squared on India daily data.
    """
    # State Branch (CNN for local patterns + LSTM for temporal)
    input_st = Input(shape=(X_train_st.shape[1], X_train_st.shape[2]), name='State_Input')
    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_st)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = LSTM(50, return_sequences=False)(x1)
    x1 = Dropout(0.2)(x1)
    
    # National Context Branch (LSTM)
    input_nat = Input(shape=(X_train_nat.shape[1], X_train_nat.shape[2]), name='National_Input')
    x2 = LSTM(32, return_sequences=False)(input_nat)
    x2 = Dropout(0.2)(x2)
    
    # Merging
    combined = Concatenate()([x1, x2])
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.1)(z)
    output = Dense(1)(z)
    
    model = Model(inputs=[input_st, input_nat], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', r2_score])
    
    es = EarlyStopping(monitor='val_r2_score', mode='max', patience=20, restore_best_weights=True)
    model.fit(
        x=[X_train_st, X_train_nat], y=y_train,
        validation_data=([X_val_st, X_val_nat], y_val),
        epochs=epochs, batch_size=32, callbacks=[es], verbose=0
    )
    return model

def train_dlinear_model(X_train, y_train, X_val, y_val, epochs=100, window_size=7):
    """
    DLinear model: Decomposition Linear.
    Decomposes the time series into Trend and Seasonality.
    """
    input_layer = Input(shape=(X_train.shape[1], 1))
    
    # 1. Decomposition Component (Simple Moving Average for Trend)
    # Using AveragePooling1D to simulate moving average
    kernel_size = window_size
    padding = 'same'
    trend = AveragePooling1D(pool_size=kernel_size, strides=1, padding=padding)(input_layer)
    seasonal = Subtract()([input_layer, trend])
    
    # 2. Linear Layers
    # Flatten and then Dense is equivalent to a weighted linear sum per time series
    trend_flat = Flatten()(trend)
    seasonal_flat = Flatten()(seasonal)
    
    trend_out = Dense(1, name='trend_dense')(trend_flat)
    seasonal_out = Dense(1, name='seasonal_dense')(seasonal_flat)
    
    # 3. Summation
    output = Add()([trend_out, seasonal_out])
    
    model = Model(inputs=input_layer, outputs=output)
    # Higher learning rate is often better for simple Linear models
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mae', r2_score])
    
    es = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=32, callbacks=[es], verbose=0)
    
    return model

def train_sarima_model(train_df, order=(1,1,1), seasonal_order=(1,1,1,7)):
    m = SARIMAX(train_df['Load'], order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False)
    return m.fit(disp=False)


