import pandas as pd
import numpy as np
from config import TIREX_AVAILABLE

if TIREX_AVAILABLE:
    import torch
    from tirex import load_model

import streamlit as st

@st.cache_resource
def load_tirex_model():
    """Cache the model loading so it doesn't reload 35M params every run."""
    print("🚀 Loading TiRex Foundation Model (xLSTM)...")
    return load_model("NX-AI/TiRex")

def get_tirex_prediction(historical_series, prediction_length):
    """
    Stable Foundation Model Forecast: 
    Log-Transformation + Instance Normalization + 1024-step Context.
    """
    if not TIREX_AVAILABLE:
        return np.zeros(prediction_length)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_tirex_model()
        model.to(device)
        
        # 1. Log Transformation (Stabilizes variance)
        y_raw = (historical_series.values if isinstance(historical_series, pd.Series) else historical_series).astype(float)
        y_log = np.log1p(y_raw)
        
        # 2. Context Window (1024 steps for maximum pattern memory)
        context_log = y_log[-1024:]
        
        # 3. Instance Normalization (per-series scaling)
        mu = np.mean(context_log)
        sigma = np.std(context_log) + 1e-6
        norm_series = (context_log - mu) / sigma
        
        # 4. Predict
        data = torch.from_numpy(norm_series).float().to(device)
        if data.ndim == 1:
            data = data.unsqueeze(0)
            
        with torch.no_grad():
            quantiles, mean = model.forecast(context=data, prediction_length=prediction_length)
        
        # Point prediction (Mean)
        pred_norm = mean.cpu().numpy().flatten()
        
        # 5. Inverse Normalization
        pred_log = (pred_norm * sigma) + mu
        
        # 6. Inverse Log-Transform
        final_forecast = np.expm1(pred_log)
        
        return final_forecast

    except Exception as e:
        print(f"TiRex Prediction Error: {e}")
        return np.zeros(prediction_length)

def prepare_tirex_context(df, lookback=1024): 
    """Slice the dataframe to provide a larger context window for TiRex."""
    return df['Load'].tail(lookback)
