import numpy as np
import scipy.stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'R²': r2}

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

def diebold_mariano_test(e1, e2, loss='abs'):
    if loss == 'abs':
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d  = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d) + 1e-12)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def detect_anomalies(y_true, y_pred, threshold=3):
    residuals = y_true - y_pred
    z = np.abs((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8))
    return z > threshold, residuals
