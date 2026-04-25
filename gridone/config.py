import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    LSTM_AVAILABLE = True
    DLINEAR_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False
    DLINEAR_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except Exception:
    SARIMA_AVAILABLE = False

try:
    import gridstatus
    GRIDSTATUS_AVAILABLE = True
except Exception:
    GRIDSTATUS_AVAILABLE = False

try:
    import holidays
    US_HOLIDAYS = holidays.UnitedStates()
    INDIA_HOLIDAYS = holidays.India()
except Exception:
    US_HOLIDAYS = None
    INDIA_HOLIDAYS = None

try:
    import tirex
    TIREX_AVAILABLE = True
except Exception:
    TIREX_AVAILABLE = False



INDIA_STATES = {
    'Andhra Pradesh':        {'base': 220, 'noise': 20, 'region': 'south'},
    'Arunachal Pradesh':     {'base': 9,   'noise': 1,  'region': 'east'},
    'Assam':                 {'base': 38,  'noise': 4,  'region': 'east'},
    'Bihar':                 {'base': 125, 'noise': 12, 'region': 'east'},
    'Chandigarh':            {'base': 9,   'noise': 1,  'region': 'north'},
    'Chhattisgarh':          {'base': 90,  'noise': 8,  'region': 'central'},
    'Delhi':                 {'base': 85,  'noise': 10, 'region': 'north'},
    'Goa':                   {'base': 17,  'noise': 2,  'region': 'west'},
    'Gujarat':               {'base': 380, 'noise': 30, 'region': 'west'},
    'Haryana':               {'base': 185, 'noise': 15, 'region': 'north'},
    'HP':                    {'base': 48,  'noise': 5,  'region': 'north'},
    'J&K(UT) & Ladakh(UT)':  {'base': 40,  'noise': 5,  'region': 'north'},
    'Jharkhand':             {'base': 80,  'noise': 8,  'region': 'east'},
    'Karnataka':             {'base': 275, 'noise': 22, 'region': 'south'},
    'Kerala':                {'base': 87,  'noise': 8,  'region': 'south'},
    'MP':                    {'base': 285, 'noise': 25, 'region': 'central'},
    'Maharashtra':           {'base': 420, 'noise': 35, 'region': 'west'},
    'Manipur':               {'base': 8,   'noise': 1,  'region': 'east'},
    'Meghalaya':             {'base': 10,  'noise': 1,  'region': 'east'},
    'Mizoram':               {'base': 6,   'noise': 1,  'region': 'east'},
    'Nagaland':              {'base': 7,   'noise': 1,  'region': 'east'},
    'Odisha':                {'base': 125, 'noise': 10, 'region': 'east'},
    'Puducherry':            {'base': 6,   'noise': 1,  'region': 'south'},
    'Rajasthan':             {'base': 258, 'noise': 22, 'region': 'north'},
    'Sikkim':                {'base': 5,   'noise': 1,  'region': 'east'},
    'Tamil Nadu':            {'base': 330, 'noise': 28, 'region': 'south'},
    'Telangana':             {'base': 205, 'noise': 18, 'region': 'south'},
    'Tripura':               {'base': 8,   'noise': 1,  'region': 'east'},
    'UP':                    {'base': 355, 'noise': 30, 'region': 'north'},
    'Uttarakhand':           {'base': 47,  'noise': 5,  'region': 'north'},
    'West Bengal':           {'base': 195, 'noise': 18, 'region': 'east'},
    'DVC':                   {'base': 75,  'noise': 7,  'region': 'east'},
    'BALCO':                 {'base': 21,  'noise': 2,  'region': 'central'},
    'AMNSIL':                {'base': 4,   'noise': 1,  'region': 'east'},
    'DNHDDPDCL':             {'base': 25,  'noise': 2,  'region': 'west'},
    'Railways_ER ISTS':      {'base': 50,  'noise': 4,  'region': 'east'},
    'Railways_NR ISTS':      {'base': 52,  'noise': 4,  'region': 'north'},
}
