import sys

tex_file = r'c:\code_1\grido\CP final\FYP_Full.tex'

def generate_appendices():
    return r"""
\newpage
  \pagestyle{fancy}
\thisfancypage{
  \setlength{\fboxsep}{20pt}\doublebox}{}
\normalsize
\chapter{APPENDIX A: DETAILED MODEL ARCHITECTURES AND MATHEMATICAL DERIVATIONS}

\section{A.1 Support Vector Machines for Regression (SVR)}
While not the primary model, SVR forms the mathematical basis for margin-based forecasting. The formulation is given by:
\begin{equation}
    \min_{w,b,\xi,\xi^*} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{l}(\xi_i + \xi_i^*)
\end{equation}
subject to:
\begin{equation}
    y_i - (w \cdot \phi(x_i) + b) \leq \epsilon + \xi_i
\end{equation}
\begin{equation}
    (w \cdot \phi(x_i) + b) - y_i \leq \epsilon + \xi_i^*
\end{equation}
where $\xi_i, \xi_i^* \geq 0$. SVR was ultimately superseded by XGBoost in GridOne due to better handling of mixed tabular data structures and inherent cyclical variables.

\section{A.2 XGBoost Advanced Loss Functions}
The objective function of XGBoost at iteration $t$ is formulated as:
\begin{equation}
    \mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
\end{equation}
Where $\Omega(f)$ is the complexity penalty:
\begin{equation}
    \Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2
\end{equation}
By applying a second-order Taylor expansion, we approximate the objective:
\begin{equation}
    \mathcal{L}^{(t)} \simeq \sum_{i=1}^{n} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
\end{equation}
where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})$ and $h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)})$. This second-order approximation is the key to XGBoost's rapid convergence and high predictive accuracy on power load datasets.

\section{A.3 DLinear Trend-Seasonal Component Splitting}
The fundamental equation defining the Average Pooling utilized for the DLinear Trend extraction ($T_t$) over a lookback window $L$ is:
\begin{equation}
    T_t = \frac{1}{k} \sum_{j=0}^{k-1} X_{t-j}
\end{equation}
The seasonal residue ($S_t$) is calculated via strictly element-wise operation:
\begin{equation}
    S_t = X_t - T_t
\end{equation}

\newpage
  \pagestyle{fancy}
\thisfancypage{
  \setlength{\fboxsep}{20pt}\doublebox}{}
\normalsize
\chapter{APPENDIX B: INDIA STATE-LEVEL BASELINES}

The following tables define the empirical parameters utilized by the synthetic data generator to accurately simulate the power requirements of the 37 Indian States and Union Territories.

\begin{table}[H]
    \centering
    \begin{tabular}{|l|c|l|}
    \hline
    \textbf{State / UT} & \textbf{Base Load (MU)} & \textbf{Geographical Region} \\
    \hline
    Maharashtra & 420 & West \\
    Gujarat & 380 & West \\
    Uttar Pradesh & 355 & North \\
    Tamil Nadu & 330 & South \\
    Karnataka & 275 & South \\
    Madhya Pradesh & 285 & Central \\
    Rajasthan & 258 & North \\
    Andhra Pradesh & 220 & South \\
    Telangana & 205 & South \\
    West Bengal & 195 & East \\
    Haryana & 185 & North \\
    Bihar & 125 & East \\
    Odisha & 125 & East \\
    Chhattisgarh & 90 & Central \\
    Kerala & 87 & South \\
    Delhi & 85 & North \\
    Jharkhand & 80 & East \\
    DVC & 75 & East \\
    Himachal Pradesh & 48 & North \\
    Uttarakhand & 47 & North \\
    J\&K and Ladakh & 40 & North \\
    Assam & 38 & East \\
    DNHDDPDCL & 25 & West \\
    BALCO & 21 & Central \\
    Goa & 17 & West \\
    Meghalaya & 10 & East \\
    Chandigarh & 9 & North \\
    Arunachal Pradesh & 9 & East \\
    Manipur & 8 & East \\
    Tripura & 8 & East \\
    Nagaland & 7 & East \\
    Mizoram & 6 & East \\
    Puducherry & 6 & South \\
    Sikkim & 5 & East \\
    \hline
    \end{tabular}
    \caption{Empirical Indian State Baseline Configurations in GridOne}
\end{table}

\newpage
  \pagestyle{fancy}
\thisfancypage{
  \setlength{\fboxsep}{20pt}\doublebox}{}
\normalsize
\chapter{APPENDIX C: SYSTEM CODE LISTINGS}

\section{C.1 Feature Engineering Pipeline (US Hourly)}
The following code snippet is indicative of the raw pandas data manipulation applied to process telemetry into predictive vectors.

\begin{lstlisting}[language=Python]
def engineer_features(df):
    # Initialize temporal structural components
    df = df.copy()
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Calculate Cyclic Temporal projections
    import numpy as np
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Auto-Regressive Lags - Mitigating Data Leakage
    df['Lag_1h'] = df['Load'].shift(1)
    df['Lag_24h'] = df['Load'].shift(24)
    df['Lag_168h'] = df['Load'].shift(168)
    
    # Advanced Rolling Statistics
    df['RollMean_3h'] = df['Load'].shift(1).rolling(window=3).mean()
    df['RollMean_24h'] = df['Load'].shift(1).rolling(window=24).mean()
    df['RollStd_24h'] = df['Load'].shift(1).rolling(window=24).std()
    
    # Terminal Dropna for model ingestion
    return df.dropna()
\end{lstlisting}

\section{C.2 DLinear Architecture Integration}
\begin{lstlisting}[language=Python]
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import AveragePooling1D, Subtract, Add

def build_dlinear_model(lookback):
    inputs = Input(shape=(lookback, 1))
    
    # Branch 1: Trend Extraction via Smoothing
    trend = AveragePooling1D(pool_size=25, strides=1, 
                             padding='same')(inputs)
    trend_flat = Flatten()(trend)
    trend_out = Dense(1, name='trend_dense')(trend_flat)
    
    # Branch 2: High-Frequency Seasonal Abstraction
    seasonal = Subtract()([inputs, trend])
    seasonal_flat = Flatten()(seasonal)
    seasonal_out = Dense(1, name='seasonal_dense')(seasonal_flat)
    
    # Output Fusion
    outputs = Add()([trend_out, seasonal_out])
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
\end{lstlisting}

"""

with open(tex_file, 'a', encoding='utf-8') as f:
    # insert before the \end{document}
    pass

# Read the file, inject appendix before \end{document}
with open(tex_file, 'r', encoding='utf-8') as f:
    text = f.read()

end_idx = text.rfind(r'\end{document}')
if end_idx != -1:
    text = text[:end_idx] + generate_appendices() + "\n" + r'\end{document}'
else:
    text += generate_appendices() + "\n" + r'\end{document}'
    
with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(text)

print("Appendices appended successfully.")
