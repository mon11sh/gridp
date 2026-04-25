# GridOne: An Integrated Multi-Region, Multi-Model Framework for Power Grid Load Forecasting

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Module Breakdown](#4-module-breakdown)
5. [Data Layer](#5-data-layer)
   - 5.1 [US ISO Markets — Real-Time Data Acquisition](#51-us-iso-markets--real-time-data-acquisition)
   - 5.2 [India State-Level — Synthetic and Real Data Pipeline](#52-india-state-level--synthetic-and-real-data-pipeline)
   - 5.3 [Data Preprocessing and Cleaning](#53-data-preprocessing-and-cleaning)
6. [Feature Engineering Pipeline](#6-feature-engineering-pipeline)
   - 6.1 [US Hourly Feature Set](#61-us-hourly-feature-set)
   - 6.2 [India Daily Feature Set](#62-india-daily-feature-set)
7. [Forecasting Models](#7-forecasting-models)
   - 7.1 [Decision Tree Regressor](#71-decision-tree-regressor)
   - 7.2 [Random Forest Regressor](#72-random-forest-regressor)
   - 7.3 [XGBoost Regressor](#73-xgboost-regressor)
   - 7.4 [LSTM Neural Network (Hourly)](#74-lstm-neural-network-hourly)
   - 7.5 [Hybrid CNN-LSTM (India Daily)](#75-hybrid-cnn-lstm-india-daily)
   - 7.6 [SARIMA](#76-sarima)
   - 7.7 [TiRex — xLSTM Foundation Model](#77-tirex--xlstm-foundation-model)
   - 7.8 [DLinear — Decomposition Linear Model](#78-dlinear--decomposition-linear-model)
8. [Forecasting Strategy — Iterative Multi-Step Approach](#8-forecasting-strategy--iterative-multi-step-approach)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Visualization and UI Design](#10-visualization-and-ui-design)
11. [Scalability and Deployment Considerations](#11-scalability-and-deployment-considerations)
12. [Comparative Performance Analysis](#12-comparative-performance-analysis)
13. [Conclusion and Future Work](#13-conclusion-and-future-work)
14. [Appendix: Configuration and Dependency Tables](#14-appendix-configuration-and-dependency-tables)

---

## 1. Abstract

GridOne is an open-source, multi-region, multi-model electricity load forecasting framework for power grid research and decision-making. The framework provides a unified interface for training, evaluating, and deploying a broad spectrum of forecasting models — ranging from classical statistical approaches such as SARIMA to state-of-the-art deep learning architectures including the **TiRex (xLSTM-based)** foundation model and the Decomposition Linear (DLinear) model. GridOne is designed to operate across two geographically and structurally distinct power markets: the United States electricity grid (ISO market-level, hourly granularity) and the Indian State-level grid (daily granularity, spanning all 37 States and Union Territories). The system supports GPU-accelerated training, interactive visualization via Plotly, and incorporates rigorous statistical validation including the Diebold-Mariano test and Nash-Sutcliffe Efficiency for robust model comparison. This document provides a comprehensive technical account of the system's architecture, data pipeline, modeling philosophy, engineering decisions, and evaluation methodology for use in research and academic contexts.

---

## 2. Introduction and Motivation

Accurate electricity load forecasting is fundamental to the reliable and economical operation of modern power grids. Grid operators, independent system operators (ISOs), and energy planners depend on short-term and medium-term load forecasts to schedule generation assets, optimize transmission, prevent outages, and ensure economic dispatch. As renewable energy penetration increases and demand patterns grow more complex, the forecasting models that once sufficed — simple auto-regressive or linear statistical models — are increasingly inadequate.

The challenge of electricity load forecasting is multifaceted:
- **Non-stationarity:** Demand patterns shift with seasons, economic growth, and societal changes (e.g., remote work trends, EV adoption).
- **Multi-scale seasonality:** Demand exhibits strong intraday, weekly, and annual cycles that are difficult to capture simultaneously.
- **External dependencies:** Weather, public holidays, industrial activity, and grid maintenance all influence load in ways that are difficult to parametrize.
- **Spatial heterogeneity:** Different geographic regions, states, or ISOs exhibit fundamentally different load profiles, even under similar external conditions.

GridOne addresses these challenges by adopting a comparative, multi-model framework. Rather than prescribing a single "best" model, the system allows researchers and operators to evaluate multiple approaches simultaneously against the same dataset, with consistent feature engineering, training splits, and evaluation metrics. This enables data-driven model selection suited to the specific characteristics of each market.

The dual-region design — covering both the US (a highly deregulated, market-driven ISO structure) and India (a state-level centrally planned system) — intentionally captures diverse market structures to demonstrate the generalizability of the framework.

---

## 3. System Architecture Overview

GridOne follows a modular, layered software architecture. The application is structured as a Python package (`gridone`) with clear separation of concerns across the following functional layers:

```
gridone/
├── app.py              # Main Streamlit entrypoint. Routing between US and India modes.
├── config.py           # Global configuration: availability flags, state parameters, holiday calendars.
├── data.py             # Data acquisition, synthetic generation, preprocessing, and feature engineering.
├── training.py         # Model training functions for all 7 architectures.
├── forecasting.py      # Iterative multi-step forecasting functions for each model.
├── tirex_utils.py      # Dedicated TiRex foundation model loading and zero-shot inference utilities.
├── metrics.py          # Evaluation: MAE, RMSE, MAPE, R², NSE, Diebold-Mariano, Anomaly Detection.
├── visualization.py    # Plotly chart generators for history, comparison, residuals, and future overlays.
├── ui_india.py         # Full Streamlit UI logic for India mode.
├── ui_us.py            # Full Streamlit UI logic for US mode.
└── compile_data.py     # Utility to compile and export merged data files.
```

The Streamlit application (app.py) serves as the entry point, rendering a mode-selection interface at startup and routing the user to either the US or India forecasting pipeline. Each pipeline is self-contained in its respective UI module and calls shared services from the training, forecasting, metrics, and visualization modules.

**Design Principles:**
1. **Separation of Concerns:** Every module has a single defined responsibility — no training logic in the UI, no data logic in training.
2. **Graceful Degradation:** Optional dependencies (XGBoost, TensorFlow, PyTorch, statsmodels) are detected at runtime via `try/except` in `config.py`. If a library is unavailable, its corresponding model checkbox in the UI is automatically disabled rather than crashing the application.
3. **Caching:** Expensive data operations (live data fetches, file reads) are wrapped with Streamlit's `@st.cache_data` decorator to prevent re-fetching on every UI interaction.
4. **Hardware Awareness:** The framework utilizes TiRex's efficient architecture, which is optimized for both CPU and GPU execution without the heavy configuration overhead of Transformers.

---

## 4. Module Breakdown

### `config.py`
The configuration module serves as the single source of truth for environment setup. It attempts to import each optional library at startup and sets boolean availability flags:
- `XGBOOST_AVAILABLE` — whether the `xgboost` library is installed.
- `LSTM_AVAILABLE` / `DLINEAR_AVAILABLE` — whether TensorFlow/Keras is installed.
- `SARIMA_AVAILABLE` — whether `statsmodels` is installed.
- `GRIDSTATUS_AVAILABLE` — whether `gridstatus` is available for US data.
- `TIREX_AVAILABLE` — whether `tirex-ts` foundation model library is installed.

The module also initializes holiday calendars (`holidays.UnitedStates()` and `holidays.India()`) and the `INDIA_STATES` dictionary, which contains generation capacity baselines, noise levels, and regional identifiers (north/south/east/west/central) for all 37 Indian States and Union Territories.

### `app.py`
The main entry point renders the application title and a top-level radio button allowing the user to switch between `US Mode` and `India Mode`. It passes the Streamlit sidebar object to the selected UI module for rendering.

---

## 5. Data Layer

### 5.1 US ISO Markets — Real-Time Data Acquisition

For the US market, GridOne fetches real-time and historical load data from ISO operators through the `gridstatus` library. The `fetch_grid_load()` function in `data.py` supports six major Independent System Operators:

| ISO | Region Covered |
| :--- | :--- |
| **CAISO** | California |
| **PJM** | Mid-Atlantic & Midwest |
| **MISO** | Midwest & South |
| **NYISO** | New York State |
| **SPP** | Southwest Power Pool |
| **ISONE** | New England |

The function instantiates the appropriate grid object from the `gridstatus` library using a string lookup table, calls `.get_load()` with the requested date range, and normalizes the output into a standard two-column DataFrame: a `DatetimeIndex` and a `Load` column in MW. Timezone normalization is applied (all timestamps converted to timezone-naive UTC-aligned values) and a deduplication pass is run on the index. The result is cached for 1 hour to avoid redundant API calls during the same user session.

**Why 6 ISOs?** These markets collectively represent approximately 70% of US electricity consumption and span the most diverse regional demand profiles — from California's solar-heavy market with strong midday dips, to New England's gas-heavy winter peaks, to PJM's coal-influenced baseload profile.

### 5.2 India State-Level — Synthetic and Real Data Pipeline

India's state-level data presents unique challenges: publicly available granular historical load data is fragmented across POSOCO/NLDC reports. GridOne addresses this with a **hybrid data pipeline** that combines real data (when available from a local CSV) with a high-fidelity **physics-informed synthetic data generator** for all 37 states.

The `generate_synthetic_india_data()` function implements the following steps:

**Step 1 — Base Load Configuration:** Each state has a pre-configured `base` demand (in MU — Million Units) and `noise_lvl` derived from POSOCO data summaries and stored in the `INDIA_STATES` dictionary in `config.py`. For example, Maharashtra has a base of 420 MU and Gujarat 380 MU, reflecting their industrial load dominance.

**Step 2 — Seasonal Component:** The function models regional seasonality differently by geographic zone:
- **North/Central states** (e.g., Rajasthan, UP, MP): A compound sinusoidal model with two harmonics captures both the summer cooling peak (April-June) and a winter heating sub-peak: `base × 0.12 × sin(2π(doy−80)/365) + base × 0.06 × cos(4π×doy/365)`.
- **South/East/West states** (e.g., Tamil Nadu, Karnataka, Maharashtra): A single harmonic model `base × 0.08 × sin(2π(doy−60)/365)` reflecting the less extreme but earlier-onset summer peak.

**Step 3 — Trend Component:** A linear growth trend is superimposed across the entire date range: `linspace(0, base×0.08, n)`. This approximates India's approximately 6-8% year-on-year electricity demand growth.

**Step 4 — Weekly Cycle:** A fixed weekly pattern applies a 4% load reduction on weekends (Saturday, Sunday), reflecting reduced industrial activity: `−base × 0.04 × (dow ≥ 5)`.

**Step 5 — Stochastic Noise:** Gaussian noise `N(0, noise_lvl)` is added to simulate day-to-day variability in demand. The noise level is calibrated per state (e.g., smaller states like Mizoram/Nagaland with base ~6-8 MU have `noise=1`, large industrial states have `noise=30-35`).

**Step 6 — Real Data Integration:** If a local `data/india_master_data.csv` file exists, the function reads it and overwrites the synthetic values on any date where real data is available, creating a seamless hybrid series. The base and noise parameters are also recalibrated from the real data statistics (mean → new base, std → new noise).

**Step 7 — Outlier Removal and Smoothing:** A 14-day rolling Z-score outlier filter is applied (threshold: 3.5σ), outliers are nullified, and linear interpolation fills the gaps. A final 3-day centered rolling mean is applied to smooth the stitching boundary between real and synthetic data.

### 5.3 Data Preprocessing and Cleaning

For US hourly data, `preprocess_data()` performs:
1. **Hourly Resampling:** `resample('H').mean()` ensures a uniform hourly frequency, handles sub-hourly raw data, and creates explicit `NaN` slots for missing hours.
2. **Time-Limited Interpolation:** `interpolate(method='time', limit=6)` fills gaps up to 6 consecutive hours using time-weighted linear interpolation, appropriate for short outages or data transmission delays.
3. **Forward/Backward Fill:** Remaining edge gaps are filled with `ffill().bfill()`.

---

## 6. Feature Engineering Pipeline

Feature engineering is the most critical step differentiating model performance. GridOne implements two distinct feature sets — one for the US hourly domain and one for the India daily domain.

### 6.1 US Hourly Feature Set

The `engineer_features()` function creates the following features from the raw `Load` column:

| Feature | Description | Rationale |
| :--- | :--- | :--- |
| `Hour` | Hour of day (0–23) | Captures the intraday consumption curve |
| `DayOfWeek` | Day of week (0=Mon, 6=Sun) | Weekday vs weekend demand pattern |
| `Month` | Calendar month (1–12) | Seasonal demand shifts |
| `DayOfYear` | Day of year (1–365) | Fine-grained seasonal encoding |
| `IsWeekend` | Binary: 1 if Sat/Sun | Explicit weekend flag |
| `IsHoliday` | Binary: 1 if US Federal Holiday | Holiday load reduction signal |
| `Lag_1h` | Load 1 hour ago | Immediate auto-regressive memory |
| `Lag_24h` | Load 24 hours ago (same time yesterday) | Daily rhythm memory |
| `Lag_168h` | Load 168 hours ago (same time last week) | Weekly rhythm memory |
| `RollMean_3h` | 3-hour rolling mean (shifted by 1h) | Short-term trend |
| `RollMean_24h` | 24-hour rolling mean (shifted by 1h) | Daily average baseline |
| `RollStd_24h` | 24-hour rolling standard deviation | Volatility measure |

**Data Leakage Prevention:** All rolling features use `.shift(1)` before the rolling window to ensure the current timestep's value is never included in its own feature calculation, preventing data leakage during training.

### 6.2 India Daily Feature Set

The `engineer_features_daily()` function implements a richer, more expressive feature set optimized for the daily grain Indian data:

**Temporal Features:**
- `Year`, `DayOfWeek`, `Month`, `DayOfYear`, `IsWeekend`, `Season` (0-3 mapping quarters).
- `TimeIndex` — a monotonically increasing integer index used as a direct trend feature for linear models.

**Cyclic Encoding (Critical for Model Performance):**
Categorical time features like `Month` and `DayOfWeek` suffer from an "end-of-range boundary problem" when treated as integers. For example, a model using integer encoding would see January (1) and December (12) as maximally distant when they are actually adjacent. GridOne resolves this with sine-cosine projection:
- `Month_Sin = sin(2π × Month / 12)`, `Month_Cos = cos(2π × Month / 12)`
- `Day_Sin = sin(2π × DayOfWeek / 7)`, `Day_Cos = cos(2π × DayOfWeek / 7)`

This maps each temporal unit onto the unit circle, creating a continuous and periodic representation.

**Auto-Regressive Lag Features:**
| Feature | Lag | Captures |
| :--- | :--- | :--- |
| `Lag_1d` | 1 day | Yesterday's demand |
| `Lag_2d` | 2 days | Two days prior |
| `Lag_7d` | 7 days | Same day last week (weekly rhythm) |
| `Lag_14d` | 14 days | Bi-weekly pattern |
| `Lag_30d` | 30 days | Monthly seasonality |

**Rolling Statistics:**
- `RollMean_3d`, `RollMean_7d` — Short and medium-term demand averages.
- `RollStd_7d` — Weekly demand volatility (high values signal demand uncertainty events).
- `RollMax_7d`, `RollMin_7d` — 7-day range, captures peak/off-peak spread.
- `EWM_7d` — Exponentially weighted 7-day mean (span=7), giving more importance to recent observations.

**Momentum and Derivative Features (Novel Additions):**
- `Load_Diff_1d = Lag_1d − Lag_2d` — First-order difference, capturing the rate of change of demand.
- `Load_Diff_7d = Lag_7d − Lag_8d` — Weekly momentum.
- `Trend_6d = Lag_1d − Lag_7d` — 6-day demand gradient, a directional trend signal.
- `Vol_Interaction = RollStd_7d / (RollMean_7d + ε)` — Coefficient of variation, a normalized volatility index.

**Holiday Integration:**
- `IsIndianHoliday` — Binary flag using the `holidays.India()` calendar, which covers all major national, religious, and regional Indian public holidays. This is particularly important for India where festivals (Diwali, Holi, Eid) cause demand dips of 10–20%.

---

## 7. Forecasting Models

GridOne integrates 7 distinct forecasting model families, each representing a different paradigm in machine learning and time series modeling.

### 7.1 Decision Tree Regressor

**Architecture:** A non-parametric supervised learning model that recursively partitions the feature space into rectangular regions based on impurity minimization (Gini or MSE for regression).

**Hyperparameters:**
- `max_depth = 10` — Controls overfitting by limiting tree depth.
- `min_samples_leaf = 5` — Each leaf must contain at least 5 training samples, preventing very specific leaf conditions.
- `random_state = 42` — Ensures reproducibility.

**Role in GridOne:** Serves as a transparent, interpretable baseline. Decision trees are fast to train and provide feature importance scores, making them useful for understanding which features drive predictions. However, they are prone to overfitting and do not extrapolate beyond the training data range, which limits their utility on long future forecast horizons.

### 7.2 Random Forest Regressor

**Architecture:** An ensemble of 150 independent Decision Trees trained on bootstrapped subsets of the data (bagging). Predictions are averaged across all trees, reducing variance dramatically compared to a single tree.

**Hyperparameters:**
- `n_estimators = 150` — Number of trees in the forest.
- `max_depth = 12` — Slightly deeper than single DT to compensate for variance reduction from averaging.
- `min_samples_leaf = 3` — Less conservative than single DT since bagging already reduces variance.
- `n_jobs = −1` — Parallelizes training across all available CPU cores.
- `random_state = 42`.

**Role in GridOne:** Provides a robust ensemble baseline. Random Forest is resistant to outliers and handles correlated features better than single trees. It generalizes well and rarely produces catastrophically bad predictions, making it a reliable fallback model.

### 7.3 XGBoost Regressor

**Architecture:** An optimized implementation of gradient boosted trees that iteratively builds an ensemble where each new tree corrects the residual errors of the previous ensemble.

**Hyperparameters:**
| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| `objective` | `reg:squarederror` | Mean Squared Error loss for regression |
| `learning_rate` | 0.05 | Conservative step size for stable convergence |
| `max_depth` | 5 | Controls model complexity |
| `min_child_weight` | 2 | Prevents learning from very small leaf groups |
| `subsample` | 0.8 | Row sampling per tree — reduces overfitting |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `gamma` | 0.1 | Minimum loss reduction to make a split |
| `num_boost_round` | 1000 | Maximum number of boosting rounds |
| `early_stopping_rounds` | 100 | Halts if validation loss doesn't improve for 100 rounds |

**Training Strategy:** XGBoost is trained using a `DMatrix` (its internal optimized data format) and uses a chronological validation split for early stopping. The `eval` callback monitors both training and validation RMSE in real-time.

**Role in GridOne:** The primary high-performance model. XGBoost consistently achieves the best accuracy on tabular feature sets due to its ability to capture non-linear interactions between lag features, temporal indicators, and rolling statistics. It is also extremely fast to train (typically under 30 seconds even for large datasets).

### 7.4 LSTM Neural Network (Hourly)

**Architecture:** A two-layer Long Short-Term Memory (LSTM) network trained directly on raw scaled load sequences, without feature engineering.

**Model Structure:**
```
Input (batch, lookback=24, 1)
  → LSTM(64 units, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32 units, return_sequences=False)
  → Dropout(0.2)
  → Dense(16, activation='relu')
  → Dense(1)  ← Forecast output
```

**Hyperparameters:**
- `lookback = 24` hours (US mode) — The model sees the last 24 hours of load to predict the next hour.
- `epochs = 50` with `EarlyStopping(monitor='val_loss', patience=15)` — Halts training if validation loss stagnates for 15 epochs, best weights are restored.
- `batch_size = 32`.
- Optimizer: `Adam`.
- Loss: `mse` + `mae` + custom `r2_score`.

**Role in GridOne:** Captures temporal dependencies in raw load sequences that feature-engineered models might miss, particularly intraday cyclical patterns. The dual-LSTM stack with increasing temporal abstraction allows the model to capture both short-term fluctuations (first LSTM layer) and longer-term patterns (second LSTM layer).

### 7.5 Hybrid CNN-LSTM (India Daily)

**Architecture:** A more sophisticated two-branch neural architecture designed specifically for the India daily forecasting mode, which integrates both state-level and national-level load signals.

**Branch 1 — State Load Branch:**
```
Input: (batch, lookback=30, 1) — State-level load sequence
  → Conv1D(32 filters, kernel_size=3, padding='same', activation='relu')
  → MaxPooling1D(pool_size=2)
  → LSTM(50 units, return_sequences=False)
  → Dropout(0.2)
  → Output: State Context Vector (50-dim)
```

**Branch 2 — National Load Branch:**
```
Input: (batch, lookback=30, 1) — National aggregate load (sum of all 37 states)
  → LSTM(32 units, return_sequences=False)
  → Dropout(0.2)
  → Output: National Context Vector (32-dim)
```

**Fusion and Output:**
```
Concatenate([State Context, National Context]) → 82-dim joint representation
  → Dense(32, activation='relu')
  → Dropout(0.1)
  → Dense(1)  ← Day demand forecast in MU
```

**Rationale for Dual-Branch Architecture:** State-level load is not independent — it is strongly correlated with the national grid state. The national context branch acts as a "macro signal" that helps the model understand whether demand is broadly high or low, providing context that improves state-level predictions.

**Training Strategy:**
- `epochs = 60` with `EarlyStopping(monitor='val_r2_score', mode='max', patience=20)` — Monitors R² directly, halting when generalization stops improving.
- `lookback = 30` days.
- National context data is generated by summing all 37 synthetic state series.

### 7.6 SARIMA

**Architecture:** Seasonal AutoRegressive Integrated Moving Average — a classical statistical time series model that extends ARIMA with seasonal components.

The SARIMAX model is formulated as `SARIMA(p,d,q)(P,D,Q)[s]` where:
- `p, d, q` — Non-seasonal AR, differencing, and MA orders
- `P, D, Q` — Seasonal AR, differencing, and MA orders
- `s` — Seasonal period

**GridOne Configuration:**
- **US Mode:** `SARIMA(1,1,1)(1,1,1)[24]` — 24-hour seasonal period for intraday cycles.
- **India Mode:** `SARIMA(1,1,1)(1,1,1)[7]` — 7-day seasonal period for weekly cycles.

**Fitting:** Estimated via Maximum Likelihood Estimation (MLE). `enforce_stationarity=False` and `enforce_invertibility=False` allow the model to handle edge cases in the data without raising convergence errors.

**Role in GridOne:** Serves as a statistically rigorous benchmark. SARIMA is a gold standard in energy forecasting literature and provides important context for evaluating the marginal gains from more complex ML and DL models. It is the slowest model in the suite (MLE optimization is sequential and computationally expensive) and performs poorly when the relationship between load and time is highly non-linear.

### 7.7 TiRex — xLSTM Foundation Model

**Architecture:** TiRex is a state-of-the-art foundation model for time series forecasting based on the **xLSTM (extended Long Short-Term Memory)** architecture (Beck et al., 2024). Unlike standard Transformers which can be unstable on smaller datasets, TiRex utilizes exponential gating and memory mixing to maintain accurate state-tracking over long horizons.

**Key TiRex Components:**
1. **xLSTM Blocks:** Evolution of the traditional LSTM that overcomes the constant error carousel limitation, allowing for parallel training while maintaining recurrent state-tracking.
2. **Zero-Shot Inference:** The model is pre-trained on a massive corpus of diverse time series data, allowing it to generate high-accuracy forecasts on the GridOne datasets without requiring local fine-tuning.
3. **Contiguous Patch Masking:** Enhances the model's ability to learn temporal patterns within localized patches of data.

**GridOne Implementation:**
- Integrated via the `tirex-ts` library.
- Operates in **Zero-Shot** mode to provide a highly stable deep learning benchmark that avoids the convergence issues commonly seen in Transformers like TFT.
- Uses a **168-hour (US)** or **90-day (India)** context window for inference.

**Role in GridOne:** Replaced the Temporal Fusion Transformer (TFT) to provide a more robust and stable deep learning baseline. TiRex consistently avoids the "negative R²" problem by leveraging its pre-trained understanding of temporal rhythms.


### 7.8 DLinear — Decomposition Linear Model

**Architecture:** A lightweight yet powerful forecasting model proposed by Zeng et al. (2022) in the paper *"Are Transformers Effective for Time Series Forecasting?"* The core insight of DLinear is that explicit decomposition of a time series into trend and seasonality components, followed by independent linear processing of each, can achieve competitive or superior accuracy to far more complex architectures.

**Model Structure in GridOne (Keras Implementation):**
```
Input: (batch, lookback, 1)
  │
  ├─── Trend Branch ────────────────────────────────────────────────────
  │    AveragePooling1D(pool_size=25, strides=1, padding='same')
  │      → Trend component (moving average of the series)
  │    Flatten()
  │    Dense(1, name='trend_dense')
  │
  ├─── Seasonal Branch ─────────────────────────────────────────────────
  │    Subtract([Original, Trend])  ← ElementWise subtraction layer
  │      → Seasonal component (residual after removing trend)
  │    Flatten()
  │    Dense(1, name='seasonal_dense')
  │
  └─── Add([trend_out, seasonal_out])  ← Final forecast
```

**Decomposition Detail:**
- The **Trend** is extracted using average pooling with `kernel_size=25` (representing 25 timesteps), which is equivalent to a simple moving average — a widely used method in classical time series analysis (Moving Average decomposition).
- The **Seasonal** component is the point-wise difference between the original series and the extracted trend: `Seasonal[t] = X[t] − Trend[t]`. This is the standard additive decomposition model.

**Training Hyperparameters:**
- `epochs = 50` with `EarlyStopping(monitor='val_loss', patience=15)`.
- `lookback = 24` hours (US) / 30 days (India).
- `batch_size = 32`.
- Optimizer: `Adam`.

**Why DLinear Competes with Complex Models:**
The hypothesis that complex models are always better is challenged by the strong prior structure in electricity load data. Grid load is highly periodic and trend-stationary. By explicitly removing the trend before fitting, DLinear ensures the linear layers only need to learn the bounded, zero-mean seasonal pattern — a far easier optimization problem than modeling the raw non-stationary series. The result is a model that is 10-50× faster to train than TFT while achieving comparable or superior performance on many datasets.

**Role in GridOne:** Provides the most efficient accuracy-to-speed trade-off in the model suite. It is the recommended model for rapid exploratory analysis or when computational resources are constrained.

---

## 8. Forecasting Strategy — Iterative Multi-Step Approach

For future point-in-time predictions (beyond the historical test window), GridOne uses an **iterative multi-step forecasting** strategy for most models except **TiRex** (which natively supports multi-step zero-shot output) and **SARIMA** (which uses closed-form state-space forecasting).

The iterative approach works as follows:
1. Initialize with the last `lookback` observations from the historical data.
2. For each future timestep `t`:
   a. Construct the input feature vector or sequence from the current context window.
   b. Generate prediction `ŷ_t`.
   c. Append `ŷ_t` to the context window (replacing the oldest observation).
3. Return the array of all `ŷ_t` values and corresponding future timestamps.

This strategy is implemented separately for each model format:
- `iterative_future_forecast_sklearn()` / `iterative_future_forecast_sklearn_daily()` — For Decision Tree and Random Forest (uses `_build_hourly_row()` or `_build_daily_row()` to reconstruct full feature vectors).
- `iterative_future_forecast_xgboost()` / `iterative_future_forecast_xgboost_daily()` — Identical logic but wraps input in `xgb.DMatrix`.
- `iterative_future_forecast_lstm_us()` / `iterative_future_forecast_lstm_daily()` — Updates sequence window by dropping the oldest step and appending the new prediction.
- `iterative_future_forecast_cnn_lstm_daily()` — The dual-branch variant updates both state and national sequences separately; for the national forecast the last observed national value is propagated forward.
- `iterative_future_forecast_dlinear_us()` / `iterative_future_forecast_dlinear_daily()` — Identical rolling-window update to LSTM.
- `future_forecast_tirex()` — Delegates directly to the `tirex-ts` zero-shot prediction pipeline.

**Limitation:** Iterative strategies accumulate error over time as each prediction becomes an input to the next step. Forecast accuracy degrades beyond approximately 7-14 days for most models. This is an inherent property of autoregressive forecasting and is not specific to GridOne.

---

## 9. Evaluation Metrics

GridOne computes a comprehensive set of evaluation metrics, balancing classical statistical accuracy measures with domain-specific energy and hydrological validation tools.

### 9.1 Standard Accuracy Metrics

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ|y_i − ŷ_i|
```
Measures the average magnitude of forecast errors, without directional bias. Expressed in the original unit (MW or MU). Lower is better.

**Root Mean Square Error (RMSE):**
```
RMSE = sqrt((1/n) × Σ(y_i − ŷ_i)²)
```
Similar to MAE but penalizes large errors disproportionately, making it sensitive to spikes and outliers. Lower is better.

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (1/n) × Σ(|y_i − ŷ_i| / (|y_i| + ε)) × 100
```
A scale-independent relative error that allows comparison across different states and regions regardless of their absolute load magnitude. A small epsilon (1e-8) is added to avoid division by zero. Lower is better.

**R² — Coefficient of Determination:**
```
R² = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²
```
Measures the proportion of variance in the true load that is explained by the model. R² = 1 is a perfect fit; R² = 0 means the model performs no better than predicting the mean; R² < 0 means the model is worse than the mean. Higher is better.

### 9.2 Domain-Specific Metric

**Nash-Sutcliffe Efficiency (NSE):**
```
NSE = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²
```
Mathematically equivalent to R² but with a rich history in hydrology and energy systems modeling, where it is the standard efficiency measure for comparing forecast models. NSE > 0.75 is generally considered a "very good" model in energy literature. NSE = 1 indicates a perfect model, NSE ≤ 0 indicates the model is no better than using the historical mean.

### 9.3 Statistical Significance — Diebold-Mariano Test

The Diebold-Mariano (DM) test provides a formal hypothesis test for whether the forecast accuracy difference between two competing models is statistically significant, rather than due to random sampling variation.

**Procedure:**
1. Compute loss differentials: `d_t = |e1_t| − |e2_t|` (using absolute error loss by default).
2. Compute the DM statistic: `DM = mean(d) / sqrt(var(d) / n)`.
3. Compute two-tailed p-value from the standard normal distribution.

A p-value < 0.05 indicates statistically significant difference in forecast accuracy. GridOne computes this test for every pair of models trained in a session, displaying the DM statistic and p-value in the UI.

### 9.4 Anomaly Detection

The residual-based anomaly detector `detect_anomalies()` computes Z-scores of the prediction residuals `(y_true − ŷ)` and flags any residual exceeding 3 standard deviations as an anomaly. These anomalies are overlaid on timeline charts as orange markers, drawing attention to periods where all models collectively failed (which often correspond to real-world events: heatwaves, demand surges, unexpected shutdowns).

---

## 10. Visualization and UI Design

All visualizations are generated as interactive Plotly charts, rendered within the Streamlit application.

### `plot_load_history()` — Raw Data Preview
A single-trace line chart of the full historical load series, used for initial data exploration. Color: Royal Blue.

### `plot_full_timeline()` — Complete Timeline View
A four-trace overlay chart showing: Historical Training Data (light blue, low opacity), Training Set Fit (green dotted line), Test Set Actual (solid blue), Model Forecast on Test Set (red dashed), and Anomalies as orange X markers. Uses `hovermode='x unified'` for synchronized tooltips.

### `plot_model_comparison()` — Multi-Model Test Overlay
All model predictions for the test period displayed on a single chart against the actual load (thick black line). Each model is assigned a consistent color (`XGBoost=red`, `LSTM=blue`, `SARIMA=green`, `DT=orange`, `RF=purple`, `TFT=teal`).

### `plot_metrics_comparison()` — Metrics Bar Charts
A 2×2 grid of bar charts comparing MAE, RMSE, MAPE, and R² across all trained models simultaneously.

### `plot_residuals()` — Residual Analysis
A 2-row subplot: residuals over time (line chart, purple) and residual distribution (histogram, light blue). Allows visual inspection of systematic bias and heteroscedasticity.

### `plot_future_overlay()` — Historical + Test + Future Forecasts
A compound chart displaying historical context, test actuals, and all models' future extrapolations as dashed lines extending beyond the test period.

### `plot_india_national_overview()` — National Demand Map
A bar chart of average daily demand by state, sorted descending, with a Viridis colorscale mapping demand magnitude to color. This provides an at-a-glance visual of India's load distribution.

---

## 11. Scalability and Deployment Considerations

**Caching Strategy:** GridOne uses `@st.cache_data(ttl=3600)` for data fetch and file read operations. This is critical for the US mode where each data fetch involves a network API call that may take 5-30 seconds.

**Hardware Acceleration:** The TiRex model uses efficient torch-based computation. It automatically utilizes CUDA if an NVIDIA GPU is present, significantly speeding up the zero-shot inference process for large horizons.

**Memory Efficiency:** For India mode, national context data (sum of all 37 synthetic state series) is generated within the Streamlit spinner context and scoped to the training function call. This prevents persistent in-memory accumulation of large arrays across training runs.

**Streamlit Session State:** Each model's predictions are stored in Python dictionaries (`predictions_dict`, `metrics_dict`, `future_dict`) local to the training session, making the UI stateless across page reloads which prevents stale predictions from appearing.

---

## 12. Comparative Performance Analysis

Based on the characteristics of electricity load data and the properties of each model architecture, the models in GridOne can be ranked broadly as follows. Note that actual performance is dataset-dependent and should be validated empirically using the DM test.

| Rank | Model | Typical R² | Speed | Interpretability | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **XGBoost** | 0.90–0.98 | Very Fast | Medium (feature importance) | Production forecasting, tabular data |
| 2 | **DLinear** | 0.85–0.95 | Very Fast | High (trending vs. seasonal split) | Fast benchmarking, trend-dominant data |
| 3 | **TiRex** | 0.88–0.97 | Fast | Medium | Research, stable zero-shot forecasts |
| 4 | **Hybrid CNN-LSTM** | 0.80–0.93 | Moderate | Low | Multi-input spatial-temporal forecasting |
| 5 | **Random Forest** | 0.78–0.90 | Fast | Medium | Robust ensemble baseline |
| 6 | **Decision Tree** | 0.70–0.85 | Very Fast | High (tree visualization) | Simple, explainable baseline |
| 7 | **SARIMA** | 0.70–0.88 | Very Slow | High (component analysis) | Statistical baseline, seasonal data |

**Key Observations:**
- For structured time series with rich feature engineering, **gradient boosted models (XGBoost)** consistently outperform neural networks due to their superior handling of tabular feature interactions.
- **DLinear** surprises despite its simplicity — this is because electricity load is inherently trend-stationary and seasonal, the exact structure that DLinear's decomposition is designed to exploit.
- **TFT** achieves its strongest advantage over simpler models on **long forecast horizons** (30+ days) where complex temporal interactions between features become more decisive.
- **SARIMA** acts as the essential benchmark: if a complex ML model cannot beat SARIMA, it is likely underfitting or has a feature engineering problem.

---

## 13. Conclusion and Future Work

GridOne demonstrates that a well-designed, modular forecasting framework can bridge the gap between research-grade algorithms and practical deployment for grid operators in both developed (US ISO) and emerging (Indian state) electricity markets. By combining modern deep learning architectures with robust classical methods, systematic feature engineering, and statistically validated comparison tools, the framework provides both the flexibility for research exploration and the rigor required for operational decision-making.

**Planned Future Enhancements:**
1. **Weather Integration:** Incorporate temperature, humidity, and solar irradiance data from Open-Meteo or similar APIs, as weather is the single strongest predictor of load beyond temporal features.
2. **NHiTS / PatchTST support:** Integrate additional state-of-the-art transformer-based models for multi-horizon forecasting.
3. **Probabilistic Output Unification:** Extend all models to produce confidence intervals (currently only TFT supports native quantile output).
4. **Real POSOCO Data Integration:** Expand the India CSV pipeline to automatically parse and ingest POSOCO daily reports.
5. **Cross-State Spillover Modeling:** Implement graph-based spatial modeling to capture load transfers between neighboring states during supply shortfalls.
6. **Online Learning:** Implement incremental model updating as new data becomes available, without full retraining.

---

## 14. Appendix: Configuration and Dependency Tables

### Python Dependencies
| Library | Version | Purpose |
| :--- | :--- | :--- |
| `streamlit` | ≥ 1.30 | Web UI framework |
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scikit-learn` | ≥ 1.3 | DT, RF, metrics, scalers |
| `xgboost` | ≥ 2.0 | XGBoost model |
| `tensorflow` | ≥ 2.13 | LSTM, DLinear (Keras) |
| `torch` | ≥ 2.0 | xLSTM and TiRex backend |
| `tirex-ts` | ≥ 1.4 | TiRex architecture and pre-trained weights |
| `statsmodels` | ≥ 0.14 | SARIMA |
| `gridstatus` | ≥ 0.23 | US ISO data fetch |
| `holidays` | ≥ 0.39 | US and India holiday calendars |
| `plotly` | ≥ 5.15 | Interactive visualization |
| `scipy` | ≥ 1.10 | Diebold-Mariano test (statistics) |
| `joblib` | ≥ 1.3 | Model serialization |

### India States Coverage (INDIA_STATES Configuration)
| State/UT | Base (MU) | Region |
| :--- | :--- | :--- |
| Maharashtra | 420 | West |
| Gujarat | 380 | West |
| UP | 355 | North |
| Tamil Nadu | 330 | South |
| Rajasthan | 258 | North |
| Karnataka | 275 | South |
| MP | 285 | Central |
| Andhra Pradesh | 220 | South |
| Telangana | 205 | South |
| West Bengal | 195 | East |
| Haryana | 185 | North |
| Bihar | 125 | East |
| Odisha | 125 | East |
| Chhattisgarh | 90 | Central |
| Kerala | 87 | South |
| Delhi | 85 | North |
| Jharkhand | 80 | East |
| DVC | 75 | East |
| Assam | 38 | East |
| HP | 48 | North |
| Uttarakhand | 47 | North |
| J&K(UT) & Ladakh(UT) | 40 | North |
| DNHDDPDCL | 25 | West |
| BALCO | 21 | Central |
| Chandigarh | 9 | North |
| Arunachal Pradesh | 9 | East |
| Goa | 17 | West |
| Puducherry | 6 | South |
| Meghalaya | 10 | East |
| Manipur | 8 | East |
| Nagaland | 7 | East |
| Tripura | 8 | East |
| Mizoram | 6 | East |
| Sikkim | 5 | East |
| Railways_ER ISTS | 50 | East |
| Railways_NR ISTS | 52 | North |
| AMNSIL | 4 | East |

---

*Document Version: 1.0 — GridOne Framework — Research Paper Technical Reference*
