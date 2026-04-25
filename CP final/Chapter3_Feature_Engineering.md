# CHAPTER 3: FEATURE ENGINEERING AND PREPROCESSING

## 3.1 Overview

Feature engineering is the process of transforming raw time-series data into a structured format that exposes the underlying patterns — seasonality, trends, and auto-regressive dependencies — to machine learning models. In **GridOne**, feature engineering is tailored to the specific granularity of the target market: hourly for the US ISO markets and daily for the Indian state-level grid.

## 3.2 US Hourly Feature Pipeline

For the US datasets, which represent high-frequency consumption with strong intraday rhythms, the feature set is designed to capture the "pulse" of the day.

### 3.2.1 Temporal and Calendar Features
Raw timestamps are decomposed into categorical features that capture periodic behavior:
*   **Hour of Day (0–23):** Captures the diurnal load curve (e.g., morning rise and evening peak).
*   **Day of Week (0–6):** Differentiates between industrial-heavy weekdays and residential-focused weekends.
*   **Month and Day of Year:** Captures broad seasonal shifts driven by heating and cooling demands.
*   **Binary Flags:** `IsWeekend` and `IsHoliday` (US Federal Holidays) are used to signal expected demand dips.

### 3.2.2 Auto-Regressive (Lag) Features
Lagged values allow the model to learn from previous grid states:
*   **Lag_1h:** The load at the previous hour, providing immediate continuity.
*   **Lag_24h:** The load at the same time yesterday, capturing daily periodicity.
*   **Lag_168h:** The load at the same time last week, capturing weekly periodicity.

### 3.2.3 Rolling Window Statistics
Statistical aggregations over sliding windows provide a measure of local trend and volatility:
*   **RollMean_3h & RollMean_24h:** Smooth out short-term noise and provide a baseline demand level.
*   **RollStd_24h:** Captures demand uncertainty and volatility within a 24-hour cycle.

---

## 3.3 India Daily Feature Pipeline

The Indian state-level data operate at a daily granularity. This requires a more complex, "history-rich" feature set to compensate for the lower frequency.

### 3.3.1 Cyclic Temporal Encoding
Treating months (1–12) or days (0–6) as linear integers creates a "boundary jump" where December (12) and January (1) appear distant despite being adjacent. GridOne solves this by mapping these features onto a unit circle using **Sine-Cosine Projection**:
$$ \text{Feature}_{\sin} = \sin\left(\frac{2\pi \times \text{value}}{\text{max\_period}}\right) $$
$$ \text{Feature}_{\cos} = \cos\left(\frac{2\pi \times \text{value}}{\text{max\_period}}\right) $$
This ensures that the model perceives the circular nature of time.

### 3.3.2 Multiscale Lag and Momentum Features
To capture the rapid growth and diverse holidays in India, an intensive auto-regressive set is implemented:
*   **Lags:** 1, 2, 7, 14, and 30 days.
*   **Load_Diff_1d:** The first-order derivative (rate of change) of the load.
*   **Trend_6d:** Calculated as $(Lag_1 - Lag_7)$, representing the weekly demand gradient.
*   **EWM_7d:** Exponentially Weighted Mean, which prioritizes recent observations while maintaining historical context.

---

## 3.4 Data Preprocessing and Cleaning

Before feature calculation, the raw telemetry undergoes a rigorous cleaning pipeline:

### 3.4.1 Resampling and Interpolation
*   **US Mode:** High-frequency raw data is resampled to a strict 1-hour frequency using mean aggregation. Missing gaps of up to 6 hours are filled using **time-weighted linear interpolation**; larger gaps are flagged as anomalies.
*   **India Mode:** Synthetic and real data are merged using a 3-day centered rolling mean at the junction points to ensure a seamless "hybrid" series.

### 3.4.2 Outlier Management
A rolling Z-score filter is applied to detect spikes:
$$ Z_t = \frac{x_t - \mu_{14d}}{\sigma_{14d}} $$
Observations with $|Z_t| > 3.5$ are treated as outliers and replaced via interpolation to prevent models from learning from transient sensor errors.

### 3.4.3 Feature Scaling
Most models in GridOne (especially LSTMs, DLinear, and SARIMA) are sensitive to the absolute scale of the input data. We utilize **Min-Max Scaling** to map features into the $[0, 1]$ range:
$$ X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}} $$
This ensures that the loss functions are stable and converge more rapidly during the training phase.

## 3.5 Data Leakage Prevention

A critical design rule in GridOne is the strict prevention of data leakage. All rolling statistics and lag features are calculated using a **shifted index** (e.g., `df['Load'].shift(1).rolling(window=24).mean()`). This ensures that at any timestep $t$, the model only has access to information from $t-1$ or earlier, mirroring real-world deployment conditions.
