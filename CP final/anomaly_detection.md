# Anomaly and Outlier Detection Framework

## 1. Overview
Anomaly detection in **GridOne** is implemented as a two-stage process. Unlike many forecasting projects that treat anomalies as an afterthought, GridOne integrates detection directly into the **Data Pipeline (Input cleaning)** and the **Evaluation Suite (Result analysis)**. This ensures that models are both trained on clean data and evaluated for their ability to handle real-world shocks.

---

## 2. Stage 1: Data-Level Outlier Removal (Preprocessing)
Before any model training begins, the raw load telemetry (specifically in the Indian Grid mode) is subjected to a **Local Z-Score filter**. This is crucial for "cleaning" the training data from sensor malfunctions or transmission errors without losing the underlying periodic signals.

### 2.1 Methodology: Sliding Z-Score
Instead of using a global mean, which would be skewed by seasonal trends, GridOne uses a **14-day centered rolling window**.
*   **Metric:** For each point $x_t$, the local mean $\mu_t$ and local standard deviation $\sigma_t$ are calculated over a $\pm$7-day window.
*   **Detection Criteria:** An observation is flagged as an outlier if:
    $$ |x_t - \mu_t| > 3.5 \times \sigma_t $$
*   **Resolution:** Detected outliers are set to null and then recovered using **Linear Interpolation**, ensuring a smooth, continuous series for deep learning models.

---

## 3. Stage 2: Forecast-Based Anomaly Detection (Evaluation)
Once a model has generated predictions, GridOne performs a "Proper Anomaly Detection" by analyzing the **Residual Distribution**.

### 3.1 The Principle of Residual Analysis
The core hypothesis is that a well-trained model captures the "normal" behavior of the grid. Therefore, any significant deviation between the predicted load and the actual load (the residual) represents an anomaly in the grid itself (e.g., a sudden heatwave, industrial shutdown, or localized blackout).

### 3.2 Implementation: Residual Z-Score
Implemented in `metrics.py`, the `detect_anomalies` function performs the following:
1.  **Residual Calculation:** $E_t = Y_{true,t} - \hat{Y}_{pred,t}$
2.  **Standardization:** The residuals are standardized to have a mean of 0 and a standard deviation of 1.
3.  **Thresholding:** A threshold of $3\sigma$ (standard deviations) is applied. Any point where the residual exceeds $3\sigma$ is flagged as a statistically significant anomaly.

### 3.3 Visualization in Dashboard
These anomalies are automatically rendered in the **Streamlit UI** using:
*   **Orange 'X' markers** overlaid on the main timeline chart.
*   **Hover Tooltips** that provide the exact magnitude of the anomaly.
This allows grid researchers to immediately pinpoint timestamps where the grid behavior deviated from historical patterns.

---

## 4. Why This is "Proper" Detection
*   **Context Awareness:** By using a 14-day local window for preprocessing, we avoid flagging seasonal peaks as outliers.
*   **Model-Agnostic:** The forecast-based detection works across all 7 model families (XGBoost, LSTM, etc.).
*   **Statistically Rigorous:** Using $3.5\sigma$ for data cleaning and $3\sigma$ for forecast anomalies adheres to standard 6-sigma statistical quality control principles, minimizing false positives while capturing extreme events.
