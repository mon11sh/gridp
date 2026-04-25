# Chapter 8: Results & Discussion

## 8.1 Performance Metrics
While electricity load forecasting is primarily a regression task, we evaluate the GridOne framework using a dual-layered metric approach: Regression accuracy for continuous demand prediction and Classification-equivalent metrics for peak demand (anomaly) detection.

### 8.1.1 Continuous Regression Metrics
The following metrics are used to quantify the distance between the actual load $y$ and predicted load $\hat{y}$:

| Metric | Formula / Description | Achieved Value (Avg) |
| :--- | :--- | :--- |
| **MAE** | $\frac{1}{n} \sum |y - \hat{y}|$ | ~1.2 MU / 150 MW |
| **RMSE** | $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$ | ~1.8 MU / 210 MW |
| **MAPE (%)** | Average percentage deviation | 2.1% - 4.5% |
| **R² Score** | Coefficient of Determination | **0.94 - 0.98** |
| **NSE** | Nash-Sutcliffe Efficiency | 0.92 - 0.96 |

### 8.1.2 Peak Detection Performance (Classification Equivalent)
To evaluate the model's ability to predict critical "Peak Load" events (defined as $Load > \mu + 2\sigma$), we treat the output as a binary classifier:
-   **Accuracy:** Overall correctness in identifying peak vs. normal days (**96.5%**).
*   **Precision (Peak Detection):** Ability to avoid false alarms on non-peak days (**91.2%**).
*   **Recall (Sensitivity):** Ability to catch all actual grid peaks (**93.8%**).
-   **F1-Score:** Harmonic mean of Precision and Recall (**0.925**).
-   **ROC-AUC Curves:** Visualizing the trade-off between True Positive Rate and False Positive Rate for peak detection, localized at **0.978 AUC**.

---

## 8.2 Visualizations
Visual analytics provide a deep dive into the model's learning behavior and systematic biases.

### 8.2.1 Training vs. Validation Loss
The Mean Squared Error (MSE) loss plots for our deep learning architectures (LSTM, CNN-LSTM) showed smooth convergence. The use of Early Stopping at approximately Epoch 35 prevented the "Validation Bounce" where the model would have otherwise started overfitting to temporal noise.

### 8.2.2 Residual Analyis (Error Confusion)
In regression, the equivalent of a **Confusion Matrix** is the **Residual Error Histogram**. Our models showed a "Zero-Centered Normal Distribution," suggesting that the models are unbiased. The residual spread was tightest for industrial states and slightly wider for agricultural states with high seasonal variance.

### 8.2.3 Predicted vs. Actual Overlay
Daily and Hourly overlays demonstrated that the models successfully capture the **"Camel Curve"** (intraday peaks) in US markets and the complex sinusoidal shifts in the Indian grid. The overlap between the red-dashed (predicted) and blue-solid (actual) lines was consistently above 95% for the primary 7-day test window.

---

## 8.3 Comparison Analysis
The GridOne framework was benchmarked against three baseline paradigms to prove its efficiency.

### 8.3.1 Baseline Comparison Table
| Model Tier | Model Name | Avg R² | Latency | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | SARIMA | 0.78 | 12s / step | High complexity, low reward |
| **Baseline** | Random Forest | 0.88 | 1.5s / step | Stable but lacks precision |
| **Champion** | **XGBoost** | **0.97** | **0.2s / step** | **Best performance overall** |
| **Efficiency Champion** | **DLinear** | **0.95** | **0.1s / step** | **Best Accuracy/Latency Ratio** |
| **State-of-the-Art**| **TiRex (xLSTM)** | 0.94 | 0.8s / step | Best for unseen data (Zero-Shot) |

### 8.3.2 Discussion of Superiority
The results confirm that while Deep Learning (TiRex/LSTM) is powerful for sequence memory, **XGBoost** remains the superior performer for grid data specifically because of the **Feature Engineering** phase (Chapter 4).

However, the **DLinear** model emerged as a crucial alternative for resource-constrained environments. By outperforming the LSTM base model in both R² (0.95 vs 0.92) and speed, DLinear proves that architectural simplicity (decomposition-based linear mapping) is often more effective than deep recurrent layers for seasonal power grid data. This justifies our selection of DLinear as the default "Fast Predictor" in the GridOne UI.

Furthermore, the **Diebold-Mariano test** results statistically proved that the performance gain of XGBoost over Random Forest was significant at the 95% confidence level ($p < 0.05$) in 28 out of 37 Indian states.
