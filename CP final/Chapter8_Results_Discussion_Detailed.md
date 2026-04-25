# CHAPTER 8: RESULTS, DISCUSSION, AND PERFORMANCE ANALYSIS

## 8.1 Experimental Setup and Metrics

The GridOne framework was benchmarked across two diverse geographic domains: the US ISO markets (hourly granularity) and the Indian State-level grid (daily granularity). To ensure a robust evaluation, we utilized holding-out cross-validation with a focus on the following metrics:

*   **Mean Absolute Percentage Error (MAPE):** Measures the relative accuracy, essential for comparing states with widely varying load magnitudes (e.g., Maharashtra vs. Sikkim).
*   **Nash-Sutcliffe Efficiency (NSE):** A rigorous efficiency metric where 1.0 indicates a perfect model and values above 0.75 represent "Very Good" performance in energy modeling.
*   **Statistic Significance (Diebold-Mariano):** Used to determine if the predictive superiority of one model over another is statistically significant.

---

## 8.2 Comparative Performance Matrix

The table below summarizes the average performance across all test instances in the GridOne suite, alongside computational efficiency metrics.

| Architecture | Model Family | US R² (Avg) | India R² (Avg) | Complexity | Training Speed | Inf. Latency | Memory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Gradient Boosed | **0.982** | **0.975** | $O(TDN \log N)$ | ~15s | 0.01s | Low |
| **DLinear** | Decomposition | 0.958 | 0.942 | **$O(L)$** | **~5s** | **0.005s** | **Negligible** |
| **TiRex** | xLSTM Foundation | 0.945 | 0.938 | $O(L \log L)$ | Zero-shot | 0.45s | Medium |
| **CNN-LSTM** | Spatial-Temporal | 0.921 | 0.948 | $O(L d_h^2)$ | ~120s | 0.12s | High |
| **LSTM** | Recurrent | 0.915 | 0.902 | $O(L d_h^2)$ | ~95s | 0.10s | Moderate |
| **Random Forest** | Bagging Ensemble | 0.892 | 0.875 | $O(TN \log N)$ | ~10s | 0.05s | Low |
| **SARIMA** | Statistical | 0.765 | 0.742 | $O(P^3)$ | ~300s | 2.50s | Low |

---

## 8.3 In-depth Discussion of Model Successes

### 8.3.1 The Undisputed Champion: XGBoost
XGBoost emerged as the primary "Champion" model across almost all regions. Its superiority is attributed to its ability to handle **high-dimensional tabular interactions**. Grid load is influenced by a complex interplay of "DayOfWeek", "Hour", and "Lagged Load". XGBoost's additive tree structure captures these non-linear thresholds (e.g., "If it is a weekend AND it is 8 PM, demand drops by X%") more effectively than the smooth activations of basic neural networks.

### 3.3.2 The DLinear "Simplicity" Paradox
One of the most significant findings in this research is the performance of **DLinear**. Despite having the fewest parameters, it outperformed the complex LSTM model in both US and India markets. 
*   **Reasoning:** Electricity load is inherently **trend-stationary and seasonal**. By explicitly decomposing the series into trend and seasonal components before applying linear layers, DLinear avoids the "vanishing gradient" and "overfitting" problems that plague deeper recurrent models on noisy energy data. It provides the best **accuracy-to-compute ratio** in the framework.

### 8.3.3 TiRex and the "Cold-Start" Advantage
While XGBoost requires historical data for every new state, **TiRex (xLSTM Foundation Model)** demonstrated exceptional **Zero-Shot stability**. In scenarios where historical telemetry was simulated as missing or extremely sparse (the "Cold-Start" scenario), TiRex maintained an $R^2 > 0.90$ by leveraging its pre-trained understanding of global electricity patterns. This highlights the value of foundation models in emerging grid territories where data sensors are recently installed.

### 8.3.4 CNN-LSTM and Hybrid Spatial Context
In the India Daily mode, the **CNN-LSTM** hybrid outperformed the standard LSTM by ~4%. The 1D-Convolutional layers acted as a "feature extractor" that identified local temporal patterns (e.g., 3-day load ramps) before the LSTM processed the sequence. Furthermore, the **Dual-Branch architecture**—which integrates national-level aggregate load as a context signal—proved vital for stabilizing predictions in smaller Indian states that are influenced by national grid frequency and supply conditions.

---

## 8.4 Statistical Significance Analysis

To move beyond simple averages, we conducted the **Diebold-Mariano (DM) test** comparing XGBoost (Champion) and SARIMA (Baseline).
*   **Results:** The DM test yielded a p-value $< 0.001$ for 92% of tested regions, indicating that the move from classical statistical modeling to modern gradient boosting provides a **statistically significant** improvement in forecast reliability.
*   **Residual Analysis:** Residual histograms for the top models followed a classic Gaussian distribution centered at zero, confirming that the models are **unbiased** and have successfully extracted all learnable information from the provided feature set, leaving only stochastic noise as error.

## 8.5 Conclusion of Discussion
The results validate the **Multi-Model philosophy** of GridOne. While XGBoost is the "Go-to" for maximum accuracy, DLinear offers a viable alternative for low-latency edge deployment, and TiRex provides a "Safety Net" for newly connected grid nodes with no historical data records.
