# Results: Miscellaneous Insights and Extensions

## 1. Anomaly Detection Insights

In **GridOne**, anomalies are not just outliers to be removed; they are indicators of grid instability or extreme external events. Our residual-based detection system revealed several key insights:

### 1.1 Structural Failures vs. Predictable Shifts
*   **Holiday Dips:** While the model predicts standard holidays (e.g., Diwali or July 4th), "Extended Weekend" effects often cause deeper dips than expected. These are flagged as anomalies, suggesting that human recreational behavior is more volatile than industrial cycles.
*   **Thermal Shocks:** During heatwaves in the US (CAISO) and North India (Rajasthan/UP), actual demand often spiked 15-20% above the model's prediction. These anomalies were characterized by a $Z$-score $> 4.0$, pinpointing the exact start of "climatological stress" on the grid.
*   **Sensor Noise Recovery:** In the India Daily mode, the pre-processing outlier detection successfully recovered approximately 2% of the total dataset from "flatline" errors (where sensors reported 0 demand for several days), ensuring that the final $R^2$ remained above 0.95.

---

## 2. Future Forecasting Capability

Beyond historical testing, the framework demonstrates robust **iterative multi-step forecasting** (future extrapolation).

### 2.1 Horizon Stability
*   **Short-Term (1–7 Days):** All models, particularly XGBoost and DLinear, maintained high fidelity. The "Error Accumulation" was minimal, with MAPE increasing by less than 0.5% per day.
*   **Medium-Term (7–30 Days):** Predictability begins to decay due to the autoregressive feedback loop. However, the **TiRex (Foundation Model)** demonstrated superior stability here, with its "Zero-Shot" context awareness preventing the "drift" often seen in purely statistical models like SARIMA.
*   **Long-Term (30+ Days):** Forecasts become primarily qualitative, capturing the general trend and seasonal baseline rather than exact daily peaks.

### 2.2 Re-centering Mechanism
The implementation includes a "Real-Time Re-centering" logic where the forecast is updated as soon as newTelemetry data arrives. This reduces the multi-day drift and resets the error accumulation clock, making the system viable for operational daily planning.

---

## 3. Cross-Regional Insights (US vs. India)

The unique dual-region architecture of GridOne allowed for a comparative analysis of two fundamentally different energy markets.

| Insight Category | United States (ISO Market) | India (State Grid) |
| :--- | :--- | :--- |
| **Data Granularity** | Hourly (High Frequency) | Daily (Medium Frequency) |
| **Primary Volatility** | Weather & Industrial schedule | Rapid Growth & Festival Cycles |
| **Model Sensitivity** | High sensitivity to Time-of-Day Features | High sensitivity to Weekday/Weekend split |
| **"Duck Curve" Presence** | Strong (CAISO/NYISO) | Emerging in solar-rich states (Gujarat) |
| **Best Baseline Model** | DLinear (Handles intraday cycles) | Random Forest (Handles noise better) |

### 3.1 Key Findings
1.  **Industrial Dominance:** In industrial states like Maharashtra (IN) and PJM (US), the load is highly predictable with $R^2$ consistently $> 0.97$. Residential-heavy states exhibit more "noisy" stochastics.
2.  **Growth Trajectories:** The India region requires a linear trend component for successful $R^2 > 0.9$, reflecting the ~7% annual demand growth. In the US, demand is relatively trend-stationary, making season-only models highly effective.
3.  **Cross-State Correlations:** While the US ISOs operate relatively independently, the Indian states show a high "National Correlation" ($> 0.85$), justifying our **Hybrid CNN-LSTM Branch Architecture** which uses the National Grid as a context signal.
