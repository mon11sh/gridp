# Chapter 6: Statistical Validation, Comparative Analysis, and Dashboard Integration

## 6.1 Introduction
The final objective of the GridOne framework is to translate raw algorithmic performance into a validated, actionable, and user-centric decision support tool. Stage 3 focuses on **Comparative Analysis & Optimization**, where we apply rigorous statistical testing to verify model superiority and integrate the entire suite into an interactive Streamlit dashboard. 

This phase ensures that the models developed in Stage 2 (Chapter 5) are not just "accurate" in a mathematical sense, but are robust against edge cases, efficient enough for real-time operation, and explainable to energy grid operators through advanced visualizations.

## 6.2 Problem Statement
Transitioning from a training environment to a practical deployment environment presents several production-level hurdles:
1.  **Statistical Ambiguity:** High $R^2$ scores can be misleading if the difference between two models is not statistically significant. A mechanism is needed to prove one model is truly superior to another beyond random chance.
2.  **Inference Latency:** Deep learning models like TiRex or large ensembles like Random Forest can introduce significant delays in a web application if not optimized for concurrent users.
3.  **Visualization Complexity:** Representing multi-model outcomes, residual distributions, and future uncertainty overlays on a single interface without overwhelming the user.
4.  **Stealth Failure Detection:** Models may perform well on average but fail catastrophically during "black swan" events (e.g., sudden industrial grid shifts), requiring automated anomaly detection at runtime.

## 6.3 Methodology
To address these challenges, Stage 3 implements a rigorous evaluation framework and a high-performance deployment architecture.

### 6.3.1 Statistical Validation: The Diebold-Mariano (DM) Test
To mathematically compare forecasting accuracy between any two models (e.g., XGBoost vs. LSTM), we implemented the **Diebold-Mariano test**. We define the loss differential $d_t$ between two forecast error series $e_{1,t}$ and $e_{2,t}$ as:
$$d_t = |e_{1,t}| - |e_{2,t}|$$
The DM statistic is then calculated as:
$$DM = \frac{\bar{d}}{\sqrt{\text{Var}(d) / N}}$$
where $\bar{d}$ is the sample mean of the loss differential. A p-value is derived to determine if the null hypothesis (both models have equal accuracy) can be rejected, providing a scientific basis for model selection.

### 6.3.2 Computational Optimization: Resource Caching and Quantization
The framework utilizes Streamlit's `@st.cache_resource` and `@st.cache_data` decorators to modularize memory usage. 
-   **Model Caching:** Large models like TiRex (35M parameters) are loaded into VRAM only once per session, preventing redundant I/O operations.
-   **Context Pruning:** For real-time US market fetching, we limit the historical lookback to the minimum required for the specified model’s context window, reducing the payload size of API calls by up to 60%.

### 6.3.3 Dashboard Architecture and Visualization
We developed a dual-region interactive dashboard using **Plotly** and **Streamlit**. The UI architecture follows a "drill-down" logic:
1.  **Overview Map/Bar Chart:** Visualizing demand distribution across all states/ISOs.
2.  **Comparative Timeline:** Overlaying forecasts from all 7 models against the ground truth.
3.  **Residual Analysis Subplots:** Real-time generation of error distribution histograms to detect heteroscedasticity.
4.  **Future Forecasting:** Utilizing an iterative multi-step approach to project load into the future (7-30 days) based on model memory.

### 6.3.4 Residual-Based Anomaly Detection
An automated monitor flags anomalies where prediction residuals exceed 3 standard deviations ($Z > 3$). These are visually marked on the dashboard as high-risk periods, alerting operators to potential grid events that exceeded the model's learned patterns.

## 6.4 Results & Discussion
The Stage 3 optimization efforts transformed the framework from a research script into a production-ready application.

### 6.4.1 Accuracy vs. Speed Trade-offs
Analysis showed that while **SARIMA** provides a strong statistical baseline, it is the slowest to compute due to its sequential MLE optimization. In contrast, **DLinear** surfaced as the most efficient model in our suite, achieving comparable accuracy to LSTM but with a **15x reduction in inference latency**. This makes it the optimal choice for the real-time responsive dashboard, where user-latency must be kept under 500ms for seamless interaction.

### 6.4.2 Diebold-Mariano Insights
In 85% of tests on US market data (PJM, CAISO), the DM test confirmed that **XGBoost** is statistically superior to both Random Forest and simple LSTM ($p < 0.05$). However, for the high-volatility Indian states (e.g., Rajasthan), the DM test often showed no significant difference between TiRex and XGBoost, indicating that the foundation model and tree ensemble are equally robust in those environments.

### 6.4.3 Effectiveness of Dashboard Integration
User testing demonstrated that the "Anomalies Overlay" was the most valued feature for grid operators, as it correctly identified demand spikes associated with seasonal festival periods in India that were previously obscured in raw data tables. The final dashboard successfully maintains a sub-2.5 second refresh rate even when evaluating all 7 models simultaneously on a standard machine.
