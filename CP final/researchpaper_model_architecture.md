# Model Architecture and Research Methodology

## 1. System Architecture Overview

GridOne is designed with a tiered, modular architecture to support multi-region electricity forecasting. The system integrates the following sequential processing layers:

1.  **Data Acquisition Layer:** Handles real-time API polling for US ISOs (via `gridstatus`) and executes a physics-informed synthetic generation engine for 37 Indian states.
2.  **Featurization Engine:** Implements high-dimensional feature engineering, including cyclic temporal encoding, multi-scale auto-regressive lags, and momentum-based derivatives.
3.  **Model Zoo:** A unified repository of 7 distinct forecasting families, ranging from classical econometrics to state-of-the-art foundation models.
4.  **Forecasting Strategy:** Employs an iterative multi-step approach for future extrapolation, maintaining state consistency across the prediction horizon.
5.  **Validation & Metrics Layer:** Executes rigorous statistical benchmarking using Nash-Sutcliffe Efficiency (NSE) and Diebold-Mariano tests.

---

## 2. Theoretical Framework of Forecasting Models

GridOne benchmarks seven distinct model architectures, each chosen for its unique ability to handle different time-series signals.

### 2.1 Classical Statistical: SARIMA
The **Seasonal AutoRegressive Integrated Moving Average** model serves as the primary statistical baseline. Formulated as $SARIMA(p,d,q)(P,D,Q)_s$, it captures linear dependencies and explicit seasonality ($s=24$ for hourly US data, $s=7$ for daily India data).

### 2.2 Ensemble learning: XGBoost
**Extreme Gradient Boosting** is the primary high-performance model for tabular feature sets. It utilizes a gradient-based optimization of a regularized objective function, effectively capturing complex non-linear interactions between temporal features and historical lags that neural networks often struggle to learn in low-data regimes.

### 2.3 Sequential Deep Learning: LSTM
**Long Short-Term Memory** networks are deployed to capture long-range temporal dependencies. The architecture comprises a two-layer stacked LSTM with dropout regularization, processing scaled sequences of load data to learn the historical continuity of the grid.

### 2.4 Spatial-Temporal Hybrid: CNN-LSTM
Developed specifically for the Indian daily mode, this **dual-branch architecture** processes state-level load and national-level load simultaneously.
*   **Branch 1 (local):** Uses 1D Convolutional layers to extract local patterns, followed by an LSTM layer.
*   **Branch 2 (global):** An independent LSTM branch that processes aggregated national demand, providing "macro-context" to the state-level prediction.
The branches are concatenated into a joint representation before passing through a dense fusion layer to produce the final MU (Million Units) forecast.

### 2.5 Decomposition-Based Linear: DLinear
**DLinear** challenges the hypothesis that complex Transformers are necessary for forecasting. It explicitly decomposes the series into a **Trend component** (via moving average pooling) and a **Seasonal component** (the residual). Two independent linear layers process these components before summing them for the final prediction, providing a highly stable and computationally efficient benchmark.

### 2.6 Foundation Model: TiRex (xLSTM)
At the frontier of the model suite is **TiRex**, a foundation model based on the **xLSTM** (extended Long Short-Term Memory) architecture. It overcomes traditional LSTM limitations through exponential gating and parallelized execution. TiRex is utilized in a **Zero-Shot** inference mode, leveraging its pre-trained understanding of international load patterns to provide accurate forecasts without requiring local retraining.

---

## 3. Iterative Multi-Step Forecasting Strategy

For multi-step-ahead forecasting, GridOne implements an **Autoregressive Feedback Loop**. At each future timestep $t+1$, the model's prediction $\hat{y}_t$ is appended back into the context window, and full feature engineering is recalculated for the subsequent step. This allows single-output architectures (like XGBoost and Decision Trees) to generate long-term forecast trajectories despite being trained on point-wise objectives.

## 4. Stability and Convergence

To ensure cross-model stability, the architecture incorporates:
*   **Early Stopping:** Monitoring validation R-squared to prevent overfitting in deep learning models.
*   **Gradient Clipping:** Applied to TiRex and LSTM backends to prevent exploding gradients during training on volatile load spikes.
*   **Feature Regularization:** Consistent Min-Max scaling across all architectures to ensure unified loss landscape normalization.
