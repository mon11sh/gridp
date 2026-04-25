# Implementation Details and Software Architecture

## 1. Technological Stack

**GridOne** is implemented exclusively in **Python 3.10+**, leveraging a heterogeneous mix of specialized libraries for data engineering, model training, and interactive visualization.

### 1.1 Core Development Frameworks
*   **Frontend & UI Orchestration:** **Streamlit** (v1.30+). Streamlit was selected for its reactive programming model and built-in support for asynchronous data loading (caching), which is critical for handling large power grid datasets.
*   **Machine Learning (Classical & Ensembles):** **Scikit-learn** and **XGBoost**. These handle the tabular forecasting models and feature scaling.
*   **Deep Learning Backends:**
    *   **TensorFlow/Keras:** Power the DLinear and LSTM architectures.
    *   **PyTorch:** Provides the underlying compute engine for the **TiRex (xLSTM)** foundation model.
*   **Data Processing:** **Pandas** and **NumPy**. Optimized for vectorized time-series operations and cyclic encoding.
*   **Visualization:** **Plotly**. Chosen over Matplotlib for its GPU-accelerated interactive web charts.

---

## 2. System modularity

The project follows a **Separation of Concerns (SoC)** philosophy, organized into discrete functional modules:

| Module | Responsibility |
| :--- | :--- |
| `app.py` | Main entry point; handles routing between US and India modes. |
| `config.py` | Environment detection, holiday calendars, and global hyperparameter constants. |
| `data.py` | Data ingestion (ISO API fetches & Synthetic generation) and feature engineering. |
| `training.py` | Abstracted model training interface for all 7 model families. |
| `forecasting.py` | Multi-step iterative forecasting logic and future extrapolation. |
| `metrics.py` | Statistical validation (MAE, RMSE, R², NSE, DM-Test). |
| `visualization.py` | Plotly-based reactive chart generation components. |

---

## 3. High-Performance Implementation Features

### 3.1 Adaptive Dependency Management
A key implementation detail is the "Graceful Degradation" system in `config.py`. The system checks for the presence of GPU-heavy libraries (like TensorFlow or XGBoost) at runtime. If a library is missing, the corresponding model is automatically disabled in the UI instead of crashing the application. This allows GridOne to run on low-power laptops as well as high-end GPU workstations.

### 3.2 Computational Optimization
*   **Stateful Caching:** We utilize `@st.cache_data` with a 1-hour Time-To-Live (TTL). This ensures that expensive operations, such as fetching real-time data from US ISO API servers, are only performed once per session.
*   **Parallelization:** Ensemble models (Random Forest, XGBoost) are configured with `n_jobs=-1` to utilize 100% of available CPU cores during training.
*   **CUDA Acceleration:** TiRex and standard LSTM models automatically detect and mount to the NVIDIA GPU (if available) to speed up sequential processing.

---

## 4. UI/UX Design Philosophy

The **Streamlit Dashboard** is designed for high-density information display:
1.  **Sidebar-Driven Control:** Users can toggle regions, select specific states/ISOs, adjust forecast horizons, and choose multiple models for head-to-head comparison without reloading the page.
2.  **Unified Hover Interaction:** All Plotly charts use `hovermode='x unified'`, allowing researchers to compare predictions across multiple models at a specific timestamp accurately.
3.  **Real-Time Metrics Grid:** Metrics are displayed in a responsive grid, updating dynamically as each model completes its training/forecasting cycle.
4.  **Anomaly Highlighting:** The UI automatically injects orange 'X' markers on historical timelines where residuals exceed the $3\sigma$ threshold, providing immediate visual diagnostic feedback.

---

## 5. Deployment and Scalability

GridOne is designed for containerized deployment (e.g., via Docker) or hosting on Streamlit Cloud. The implementation includes:
*   **Timezone Normalization:** All data is converted to UTC-naive timestamps at the ingestion layer to prevent synchronization issues between international regional datasets.
*   **Memory Management:** Large synthetic datasets for India (37 states) are generated on-demand and stored in the Streamlit Session State to prevent persistent memory leaks during long research sessions.
