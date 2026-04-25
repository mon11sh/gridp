# System Requirements Specification

This document details the software and hardware environments required to run, develop, and deploy the GridOne power grid load forecasting framework.

---

## 1. Software Requirements

GridOne is built primarily in Python and leverages a modular dependency architecture. The system identifies available libraries at runtime to enable or disable specific forecasting modules.

### 1.1 Operating System
- **Supported:** Windows 10/11, Ubuntu 20.04+, macOS (Intel and Apple Silicon).
- **Recommended:** Ubuntu 22.04 LTS for optimal GPU driver support and multi-processing efficiency.

### 1.2 Programming Language
- **Python 3.10 or higher** (Required for compatibility with `torch` 2.0+ and `tirex-ts`).

### 1.3 Core Dependencies
| Library | Minimum Version | Purpose |
| :--- | :--- | :--- |
| `streamlit` | 1.30.0 | Web interface and interactive dashboard |
| `pandas` | 2.0.0 | Data manipulation and time-series indexing |
| `numpy` | 1.24.0 | Numerical operations and array handling |
| `scikit-learn` | 1.3.0 | Machine learning baselines and data scaling |
| `plotly` | 5.15.0 | High-performance interactive visualizations |

### 1.4 Modeling Orchestration
- **Gradient Boosting:** `xgboost` ≥ 2.0.0
- **Deep Learning (Legacy/Standard):** `tensorflow` ≥ 2.13.0 (for LSTM and DLinear)
- **Deep Learning (Next-Gen):** `torch` ≥ 2.0.0 and `tirex-ts` ≥ 1.4 (for xLSTM Foundation Model)
- **Statistical Modeling:** `statsmodels` ≥ 0.14.0 (for SARIMA)

### 1.5 Data & Utilities
- **Data Fetching:** `gridstatus` ≥ 0.23.0 (for US market API)
- **Calendar Logic:** `holidays` ≥ 0.39.0 (multi-country holiday detection)
- **Serialization:** `joblib` ≥ 1.3.2 (model saving/loading)

---

## 2. Hardware Requirements

The hardware requirements scale based on whether the user is performing inference only or full model retraining.

### 2.1 Minimum Specifications (Development/Testing)
*Ideal for running the UI and performing zero-shot inference using TiRex.*
- **Processor:** 4-Core CPU (e.g., Intel Core i5 10th Gen / AMD Ryzen 5)
- **Memory (RAM):** 8 GB
- **Storage:** 500 MB available space (for codebase and compiled master data)
- **GPU:** Optional (CPU-based inference is supported but slower)

### 2.2 Recommended Specifications (Production/Full Training)
*Required for training XGBoost ensembles and fine-tuning Deep Learning models on 37+ States.*
- **Processor:** 8-Core+ CPU (e.g., Intel Core i7 12th Gen / AMD Ryzen 7)
- **Memory (RAM):** 16 GB to 32 GB (Highly recommended for massive data generation)
- **Storage:** 2 GB+ SSD (for large datasets and model checkpoints)

### 2.3 GPU Requirements (Deep Learning Acceleration)
To utilize the full potential of the **TiRex Foundation Model** and **Hybrid CNN-LSTM**, a dedicated GPU is strongly recommended:
- **Manufacturer:** NVIDIA (CUDA-enabled)
- **Model:** RTX 3060 / 4060 or higher (8 GB VRAM Minimum)
- **Platform:** CUDA Toolkit 11.8 or 12.1+

---

## 3. Environment Setup Recommendation

To avoid dependency conflicts, it is recommended to use a virtual environment:

### Using `venv` (Windows)
```powershell
# Create environment
python -m venv lenv

# Activate environment
.\lenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Hardware Optimization Notes
1. **CPU Parallelism:** XGBoost and Random Forest modules are configured with `n_jobs=-1` to utilize all available logical processors during training.
2. **TiRex Context Window:** The TiRex model is configured to use a 1024-step context for maximum pattern memory. On lower-end systems, inference time may increase significantly without GPU acceleration.
3. **Streamlit Multi-threading:** The framework is designed to handle multiple concurrent UI sessions, though memory usage will scale with the number of active users fetching massive datasets.
