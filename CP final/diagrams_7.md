# System Architecture and Process Diagrams (Chapter 7)

This document contains standardized Mermaid visualizations for the **GridOne** forecasting framework, covering data movement, feature transformations, and system interactions.

---

## 1. Data Flow Diagrams (DFD)

### 1.1 Level 0: Global Data Flow
Visualizing the end-to-end journey from the physical power grid sensors (via APIs) to the operator's dashboard.

```mermaid
graph LR
    subgraph Storage [Persistent Layer]
        DB[(Historical Load CSVs)]
    end

    subgraph Sources [Data Sources]
        API1[GridStatus - US ISOs]
        API2[NLDC - India States]
    end

    Sources --> Ingest[Data Ingestion Controller]
    DB --> Ingest
    
    Ingest --> Clean[Cleaning & Imputation]
    Clean --> FE[Feature Engineering Engine]
    FE --> Model[Multi-Model Inference]
    
    Model --> UI[Streamlit Dashboard]
    UI --> User((Grid Operator))
```

### 1.2 Level 1: Modeling Pipeline
Detailing the internal data lifecycle within the deep learning forecasting modules.

```mermaid
graph TD
    Input[Aggregated Datetime Series] --> Resample[Resample: Hourly/Daily]
    Resample --> Interpolate[Linear Interpolation]
    Interpolate --> Norm[StandardScaler / Z-Score]
    
    subgraph SequenceProcessing [Sequence Preparation]
        Norm --> Slice[Rolling Window 24h/168h]
        Slice --> Tensor[3D Tensor Generation]
    end
    
    Tensor --> Core[Neural Core: LSTM / DLinear / TFT]
    Core --> Out[Normalized Output]
    Out --> DeNorm[Inverse Transformation]
    DeNorm --> Final[Final Forecast: MW / MU]
```

---

## 2. Feature Engineering Process Diagram

Visualization of the transformation logic where raw timestamps and load values are converted into predictive signals.

```mermaid
flowchart TD
    Raw[Raw Timestamped Load] --> Time[Time Component Extractor]
    Raw --> Hist[Historical Lag Generator]
    
    subgraph Temporal [Temporal Features]
        Time --> Cyclic[Cyclic Encoding: Sin/Cos]
        Time --> Binary[Weekend / Holiday Flags]
    end
    
    subgraph AutoReg [Auto-Regressive Features]
        Hist --> Lags[Lags: 1h, 24h, 168h]
        Hist --> Roll[Rolling Mean/Std: 3h, 24h]
    end
    
    Cyclic & Binary & Lags & Roll --> Concat[Feature Vector Concatenation]
    Concat --> Train[Model Training Input]
    Concat --> Pred[Real-Time Inference Input]
```

---

## 3. UML Diagrams

### 3.1 Use Case Diagram
User interactions with the GridOne forecasting system.

```mermaid
useCaseDiagram
    actor "Researcher / Analyst" as Analyst
    actor "System Operator" as Operator
    
    package "GridOne Forecasting Platform" {
        usecase "Compare Models (R², MAPE)" as UC1
        usecase "Select Geographic Market" as UC2
        usecase "Analyze Anomalies" as UC3
        usecase "Generate Future Forecasts" as UC4
        usecase "Hyperparameter Tuning" as UC5
    }
    
    Analyst --> UC1
    Analyst --> UC2
    Analyst --> UC5
    
    Operator --> UC2
    Operator --> UC3
    Operator --> UC4
```

### 3.2 System Sequence Diagram
The interaction between the UI layer and the modeling orchestration logic.

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant DP as Data Processor
    participant ME as Model Engine
    
    User->>UI: Select Market & Date Range
    UI->>DP: Fetch Raw Series
    DP-->>UI: Return Cleaned & Standardized Data
    
    User->>UI: Click 'Run Forecast'
    UI->>DP: Engineer Features (Lags, Cyclic)
    DP-->>ME: Transmit Processed Feature Vector
    
    ME->>ME: Compute Multi-Model Ensemble
    ME-->>UI: Return Forecast Tensors & Metrics
    UI-->>User: Render Interactive Visuals (Plotly)
```

### 3.3 Class Diagram (Logical Architecture)
Representation of the modular code structure of the forecasting framework.

```mermaid
classDiagram
    class DataIngestor {
        +fetch_us_iso(market)
        +load_india_states()
        +clean_load_data(df)
    }
    
    class FeatureEngineer {
        +add_lags(df, window)
        +add_cyclic_features(df)
        +get_holiday_flags(dates)
        +create_sequences(data, lookback)
    }
    
    class ModelEngine {
        +train_xgboost(X, y)
        +train_lstm(X, y)
        +predict_future(model, horizon)
        +evaluate_metrics(y_true, y_pred)
    }
    
    class DashboardUI {
        +render_sidebar()
        +plot_comparison(results)
        +display_metrics(metrics)
    }
    
    DataIngestor ..> FeatureEngineer : feeds
    FeatureEngineer ..> ModelEngine : feeds
    DashboardUI --> DataIngestor : triggers
    DashboardUI --> ModelEngine : triggers
```
