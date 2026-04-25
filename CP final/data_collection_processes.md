# Data Collection and Processing Methodology

This document outlines the systematic processes employed for data collection, preparation, and feature engineering within the GridOne power grid load forecasting framework. The system manages two distinct datasets: high-frequency hourly data for United States energy markets and regional daily data for the Indian power grid.

---

## 1. Data Sources and Acquisition

The framework utilizes a multi-source acquisition strategy to ensure a robust representation of power grid dynamics across different geographic and economic contexts.

### 1.1 US Energy Markets (Hourly)
Data for the United States is retrieved using the `gridstatus` library, which provides a unified API to access various Independent System Operators (ISOs) and Regional Transmission Organizations (RTOs).
- **Supported Markets:** CAISO (California), PJM (Mid-Atlantic), MISO (Midcontinent), NYISO (New York), SPP (Southwest), and ISONE (New England).
- **Frequency:** Hourly interval data.
- **Metric:** Total system load/demand.

### 1.2 Indian Power Grid (Daily)
The Indian dataset is constructed through a hybrid approach combining primary source compilation and synthetic augmentation.
- **Primary Source:** Daily Power Supply Position (PSP) reports from the **National Load Despatch Centre (NLDC)**.
- **Compilation:** Excel reports (located in `gridone/data/Cleaned_Data2`) are processed and synthesized into a master record (`india_master_data.csv`).
- **Coverage:** State-level and Union Territory-level daily demand data.

---

## 2. Hybrid Data Generation (India)

To facilitate granular state-level analysis where historical records might be sparse or inconsistent, GridOne employs a sophisticated **Synthetic Data Augmentation Engine**.

### 2.1 State-Specific Configuration
Each of the 37 Indian states/UTs is assigned a configuration profile including:
- **Base Load:** The baseline power demand.
- **Noise Level:** Volatility parameters characteristic of that region.
- **Regional Grouping:** Classification into North, Central, West, South, or East regions to model climatic and seasonal similarities.

### 2.2 Augmentation Logic
For states with partial data, the system generates synthetic loads based on:
- **Sinusoidal Seasonality:** Modeling annual temperature variations and crop cycles.
- **Linear Trend:** Accounting for infrastructure growth and urbanization.
- **Weekly Patterns:** Capturing reduced industrial demand during weekends.
- **Gaussian Noise:** Introducing stochasticity inherent in real-world energy consumption.

**Data Merging:** Real-world NLDC data always takes precedence. Where historical records exist for a specific date and state, they overwrite synthetic values, ensuring the model remains grounded in reality.

---

## 3. Data Refinement and Integrity

Raw data undergoes a multi-stage cleaning pipeline to ensure high $R^2$ performance during model training.

1.  **Resampling and Interpolation:** Hourly data is resampled to ensure no gaps exist in the time series. Missing values (up to 6 consecutive hours) are filled using time-based linear interpolation.
2.  **Outlier Removal:** A local Z-score mechanism identifies anomalies. For daily data, a 14-day rolling window is used to detect and remove demand spikes that exceed 3.5 standard deviations.
3.  **Short-term Smoothing:** A 3-day (daily) or multi-hour (hourly) rolling mean is applied to "stitch" together real and synthetic data transitions, reducing raw noise without losing seasonal signal.

---

## 4. Feature Engineering

The system transforms raw load values into a high-dimensional feature set designed to capture temporal hierarchies and momentum.

### 4.1 Temporal & Cyclic Features
- **Calendar Components:** Extraction of Year, Month, Day, Hour, Day of Week, and Season.
- **Cyclic Encoding:** Month and Day are transformed into Sine and Cosine components ($sin(2\pi x/T)$, $cos(2\pi x/T)$). This preserves the continuity between 11:00 PM and 1:00 AM, or December and January.
- **Holiday Integration:** Comprehensive holiday lists for both US and India are utilized to account for significant shifts in commercial and industrial load.

### 4.2 Auto-regressive & Statistical Features
- **Lag Features:**
    - **Hourly:** 1h, 24h (1 day), 168h (1 week).
    - **Daily:** 1d, 2d, 7d, 14d, 30d.
- **Rolling Statistics:** Rolling means, standard deviations, and min/max values across various windows (3h, 24h, 3d, 7d).
- **Momentum Features:** First-order differences (e.g., $Load_{t-1} - Load_{t-2}$) to capture the gradient of demand changes.
- **Volatility Interaction:** Ratio of rolling standard deviation to rolling mean to model demand stability.

---

## 5. Storage and Structure

Processed data is organized logically within the project directory:

```text
gridone/
├── data/
│   ├── Cleaned_Data2/      # Raw NLDC Excel reports
│   └── india_master_data.csv # Compiled historical data
└── data.py                 # Core logic for fetching and processing
```

This standardized pipeline ensures that models receive consistent, high-quality inputs regardless of the target market or region.
