# ABSTRACT

Efficient and accurate Short-Term Load Forecasting (STLF) has become a cornerstone of the modern electrical grid’s stability, especially as the global energy landscape transitions toward a high-frequency, variable-renewable-heavy paradigm. This project presents **GridOne**, a unified, multi-region framework designed to bridge the gap between high-transparency markets (United States ISOs) and emerging grid territories (Indian National Grid). 

GridOne addresses the chronic fragmentation in forecasting tools by integrating a modular, scalable architecture that supports seven distinct modeling families, ranging from classical statistical methods (SARIMA) and gradient-boosted ensembles (XGBoost) to state-of-the-art deep sequential models (LSTMs, CNN-LSTM hybrids) and time-series foundation models (TiRex based on xLSTM). To overcome the data accessibility barriers in the Indian subcontinent, the framework implements a physics-informed synthetic generation engine that simulates realistic state-level demand patterns for 37 States and Union Territories.

Our benchmarking results, validated through robust statistical frameworks such as Nash-Sutcliffe Efficiency (NSE) and Diebold-Mariano tests, demonstrate that GridOne provides a superior predictive baseline. Specifically, XGBoost and DLinear models show exceptional stability in US hourly markets ($R^2 \approx 0.98$), while the CNN-LSTM hybrid effectively captures the spatial-temporal volatile demand signatures of Indian states ($R^2 \approx 0.95$). Furthermore, the integration of the TiRex foundation model demonstrates successful "Zero-Shot" forecasting capabilities, offering a promising solution for "cold-start" grid scenarios where historical telemetry is unavailable. 

GridOne establishes an open-source, extensible infrastructure for grid operators and energy researchers, providing the necessary tools to navigate the complexities of a decarbonized, decentralized, and digitalized energy future.

**Keywords**: Power Grid Load Forecasting, XGBoost, LSTM, xLSTM, Foundation Models, Multi-Region Energy Analytics, Synthetic Data Generation.
