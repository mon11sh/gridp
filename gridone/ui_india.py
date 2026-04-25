import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import INDIA_STATES, XGBOOST_AVAILABLE, LSTM_AVAILABLE, SARIMA_AVAILABLE, TIREX_AVAILABLE, DLINEAR_AVAILABLE
from data import generate_synthetic_india_data, load_real_india_data, engineer_features_daily, create_lstm_sequences
from training import train_decision_tree_model, train_random_forest_model, train_xgboost_model, train_lstm_model, train_cnn_lstm_model, train_sarima_model, train_dlinear_model
from forecasting import iterative_future_forecast_sklearn_daily, iterative_future_forecast_xgboost_daily, iterative_future_forecast_lstm_daily, iterative_future_forecast_cnn_lstm_daily, iterative_future_forecast_dlinear_daily, future_forecast_tirex
from metrics import calculate_metrics, nse, diebold_mariano_test, detect_anomalies
from visualization import plot_load_history, plot_india_national_overview, plot_full_timeline, plot_model_comparison, plot_metrics_comparison, plot_residuals, plot_future_overlay

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go

if XGBOOST_AVAILABLE:
    import xgboost as xgb

def run_india_mode(sidebar):
    with sidebar:
        st.header('⚙️ India Configuration')
        state_name = st.selectbox('Select State', list(INDIA_STATES.keys()))
        st.subheader('Data Range')
        # Load real data info for better defaults
        real_df = load_real_india_data()
        default_start = datetime(2023, 4, 1).date()
        default_end = datetime(2025, 10, 31).date()
        
        if real_df is not None:
             state_real = real_df[real_df['States'] == state_name]
             if not state_real.empty:
                 default_start = state_real['Date'].min().date()
                 default_end = state_real['Date'].max().date()

        c1, c2 = st.columns(2)
        with c1: start_date = st.date_input('Start', value=default_start)
        with c2: end_date   = st.date_input('End',   value=default_end)
        days_total = (end_date - start_date).days
        st.caption(f'📅 {days_total} days of synthetic data')
        st.subheader('Models')
        use_dt     = st.checkbox('Decision Tree',  value=True,  key='in_dt')
        use_rf     = st.checkbox('Random Forest',  value=True,  key='in_rf')
        use_xgb    = st.checkbox('XGBoost',        value=XGBOOST_AVAILABLE, key='in_xgb', disabled=not XGBOOST_AVAILABLE)
        use_lstm   = st.checkbox('LSTM',            value=LSTM_AVAILABLE,    key='in_lstm', disabled=not LSTM_AVAILABLE)
        use_sarima = st.checkbox('SARIMA (slow)',   value=False, key='in_sarima', disabled=not SARIMA_AVAILABLE)
        use_tirex    = st.checkbox('TiRex (xLSTM)', value=TIREX_AVAILABLE, key='in_tirex', disabled=not TIREX_AVAILABLE)
        use_dlinear = st.checkbox('DLinear',          value=DLINEAR_AVAILABLE, key='in_dlinear', disabled=not DLINEAR_AVAILABLE)
        st.subheader('Parameters')
        max_test_in = max(15, days_total // 4)
        test_days   = st.slider('Test Set (days)', 7, min(180, max_test_in), min(60, max_test_in))
        forecast_future = st.checkbox('Forecast Future', value=True, key='in_ff')
        future_days = st.slider('Days Ahead', 7, 90, 30) if forecast_future else 0
        st.divider()
        show_national = st.checkbox('Show National Overview', value=True)
        run_btn = st.button('🚀 Run India Forecast', type='primary', use_container_width=True)

    if not run_btn:
        st.info('👈 Select a state and click **Run India Forecast**')
        with st.expander('ℹ️ About India Mode'):
            st.markdown('''
**India Grid Forecasting** uses synthetic data generated for all 37 states based on realistic demand baselines,
seasonal patterns (summer/winter peaks), weekly cycles, and year-on-year growth.

**Metrics are identical to US mode:** MAE · RMSE · MAPE · R² · NSE · Diebold-Mariano Test

**Frequency:** Daily (vs hourly for US)  
**Target:** Day Demand (MU — Million Units)
            ''')
        if show_national:
            st.subheader('🗺️ National Overview (Synthetic Baseline)')
            nat_end   = datetime(2026, 1, 31).date()
            nat_start = datetime(2025, 1, 1).date()
            all_states_frames = []
            with st.spinner('Generating national data…'):
                for sn in INDIA_STATES:
                    sdf = generate_synthetic_india_data(sn, nat_start, nat_end)
                    sdf['State'] = sn
                    all_states_frames.append(sdf)
            national_df = pd.concat(all_states_frames)
            st.plotly_chart(plot_india_national_overview(national_df), use_container_width=True)
            total_by_date = national_df.groupby(national_df.index)['Load'].sum()
            fig_nat = go.Figure()
            fig_nat.add_trace(go.Scatter(x=total_by_date.index, y=total_by_date.values,
                                         mode='lines', name='National Total',
                                         line=dict(color='#e74c3c', width=2)))
            fig_nat.update_layout(title='National Total Daily Demand (All 37 States)',
                                  xaxis_title='Date', yaxis_title='Total Demand (MU)', height=350)
            st.plotly_chart(fig_nat, use_container_width=True)
        return

    with st.spinner(f'⚙️ Generating synthetic data for {state_name}…'):
        df = generate_synthetic_india_data(state_name, start_date, end_date)

    st.success(f'✅ Generated {len(df)} days ({df.index.min().date()} → {df.index.max().date()})')

    with st.expander('📊 Raw Synthetic Data'):
        st.dataframe(df.tail(20), use_container_width=True)
        st.plotly_chart(plot_load_history(df, 'MU'), use_container_width=True)

    if len(df) < test_days + 30:
        st.error('Not enough data for selected test size.'); return

    df_features = engineer_features_daily(df)
    
    # Chronological Split
    total_len = len(df_features)
    test_idx = total_len - test_days
    
    train_feat = df_features.iloc[:test_idx]
    test_feat  = df_features.iloc[test_idx:]
    
    # We need train_df and test_df for some models (LSTM/SARIMA)
    train_df = df.loc[train_feat.index]
    test_df  = df.loc[test_feat.index]

    # NEW: Generate National Baseline (Sum of all synthetic states) for CNN-LSTM
    with st.spinner('🌐 Generating National context...'):
        all_frames = []
        for sn in INDIA_STATES:
            # Reusing generate_synthetic_india_data with same range
            sdf = generate_synthetic_india_data(sn, start_date, end_date)
            all_frames.append(sdf[['Load']])
        nat_df = pd.concat(all_frames, axis=1).sum(axis=1).to_frame(name='Load')
        train_nat = nat_df.loc[train_feat.index]
        test_nat  = nat_df.loc[test_feat.index]

    # Features & Targets
    feat_cols = [c for c in df_features.columns if c != 'Load']
    X_train_all = train_feat[feat_cols]
    y_train_all = train_feat['Load']
    X_test      = test_feat[feat_cols]
    y_test      = test_feat['Load']
    
    # Validation Split (Inner Chronological)
    val_size = max(1, int(len(X_train_all) * 0.15))
    X_tr = X_train_all.iloc[:-val_size]
    y_tr = y_train_all.iloc[:-val_size]
    X_val = X_train_all.iloc[-val_size:]
    y_val = y_train_all.iloc[-val_size:]
    
    # Scaling
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    predictions_dict = {}
    metrics_dict     = {}
    future_dict      = {}

    if use_dt:
        with st.spinner('🌳 Training Decision Tree…'):
            dt_model = train_decision_tree_model(X_tr_sc, y_tr.values)
            dt_pred  = dt_model.predict(X_test_sc)
            predictions_dict['Decision Tree'] = dt_pred
            metrics_dict['Decision Tree']     = calculate_metrics(y_test.values, dt_pred)
            metrics_dict['Decision Tree']['NSE'] = nse(y_test.values, dt_pred)
            if forecast_future and future_days > 0:
                fd, fp = iterative_future_forecast_sklearn_daily(df_features, dt_model, scaler, future_days)
                future_dict['Decision Tree'] = (fd, fp)
        st.success('✅ Decision Tree trained')

    if use_rf:
        with st.spinner('🌲 Training Random Forest…'):
            rf_model = train_random_forest_model(X_tr_sc, y_tr.values)
            rf_pred  = rf_model.predict(X_test_sc)
            predictions_dict['Random Forest'] = rf_pred
            metrics_dict['Random Forest']     = calculate_metrics(y_test.values, rf_pred)
            metrics_dict['Random Forest']['NSE'] = nse(y_test.values, rf_pred)
            if forecast_future and future_days > 0:
                fd, fp = iterative_future_forecast_sklearn_daily(df_features, rf_model, scaler, future_days)
                future_dict['Random Forest'] = (fd, fp)
        st.success('✅ Random Forest trained')

    if use_xgb and XGBOOST_AVAILABLE:
        with st.spinner('🤖 Training XGBoost…'):
            xgb_model = train_xgboost_model(X_tr_sc, y_tr.values, X_val_sc, y_val.values)
            xgb_pred  = xgb_model.predict(xgb.DMatrix(X_test_sc))
            predictions_dict['XGBoost'] = xgb_pred
            metrics_dict['XGBoost']     = calculate_metrics(y_test.values, xgb_pred)
            metrics_dict['XGBoost']['NSE'] = nse(y_test.values, xgb_pred)
            if forecast_future and future_days > 0:
                fd, fp = iterative_future_forecast_xgboost_daily(df_features, xgb_model, scaler, future_days)
                future_dict['XGBoost'] = (fd, fp)
        st.success('✅ XGBoost trained')

    if use_lstm and LSTM_AVAILABLE:
        with st.spinner('🧠 Training Hybrid CNN-LSTM…'):
            scaler_st = MinMaxScaler()
            scaler_nat = MinMaxScaler()
            
            scaled_st = scaler_st.fit_transform(train_df[['Load']].values)
            scaled_nat = scaler_nat.fit_transform(train_nat[['Load']].values)
            
            lookback = 30 # Reduced from 60 to focus on recent patterns and improve R2
            
            X_st,  y_st  = create_lstm_sequences(scaled_st,  lookback)
            X_nat, _     = create_lstm_sequences(scaled_nat, lookback)
            
            # Ensure shape (samples, lookback, features)
            X_st = X_st.reshape(-1, lookback, 1)
            X_nat = X_nat.reshape(-1, lookback, 1)
            
            vs = max(1, int(len(X_st)*0.85))
            
            # CNN-LSTM Architecture from dru/model.ipynb
            lstm_model = train_cnn_lstm_model(
                X_st[:vs], X_nat[:vs], y_st[:vs],
                X_st[vs:], X_nat[vs:], y_st[vs:], 
                epochs=60
            )
            
            # Prediction on Test Set
            test_scaled_st = scaler_st.transform(test_df[['Load']].values)
            test_scaled_nat = scaler_nat.transform(test_nat[['Load']].values)
            
            # Combine lookback from train + test for iterative prediction style
            comb_st = np.vstack([scaled_st[-lookback:], test_scaled_st])
            comb_nat = np.vstack([scaled_nat[-lookback:], test_scaled_nat])
            
            lstm_preds = []
            for i in range(lookback, len(comb_st)):
                s_st = comb_st[i-lookback:i].reshape(1, lookback, 1)
                s_nat = comb_nat[i-lookback:i].reshape(1, lookback, 1)
                lstm_preds.append(lstm_model.predict([s_st, s_nat], verbose=0)[0,0])
            
            lstm_pred = scaler_st.inverse_transform(
                np.array(lstm_preds).reshape(-1,1)).flatten()[:len(test_df)]
                
            predictions_dict['LSTM'] = lstm_pred
            metrics_dict['LSTM']     = calculate_metrics(test_df['Load'].values, lstm_pred)
            metrics_dict['LSTM']['NSE'] = nse(test_df['Load'].values, lstm_pred)
            
            if forecast_future and future_days > 0:
                fd, fp = iterative_future_forecast_cnn_lstm_daily(
                    df, nat_df, lstm_model, scaler_st, scaler_nat, lookback, future_days
                )
                future_dict['LSTM'] = (fd, fp)
        st.success('✅ CNN-LSTM trained')

    if use_sarima and SARIMA_AVAILABLE:
        with st.spinner('📈 Training SARIMA (weekly seasonal)…'):
            sarima_model = train_sarima_model(train_df, order=(1,1,1), seasonal_order=(1,1,1,7))
            sarima_pred  = sarima_model.forecast(steps=len(test_df))
            predictions_dict['SARIMA'] = sarima_pred
            metrics_dict['SARIMA']     = calculate_metrics(test_df['Load'].values, sarima_pred)
            metrics_dict['SARIMA']['NSE'] = nse(test_df['Load'].values, sarima_pred)
            if forecast_future and future_days > 0:
                total_fc = sarima_model.forecast(steps=len(test_df)+future_days)
                sarima_fut = total_fc.tail(future_days)
                future_dict['SARIMA'] = (sarima_fut.index, sarima_fut.values)
        st.success('✅ SARIMA trained')

    if use_tirex and TIREX_AVAILABLE:
        from tirex_utils import get_tirex_prediction
        with st.spinner('🔭 Running TiRex Zero-Shot Forecast...'):
            tirex_pred = get_tirex_prediction(train_df['Load'], len(test_df))
            
            # Ensure length consistency
            tirex_pred = tirex_pred[:len(test_df)]
            
            predictions_dict['TiRex'] = tirex_pred
            metrics_dict['TiRex']     = calculate_metrics(test_df['Load'].values, tirex_pred)
            metrics_dict['TiRex']['NSE'] = nse(test_df['Load'].values, tirex_pred)
            
            if forecast_future and future_days > 0:
                fd, fp = future_forecast_tirex(df, future_days)
                future_dict['TiRex'] = (fd, fp)
        st.success('✅ TiRex forecast completed')

    if use_dlinear and DLINEAR_AVAILABLE:
        with st.spinner('📐 Training DLinear…'):
            scaler_dl = MinMaxScaler()
            scaled_train = scaler_dl.fit_transform(train_df[['Load']].values)
            lookback = 30
            X_ltr, y_ltr = create_lstm_sequences(scaled_train, lookback)
            X_ltr = X_ltr.reshape(-1, lookback, 1)
            vs = int(len(X_ltr)*0.85)
            dl_model = train_dlinear_model(X_ltr[:vs], y_ltr[:vs], X_ltr[vs:], y_ltr[vs:], epochs=100)
            
            test_scaled = scaler_dl.transform(test_df[['Load']].values)
            combined    = np.vstack([scaled_train[-lookback:], test_scaled])
            dl_preds  = []
            for i in range(lookback, len(combined)):
                seq  = combined[i-lookback:i].reshape(1, lookback, 1)
                dl_preds.append(dl_model.predict(seq, verbose=0)[0,0])
            
            dl_pred = scaler_dl.inverse_transform(np.array(dl_preds).reshape(-1,1)).flatten()[:len(test_df)]
            predictions_dict['DLinear'] = dl_pred
            metrics_dict['DLinear']     = calculate_metrics(test_df['Load'].values, dl_pred)
            metrics_dict['DLinear']['NSE'] = nse(test_df['Load'].values, dl_pred)
            
            if forecast_future and future_days > 0:
                fd, fp = iterative_future_forecast_dlinear_daily(df, dl_model, scaler_dl, lookback, future_days)
                future_dict['DLinear'] = (fd, fp)
        st.success('✅ DLinear trained')


    st.header(f'📈 Results — {state_name}')

    if metrics_dict:
        st.subheader('📊 Performance Metrics')
        mdf = pd.DataFrame(metrics_dict).T
        st.dataframe(mdf.style
                     .highlight_min(axis=0, subset=['MAE','RMSE','MAPE (%)'])
                     .highlight_max(axis=0, subset=['R²', 'NSE']), use_container_width=True)

        st.subheader('📐 Diebold–Mariano Test')
        models_list = list(predictions_dict.keys())
        if len(models_list) >= 2:
            for i in range(len(models_list)):
                for j in range(i+1, len(models_list)):
                    m1, m2 = models_list[i], models_list[j]
                    e1 = test_df['Load'].values - predictions_dict[m1]
                    e2 = test_df['Load'].values - predictions_dict[m2]
                    dm, pv = diebold_mariano_test(e1, e2)
                    st.write(f'**{m1} vs {m2}:** DM = `{dm:.4f}`, p = `{pv:.4f}`')
        st.plotly_chart(plot_metrics_comparison(metrics_dict), use_container_width=True)

    if predictions_dict:
        st.subheader('🔮 Model Comparison: Test Period')
        st.plotly_chart(plot_model_comparison(test_df, predictions_dict, 'MU'), use_container_width=True)

    if forecast_future and future_dict:
        st.subheader('🚀 Future Forecasts')
        st.plotly_chart(plot_future_overlay(train_df, test_df, future_dict, 'MU'), use_container_width=True)
        future_export = pd.DataFrame()
        for mn, (fd, fp) in future_dict.items():
            if future_export.empty: future_export['Date'] = fd
            future_export[f'{mn}_Forecast_MU'] = fp
        st.download_button('⬇️ Download Future Forecasts (CSV)',
                           future_export.to_csv(index=False),
                           f'India_{state_name}_future_{datetime.now():%Y%m%d}.csv', 'text/csv')

    for model_name in ['Decision Tree','Random Forest','XGBoost','LSTM','SARIMA','TiRex','DLinear']:
        if model_name not in predictions_dict: continue
        icons = {'Decision Tree':'🌳','Random Forest':'🌲','XGBoost':'🤖','LSTM':'🧠','SARIMA':'📈', 'TiRex':'🔭', 'DLinear':'📐'}
        with st.expander(f'{icons.get(model_name,"📊")} {model_name} Details'):
            am, res = detect_anomalies(test_df['Load'].values, predictions_dict[model_name])
            if model_name in ('XGBoost',) and use_xgb:
                X_tr_full = scaler.transform(X_train_all)
                tr_pred = xgb_model.predict(xgb.DMatrix(X_tr_full))
                st.plotly_chart(plot_full_timeline(train_feat, test_feat, tr_pred,
                                                   predictions_dict[model_name], am, 'MU'),
                                use_container_width=True)
            st.plotly_chart(plot_residuals(res, test_df.index), use_container_width=True)
            st.info(f'Anomalies detected: {am.sum()}')


    if show_national:
        st.subheader('🗺️ National Overview (All 37 States, Synthetic)')
        all_frames = []
        with st.spinner('Generating all states…'):
            for sn in INDIA_STATES:
                sdf = generate_synthetic_india_data(sn, start_date, end_date)
                sdf['State'] = sn
                all_frames.append(sdf)
        national = pd.concat(all_frames)
        st.plotly_chart(plot_india_national_overview(national), use_container_width=True)
        top5 = national.groupby('State')['Load'].mean().nlargest(5).index.tolist()
        fig_top = go.Figure()
        for sn in top5:
            sdata = national[national['State']==sn]
            fig_top.add_trace(go.Scatter(x=sdata.index, y=sdata['Load'], mode='lines', name=sn))
        fig_top.update_layout(title='Top 5 States — Daily Demand Over Time',
                              xaxis_title='Date', yaxis_title='Load (MU)',
                              hovermode='x unified', height=400)
        st.plotly_chart(fig_top, use_container_width=True)

    st.subheader('💾 Export')
    res_df = test_df[['Load']].copy(); res_df.columns = ['Actual_MU']
    for mn, p in predictions_dict.items():
        res_df[f'{mn}_Pred'] = p; res_df[f'{mn}_Residual'] = res_df['Actual_MU'] - p
    st.download_button('⬇️ Download Test Results (CSV)', res_df.to_csv(),
                       f'India_{state_name}_test_{datetime.now():%Y%m%d}.csv', 'text/csv')
