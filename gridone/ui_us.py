import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import XGBOOST_AVAILABLE, LSTM_AVAILABLE, SARIMA_AVAILABLE, GRIDSTATUS_AVAILABLE, TIREX_AVAILABLE, DLINEAR_AVAILABLE
from data import fetch_grid_load, preprocess_data, engineer_features, create_lstm_sequences
from training import train_decision_tree_model, train_random_forest_model, train_xgboost_model, train_lstm_model, train_sarima_model, train_dlinear_model
from forecasting import iterative_future_forecast_sklearn, iterative_future_forecast_xgboost, iterative_future_forecast_lstm_us, iterative_future_forecast_dlinear_us, future_forecast_tirex
from metrics import calculate_metrics, nse, diebold_mariano_test, detect_anomalies
from visualization import plot_load_history, plot_full_timeline, plot_model_comparison, plot_metrics_comparison, plot_residuals, plot_future_overlay

from sklearn.preprocessing import StandardScaler, MinMaxScaler

if XGBOOST_AVAILABLE:
    import xgboost as xgb

def run_us_mode(sidebar):
    with sidebar:
        st.header('⚙️ US Configuration')
        market = st.selectbox('Select Market', ['CAISO','PJM','MISO','NYISO','SPP','ISONE'])
        st.subheader('Historical Data Range')
        default_end   = datetime.now().date()
        default_start = default_end - timedelta(days=120)
        c1, c2 = st.columns(2)
        with c1: start_date = st.date_input('Start', value=default_start,
                                             min_value=datetime(2018,1,1).date(),
                                             max_value=datetime.now().date())
        with c2: end_date   = st.date_input('End',   value=default_end,
                                             min_value=datetime(2018,1,1).date(),
                                             max_value=datetime.now().date())
        days_diff = (end_date - start_date).days
        st.caption(f'📅 {days_diff} days selected')
        if days_diff < 30: st.warning('⚠️ Select at least 30 days')
        st.subheader('Models')
        use_dt     = st.checkbox('Decision Tree',  value=True)
        use_rf     = st.checkbox('Random Forest',  value=True)
        use_xgb    = st.checkbox('XGBoost',        value=XGBOOST_AVAILABLE, disabled=not XGBOOST_AVAILABLE)
        use_lstm   = st.checkbox('LSTM',            value=LSTM_AVAILABLE,    disabled=not LSTM_AVAILABLE)
        use_sarima = st.checkbox('SARIMA (slow)',   value=False,             disabled=not SARIMA_AVAILABLE)
        use_tirex    = st.checkbox('TiRex (xLSTM)', value=TIREX_AVAILABLE, disabled=not TIREX_AVAILABLE)
        use_dlinear = st.checkbox('DLinear',          value=DLINEAR_AVAILABLE, disabled=not DLINEAR_AVAILABLE)
        st.subheader('Parameters')
        max_test  = max(30, days_diff // 2)
        test_size = st.slider('Test Set (days)', 7, min(60, max_test), 14)
        forecast_future = st.checkbox('Forecast Future Dates', value=False)
        future_days = st.slider('Days Ahead', 1, 30, 7) if forecast_future else 0
        st.divider()
        run_btn = st.button('🚀 Run US Forecast', type='primary', use_container_width=True)

    if not GRIDSTATUS_AVAILABLE:
        st.error('GridStatus library required.'); return
    if not run_btn:
        st.info('👈 Configure and click **Run US Forecast**')
        with st.expander('ℹ️ Models'):
            st.markdown('**Decision Tree** · **Random Forest** · **XGBoost** · **LSTM** · **SARIMA**')
        return

    with st.spinner('⏳ Fetching data…'):
        try:
            df = fetch_grid_load(
                market,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date,   datetime.max.time()))
        except Exception as e:
            st.error(f'Failed: {e}'); return

    with st.expander('📊 Raw Data'):
        st.dataframe(df.head(24), use_container_width=True)
        st.plotly_chart(plot_load_history(df, 'MW'), use_container_width=True)

    df_clean = preprocess_data(df)
    min_needed = test_size * 24 + 500
    if len(df_clean) < min_needed:
        st.error(f'Not enough data. Got {len(df_clean)}, need {min_needed}.'); return
    test_samples = test_size * 24
    train_df = df_clean.iloc[:-test_samples]
    test_df  = df_clean.iloc[-test_samples:]
    st.success(f'✅ Train: {len(train_df)} | Test: {len(test_df)}')

    df_features = engineer_features(df_clean)
    test_feat   = df_features.iloc[-test_samples:]
    train_feat  = df_features.iloc[:-test_samples]
    feat_cols   = [c for c in df_features.columns if c != 'Load']
    X_train_all = train_feat[feat_cols]; y_train_all = train_feat['Load']
    X_test      = test_feat[feat_cols];  y_test      = test_feat['Load']
    val_size    = max(1, int(len(X_train_all)*0.15))
    X_tr = X_train_all.iloc[:-val_size]; y_tr = y_train_all.iloc[:-val_size]
    X_val= X_train_all.iloc[-val_size:]; y_val= y_train_all.iloc[-val_size:]
    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_test_sc= scaler.transform(X_test)

    predictions_dict  = {}
    metrics_dict      = {}
    future_dict       = {}

    if use_dt:
        with st.spinner('🌳 Training Decision Tree…'):
            dt_model = train_decision_tree_model(X_tr_sc, y_tr.values)
            dt_pred  = dt_model.predict(X_test_sc)
            predictions_dict['Decision Tree'] = dt_pred
            metrics_dict['Decision Tree']     = calculate_metrics(y_test.values, dt_pred)
            metrics_dict['Decision Tree']['NSE'] = nse(y_test.values, dt_pred)
            if forecast_future and future_days > 0:
                fd, fp = iterative_future_forecast_sklearn(df_features, dt_model, scaler, future_days*24)
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
                fd, fp = iterative_future_forecast_sklearn(df_features, rf_model, scaler, future_days*24)
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
                fd, fp = iterative_future_forecast_xgboost(df_features, xgb_model, scaler, future_days*24)
                future_dict['XGBoost'] = (fd, fp)
        st.success('✅ XGBoost trained')

    if use_lstm and LSTM_AVAILABLE:
        with st.spinner('🧠 Training LSTM…'):
            scaler_lstm  = MinMaxScaler()
            scaled_train = scaler_lstm.fit_transform(train_df[['Load']].values)
            lookback = 24
            X_ltr, y_ltr = create_lstm_sequences(scaled_train, lookback)
            X_ltr = X_ltr.reshape(-1, lookback, 1)
            vs = int(len(X_ltr)*0.85)
            lstm_model = train_lstm_model(X_ltr[:vs], y_ltr[:vs], X_ltr[vs:], y_ltr[vs:], epochs=50)
            test_scaled = scaler_lstm.transform(test_df[['Load']].values)
            combined    = np.vstack([scaled_train[-lookback:], test_scaled])
            lstm_preds  = []
            for i in range(lookback, len(combined)):
                seq  = combined[i-lookback:i].reshape(1, lookback, 1)
                lstm_preds.append(lstm_model.predict(seq, verbose=0)[0,0])
            lstm_pred = scaler_lstm.inverse_transform(
                np.array(lstm_preds).reshape(-1,1)).flatten()[:len(test_df)]
            predictions_dict['LSTM'] = lstm_pred
            metrics_dict['LSTM']     = calculate_metrics(test_df['Load'].values, lstm_pred)
            metrics_dict['LSTM']['NSE'] = nse(test_df['Load'].values, lstm_pred)
            if forecast_future and future_days > 0:
                fd, fp = iterative_future_forecast_lstm_us(df_clean, lstm_model, scaler_lstm,
                                                           lookback, future_days*24)
                future_dict['LSTM'] = (fd, fp)
        st.success('✅ LSTM trained')

    if use_sarima and SARIMA_AVAILABLE:
        with st.spinner('📈 Training SARIMA…'):
            sarima_model = train_sarima_model(train_df, seasonal_order=(1,1,1,24))
            sarima_pred  = sarima_model.forecast(steps=len(test_df))
            predictions_dict['SARIMA'] = sarima_pred
            metrics_dict['SARIMA']     = calculate_metrics(test_df['Load'].values, sarima_pred)
            metrics_dict['SARIMA']['NSE'] = nse(test_df['Load'].values, sarima_pred)
            if forecast_future and future_days > 0:
                total_fc = sarima_model.forecast(steps=len(test_df)+future_days*24)
                sarima_fut = total_fc.tail(future_days*24)
                future_dict['SARIMA'] = (sarima_fut.index, sarima_fut.values)
        st.success('✅ SARIMA trained')

    if use_tirex and TIREX_AVAILABLE:
        from tirex_utils import get_tirex_prediction
        with st.spinner('🔭 Running TiRex Zero-Shot Forecast (Hourly)…'):
            tirex_pred = get_tirex_prediction(train_df['Load'], len(test_df))
            
            tirex_pred = tirex_pred[:len(test_df)]
            predictions_dict['TiRex'] = tirex_pred
            metrics_dict['TiRex']     = calculate_metrics(y_test.values, tirex_pred)
            metrics_dict['TiRex']['NSE'] = nse(y_test.values, tirex_pred)
            
            if forecast_future and future_days > 0:
                fd, fp = future_forecast_tirex(df_clean, future_days*24)
                future_dict['TiRex'] = (fd, fp[:future_days*24])
        st.success('✅ TiRex forecast completed')

    if use_dlinear and DLINEAR_AVAILABLE:
        with st.spinner('📐 Training DLinear…'):
            scaler_dl = MinMaxScaler()
            scaled_train = scaler_dl.fit_transform(train_df[['Load']].values)
            lookback = 24
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
                fd, fp = iterative_future_forecast_dlinear_us(df_clean, dl_model, scaler_dl,
                                                           lookback, future_days*24)
                future_dict['DLinear'] = (fd, fp)
        st.success('✅ DLinear trained')


    st.header('📈 Results')
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
        st.plotly_chart(plot_model_comparison(test_df, predictions_dict, 'MW'), use_container_width=True)

    if forecast_future and future_dict:
        st.subheader('🚀 Future Forecasts')
        st.plotly_chart(plot_future_overlay(train_df, test_df, future_dict, 'MW'), use_container_width=True)
        future_export = pd.DataFrame()
        for mn, (fd, fp) in future_dict.items():
            if future_export.empty: future_export['Date'] = fd
            future_export[f'{mn}_Forecast'] = fp
        st.download_button('⬇️ Download Future Forecasts (CSV)',
                           future_export.to_csv(index=False),
                           f'{market}_future_{datetime.now():%Y%m%d}.csv', 'text/csv')

    if 'Decision Tree' in predictions_dict:
        with st.expander('🌳 Decision Tree Details'):
            am, res = detect_anomalies(test_df['Load'].values, predictions_dict['Decision Tree'])
            st.plotly_chart(plot_residuals(res, test_df.index), use_container_width=True)
            st.info(f'Anomalies detected: {am.sum()}')

    if 'Random Forest' in predictions_dict:
        with st.expander('🌲 Random Forest Details'):
            am, res = detect_anomalies(test_df['Load'].values, predictions_dict['Random Forest'])
            st.plotly_chart(plot_residuals(res, test_df.index), use_container_width=True)
            st.info(f'Anomalies detected: {am.sum()}')

    if 'XGBoost' in predictions_dict and use_xgb:
        with st.expander('🤖 XGBoost Details'):
            am, res = detect_anomalies(test_df['Load'].values, predictions_dict['XGBoost'])
            X_tr_full = scaler.transform(X_train_all)
            train_pred_xgb = xgb_model.predict(xgb.DMatrix(X_tr_full))
            fig = plot_full_timeline(train_feat, test_feat, train_pred_xgb,
                                     predictions_dict['XGBoost'], am, 'MW')
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(plot_residuals(res, test_df.index), use_container_width=True)

    if 'LSTM' in predictions_dict:
        with st.expander('🧠 LSTM Details'):
            res = test_df['Load'].values - predictions_dict['LSTM']
            st.plotly_chart(plot_residuals(res, test_df.index), use_container_width=True)

    if 'SARIMA' in predictions_dict:
        with st.expander('📈 SARIMA Details'):
            res = test_df['Load'].values - predictions_dict['SARIMA']
            st.plotly_chart(plot_residuals(res, test_df.index), use_container_width=True)

    if 'TiRex' in predictions_dict:
        with st.expander('🔭 TiRex Details'):
            res = test_df['Load'].values - predictions_dict['TiRex']
            st.plotly_chart(plot_residuals(res, test_df.index), use_container_width=True)

    if 'DLinear' in predictions_dict:
        with st.expander('📐 DLinear Details'):
            res = test_df['Load'].values - predictions_dict['DLinear']
            st.plotly_chart(plot_residuals(res, test_df.index), use_container_width=True)


    st.subheader('💾 Export')
    res_df = test_df[['Load']].copy(); res_df.columns = ['Actual']
    for mn, p in predictions_dict.items():
        res_df[f'{mn}_Pred'] = p; res_df[f'{mn}_Residual'] = res_df['Actual'] - p
    st.download_button('⬇️ Download Test Results (CSV)', res_df.to_csv(),
                       f'{market}_test_{datetime.now():%Y%m%d}.csv', 'text/csv')
