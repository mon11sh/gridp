import streamlit as st
from config import XGBOOST_AVAILABLE, LSTM_AVAILABLE, SARIMA_AVAILABLE, GRIDSTATUS_AVAILABLE
from ui_us import run_us_mode
from ui_india import run_india_mode

def main():
    st.set_page_config(
        page_title='Grid Load Forecasting — US & India',
        page_icon='⚡',
        layout='wide'
    )

    st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #f39c12, #e74c3c, #9b59b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #7f8c8d; font-size: 1rem; margin-bottom: 1.5rem; }
    </style>
    <p class="main-title">⚡ Power Grid Load Forecasting</p>
    <p class="subtitle">US Markets (CAISO · PJM · MISO · NYISO · SPP · ISONE) &nbsp;|&nbsp;
       India Grid (37 States, Synthetic) &nbsp;|&nbsp;
       Models: Decision Tree · Random Forest · XGBoost · LSTM · SARIMA</p>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/lightning-bolt.png", width=60)
        mode = st.radio('🌍 Select Region', ['🇺🇸 US Market', '🇮🇳 India Grid'],
                        help='US mode fetches live data via GridStatus API. India mode uses synthetic data.')
        st.divider()

    if mode == '🇺🇸 US Market':
        with st.expander('📦 Library Status', expanded=False):
            st.write(f"XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}")
            st.write(f"LSTM (TF): {'✅' if LSTM_AVAILABLE else '❌'}")
            st.write(f"SARIMA: {'✅' if SARIMA_AVAILABLE else '❌'}")
            st.write(f"GridStatus: {'✅' if GRIDSTATUS_AVAILABLE else '❌'}")
        run_us_mode(st.sidebar)
    else:
        st.info('🇮🇳 **India Grid Mode** — Synthetic daily load data for all 37 Indian states. '
                'Same models and metrics as US mode.')
        run_india_mode(st.sidebar)

if __name__ == '__main__':
    main()
