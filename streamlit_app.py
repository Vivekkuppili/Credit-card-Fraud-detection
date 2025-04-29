import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Fraud Detection Pro Dashboard", layout="wide")

# --- Sidebar Controls ---
st.sidebar.title('Fraud Detection')
dataset = st.sidebar.selectbox('Dataset', ['creditcard.csv', 'synthetic_fraud_large.csv'], key='dataset')
model_name = st.sidebar.selectbox('Model', ['Logistic Regression', 'Random Forest', 'XGBoost'], key='model')

# File uploader with unique key so changing dataset/model resets file
uploaded_file = st.sidebar.file_uploader('Upload Transactions (CSV)', type=['csv'], key=f'file_{dataset}_{model_name}')

# --- Session State for Run Logic ---
if 'run_clicked' not in st.session_state:
    st.session_state['run_clicked'] = False
if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None
if 'last_dataset' not in st.session_state:
    st.session_state['last_dataset'] = dataset
if 'last_model' not in st.session_state:
    st.session_state['last_model'] = model_name

# Reset run state if file, dataset, or model changes
if uploaded_file is not None and (
    st.session_state['last_file'] != uploaded_file or
    st.session_state['last_dataset'] != dataset or
    st.session_state['last_model'] != model_name):
    st.session_state['run_clicked'] = False
    st.session_state['last_file'] = uploaded_file
    st.session_state['last_dataset'] = dataset
    st.session_state['last_model'] = model_name

run_disabled = uploaded_file is None
run = st.sidebar.button('Run', disabled=run_disabled)
if run:
    st.session_state['run_clicked'] = True

# --- App State (placeholder/demo) ---
state = {
    'model_name': model_name,
    'dataset_name': dataset,
    'total_txns': 20000,
    'fraud_txns': 85,
    'fraud_rate': 85/20000,
    'last_retrain': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'fraud_predictions': None,
}

st.title('🏦 Fraud Detection Dashboard')
st.info('**How to Use:** 1. Select your dataset and model. 2. Upload a CSV of transactions. 3. Click "Run" to see results.')

# --- Home Tab: Dashboard + Predictions ---
if uploaded_file is not None and st.session_state['run_clicked']:
    data = pd.read_csv(uploaded_file)
    # Placeholder: random predictions and confidence
    data.loc[:, 'Fraud_Prediction'] = (data.index % 2 == 0).astype(int)
    data.loc[:, 'Confidence'] = 0.9 - 0.4 * (data.index % 2)
    state['fraud_predictions'] = data[data['Fraud_Prediction'] == 1].copy()
    st.subheader('Prediction Results')
    st.dataframe(data)
    st.success(f"Predicted {data['Fraud_Prediction'].sum()} frauds out of {len(data)} transactions.")
else:
    st.subheader('Dashboard')
    col1, col2, col3 = st.columns(3)
    col1.metric('Model', state.get('model_name', 'N/A'))
    col2.metric('Dataset', state.get('dataset_name', 'N/A'))
    col3.metric('Total Transactions', state.get('total_txns', 0))
    st.info('Upload a CSV file and click "Run" to see fraud statistics and predictions.')

# --- Anomaly Explanation Tab ---
st.markdown('---')
st.header('🧠 Anomaly Explanation')
from dashboard import render_dashboard
from live_predictions import render_live_predictions
from anomaly_explanation import render_anomaly_explanation
from drift_detection import render_drift_detection
from model_training import render_model_training
from logs_reports import render_logs_reports
render_logs_reports(state)

st.caption('Professional Fraud Detection Tool | Demo Mode')
