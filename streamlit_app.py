import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Fraud Detection Pro Dashboard", layout="wide")

# --- Sidebar Controls ---
st.sidebar.title('Fraud Detection')
dataset = st.sidebar.selectbox('Dataset', ['creditcard.csv', 'synthetic_fraud_large.csv'], key='dataset')
model_name = st.sidebar.selectbox('Model', ['Logistic Regression', 'Random Forest', 'XGBoost'], key='model')

uploaded_file = st.sidebar.file_uploader('Upload Transactions (CSV)', type=['csv'], key=f'file_{dataset}_{model_name}')

if 'run_clicked' not in st.session_state:
    st.session_state['run_clicked'] = False
if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None
if 'last_dataset' not in st.session_state:
    st.session_state['last_dataset'] = dataset
if 'last_model' not in st.session_state:
    st.session_state['last_model'] = model_name

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

state = {
    'model_name': model_name,
    'dataset_name': dataset,
    'total_txns': 20000,
    'fraud_txns': 85,
    'fraud_rate': 85/20000,
    'last_retrain': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'fraud_predictions': None,
    'drift_alert': False,
    'auto_retrain': False,
}

st.title('🏦 Fraud Detection Dashboard')
st.info('**How to Use:** 1. Select your dataset and model. 2. Upload a CSV of transactions. 3. Click "Run" to see results.')

# --- Dashboard Section ---
def render_dashboard(state):
    col1, col2, col3 = st.columns(3)
    col1.metric('Model', state.get('model_name', 'N/A'))
    col2.metric('Dataset', state.get('dataset_name', 'N/A'))
    col3.metric('Total Transactions', state.get('total_txns', 0))
    st.metric('Fraud Transactions', state.get('fraud_txns', 0))
    st.metric('Fraud Detection Rate', f"{state.get('fraud_rate', 0):.2%}")
    st.metric('Last Retrained', state.get('last_retrain', 'Never'))
    if state.get('drift_alert', False):
        st.error('⚠️ Drift detected! Model retraining recommended.')
    if not state.get('auto_retrain', False):
        if st.button('Retrain Model'):
            state['retrain_trigger'] = True
            st.success('Retrain triggered!')

# --- Live Predictions Section ---
def render_live_predictions(state):
    st.title('🔍 Live Predictions')
    uploaded_file = st.file_uploader('Upload CSV for Batch Prediction', type=['csv'], key='live_pred_upload')
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        placeholder = st.empty()
        for i in range(1, len(data)+1):
            batch = data.iloc[:i]
            batch['Fraud_Prediction'] = (batch.index % 2 == 0).astype(int)
            batch['Confidence'] = 0.9 - 0.4 * (batch.index % 2)
            placeholder.dataframe(batch)
            import time
            time.sleep(0.5)
        st.success('Batch prediction complete!')
    else:
        st.info('Upload a CSV file to simulate live predictions.')

# --- Anomaly Explanation Section ---
def render_anomaly_explanation(state):
    st.title('🧠 Anomaly Explanation')
    frauds = state.get('fraud_predictions')
    if frauds is not None and not frauds.empty:
        txn = st.selectbox('Select a fraud transaction', frauds.index)
        st.write('SHAP Force Plot for Transaction', txn)
        st.image('https://shap.readthedocs.io/en/latest/_images/force_plot.png', caption='SHAP Force Plot (Demo)')
        st.write('Top contributing features:')
        st.write({'V14': 2.1, 'V10': -1.7, 'Amount': 0.9})
    else:
        st.info('No fraud predictions available. Run predictions first.')

# --- Drift Detection Section ---
def render_drift_detection(state):
    st.title('📈 Drift Detection')
    uploaded_file = st.file_uploader('Upload for Drift Check', type=['csv'], key='drift_upload')
    if uploaded_file:
        features = ['V1', 'V2', 'V3', 'Amount']
        drift_scores = np.random.rand(len(features))
        fig, ax = plt.subplots()
        ax.bar(features, drift_scores)
        ax.set_ylabel('Drift Score (KS)')
        st.pyplot(fig)
        st.write('Drifted features:', [f for f, s in zip(features, drift_scores) if s > 0.5])
        st.success('Drift analysis complete!')
    st.write('Drift log (last 3 checks):')
    st.table({'Timestamp': ['2025-04-22 13:00', '2025-04-21 10:30', '2025-04-20 09:15'], 'Drifted Features': [['V1'], ['V2', 'V3'], []]})

# --- Model Training Section ---
def render_model_training(state):
    st.title('🛠️ Model Training')
    st.write('Select model and hyperparameters below.')
    st.selectbox('Model', ['Logistic Regression', 'Random Forest', 'XGBoost'], key='model_select')
    st.checkbox('Enable Hyperparameter Tuning', key='tune')
    if st.button('Train Model'):
        st.success('Model training started!')
        st.write({'precision': 0.88, 'recall': 0.87, 'f1': 0.87, 'roc_auc': 0.99})
        fig, ax = plt.subplots()
        ax.imshow([[0,1],[1,0]], cmap='Blues')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

# --- Logs & Reports Section ---
def render_logs_reports(state):
    st.title('📚 Logs & Reports')
    logs = [
        '[2025-04-22 13:00] Loaded dataset creditcard.csv',
        '[2025-04-22 13:01] SMOTE enabled',
        '[2025-04-22 13:02] Model trained: Random Forest',
        '[2025-04-22 13:03] Drift detected: V1',
        '[2025-04-22 13:04] Prediction: Fraud',
    ]
    st.write('### Pipeline Logs')
    st.code('\n'.join(logs))
    st.write('Download artifacts:')
    st.download_button('Download Model', data=b'model', file_name='model.pkl')
    st.download_button('Download SHAP Plot', data=b'shap', file_name='shap.png')
    st.download_button('Download Drift Report', data=b'drift', file_name='drift_report.txt')
    st.write('### Model Performance History')
    st.table({'Timestamp': ['2025-04-22', '2025-04-21'], 'Precision': [0.88, 0.86], 'Recall': [0.87, 0.85], 'ROC-AUC': [0.99, 0.98]})

# --- Main App Layout ---
if uploaded_file is not None and st.session_state['run_clicked']:
    data = pd.read_csv(uploaded_file)
    data.loc[:, 'Fraud_Prediction'] = (data.index % 2 == 0).astype(int)
    data.loc[:, 'Confidence'] = 0.9 - 0.4 * (data.index % 2)
    state['fraud_predictions'] = data[data['Fraud_Prediction'] == 1].copy()
    st.subheader('Prediction Results')
    st.dataframe(data)
    st.success(f"Predicted {data['Fraud_Prediction'].sum()} frauds out of {len(data)} transactions.")
    render_dashboard(state)
    render_live_predictions(state)
    render_anomaly_explanation(state)
    render_drift_detection(state)
    render_model_training(state)
    render_logs_reports(state)
else:
    st.subheader('Dashboard')
    render_dashboard(state)
    render_logs_reports(state)

st.caption('Professional Fraud Detection Tool | Demo Mode')
