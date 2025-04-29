# BACKEND: Fraud Detection API and Pipeline (merged version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
import logging
import yaml
import os
from flask import Flask, request, jsonify
import shap

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Feature Engineering ---
def add_time_features(df):
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] % 24).astype(int)
        df['Is_Night'] = ((df['Hour'] < 6) | (df['Hour'] > 20)).astype(int)
        df = df.sort_values('Time')
        df['Time_Since_Prev'] = df['Time'].diff().fillna(0)
    return df

def add_sequence_features(df):
    if 'UserID' in df.columns:
        df = df.sort_values(['UserID', 'Time'])
        df['Prev_Amount'] = df.groupby('UserID')['Amount'].shift(1).fillna(0)
        df['Amt_Diff'] = df['Amount'] - df['Prev_Amount']
    return df

# --- Config Loader ---
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Main ML Pipeline ---
def fraud_pipeline(config, input_df=None):
    try:
        nrows = config.get('quick_mode_nrows', None)
        if config.get('quick_mode', False):
            nrows = nrows or 20000
        if input_df is not None:
            df = input_df.copy()
        else:
            df = pd.read_csv(config['dataset_path'], nrows=nrows)
        # Feature engineering
        if set(['V1','V28','Amount','Time']).issubset(df.columns):
            df['Amount_Rolling_Mean10'] = df['Amount'].rolling(window=10, min_periods=1).mean()
            df['Hour'] = (df['Time'] // 3600) % 24
            df['Is_Night'] = ((df['Hour'] < 6) | (df['Hour'] > 20)).astype(int)
            features = [col for col in df.columns if col not in ['Class','Time']]
            X = df[features]
            y = df['Class'] if 'Class' in df.columns else None
        else:
            df = add_time_features(df)
            df = add_sequence_features(df)
            df['Amount_Rolling_Mean10'] = df['Amount'].rolling(window=10, min_periods=1).mean()
            y = df['Class'] if 'Class' in df.columns else None
            X = df.drop(['Class'], axis=1) if 'Class' in df.columns else df
            categorical_cols = [col for col in ['MerchantType', 'Location', 'Device'] if col in X.columns]
            if categorical_cols:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # SMOTE
        if config.get('skip_smote', False) or y is None:
            X_res, y_res = X_scaled, y
        else:
            smote = SMOTE(random_state=config.get('random_seed', 42))
            X_res, y_res = smote.fit_resample(X_scaled, y)
        # Chronological split
        n = len(X_res)
        split_idx = int(n * 0.8)
        X_train, X_test = X_res[:split_idx], X_res[split_idx:]
        y_train, y_test = (y_res[:split_idx], y_res[split_idx:]) if y is not None else (None, None)
        # Model selection
        if config.get('tune', False):
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
            grid = GridSearchCV(RandomForestClassifier(random_state=config.get('random_seed', 42)), param_grid, cv=3, scoring='roc_auc')
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            models = {
                'logistic_regression': LogisticRegression(max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=config.get('random_seed', 42)),
                'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=config.get('random_seed', 42))
            }
            model = models[config.get('model', 'random_forest')]
            if y_train is not None:
                model.fit(X_train, y_train)
        # If input_df is provided, do prediction
        if input_df is not None:
            preds = model.predict(X_scaled)
            proba = model.predict_proba(X_scaled)[:,1]
            return preds, proba, model, X, scaler
        # Otherwise, run evaluation
        if y_test is not None:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            logging.info(f"Test Accuracy: {accuracy:.4f}")
            return {'accuracy': accuracy, 'roc_auc': roc_auc, 'cm': cm, 'cr': cr}
        return None
    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        raise

# --- Flask API ---
app = Flask(__name__)
config = load_config()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame(data)
    preds, proba, model, X, scaler = fraud_pipeline(config, input_df)
    result = 'fraud' if preds[0] == 1 else 'not fraud'
    explanation = None
    if preds[0] == 1:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[[0]])
        if isinstance(shap_values, list) and len(shap_values) == 2:
            feature_importances = dict(sorted(zip(X.columns, shap_values[1][0]), key=lambda x: abs(x[1]), reverse=True)[:3])
        else:
            feature_importances = dict(sorted(zip(X.columns, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)[:3])
        explanation = feature_importances
    return jsonify({"prediction": result, "explanation": explanation, "probability": float(proba[0])})

@app.route('/drift_detect', methods=['POST'])
def drift_detect():
    data = request.get_json()
    train_sample = pd.read_csv(config['dataset_path'], nrows=10000)
    new_data = pd.DataFrame(data)
    from scipy.stats import ks_2samp
    drifted_features = []
    for col in train_sample.columns:
        if col in new_data.columns and train_sample[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            stat, p = ks_2samp(train_sample[col].dropna(), new_data[col].dropna())
            if p < 0.05:
                drifted_features.append(col)
    return jsonify({"drifted_features": drifted_features})

if __name__ == "__main__":
    app.run(debug=True)
