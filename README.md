# Fraud Detection System

## Overview
A robust machine learning pipeline to detect fraudulent transactions using real-world and synthetic data. The system features advanced feature engineering, production-like evaluation, explainability (SHAP), and both API and web demo interfaces.

## Workflow Summary (Advanced)
1. **Configurable Pipeline:** All options (data, features, models, SMOTE, tuning, etc.) are set in `config.yaml`.
2. **Data Loading:** Supports full or sampled data (quick mode). Loads Kaggle creditcard.csv or synthetic data.
3. **Feature Engineering:** Rolling means, time-based features (Hour, Is_Night, Time_Since_Prev), and more.
4. **Preprocessing:** Scales features, handles categorical data, SMOTE (optional).
5. **Production-like Split:** Chronological split (first 80% train, last 20% test).
6. **Model Training:** Random Forest, Logistic Regression, XGBoost; optional hyperparameter tuning.
7. **Evaluation:** Precision, recall, F1, ROC-AUC, confusion matrix.
8. **Explainability:** SHAP summary plot for global feature importances.
9. **Real-Time Fraud Prediction:**
    - **Flask API (`app.py`):** `/predict` endpoint returns prediction and top SHAP features for flagged frauds.
    - **Streamlit app:** Interactive demo for batch and single predictions.
10. **Drift Detection:**
    - **Script:** `drift_detection.py` compares new data to training data using KS-test.
    - **API:** `/drift_detect` endpoint in Flask returns features with significant drift.
11. **Self-Improving Pipeline:**
    - **Script:** `retrain.py` appends new labeled data and retrains the model automatically.
12. **Real-Time Transaction Feed Simulation:**
    - **Script:** `real_time_feed.py` streams transactions to the API, simulating live banking data.
13. **Testing:** Unit tests for feature engineering and evaluation.

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Edit `config.yaml` as needed.**
3. **Run main pipeline:**
   ```bash
   python fraud_detection.py --config config.yaml
   ```
4. **Run Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```
5. **Run Flask API:**
   ```bash
   python app.py
   ```
6. **Drift Detection:**
   - **Script:**
     ```bash
     python drift_detection.py
     ```
   - **API:**
     ```bash
     # POST new data as JSON to http://localhost:5000/drift_detect
     ```
7. **Self-Improving Pipeline (Auto-Retrain):**
   ```bash
   python retrain.py --config config.yaml --new_data new_transactions.csv
   ```
8. **Real-Time Transaction Feed Simulation:**
   ```bash
   python real_time_feed.py --csv creditcard.csv --api_url http://localhost:5000/predict --delay 1.0
   ```

## Key Features
- Config-driven, modular codebase
- Handles imbalanced data (SMOTE optional)
- Advanced feature engineering (including Time_Since_Prev)
- Chronological (production-like) train/test split
- Hyperparameter tuning (GridSearchCV)
- Model explainability (SHAP)
- **Drift detection** (script & API)
- **Self-improving pipeline** (auto-retrain)
- **Anomaly explanation** (per-fraud SHAP reasons)
- **Real-time transaction feed simulation**
- Streamlit web demo and Flask API
- Robust logging and error handling
- Unit tests for reliability

## Directory Structure
- `fraud_detection.py` - Main pipeline
- `fraud_utils.py` - Feature engineering utilities
- `drift_detection.py` - Drift detection script
- `retrain.py` - Self-improving (auto-retrain) script
- `real_time_feed.py` - Real-time transaction feed simulator
- `config.yaml` - Configuration file
- `streamlit_app.py` - Web demo app
- `app.py` - Flask API (with SHAP explanations & drift endpoint)
- `tests/` - Unit tests
- `requirements.txt` - Dependencies

## Extending
- Add more features/models in `fraud_detection.py` and `fraud_utils.py`
- Add more tests in `tests/`
- Enhance Streamlit/Flask apps for deployment

## Author
Your Name Here
