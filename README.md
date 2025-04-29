# Fraud Detection System

## Overview
A robust machine learning pipeline to detect fraudulent transactions using real-world and synthetic data. The system features advanced feature engineering, production-like evaluation, explainability (SHAP), and both API and web demo interfaces.

## New Structure (2025)
- **Backend:** All backend logic (ML pipeline, Flask API, SHAP explanations, drift detection) is now in `backend.py`.
- **Frontend:** All Streamlit UI logic is in `streamlit_app.py`.
- This makes deployment and maintenance easier—just two main files for all code!

## Workflow Summary
1. **Configurable Pipeline:** All options (data, features, models, SMOTE, tuning, etc.) are set in `config.yaml`.
2. **Data Loading:** Supports full or sampled data (quick mode). Loads Kaggle creditcard.csv or synthetic data.
3. **Feature Engineering:** Rolling means, time-based features (Hour, Is_Night, Time_Since_Prev), and more.
4. **Preprocessing:** Scales features, handles categorical data, SMOTE (optional).
5. **Production-like Split:** Chronological split (first 80% train, last 20% test).
6. **Model Training:** Random Forest, Logistic Regression, XGBoost; optional hyperparameter tuning.
7. **Evaluation:** Precision, recall, F1, ROC-AUC, confusion matrix.
8. **Explainability:** SHAP summary plot for global feature importances.
9. **Real-Time Fraud Prediction:**
    - **Flask API (`backend.py`):** `/predict` endpoint returns prediction and top SHAP features for flagged frauds.
    - **Streamlit app (`streamlit_app.py`):** Interactive demo for batch and single predictions.
10. **Drift Detection:**
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
3. **Run backend API (Flask):**
   ```bash
   python backend.py
   ```
4. **Run Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```
5. **Self-Improving Pipeline (Auto-Retrain):**
   ```bash
   python retrain.py --config config.yaml --new_data new_transactions.csv
   ```
6. **Real-Time Transaction Feed Simulation:**
   ```bash
   python real_time_feed.py --csv creditcard.csv --api_url http://localhost:5000/predict --delay 1.0
   ```

## Key Features
- Single-file backend (`backend.py`) and frontend (`streamlit_app.py`) for easy deployment and maintenance
- Config-driven pipeline
- Handles imbalanced data (SMOTE optional)
- Advanced feature engineering (including Time_Since_Prev)
- Chronological (production-like) train/test split
- Hyperparameter tuning (GridSearchCV)
- Model explainability (SHAP)
- Drift detection (API)
- Self-improving pipeline (auto-retrain)
- Anomaly explanation (per-fraud SHAP reasons)
- Real-time transaction feed simulation
- Streamlit web demo and Flask API
- Robust logging and error handling
- Unit tests for reliability

## Directory Structure
- `backend.py` - All backend logic (Flask API + ML pipeline)
- `streamlit_app.py` - Web demo app (all frontend logic)
- `retrain.py` - Self-improving (auto-retrain) script
- `real_time_feed.py` - Real-time transaction feed simulator
- `config.yaml` - Configuration file
- `tests/` - Unit tests
- `requirements.txt` - Dependencies
- `.gitignore` - Git ignore rules

## Extending
- Add more features/models in `backend.py`
- Add more UI features in `streamlit_app.py`
- Add more tests in `tests/`

## Author
Your Name Here
