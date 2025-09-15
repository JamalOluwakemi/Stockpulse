import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'selected_features.csv')
REPORTS_DIR = os.path.join(BASE_DIR, 'Data', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(CSV_PATH)

# --- Features to use ---
features = ['Close', 'Volume', 'Daily_Return', 'Volatility_7', 'Volatility_30']
X = df[features].copy()

# --- Handle missing values ---
X = X.fillna(0)

# --- Standardize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Isolation Forest ---
contamination_rate = 0.05
iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
iso_predictions = iso_forest.fit_predict(X_scaled)

# --- Assign anomalies ---
df['Anomaly'] = (iso_predictions == -1).astype(int)

# --- Summary / Metrics ---
total_rows = len(df)
num_anomalies = df['Anomaly'].sum()
anomaly_fraction = num_anomalies / total_rows

# Check if ground truth labels exist
if 'GroundTruth' in df.columns:
    y_true = df['GroundTruth']
    y_pred = df['Anomaly']

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    summary_text = f"""
Model Evaluation Report
=======================
Total rows: {total_rows}
Number of anomalies detected: {num_anomalies}
Anomaly fraction: {anomaly_fraction:.2%}
Contamination parameter: {contamination_rate:.2%}

Precision: {precision:.2f}
Recall: {recall:.2f}
F1-score: {f1:.2f}
"""
else:
    summary_text = f"""
Model Evaluation Report
=======================
Total rows: {total_rows}
Number of anomalies detected: {num_anomalies}
Anomaly fraction: {anomaly_fraction:.2%}
Contamination parameter: {contamination_rate:.2%}

(No ground truth labels found, skipping precision/recall/F1)
"""

print(summary_text)

# --- Save results ---
RESULT_CSV = os.path.join(BASE_DIR, 'Data', 'processed', 'modeling_results.csv')
df.to_csv(RESULT_CSV, index=False)

METRICS_TXT = os.path.join(REPORTS_DIR, 'metrics.txt')
with open(METRICS_TXT, 'w') as f:
    f.write(summary_text)

print(f"All done! Dataset saved as '{RESULT_CSV}' and metrics report saved as '{METRICS_TXT}'")
