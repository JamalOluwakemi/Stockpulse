import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_predictions = iso_forest.fit_predict(X_scaled)

# --- Assign anomalies ---
df['Anomaly'] = (iso_predictions == -1).astype(int)

# --- Summary ---
total_rows = len(df)
num_anomalies = df['Anomaly'].sum()
print(f"Total rows: {total_rows}")
print(f"Number of anomalies detected: {num_anomalies}")
print(f"Anomaly fraction: {num_anomalies / total_rows:.2%}")

# Save results
RESULT_CSV = os.path.join(BASE_DIR, 'Data', 'processed', 'modeling_results.csv')
df.to_csv(RESULT_CSV, index=False)
print(f"All done! Your dataset with anomaly labels is saved as '{RESULT_CSV}'")
