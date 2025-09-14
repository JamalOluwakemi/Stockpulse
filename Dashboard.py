import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SAMPLE_DIR = os.path.join(BASE_DIR, 'static', 'samples')
REPORTS_DIR = os.path.join(BASE_DIR, 'Data', 'reports')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# --- Helper Functions ---
def plot_stock(df, plot_filename):
    """Generate plot of Close price and highlight anomalies."""
    if 'Date' not in df.columns or 'Close' not in df.columns:
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')

    if 'Anomaly' in df.columns:
        anomalies = df[df['Anomaly'] == 1]
        if not anomalies.empty:
            plt.scatter(anomalies['Date'], anomalies['Close'], color='red', label='Anomalies')

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(UPLOAD_FOLDER, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_filename

def load_csv(filepath):
    """Load CSV and parse dates."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

def detect_anomalies(df):
    """Run Isolation Forest on selected features."""
    features = ['Close', 'Volume', 'Daily_Return', 'Volatility_7', 'Volatility_30']
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        df['Anomaly'] = 0
        return df

    X = df[available_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = (iso_forest.fit_predict(X_scaled) == -1).astype(int)
    return df

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for('results', filename=file.filename))
        else:
            flash("Please upload a valid CSV file.")
            return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        flash("File not found.")
        return redirect(url_for('index'))

    # Load CSV and detect anomalies
    df = load_csv(file_path)
    df = detect_anomalies(df)

    # Keep only anomalies for report
    anomalies = df[df['Anomaly'] == 1]
    report_filename = f"anomaly_report_{filename}"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    anomalies.to_csv(report_path, index=False)

    # Generate plot for full dataset
    plot_file = None
    if 'Date' in df.columns and 'Close' in df.columns:
        plot_filename = f"{os.path.splitext(filename)[0]}_plot.png"
        plot_file = plot_stock(df, plot_filename)

    # Show all rows in the table, highlight anomalies
    tables = [df.to_dict(orient='records')]
    columns = df.columns.tolist()

    has_anomalies = not anomalies.empty

    return render_template(
        'results.html',
        tables=tables,
        columns=columns,
        plot_file=plot_file,
        report_file=report_filename,
        has_anomalies=has_anomalies
    )

@app.route('/plot/<filename>')
def serve_plot(filename):
    plot_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(plot_path):
        flash("Plot not found.")
        return redirect(url_for('index'))
    return send_file(plot_path)

@app.route('/download_report/<filename>')
def download_report(filename):
    report_file = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(report_file):
        flash("Report not found.")
        return redirect(url_for('index'))
    return send_file(report_file, as_attachment=True)

@app.route('/download_sample')
def download_sample():
    sample_file = os.path.join(SAMPLE_DIR, 'sample_stock.csv')
    return send_file(sample_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
