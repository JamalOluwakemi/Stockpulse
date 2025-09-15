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

def detect_anomalies(df, filename=None):
    """Run Isolation Forest on selected features and generate metrics."""
    features = ['Close', 'Volume', 'Daily_Return', 'Volatility_7', 'Volatility_30']
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        df['Anomaly'] = 0
        return df, None

    # Standardize and run Isolation Forest
    X = df[available_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    contamination_rate = 0.05
    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
    df['Anomaly'] = (iso_forest.fit_predict(X_scaled) == -1).astype(int)

    # --- Metrics ---
    total_rows = len(df)
    num_anomalies = df['Anomaly'].sum()
    anomaly_fraction = num_anomalies / total_rows if total_rows > 0 else 0

    metrics_text = f"""
    Model Evaluation Report
    =======================
    File: {filename if filename else "N/A"}
    Total rows: {total_rows}
    Number of anomalies detected: {num_anomalies}
    Anomaly fraction: {anomaly_fraction:.2%}
    Contamination parameter: {contamination_rate:.2%}
    """

    # Save metrics file
    metrics_file = os.path.join(REPORTS_DIR, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(metrics_text)

    return df, metrics_text


# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part in request.")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for('results', filename=file.filename))
        else:
            flash("Please upload a valid CSV file.")
            return redirect(request.url)

    return render_template('index.html')


@app.route('/results/<filename>')
def results(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        flash("File not found.")
        return redirect(url_for('index'))

    # Load CSV and detect anomalies + metrics
    df = load_csv(file_path)
    df, metrics_content = detect_anomalies(df, filename)

    # Keep only anomalies for report
    anomalies = df[df['Anomaly'] == 1]
    report_filename = f"anomaly_report_{filename}"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    anomalies.to_csv(report_path, index=False)

    # Generate plot
    plot_file = None
    if 'Date' in df.columns and 'Close' in df.columns:
        plot_filename = f"{os.path.splitext(filename)[0]}_plot.png"
        plot_file = plot_stock(df, plot_filename)

    # Show results
    tables = [df.to_dict(orient='records')]
    columns = df.columns.tolist()
    has_anomalies = not anomalies.empty

    return render_template(
        'results.html',
        tables=tables,
        columns=columns,
        plot_file=plot_file,
        report_file=report_filename,
        has_anomalies=has_anomalies,
        metrics=metrics_content
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
