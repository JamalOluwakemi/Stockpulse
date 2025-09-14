# 📊 StockPulse Dashboard  

StockPulse is a **Flask-based anomaly detection dashboard** for stock market data.  
It allows users to upload CSV stock datasets, run anomaly detection and feature engineering, and visualize results through an interactive web UI.  

The dashboard highlights anomalies in stock prices and provides downloadable anomaly reports. It also includes charts for visual analysis of stock trends with anomalies.

---

## Project Structure  

/StockPulse
│
├── Dashboard.py                  # Main Flask application (serves UI & routes)
├── anomaly_detection.py          # Core anomaly detection & modeling logic
│   # - Cleans input data
│   # - Runs anomaly detection algorithms (Isolation Forest, SVM, etc.)
│   # - Returns results to Dashboard.py for visualization
│
├── requirements.txt              # Full dependencies (for dev & notebooks)
├── requirements-min.txt          # Minimal dependencies (for dashboard only)
├── README.md                     # Project documentation (this file)
├── .gitignore                    # Ignore unnecessary files in version control
│
├── templates/                    # HTML templates (Flask views)
│   ├── base.html                 # Shared layout
│   ├── index.html                # Upload page
│   ├── results.html              # Visualization page
│   └── ...
│
├── static/                       # Static assets (CSS, JS, images, sample data)
│   ├── css/                      # Stylesheets
│   ├── js/                       # JavaScript files
│   └── samples/
│       └── sample_stock.csv      # Example stock dataset
│
├── uploads/                      # User-uploaded CSV files (via dashboard UI)
│
├── data/                         # Dataset storage
│   ├── raw/                      # Raw datasets (original CSVs before cleaning)
│   ├── processed/                # Cleaned & feature-engineered datasets
│   │   ├── cleaned_stock.csv     # Example processed dataset
│   │   └── other_files.csv       # Additional generated datasets
│   └── reports/                  # Generated anomaly reports
│       ├── anomaly_report_*.csv  # Anomaly detection CSV reports for uploaded datasets
│       └── ...
│
└── notebooks/                    # Jupyter notebooks for EDA & prototyping
    ├── eda.ipynb                 # Exploratory Data Analysis
    └── feature_engineering.ipynb # Feature engineering & preprocessing

---

## How to Run  

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JamalOluwakemi/StockPulse.git
   cd StockPulse
