# 📊 StockPulse Dashboard  

StockPulse is a **Flask-based anomaly detection dashboard** for stock market data.  
It allows users to upload CSV stock datasets, run anomaly detection and feature engineering, and visualize results through an interactive web UI.  

The dashboard highlights anomalies in stock prices and provides downloadable anomaly reports. It also includes charts for visual analysis of stock trends with anomalies.

---

## Demo Video
A screen recording showing the StockPulse dashboard in action is included in the `StockPulse_Demo` folder.


## Features  

- **Upload Stock Data:** Upload CSV files containing stock market data with fields like Date, Close, Volume, etc.  
- **Anomaly Detection:** Uses **Isolation Forest** and other algorithms to detect unusual spikes, dips, or patterns in stock prices.  
- **Highlight Anomalies:** Detected anomalies are highlighted in the results table for easy visualization.  
- **Interactive Charts:** View stock price trends over time with anomalies marked on the charts.  
- **Download Reports:** Export anomaly detection results as CSV reports for further analysis or documentation.  
- **Sample Data Provided:** Test the dashboard quickly with a sample stock CSV included in the project.  
- **Simple & Professional UI:** User-friendly dashboard built with Flask and Bootstrap for easy navigation and interaction.  

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
└── StockPulse_Demo/
    └── StockPulse_Dashboard_Demo.mp4

---

## How to Run  

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JamalOluwakemi/StockPulse.git
   cd StockPulse
