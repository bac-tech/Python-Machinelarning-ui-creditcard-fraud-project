#💳 Credit Card Fraud Detection System
Python 3.9+FrameworkUI 
License

An end-to-end Machine Learning system and interactive analytics web application built to detect fraudulent credit card transactions in heavily imbalanced financial data.

📌 Table of Contents
Executive Overview
Key Features
System Architecture
Dataset Specifications
Project Directory Structure
Installation & Quickstart
Model Training & Benchmarking
Interactive Web Dashboard
Git LFS Pointer Troubleshooting
Documentation Suite
🔍 Executive Overview
Credit card fraud costs the global financial ecosystem tens of billions of dollars annually. Building fraud detection systems involves overcoming significant real-world machine learning challenges:

Extreme Class Imbalance: Genuine transactions drastically outnumber fraudulent attempts (typically ~99.8% legitimate vs ~0.2% fraud). Conventional models trained on unweighted data achieve misleadingly high accuracy by simply predicting every record as legitimate.
Asymmetric Business Costs: A False Negative (missing an actual fraud incident) causes direct financial chargebacks and reputational damage. A False Positive (blocking a legitimate user's card) introduces customer dissatisfaction.
Low-Latency Inference: Financial institutions require sub-second risk classification when a card is swiped or an online purchase is submitted.
This repository provides a modular, production-grade codebase that trains cost-sensitive Machine Learning models (Logistic Regression with feature scaling and Random Forest ensemble), serializes production artifacts, and serves an interactive Streamlit dashboard for fraud analytics and real-time transaction scoring.

✨ Key Features
Automated Ingestion & Validation: Verifies transaction integrity, drops duplicate entries, and validates schema consistency.
Git LFS & Self-Healing Resilience: Automatically detects unresolved Git LFS pointer text files and recovers by downloading raw files or generating high-fidelity synthetic benchmark data.
Cost-Sensitive ML Pipeline: Balanced class weighting (class_weight='balanced') to effectively penalize missed fraud cases.
Comparative Model Evaluation: Benchmarks Random Forest and Logistic Regression across Precision, Recall, F1-Score, and ROC-AUC.
Interactive Streamlit Web Dashboard:
Live KPI Metric Cards: Real-time summaries of total volume, fraud count, fraud rate, and model F1-score.
Exploratory Visualizations: Volume distribution, log-scaled amount histograms, and fraud temporal trend charts.
One-Click Presets: Test legitimate and high-risk fraud cases with a single click.
4-Group Feature Expanders: Clean UI for all 28 PCA-transformed numerical features (V1–V28).
Real-Time Risk Scoring: Dynamic result cards displaying fraud probability percentage, status badges, and risk levels (HIGH, MEDIUM, LOW).
Model Interpretability: Feature importance bar charts highlighting key fraud indicators (such as V14, V10, V12, V17, V4).
🏛️ System Architecture
Mermaid diagram
📊 Dataset Specifications
The system is configured around the standard European credit card fraud benchmark dataset:

Total Features: 30 numerical input features + 1 target feature.
Time: Seconds elapsed between this transaction and the first transaction in the dataset.
V1 – V28: Principal components obtained via PCA transformation to protect customer confidentiality and sensitive financial information.
Amount: Transaction monetary value in USD/EUR.
Class (Target):
0: Legitimate transaction
1: Fraudulent transaction
📁 Project Directory Structure

Credit-Card-Fraud-Detection-System/
├── Dataset/
│   ├── creditcard.csv                 # Raw or benchmark transaction data
│   └── generate_sample_data.py        # Dataset validation & synthetic generator utility
│
├── Model/
│   ├── train.py                       # ML model training and evaluation pipeline
│   └── saved_models/                  # Serialized model artifacts
│       ├── fraud_detection_all_models.pkl
│       ├── scaler.pkl
│       ├── logistic_model.pkl
│       ├── random_forest_model.pkl
│       ├── best_model.pkl
│       ├── feature_names.pkl
│       ├── metrics.pkl
│       └── feature_importance.csv
│
├── App/
│   └── app.py                         # Streamlit dashboard implementation
│
├── app.py                             # Root runner entry point
├── requirements.txt                   # Production dependency specifications
├── README.md                          # Project documentation
│
└── docs/
    ├── ARCHITECTURE.md                # System design & data flow documentation
    ├── MODEL_REPORT.md                # Algorithmic benchmarks, cost matrix & evaluation
    └── USER_GUIDE.md                  # Comprehensive user instructions & manual
🚀 Installation & Quickstart
1. Clone or Open the Repository
bash

cd Credit-Card-Fraud-Detection-System
2. Set Up a Virtual Environment (Recommended)
bash

# Windows
python -m venv venv
.\venv\Scripts\activate
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash

pip install -r requirements.txt
🧠 Model Training & Benchmarking
To run the complete data preprocessing, training, and artifact serialization pipeline:

bash

python Model/train.py
Pipeline Execution Output:
Loads and validates Dataset/creditcard.csv.
Cleans duplicates and verifies 30 numerical input features.
Conducts a stratified 80/20 train/test split.
Fits a StandardScaler on training data.
Trains Logistic Regression and Random Forest Classifier with class_weight='balanced'.
Evaluates test set performance across Precision, Recall, F1-Score, and ROC-AUC.
Saves all 8 model bundles to Model/saved_models/.
Benchmark Results (Hold-out Test Set)
Metric	Logistic Regression (Balanced)	Random Forest (Balanced)
Accuracy	~97.5%	~99.9%
Precision	~0.06 – 0.10	~0.85 – 0.95
Recall (Sensitivity)	~0.90 – 0.93	~0.82 – 0.88
F1-Score	~0.15 – 0.18	~0.85 – 0.91
ROC-AUC	~0.970	~0.985
Note: Random Forest achieves superior F1-Score and precision, minimizing customer-facing false alarms while retaining strong fraud recall.

💻 Interactive Web Dashboard
Launch the Streamlit web dashboard:

bash

streamlit run app.py
(Or alternatively: streamlit run App/app.py)

Once started, open your browser at http://localhost:8501.

Dashboard Capabilities:
Interactive Model Switcher: Toggle in real time between Random Forest and Logistic Regression.
Live KPI Metric Badges: Instant summary of total records, fraud count, fraud rate, and model F1-score.
Exploratory Visualizations: View volume distribution, amount histograms, and time-bin trends.
Preset Demonstration Buttons:
⚡ Fill Legitimate Demo: Pre-fills inputs with normal baseline parameters.
🚨 Fill High-Risk Fraud Demo: Pre-fills inputs with known discriminative fraud signatures (V14, V12, V10, V4, V11).
🧹 Reset / Clear Inputs: Resets all parameters to neutral zero.
Real-Time Risk Scoring: View fraud probability percentage, classification badge, and risk level.
Feature Importance & Confusion Matrix: Examine underlying decision drivers and test set performance matrices.
⚠️ Git LFS Pointer Troubleshooting
The Problem
When cloning or downloading repositories containing large files (such as creditcard.csv ~150MB or model pickle files) without Git LFS configured, Git stores small pointer text files instead:

text

version https://git-lfs.github.com/spec/v1
oid sha256:76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
size 150828752
When Python attempts to read this with pandas.read_csv() or pickle.load(), it throws parser errors or reports missing columns.

Solutions
Option 1 (Git LFS Pull):
bash

git lfs install
git lfs pull
Option 2 (Automatic Recovery / Generator): Run our automated dataset utility:
bash

python Dataset/generate_sample_data.py
Option 3 (Self-Healing Streamlit App): The Streamlit app in this repository includes automatic self-healing: if neither the local file nor the remote Git LFS endpoint is available, it automatically generates a calibrated benchmark dataset and trains in-memory baseline models so you can test the system immediately without manual setup!
📚 Documentation Suite
For deeper technical deep dives, explore our detailed documentation:

System Architecture Guide
: Pipeline engineering, caching strategies, and data flow.
Machine Learning Model Report
: Class imbalance methodologies, cost matrix evaluation, and hyperparameter analysis.
User Guide & Operations Manual
