# 💳 Credit Card Fraud Detection System

An interactive Machine Learning web application built using **Python**, **Streamlit**, and **Scikit-Learn** to detect fraudulent credit card transactions in real-time.

---

## 📌 Project Overview

Credit card fraud is a significant issue in financial services. This project implements machine learning algorithms (such as Logistic Regression and Random Forest) to identify suspicious transactions. The user interface allows users and analysts to test transaction parameters dynamically and evaluate risk metrics in real-time.

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection-System/
├── app/
│   └── app.py                  # Main Streamlit dashboard interface
├── Dataset/
│   └── creditcard.csv          # Credit card transaction dataset (Tracked with Git LFS)
├── Model/
│   ├── Model_train.py          # Preprocessing & model training pipeline
│   ├── project internship.ipynb # Exploratory Data Analysis & experimentation
│   └── saved_models/           # Saved model artifacts (.pkl files)
│       ├── fraud_detection_all_models.pkl
│       ├── logistic_model.pkl
│       ├── random_forest_model.pkl
│       └── scaler.pkl
├── .gitattributes              # Git LFS Configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
🚀 Key Features
Data Preprocessing & Feature Scaling: Robust handling of imbalanced transaction data with standard scaling.

Multiple Model Evaluation: Evaluates both Logistic Regression and Ensemble Random Forest Classifiers.

Interactive Dashboard: Dynamic input controls, test transaction triggers, and visual analytics via Streamlit.

Git LFS Integration: Large model artifacts and datasets are managed efficiently for web deployment.

🛠️ Tech Stack
Language: Python

Machine Learning: Scikit-Learn, Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Web Interface: Streamlit

⚙️ Installation & Local Setup

1. Create a virtual environment (Optional but Recommended)
2 .Clone:[https://github.com/bac-tech/Python-Machinelarning-ui-creditcard-fraud-project](https://github.com/bac-tech/Python-Machinelarning-ui-creditcard-fraud-project)
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
Bash
pip install -r requirements.txt
💻 Running the Application
Train the Models
To execute data preprocessing and train the classification models:

Bash
python Model/Model_train.py
Launch Streamlit Dashboard
To run the interactive web interface locally:

Bash
streamlit run app/app.py
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
