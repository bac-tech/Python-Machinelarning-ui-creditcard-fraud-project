import io
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
REPO_OWNER = "bac-tech"
REPO_NAME = "Credit-Card-Fraud-Detection-System"
BRANCH = "main"

MODEL_RELATIVE_PATH = "Model/saved_models/fraud_detection_all_models.pkl"
DATASET_RELATIVE_PATH = "Dataset/creditcard.csv"

LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, MODEL_RELATIVE_PATH)
LOCAL_DATASET_PATH = os.path.join(ROOT_DIR, DATASET_RELATIVE_PATH)

MODEL_MEDIA_URL = (
    f"https://media.githubusercontent.com/media/{REPO_OWNER}/"
    f"{REPO_NAME}/{BRANCH}/{MODEL_RELATIVE_PATH}"
)
DATASET_MEDIA_URL = (
    f"https://media.githubusercontent.com/media/{REPO_OWNER}/"
    f"{REPO_NAME}/{BRANCH}/{DATASET_RELATIVE_PATH}"
)


def _is_lfs_pointer(data: bytes) -> bool:
    return data.startswith(b"version https://git-lfs.github.com/spec/v1")


@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    errors = []

    if os.path.isfile(LOCAL_MODEL_PATH):
        try:
            with open(LOCAL_MODEL_PATH, "rb") as f:
                raw = f.read()
            if not _is_lfs_pointer(raw):
                return pickle.load(io.BytesIO(raw)), "local"
            errors.append("Local model is a Git LFS pointer.")
        except Exception as exc:
            errors.append(f"Local model error: {exc}")

    try:
        response = requests.get(
            MODEL_MEDIA_URL,
            timeout=120,
            allow_redirects=True,
        )
        response.raise_for_status()
        if _is_lfs_pointer(response.content):
            raise RuntimeError("GitHub returned an LFS pointer instead of the model.")
        package = pickle.load(io.BytesIO(response.content))
        return package, "GitHub LFS"
    except Exception as exc:
        errors.append(f"GitHub LFS model download failed: {exc}")

    return None, " | ".join(errors)


@st.cache_data(show_spinner="Loading transaction dataset...")
def load_dataset():
    if os.path.isfile(LOCAL_DATASET_PATH):
        try:
            df = pd.read_csv(LOCAL_DATASET_PATH)
            return df, None
        except Exception as exc:
            local_error = str(exc)
    else:
        local_error = "Local dataset file not found."

    try:
        response = requests.get(
            DATASET_MEDIA_URL,
            timeout=180,
            allow_redirects=True,
        )
        response.raise_for_status()
        if _is_lfs_pointer(response.content):
            raise RuntimeError("GitHub returned an LFS pointer instead of the dataset.")
        df = pd.read_csv(io.BytesIO(response.content))
        return df, None
    except Exception as exc:
        return None, f"{local_error} GitHub LFS download failed: {exc}"


def build_default_features():
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


data_pkg, model_source = load_model()

if data_pkg is None:
    st.error("❌ ML model could not be loaded.")
    st.info(
        "The repository stores the trained model with Git LFS. "
        "This app now tries both the local file and GitHub's LFS media endpoint."
    )
    st.code(model_source)
    st.stop()

scaler = data_pkg.get("scaler")
lr_model = data_pkg.get("logistic_model")
rf_model = data_pkg.get("random_forest_model")
best_model_name = data_pkg.get("best_model_name", "Random Forest")
feature_names = data_pkg.get("feature_names", build_default_features())

if scaler is None or lr_model is None or rf_model is None:
    st.error("❌ Model package is incomplete.")
    st.write("Required keys: scaler, logistic_model, random_forest_model.")
    st.stop()

df, dataset_error = load_dataset()

if dataset_error:
    st.warning(
        "Dataset could not be loaded. Prediction still works, but dataset-based "
        "KPIs and charts are unavailable."
    )

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #eef4ff 50%, #f8f5ff 100%);
        color: #1e293b;
    }
    .block-container {
        max-width: 1450px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .dashboard-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .title-text {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-title {
        color: #173b8f !important;
        font-size: 20px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 15px;
        padding: 10px 12px;
        border-left: 6px solid #6366f1;
        background: rgba(99, 102, 241, 0.08);
        border-radius: 0 12px 12px 0;
    }
    .kpi-card {
        min-height: 150px;
        padding: 18px 16px;
        border-radius: 18px;
        background: linear-gradient(145deg, #ffffff 0%, #eef2ff 100%);
        border: 1px solid #c7d2fe;
        box-shadow: 0 8px 22px rgba(30, 64, 175, 0.12);
        text-align: center;
        margin-bottom: 10px;
    }
    .kpi-icon {
        font-size: 27px;
        margin-bottom: 5px;
    }
    .kpi-heading {
        color: #1e3a8a !important;
        font-size: 13px;
        font-weight: 850;
        letter-spacing: 0.4px;
        opacity: 1 !important;
    }
    .kpi-value {
        color: #111827 !important;
        font-size: 30px;
        line-height: 1.25;
        font-weight: 900;
        margin-top: 8px;
    }
    .kpi-subtitle {
        color: #64748b !important;
        font-size: 11px;
        font-weight: 600;
        margin-top: 5px;
    }
    div[data-testid="stMetricLabel"] p {
        color: #334155 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #172554 !important;
        font-weight: 850 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #475569 !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid #c7d2fe;
        border-radius: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(30, 64, 175, 0.07);
    }
    div[data-testid="stExpander"] summary {
        font-weight: 800;
        color: #1e3a8a;
    }
    div[data-testid="stButton"] > button {
        border-radius: 12px;
        font-weight: 750;
        min-height: 46px;
    }
    div[data-testid="stNumberInput"] label {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    .result-card {
        text-align: center;
        padding: 25px;
        border-radius: 18px;
        margin-top: 15px;
        color: white;
    }
    .result-card h1 {
        font-size: 26px;
        margin-bottom: 10px;
        color: white;
    }
    .risk-number {
        font-size: 40px;
        font-weight: 800;
        color: white;
    }
    .legitimate {
        background: linear-gradient(135deg, #0f5132, #198754);
        border: 2px solid #20c997;
    }
    .fraud {
        background: linear-gradient(135deg, #842029, #dc3545);
        border: 2px solid #ff6b6b;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .dashboard-title {
            font-size: 24px !important;
            text-align: center;
        }
        .section-title {
            font-size: 18px !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 10px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-title">💳 <span class="title-text">'
    "Credit Card Fraud Detection Dashboard</span></div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## ⚙️ Model Configuration")
model_choice = st.sidebar.radio(
    "Select ML Algorithm:",
    ["Random Forest", "Logistic Regression"],
    index=0 if best_model_name == "Random Forest" else 1,
)
selected_model = rf_model if model_choice == "Random Forest" else lr_model

st.sidebar.success(f"Model loaded from: {model_source}")
st.sidebar.info(f"Best trained model: {best_model_name}")

selected_metrics = (
    data_pkg.get("random_forest_metrics", {})
    if model_choice == "Random Forest"
    else data_pkg.get("logistic_metrics", {})
)

st.markdown('<div class="section-title">1. Top KPIs</div>', unsafe_allow_html=True)

if df is not None and "Class" in df.columns:
    total_transactions = len(df)
    fraud_cases = int((df["Class"] == 1).sum())
    legitimate_cases = int((df["Class"] == 0).sum())
    fraud_rate = fraud_cases / total_transactions * 100 if total_transactions else 0
else:
    total_transactions = legitimate_cases = fraud_cases = 0
    fraud_rate = 0

k1, k2, k3, k4 = st.columns(4)

kpi_cards = [
    (
        "💳",
        "TOTAL TRANSACTIONS",
        f"{total_transactions:,}" if total_transactions else "N/A",
        "All transactions in dataset",
    ),
    (
        "🚨",
        "FRAUD CASES",
        f"{fraud_cases:,}" if total_transactions else "N/A",
        "Detected fraud transactions",
    ),
    (
        "📊",
        "FRAUD RATE",
        f"{fraud_rate:.2f}%" if total_transactions else "N/A",
        "Fraud percentage of total",
    ),
    (
        "🎯",
        "MODEL F1-SCORE",
        f"{selected_metrics.get('f1', 0):.4f}" if selected_metrics else "N/A",
        f"{model_choice} performance",
    ),
]

for col, (icon, heading, value, subtitle) in zip((k1, k2, k3, k4), kpi_cards):
    with col:
        st.markdown(
            f'''
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-heading">{heading}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-subtitle">{subtitle}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

if df is not None and {"Class", "Amount", "Time"}.issubset(df.columns):
    st.markdown('<div class="section-title">2. Transaction Overview</div>', unsafe_allow_html=True)
    ov_col1, ov_col2 = st.columns(2)

    with ov_col1:
        st.subheader("Fraud vs Legitimate Transactions")
        chart_data = pd.DataFrame(
            {
                "Type": ["Legitimate", "Fraud"],
                "Count": [legitimate_cases, fraud_cases],
            }
        )
        st.bar_chart(chart_data.set_index("Type"))

    with ov_col2:
        st.subheader("Amount Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["Amount"], bins=40)
        ax.set_xlabel("Amount")
        ax.set_ylabel("Frequency")
        st.pyplot(fig, clear_figure=True)

    st.markdown('<div class="section-title">3. Fraud Trend</div>', unsafe_allow_html=True)
    fraud_df = df[df["Class"] == 1].copy()
    if not fraud_df.empty:
        fraud_df["Time Bin"] = pd.cut(fraud_df["Time"], bins=15)
        trend = fraud_df.groupby("Time Bin", observed=False).size()
        trend_df = pd.DataFrame(
            {
                "Time Bins": [str(x) for x in trend.index],
                "Fraud Cases": trend.values,
            }
        )
        st.line_chart(trend_df.set_index("Time Bins"))

st.markdown(
    '<div class="section-title">4. Transaction Prediction</div>',
    unsafe_allow_html=True,
)

if "v_values" not in st.session_state:
    st.session_state.v_values = {f"V{i}": 0.0 for i in range(1, 29)}
if "time_val" not in st.session_state:
    st.session_state.time_val = 50000.0
if "amount_val" not in st.session_state:
    st.session_state.amount_val = 100.0


def apply_input_values(values, time_value, amount_value):
    # Update the actual widget keys so the UI visibly changes immediately.
    st.session_state["input_time"] = float(time_value)
    st.session_state["input_amount"] = float(amount_value)
    st.session_state.time_val = float(time_value)
    st.session_state.amount_val = float(amount_value)
    for i in range(1, 29):
        st.session_state[f"input_V{i}"] = float(values[i - 1])


def clear_prediction_inputs():
    # Clear every widget key, not only helper state, so Reset really resets the form.
    st.session_state["input_time"] = 50000.0
    st.session_state["input_amount"] = 100.0
    st.session_state.time_val = 50000.0
    st.session_state.amount_val = 100.0
    for i in range(1, 29):
        st.session_state[f"input_V{i}"] = 0.0
    st.session_state.pop("prediction_result", None)


b1, b2, b3 = st.columns(3)

with b1:
    if st.button("⚡ Fill Demo Case", use_container_width=True):
        apply_input_values([0.0] * 28, 1000.0, 150.0)
        st.rerun()

with b2:
    if st.button("🚨 Fill High-Risk Demo", use_container_width=True):
        apply_input_values(
            [-5.0, -2.0, -1.0, -3.0, -1.0, -2.0, -1.0,
             -4.0, -2.0, -3.0, -1.0, -2.0, -1.0, -2.0,
             -3.0, -2.0, -1.0, -2.0, -3.0, -1.0, -2.0,
             -3.0, -2.0, -1.0, -2.0, -3.0, -1.0, -2.0],
            406.0,
            1150.0,
        )
        st.rerun()

with b3:
    if st.button("🧹 Reset / Clear All", use_container_width=True):
        clear_prediction_inputs()
        st.rerun()

time_input = st.number_input(
    "⏱️ Transaction Time",
    min_value=0.0,
    value=float(st.session_state.get("input_time", 50000.0)),
    key="input_time",
)
amount_input = st.number_input(
    "💰 Transaction Amount",
    min_value=0.0,
    value=float(st.session_state.get("input_amount", 100.0)),
    format="%.2f",
    key="input_amount",
)

st.subheader("🔎 Transaction Features")
st.caption("V1–V28 are divided into 4 groups. Each group contains exactly 7 individual inputs. Enter the values from your transaction record.")

v_values = {}
groups = [
    (1, 7, "📘 Group 1 — V1 to V7"),
    (8, 14, "📗 Group 2 — V8 to V14"),
    (15, 21, "📙 Group 3 — V15 to V21"),
    (22, 28, "📕 Group 4 — V22 to V28"),
]

for start, end, label in groups:
    with st.expander(label, expanded=(start == 1)):
        cols = st.columns(4)
        for offset, i in enumerate(range(start, end + 1)):
            with cols[offset % 4]:
                v_values[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=float(st.session_state.get(f"input_V{i}", 0.0)),
                    format="%.6f",
                    key=f"input_V{i}",
                    help=f"Transaction feature V{i}",
                )

predict_click = st.button(
    "🔍 Predict Transaction",
    type="primary",
    use_container_width=True,
)

if predict_click:
    input_dict = {"Time": time_input}
    input_dict.update(v_values)
    input_dict["Amount"] = amount_input

    try:
        input_df = pd.DataFrame([input_dict])[feature_names]

        if model_choice == "Logistic Regression":
            model_input = scaler.transform(input_df)
        else:
            model_input = input_df

        prediction = int(selected_model.predict(model_input)[0])
        probability = float(selected_model.predict_proba(model_input)[0][1] * 100)

        risk = "HIGH" if probability >= 70 else "MEDIUM" if probability >= 30 else "LOW"
        status = "FRAUD" if prediction == 1 else "LEGITIMATE"

        st.session_state.prediction_result = {
            "prediction": prediction,
            "prob": probability,
            "risk": risk,
            "status": status,
            "amount": amount_input,
            "model": model_choice,
        }
    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)

st.markdown('<div class="section-title">5. Prediction Result</div>', unsafe_allow_html=True)

result = st.session_state.get("prediction_result")

if result:
    is_fraud = result["prediction"] == 1
    card_class = "fraud" if is_fraud else "legitimate"
    title = "🚨 FRAUDULENT TRANSACTION" if is_fraud else "✅ LEGITIMATE TRANSACTION"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <h1>{title}</h1>
            <div class="risk-number">{result["prob"]:.2f}%</div>
            <p>Fraud Probability</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Transaction Amount", f"₹{result['amount']:,.2f}")
    r2.metric("Status", result["status"])
    r3.metric("Risk Level", result["risk"])
    r4.metric("Selected Model", result["model"])

st.markdown('<div class="section-title">6. Model Performance</div>', unsafe_allow_html=True)

if selected_metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{selected_metrics.get('precision', 0):.4f}")
    m2.metric("Recall", f"{selected_metrics.get('recall', 0):.4f}")
    m3.metric("F1-Score", f"{selected_metrics.get('f1', 0):.4f}")
    m4.metric("ROC-AUC", f"{selected_metrics.get('roc_auc', 0):.4f}")

    cm = selected_metrics.get("confusion_matrix")
    if cm:
        st.subheader("Confusion Matrix")
        cm_array = np.array(cm)
        cm_df = pd.DataFrame(
            cm_array,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Pred Legitimate", "Pred Fraud"],
        )
        st.dataframe(cm_df, use_container_width=True)

if df is not None and {"Time", "Amount", "Class"}.issubset(df.columns):
    st.markdown(
        '<div class="section-title">7. Sample Transactions</div>',
        unsafe_allow_html=True,
    )
    sample_df = df[["Time", "Amount", "Class"]].tail(50).copy()
    sample_df["Status"] = sample_df["Class"].map(
        {0: "Legitimate", 1: "Fraud"}
    )
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="section-title">8. Feature Importance</div>',
    unsafe_allow_html=True,
)

feature_importance_path = os.path.join(
    ROOT_DIR,
    "Model",
    "saved_models",
    "feature_importance.csv",
)

try:
    if os.path.isfile(feature_importance_path):
        fi = pd.read_csv(feature_importance_path)
        if not fi.empty and {"Feature", "Importance"}.issubset(fi.columns):
            st.bar_chart(fi.head(10).set_index("Feature"))
    else:
        st.caption("Feature importance file is not available locally.")
except Exception as exc:
    st.caption(f"Feature importance could not be displayed: {exc}")
