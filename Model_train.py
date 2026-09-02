import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Dataset",
    "creditcard.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


print("=" * 70)
print("CREDIT CARD FRAUD DETECTION SYSTEM")
print("MODEL TRAINING")
print("=" * 70)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[1] Loading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv("C:/Users/CHETNA/Downloads/Credit-Card-Fraud-Detection-System-main/Credit-Card-Fraud-Detection-System-main/Dataset/creditcard.csv")

print("Dataset Shape:", df.shape)


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\n[2] Dataset Information")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

print("\n[3] Removing duplicates...")

before = len(df)

df.drop_duplicates(
    inplace=True
)

after = len(df)

print(
    "Duplicates Removed:",
    before - after
)

print(
    "Dataset Shape After Duplicate Removal:",
    df.shape
)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Time",
    "Amount",
    "Class"
]

# Add V1 to V28
for i in range(1, 29):
    required_columns.append(f"V{i}")


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns: "
        + ", ".join(missing_columns)
    )


print("\n[4] Required columns verified successfully.")


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

print("\n[5] Class Distribution")

class_counts = df["Class"].value_counts()

print(class_counts)

print("\nClass Percentage:")

class_percentage = (
    df["Class"]
    .value_counts(normalize=True)
    * 100
)

print(class_percentage)


fraud_count = int(
    (df["Class"] == 1).sum()
)

legitimate_count = int(
    (df["Class"] == 0).sum()
)

print("\nLegitimate Transactions:", legitimate_count)
print("Fraud Transactions:", fraud_count)


# ============================================================
# 7. SELECT FEATURES
# ============================================================

print("\n[6] Selecting features...")

feature_names = [
    "Time"
]

# V1 to V28
for i in range(1, 29):
    feature_names.append(f"V{i}")

feature_names.append(
    "Amount"
)


X = df[
    feature_names
].copy()

y = df[
    "Class"
].copy()


print("\nSelected Features:")
print(feature_names)

print(
    "\nTotal Input Features:",
    len(feature_names)
)

print(
    "Target Column: Class"
)


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

print("\n[7] Handling missing values...")

missing_before = X.isnull().sum().sum()

print(
    "Missing Values Before:",
    missing_before
)


if missing_before > 0:

    for column in X.columns:

        if X[column].isnull().any():

            X[column] = X[column].fillna(
                X[column].median()
            )


missing_after = X.isnull().sum().sum()

print(
    "Missing Values After:",
    missing_after
)


# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

print("\n[8] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    "Training Records:",
    len(X_train)
)

print(
    "Testing Records:",
    len(X_test)
)

print(
    "\nTraining Class Distribution:"
)

print(
    y_train.value_counts()
)

print(
    "\nTesting Class Distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# 10. FEATURE SCALING
# ============================================================

print("\n[9] Scaling features for Logistic Regression...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 11. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n[10] Training Logistic Regression...")

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

lr_model.fit(
    X_train_scaled,
    y_train
)

print(
    "Logistic Regression training completed."
)


# ============================================================
# 12. TRAIN RANDOM FOREST
# ============================================================

print("\n[11] Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    max_depth=None
)

rf_model.fit(
    X_train,
    y_train
)

print(
    "Random Forest training completed."
)


# ============================================================
# 13. LOGISTIC REGRESSION PREDICTION
# ============================================================

print("\n[12] Logistic Regression Prediction...")

lr_pred = lr_model.predict(
    X_test_scaled
)

lr_probability = lr_model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 14. RANDOM FOREST PREDICTION
# ============================================================

print("\n[13] Random Forest Prediction...")

rf_pred = rf_model.predict(
    X_test
)

rf_probability = rf_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. METRICS FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

    return {

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability
            )
        ),

        "confusion_matrix": (
            confusion_matrix(
                y_true,
                y_pred
            ).tolist()
        )
    }


# ============================================================
# 16. CALCULATE METRICS
# ============================================================

print("\n[14] Calculating model metrics...")

lr_metrics = calculate_metrics(
    y_test,
    lr_pred,
    lr_probability
)

rf_metrics = calculate_metrics(
    y_test,
    rf_pred,
    rf_probability
)


# ============================================================
# 17. DISPLAY LOGISTIC REGRESSION RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 70)

print(
    "Accuracy :",
    round(lr_metrics["accuracy"], 4)
)

print(
    "Precision:",
    round(lr_metrics["precision"], 4)
)

print(
    "Recall   :",
    round(lr_metrics["recall"], 4)
)

print(
    "F1 Score :",
    round(lr_metrics["f1"], 4)
)

print(
    "ROC-AUC  :",
    round(lr_metrics["roc_auc"], 4)
)

print(
    "\nConfusion Matrix:"
)

print(
    np.array(
        lr_metrics["confusion_matrix"]
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        lr_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. DISPLAY RANDOM FOREST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)

print(
    "Accuracy :",
    round(rf_metrics["accuracy"], 4)
)

print(
    "Precision:",
    round(rf_metrics["precision"], 4)
)

print(
    "Recall   :",
    round(rf_metrics["recall"], 4)
)

print(
    "F1 Score :",
    round(rf_metrics["f1"], 4)
)

print(
    "ROC-AUC  :",
    round(rf_metrics["roc_auc"], 4)
)

print(
    "\nConfusion Matrix:"
)

print(
    np.array(
        rf_metrics["confusion_matrix"]
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 19. MODEL SELECTION
# ============================================================

print("\n")
print("=" * 70)
print("MODEL SELECTION")
print("=" * 70)


# Fraud detectionમાં F1 score મહત્વપૂર્ણ છે.
# F1 પછી ROC-AUC ને tie-breaker તરીકે ઉપયોગ કરીએ છીએ.

if (
    rf_metrics["f1"] > lr_metrics["f1"]
):

    best_model_name = "Random Forest"
    best_model = rf_model

elif (
    rf_metrics["f1"] < lr_metrics["f1"]
):

    best_model_name = "Logistic Regression"
    best_model = lr_model

else:

    if (
        rf_metrics["roc_auc"]
        >=
        lr_metrics["roc_auc"]
    ):

        best_model_name = "Random Forest"
        best_model = rf_model

    else:

        best_model_name = "Logistic Regression"
        best_model = lr_model


print(
    "Selected Best Model:",
    best_model_name
)


# ============================================================
# 20. SAVE ALL MODELS PACKAGE
# ============================================================

print("\n[15] Saving models...")


all_in_one_pkg = {

    # Scaler
    "scaler": scaler,

    # Models
    "logistic_model": lr_model,

    "random_forest_model": rf_model,

    # Best Model
    "best_model": best_model,

    "best_model_name": best_model_name,

    # Features
    "feature_names": feature_names,

    "training_features": feature_names,

    # Target
    "target_column": "Class",

    # Metrics
    "logistic_metrics": lr_metrics,

    "random_forest_metrics": rf_metrics,

    # Class information
    "class_labels": {
        0: "Legitimate",
        1: "Fraud"
    }
}


# ============================================================
# 21. SAVE MAIN PACKAGE
# ============================================================

main_model_path = os.path.join(
    MODEL_DIR,
    "fraud_detection_all_models.pkl"
)


with open(
    main_model_path,
    "wb"
) as file:

    pickle.dump(
        all_in_one_pkg,
        file
    )


# ============================================================
# 22. SAVE SCALER
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


# ============================================================
# 23. SAVE LOGISTIC REGRESSION
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "logistic_model.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        lr_model,
        file
    )


# ============================================================
# 24. SAVE RANDOM FOREST
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        rf_model,
        file
    )


# ============================================================
# 25. SAVE BEST MODEL
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        best_model,
        file
    )


# ============================================================
# 26. SAVE FEATURE NAMES
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "feature_names.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        feature_names,
        file
    )


# ============================================================
# 27. SAVE METRICS
# ============================================================

metrics_data = {

    "logistic_regression": lr_metrics,

    "random_forest": rf_metrics,

    "best_model": best_model_name
}


with open(
    os.path.join(
        MODEL_DIR,
        "metrics.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        metrics_data,
        file
    )


# ============================================================
# 28. FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)


feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": rf_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 29. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance_path = os.path.join(
    MODEL_DIR,
    "feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_path,
    index=False
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY! ✅")
print("=" * 70)

print("\nDataset:")
print(
    os.path.basename(DATASET_PATH)
)

print("\nTotal Records:")
print(
    len(df)
)

print("\nTotal Features:")
print(
    len(feature_names)
)

print("\nFeatures Used:")
print(
    ", ".join(feature_names)
)

print("\nTarget:")
print(
    "Class (0 = Legitimate, 1 = Fraud)"
)

print("\nBest Model:")
print(
    best_model_name
)

print("\nSaved Models Directory:")
print(
    MODEL_DIR
)

print("\nGenerated Files:")

print(
    "1. fraud_detection_all_models.pkl"
)

print(
    "2. scaler.pkl"
)

print(
    "3. logistic_model.pkl"
)

print(
    "4. random_forest_model.pkl"
)

print(
    "5. best_model.pkl"
)

print(
    "6. feature_names.pkl"
)

print(
    "7. metrics.pkl"
)

print(
    "8. feature_importance.csv"
)

print("\nDone!")