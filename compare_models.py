import pandas as pd
import os

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VAL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "validation_processed.csv.gz"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PRODUCTION_MODEL = os.path.join(
    MODEL_DIR,
    "xgboost_fraud_model_production.json"
)

CANDIDATE_MODEL = os.path.join(
    MODEL_DIR,
    "xgboost_fraud_model_candidate.json"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation data...")

validation = pd.read_csv(VAL_FILE)

TARGET = "isFraud"

DROP_COLUMNS = [
    TARGET,
    "isFlaggedFraud"
]

X_val = validation.drop(
    columns=DROP_COLUMNS
)

y_val = validation[TARGET]

print("Validation shape:", validation.shape)
print("Features:", X_val.shape[1])


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

print("\nLoading production model...")

production_model = XGBClassifier()

production_model.load_model(
    PRODUCTION_MODEL
)


# ============================================================
# LOAD CANDIDATE MODEL
# ============================================================

print("Loading candidate model...")

candidate_model = XGBClassifier()

candidate_model.load_model(
    CANDIDATE_MODEL
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, name):

    print(
        f"\nEvaluating {name}..."
    )

    y_prob = model.predict_proba(
        X_val
    )[:, 1]

    threshold = 0.5

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_val,
        y_prob
    )

    pr_auc = average_precision_score(
        y_val,
        y_prob
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_val,
        y_pred
    )

    return {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


# ============================================================
# EVALUATE BOTH MODELS
# ============================================================

production_results = evaluate_model(
    production_model,
    "Production"
)

candidate_results = evaluate_model(
    candidate_model,
    "Candidate"
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    f"{'Metric':<15}"
    f"{'Production':<15}"
    f"{'Candidate':<15}"
)

print("-" * 45)

metrics = [
    ("ROC-AUC", "roc_auc"),
    ("PR-AUC", "pr_auc"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1")
]

for label, key in metrics:

    print(
        f"{label:<15}"
        f"{production_results[key]:<15.6f}"
        f"{candidate_results[key]:<15.6f}"
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\nProduction Confusion Matrix:")

print(
    production_results[
        "confusion_matrix"
    ]
)

print("\nCandidate Confusion Matrix:")

print(
    candidate_results[
        "confusion_matrix"
    ]
)


# ============================================================
# DIFFERENCES
# ============================================================

print("\n==========================================")
print("METRIC DIFFERENCES")
print("==========================================")

for label, key in metrics:

    difference = (
        candidate_results[key]
        - production_results[key]
    )

    print(
        f"{label:<15}"
        f"{difference:+.6f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON COMPLETE")
print("==========================================")

print(
    "\nNo model was replaced or modified."
)