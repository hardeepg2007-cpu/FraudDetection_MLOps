import pandas as pd
import numpy as np
import os

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_fraud_model_production.json"
)

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data",
    "train_processed.csv.gz"
)

VAL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "validation_processed.csv.gz"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "test_processed.csv.gz"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)

TARGET = "isFraud"
DROP_COLUMNS = [TARGET, "isFlaggedFraud"]

print("Loading production model...")

model = XGBClassifier()
model.load_model(MODEL_FILE)

print("Production model loaded successfully.")

datasets = {
    "train": TRAIN_FILE,
    "validation": VAL_FILE,
    "test": TEST_FILE
}

results = []

for period, file_path in datasets.items():

    print(f"\nLoading {period} data...")

    df = pd.read_csv(file_path)

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    print(f"{period} rows:", len(df))
    print(f"{period} frauds:", int(y.sum()))

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    fraud_rate = y.mean()

    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    results.append({
        "period": period,
        "rows": len(df),
        "fraud_count": int(y.sum()),
        "fraud_rate": fraud_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(f"Fraud rate : {fraud_rate:.6f}")
    print(f"ROC-AUC    : {roc_auc:.6f}")
    print(f"PR-AUC     : {pr_auc:.6f}")
    print(f"Precision  : {precision:.6f}")
    print(f"Recall     : {recall:.6f}")
    print(f"F1-score   : {f1:.6f}")


results_df = pd.DataFrame(results)

output_file = os.path.join(
    RESULT_DIR,
    "concept_drift_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\n==========================================")
print("CONCEPT DRIFT MONITORING COMPLETE")
print("==========================================")

print("\nResults:")
print(results_df.to_string(index=False))

print("\nResults saved to:")
print(output_file)