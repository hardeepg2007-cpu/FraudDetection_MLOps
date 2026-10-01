import pandas as pd
import os

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ==========================================
# 1. File paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_FILE = os.path.join(
    BASE_DIR, "data", "test_processed.csv.gz"
)

MODEL_FILE = os.path.join(
    BASE_DIR, "models", "xgboost_fraud_model.json"
)

RESULT_DIR = os.path.join(
    BASE_DIR, "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================
# 2. Load test data
# ==========================================

print("Loading test data...")

test = pd.read_csv(TEST_FILE)

print("Test shape:", test.shape)


# ==========================================
# 3. Separate features and target
# ==========================================

TARGET = "isFraud"

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]


print("\nNormal transactions:", (y_test == 0).sum())
print("Fraud transactions:", (y_test == 1).sum())


# ==========================================
# 4. Load trained XGBoost model
# ==========================================

print("\nLoading trained model...")

model = XGBClassifier()

model.load_model(MODEL_FILE)

print("Model loaded successfully.")


# ==========================================
# 5. Generate predictions
# ==========================================

print("\nGenerating test predictions...")

y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.5

y_pred = (y_prob >= threshold).astype(int)


# ==========================================
# 6. Calculate metrics
# ==========================================

roc_auc = roc_auc_score(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)


# ==========================================
# 7. Display results
# ==========================================

print("\n==========================================")
print("TEST SET PERFORMANCE")
print("==========================================")

print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ==========================================
# 8. Save results
# ==========================================

results = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "threshold": threshold,
    "true_negatives": cm[0, 0],
    "false_positives": cm[0, 1],
    "false_negatives": cm[1, 0],
    "true_positives": cm[1, 1]
}

results_path = os.path.join(
    RESULT_DIR,
    "test_evaluation_results.csv"
)

pd.DataFrame([results]).to_csv(
    results_path,
    index=False
)


print("\nTest results saved to:")

print(results_path)

print("\n==========================================")
print("TEST EVALUATION COMPLETE!")
print("==========================================")