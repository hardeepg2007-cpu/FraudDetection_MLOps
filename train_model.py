import pandas as pd
import numpy as np
import os
import joblib

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

TRAIN_FILE = os.path.join(BASE_DIR, "data", "train_processed.csv.gz")
VAL_FILE = os.path.join(BASE_DIR, "data", "validation_processed.csv.gz")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================
# 2. Load datasets
# ==========================================

print("Loading training data...")
train = pd.read_csv(TRAIN_FILE)

print("Loading validation data...")
validation = pd.read_csv(VAL_FILE)

print("\nTraining shape:", train.shape)
print("Validation shape:", validation.shape)


# ==========================================
# 3. Separate features and target
# ==========================================

TARGET = "isFraud"

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_val = validation.drop(columns=[TARGET])
y_val = validation[TARGET]


# ==========================================
# 4. Calculate class imbalance
# ==========================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nNormal transactions:", negative)
print("Fraud transactions:", positive)
print("Scale positive weight:", scale_pos_weight)


# ==========================================
# 5. Create XGBoost model
# ==========================================

model = XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. Train model
# ==========================================

print("\nStarting XGBoost training...")
print("This may take some time.")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


# ==========================================
# 7. Predictions
# ==========================================

print("\nGenerating validation predictions...")

y_prob = model.predict_proba(X_val)[:, 1]

# Default threshold
threshold = 0.5

y_pred = (y_prob >= threshold).astype(int)


# ==========================================
# 8. Evaluation
# ==========================================

roc_auc = roc_auc_score(y_val, y_prob)
pr_auc = average_precision_score(y_val, y_prob)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))


# ==========================================
# 9. Save model
# ==========================================

model_path = os.path.join(MODEL_DIR, "xgboost_fraud_model.json")

model.save_model(model_path)

print("\nModel saved to:")
print(model_path)


# ==========================================
# 10. Save evaluation results
# ==========================================

results = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "scale_pos_weight": scale_pos_weight,
    "threshold": threshold
}

results_path = os.path.join(RESULT_DIR, "evaluation_results.csv")

pd.DataFrame([results]).to_csv(results_path, index=False)

print("\nEvaluation results saved to:")
print(results_path)

print("\n==========================================")
print("TRAINING COMPLETE!")
print("==========================================")