import pandas as pd
import os
import mlflow
import json

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

train = pd.read_csv(TRAIN_FILE)

print("Training data loaded.")

print("\nLoading validation data...")

validation = pd.read_csv(VAL_FILE)

print("Validation data loaded.")


print("\nTraining shape:", train.shape)
print("Validation shape:", validation.shape)


# ============================================================
# PREPARE FEATURES
# ============================================================

TARGET = "isFraud"

DROP_COLUMNS = [
    TARGET,
    "isFlaggedFraud"
]

X_train = train.drop(
    columns=DROP_COLUMNS
)

y_train = train[TARGET]

X_val = validation.drop(
    columns=DROP_COLUMNS
)

y_val = validation[TARGET]


print("\nNumber of training features:", X_train.shape[1])


# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nFraud cases:", positive)
print("Normal cases:", negative)

print(
    "Scale pos weight:",
    scale_pos_weight
)

# ============================================================
# START MLFLOW RUN
# ============================================================

mlflow.set_experiment(
    "Fraud_Detection_MLOps"
)

mlflow.start_run(
    run_name="XGBoost_Candidate_Retraining"
)

print("\nMLflow retraining run started.")

# ============================================================
# LOAD DATA DRIFT INFORMATION
# ============================================================

DATA_DRIFT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "data_drift_summary.json"
)

with open(DATA_DRIFT_FILE, "r") as f:
    drift = json.load(f)

drift_percentage = float(
    drift["drift_percentage"]
)

drifted_columns = float(
    drift["drifted_columns"]
)

total_columns = float(
    drift["total_columns"]
)

# ============================================================
# CREATE CANDIDATE MODEL
# ============================================================

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

# ============================================================
# LOG MLFLOW PARAMETERS
# ============================================================

mlflow.log_param(
    "model",
    "XGBoost"
)

mlflow.log_param(
    "n_estimators",
    1000
)

mlflow.log_param(
    "max_depth",
    8
)

mlflow.log_param(
    "learning_rate",
    0.05
)

mlflow.log_param(
    "subsample",
    0.8
)

mlflow.log_param(
    "colsample_bytree",
    0.8
)

mlflow.log_param(
    "scale_pos_weight",
    float(scale_pos_weight)
)

mlflow.log_param(
    "candidate_model",
    "xgboost_fraud_model_candidate.json"
)


# ============================================================
# LOG DATA DRIFT METRICS
# ============================================================

mlflow.log_metric(
    "drift_percentage",
    drift_percentage
)

mlflow.log_metric(
    "drifted_columns",
    drifted_columns
)

mlflow.log_metric(
    "total_columns",
    total_columns
)

print("\nMLflow drift metrics logged.")
# ============================================================
# TRAIN CANDIDATE MODEL
# ============================================================

print("\n==========================================")
print("STARTING CANDIDATE MODEL TRAINING")
print("==========================================")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

y_prob = model.predict_proba(
    X_val
)[:, 1]

threshold = 0.5

y_pred = (
    y_prob >= threshold
).astype(int)


# ============================================================
# EVALUATION
# ============================================================

roc_auc = roc_auc_score(
    y_val,
    y_prob
)

pr_auc = average_precision_score(
    y_val,
    y_prob
)

# ============================================================
# LOG MLFLOW METRICS
# ============================================================

mlflow.log_metric(
    "candidate_roc_auc",
    float(roc_auc)
)

mlflow.log_metric(
    "candidate_pr_auc",
    float(pr_auc)
)

mlflow.log_metric(
    "validation_rows",
    float(len(y_val))
)

mlflow.log_metric(
    "training_rows",
    float(len(y_train))
)

print("\nMLflow metrics logged.")


print("\n==========================================")
print("CANDIDATE MODEL PERFORMANCE")
print("==========================================")

print(
    f"ROC-AUC : {roc_auc:.6f}"
)

print(
    f"PR-AUC  : {pr_auc:.6f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        digits=4
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        y_pred
    )
)


# ============================================================
# SAVE CANDIDATE MODEL
# ============================================================

candidate_path = os.path.join(
    MODEL_DIR,
    "xgboost_fraud_model_candidate.json"
)

model.save_model(
    candidate_path
)


print("\nCandidate model saved to:")

print(candidate_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("CANDIDATE MODEL TRAINING COMPLETE")
print("==========================================")

print(
    "Production model was NOT changed."
)

# ============================================================
# END MLFLOW RUN
# ============================================================

mlflow.end_run()

print("\nMLflow retraining run completed.")