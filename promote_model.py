import os
import shutil
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
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

BACKUP_MODEL = os.path.join(
    MODEL_DIR,
    "xgboost_fraud_model_production_backup.json"
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


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading production model...")

production_model = XGBClassifier()

production_model.load_model(
    PRODUCTION_MODEL
)


print("Loading candidate model...")

candidate_model = XGBClassifier()

candidate_model.load_model(
    CANDIDATE_MODEL
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model):

    y_prob = model.predict_proba(
        X_val
    )[:, 1]

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_val,
            y_prob
        ),

        "pr_auc": average_precision_score(
            y_val,
            y_prob
        ),

        "precision": precision_score(
            y_val,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_val,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_val,
            y_pred,
            zero_division=0
        )
    }


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating production model...")

production_results = evaluate_model(
    production_model
)


print("Evaluating candidate model...")

candidate_results = evaluate_model(
    candidate_model
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("PROMOTION EVALUATION")
print("==========================================")

for metric in [
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1"
]:

    production_value = production_results[metric]
    candidate_value = candidate_results[metric]

    print(
        f"{metric.upper():<10}"
        f" Production: {production_value:.6f}"
        f"  Candidate: {candidate_value:.6f}"
    )

# ============================================================
# PROMOTION RULE
# ============================================================

candidate_is_better_or_equal = (

    candidate_results["roc_auc"]
    >= production_results["roc_auc"]

    and

    candidate_results["pr_auc"]
    >= production_results["pr_auc"]

    and

    candidate_results["precision"]
    >= production_results["precision"]

    and

    candidate_results["recall"]
    >= production_results["recall"]

    and

    candidate_results["f1"]
    >= production_results["f1"]
)


# At least one metric must improve
candidate_has_improvement = (

    candidate_results["roc_auc"]
    > production_results["roc_auc"]

    or

    candidate_results["pr_auc"]
    > production_results["pr_auc"]

    or

    candidate_results["precision"]
    > production_results["precision"]

    or

    candidate_results["recall"]
    > production_results["recall"]

    or

    candidate_results["f1"]
    > production_results["f1"]
)


promotion_approved = (
    candidate_is_better_or_equal
    and candidate_has_improvement
)
# ============================================================
# PROMOTE CANDIDATE
# ============================================================

if promotion_approved:

    print("\n==========================================")
    print("PROMOTION APPROVED")
    print("==========================================")

    print(
        "\nBacking up current production model..."
    )

    shutil.copy2(
        PRODUCTION_MODEL,
        BACKUP_MODEL
    )

    print(
        "Production backup created:"
    )

    print(BACKUP_MODEL)

    print(
        "\nPromoting candidate model..."
    )

    shutil.copy2(
        CANDIDATE_MODEL,
        PRODUCTION_MODEL
    )

    print(
        "\nCandidate model promoted successfully."
    )

else:

    print("\n==========================================")
    print("PROMOTION REJECTED")
    print("==========================================")

    print(
        "\nProduction model will remain unchanged."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("MODEL PROMOTION PROCESS COMPLETE")
print("==========================================")