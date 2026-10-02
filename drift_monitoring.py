import pandas as pd
import os
import json

from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data",
    "train_processed.csv.gz"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "test_processed.csv.gz"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# PRODUCTION MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "step",
    "hour",
    "day",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_error",
    "dest_balance_error",
    "amount_to_orig_balance",
    "amount_to_dest_balance",
    "orig_balance_zero",
    "dest_balance_zero",
    "dest_is_merchant",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

print("Loading reference data...")

reference = pd.read_csv(
    TRAIN_FILE,
    usecols=MODEL_FEATURES
)

print("Reference data loaded.")


# ============================================================
# LOAD CURRENT DATA
# ============================================================

print("\nLoading current data...")

current = pd.read_csv(
    TEST_FILE,
    usecols=MODEL_FEATURES
)

print("Current data loaded.")


# ============================================================
# DISPLAY DATA SHAPES
# ============================================================

print("\nReference shape:", reference.shape)
print("Current shape:", current.shape)

print("\nNumber of features monitored:", len(MODEL_FEATURES))


# ============================================================
# GENERATE DATA DRIFT REPORT
# ============================================================

print("\nGenerating data drift report...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

snapshot = report.run(
    current_data=current,
    reference_data=reference
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    REPORT_DIR,
    "data_drift_report.html"
)

snapshot.save_html(report_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("DATA DRIFT MONITORING COMPLETE")
print("==========================================")

print("\nFeatures monitored:", len(MODEL_FEATURES))

print("\nReport saved to:")
print(report_path)

# ============================================================
# DATA DRIFT SUMMARY
# ============================================================

total_columns = len(MODEL_FEATURES)

# Get the drift results from Evidently
drift_results = snapshot.dict()

drifted_columns = 0

for metric in drift_results.get("metrics", []):
    metric_value = metric.get("value")

    if isinstance(metric_value, dict):
        if "count" in metric_value:
            drifted_columns = metric_value["count"]

        elif "share" in metric_value:
            drifted_columns = round(
                metric_value["share"] * total_columns
            )

dataset_drift = (
    drifted_columns / total_columns >= 0.50
)

drift_percentage = (
    drifted_columns / total_columns
) * 100

data_drift_summary = {
    "dataset_drift": dataset_drift,
    "drifted_columns": drifted_columns,
    "total_columns": total_columns,
    "drift_percentage": drift_percentage
}

summary_path = os.path.join(
    REPORT_DIR,
    "data_drift_summary.json"
)

with open(summary_path, "w") as f:
    json.dump(data_drift_summary, f, indent=4)

print("\nData drift summary saved to:")
print(summary_path)

# ============================================================
# RETRAINING TRIGGER
# ============================================================

RETRAINING_THRESHOLD = 25.0

if drift_percentage >= RETRAINING_THRESHOLD:

    print("\n⚠️ RETRAINING TRIGGERED")
    print(
        f"Drift percentage: {drift_percentage:.1f}%"
    )
    print(
        f"Threshold: {RETRAINING_THRESHOLD:.1f}%"
    )
    print(
        "Recommendation: Retraining should be started."
    )

else:

    print("\n✅ NO RETRAINING REQUIRED")
    print(
        f"Drift percentage: {drift_percentage:.1f}%"
    )
    print(
        f"Threshold: {RETRAINING_THRESHOLD:.1f}%"
    )