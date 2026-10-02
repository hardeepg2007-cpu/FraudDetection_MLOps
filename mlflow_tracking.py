import mlflow
import pandas as pd
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "production_evaluation_results.csv"
)

CONCEPT_DRIFT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "concept_drift_results.csv"
)

DATA_DRIFT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "data_drift_summary.json"
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("Loading evaluation results...")

results = pd.read_csv(
    RESULT_FILE
).iloc[0]


print("Loading concept drift results...")

concept_drift = pd.read_csv(
    CONCEPT_DRIFT_FILE
)


print("Starting MLflow run...")


# ============================================================
# MLflow EXPERIMENT
# ============================================================

mlflow.set_experiment(
    "Fraud_Detection_MLOps"
)


# ============================================================
# START RUN
# ============================================================

with mlflow.start_run(
    run_name="XGBoost_Production_Model"
):

    # --------------------------------------------------------
    # MODEL PARAMETERS
    # --------------------------------------------------------

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "objective",
        "binary:logistic"
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
        "threshold",
        float(results["threshold"])
    )

    mlflow.log_param(
        "removed_feature",
        "isFlaggedFraud"
    )


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    mlflow.log_metric(
        "roc_auc",
        float(results["roc_auc"])
    )

    mlflow.log_metric(
        "pr_auc",
        float(results["pr_auc"])
    )


    # --------------------------------------------------------
    # FRAUD RATE
    # --------------------------------------------------------

    validation_row = concept_drift[
        concept_drift["period"] == "validation"
    ]

    if not validation_row.empty:

        fraud_rate = float(
            validation_row.iloc[0]["fraud_rate"]
        )

        mlflow.log_metric(
            "validation_fraud_rate",
            fraud_rate
        )


    # --------------------------------------------------------
    # DATA DRIFT
    # --------------------------------------------------------

    if os.path.exists(DATA_DRIFT_FILE):

        import json

        with open(
            DATA_DRIFT_FILE,
            "r"
        ) as f:

            drift = json.load(f)

        mlflow.log_metric(
            "drift_percentage",
            float(drift["drift_percentage"])
        )

        mlflow.log_metric(
            "drifted_columns",
            float(drift["drifted_columns"])
        )

        mlflow.log_metric(
            "total_columns",
            float(drift["total_columns"])
        )


    # --------------------------------------------------------
    # ARTIFACTS
    # --------------------------------------------------------

    mlflow.log_artifact(
        RESULT_FILE
    )

    mlflow.log_artifact(
        CONCEPT_DRIFT_FILE
    )

    if os.path.exists(DATA_DRIFT_FILE):

        mlflow.log_artifact(
            DATA_DRIFT_FILE
        )


    # --------------------------------------------------------
    # RUN INFORMATION
    # --------------------------------------------------------

    print("\nMLflow run completed successfully.")

    print("Run ID:")

    print(
        mlflow.active_run().info.run_id
    )


print(
    "\nExperiment: Fraud_Detection_MLOps"
)