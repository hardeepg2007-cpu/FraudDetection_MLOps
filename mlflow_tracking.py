import mlflow
import mlflow.xgboost
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "production_evaluation_results.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_fraud_model_production.json"
)

print("Loading evaluation results...")
results = pd.read_csv(RESULT_FILE).iloc[0]

print("Starting MLflow run...")

mlflow.set_experiment("Fraud_Detection_MLOps")

with mlflow.start_run(run_name="XGBoost_Production_Model"):

    # Log parameters
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("objective", "binary:logistic")
    mlflow.log_param("n_estimators", 1000)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.log_param("threshold", float(results["threshold"]))
    mlflow.log_param("removed_feature", "isFlaggedFraud")

    # Log metrics
    mlflow.log_metric("roc_auc", float(results["roc_auc"]))
    mlflow.log_metric("pr_auc", float(results["pr_auc"]))

    # Log evaluation file
    mlflow.log_artifact(RESULT_FILE)

    print("\nMLflow run completed successfully.")

    print("Run ID:")
    print(mlflow.active_run().info.run_id)

print("\nExperiment: Fraud_Detection_MLOps")