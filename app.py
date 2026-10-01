from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json
from io import BytesIO
from fastapi.responses import FileResponse, StreamingResponse


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost-based financial transaction fraud detection API",
    version="2.0"
)

# Local development remains available; this also permits Vercel production and
# preview deployments to call the public API.
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    r"https://.*\.vercel\.app",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175"
    ],
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_fraud_model_production.json"
)


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)


# ============================================================
# RAW TRANSACTION INPUT
# ============================================================

class Transaction(BaseModel):

    step: float
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Hour and day from step
    # --------------------------------------------------------

    df["hour"] = df["step"] % 24
    df["day"] = (df["step"] // 24) + 1

    # --------------------------------------------------------
    # Balance errors
    # --------------------------------------------------------

    df["orig_balance_error"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    df["dest_balance_error"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    # --------------------------------------------------------
    # Amount ratios
    # --------------------------------------------------------

    df["amount_to_orig_balance"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / df["oldbalanceOrg"],
        0
    )

    df["amount_to_dest_balance"] = np.where(
        df["oldbalanceDest"] > 0,
        df["amount"] / df["oldbalanceDest"],
        0
    )

    # --------------------------------------------------------
    # Zero balance indicators
    # --------------------------------------------------------

    df["orig_balance_zero"] = (
        df["newbalanceOrig"] == 0
    ).astype(int)

    df["dest_balance_zero"] = (
        df["newbalanceDest"] == 0
    ).astype(int)

    # --------------------------------------------------------
    # Merchant indicator
    #
    # In the original dataset, merchant destinations are
    # represented by destination account patterns.
    # For API input, we use destination balance information
    # as a simple indicator.
    # --------------------------------------------------------

    df["dest_is_merchant"] = (
        (df["oldbalanceDest"] == 0) &
        (df["newbalanceDest"] > 0)
    ).astype(int)

    # --------------------------------------------------------
    # One-hot encode transaction type
    # --------------------------------------------------------

    transaction_types = [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    for transaction_type in transaction_types:

        column_name = "type_" + transaction_type

        df[column_name] = (
            df["type"].astype(str).str.upper()
            == transaction_type
        ).astype(int)

    # --------------------------------------------------------
    # Select EXACT model features
    # --------------------------------------------------------

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

    return df[MODEL_FEATURES]


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API is running",
        "model": "XGBoost Production Model",
        "version": "2.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "xgboost_fraud_model_production.json"
    }


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert request to DataFrame

    data = transaction.model_dump()

    df = pd.DataFrame([data])

    # Create model features

    X = create_features(df)

    # Predict probability

    probability = float(
        model.predict_proba(X)[0][1]
    )

    # Classification threshold

    threshold = 0.5

    prediction = int(
        probability >= threshold
    )

    # Result

    if prediction == 1:
        result = "FRAUD"
    else:
        result = "LEGITIMATE"

    return {

        "prediction": prediction,

        "result": result,

        "fraud_probability": probability,

        "threshold": threshold
    }


# ============================================================
# CSV BATCH PREDICTION
# ============================================================

@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Check file extension
    # --------------------------------------------------------

    if not file.filename.lower().endswith(".csv"):

        return {
            "error": "Please upload a CSV file."
        }

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(file.file)

    # --------------------------------------------------------
    # Required RAW columns
    # --------------------------------------------------------

    REQUIRED_COLUMNS = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    # --------------------------------------------------------
    # Check missing columns
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        return {

            "error": "Missing required columns",

            "missing_columns": missing_columns
        }

    # --------------------------------------------------------
    # Create model features
    # --------------------------------------------------------

    X = create_features(df)

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)[:, 1]

    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Add results
    # --------------------------------------------------------

    df["prediction"] = predictions

    df["result"] = df["prediction"].map({

        0: "LEGITIMATE",

        1: "FRAUD"
    })

    df["fraud_probability"] = probabilities

    # --------------------------------------------------------
    # Return an in-memory download. This avoids a shared predictions.csv file
    # being overwritten when multiple public users upload at the same time.
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        csv_buffer,
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        media_type="text/csv"
    )



# ============================================================
# CONCEPT DRIFT MONITORING
# ============================================================

@app.get("/concept-drift")
def get_concept_drift():

    file_path = os.path.join(
        BASE_DIR,
        "results",
        "concept_drift_results.csv"
    )

    if not os.path.exists(file_path):
        return {
            "error": "Concept drift results file not found"
        }

    df = pd.read_csv(file_path)

    return df.to_dict(orient="records")


@app.get("/data-drift")
def get_data_drift():

    file_path = os.path.join(
        BASE_DIR,
        "results",
        "data_drift_summary.json"
    )

    if not os.path.exists(file_path):
        return {
            "error": "Data drift summary file not found"
        }

    with open(file_path, "r") as f:
        return json.load(f)


# ============================================================
# DATA DRIFT REPORT
# ============================================================

@app.get("/data_drift_report.html")
def get_data_drift_report():

    file_path = os.path.join(
        BASE_DIR,
        "results",
        "data_drift_report.html"
    )

    if not os.path.exists(file_path):
        return {
            "error": "Data drift report not found"
        }

    return FileResponse(
        file_path,
        media_type="text/html"
    )
