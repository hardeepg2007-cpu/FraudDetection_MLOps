from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import os
from fastapi.responses import FileResponse
app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost-based financial transaction fraud detection API",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_fraud_model_production.json"
)

# Load production model
model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)


class Transaction(BaseModel):
    step: float
    hour: float
    day: float
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    orig_balance_error: float
    dest_balance_error: float
    amount_to_orig_balance: float
    amount_to_dest_balance: float
    orig_balance_zero: float
    dest_balance_zero: float
    dest_is_merchant: float
    type_CASH_IN: float
    type_CASH_OUT: float
    type_DEBIT: float
    type_PAYMENT: float
    type_TRANSFER: float


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "model": "XGBoost Production Model"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "xgboost_fraud_model_production.json"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    data = transaction.model_dump()

    df = pd.DataFrame([data])

    probability = float(model.predict_proba(df)[0][1])

    threshold = 0.5

    prediction = int(probability >= threshold)

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

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):

    # Check that uploaded file is CSV
    if not file.filename.lower().endswith(".csv"):
        return {"error": "Please upload a CSV file."}

    # Read CSV
    df = pd.read_csv(file.file)

    # Features required by the production model
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

    # Check for missing columns
    missing_columns = [
        col for col in MODEL_FEATURES
        if col not in df.columns
    ]

    if missing_columns:
        return {
            "error": "Missing required columns",
            "missing_columns": missing_columns
        }

    # Select model input columns
    X = df[MODEL_FEATURES]

    # Predict fraud probability
    probabilities = model.predict_proba(X)[:, 1]

    # Apply threshold
    predictions = (probabilities >= 0.5).astype(int)

    # Add prediction results to CSV
    df["prediction"] = predictions

    df["result"] = df["prediction"].map({
        0: "LEGITIMATE",
        1: "FRAUD"
    })

    df["fraud_probability"] = probabilities

    # Save output CSV
    output_file = os.path.join(BASE_DIR, "predictions.csv")

    df.to_csv(output_file, index=False)

    # Return downloadable file
    return FileResponse(
        path=output_file,
        filename="predictions.csv",
        media_type="text/csv"
    )