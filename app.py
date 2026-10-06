from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json
import sqlite3
import io
from io import BytesIO
from fastapi.responses import FileResponse, StreamingResponse
from datetime import datetime


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

DB_FILE = os.path.join(BASE_DIR, "prediction_history.db")


def _get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT DEFAULT 'manual',
        Time TEXT,
        transaction_id TEXT,
        type TEXT,
        amount REAL,
        currency TEXT,
        device_type TEXT,
        new_device TEXT,
        new_location TEXT,
        previous_fraud_count INTEGER,
        oldbalanceOrg REAL,
        newbalanceOrig REAL,
        oldbalanceDest REAL,
        newbalanceDest REAL,
        hour REAL,
        day REAL,
        prediction INTEGER,
        result TEXT,
        fraud_probability REAL,
        threshold REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS csv_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
        total_rows INTEGER,
        fraud_count INTEGER,
        legitimate_count INTEGER,
        fraud_rate REAL,
        avg_fraud_probability REAL,
        highest_fraud_probability REAL,
        csv_data TEXT
    )
    ''')
    try:
        cursor.execute("ALTER TABLE prediction_history ADD COLUMN detection_method TEXT DEFAULT 'xgboost'")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE prediction_history ADD COLUMN rule_reason TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


init_db()

# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)


# ============================================================
# RAW TRANSACTION INPUT
# ============================================================

class Transaction(BaseModel):
    step: Optional[float] = None
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    hour: Optional[float] = None
    day: Optional[float] = None
    Time: Optional[str] = None
    transaction_id: Optional[str] = None
    currency: Optional[str] = None
    device_type: Optional[str] = None
    new_device: Optional[str] = None
    new_location: Optional[str] = None
    previous_fraud_count: Optional[int] = None
    # Legacy fields kept for backward compatibility
    location: Optional[str] = None
    channel: Optional[str] = None
    previous_transaction_count: Optional[int] = None


# ============================================================
# FEATURE ENGINEERING (PRESERVED EXACTLY)
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


def _derive_step(data):
    """Derive step from Time string, or from hour/day, or default to 0."""
    if data.get("step") is not None:
        return

    time_str = data.get("Time")
    if time_str:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                dt = datetime.strptime(time_str.strip(), fmt)
                data["hour"] = float(dt.hour)
                data["day"] = float(dt.day)
                data["step"] = (data["day"] - 1) * 24 + data["hour"]
                return
            except ValueError:
                pass

    h = data.get("hour", 0) or 0
    d = data.get("day", 1) or 1
    data["step"] = (d - 1) * 24 + h


def _next_transaction_id():
    """Auto-generate a sequential transaction ID."""
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prediction_history")
    row = cursor.fetchone()
    conn.close()
    next_id = (row[0] or 0) + 1
    return f"TXN{next_id:05d}"


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
# INPUT VALIDATION
# ============================================================

class ValidationError(BaseModel):
    field: str
    message: str


class ValidationResult(BaseModel):
    valid: bool
    errors: List[ValidationError] = []


@app.post("/validate", response_model=ValidationResult)
def validate_transaction(transaction: Transaction):
    errors = []

    # 1. Negative amount
    if transaction.amount < 0:
        errors.append(ValidationError(
            field="amount",
            message="Amount cannot be negative."
        ))
    # 2. Zero amount
    elif transaction.amount == 0:
        errors.append(ValidationError(
            field="amount",
            message="Amount must be greater than zero."
        ))
    # 3. Amount > Old Balance (Origin)
    elif transaction.amount > 0 and transaction.oldbalanceOrg >= 0 and transaction.amount > transaction.oldbalanceOrg:
        errors.append(ValidationError(
            field="amount",
            message="Transaction amount cannot exceed the available origin balance."
        ))

    # 4. Negative Old Balance (Origin)
    if transaction.oldbalanceOrg < 0:
        errors.append(ValidationError(
            field="oldbalanceOrg",
            message="Balance cannot be negative."
        ))

    # 5. Negative Previous Fraud Count
    if transaction.previous_fraud_count is not None and transaction.previous_fraud_count < 0:
        errors.append(ValidationError(
            field="previous_fraud_count",
            message="Previous fraud count cannot be negative."
        ))

    # 6. Invalid transaction type
    valid_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    if transaction.type.upper() not in valid_types:
        errors.append(ValidationError(
            field="type",
            message=f"Invalid transaction type. Must be one of: {', '.join(valid_types)}"
        ))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

def check_fraud_rules(data: dict):
    try:
        amount = float(data.get("amount", 0))
    except (ValueError, TypeError):
        amount = 0.0
    try:
        oldbalanceOrg = float(data.get("oldbalanceOrg", 0))
    except (ValueError, TypeError):
        oldbalanceOrg = 0.0
    try:
        newbalanceOrig = float(data.get("newbalanceOrig", 0))
    except (ValueError, TypeError):
        newbalanceOrig = 0.0
    txn_type = str(data.get("type", "")).upper()

    if amount < 0:
        return True, "Invalid negative transaction amount"
    if amount == 0:
        return True, "Transaction amount cannot be zero"
    if oldbalanceOrg < 0:
        return True, "Invalid negative origin balance"
    if amount > oldbalanceOrg:
        return True, "Amount exceeds available origin balance"
    
    if txn_type in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        if abs(oldbalanceOrg - amount - newbalanceOrig) > 0.01:
            return True, "Origin balance does not match the transaction amount"
            
    return False, None


@app.post("/predict")
def predict(transaction: Transaction):

    # Convert request to DataFrame
    data = transaction.model_dump()

    _derive_step(data)

    # Auto-generate transaction_id if not provided
    txn_id = data.get("transaction_id") or _next_transaction_id()
    data["transaction_id"] = txn_id

    is_rule_fraud, rule_reason = check_fraud_rules(data)

    if is_rule_fraud:
        prediction = 1
        probability = 1.0
        result = "FRAUD"
        detection_method = "rule-based"
        r_reason = rule_reason
        threshold = 0.5
    else:
        df = pd.DataFrame([data])
        X = create_features(df)
        probability = float(model.predict_proba(X)[0][1])
        threshold = 0.5
        prediction = int(probability >= threshold)
        result = "FRAUD" if prediction == 1 else "LEGITIMATE"
        detection_method = "xgboost"
        r_reason = None

    response = {
        "prediction": prediction,
        "result": result,
        "fraud_probability": probability,
        "threshold": threshold,
        "detection_method": detection_method,
        "rule_reason": r_reason,
        "Time": data.get("Time"),
        "transaction_id": txn_id,
        "currency": data.get("currency"),
        "device_type": data.get("device_type"),
        "new_device": data.get("new_device"),
        "new_location": data.get("new_location"),
        "previous_fraud_count": data.get("previous_fraud_count"),
        "type": data.get("type"),
        "amount": data.get("amount"),
        "oldbalanceOrg": data.get("oldbalanceOrg"),
        "newbalanceOrig": data.get("newbalanceOrig"),
        "oldbalanceDest": data.get("oldbalanceDest"),
        "newbalanceDest": data.get("newbalanceDest"),
        "hour": data.get("hour"),
        "day": data.get("day"),
    }

    # Save to history DB
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prediction_history (
            source, Time, transaction_id, type, amount, currency,
            device_type, new_device, new_location,
            previous_fraud_count, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest, hour, day,
            prediction, result, fraud_probability, threshold,
            detection_method, rule_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        'manual',
        response["Time"], response["transaction_id"], response["type"], response["amount"],
        response["currency"], response["device_type"],
        response["new_device"], response["new_location"],
        response["previous_fraud_count"],
        response["oldbalanceOrg"], response["newbalanceOrig"], response["oldbalanceDest"],
        response["newbalanceDest"], response["hour"], response["day"],
        response["prediction"], response["result"], response["fraud_probability"], response["threshold"],
        response["detection_method"], response["rule_reason"]
    ))
    conn.commit()
    conn.close()

    return response


# ============================================================
# CSV BATCH PREDICTION
# ============================================================

@app.post("/predict_csv")
async def predict_csv(
    request: Request,
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

    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

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
    # Apply rules
    # --------------------------------------------------------
    rule_results = df.apply(lambda row: check_fraud_rules(row.to_dict()), axis=1)
    is_rule_fraud = rule_results.apply(lambda x: x[0])
    rule_reasons = rule_results.apply(lambda x: x[1])

    # --------------------------------------------------------
    # Create model features
    # --------------------------------------------------------

    X = create_features(df)

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)[:, 1]

    # --------------------------------------------------------
    # Apply threshold & override with rules
    # --------------------------------------------------------

    predictions = (probabilities >= 0.5).astype(int)
    
    probabilities[is_rule_fraud] = 1.0
    predictions[is_rule_fraud] = 1

    # --------------------------------------------------------
    # Add results
    # --------------------------------------------------------

    df["prediction"] = predictions

    df["result"] = df["prediction"].map({
        0: "LEGITIMATE",
        1: "FRAUD"
    })

    df["fraud_probability"] = probabilities

    df["threshold"] = 0.5
    
    df["detection_method"] = np.where(is_rule_fraud, "rule-based", "xgboost")
    df["rule_reason"] = rule_reasons

    # --------------------------------------------------------
    # Store CSV analysis in history
    # --------------------------------------------------------
    fraud_count = int(predictions.sum())
    total_rows = len(df)
    legit_count = total_rows - fraud_count
    fraud_rate = fraud_count / total_rows if total_rows > 0 else 0.0
    avg_prob = float(probabilities.mean()) if total_rows > 0 else 0.0
    max_prob = float(probabilities.max()) if total_rows > 0 else 0.0

    csv_result_buffer = BytesIO()
    df.to_csv(csv_result_buffer, index=False)
    csv_data_str = csv_result_buffer.getvalue().decode("utf-8")

    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO csv_history (
            filename, total_rows, fraud_count, legitimate_count,
            fraud_rate, avg_fraud_probability, highest_fraud_probability, csv_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        file.filename, total_rows, fraud_count, legit_count,
        fraud_rate, avg_prob, max_prob, csv_data_str
    ))
    conn.commit()
    conn.close()

    # --------------------------------------------------------
    # Return JSON if client requested json, else StreamingResponse
    # --------------------------------------------------------
    accept_hdr = request.headers.get("accept", "")
    if "application/json" in accept_hdr or request.query_params.get("format") == "json":
        return {
            "filename": file.filename,
            "total": total_rows,
            "fraud": fraud_count,
            "legitimate": legit_count,
            "fraud_rate": fraud_rate,
            "avg_fraud_probability": avg_prob,
            "highest_fraud_probability": max_prob,
            "predictions": df.to_dict(orient="records"),
        }

    csv_buffer = BytesIO(csv_data_str.encode("utf-8"))
    return StreamingResponse(
        csv_buffer,
        headers={"Content-Disposition": f"attachment; filename={file.filename or 'predictions.csv'}"},
        media_type="text/csv"
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.get("/prediction-history")
def get_prediction_history():
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/prediction-history/stats")
def get_prediction_history_stats():
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(prediction), AVG(fraud_probability), MAX(fraud_probability) FROM prediction_history")
    row = cursor.fetchone()
    conn.close()

    total = row[0] or 0
    fraud = int(row[1] or 0)
    legit = total - fraud
    fraud_rate = fraud / total if total > 0 else 0.0
    avg_prob = row[2] or 0.0
    max_prob = row[3] or 0.0

    return {
        "total_predictions": total,
        "fraud_predictions": fraud,
        "legitimate_predictions": legit,
        "fraud_rate": fraud_rate,
        "avg_fraud_probability": avg_prob,
        "highest_fraud_probability": max_prob
    }


@app.delete("/prediction-history")
def delete_prediction_history():
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
    return {"message": "Prediction history cleared"}


@app.get("/prediction-history/export")
def export_prediction_history():
    conn = _get_db()
    df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY created_at DESC", conn)
    conn.close()

    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        csv_buffer,
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
        media_type="text/csv"
    )


# ============================================================
# CSV HISTORY
# ============================================================

@app.get("/csv-history")
def get_csv_history():
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, uploaded_at, total_rows, fraud_count, legitimate_count, fraud_rate, avg_fraud_probability, highest_fraud_probability FROM csv_history ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/csv-history/{csv_id}")
def get_csv_history_detail(csv_id: int):
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM csv_history WHERE id = ?", (csv_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"error": "CSV history record not found"}
    result = dict(row)
    # Parse csv_data back into records
    csv_data_str = result.pop("csv_data", "")
    if csv_data_str:
        df = pd.read_csv(io.StringIO(csv_data_str))
        result["predictions"] = df.to_dict(orient="records")
        result["columns"] = list(df.columns)
    else:
        result["predictions"] = []
        result["columns"] = []
    return result


@app.delete("/csv-history")
def delete_csv_history():
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM csv_history")
    conn.commit()
    conn.close()
    return {"message": "CSV history cleared"}


@app.get("/csv-history/{csv_id}/export")
def export_csv_history(csv_id: int):
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT csv_data, filename FROM csv_history WHERE id = ?", (csv_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"error": "CSV history record not found"}
    csv_data_str = row[0] or ""
    filename = row[1] or "predictions.csv"
    return StreamingResponse(
        BytesIO(csv_data_str.encode("utf-8")),
        headers={"Content-Disposition": f"attachment; filename={filename}"},
        media_type="text/csv"
    )


# ============================================================
# COMBINED & COMPLETE ANALYTICS DATASET
# ============================================================

@app.get("/combined-stats")
def get_combined_stats():
    """Combined analytics summary from manual predictions + CSV predictions."""
    conn = _get_db()
    cursor = conn.cursor()

    # Manual prediction stats
    cursor.execute("SELECT COUNT(*), SUM(prediction), AVG(fraud_probability), MAX(fraud_probability) FROM prediction_history")
    m = cursor.fetchone()
    m_total = m[0] or 0
    m_fraud = int(m[1] or 0)
    m_avg = m[2] or 0.0
    m_max = m[3] or 0.0

    # CSV stats
    cursor.execute("SELECT SUM(total_rows), SUM(fraud_count), AVG(avg_fraud_probability), MAX(highest_fraud_probability) FROM csv_history")
    c = cursor.fetchone()
    c_total = int(c[0] or 0)
    c_fraud = int(c[1] or 0)
    c_avg = c[2] or 0.0
    c_max = c[3] or 0.0

    conn.close()

    total = m_total + c_total
    fraud = m_fraud + c_fraud
    legit = total - fraud
    fraud_rate = fraud / total if total > 0 else 0.0

    # Weighted average
    if total > 0:
        avg_prob = (m_avg * m_total + c_avg * c_total) / total if (m_total + c_total) > 0 else 0.0
    else:
        avg_prob = 0.0

    max_prob = max(m_max, c_max)

    return {
        "total_predictions": total,
        "fraud_predictions": fraud,
        "legitimate_predictions": legit,
        "fraud_rate": fraud_rate,
        "avg_fraud_probability": avg_prob,
        "highest_fraud_probability": max_prob,
        "manual_total": m_total,
        "manual_fraud": m_fraud,
        "csv_total": c_total,
        "csv_fraud": c_fraud,
    }


@app.get("/analytics-dataset")
def get_analytics_dataset():
    """Returns detailed real analytics records from both manual predictions and all CSV analyses."""
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_history ORDER BY created_at DESC")
    manual_rows = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM csv_history ORDER BY uploaded_at DESC")
    csv_rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    manual_records = []
    for r in manual_rows:
        manual_records.append({
            "source": "manual",
            "type": r.get("type", "UNKNOWN"),
            "prediction": r.get("prediction", 0),
            "result": r.get("result", "LEGITIMATE"),
            "fraud_probability": r.get("fraud_probability", 0.0),
            "Time": r.get("Time") or r.get("created_at", ""),
        })

    csv_records = []
    for c in csv_rows:
        csv_data_str = c.get("csv_data", "")
        if csv_data_str:
            try:
                df = pd.read_csv(io.StringIO(csv_data_str))
                for _, row in df.iterrows():
                    csv_records.append({
                        "source": "csv",
                        "type": str(row.get("type", "UNKNOWN")),
                        "prediction": int(row.get("prediction", 0)),
                        "result": str(row.get("result", "LEGITIMATE")),
                        "fraud_probability": float(row.get("fraud_probability", 0.0)),
                        "Time": str(row.get("Time", c.get("uploaded_at", ""))),
                    })
            except Exception:
                pass

    return {
        "manual": manual_records,
        "csv": csv_records,
        "combined": manual_records + csv_records,
    }


# ============================================================
# CONCEPT DRIFT MONITORING (PRESERVED EXACTLY)
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
# DATA DRIFT REPORT (PRESERVED EXACTLY)
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
