import zipfile
import pandas as pd
import os
import gc

ZIP_FILE = r"C:\Users\24ads35\Downloads\ezyZip.zip"
OUTPUT_FILE = r"C:\Users\24ads35\FraudDetection_MLOps\data\train_processed.csv.gz"

print("Opening original ZIP file...")
print("This may take a few minutes.")

first_chunk = True
total_rows = 0

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    with z.open("AIML Dataset.csv") as f:

        for df in pd.read_csv(f, chunksize=200000):

            # Training period: steps 1 to 520
            df = df[df["step"] <= 520].copy()

            if len(df) == 0:
                continue

            # Time features
            df["hour"] = (df["step"] % 24).astype("int16")
            df["day"] = (df["step"] // 24).astype("int16")

            # Balance features
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

            # Ratio features
            df["amount_to_orig_balance"] = (
                df["amount"] / (df["oldbalanceOrg"] + 1)
            )

            df["amount_to_dest_balance"] = (
                df["amount"] / (df["oldbalanceDest"] + 1)
            )

            # Zero-balance indicators
            df["orig_balance_zero"] = (
                df["oldbalanceOrg"] == 0
            ).astype("int8")

            df["dest_balance_zero"] = (
                df["oldbalanceDest"] == 0
            ).astype("int8")

            # Merchant indicator
            df["dest_is_merchant"] = (
                df["nameDest"].astype(str).str.startswith("M")
            ).astype("int8")

            # One-hot encode transaction type
            dummies = pd.get_dummies(
                df["type"],
                prefix="type",
                dtype="int8"
            )

            required_types = [
                "type_CASH_IN",
                "type_CASH_OUT",
                "type_DEBIT",
                "type_PAYMENT",
                "type_TRANSFER"
            ]

            for col in required_types:
                if col not in dummies:
                    dummies[col] = 0

            dummies = dummies[required_types]

            # Final feature columns
            columns = [
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
                "dest_is_merchant"
            ]

            output = pd.concat(
                [
                    df[columns],
                    dummies,
                    df[["isFlaggedFraud", "isFraud"]]
                ],
                axis=1
            )

            output.to_csv(
                OUTPUT_FILE,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
                compression="gzip"
            )

            first_chunk = False
            total_rows += len(output)

            print(
                "Processed training rows:",
                f"{total_rows:,}"
            )

            del df
            del output
            gc.collect()

print()
print("======================================")
print("TRAIN FILE CREATED SUCCESSFULLY")
print("======================================")
print("Rows:", f"{total_rows:,}")
print("File:", OUTPUT_FILE)