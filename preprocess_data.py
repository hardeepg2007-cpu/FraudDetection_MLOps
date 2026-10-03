import os
import zipfile
import pandas as pd
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_FILE = os.path.join(BASE_DIR, "financial-fraud-detection-dataset.zip")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "train_processed.csv.gz")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation_processed.csv.gz")
TEST_FILE = os.path.join(DATA_DIR, "test_processed.csv.gz")

REQUIRED_TYPES = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

FEATURE_COLUMNS = [
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
]


def preprocess_chunk(df):
    df = df.copy()

    df["hour"] = (df["step"] % 24).astype("int16")
    df["day"] = (df["step"] // 24).astype("int16")

    df["orig_balance_error"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )

    df["dest_balance_error"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    df["amount_to_orig_balance"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    )

    df["amount_to_dest_balance"] = (
        df["amount"] / (df["oldbalanceDest"] + 1)
    )

    df["orig_balance_zero"] = (
        df["oldbalanceOrg"] == 0
    ).astype("int8")

    df["dest_balance_zero"] = (
        df["oldbalanceDest"] == 0
    ).astype("int8")

    df["dest_is_merchant"] = (
        df["nameDest"].astype(str).str.startswith("M")
    ).astype("int8")

    dummies = pd.get_dummies(
        df["type"],
        prefix="type",
        dtype="int8"
    )

    for col in REQUIRED_TYPES:
        if col not in dummies:
            dummies[col] = 0

    dummies = dummies[REQUIRED_TYPES]

    output = pd.concat(
        [
            df[FEATURE_COLUMNS],
            dummies,
            df[["isFlaggedFraud", "isFraud"]],
        ],
        axis=1
    )

    return output


print("=" * 60)
print("FRAUD DETECTION DATA PREPROCESSING")
print("=" * 60)

if not os.path.exists(ZIP_FILE):
    raise FileNotFoundError(
        f"Dataset ZIP not found: {ZIP_FILE}"
    )

print("\nOpening dataset...")

with zipfile.ZipFile(ZIP_FILE, "r") as z:

    csv_name = "Synthetic_Financial_datasets_log.csv"

    if csv_name not in z.namelist():
        raise FileNotFoundError(
            f"{csv_name} not found inside ZIP file."
        )

    with z.open(csv_name) as f:

        first_write = {
            "train": True,
            "validation": True,
            "test": True,
        }

        row_counts = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        for df in pd.read_csv(f, chunksize=200000):

            # Keep only the three required periods.
            df = df[
                (df["step"] <= 520)
                | (
                    (df["step"] >= 521)
                    & (df["step"] <= 631)
                )
                | (
                    (df["step"] >= 632)
                    & (df["step"] <= 743)
                )
            ].copy()

            if len(df) == 0:
                continue

            processed = preprocess_chunk(df)

            train = processed[
                processed["step"] <= 520
            ]

            validation = processed[
                (processed["step"] >= 521)
                & (processed["step"] <= 631)
            ]

            test = processed[
                (processed["step"] >= 632)
                & (processed["step"] <= 743)
            ]

            if len(train) > 0:
                train.to_csv(
                    TRAIN_FILE,
                    mode="w" if first_write["train"] else "a",
                    header=first_write["train"],
                    index=False,
                    compression="gzip",
                )
                first_write["train"] = False
                row_counts["train"] += len(train)

            if len(validation) > 0:
                validation.to_csv(
                    VALIDATION_FILE,
                    mode="w" if first_write["validation"] else "a",
                    header=first_write["validation"],
                    index=False,
                    compression="gzip",
                )
                first_write["validation"] = False
                row_counts["validation"] += len(validation)

            if len(test) > 0:
                test.to_csv(
                    TEST_FILE,
                    mode="w" if first_write["test"] else "a",
                    header=first_write["test"],
                    index=False,
                    compression="gzip",
                )
                first_write["test"] = False
                row_counts["test"] += len(test)

            print(
                f"Processed: "
                f"Train={row_counts['train']:,}, "
                f"Validation={row_counts['validation']:,}, "
                f"Test={row_counts['test']:,}"
            )

            del df
            del processed
            del train
            del validation
            del test

            gc.collect()


print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nTrain rows:      {row_counts['train']:,}")
print(f"Validation rows: {row_counts['validation']:,}")
print(f"Test rows:       {row_counts['test']:,}")

print("\nFiles created:")
print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)

print("=" * 60)