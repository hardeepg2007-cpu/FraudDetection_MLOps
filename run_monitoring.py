import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


print("=" * 60)
print("FRAUD DETECTION - MLOPS MONITORING PIPELINE")
print("=" * 60)


# ============================================================
# STEP 1 — DATA DRIFT MONITORING
# ============================================================

print("\n[1/2] Running Data Drift Monitoring...")
print("-" * 60)

result = subprocess.run(
    [sys.executable, "drift_monitoring.py"],
    cwd=BASE_DIR
)

if result.returncode != 0:
    print("\nData drift monitoring failed.")
    sys.exit(1)

print("\nData drift monitoring completed successfully.")


# ============================================================
# STEP 2 — CONCEPT DRIFT MONITORING
# ============================================================

print("\n[2/2] Running Concept Drift Monitoring...")
print("-" * 60)

result = subprocess.run(
    [sys.executable, "concept_drift_monitoring.py"],
    cwd=BASE_DIR
)

if result.returncode != 0:
    print("\nConcept drift monitoring failed.")
    sys.exit(1)

print("\nConcept drift monitoring completed successfully.")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MLOPS MONITORING PIPELINE COMPLETED")
print("=" * 60)

print("\nGenerated monitoring files:")

print(
    os.path.join(
        BASE_DIR,
        "results",
        "data_drift_report.html"
    )
)

print(
    os.path.join(
        BASE_DIR,
        "results",
        "concept_drift_results.csv"
    )
)

print("\nMonitoring workflow completed successfully.")