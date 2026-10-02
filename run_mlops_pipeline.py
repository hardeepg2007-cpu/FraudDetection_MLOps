import os
import subprocess


# ============================================================
# MLOPS AUTOMATIC PIPELINE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def run_step(script_name):

    print("\n" + "=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        ["python", script_name],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print("\n❌ STEP FAILED")
        print(f"Script: {script_name}")

        raise SystemExit(1)

    print(f"\n✅ {script_name} completed successfully.")


# ============================================================
# PIPELINE
# ============================================================

print("\n")
print("=" * 60)
print("AUTOMATIC MLOPS PIPELINE")
print("=" * 60)


# Step 1: Data drift monitoring
run_step(
    "drift_monitoring.py"
)

# Step 2: Candidate model retraining
run_step(
    "retrain_model.py"
)


# Step 3: Compare production and candidate models
run_step(
    "compare_models.py"
)


# Step 4: Promote candidate model if it passes evaluation
run_step(
    "promote_model.py"
)


print("\n" + "=" * 60)
print("MLOPS PIPELINE COMPLETE")
print("=" * 60)