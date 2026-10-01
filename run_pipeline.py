import subprocess
import sys
from pathlib import Path


# ============================================================
# AI-POWERED RAILWAY MAINTENANCE PLANNING
# MASTER PIPELINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL1 = BASE_DIR / "train_priority_model.py"
MODEL2 = BASE_DIR / "block_planner.py"
MODEL3 = BASE_DIR / "emergency_replanner.py"


def run_module(script, module_name):
    """
    Run one AI module and stop the pipeline if it fails.
    """

    print("\n")
    print("=" * 75)
    print(f"RUNNING {module_name}")
    print("=" * 75)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print("\n" + "!" * 75)
        print(f"{module_name} FAILED")
        print("!" * 75)

        sys.exit(result.returncode)

    print("\n" + "-" * 75)
    print(f"{module_name} COMPLETED SUCCESSFULLY")
    print("-" * 75)


# ============================================================
# CHECK FILES
# ============================================================

required_scripts = [
    MODEL1,
    MODEL2,
    MODEL3
]

for script in required_scripts:

    if not script.exists():

        print(f"\nERROR: {script.name} not found.")

        print(
            "\nMake sure these files are in the same folder:"
        )

        print("  train_priority_model.py")
        print("  block_planner.py")
        print("  model3_emergency.py")
        print("  run_pipeline.py")

        sys.exit(1)


# ============================================================
# PIPELINE START
# ============================================================

print("\n")
print("=" * 75)
print(" AI-POWERED AUTOMATIC BLOCK PLANNING")
print(" INDIAN RAILWAYS - COMPLETE AI PIPELINE")
print("=" * 75)

print("\nPipeline:")
print("1. Maintenance Priority Engine")
print("2. AI Block Planner")
print("3. Emergency Dynamic Replanner")


# ============================================================
# MODEL 1
# ============================================================

run_module(
    MODEL1,
    "MODEL 1 - MAINTENANCE PRIORITY ENGINE"
)


# ============================================================
# MODEL 2
# ============================================================

run_module(
    MODEL2,
    "MODEL 2 - AI BLOCK PLANNER"
)


# ============================================================
# MODEL 3
# ============================================================

run_module(
    MODEL3,
    "MODEL 3 - EMERGENCY DYNAMIC REPLANNER"
)


# ============================================================
# FINAL OUTPUTS
# ============================================================

priority_model = BASE_DIR / "models" / "priority_model.pkl"

block_plan = (
    BASE_DIR /
    "outputs" /
    "ai_block_plan_24hr.csv"
)

emergency_plan = (
    BASE_DIR /
    "outputs" /
    "ai_emergency_replanned_schedule.csv"
)


print("\n")
print("=" * 75)
print(" COMPLETE AI PIPELINE FINISHED")
print("=" * 75)

print("\nGenerated Outputs:")

if priority_model.exists():
    print("✓ Model 1: priority_model.pkl")
else:
    print("✗ Model 1 output not found")


if block_plan.exists():
    print("✓ Model 2: ai_block_plan_24hr.csv")
else:
    print("✗ Model 2 output not found")


if emergency_plan.exists():
    print("✓ Model 3: ai_emergency_replanned_schedule.csv")
else:
    print("✗ Model 3 output not found")


print("\n")
print("=" * 75)
print(" END-TO-END AI PIPELINE SUCCESSFUL")
print("=" * 75)

print("\nDecision Flow:")
print("Maintenance Tasks")
print("       ↓")
print("AI Priority Scoring")
print("       ↓")
print("Optimal Maintenance Block")
print("       ↓")
print("Emergency Detected")
print("       ↓")
print("Dynamic Replanning")
print("       ↓")
print("Final Conflict-Free Schedule")

print("\nReady for dashboard integration.")