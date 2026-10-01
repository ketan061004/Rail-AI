
import os
import joblib
import numpy as np
import pandas as pd

# ============================================================
# MODEL 2 - AI BLOCK PLANNER
# Uses Model 1 predicted priority scores
# ============================================================

MODEL_PATH = "models/priority_model.pkl"
OUTPUT_PATH = "outputs/ai_block_plan_24hr.csv"

# ------------------------------------------------------------
# Load Model 1
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model 1 not found at {MODEL_PATH}. "
        "Run train_priority_model.py first."
    )

priority_model = joblib.load(MODEL_PATH)

print("\nMODEL 2 - AI BLOCK PLANNER")
print("Model 1 priority model loaded successfully.")


# ------------------------------------------------------------
# Maintenance tasks
# These represent existing maintenance tasks coming from
# railway maintenance systems in our prototype.
# ------------------------------------------------------------

tasks = pd.DataFrame([
    {
        "task_id": "MT-001",
        "department": "Engineering",
        "maintenance_type": "Track Repair",
        "location_km": 45.2,
        "required_duration": 90,
        "section": "Section-A",

        # Features used by Model 1
        "rail_wear_mm": 7.5,
        "track_vibration_level": 7.2,
        "wheel_wear_percent": 65,
        "axle_temperature_c": 78,
        "bearing_temperature_c": 82,
        "traction_motor_temp_c": 88,
        "delay_minutes": 18,
        "last_maintenance_days": 120,
        "load_factor_percent": 82,
        "daily_trips": 14,
        "inspection_score": 58,
        "sensor_health_index": 91,
    },

    {
        "task_id": "MT-002",
        "department": "S&T",
        "maintenance_type": "Signal Repair",
        "location_km": 47.8,
        "required_duration": 60,
        "section": "Section-A",

        "rail_wear_mm": 5.2,
        "track_vibration_level": 5.5,
        "wheel_wear_percent": 48,
        "axle_temperature_c": 70,
        "bearing_temperature_c": 75,
        "traction_motor_temp_c": 82,
        "delay_minutes": 12,
        "last_maintenance_days": 95,
        "load_factor_percent": 74,
        "daily_trips": 12,
        "inspection_score": 65,
        "sensor_health_index": 94,
    },

    {
        "task_id": "MT-003",
        "department": "Traction",
        "maintenance_type": "OHE Maintenance",
        "location_km": 52.1,
        "required_duration": 120,
        "section": "Section-B",

        "rail_wear_mm": 4.8,
        "track_vibration_level": 5.0,
        "wheel_wear_percent": 45,
        "axle_temperature_c": 68,
        "bearing_temperature_c": 73,
        "traction_motor_temp_c": 84,
        "delay_minutes": 10,
        "last_maintenance_days": 90,
        "load_factor_percent": 70,
        "daily_trips": 10,
        "inspection_score": 68,
        "sensor_health_index": 95,
    },

    {
        "task_id": "MT-004",
        "department": "Engineering",
        "maintenance_type": "Track Inspection",
        "location_km": 55.4,
        "required_duration": 60,
        "section": "Section-B",

        "rail_wear_mm": 3.8,
        "track_vibration_level": 4.2,
        "wheel_wear_percent": 35,
        "axle_temperature_c": 65,
        "bearing_temperature_c": 70,
        "traction_motor_temp_c": 80,
        "delay_minutes": 6,
        "last_maintenance_days": 70,
        "load_factor_percent": 65,
        "daily_trips": 8,
        "inspection_score": 75,
        "sensor_health_index": 97,
    }
])


# ------------------------------------------------------------
# Prepare input for Model 1
#
# IMPORTANT:
# Model 1 was trained using the original railway dataset.
# Therefore we provide the same feature names expected by
# the model wherever possible.
# ------------------------------------------------------------

MODEL1_FEATURES = [
    "region",
    "season",
    "train_type",
    "train_age_years",
    "average_speed_kmph",
    "distance_travelled_km",
    "track_temperature_c",
    "rail_wear_mm",
    "track_vibration_level",
    "ballast_condition",
    "track_curvature_degree",
    "ambient_temperature_c",
    "humidity_percent",
    "rainfall_mm",
    "wind_speed_kmph",
    "wheel_wear_percent",
    "axle_temperature_c",
    "brake_pressure_psi",
    "brake_pad_wear_percent",
    "bearing_temperature_c",
    "battery_voltage",
    "traction_motor_temp_c",
    "signal_system_status",
    "power_consumption_kw",
    "load_factor_percent",
    "daily_trips",
    "delay_minutes",
    "last_maintenance_days",
    "inspection_score",
    "sensor_health_index"
]


# ------------------------------------------------------------
# Create prototype input compatible with Model 1
# ------------------------------------------------------------

model_input = pd.DataFrame()

for feature in MODEL1_FEATURES:

    if feature in tasks.columns:
        model_input[feature] = tasks[feature]

    elif feature == "region":
        model_input[feature] = "Central"

    elif feature == "season":
        model_input[feature] = "Winter"

    elif feature == "train_type":
        model_input[feature] = "Express"

    elif feature == "ballast_condition":
        model_input[feature] = "Good"

    elif feature == "signal_system_status":
        model_input[feature] = "Normal"

    elif feature == "train_age_years":
        model_input[feature] = 8

    elif feature == "average_speed_kmph":
        model_input[feature] = 100

    elif feature == "distance_travelled_km":
        model_input[feature] = 250000

    elif feature == "track_temperature_c":
        model_input[feature] = 35

    elif feature == "track_curvature_degree":
        model_input[feature] = 2.5

    elif feature == "ambient_temperature_c":
        model_input[feature] = 30

    elif feature == "humidity_percent":
        model_input[feature] = 65

    elif feature == "rainfall_mm":
        model_input[feature] = 5

    elif feature == "wind_speed_kmph":
        model_input[feature] = 10

    elif feature == "brake_pressure_psi":
        model_input[feature] = 110

    elif feature == "brake_pad_wear_percent":
        model_input[feature] = 45

    elif feature == "battery_voltage":
        model_input[feature] = 24

    elif feature == "power_consumption_kw":
        model_input[feature] = 500

    else:
        model_input[feature] = 0


# ------------------------------------------------------------
# Predict priority using Model 1
# ------------------------------------------------------------

predicted_priority = priority_model.predict(model_input)

# Keep scores in 0-100 range
predicted_priority = np.clip(predicted_priority, 0, 100)

tasks["priority_score"] = np.round(predicted_priority, 2)


print("\nMODEL 1 → MODEL 2 DATA FLOW")
print("--------------------------------")

for _, row in tasks.iterrows():
    print(
        f"{row['task_id']} | "
        f"{row['maintenance_type']} | "
        f"AI Priority = {row['priority_score']}/100"
    )


# ============================================================
# TRAIN TIMETABLE
# ============================================================

trains = [
    # Section A
    ("T001", "Section-A", "01:00", "01:15"),
    ("T002", "Section-A", "03:00", "03:15"),
    ("T003", "Section-A", "05:00", "05:15"),
    ("T004", "Section-A", "07:00", "07:15"),
    ("T005", "Section-A", "09:00", "09:15"),
    ("T006", "Section-A", "11:00", "11:15"),
    ("T007", "Section-A", "14:00", "14:15"),
    ("T008", "Section-A", "17:00", "17:15"),
    ("T009", "Section-A", "20:00", "20:15"),
    ("T010", "Section-A", "22:00", "22:15"),

    # Section B
    ("T011", "Section-B", "02:00", "02:15"),
    ("T012", "Section-B", "04:00", "04:15"),
    ("T013", "Section-B", "06:00", "06:15"),
    ("T014", "Section-B", "08:00", "08:15"),
    ("T015", "Section-B", "10:00", "10:15"),
    ("T016", "Section-B", "12:00", "12:15"),
    ("T017", "Section-B", "15:00", "15:15"),
    ("T018", "Section-B", "18:00", "18:15"),
    ("T019", "Section-B", "21:00", "21:15"),
    ("T020", "Section-B", "23:00", "23:15"),
]


def time_to_minutes(time_str):
    h, m = map(int, time_str.split(":"))
    return h * 60 + m


train_data = []

for train_id, section, start, end in trains:
    train_data.append({
        "train_id": train_id,
        "section": section,
        "start": time_to_minutes(start),
        "end": time_to_minutes(end)
    })

train_df = pd.DataFrame(train_data)


# ============================================================
# GENERATE 24-HOUR CANDIDATE BLOCKS
# ============================================================

candidate_windows = []

durations = [60, 90, 120]

for start in range(0, 1440, 30):

    for duration in durations:

        end = start + duration

        if end <= 1440:

            candidate_windows.append({
                "start": start,
                "end": end,
                "duration": duration
            })

print(f"\nCandidate block windows generated: {len(candidate_windows)}")


# ============================================================
# TRAIN CONFLICT CALCULATION
# ============================================================

def calculate_train_conflicts(section, block_start, block_end):

    conflicts = []

    section_trains = train_df[
        train_df["section"] == section
    ]

    for _, train in section_trains.iterrows():

        overlap = (
            block_start < train["end"]
            and block_end > train["start"]
        )

        if overlap:
            conflicts.append(train["train_id"])

    return conflicts


# ============================================================
# MAINTENANCE SCHEDULING
# ============================================================

occupied_blocks = {
    "Section-A": [],
    "Section-B": []
}

scheduled = []


# Highest priority tasks scheduled first
tasks = tasks.sort_values(
    by="priority_score",
    ascending=False
).reset_index(drop=True)


for _, task in tasks.iterrows():

    best_option = None

    for window in candidate_windows:

        start = window["start"]
        end = window["end"]

        # Duration must be sufficient
        if window["duration"] < task["required_duration"]:
            continue

        # Check maintenance overlap
        maintenance_conflict = False

        for occupied in occupied_blocks[task["section"]]:

            if (
                start < occupied["end"]
                and end > occupied["start"]
            ):
                maintenance_conflict = True
                break

        if maintenance_conflict:
            continue

        # Check train conflicts
        conflicts = calculate_train_conflicts(
            task["section"],
            start,
            end
        )

        affected_trains = len(conflicts)

        # Operational disruption cost
        disruption_cost = (
            affected_trains * 25
            + window["duration"] * 0.10
        )

        # Higher priority gives greater benefit
        priority_benefit = task["priority_score"] * 0.50

        optimization_score = (
            priority_benefit - disruption_cost
        )

        option = {
            "start": start,
            "end": end,
            "duration": window["duration"],
            "affected_trains": affected_trains,
            "conflicting_trains": conflicts,
            "optimization_score": optimization_score
        }

        # Best:
        # 1. Fewest affected trains
        # 2. Highest optimization score
        # 3. Shortest duration

        if best_option is None:

            best_option = option

        else:

            current_key = (
                best_option["affected_trains"],
                -best_option["optimization_score"],
                best_option["duration"]
            )

            new_key = (
                option["affected_trains"],
                -option["optimization_score"],
                option["duration"]
            )

            if new_key < current_key:
                best_option = option

    # --------------------------------------------------------
    # Schedule selected block
    # --------------------------------------------------------

    if best_option is not None:

        occupied_blocks[task["section"]].append({
            "start": best_option["start"],
            "end": best_option["end"]
        })

        scheduled.append({
            "task_id": task["task_id"],
            "department": task["department"],
            "maintenance_type": task["maintenance_type"],
            "location_km": task["location_km"],
            "priority_score": task["priority_score"],
            "required_duration": task["required_duration"],
            "section": task["section"],
            "block_start": best_option["start"],
            "block_end": best_option["end"],
            "block_duration": best_option["duration"],
            "affected_trains": best_option["affected_trains"],
            "conflicting_trains": (
                ", ".join(best_option["conflicting_trains"])
                if best_option["conflicting_trains"]
                else "None"
            ),
            "optimization_score": round(
                best_option["optimization_score"], 2
            ),
            "status": "RECOMMENDED"
        })


# ============================================================
# FORMAT TIME
# ============================================================

def minutes_to_time(minutes):

    hours = (minutes // 60) % 24
    mins = minutes % 60

    return f"{hours:02d}:{mins:02d}"


result_df = pd.DataFrame(scheduled)

result_df["block_start"] = result_df["block_start"].apply(
    minutes_to_time
)

result_df["block_end"] = result_df["block_end"].apply(
    minutes_to_time
)


# Sort by block time
result_df["_sort_time"] = result_df["block_start"].apply(
    time_to_minutes
)

result_df = result_df.sort_values(
    "_sort_time"
).drop(columns=["_sort_time"])


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 75)
print("AI 24-HOUR BLOCK PLAN")
print("=" * 75)

print(
    result_df[
        [
            "task_id",
            "department",
            "maintenance_type",
            "priority_score",
            "section",
            "block_start",
            "block_end",
            "block_duration",
            "affected_trains",
            "optimization_score",
            "status"
        ]
    ].to_string(index=False)
)


# ============================================================
# FINAL CONFLICT CHECK
# ============================================================

print("\nFINAL SCHEDULE CONFLICT CHECK")

conflict_found = False

for section in occupied_blocks:

    blocks = occupied_blocks[section]

    for i in range(len(blocks)):

        for j in range(i + 1, len(blocks)):

            a = blocks[i]
            b = blocks[j]

            if (
                a["start"] < b["end"]
                and b["start"] < a["end"]
            ):
                conflict_found = True


if conflict_found:

    print("WARNING: Maintenance conflict detected.")

else:

    print("SUCCESS: Final schedule is conflict-free.")


# ============================================================
# SAVE OUTPUT
# ============================================================

os.makedirs("outputs", exist_ok=True)

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved AI block plan to: {OUTPUT_PATH}"
)

