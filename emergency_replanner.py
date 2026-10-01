
import pandas as pd
import numpy as np
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")



# ============================================================
# MODEL 3 - AI DYNAMIC EMERGENCY REPLANNING
#
# MODEL 1 → PRIORITY
# MODEL 2 → BLOCK PLANNING
# MODEL 3 → EMERGENCY REPLANNING
# ============================================================

INPUT_PATH = "outputs/ai_block_plan_24hr.csv"
OUTPUT_PATH = "outputs/ai_emergency_replanned_schedule.csv"


print("\n" + "=" * 75)
print("MODEL 3 - AI DYNAMIC EMERGENCY REPLANNING")
print("=" * 75)


# ============================================================
# LOAD MODEL 2 OUTPUT
# ============================================================

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Model 2 output not found: {INPUT_PATH}"
    )

plan_df = pd.read_csv(INPUT_PATH)


if "priority_score" not in plan_df.columns:
    raise ValueError(
        "priority_score column missing from Model 2 output."
    )


# ============================================================
# MODEL 1 → MODEL 2 → MODEL 3
# ============================================================

print("\nMODEL 1 -> MODEL 2 -> MODEL 3 DATA FLOW")
print("----------------------------------------")

for _, row in plan_df.iterrows():

    print(
        f"{row['task_id']} | "
        f"AI Priority = {float(row['priority_score']):.2f}/100"
    )


# ============================================================
# CURRENT MODEL 2 PLAN
# ============================================================

print("\nCURRENT MODEL 2 BLOCK PLAN")
print("----------------------------------------")

for _, row in plan_df.iterrows():

    print(
        f"{row['task_id']:8} "
        f"{row['section']:10} "
        f"{row['block_start']} - {row['block_end']} "
        f"Priority: {float(row['priority_score']):.2f}"
    )


# ============================================================
# EMERGENCY
# ============================================================

emergency = {
    "task_id": "EMG-001",
    "department": "Engineering",
    "maintenance_type": "Critical Track Repair",
    "location_km": 45.0,
    "priority_score": 98.0,
    "duration_minutes": 90,
    "section": "Section-A",
    "detected_at": 23 * 60
}


print("\nEMERGENCY DEFECT DETECTED")
print("----------------------------------------")

print(f"Task ID          : {emergency['task_id']}")
print(f"Department       : {emergency['department']}")
print(f"Type             : {emergency['maintenance_type']}")
print(f"Location         : KM {emergency['location_km']}")
print(f"Priority         : {emergency['priority_score']:.0f}/100")
print(f"Duration         : {emergency['duration_minutes']} minutes")
print(f"Section          : {emergency['section']}")
print("Detected At      : 23:00")


# ============================================================
# TIME FUNCTIONS
# ============================================================

def time_to_minutes(value):

    hour, minute = map(int, str(value).split(":"))

    return hour * 60 + minute


def display_time(minutes):

    day = minutes // 1440
    minute_of_day = minutes % 1440

    hour = minute_of_day // 60
    minute = minute_of_day % 60

    time = f"{hour:02d}:{minute:02d}"

    if day == 0:
        return time

    return f"D+{day} {time}"


# ============================================================
# CONVERT MODEL 2 PLAN TO ROLLING TIMELINE
#
# Detection = 23:00
#
# Therefore:
#
# 00:00 → D+1 00:00
# 01:30 → D+1 01:30
# 03:00 → D+1 03:00
# ============================================================

detection = emergency["detected_at"]

tasks = []


for _, row in plan_df.iterrows():

    start_original = time_to_minutes(
        row["block_start"]
    )

    # Prefer block_duration from Model 2
    if "block_duration" in row:

        duration = int(row["block_duration"])

    elif "required_duration" in row:

        duration = int(row["required_duration"])

    else:

        end_original = time_to_minutes(
            row["block_end"]
        )

        duration = end_original - start_original

        if duration <= 0:
            duration += 1440


    # Move daily schedule into next occurrence
    if start_original < detection:

        start = start_original + 1440

    else:

        start = start_original


    end = start + duration


    tasks.append({

        "task_id": row["task_id"],
        "department": row["department"],
        "maintenance_type": row["maintenance_type"],
        "location_km": row["location_km"],
        "priority_score": float(row["priority_score"]),
        "section": row["section"],

        "start": start,
        "end": end,
        "duration": duration,

        "status": "UNCHANGED"
    })


# ============================================================
# TRAIN TIMETABLE
# ============================================================

train_schedule = [

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
    ("T020", "Section-B", "23:00", "23:15")
]


trains = []


for train_id, section, start, end in train_schedule:

    start_min = time_to_minutes(start)
    end_min = time_to_minutes(end)

    duration = end_min - start_min

    if start_min < detection:

        start_min += 1440

    end_min = start_min + duration


    trains.append({

        "train_id": train_id,
        "section": section,
        "start": start_min,
        "end": end_min
    })


# ============================================================
# TRAIN CONFLICT FUNCTION
# ============================================================

def train_conflicts(section, start, end):

    conflicts = []

    for train in trains:

        if train["section"] != section:
            continue

        overlap = (
            start < train["end"]
            and end > train["start"]
        )

        if overlap:

            conflicts.append(
                train["train_id"]
            )

    return conflicts


# ============================================================
# STEP 1
# FIND BEST EMERGENCY WINDOW
#
# Existing maintenance blocks are NOT hard constraints.
# Model 3 is allowed to displace them.
# ============================================================

best_emergency = None


for start in range(
    detection,
    detection + 1440,
    30
):

    end = (
        start
        + emergency["duration_minutes"]
    )


    if end > detection + 1440:
        continue


    conflicts = train_conflicts(
        emergency["section"],
        start,
        end
    )


    affected_trains = len(conflicts)


    disruption_cost = (
        affected_trains * 25
        + emergency["duration_minutes"] * 0.10
    )


    priority_benefit = (
        emergency["priority_score"] * 0.50
    )


    optimization_score = (
        priority_benefit
        - disruption_cost
    )


    candidate = {

        "start": start,
        "end": end,

        "affected_trains": affected_trains,

        "train_conflicts": conflicts,

        "optimization_score":
            optimization_score
    }


    if best_emergency is None:

        best_emergency = candidate

    else:

        current_key = (

            best_emergency["affected_trains"],

            -best_emergency[
                "optimization_score"
            ],

            best_emergency["start"]
        )


        new_key = (

            candidate["affected_trains"],

            -candidate[
                "optimization_score"
            ],

            candidate["start"]
        )


        if new_key < current_key:

            best_emergency = candidate


# ============================================================
# EMERGENCY DECISION
# ============================================================

emergency_start = best_emergency["start"]
emergency_end = best_emergency["end"]


print("\nAI EMERGENCY REPLANNING DECISION")
print("----------------------------------------")

print(
    f"Recommended Emergency Block : "
    f"{display_time(emergency_start)} - "
    f"{display_time(emergency_end)}"
)


print(
    f"Affected Trains             : "
    f"{best_emergency['affected_trains']}"
)


print(
    f"Conflicting Trains          : "
    f"{', '.join(best_emergency['train_conflicts']) 
    if best_emergency['train_conflicts']
    else 'None'}"
)


print(
    f"Operational Disruption Cost : "
    f"{best_emergency['affected_trains'] * 25 + 9:.1f}"
)


print(
    f"Optimization Score          : "
    f"{best_emergency['optimization_score']:.2f}"
)


# ============================================================
# STEP 2
# FIND TASKS THAT CONFLICT WITH EMERGENCY
# ============================================================

conflicting_tasks = []


for task in tasks:

    if task["section"] != emergency["section"]:
        continue


    overlap = (

        emergency_start < task["end"]

        and

        emergency_end > task["start"]
    )


    if overlap:

        conflicting_tasks.append(task)


# ============================================================
# DISPLAY CONFLICTS
# ============================================================

print("\nMAINTENANCE CONFLICT ANALYSIS")
print("----------------------------------------")


if len(conflicting_tasks) == 0:

    print(
        "No maintenance tasks conflict with "
        "the emergency block."
    )

else:

    for task in conflicting_tasks:

        print(
            f"{task['task_id']} | "
            f"Priority {task['priority_score']:.2f} | "
            f"{display_time(task['start'])} - "
            f"{display_time(task['end'])}"
        )


# ============================================================
# STEP 3
# EMERGENCY TASK
# ============================================================

emergency_task = {

    "task_id": emergency["task_id"],
    "department": emergency["department"],
    "maintenance_type": emergency["maintenance_type"],
    "location_km": emergency["location_km"],
    "priority_score": emergency["priority_score"],
    "section": emergency["section"],

    "start": emergency_start,
    "end": emergency_end,

    "duration": emergency["duration_minutes"],

    "status": "EMERGENCY-NEW"
}


# ============================================================
# STEP 4
# RESCHEDULE LOWER-PRIORITY CONFLICTING TASKS
#
# Important:
# Higher-priority tasks are preserved first.
# ============================================================

print("\nDYNAMIC RESCHEDULING")
print("----------------------------------------")


# Lowest priority first
conflicting_tasks.sort(
    key=lambda x: x["priority_score"]
)


# ------------------------------------------------------------
# Create final task list
# ------------------------------------------------------------

final_tasks = []


# Add emergency first
final_tasks.append(emergency_task)


# Add all non-conflicting normal tasks
for task in tasks:

    if task not in conflicting_tasks:

        final_tasks.append(task)


# ============================================================
# RESCHEDULE EACH CONFLICTING TASK
# ============================================================

for task in conflicting_tasks:

    duration = task["duration"]

    found_slot = None


    # --------------------------------------------------------
    # Search forward from emergency end
    # --------------------------------------------------------

    for candidate_start in range(

        emergency_end,

        detection + 1440,

        30
    ):

        candidate_end = (
            candidate_start
            + duration
        )


        if candidate_end > detection + 1440:

            continue


        # ----------------------------------------------------
        # Check against emergency
        # ----------------------------------------------------

        if (

            candidate_start < emergency_end

            and

            candidate_end > emergency_start
        ):

            continue


        # ----------------------------------------------------
        # Check against ALL other final tasks
        # ----------------------------------------------------

        slot_conflict = False


        for existing in final_tasks:

            if existing["section"] != task["section"]:

                continue


            if existing["task_id"] == task["task_id"]:

                continue


            overlap = (

                candidate_start
                < existing["end"]

                and

                candidate_end
                > existing["start"]
            )


            if overlap:

                slot_conflict = True

                break


        if slot_conflict:

            continue


        # ----------------------------------------------------
        # Check train conflicts
        # ----------------------------------------------------

        train_conflict_list = train_conflicts(

            task["section"],

            candidate_start,

            candidate_end
        )


        # ----------------------------------------------------
        # First feasible forward slot
        # ----------------------------------------------------

        found_slot = {

            "start": candidate_start,

            "end": candidate_end,

            "train_conflicts":
                train_conflict_list
        }

        break


    # ========================================================
    # APPLY RESCHEDULING
    # ========================================================

    if found_slot is not None:

        task["start"] = found_slot["start"]
        task["end"] = found_slot["end"]
        task["status"] = "RESCHEDULED"


        final_tasks.append(task)


        print(

            f"{task['task_id']} -> "
            f"{display_time(task['start'])} - "
            f"{display_time(task['end'])} "
            f"[RESCHEDULED]"
        )


        print(
            "   Reason: Moved because critical "
            "emergency maintenance has higher priority"
        )


    else:

        task["status"] = "MANUAL-REVIEW"

        final_tasks.append(task)


        print(

            f"{task['task_id']} -> "
            "MANUAL REVIEW REQUIRED"
        )


# ============================================================
# FINAL CONFLICT CHECK
# ============================================================

print("\nFINAL SCHEDULE CONFLICT CHECK")
print("----------------------------------------")


conflict_found = False


for i in range(len(final_tasks)):

    for j in range(i + 1, len(final_tasks)):

        task_a = final_tasks[i]
        task_b = final_tasks[j]


        if task_a["section"] != task_b["section"]:

            continue


        overlap = (

            task_a["start"]
            < task_b["end"]

            and

            task_b["start"]
            < task_a["end"]
        )


        if overlap:

            print(

                f"CONFLICT: "
                f"{task_a['task_id']} ↔ "
                f"{task_b['task_id']}"
            )

            conflict_found = True


if not conflict_found:

    print(
        "SUCCESS: Final schedule is conflict-free."
    )

else:

    print(
        "WARNING: Final schedule contains conflicts."
    )


# ============================================================
# CREATE OUTPUT
# ============================================================

output_rows = []


for task in final_tasks:

    output_rows.append({

        "task_id": task["task_id"],

        "department": task["department"],

        "maintenance_type":
            task["maintenance_type"],

        "location_km":
            task["location_km"],

        "priority_score":
            round(
                task["priority_score"],
                2
            ),

        "section":
            task["section"],

        "block_start":
            display_time(task["start"]),

        "block_end":
            display_time(task["end"]),

        "duration_minutes":
            task["duration"],

        "status":
            task["status"]
    })


final_df = pd.DataFrame(output_rows)


# ============================================================
# SORT BY ACTUAL TIME
# ============================================================

final_df["_sort"] = [

    task["start"]

    for task in final_tasks
]


final_df = (

    final_df

    .sort_values("_sort")

    .drop(columns=["_sort"])
)


# ============================================================
# DISPLAY FINAL PLAN
# ============================================================

print("\nFINAL DYNAMIC PLAN")
print("----------------------------------------")

print(
    final_df.to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

os.makedirs("outputs", exist_ok=True)


final_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nSaved final emergency plan to: "
    f"{OUTPUT_PATH}"
)

