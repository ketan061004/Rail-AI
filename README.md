# 🚆 RailAI Operations Control

### AI-Powered Automatic Block Planning & Emergency Replanning for Indian Railways

> **An intelligent railway operations decision-support system that prioritizes maintenance, identifies optimal maintenance blocks, and dynamically replans schedules during emergencies.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![Scikit--Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75?logo=plotly)](https://plotly.com/python/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## 📌 Overview

**RailAI Operations Control** is an AI-assisted railway maintenance planning and operations decision-support platform designed around the challenges of coordinating maintenance activities with train operations.

The system follows a three-stage intelligence pipeline:

```text
Railway Maintenance & Condition Data
                │
                ▼
      ┌─────────────────────┐
      │ Model 1             │
      │ Maintenance Priority│
      │ Engine              │
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │ Model 2             │
      │ AI Block Planner    │
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │ Model 3             │
      │ Emergency Dynamic   │
      │ Replanner           │
      └──────────┬──────────┘
                 │
                 ▼
       Conflict-Aware Schedule
```

The project also provides an interactive **Streamlit Operations Control dashboard** where users can assess railway conditions, generate maintenance blocks, introduce emergency scenarios, and visualize schedule changes.

---

# 🎯 Problem Statement

Railway maintenance activities involve multiple departments such as:

* 🛤️ Engineering
* 🚦 Signal & Telecommunication (S&T)
* ⚡ Traction / OHE
* 🌉 Bridge & Works

Maintenance activities need access to railway sections while train services continue to operate.

A poorly coordinated maintenance block can result in:

* Train conflicts
* Operational disruption
* Inefficient utilization of maintenance windows
* Delayed maintenance activities
* Difficult emergency rescheduling

RailAI addresses this problem by combining **AI-based maintenance prioritization with timetable-aware block planning and emergency replanning**.

---

# 💡 Solution

RailAI transforms railway maintenance planning into a sequential AI-assisted decision process.

### 1️⃣ Maintenance Priority

The system analyzes railway and train-condition parameters and generates a priority score from **0–100**.

Higher priority indicates a greater need for maintenance attention.

The priority engine considers factors including:

* Rail wear
* Track vibration
* Wheel wear
* Axle temperature
* Bearing temperature
* Traction motor temperature
* Train delays
* Days since last maintenance
* Load factor
* Daily trips
* Inspection score
* Sensor health

The training pipeline creates a weighted maintenance-priority target and trains a **Random Forest Regressor** to predict it.

---

### 2️⃣ Optimal Block Planning

After maintenance tasks receive priority scores, the block planner searches possible maintenance windows across a 24-hour timetable.

The planner considers:

* Maintenance duration
* Railway section
* Train timetable
* Number of affected trains
* Maintenance conflicts
* Operational disruption
* Maintenance priority

Candidate windows are evaluated and the system selects suitable blocks while attempting to minimize train disruption.

---

### 3️⃣ Emergency Dynamic Replanning

Railway operations can change unexpectedly due to critical defects or infrastructure failures.

RailAI can introduce an emergency maintenance requirement and dynamically modify the existing schedule.

The emergency replanner:

1. Detects an emergency requirement.
2. Determines a feasible emergency block.
3. Identifies conflicting maintenance activities.
4. Protects higher-priority work.
5. Reschedules lower-priority conflicting tasks.
6. Checks the resulting schedule for conflicts.
7. Generates the final dynamic schedule.

If a suitable slot cannot be found, the task can be marked for **manual review**.

---

# 🧠 AI Pipeline

## Model 1 — Maintenance Priority Engine

```text
Railway Condition Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Weighted Priority Score
        │
        ▼
Random Forest Regressor
        │
        ▼
Priority Score: 0–100
```

The model uses a preprocessing pipeline containing:

* Median imputation for numerical data
* Most-frequent imputation for categorical data
* One-hot encoding
* Random Forest regression

The current implementation uses **150 trees**, maximum depth of **18**, and a fixed random state for reproducibility.

---

# ⚙️ Priority Scoring

The prototype generates the maintenance-priority target using weighted operational and condition indicators.

| Feature                    |   Weight |
| -------------------------- | -------: |
| Rail wear                  |      18% |
| Track vibration            |      12% |
| Wheel wear                 |      10% |
| Axle temperature           |       8% |
| Bearing temperature        |       8% |
| Traction motor temperature |       6% |
| Delay                      |      10% |
| Days since maintenance     |      10% |
| Load factor                |       6% |
| Daily trips                |       4% |
| Inspection risk            |       5% |
| Sensor health risk         |       3% |
| **Total**                  | **100%** |

These weights are implemented in the project's priority-score generation pipeline.

---

# 🚦 Block Planning Logic

The block planner creates candidate maintenance windows in 30-minute increments.

For each candidate block, it evaluates:

```text
                 Candidate Block
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 Maintenance Conflict?        Train Conflict?
          │                         │
          └────────────┬────────────┘
                       ▼
               Disruption Cost
                       │
                       ▼
               Priority Benefit
                       │
                       ▼
             Optimization Score
                       │
                       ▼
             Recommended Block
```

The prototype uses train-conflict count and block duration as components of operational disruption, while maintenance priority contributes a benefit to the optimization score.

---

# 🚨 Emergency Replanning

When a critical maintenance requirement appears, RailAI evaluates the existing schedule and searches for a new feasible arrangement.

Example flow:

```text
Existing Schedule
       │
       ▼
Emergency Detected
       │
       ▼
Find Emergency Window
       │
       ▼
Identify Conflicting Tasks
       │
       ▼
Compare Priorities
       │
       ├───────────────┐
       ▼               ▼
Higher Priority     Lower Priority
Protected           Reschedule
       │               │
       └───────┬───────┘
               ▼
       Conflict Check
               │
               ▼
      Final Dynamic Schedule
```

The current emergency scenario uses a critical track repair with a high priority and searches forward in 30-minute increments for feasible rescheduling slots.

---

# 🖥️ Dashboard

The Streamlit dashboard provides three major sections.

### Step 01 — Maintenance Assessment

Users can enter railway and train-condition information such as:

* Railway region
* Season
* Train type
* Train age
* Average speed
* Rail wear
* Track vibration
* Environmental conditions
* Wheel wear
* Axle temperature
* Bearing temperature
* Brake pressure
* Traction motor temperature
* Signal status
* Load factor
* Delays
* Inspection score
* Sensor health

The dashboard then generates:

```text
Maintenance Priority
        │
        ├── Score /100
        ├── Priority Level
        └── AI Engine Used
```

The dashboard also provides fallback scoring when the trained model is unavailable.

---

### Step 02 — Optimal Block Planning

Users can specify:

* Maintenance type
* Railway section
* Location
* Required duration
* Preferred start time
* Traffic level
* Maintenance priority
* Night-block availability

The dashboard displays:

* Recommended block
* Priority
* Number of affected trains
* Optimization score
* Top candidate windows
* 24-hour maintenance timeline

---

### Step 03 — Emergency Replanning

Users can simulate:

* Critical Track Defect
* Signal Failure
* OHE Failure
* Bridge Issue

The dashboard then displays:

* Emergency block
* Emergency priority
* Number of rescheduled tasks
* Affected section
* Before/after schedule
* Updated maintenance schedule

---

# 📂 Project Structure

```text
railai-operations-control/
│
├── app.py
│   └── Streamlit Operations Control Dashboard
│
├── train_priority_model.py
│   └── Model 1: Maintenance Priority Engine
│
├── block_planner.py
│   └── Model 2: AI Block Planner
│
├── emergency_replanner.py
│   └── Model 3: Emergency Dynamic Replanner
│
├── run_pipeline.py
│   └── End-to-end AI pipeline runner
│
├── requirements.txt
│   └── Python dependencies
│
├── models/
│   └── priority_model.pkl
│
└── outputs/
    ├── ai_block_plan_24hr.csv
    └── ai_emergency_replanned_schedule.csv
```

The repository currently contains these core Python modules and dependency file.

---

# 🛠️ Tech Stack

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| 🐍 Python        | Core development          |
| 🧠 Scikit-learn  | Machine learning          |
| 🌲 Random Forest | Priority prediction       |
| 🐼 Pandas        | Data processing           |
| 🔢 NumPy         | Numerical computation     |
| 📊 Plotly        | Interactive visualization |
| 🎛️ Streamlit    | Operations dashboard      |
| 💾 Joblib        | Model serialization       |
| 📄 CSV           | Pipeline data exchange    |

The project's current dependency list includes Streamlit, Pandas, NumPy, Plotly, Scikit-learn, and Joblib.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Mariya-saifee/railai-operations-control.git
cd railai-operations-control
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the URL displayed by Streamlit, typically:

```text
http://localhost:8501
```

The dashboard loads the trained priority model from:

```text
models/priority_model.pkl
```

If the model is unavailable, the application contains a prototype fallback priority engine.

---

# 🤖 Train the AI Model

The priority model is trained using:

```bash
python train_priority_model.py
```

The training script:

1. Loads the railway maintenance dataset.
2. Handles missing values.
3. Generates maintenance-priority scores.
4. Separates numerical and categorical features.
5. Builds preprocessing pipelines.
6. Trains the Random Forest model.
7. Evaluates predictions using MAE and R².
8. Saves the trained model.

The resulting model is saved as:

```text
models/priority_model.pkl
```

> **Dataset note:** The current training script contains a local dataset path. Before running it on another machine, update `FILE_PATH` in `train_priority_model.py` to point to your dataset.

---

# 🔄 Run the Complete AI Pipeline

The project includes a master pipeline:

```bash
python run_pipeline.py
```

The pipeline executes:

```text
Model 1
   ↓
Maintenance Priority

Model 2
   ↓
24-Hour Block Planning

Model 3
   ↓
Emergency Dynamic Replanning
```

The pipeline checks that the required scripts exist and stops if an individual module fails.

---

# 📤 Generated Outputs

After successful execution, the pipeline generates:

### Trained model

```text
models/priority_model.pkl
```

### AI block plan

```text
outputs/ai_block_plan_24hr.csv
```

### Emergency-replanned schedule

```text
outputs/ai_emergency_replanned_schedule.csv
```

These outputs form the data flow between the three AI stages.

---

# 🔗 End-to-End Data Flow

```text
                  ┌───────────────────────┐
                  │ Railway Condition    │
                  │ & Maintenance Data   │
                  └───────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ MODEL 1                │
                 │ Maintenance Priority   │
                 │ Random Forest          │
                 └────────────┬───────────┘
                              │
                       Priority Score
                              │
                              ▼
                 ┌────────────────────────┐
                 │ MODEL 2                │
                 │ AI Block Planner       │
                 │                        │
                 │ Train Conflict         │
                 │ + Maintenance Conflict │
                 │ + Disruption Cost      │
                 └────────────┬───────────┘
                              │
                       24-Hour Block Plan
                              │
                              ▼
                 ┌────────────────────────┐
                 │ MODEL 3                │
                 │ Emergency Replanner    │
                 │                        │
                 │ Emergency Detection    │
                 │ Conflict Analysis      │
                 │ Dynamic Rescheduling   │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ FINAL SCHEDULE         │
                 │                        │
                 │ Conflict-Aware         │
                 │ Maintenance Plan       │
                 └────────────────────────┘
```

---

# 📊 Example Use Case

Consider four maintenance tasks:

| Task   | Department  | Maintenance      | Section   |
| ------ | ----------- | ---------------- | --------- |
| MT-001 | Engineering | Track Repair     | Section-A |
| MT-002 | S&T         | Signal Repair    | Section-A |
| MT-003 | Traction    | OHE Maintenance  | Section-B |
| MT-004 | Engineering | Track Inspection | Section-B |

The system:

**Step 1:** Calculates a priority score for each task.

**Step 2:** Searches available maintenance windows.

**Step 3:** Checks train conflicts.

**Step 4:** Generates the 24-hour maintenance plan.

**Step 5:** Introduces an emergency defect.

**Step 6:** Identifies conflicting work.

**Step 7:** Protects the emergency maintenance requirement and moves lower-priority work where feasible.

**Step 8:** Produces the updated schedule.

---

# 🧪 Prototype vs Production

This repository is a **prototype / decision-support system**.

It currently uses simulated/sample railway schedules and maintenance scenarios rather than live railway operational systems.

For production deployment, the system would require integration with authoritative railway data sources, real-time train movement information, infrastructure monitoring systems, operational rules, safety procedures, and appropriate authorization workflows.

### Important

> ⚠️ **RailAI recommendations are advisory. They are not intended to directly control railway infrastructure or replace decisions made by authorized railway operating and maintenance personnel.**

The dashboard itself explicitly describes the system as a prototype decision-support system and keeps final block authorization with designated railway personnel.

---

# 🔮 Future Scope

Potential future improvements include:

* 🔴 Real-time train tracking
* 📡 IoT / sensor integration
* 🛰️ Live railway telemetry
* 🗺️ GIS-based railway visualization
* 🤖 Advanced optimization algorithms
* 🧠 Deep-learning failure prediction
* 📈 Predictive maintenance forecasting
* 🚦 Real-time traffic-aware block planning
* 🔄 Continuous schedule optimization
* 🏢 Multi-department coordination
* 📱 Mobile operations interface
* 🔐 Role-based access control
* 📜 Audit logs and explainable AI
* ☁️ Cloud deployment
* 🔌 Integration with railway operational systems
* 🧪 Automated simulation and stress testing

---

# ⚠️ Limitations

The current implementation has several prototype-level limitations:

* Train schedules are predefined in the code.
* Maintenance tasks are currently represented using prototype/sample data.
* Emergency scenarios are simulated.
* The training script uses a machine-specific dataset path that must be changed for other environments.
* The optimization logic is heuristic rather than a full railway-grade constraint solver.
* No live railway infrastructure control is implemented.
* AI outputs require human review before operational use.

---

# 🤝 Contributing

Contributions are welcome.

### 1. Fork the repository

```bash
git fork
```

### 2. Create a feature branch

```bash
git checkout -b feature/your-feature
```

### 3. Make your changes

```bash
git add .
git commit -m "Add your feature"
```

### 4. Push the branch

```bash
git push origin feature/your-feature
```

### 5. Open a Pull Request

Please keep contributions focused and document any changes to the AI pipeline or scheduling logic.

---

# 📜 License

This project is intended as an educational and prototype implementation.

If this project is to be distributed as open source, add an appropriate license file such as:

```text
LICENSE
```

and update this section accordingly.

---

# 👩‍💻 Project

**RailAI Operations Control**

AI-Powered Automatic Block Planning & Emergency Replanning for Indian Railways

**Repository:**
https://github.com/Mariya-saifee/railai-operations-control

---

## ⭐ Key Idea

> **Prioritize smarter. Plan better. Respond faster.**

RailAI combines **machine learning, railway maintenance intelligence, timetable-aware scheduling, and dynamic emergency replanning** into a unified decision-support workflow.

```text
ASSESS
   ↓
PRIORITIZE
   ↓
PLAN
   ↓
MONITOR
   ↓
REPLAN
   ↓
RESPOND
```

**RailAI — Intelligent Operations for Safer, Smarter Railway Maintenance.**
