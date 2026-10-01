import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

FILE_PATH = r"C:\Users\MARIYA\Downloads\indian_railway_predictive_maintenance_100k\indian_railway_predictive_maintenance_100k.csv"

df = pd.read_csv(FILE_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. CREATE AI MAINTENANCE PRIORITY SCORE
# ============================================================

print("\nCreating maintenance priority score...")


def normalize(series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(0, index=series.index)

    return (series - min_val) / (max_val - min_val)


# Fill missing values temporarily for score calculation
score_features = [
    "rail_wear_mm",
    "track_vibration_level",
    "wheel_wear_percent",
    "axle_temperature_c",
    "bearing_temperature_c",
    "traction_motor_temp_c",
    "delay_minutes",
    "last_maintenance_days",
    "load_factor_percent",
    "daily_trips",
    "inspection_score",
    "sensor_health_index"
]

for col in score_features:
    df[col] = df[col].fillna(df[col].median())


# Higher value = higher maintenance priority
rail_wear_score = normalize(df["rail_wear_mm"])
vibration_score = normalize(df["track_vibration_level"])
wheel_wear_score = normalize(df["wheel_wear_percent"])
axle_temp_score = normalize(df["axle_temperature_c"])
bearing_temp_score = normalize(df["bearing_temperature_c"])
motor_temp_score = normalize(df["traction_motor_temp_c"])
delay_score = normalize(df["delay_minutes"])
maintenance_age_score = normalize(df["last_maintenance_days"])
load_score = normalize(df["load_factor_percent"])
trips_score = normalize(df["daily_trips"])

# For inspection and sensor health:
# Lower health/inspection = higher priority
inspection_risk = 1 - normalize(df["inspection_score"])
sensor_risk = 1 - normalize(df["sensor_health_index"])


# Weighted priority score
df["priority_score"] = (
    rail_wear_score * 0.18 +
    vibration_score * 0.12 +
    wheel_wear_score * 0.10 +
    axle_temp_score * 0.08 +
    bearing_temp_score * 0.08 +
    motor_temp_score * 0.06 +
    delay_score * 0.10 +
    maintenance_age_score * 0.10 +
    load_score * 0.06 +
    trips_score * 0.04 +
    inspection_risk * 0.05 +
    sensor_risk * 0.03
) * 100


print("\nPriority score created.")

print(df["priority_score"].describe())


# ============================================================
# 3. REMOVE MISSING TARGET VALUES
# ============================================================

df = df.dropna(subset=["priority_score"]).copy()

print("\nRecords after removing missing priority scores:", len(df))


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

X = df.drop(
    columns=[
        "priority_score",
        "maintenance_required",
        "risk_score",
        "failure_type",
        "failure_severity",
        "train_id"
    ],
    errors="ignore"
)

y = df["priority_score"]


# ============================================================
# 5. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_columns
        ),
        (
            "cat",
            categorical_transformer,
            categorical_columns
        )
    ]
)


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 10. MODEL PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)


# Keep scores between 0 and 100
y_pred = np.clip(y_pred, 0, 100)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")


# ============================================================
# 12. SAMPLE PRIORITY PREDICTIONS
# ============================================================

print("\n" + "=" * 50)
print("SAMPLE PRIORITY PREDICTIONS")
print("=" * 50)


sample_results = pd.DataFrame({
    "Actual Priority": y_test.values[:10],
    "AI Priority": y_pred[:10]
})


sample_results["AI Priority"] = sample_results["AI Priority"].round(2)
sample_results["Actual Priority"] = sample_results["Actual Priority"].round(2)


print(sample_results.to_string(index=False))


# ============================================================
# 13. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/priority_model.pkl"

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print("Saved at:", MODEL_PATH)

print("\nPriority AI model is ready!")