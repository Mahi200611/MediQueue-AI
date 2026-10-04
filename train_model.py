import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/healthcare_appointments.csv")

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "Day",
    "Department",
    "Patient_Type",
    "Appointment_Hour",
    "Appointment_Minute",
    "Existing_Appointments",
    "Previous_Visits",
    "Emergency"
]

X = df[features]

# Target 1
y_queue = df["Queue_Number"]

# Target 2
y_wait = df["Waiting_Time_Minutes"]


# ==========================================
# 3. CATEGORICAL AND NUMERICAL FEATURES
# ==========================================

categorical_features = [
    "Day",
    "Department",
    "Patient_Type",
    "Emergency"
]

numerical_features = [
    "Appointment_Hour",
    "Appointment_Minute",
    "Existing_Appointments",
    "Previous_Visits"
]


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 5. QUEUE NUMBER MODEL
# ==========================================

queue_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                max_depth=15
            )
        )
    ]
)


# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_queue,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 7. TRAIN QUEUE MODEL
# ==========================================

print("\nTraining Queue Prediction Model...")

queue_model.fit(X_train, y_train)


# ==========================================
# 8. EVALUATE QUEUE MODEL
# ==========================================

queue_predictions = queue_model.predict(X_test)

queue_mae = mean_absolute_error(
    y_test,
    queue_predictions
)

queue_r2 = r2_score(
    y_test,
    queue_predictions
)

print("\nQueue Prediction Results")
print("------------------------")
print("MAE:", round(queue_mae, 2))
print("R2 Score:", round(queue_r2, 3))


# ==========================================
# 9. TRAIN WAITING TIME MODEL
# ==========================================

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X,
    y_wait,
    test_size=0.2,
    random_state=42
)


wait_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                max_depth=15
            )
        )
    ]
)


print("\nTraining Waiting Time Model...")

wait_model.fit(X_train_w, y_train_w)


# ==========================================
# 10. EVALUATE WAITING TIME MODEL
# ==========================================

wait_predictions = wait_model.predict(X_test_w)

wait_mae = mean_absolute_error(
    y_test_w,
    wait_predictions
)

wait_r2 = r2_score(
    y_test_w,
    wait_predictions
)

print("\nWaiting Time Prediction Results")
print("-------------------------------")
print("MAE:", round(wait_mae, 2))
print("R2 Score:", round(wait_r2, 3))


# ==========================================
# 11. SAVE MODELS
# ==========================================

joblib.dump(
    queue_model,
    "models/queue_model.pkl"
)

joblib.dump(
    wait_model,
    "models/wait_time_model.pkl"
)


# ==========================================
# 12. FINAL MESSAGE
# ==========================================

print("\n====================================")
print("       MEDIQUEUE AI TRAINING")
print("====================================")

print("Queue model saved successfully!")
print("Waiting time model saved successfully!")

print("\nModels:")
print("models/queue_model.pkl")
print("models/wait_time_model.pkl")

print("\nTraining completed successfully!")