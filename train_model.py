"""Retrain the model using student_performance.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "student_performance.csv")
features = ["study_hours", "attendance_percent", "previous_score", "sleep_hours", "assignments_completed"]
X, y = df[features], df["final_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "mae": round(float(mean_absolute_error(y_test, pred)), 2),
    "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
    "r2": round(float(r2_score(y_test, pred)), 3),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test))
}
dump({"model": model, "features": features, "metrics": metrics}, ROOT / "student_model.joblib")
print("Model retrained.")
print(metrics)
