"""Pure-Python helper functions for the Student Performance Predictor.

Nothing in this file imports Streamlit, so it can be tested on its own.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "study_hours",
    "attendance_percent",
    "previous_score",
    "sleep_hours",
    "assignments_completed",
]
TARGET = "final_score"

LABELS = {
    "study_hours": "Study hours / day",
    "attendance_percent": "Attendance (%)",
    "previous_score": "Previous score (%)",
    "sleep_hours": "Sleep hours / night",
    "assignments_completed": "Assignments (of 10)",
    "final_score": "Final score",
}

# Range seen in the training data; inputs are kept inside it.
BOUNDS = {
    "study_hours": (0.5, 8.0),
    "attendance_percent": (45, 100),
    "previous_score": (25, 100),
    "sleep_hours": (4.0, 9.0),
    "assignments_completed": (0, 10),
}

PRESETS = {
    "Custom": None,
    "Struggling student": {
        "study_hours": 1.0, "attendance_percent": 55, "previous_score": 40,
        "sleep_hours": 5.0, "assignments_completed": 2,
    },
    "Average student": {
        "study_hours": 3.5, "attendance_percent": 72, "previous_score": 62,
        "sleep_hours": 6.5, "assignments_completed": 5,
    },
    "High achiever": {
        "study_hours": 6.5, "attendance_percent": 95, "previous_score": 85,
        "sleep_hours": 7.5, "assignments_completed": 10,
    },
}


# --------------------------------------------------------------------------
# Single prediction helpers
# --------------------------------------------------------------------------
def to_row(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([{f: inputs[f] for f in FEATURES}])


def predict_score(model, inputs: dict) -> float:
    """Predict one student's final score, kept inside 0-100."""
    raw = float(model.predict(to_row(inputs))[0])
    return max(0.0, min(100.0, raw))


def grade_for(score: float) -> str:
    for cutoff, letter in [(80, "A"), (70, "B"), (60, "C"), (50, "D"), (40, "E")]:
        if score >= cutoff:
            return letter
    return "F"


def pass_probability(score: float, rmse: float, pass_mark: float) -> float:
    """P(actual score >= pass_mark), assuming errors are roughly normal
    with standard deviation equal to the model's RMSE."""
    rmse = max(rmse, 1e-6)
    z = (pass_mark - score) / rmse
    return 0.5 * math.erfc(z / math.sqrt(2))


def risk_level(prob: float) -> tuple[str, str]:
    """Return (label, emoji) from the probability of passing."""
    if prob >= 0.85:
        return "Low", "🟢"
    if prob >= 0.60:
        return "Medium", "🟡"
    return "High", "🔴"


def prediction_interval(score: float, rmse: float, z: float = 1.96) -> tuple[float, float]:
    return max(0.0, score - z * rmse), min(100.0, score + z * rmse)


def percentile_rank(score: float, series: pd.Series) -> float:
    return float((series < score).mean() * 100)


# --------------------------------------------------------------------------
# Advice / what-if
# --------------------------------------------------------------------------
# (feature, step, text) -- previous_score is excluded because it can't be changed.
_ACTIONS = [
    ("study_hours", 1.0, "Study 1 more hour per day"),
    ("attendance_percent", 10, "Raise attendance by 10 points"),
    ("assignments_completed", 2, "Complete 2 more assignments"),
    ("sleep_hours", 1.0, "Sleep 1 more hour (up to 8 h)"),
]


def improvement_suggestions(model, inputs: dict) -> pd.DataFrame:
    """Predicted score gain for realistic improvements, best first."""
    base = float(model.predict(to_row(inputs))[0])
    rows = []
    for feat, step, text in _ACTIONS:
        hi = 8.0 if feat == "sleep_hours" else BOUNDS[feat][1]
        new_val = min(hi, inputs[feat] + step)
        if new_val <= inputs[feat]:
            continue
        changed = dict(inputs, **{feat: new_val})
        gain = float(model.predict(to_row(changed))[0]) - base
        if gain > 0.05:
            rows.append({"action": text, "feature": feat, "gain": gain})
    out = pd.DataFrame(rows, columns=["action", "feature", "gain"])
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def apply_changes(inputs: dict, deltas: dict) -> dict:
    """Return a new input dict with deltas added and clipped to BOUNDS."""
    new = dict(inputs)
    for feat, d in deltas.items():
        lo, hi = BOUNDS[feat]
        new[feat] = min(hi, max(lo, inputs[feat] + d))
    return new


def sensitivity_curve(model, inputs: dict, feature: str, n: int = 40) -> pd.DataFrame:
    """Predicted score as one feature sweeps across its range."""
    lo, hi = BOUNDS[feature]
    if feature == "assignments_completed":
        xs = np.arange(int(lo), int(hi) + 1)
    else:
        xs = np.linspace(lo, hi, n)
    frame = pd.DataFrame([dict(inputs, **{feature: float(x)}) for x in xs])[FEATURES]
    preds = np.clip(model.predict(frame), 0, 100)
    return pd.DataFrame({"x": xs, "predicted": preds})


# --------------------------------------------------------------------------
# Model insight helpers
# --------------------------------------------------------------------------
def feature_importance(model) -> pd.DataFrame:
    """Standardised coefficients (linear) or impurity importances (trees)."""
    est = model.named_steps["regressor"] if hasattr(model, "named_steps") else model
    if hasattr(est, "coef_"):
        # The pipeline standardises inputs, so coefficients are comparable.
        values = np.asarray(est.coef_, dtype=float)
    else:
        values = np.asarray(est.feature_importances_, dtype=float)
    df = pd.DataFrame({"feature": FEATURES, "effect": values})
    df["share_pct"] = df["effect"].abs() / df["effect"].abs().sum() * 100
    df["label"] = df["feature"].map(LABELS)
    return df.sort_values("share_pct", ascending=False).reset_index(drop=True)


def holdout_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Re-create the train/test split used in train_model.py."""
    X, y = df[FEATURES], df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = np.clip(model.predict(X_test), 0, 100)
    return pd.DataFrame({"actual": y_test.values, "predicted": pred})


def _pipe(regressor) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Train several regressors on the same split and compare them."""
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, reg in candidates.items():
        pipe = _pipe(reg).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        cv_r2 = cross_val_score(_pipe(reg), X, y, cv=cv, scoring="r2").mean()
        rows.append({
            "Model": name,
            "MAE": round(float(mean_absolute_error(y_te, pred)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2),
            "R²": round(float(r2_score(y_te, pred)), 3),
            "5-fold CV R²": round(float(cv_r2), 3),
        })
    return pd.DataFrame(rows).sort_values("5-fold CV R²", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Batch prediction
# --------------------------------------------------------------------------
def batch_predict(model, df_in: pd.DataFrame, rmse: float, pass_mark: float):
    """Predict for every row of an uploaded table.

    Returns (result_df, info). Raises ValueError if columns are missing.
    """
    missing = [c for c in FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    X = df_in[FEATURES].apply(pd.to_numeric, errors="coerce")
    n_missing = int(X.isna().sum().sum())
    lows = pd.Series({f: BOUNDS[f][0] for f in FEATURES})
    highs = pd.Series({f: BOUNDS[f][1] for f in FEATURES})
    n_clipped = int(((X < lows) | (X > highs)).sum().sum())
    X = X.clip(lower=lows, upper=highs, axis=1)

    preds = np.clip(model.predict(X), 0, 100)
    out = df_in.copy()
    out["predicted_score"] = np.round(preds, 1)
    out["grade"] = [grade_for(p) for p in preds]
    probs = np.array([pass_probability(p, rmse, pass_mark) for p in preds])
    out["pass_probability_%"] = np.round(probs * 100, 1)
    out["risk_level"] = [risk_level(p)[0] for p in probs]
    info = {"rows": len(out), "missing_filled": n_missing, "values_clipped": n_clipped}
    return out, info


def csv_template() -> str:
    sample = pd.DataFrame([
        {"study_hours": 3.5, "attendance_percent": 80, "previous_score": 65, "sleep_hours": 7.0, "assignments_completed": 7},
        {"study_hours": 1.0, "attendance_percent": 55, "previous_score": 40, "sleep_hours": 5.0, "assignments_completed": 2},
        {"study_hours": 6.5, "attendance_percent": 95, "previous_score": 85, "sleep_hours": 7.5, "assignments_completed": 10},
    ])
    return sample.to_csv(index=False)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def build_report(name, inputs, score, grade, prob, risk, interval, suggestions, pass_mark, percentile) -> str:
    lines = [
        "STUDENT PERFORMANCE REPORT",
        "=" * 34,
        f"Student : {name or 'Not specified'}",
        f"Date    : {datetime.now():%d %b %Y, %H:%M}",
        "",
        "INPUTS",
        "-" * 34,
    ]
    for f in FEATURES:
        lines.append(f"{LABELS[f]:<24}: {inputs[f]}")
    lines += [
        "",
        "PREDICTION",
        "-" * 34,
        f"Estimated final score   : {score:.1f}/100 (grade {grade})",
        f"Likely range (95%)      : {interval[0]:.1f} - {interval[1]:.1f}",
        f"Pass mark               : {pass_mark}",
        f"Chance of passing       : {prob * 100:.0f}%",
        f"Risk level              : {risk}",
        f"Better than             : {percentile:.0f}% of students in the sample data",
        "",
        "HOW TO IMPROVE",
        "-" * 34,
    ]
    if len(suggestions):
        for _, r in suggestions.iterrows():
            lines.append(f"- {r['action']}: about +{r['gain']:.1f} points")
    else:
        lines.append("- Already near the top of the range. Keep up the habits!")
    lines += ["", "Note: educational demo using synthetic data. Estimates, not guarantees."]
    return "\n".join(lines)
