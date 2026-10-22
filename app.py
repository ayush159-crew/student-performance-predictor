import streamlit as st
import pandas as pd
from joblib import load
from pathlib import Path

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
ROOT = Path(__file__).parent
bundle = load(ROOT / "student_model.joblib")
model = bundle["model"]
metrics = bundle["metrics"]

st.title("🎓 Student Performance Predictor")
st.write("A beginner-friendly machine learning website that estimates a student's final score from study habits and academic indicators.")
st.info("Educational demo only. The bundled dataset is synthetic sample data, not real student records. Predictions are estimates, not guarantees.")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Enter student details")
    study_hours = st.slider("Study hours per day", 0.5, 8.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 45, 100, 80)
    previous_score = st.slider("Previous exam score (%)", 25, 100, 65)
    sleep_hours = st.slider("Sleep hours per night", 4.0, 9.0, 7.0, 0.5)
    assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)
    if st.button("Predict final score", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance_percent": attendance,
            "previous_score": previous_score,
            "sleep_hours": sleep_hours,
            "assignments_completed": assignments
        }])
        score = float(model.predict(row)[0])
        score = max(0.0, min(100.0, score))
        st.session_state["prediction"] = score

with right:
    st.subheader("Prediction")
    if "prediction" in st.session_state:
        score = st.session_state["prediction"]
        st.metric("Estimated final score", f"{score:.1f}/100")
        st.progress(int(round(score)))
        if score >= 80:
            st.success("High estimated performance")
        elif score >= 60:
            st.info("Moderate estimated performance")
        elif score >= 40:
            st.warning("Needs improvement")
        else:
            st.error("At-risk estimate — consider additional academic support")
        st.caption("This estimate is produced by the trained Linear Regression model.")
    else:
        st.write("Adjust the inputs and click **Predict final score** to see an estimate.")
    st.divider()
    st.subheader("Model evaluation")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", str(metrics["mae"]))
    c2.metric("RMSE", str(metrics["rmse"]))
    c3.metric("R²", str(metrics["r2"]))
    st.caption(f"Trained on {metrics['train_rows']} rows; evaluated on {metrics['test_rows']} held-out rows. MAE is average absolute error in score points.")

st.divider()
st.subheader("Explore the sample dataset")
data = pd.read_csv(ROOT / "student_performance.csv")
st.write("The CSV contains generated example records. It is included so you can inspect the data used to train the model.")
st.dataframe(data.head(10), use_container_width=True)
st.download_button("Download sample dataset (CSV)", data.to_csv(index=False), file_name="student_performance.csv", mime="text/csv")
