from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load

import helpers as h

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
ROOT = Path(__file__).parent


# --------------------------------------------------------------------------
# Cached loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return load(ROOT / "student_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "student_performance.csv")


@st.cache_data
def cached_model_comparison(df):
    return h.compare_models(df)


bundle = load_bundle()
model = bundle["model"]
metrics = bundle["metrics"]
data = load_data()
rmse = float(metrics["rmse"])

# --------------------------------------------------------------------------
# Sidebar: student profile
# --------------------------------------------------------------------------
DEFAULTS = h.PRESETS["Average student"]
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)
st.session_state.setdefault("history", [])


def apply_preset():
    preset = h.PRESETS[st.session_state["preset"]]
    if preset:
        for key, val in preset.items():
            st.session_state[key] = val


with st.sidebar:
    st.header("🧑‍🎓 Student profile")
    student_name = st.text_input("Student name (optional)", placeholder="e.g. Aman")
    st.selectbox("Quick preset", list(h.PRESETS), key="preset", on_change=apply_preset)

    st.slider("Study hours per day", 0.5, 8.0, step=0.5, key="study_hours")
    st.slider("Attendance (%)", 45, 100, key="attendance_percent")
    st.slider("Previous exam score (%)", 25, 100, key="previous_score")
    st.slider("Sleep hours per night", 4.0, 9.0, step=0.5, key="sleep_hours")
    st.slider("Assignments completed (out of 10)", 0, 10, key="assignments_completed")

    st.divider()
    pass_mark = st.slider("Pass mark", 30, 60, 40, help="Used for pass probability and risk level.")

inputs = {f: st.session_state[f] for f in h.FEATURES}

# --------------------------------------------------------------------------
# Core calculations (shared by all tabs)
# --------------------------------------------------------------------------
score = h.predict_score(model, inputs)
grade = h.grade_for(score)
prob = h.pass_probability(score, rmse, pass_mark)
risk_label, risk_icon = h.risk_level(prob)
low, high = h.prediction_interval(score, rmse)
percentile = h.percentile_rank(score, data["final_score"])
suggestions = h.improvement_suggestions(model, inputs)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.write(
    "Estimate a student's final score from study habits, then explore what-if scenarios, "
    "predict for a whole class, and see how the model works."
)
st.info(
    "Educational demo only. The bundled dataset is synthetic sample data, not real student "
    "records. Predictions are estimates, not guarantees."
)

tab_predict, tab_whatif, tab_batch, tab_data, tab_model, tab_track = st.tabs([
    "🎯 Predict", "🔬 What-If Lab", "📂 Batch Predict",
    "📊 Data Analytics", "🧠 Model Insights", "📈 Progress Tracker",
])

# ==========================================================================
# TAB 1: Predict
# ==========================================================================
with tab_predict:
    who = student_name.strip() or "this student"
    st.subheader(f"Prediction for {who}")

    g_col, d_col = st.columns([1, 1], gap="large")
    with g_col:
        bar_color = "#16a34a" if score >= 60 else "#f59e0b" if score >= pass_mark else "#dc2626"
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(score, 1),
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, pass_mark], "color": "#fde2e1"},
                    {"range": [pass_mark, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#dbeafe"},
                    {"range": [80, 100], "color": "#d1fae5"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": pass_mark},
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=30, r=30, t=30, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption(f"Black line = pass mark ({pass_mark}). Shaded bands: fail, borderline, good, excellent.")

    with d_col:
        m1, m2 = st.columns(2)
        m1.metric("Estimated final score", f"{score:.1f}/100")
        m2.metric("Grade", grade)
        m3, m4 = st.columns(2)
        m3.metric("Chance of passing", f"{prob * 100:.0f}%")
        m4.metric("Risk level", f"{risk_icon} {risk_label}")
        st.write(f"**Likely range (95%):** {low:.1f} – {high:.1f}")
        st.write(f"**Class standing:** better than **{percentile:.0f}%** of students in the sample data.")

        if score >= 80:
            st.success("High estimated performance")
        elif score >= 60:
            st.info("Moderate-to-good estimated performance")
        elif score >= pass_mark:
            st.warning("Borderline: passing, but there is room to improve")
        else:
            st.error("At-risk estimate: consider additional academic support")

    st.divider()
    st.subheader("How this student compares with the class average")
    cols = st.columns(len(h.FEATURES))
    for col, feat in zip(cols, h.FEATURES):
        avg = float(data[feat].mean())
        diff = inputs[feat] - avg
        col.metric(h.LABELS[feat], f"{inputs[feat]}", f"{diff:+.1f} vs avg {avg:.1f}")

    st.divider()
    st.subheader("💡 Personalised improvement tips")
    if len(suggestions):
        st.caption("Estimated gain in final score from each change, with everything else unchanged.")
        for _, row in suggestions.iterrows():
            st.markdown(f"- **{row['action']}** → about **+{row['gain']:.1f}** points")
        combined = h.apply_changes(inputs, {"study_hours": 1.0, "attendance_percent": 10, "assignments_completed": 2})
        combined_score = h.predict_score(model, combined)
        st.success(
            f"Doing the study, attendance and assignment tips together could lift the score "
            f"from **{score:.1f}** to about **{combined_score:.1f}**."
        )
    else:
        st.success("This student is already near the top of the range on every habit. Keep it up!")

    st.divider()
    a1, a2, _ = st.columns([1, 1, 2])
    if a1.button("💾 Save to Progress Tracker", type="primary"):
        st.session_state["history"].append({
            "time": datetime.now().strftime("%d %b %H:%M"),
            "student": student_name.strip() or "Student",
            **inputs,
            "predicted_score": round(score, 1),
        })
        st.toast("Saved to the Progress Tracker tab")

    report_text = h.build_report(
        student_name.strip(), inputs, score, grade, prob, f"{risk_label}",
        (low, high), suggestions, pass_mark, percentile,
    )
    a2.download_button("📄 Download report", report_text, file_name="student_report.txt", mime="text/plain")

# ==========================================================================
# TAB 2: What-If Lab
# ==========================================================================
with tab_whatif:
    st.subheader("What if the student changed their habits?")
    st.write("Move the sliders to change the current profile (set in the sidebar) and compare the outcomes.")

    w1, w2 = st.columns([1, 1.2], gap="large")
    with w1:
        d_study = st.slider("Change in study hours / day", -3.0, 3.0, 0.0, 0.5)
        d_att = st.slider("Change in attendance (points)", -30, 30, 0, 5)
        d_sleep = st.slider("Change in sleep hours", -3.0, 3.0, 0.0, 0.5)
        d_assign = st.slider("Change in assignments completed", -5, 5, 0, 1)

    scenario = h.apply_changes(inputs, {
        "study_hours": d_study,
        "attendance_percent": d_att,
        "sleep_hours": d_sleep,
        "assignments_completed": d_assign,
    })
    s_score = h.predict_score(model, scenario)
    s_prob = h.pass_probability(s_score, rmse, pass_mark)

    with w2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Current score", f"{score:.1f}")
        c2.metric("Scenario score", f"{s_score:.1f}", f"{s_score - score:+.1f}")
        c3.metric("Chance of passing", f"{s_prob * 100:.0f}%", f"{(s_prob - prob) * 100:+.0f} pts")

        compare = pd.DataFrame({
            "Habit": [h.LABELS[f] for f in h.FEATURES],
            "Current": [inputs[f] for f in h.FEATURES],
            "Scenario": [scenario[f] for f in h.FEATURES],
        })
        st.dataframe(compare, hide_index=True)
        st.caption("Values are kept within the range seen in the training data.")

    st.divider()
    st.subheader("Sensitivity: how does one habit affect the score?")
    feat_choice = st.selectbox(
        "Habit to vary", h.FEATURES, format_func=lambda f: h.LABELS[f], key="sens_feature"
    )
    curve = h.sensitivity_curve(model, inputs, feat_choice)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["x"], y=curve["predicted"], mode="lines", name="Predicted score"))
    fig.add_trace(go.Scatter(
        x=[inputs[feat_choice]], y=[score], mode="markers",
        marker=dict(size=13, color="#dc2626"), name="Current student",
    ))
    fig.add_hline(y=pass_mark, line_dash="dash", annotation_text="Pass mark")
    fig.update_layout(
        xaxis_title=h.LABELS[feat_choice], yaxis_title="Predicted final score",
        height=380, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 3: Batch Predict
# ==========================================================================
with tab_batch:
    st.subheader("Predict for a whole class")
    st.write(
        "Upload a CSV with these columns: "
        + ", ".join(f"`{f}`" for f in h.FEATURES)
        + ". Extra columns (like name or roll number) are kept in the output."
    )
    st.download_button("⬇️ Download CSV template", h.csv_template(), file_name="batch_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    use_sample = st.checkbox("Or try it with 25 students from the sample dataset")

    batch_df = None
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:  # unreadable file
            st.error(f"Could not read that file as CSV: {exc}")
    elif use_sample:
        batch_df = data.sample(25, random_state=1).drop(columns=["final_score"]).reset_index(drop=True)

    if batch_df is not None:
        try:
            result, info = h.batch_predict(model, batch_df, rmse, pass_mark)
        except ValueError as exc:
            st.error(str(exc))
        else:
            if info["missing_filled"]:
                st.warning(f"{info['missing_filled']} missing/non-numeric value(s) were filled with the training median.")
            if info["values_clipped"]:
                st.warning(f"{info['values_clipped']} value(s) outside the training range were clipped.")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Students", info["rows"])
            k2.metric("Average predicted score", f"{result['predicted_score'].mean():.1f}")
            k3.metric("High-risk students", int((result["risk_level"] == "High").sum()))
            k4.metric("Expected to pass", f"{(result['predicted_score'] >= pass_mark).mean() * 100:.0f}%")

            order = ["Low", "Medium", "High"]
            counts = result["risk_level"].value_counts().reindex(order, fill_value=0)
            risk_fig = go.Figure(go.Bar(
                x=order, y=counts.values, marker_color=["#16a34a", "#f59e0b", "#dc2626"],
            ))
            risk_fig.update_layout(
                title="Students by risk level", height=300,
                margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Students",
            )
            st.plotly_chart(risk_fig, use_container_width=True)

            st.dataframe(result)
            st.download_button(
                "⬇️ Download predictions (CSV)", result.to_csv(index=False),
                file_name="batch_predictions.csv", mime="text/csv",
            )

# ==========================================================================
# TAB 4: Data Analytics
# ==========================================================================
with tab_data:
    st.subheader("Explore the sample dataset")
    st.caption("500 generated example records used to train the model.")

    st.dataframe(data.head(10))
    st.download_button(
        "Download sample dataset (CSV)", data.to_csv(index=False),
        file_name="student_performance.csv", mime="text/csv",
    )

    with st.expander("Summary statistics"):
        st.dataframe(data.describe().T.round(2))

    r1, r2 = st.columns(2, gap="large")
    with r1:
        hist = go.Figure(go.Histogram(x=data["final_score"], nbinsx=25, marker_color="#6366f1"))
        hist.add_vline(x=score, line_dash="dash", line_color="#dc2626", annotation_text="This student")
        hist.update_layout(
            title="Distribution of final scores", xaxis_title="Final score", yaxis_title="Students",
            height=360, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(hist, use_container_width=True)
    with r2:
        grades = data["final_score"].apply(h.grade_for).value_counts().reindex(list("ABCDEF"), fill_value=0)
        gfig = go.Figure(go.Bar(x=grades.index, y=grades.values, marker_color="#0ea5e9"))
        gfig.update_layout(
            title="Grade distribution", xaxis_title="Grade", yaxis_title="Students",
            height=360, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(gfig, use_container_width=True)

    corr = data[h.FEATURES + [h.TARGET]].corr()
    names = [h.LABELS[c] for c in corr.columns]
    heat = go.Figure(go.Heatmap(
        z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu",
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    heat.update_layout(title="Correlation heatmap", height=460, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(heat, use_container_width=True)

    st.markdown("**Relationship with final score**")
    x_feat = st.selectbox("Habit", h.FEATURES, format_func=lambda f: h.LABELS[f], key="scatter_feature")
    slope, intercept = np.polyfit(data[x_feat], data[h.TARGET], 1)
    xs = np.linspace(data[x_feat].min(), data[x_feat].max(), 50)
    sc = go.Figure()
    sc.add_trace(go.Scatter(
        x=data[x_feat], y=data[h.TARGET], mode="markers", name="Students",
        marker=dict(size=6, opacity=0.55, color="#6366f1"),
    ))
    sc.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Trend", line=dict(color="#f59e0b", width=3)))
    sc.add_trace(go.Scatter(
        x=[inputs[x_feat]], y=[score], mode="markers", name="This student",
        marker=dict(size=14, color="#dc2626", symbol="star"),
    ))
    sc.update_layout(
        xaxis_title=h.LABELS[x_feat], yaxis_title="Final score",
        height=400, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(sc, use_container_width=True)

# ==========================================================================
# TAB 5: Model Insights
# ==========================================================================
with tab_model:
    st.subheader("Model evaluation")
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE", str(metrics["mae"]), help="Average absolute error in score points.")
    e2.metric("RMSE", str(metrics["rmse"]), help="Like MAE but punishes big mistakes more.")
    e3.metric("R²", str(metrics["r2"]), help="Share of score variation the model explains (1.0 = perfect).")
    st.caption(
        f"Trained on {metrics['train_rows']} rows; evaluated on {metrics['test_rows']} held-out rows. "
        "Pass probability and the 95% range use RMSE as the typical prediction error."
    )

    st.divider()
    i1, i2 = st.columns(2, gap="large")
    imp = h.feature_importance(model)
    with i1:
        st.markdown("**What drives the score?**")
        imp_plot = imp.sort_values("share_pct")
        ifig = go.Figure(go.Bar(
            x=imp_plot["share_pct"], y=imp_plot["label"], orientation="h", marker_color="#6366f1",
            text=imp_plot["share_pct"].round(1).astype(str) + "%", textposition="outside",
        ))
        ifig.update_layout(
            xaxis_title="Share of model influence (%)", height=340,
            margin=dict(l=10, r=40, t=10, b=10),
        )
        st.plotly_chart(ifig, use_container_width=True)
        top = imp.iloc[0]
        st.caption(f"**{top['label']}** has the biggest influence on the predicted score.")

    with i2:
        st.markdown("**Actual vs predicted (held-out test data)**")
        hold = h.holdout_predictions(model, data)
        lo_v = float(min(hold.min()))
        hi_v = float(max(hold.max()))
        afig = go.Figure()
        afig.add_trace(go.Scatter(
            x=hold["actual"], y=hold["predicted"], mode="markers", name="Test students",
            marker=dict(size=7, opacity=0.7, color="#0ea5e9"),
        ))
        afig.add_trace(go.Scatter(
            x=[lo_v, hi_v], y=[lo_v, hi_v], mode="lines", name="Perfect prediction",
            line=dict(dash="dash", color="#6b7280"),
        ))
        afig.update_layout(
            xaxis_title="Actual final score", yaxis_title="Predicted final score",
            height=340, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(afig, use_container_width=True)
        st.caption("Points close to the dashed line are accurate predictions.")

    st.markdown("**Prediction errors (residuals)**")
    resid = hold["actual"] - hold["predicted"]
    rfig = go.Figure(go.Histogram(x=resid, nbinsx=20, marker_color="#f59e0b"))
    rfig.update_layout(
        xaxis_title="Actual − predicted (score points)", yaxis_title="Students",
        height=300, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(rfig, use_container_width=True)

    st.divider()
    st.subheader("Model comparison")
    st.write("How the current Linear Regression model stacks up against other popular algorithms on the same data.")
    if st.button("Run model comparison"):
        with st.spinner("Training models..."):
            st.session_state["model_cmp"] = cached_model_comparison(data)
    if "model_cmp" in st.session_state:
        cmp_df = st.session_state["model_cmp"]
        st.dataframe(cmp_df, hide_index=True)
        best = cmp_df.iloc[0]
        st.caption(
            f"Best 5-fold cross-validated R²: **{best['Model']}** ({best['5-fold CV R²']}). "
            "On small, mostly linear data, simple models often beat complex ones."
        )

# ==========================================================================
# TAB 6: Progress Tracker
# ==========================================================================
with tab_track:
    st.subheader("Progress tracker")
    st.write(
        "Save predictions from the **Predict** tab over time (for example, after each exam) "
        "to see whether a student's estimated score is trending up or down."
    )
    history = pd.DataFrame(st.session_state["history"])
    if history.empty:
        st.info("Nothing saved yet. Go to the Predict tab and click **Save to Progress Tracker**.")
    else:
        students = sorted(history["student"].unique())
        pick = st.selectbox("Student", ["All students"] + students)
        view = history if pick == "All students" else history[history["student"] == pick]
        view = view.reset_index(drop=True)
        view.index = view.index + 1

        if len(view) >= 2:
            change = view["predicted_score"].iloc[-1] - view["predicted_score"].iloc[0]
            t1, t2, t3 = st.columns(3)
            t1.metric("Entries", len(view))
            t2.metric("Latest score", f"{view['predicted_score'].iloc[-1]:.1f}")
            t3.metric("Change since first", f"{change:+.1f}")

            tfig = go.Figure()
            for name, grp in view.groupby("student", sort=False):
                tfig.add_trace(go.Scatter(
                    x=list(range(1, len(grp) + 1)), y=grp["predicted_score"],
                    mode="lines+markers", name=name,
                ))
            tfig.add_hline(y=pass_mark, line_dash="dash", annotation_text="Pass mark")
            tfig.update_layout(
                xaxis_title="Entry number", yaxis_title="Predicted score",
                height=360, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(tfig, use_container_width=True)
        else:
            st.caption("Save at least two entries to see a trend chart.")

        st.dataframe(view)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("⬇️ Download history", history.to_csv(index=False), file_name="progress_history.csv", mime="text/csv")
        if d2.button("🗑️ Clear history"):
            st.session_state["history"] = []
            st.rerun()

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
**How it works:** a Linear Regression model (with median imputation and feature scaling) was trained
on 500 synthetic student records using five inputs: study hours, attendance, previous score, sleep
and assignments completed.

**Pass probability** assumes prediction errors are roughly bell-shaped with a spread equal to the
model's RMSE. **Risk level:** Low ≥ 85% chance of passing, Medium ≥ 60%, otherwise High.

**Limitations:** the data is synthetic, so real-world results will differ. The model only sees five
factors and shows correlation, not proof of cause and effect.
"""
    )
