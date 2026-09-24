
import io
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = "models/random_forest_model.pkl"
KMEANS_PATH = "models/kmeans_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/student_performance.csv"

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]

model = joblib.load(MODEL_PATH)
kmeans = joblib.load(KMEANS_PATH)
scaler = joblib.load(SCALER_PATH)
data = pd.read_csv(DATA_PATH)

st.title("🎓 Student Performance Analytics & Prediction")
st.caption("K-Means Clustering + Random Forest")

with st.sidebar:
    st.header("Student Information")
    student_id = st.text_input("Student ID", "S101")
    student_name = st.text_input("Student Name", "Student")
    st.divider()
    st.info("Enter the student's academic information, predict performance, and download a progress report.")

st.subheader("Enter Student Details")

c1, c2, c3 = st.columns(3)
with c1:
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    study_hours = st.slider("Study Hours / Day", 0.0, 10.0, 3.0, 0.5)
with c2:
    internal = st.slider("Internal Mark", 0, 100, 65)
    assignment = st.slider("Assignment Score", 0, 100, 70)
with c3:
    previous = st.slider("Previous Mark", 0, 100, 65)

student = pd.DataFrame([{
    "Attendance": attendance,
    "Study_Hours": study_hours,
    "Internal_Mark": internal,
    "Assignment": assignment,
    "Previous_Mark": previous
}])

def make_recommendations(row, prediction):
    recs = []
    if row["Attendance"] < 75:
        recs.append("Improve attendance and maintain regular class participation.")
    if row["Study_Hours"] < 3:
        recs.append("Increase focused study time gradually and maintain a daily study routine.")
    if row["Internal_Mark"] < 50:
        recs.append("Focus on internal-test preparation and clarify difficult topics.")
    if row["Assignment"] < 60:
        recs.append("Complete assignments on time and review feedback.")
    if row["Previous_Mark"] < 50:
        recs.append("Revise previous topics and strengthen basic concepts.")
    if not recs:
        recs.append("Maintain the current study routine and continue regular revision.")
    if prediction == "Low":
        recs.append("Consider discussing the result with a teacher/mentor and creating a short improvement plan.")
    elif prediction == "Medium":
        recs.append("Focus on the lowest-scoring areas to move toward a higher performance level.")
    else:
        recs.append("Maintain consistent performance and continue challenging yourself.")
    return recs

def build_progress_html(name, sid, row, prediction, cluster, confidence, recommendations):
    esc = lambda x: str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    rec_html = ''.join(f'<li>{esc(r)}</li>' for r in recommendations)
    labels = {
        'Attendance': f'{row["Attendance"]:.1f}%',
        'Study_Hours': f'{row["Study_Hours"]:.1f} hours/day',
        'Internal_Mark': f'{row["Internal_Mark"]:.1f}/100',
        'Assignment': f'{row["Assignment"]:.1f}/100',
        'Previous_Mark': f'{row["Previous_Mark"]:.1f}/100',
    }
    rows = ''.join(f'<tr><td>{esc(key.replace("_", " ") )}</td><td>{esc(labels[key])}</td></tr>' for key in FEATURES)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Student Progress Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;color:#222}}h1{{text-align:center}}.sub{{text-align:center;color:#666}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}th,td{{border:1px solid #aaa;padding:9px}}th{{background:#eee}}
.box{{border:1px solid #ccc;padding:12px;border-radius:8px;margin:12px 0}}@media print{{body{{margin:15px}}}}</style>
</head><body><h1>Student Progress Report</h1><p class='sub'>Student Performance Analysis &amp; Prediction System</p>
<div class='box'><b>Student Name:</b> {esc(name)}<br><b>Student ID:</b> {esc(sid)}<br>
<b>Report Date:</b> {datetime.now().strftime('%d-%m-%Y')}<br><b>Method:</b> K-Means + Random Forest</div>
<h2>1. Prediction Summary</h2><table><tr><td>Predicted Performance</td><td>{esc(prediction)}</td></tr>
<tr><td>Student Cluster</td><td>{cluster}</td></tr><tr><td>Model Confidence</td><td>{confidence:.1f}%</td></tr></table>
<h2>2. Academic Inputs</h2><table><tr><th>Feature</th><th>Value</th></tr>{rows}</table>
<h2>3. Suggested Improvement Plan</h2><ul>{rec_html}</ul>
<p style='color:#666;font-size:12px'>This report is for academic demonstration. Model outputs should support, not replace, teacher or academic judgment.</p>
</body></html>"""

if st.button("🔮 Predict Performance", type="primary", width="stretch"):
    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]
    confidence = float(probabilities.max()) * 100
    cluster = int(kmeans.predict(scaler.transform(student))[0])
    recommendations = make_recommendations(student.iloc[0], prediction)

    st.session_state["prediction"] = prediction
    st.session_state["confidence"] = confidence
    st.session_state["cluster"] = cluster
    st.session_state["recommendations"] = recommendations
    st.session_state["student_row"] = student.copy()

if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    confidence = st.session_state["confidence"]
    cluster = st.session_state["cluster"]
    recommendations = st.session_state["recommendations"]
    current_student = st.session_state["student_row"]

    st.divider()
    st.subheader("📋 Prediction Result")

    r1, r2, r3 = st.columns(3)
    r1.success(f"Predicted Performance: {prediction}")
    r2.info(f"Student Cluster: {cluster}")
    r3.metric("Model Confidence", f"{confidence:.1f}%")

    st.subheader("📊 Student Input Summary")
    st.dataframe(current_student, width="stretch", hide_index=True)

    st.subheader("💡 Progress Recommendations")
    for rec in recommendations:
        st.write("• " + rec)

    report_html = build_progress_html(student_name, student_id, current_student.iloc[0], prediction, cluster, confidence, recommendations)
    st.download_button(
        "📥 Download Progress Report (HTML)",
        data=report_html,
        file_name=f"{student_id}_Student_Progress_Report.html",
        mime="text/html",
        width="stretch"
    )
    st.caption("For PDF: open the downloaded HTML report in a browser → Print → Save as PDF.")

st.divider()
st.subheader("📈 Dataset Overview")
m1, m2, m3 = st.columns(3)
m1.metric("Total Students", len(data))
m2.metric("Features", len(FEATURES))
m3.metric("Performance Classes", data["Performance"].nunique())

st.caption("Academic prototype. Predictions should support, not replace, academic judgment.")
