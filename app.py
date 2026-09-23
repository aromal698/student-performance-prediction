import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

from io import BytesIO


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "data/student_performance.csv"

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]

SUBJECTS = [
    "Mathematics",
    "Data Structures",
    "Database Management System",
    "Artificial Intelligence",
    "Machine Learning",
    "Survey and Geomatics"
]

TEACHER_PASSWORD = "*ktutech"


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.low-box {
    padding: 15px;
    border: 2px solid red;
    border-radius: 12px;
    margin-bottom: 10px;
}

.average-box {
    padding: 15px;
    border: 2px solid orange;
    border-radius: 12px;
    margin-bottom: 10px;
}

.above-box {
    padding: 15px;
    border: 2px solid #e6c300;
    border-radius: 12px;
    margin-bottom: 10px;
}

.good-box {
    padding: 15px;
    border: 2px solid green;
    border-radius: 12px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)

    return data


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_models():

    data = load_data()

    required_columns = FEATURES + ["Performance"]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in dataset: "
            + ", ".join(missing)
        )

    X = data[FEATURES].copy()

    y = data["Performance"].copy()

    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    data["Cluster"] = clusters

    try:
        silhouette = silhouette_score(
            X_scaled,
            clusters
        )
    except Exception:
        silhouette = 0.0

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report_text = classification_report(
        y_test,
        y_pred
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    return (
        data,
        scaler,
        kmeans,
        model,
        accuracy,
        silhouette,
        report_text,
        report_dict,
        matrix
    )


# ============================================================
# ATTENDANCE CONVERSION
# ============================================================

def attendance_mark(attendance):

    if attendance >= 90:
        return 5

    elif attendance >= 80:
        return 4

    elif attendance >= 70:
        return 3

    elif attendance >= 60:
        return 2

    elif attendance >= 10:
        return 1

    return 0


# ============================================================
# PERFORMANCE LEVEL
# ============================================================

def get_performance_level(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    attendance_percent = (
        attendance_mark(attendance) / 5
    ) * 100

    internal_percent = (
        internal / 40
    ) * 100

    assignment_percent = (
        assignment / 15
    ) * 100

    previous_percent = (
        previous / 60
    ) * 100

    study_percent = (
        study_hours / 6
    ) * 100

    study_percent = min(
        study_percent,
        100
    )

    overall = np.mean([
        attendance_percent,
        internal_percent,
        assignment_percent,
        previous_percent,
        study_percent
    ])

    if overall < 50:

        return (
            overall,
            "Low Performance",
            "🔴"
        )

    elif overall < 65:

        return (
            overall,
            "Average Performance",
            "🟠"
        )

    elif overall < 80:

        return (
            overall,
            "Above Average",
            "🟡"
        )

    else:

        return (
            overall,
            "Good Performance",
            "🟢"
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def recommendations(
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
    level
):

    result = []

    if attendance < 75:

        result.append(
            "Improve attendance and attend classes regularly."
        )

    if study_hours < 2:

        result.append(
            "Increase daily study time gradually."
        )

    if internal < 20:

        result.append(
            "Focus more on internal examinations."
        )

    if assignment < 8:

        result.append(
            "Complete assignments regularly and on time."
        )

    if previous < 30:

        result.append(
            "Revise previous topics and strengthen basic concepts."
        )

    if level == "Low Performance":

        result.append(
            "Prepare a weekly study timetable."
        )

        result.append(
            "Ask teachers for help with difficult topics."
        )

    elif level == "Average Performance":

        result.append(
            "Maintain regular revision and improve consistency."
        )

        result.append(
            "Practice more questions before examinations."
        )

    elif level == "Above Average":

        result.append(
            "Make small improvements in attendance and internal marks."
        )

        result.append(
            "Continue regular study and revision."
        )

    else:

        result.append(
            "Excellent performance. Maintain the same consistency."
        )

        result.append(
            "Continue regular attendance and assignment submission."
        )

    return list(dict.fromkeys(result))


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_input(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    # Convert UI values to approximately 0-100 scale
    # used by the existing model.

    study_model = (
        study_hours / 6
    ) * 100

    internal_model = (
        internal / 40
    ) * 100

    assignment_model = (
        assignment / 15
    ) * 100

    previous_model = (
        previous / 60
    ) * 100

    return pd.DataFrame(
        [[
            attendance,
            study_model,
            internal_model,
            assignment_model,
            previous_model
        ]],
        columns=FEATURES
    )


# ============================================================
# PREDICT
# ============================================================

def predict_student(
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
    model,
    scaler,
    kmeans
):

    model_input = prepare_input(
        attendance,
        study_hours,
        internal,
        assignment,
        previous
    )

    # Random Forest
    prediction = model.predict(
        model_input
    )[0]

    probability = model.predict_proba(
        model_input
    )[0]

    confidence = max(
        probability
    ) * 100

    # K-Means
    scaled = scaler.transform(
        model_input
    )

    cluster = int(
        kmeans.predict(
            scaled
        )[0]
    )

    # Performance level
    overall, level, symbol = get_performance_level(
        attendance,
        study_hours,
        internal,
        assignment,
        previous
    )

    suggestions = recommendations(
        attendance,
        study_hours,
        internal,
        assignment,
        previous,
        level
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "cluster": cluster,
        "overall": overall,
        "level": level,
        "symbol": symbol,
        "suggestions": suggestions,
        "attendance_mark": attendance_mark(attendance)
    }


# ============================================================
# CREATE HTML PROGRESS REPORT
# ============================================================

def create_progress_report(
    student_name,
    university_id,
    semester,
    branch,
    dob_year,
    results
):

    rows = ""

    for subject, result in results.items():

        suggestions = "<br>".join(
            [
                "• " + item
                for item in result["suggestions"]
            ]
        )

        rows += f"""
        <tr>
            <td>{subject}</td>
            <td>{result["symbol"]} {result["level"]}</td>
            <td>{result["overall"]:.1f}%</td>
            <td>{result["attendance"]:.0f}%</td>
            <td>{result["attendance_mark"]}/5</td>
            <td>{result["internal"]:.0f}/40</td>
            <td>{result["previous"]:.0f}/60</td>
            <td>{result["assignment"]:.0f}/15</td>
            <td>{result["study_hours"]:.1f}/6</td>
            <td>{result["prediction"]}</td>
        </tr>

        <tr>
            <td colspan="10">
                <b>Suggestions:</b><br>
                {suggestions}
            </td>
        </tr>
        """

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Student Progress Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 35px;
    color: #222;
}}

h1 {{
    text-align: center;
    font-size: 25px;
}}

h2 {{
    margin-top: 25px;
}}

.subtitle {{
    text-align: center;
    font-size: 15px;
}}

.info {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}}

.info td {{
    border: 1px solid #999;
    padding: 8px;
}}

.info td:first-child {{
    font-weight: bold;
    width: 30%;
}}

.report {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
    font-size: 11px;
}}

.report th,
.report td {{
    border: 1px solid #888;
    padding: 7px;
    text-align: center;
}}

.report th {{
    background: #eeeeee;
}}

.report td[colspan] {{
    text-align: left;
    background: #fafafa;
}}

.note {{
    margin-top: 25px;
    padding: 12px;
    border: 1px solid #aaa;
    background: #f5f5f5;
    font-size: 12px;
}}

</style>

</head>

<body>

<h1>STUDENT PERFORMANCE PROGRESS REPORT</h1>

<div class="subtitle">
B.Tech S3 – Artificial Intelligence & Data Science
</div>

<div class="subtitle">
K-Means Clustering + Random Forest Classification
</div>

<h2>1. Student Information</h2>

<table class="info">

<tr>
<td>Student Name</td>
<td>{student_name}</td>
</tr>

<tr>
<td>University ID</td>
<td>{university_id}</td>
</tr>

<tr>
<td>Semester</td>
<td>{semester}</td>
</tr>

<tr>
<td>Branch</td>
<td>{branch}</td>
</tr>

<tr>
<td>DOB Year</td>
<td>{dob_year}</td>
</tr>

</table>


<h2>2. Subject-wise Performance</h2>

<table class="report">

<tr>

<th>Subject</th>
<th>Performance</th>
<th>Overall</th>
<th>Attendance</th>
<th>Att. Mark</th>
<th>Internal</th>
<th>Previous</th>
<th>Assignment</th>
<th>Study Hours</th>
<th>ML Prediction</th>

</tr>

{rows}

</table>


<h2>3. Attendance Conversion</h2>

<table class="report">

<tr>
<th>Attendance</th>
<th>Converted Mark</th>
</tr>

<tr>
<td>90–100%</td>
<td>5/5</td>
</tr>

<tr>
<td>80–89%</td>
<td>4/5</td>
</tr>

<tr>
<td>70–79%</td>
<td>3/5</td>
</tr>

<tr>
<td>60–69%</td>
<td>2/5</td>
</tr>

<tr>
<td>10–59%</td>
<td>1/5</td>
</tr>

<tr>
<td>Below 10%</td>
<td>0/5</td>
</tr>

</table>


<h2>4. Performance Indicators</h2>

<table class="report">

<tr>
<th>Indicator</th>
<th>Performance Level</th>
</tr>

<tr>
<td>🔴 Red</td>
<td>Low Performance</td>
</tr>

<tr>
<td>🟠 Orange</td>
<td>Average Performance</td>
</tr>

<tr>
<td>🟡 Yellow</td>
<td>Above Average</td>
</tr>

<tr>
<td>🟢 Green</td>
<td>Good Performance</td>
</tr>

</table>


<div class="note">

<b>Note:</b>

The performance indicator scale is a project-defined
academic monitoring scale and is not an official KTU
grading standard.

The machine-learning prediction is intended to support
academic monitoring and should not replace teacher
evaluation.

</div>

</body>

</html>
"""

    return html


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        '<div class="title">🎓 Student Performance Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'B.Tech S3 – Artificial Intelligence & Data Science'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Choose Login")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):

            st.session_state.login_type = "student"

            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):

            st.session_state.login_type = "teacher"

            st.rerun()


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.title("🎓 Student Login")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        valid = False

        if len(password) == 9:

            if password[:5].upper() == "BTECH":

                try:

                    year = int(
                        password[5:]
                    )

                    if 2000 <= year <= 2020:

                        valid = True

                except ValueError:

                    valid = False

        if valid:

            st.session_state.logged_in = True

            st.session_state.role = "student"

            st.session_state.dob_year = int(
                password[5:]
            )

            st.rerun()

        else:

            st.error(
                "Invalid student password."
            )

    if st.button("← Back"):

        st.session_state.login_type = None

        st.rerun()


# ============================================================
# TEACHER LOGIN
# ============================================================

def teacher_login():

    st.title("👨‍🏫 Teacher Login")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        if password == TEACHER_PASSWORD:

            st.session_state.logged_in = True

            st.session_state.role = "teacher"

            st.rerun()

        else:

            st.error(
                "Invalid teacher password."
            )

    if st.button("← Back"):

        st.session_state.login_type = None

        st.rerun()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    (
        data,
        scaler,
        kmeans,
        model,
        accuracy,
        silhouette,
        report_text,
        report_dict,
        matrix
    ) = train_models()

    st.title("🎓 Student Dashboard")

    col1, col2 = st.columns([4, 1])

    with col1:

        st.success(
            "Student login successful."
        )

    with col2:

        if st.button("Logout"):

            st.session_state.clear()

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # STUDENT INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "1. Student Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input(
            "Student Name"
        )

        university_id = st.text_input(
            "University ID"
        )

    with col2:

        semester = st.selectbox(
            "Semester",
            [
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
                "S6",
                "S7",
                "S8"
            ],
            index=2
        )

        branch = st.text_input(
            "Branch",
            value="B.Tech AI & Data Science"
        )

    st.divider()

    # --------------------------------------------------------
    # SUBJECT DATA
    # --------------------------------------------------------

    st.subheader(
        "2. Subject-wise Academic Details"
    )

    subject_data = {}

    for subject in SUBJECTS:

        st.markdown(
            f"### 📘 {subject}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
                key=f"{subject}_attendance"
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=6.0,
                value=3.0,
                step=0.5,
                key=f"{subject}_study"
            )

        with c2:

            internal = st.number_input(
                "Internal Mark / 40",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                step=1.0,
                key=f"{subject}_internal"
            )

            assignment = st.number_input(
                "Assignment / 15",
                min_value=0.0,
                max_value=15.0,
                value=10.0,
                step=1.0,
                key=f"{subject}_assignment"
            )

        with c3:

            previous = st.number_input(
                "Previous Mark / 60",
                min_value=0.0,
                max_value=60.0,
                value=35.0,
                step=1.0,
                key=f"{subject}_previous"
            )

            st.write(
                "Attendance Mark:",
                f"**{attendance_mark(attendance)}/5**"
            )

        subject_data[subject] = {
            "attendance": attendance,
            "study_hours": study_hours,
            "internal": internal,
            "assignment": assignment,
            "previous": previous
        }

        st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Student Performance",
        type="primary",
        width="stretch"
    ):

        if not student_name.strip():

            st.warning(
                "Please enter Student Name."
            )

            return

        if not university_id.strip():

            st.warning(
                "Please enter University ID."
            )

            return

        results = {}

        for subject, values in subject_data.items():

            prediction = predict_student(
                values["attendance"],
                values["study_hours"],
                values["internal"],
                values["assignment"],
                values["previous"],
                model,
                scaler,
                kmeans
            )

            results[subject] = {
                **values,
                **prediction
            }

        st.session_state.results = results

        st.session_state.student_name = student_name

        st.session_state.university_id = university_id

        st.session_state.semester = semester

        st.session_state.branch = branch

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "results" in st.session_state:

        results = st.session_state.results

        st.divider()

        st.subheader(
            "3. Subject-wise Performance"
        )

        for subject, result in results.items():

            level = result["level"]

            if level == "Low Performance":
                box = "low-box"

            elif level == "Average Performance":
                box = "average-box"

            elif level == "Above Average":
                box = "above-box"

            else:
                box = "good-box"

            st.markdown(
                f"""
                <div class="{box}">

                <h3>
                {result["symbol"]} {subject}
                </h3>

                <b>Performance:</b>
                {result["level"]}

                <br>

                <b>Overall Indicator:</b>
                {result["overall"]:.1f}%

                <br>

                <b>ML Prediction:</b>
                {result["prediction"]}

                <br>

                <b>Model Confidence:</b>
                {result["confidence"]:.1f}%

                <br>

                <b>K-Means Cluster:</b>
                {result["cluster"]}

                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.subheader(
            "4. Overall Academic Information"
        )

        table = []

        for subject, result in results.items():

            table.append({
                "Subject": subject,
                "Attendance": (
                    f'{result["attendance"]:.0f}%'
                ),
                "Attendance Mark": (
                    f'{result["attendance_mark"]}/5'
                ),
                "Internal": (
                    f'{result["internal"]:.0f}/40'
                ),
                "Previous": (
                    f'{result["previous"]:.0f}/60'
                ),
                "Assignment": (
                    f'{result["assignment"]:.0f}/15'
                ),
                "Study Hours": (
                    f'{result["study_hours"]:.1f}/6'
                ),
                "Prediction": result["prediction"],
                "Level": result["level"]
            })

        result_df = pd.DataFrame(table)

        st.dataframe(
            result_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # SUGGESTIONS
        # ----------------------------------------------------

        st.subheader(
            "5. Improvement / Maintenance"
        )

        for subject, result in results.items():

            with st.expander(
                f'{result["symbol"]} {subject}'
            ):

                for item in result["suggestions"]:

                    st.write(
                        "• " + item
                    )

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------

        st.subheader(
            "6. Progress Report"
        )

        html_report = create_progress_report(
            st.session_state.student_name,
            st.session_state.university_id,
            st.session_state.semester,
            st.session_state.branch,
            st.session_state.dob_year,
            results
        )

        st.download_button(
            label="📥 Download Progress Report",
            data=html_report,
            file_name=(
                f'{st.session_state.university_id}'
                '_Progress_Report.html'
            ),
            mime="text/html",
            width="stretch"
        )

        st.info(
            "Open the downloaded HTML report in a browser and "
            "use Print → Save as PDF if you need a PDF copy."
        )


# ============================================================
# TEACHER DASHBOARD
# ============================================================

def teacher_dashboard():

    (
        data,
        scaler,
        kmeans,
        model,
        accuracy,
        silhouette,
        report_text,
        report_dict,
        matrix
    ) = train_models()

    st.title(
        "👨‍🏫 Teacher Dashboard"
    )

    if st.button("Logout"):

        st.session_state.clear()

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    st.subheader(
        "📊 Model Performance"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with c2:

        st.metric(
            "K-Means Silhouette Score",
            f"{silhouette:.4f}"
        )

    with c3:

        st.metric(
            "Dataset Records",
            len(data)
        )

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "📋 Classification Report"
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    st.dataframe(
        report_df.round(3),
        width="stretch"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "🔢 Confusion Matrix"
    )

    labels = sorted(
        data["Performance"].unique()
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    st.dataframe(
        matrix_df,
        width="stretch"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🌳 Feature Importance"
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )

    st.dataframe(
        importance_df.round(4),
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    st.subheader(
        "🔵 K-Means Cluster Distribution"
    )

    cluster_counts = (
        data["Cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_df = pd.DataFrame({
        "Cluster": cluster_counts.index.astype(str),
        "Students": cluster_counts.values
    })

    st.bar_chart(
        cluster_df.set_index(
            "Cluster"
        )
    )

    # --------------------------------------------------------
    # PERFORMANCE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📈 Performance Distribution"
    )

    performance_counts = (
        data["Performance"]
        .value_counts()
    )

    performance_df = pd.DataFrame({
        "Performance": performance_counts.index,
        "Students": performance_counts.values
    })

    st.bar_chart(
        performance_df.set_index(
            "Performance"
        )
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader(
        "📁 Dataset"
    )

    st.dataframe(
        data,
        width="stretch",
        hide_index=True
    )

    csv = data.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Dataset",
        data=csv,
        file_name="student_performance_dataset.csv",
        mime="text/csv",
        width="stretch"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if "login_type" not in st.session_state:

        st.session_state.login_type = None

    if "logged_in" not in st.session_state:

        st.session_state.logged_in = False

    # --------------------------------------------------------
    # LOGGED IN
    # --------------------------------------------------------

    if st.session_state.logged_in:

        if st.session_state.role == "student":

            student_dashboard()

        elif st.session_state.role == "teacher":

            teacher_dashboard()

        return

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    if st.session_state.login_type is None:

        login_page()

    elif st.session_state.login_type == "student":

        student_login()

    elif st.session_state.login_type == "teacher":

        teacher_login()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
