import os
import html
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# BRANCHES AND SUBJECTS
# =========================================================

BRANCH_SUBJECTS = {
    "Artificial Intelligence and Data Science": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Artificial Intelligence",
        "Machine Learning",
        "Data Science"
    ],

    "Computer Science and Engineering": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Operating Systems",
        "Computer Networks",
        "Object Oriented Programming"
    ],

    "Computer Science and Engineering (AI)": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Artificial Intelligence",
        "Machine Learning",
        "Computer Networks"
    ],

    "Computer Science and Engineering (Data Science)": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Data Analytics",
        "Machine Learning",
        "Statistics"
    ],

    "Information Technology": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Computer Networks",
        "Web Technology",
        "Operating Systems"
    ],

    "Cyber Security": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Cyber Security",
        "Computer Networks",
        "Cryptography"
    ],

    "Electronics and Communication Engineering": [
        "Mathematics",
        "Electronic Devices",
        "Digital Electronics",
        "Signals and Systems",
        "Communication Engineering",
        "Microprocessors"
    ],

    "Electrical and Electronics Engineering": [
        "Mathematics",
        "Circuit Theory",
        "Digital Electronics",
        "Electrical Machines",
        "Power Systems",
        "Control Systems"
    ],

    "Electronics and Instrumentation Engineering": [
        "Mathematics",
        "Electronic Devices",
        "Measurements",
        "Control Systems",
        "Digital Electronics",
        "Instrumentation"
    ],

    "Mechanical Engineering": [
        "Mathematics",
        "Engineering Mechanics",
        "Thermodynamics",
        "Fluid Mechanics",
        "Manufacturing Process",
        "Machine Design"
    ],

    "Civil Engineering": [
        "Mathematics",
        "Engineering Mechanics",
        "Surveying",
        "Strength of Materials",
        "Fluid Mechanics",
        "Construction Technology"
    ]
}


# =========================================================
# DEMO TEACHER ACCOUNTS
# =========================================================

TEACHERS = {
    "teacher_aids": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Data Science"
    },

    "teacher_cse": {
        "password": "ktutech",
        "branch": "Computer Science and Engineering"
    },

    "teacher_cseai": {
        "password": "ktutech",
        "branch": "Computer Science and Engineering (AI)"
    },

    "teacher_ds": {
        "password": "ktutech",
        "branch": "Computer Science and Engineering (Data Science)"
    },

    "teacher_it": {
        "password": "ktutech",
        "branch": "Information Technology"
    },

    "teacher_cyber": {
        "password": "ktutech",
        "branch": "Cyber Security"
    },

    "teacher_ece": {
        "password": "ktutech",
        "branch": "Electronics and Communication Engineering"
    },

    "teacher_eee": {
        "password": "ktutech",
        "branch": "Electrical and Electronics Engineering"
    },

    "teacher_eie": {
        "password": "ktutech",
        "branch": "Electronics and Instrumentation Engineering"
    },

    "teacher_me": {
        "password": "ktutech",
        "branch": "Mechanical Engineering"
    },

    "teacher_ce": {
        "password": "ktutech",
        "branch": "Civil Engineering"
    }
}


# =========================================================
# DATA PATHS
# =========================================================

DATA_DIR = "data"

DATASET_PATH = os.path.join(
    DATA_DIR,
    "student_performance.csv"
)

REPORT_PATH = os.path.join(
    DATA_DIR,
    "student_reports.csv"
)


# =========================================================
# CREATE DATA FOLDER
# =========================================================

os.makedirs(DATA_DIR, exist_ok=True)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "student_logged_in" not in st.session_state:
    st.session_state.student_logged_in = False

if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False

if "teacher_username" not in st.session_state:
    st.session_state.teacher_username = ""

if "student_password" not in st.session_state:
    st.session_state.student_password = ""


# =========================================================
# TITLE
# =========================================================

st.title("🎓 Student Performance Prediction")


# =========================================================
# ATTENDANCE CONVERSION
# =========================================================

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

    else:
        return 0


# =========================================================
# PERFORMANCE ANALYSIS
# =========================================================

def performance_analysis(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    attendance_m = attendance_mark(attendance)

    attendance_percent = (attendance_m / 5) * 100

    study_percent = min(
        (study_hours / 6) * 100,
        100
    )

    internal_percent = (
        internal / 40
    ) * 100

    assignment_percent = (
        assignment / 15
    ) * 100

    previous_percent = (
        previous / 60
    ) * 100

    overall = np.mean([
        attendance_percent,
        study_percent,
        internal_percent,
        assignment_percent,
        previous_percent
    ])

    if overall < 50:

        level = "Low Performance"
        symbol = "🔴"

    elif overall < 65:

        level = "Average Performance"
        symbol = "🟠"

    elif overall < 80:

        level = "Above Average Performance"
        symbol = "🟡"

    else:

        level = "Good Performance"
        symbol = "🟢"

    return {
        "attendance_mark": attendance_m,
        "attendance_percent": attendance_percent,
        "study_percent": study_percent,
        "internal_percent": internal_percent,
        "assignment_percent": assignment_percent,
        "previous_percent": previous_percent,
        "overall": overall,
        "level": level,
        "symbol": symbol
    }


# =========================================================
# RECOMMENDATIONS
# =========================================================

def recommendations(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    rec = []

    if attendance < 80:
        rec.append(
            "Improve attendance and attend classes regularly."
        )

    if study_hours < 3:
        rec.append(
            "Increase daily study time gradually."
        )

    if internal < 20:
        rec.append(
            "Focus more on internal examination preparation."
        )

    if assignment < 8:
        rec.append(
            "Complete assignments regularly and on time."
        )

    if previous < 30:
        rec.append(
            "Revise previous topics and strengthen fundamentals."
        )

    if not rec:
        rec.append(
            "Maintain your current performance and continue regular revision."
        )

    return rec


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_PATH):
        return None

    try:
        return pd.read_csv(DATASET_PATH)

    except Exception:
        return None


# =========================================================
# TRAIN ML MODELS
# =========================================================

@st.cache_resource
def train_models():

    data = load_dataset()

    if data is None:
        return None

    required_columns = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Performance"
    ]

    for col in required_columns:

        if col not in data.columns:
            return None

    X = data[
        [
            "Attendance",
            "Study_Hours",
            "Internal_Mark",
            "Assignment",
            "Previous_Mark"
        ]
    ].copy()

    y = data["Performance"]

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    valid = X.notna().all(axis=1) & y.notna()

    X = X[valid]
    y = y[valid]

    if len(X) < 10:
        return None

    # -----------------------------
    # K-MEANS
    # -----------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        X_scaled
    )

    try:

        silhouette = silhouette_score(
            X_scaled,
            clusters
        )

    except Exception:

        silhouette = 0

    # -----------------------------
    # RANDOM FOREST
    # -----------------------------

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

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return {
        "model": model,
        "kmeans": kmeans,
        "scaler": scaler,
        "accuracy": accuracy,
        "silhouette": silhouette
    }


# =========================================================
# SAVE STUDENT REPORT
# =========================================================

def save_student_report(record):

    df_new = pd.DataFrame(
        [record]
    )

    if os.path.exists(REPORT_PATH):

        try:

            df_old = pd.read_csv(
                REPORT_PATH
            )

            df = pd.concat(
                [df_old, df_new],
                ignore_index=True
            )

        except Exception:

            df = df_new

    else:

        df = df_new

    df.to_csv(
        REPORT_PATH,
        index=False
    )


# =========================================================
# GENERATE HTML REPORT
# =========================================================

def create_html_report(
    student_name,
    university_id,
    semester,
    branch,
    subject_results,
    ml_results
):

    rows = ""

    for item in subject_results:

        rows += f"""
        <tr>
            <td>{html.escape(str(item["subject"]))}</td>
            <td>{item["attendance"]}%</td>
            <td>{item["study_hours"]}</td>
            <td>{item["internal"]}/40</td>
            <td>{item["assignment"]}/15</td>
            <td>{item["previous"]}/60</td>
            <td>{item["symbol"]} {html.escape(item["level"])}</td>
            <td>{item["overall"]:.1f}%</td>
        </tr>
        """

    ml_prediction = html.escape(
        str(ml_results["prediction"])
    )

    confidence = ml_results["confidence"]

    cluster = ml_results["cluster"]

    accuracy = ml_results["accuracy"] * 100

    document = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Student Progress Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    color: #222;
}}

h1 {{
    text-align: center;
}}

h2 {{
    margin-top: 30px;
}}

.info {{
    border: 1px solid #ccc;
    padding: 15px;
    margin-bottom: 20px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}}

th, td {{
    border: 1px solid #aaa;
    padding: 8px;
    text-align: center;
}}

th {{
    background: #eeeeee;
}}

.note {{
    margin-top: 30px;
    padding: 15px;
    border: 1px solid #ccc;
}}

@media print {{

    body {{
        margin: 20px;
    }}

}}

</style>

</head>

<body>

<h1>🎓 Student Progress Report</h1>

<div class="info">

<p><b>Student Name:</b>
{html.escape(student_name)}
</p>

<p><b>University ID:</b>
{html.escape(university_id)}
</p>

<p><b>Semester:</b>
{html.escape(semester)}
</p>

<p><b>Branch:</b>
{html.escape(branch)}
</p>

</div>


<h2>Subject-wise Performance</h2>

<table>

<tr>

<th>Subject</th>
<th>Attendance</th>
<th>Study Hours</th>
<th>Internal</th>
<th>Assignment</th>
<th>Previous Mark</th>
<th>Performance</th>
<th>Overall</th>

</tr>

{rows}

</table>


<h2>Performance Indicator</h2>

<p>
🔴 Low Performance
</p>

<p>
🟠 Average Performance
</p>

<p>
🟡 Above Average Performance
</p>

<p>
🟢 Good Performance
</p>


<h2>Machine Learning Prediction</h2>

<div class="info">

<p>
<b>Random Forest Prediction:</b>
{ml_prediction}
</p>

<p>
<b>Model Confidence:</b>
{confidence:.1f}%
</p>

<p>
<b>K-Means Cluster:</b>
{cluster}
</p>

<p>
<b>Random Forest Test Accuracy:</b>
{accuracy:.2f}%
</p>

</div>


<h2>Improvement Suggestions</h2>

<ul>
"""

    for item in subject_results:

        document += f"""

<li>
<b>{html.escape(item["subject"])}:</b>
"""

        for rec in item["recommendations"]:

            document += f"""

{html.escape(rec)}<br>

"""

        document += "</li>"

    document += """

</ul>


<div class="note">

<b>Note:</b>

This report is generated using machine learning
and academic performance indicators. The prediction
is intended to support academic monitoring and should
not replace teacher or institutional evaluation.

</div>

</body>

</html>
"""

    return document


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.subheader("Choose Login")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):

            st.session_state.page = "student_login"

            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):

            st.session_state.page = "teacher_login"

            st.rerun()


# =========================================================
# STUDENT LOGIN
# =========================================================

elif st.session_state.page == "student_login":

    st.subheader("🎓 Student Login")

    password = st.text_input(
        "Enter Student Password",
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        valid = False

        if len(password) == 9:

            if password[:5].upper() == "BTECH":

                year_text = password[5:]

                if year_text.isdigit():

                    year = int(year_text)

                    if 2000 <= year <= 2020:

                        valid = True

        if valid:

            st.session_state.student_logged_in = True

            st.session_state.student_password = password

            st.session_state.page = "student_dashboard"

            st.rerun()

        else:

            st.error(
                "Invalid student password."
            )

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.rerun()


# =========================================================
# TEACHER LOGIN
# =========================================================

elif st.session_state.page == "teacher_login":

    st.subheader("👨‍🏫 Teacher Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        if (
            username in TEACHERS
            and
            password == TEACHERS[username]["password"]
        ):

            st.session_state.teacher_logged_in = True

            st.session_state.teacher_username = username

            st.session_state.page = "teacher_dashboard"

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.rerun()


# =========================================================
# STUDENT DASHBOARD
# =========================================================

elif st.session_state.page == "student_dashboard":

    if not st.session_state.student_logged_in:

        st.session_state.page = "home"

        st.rerun()

    st.subheader("🎓 Student Dashboard")

    if st.button(
        "Logout"
    ):

        st.session_state.student_logged_in = False

        st.session_state.page = "home"

        st.rerun()

    st.divider()

    # -----------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------

    st.header("Student Information")

    c1, c2 = st.columns(2)

    with c1:

        student_name = st.text_input(
            "Student Name"
        )

        university_id = st.text_input(
            "University ID"
        )

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
            ]
        )

    with c2:

        branch = st.selectbox(
            "Branch",
            list(BRANCH_SUBJECTS.keys())
        )

    subjects = BRANCH_SUBJECTS[branch]

    st.info(
        "Exactly 6 subjects are required."
    )

    selected_subjects = st.multiselect(
        "Select 6 Subjects",
        subjects,
        max_selections=6
    )

    if len(selected_subjects) != 6:

        st.warning(
            "Please select exactly 6 subjects."
        )

    # -----------------------------------------------
    # SUBJECT INPUTS
    # -----------------------------------------------

    subject_inputs = {}

    if len(selected_subjects) == 6:

        st.header("Academic Information")

        for subject in selected_subjects:

            st.subheader(
                f"📚 {subject}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                attendance = st.number_input(
                    "Attendance (%)",
                    min_value=0,
                    max_value=100,
                    value=80,
                    key=f"attendance_{subject}"
                )

            with c2:

                study_hours = st.number_input(
                    "Study Hours / Day",
                    min_value=0.0,
                    max_value=6.0,
                    value=3.0,
                    step=0.5,
                    key=f"study_{subject}"
                )

            with c3:

                internal = st.number_input(
                    "Internal Mark / 40",
                    min_value=0,
                    max_value=40,
                    value=25,
                    key=f"internal_{subject}"
                )

            with c4:

                assignment = st.number_input(
                    "Assignment / 15",
                    min_value=0,
                    max_value=15,
                    value=10,
                    key=f"assignment_{subject}"
                )

            with c5:

                previous = st.number_input(
                    "Previous Mark / 60",
                    min_value=0,
                    max_value=60,
                    value=40,
                    key=f"previous_{subject}"
                )

            subject_inputs[subject] = {
                "attendance": attendance,
                "study_hours": study_hours,
                "internal": internal,
                "assignment": assignment,
                "previous": previous
            }

    # -----------------------------------------------
    # PREDICT
    # -----------------------------------------------

    if len(selected_subjects) == 6:

        if st.button(
            "🔍 Predict Performance",
            width="stretch"
        ):

            if not student_name.strip():

                st.error(
                    "Please enter student name."
                )

            elif not university_id.strip():

                st.error(
                    "Please enter university ID."
                )

            else:

                models = train_models()

                if models is None:

                    st.error(
                        "Dataset/model could not be loaded. "
                        "Check data/student_performance.csv."
                    )

                else:

                    subject_results = []

                    ml_predictions = []

                    confidence_values = []

                    clusters = []

                    for subject in selected_subjects:

                        values = subject_inputs[subject]

                        analysis = performance_analysis(
                            values["attendance"],
                            values["study_hours"],
                            values["internal"],
                            values["assignment"],
                            values["previous"]
                        )

                        recs = recommendations(
                            values["attendance"],
                            values["study_hours"],
                            values["internal"],
                            values["assignment"],
                            values["previous"]
                        )

                        # ----------------------------------
                        # Convert UI values to model scale
                        # ----------------------------------

                        model_input = pd.DataFrame(
                            [[
                                values["attendance"],
                                values["study_hours"] / 6 * 10,
                                values["internal"] / 40 * 100,
                                values["assignment"] / 15 * 100,
                                values["previous"] / 60 * 100
                            ]],
                            columns=[
                                "Attendance",
                                "Study_Hours",
                                "Internal_Mark",
                                "Assignment",
                                "Previous_Mark"
                            ]
                        )

                        # ----------------------------------
                        # Random Forest
                        # ----------------------------------

                        prediction = models["model"].predict(
                            model_input
                        )[0]

                        probabilities = models[
                            "model"
                        ].predict_proba(
                            model_input
                        )[0]

                        confidence = float(
                            max(probabilities)
                        ) * 100

                        # ----------------------------------
                        # K-Means
                        # ----------------------------------

                        scaled_input = models[
                            "scaler"
                        ].transform(
                            model_input
                        )

                        cluster = int(
                            models["kmeans"].predict(
                                scaled_input
                            )[0]
                        )

                        ml_predictions.append(
                            str(prediction)
                        )

                        confidence_values.append(
                            confidence
                        )

                        clusters.append(
                            cluster
                        )

                        subject_results.append({
                            "subject": subject,
                            "attendance": values["attendance"],
                            "study_hours": values["study_hours"],
                            "internal": values["internal"],
                            "assignment": values["assignment"],
                            "previous": values["previous"],
                            "level": analysis["level"],
                            "symbol": analysis["symbol"],
                            "overall": analysis["overall"],
                            "recommendations": recs
                        })

                    # --------------------------------------
                    # OVERALL ML RESULT
                    # --------------------------------------

                    prediction_counts = pd.Series(
                        ml_predictions
                    ).value_counts()

                    overall_prediction = (
                        prediction_counts.index[0]
                    )

                    overall_confidence = np.mean(
                        confidence_values
                    )

                    overall_cluster = int(
                        round(
                            np.mean(clusters)
                        )
                    )

                    ml_results = {
                        "prediction": overall_prediction,
                        "confidence": overall_confidence,
                        "cluster": overall_cluster,
                        "accuracy": models["accuracy"]
                    }

                    # --------------------------------------
                    # DISPLAY RESULTS
                    # --------------------------------------

                    st.divider()

                    st.header(
                        "📊 Performance Result"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Random Forest Prediction",
                            overall_prediction
                        )

                    with col2:

                        st.metric(
                            "Confidence",
                            f"{overall_confidence:.1f}%"
                        )

                    with col3:

                        st.metric(
                            "K-Means Cluster",
                            overall_cluster
                        )

                    # --------------------------------------
                    # SUBJECT RESULTS
                    # --------------------------------------

                    st.header(
                        "📚 Subject-wise Performance"
                    )

                    for result in subject_results:

                        with st.container(
                            border=True
                        ):

                            st.subheader(
                                f"{result['symbol']} "
                                f"{result['subject']}"
                            )

                            st.write(
                                f"**Performance:** "
                                f"{result['level']}"
                            )

                            st.write(
                                f"**Overall Score:** "
                                f"{result['overall']:.1f}%"
                            )

                            st.write(
                                "**Improvement / Feedback:**"
                            )

                            for rec in result[
                                "recommendations"
                            ]:

                                st.write(
                                    f"• {rec}"
                                )

                    # --------------------------------------
                    # SAVE REPORT
                    # --------------------------------------

                    for result in subject_results:

                        save_student_report({
                            "Student_Name": student_name,
                            "University_ID": university_id,
                            "Semester": semester,
                            "Branch": branch,
                            "Subject": result["subject"],
                            "Attendance": result["attendance"],
                            "Study_Hours": result["study_hours"],
                            "Internal_Mark": result["internal"],
                            "Assignment": result["assignment"],
                            "Previous_Mark": result["previous"],
                            "Performance_Level": result["level"],
                            "Overall_Percent": round(
                                result["overall"],
                                2
                            ),
                            "ML_Prediction": overall_prediction,
                            "ML_Confidence": round(
                                overall_confidence,
                                2
                            ),
                            "Cluster": overall_cluster
                        })

                    # --------------------------------------
                    # HTML REPORT
                    # --------------------------------------

                    report_html = create_html_report(
                        student_name,
                        university_id,
                        semester,
                        branch,
                        subject_results,
                        ml_results
                    )

                    st.download_button(
                        label="📥 Download Progress Report",
                        data=report_html,
                        file_name=(
                            f"{university_id}_"
                            "Progress_Report.html"
                        ),
                        mime="text/html",
                        width="stretch"
                    )

                    st.success(
                        "Progress report generated successfully."
                    )

                    st.info(
                        "Open the downloaded HTML report → "
                        "Print → Save as PDF."
                    )


# =========================================================
# TEACHER DASHBOARD
# =========================================================

elif st.session_state.page == "teacher_dashboard":

    if not st.session_state.teacher_logged_in:

        st.session_state.page = "home"

        st.rerun()

    username = (
        st.session_state.teacher_username
    )

    teacher_branch = TEACHERS[
        username
    ]["branch"]

    st.subheader(
        "👨‍🏫 Teacher Dashboard"
    )

    st.write(
        f"**Assigned Branch:** {teacher_branch}"
    )

    if st.button(
        "Logout"
    ):

        st.session_state.teacher_logged_in = False

        st.session_state.teacher_username = ""

        st.session_state.page = "home"

        st.rerun()

    st.divider()

    # -----------------------------------------------
    # LOAD REPORTS
    # -----------------------------------------------

    if not os.path.exists(REPORT_PATH):

        st.info(
            "No student reports are available yet."
        )

    else:

        try:

            reports = pd.read_csv(
                REPORT_PATH
            )

        except Exception:

            reports = pd.DataFrame()

        if reports.empty:

            st.info(
                "No student reports are available yet."
            )

        else:

            # -----------------------------------------
            # BRANCH FILTER
            # -----------------------------------------

            branch_reports = reports[
                reports["Branch"].astype(str)
                == str(teacher_branch)
            ].copy()

            st.header(
                "Students in Your Branch"
            )

            if branch_reports.empty:

                st.info(
                    "No students from your branch "
                    "have submitted reports."
                )

            else:

                # -------------------------------------
                # SUMMARY
                # -------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Student Records",
                        len(branch_reports)
                    )

                with col2:

                    st.metric(
                        "Students",
                        branch_reports[
                            "University_ID"
                        ].nunique()
                    )

                with col3:

                    avg = branch_reports[
                        "Overall_Percent"
                    ].mean()

                    st.metric(
                        "Average Performance",
                        f"{avg:.1f}%"
                    )

                # -------------------------------------
                # STUDENT SELECTION
                # -------------------------------------

                student_ids = sorted(
                    branch_reports[
                        "University_ID"
                    ].astype(str).unique()
                )

                selected_student = st.selectbox(
                    "Select Student",
                    student_ids
                )

                student_data = branch_reports[
                    branch_reports[
                        "University_ID"
                    ].astype(str)
                    == str(selected_student)
                ].copy()

                if not student_data.empty:

                    student_name = student_data[
                        "Student_Name"
                    ].iloc[0]

                    st.header(
                        f"👤 {student_name}"
                    )

                    st.write(
                        f"**University ID:** "
                        f"{selected_student}"
                    )

                    st.write(
                        f"**Semester:** "
                        f"{student_data['Semester'].iloc[0]}"
                    )

                    st.write(
                        f"**Branch:** "
                        f"{teacher_branch}"
                    )

                    # ---------------------------------
                    # TABLE
                    # ---------------------------------

                    display_columns = [
                        "Subject",
                        "Attendance",
                        "Study_Hours",
                        "Internal_Mark",
                        "Assignment",
                        "Previous_Mark",
                        "Performance_Level",
                        "Overall_Percent",
                        "ML_Prediction",
                        "ML_Confidence",
                        "Cluster"
                    ]

                    st.dataframe(
                        student_data[
                            display_columns
                        ],
                        width="stretch"
                    )

                    # ---------------------------------
                    # AVERAGE
                    # ---------------------------------

                    overall_average = student_data[
                        "Overall_Percent"
                    ].mean()

                    st.metric(
                        "Student Overall Average",
                        f"{overall_average:.1f}%"
                    )

                    # ---------------------------------
                    # TEACHER REPORT
                    # ---------------------------------

                    subject_results = []

                    for _, row in student_data.iterrows():

                        level = str(
                            row["Performance_Level"]
                        )

                        if "Low" in level:

                            symbol = "🔴"

                        elif "Average" in level:

                            symbol = "🟠"

                        elif "Above" in level:

                            symbol = "🟡"

                        else:

                            symbol = "🟢"

                        recs = recommendations(
                            float(row["Attendance"]),
                            float(row["Study_Hours"]),
                            float(row["Internal_Mark"]),
                            float(row["Assignment"]),
                            float(row["Previous_Mark"])
                        )

                        subject_results.append({
                            "subject": row["Subject"],
                            "attendance": row["Attendance"],
                            "study_hours": row["Study_Hours"],
                            "internal": row["Internal_Mark"],
                            "assignment": row["Assignment"],
                            "previous": row["Previous_Mark"],
                            "level": level,
                            "symbol": symbol,
                            "overall": float(
                                row["Overall_Percent"]
                            ),
                            "recommendations": recs
                        })

                    ml_results = {
                        "prediction": str(
                            student_data[
                                "ML_Prediction"
                            ].iloc[0]
                        ),
                        "confidence": float(
                            student_data[
                                "ML_Confidence"
                            ].iloc[0]
                        ),
                        "cluster": int(
                            student_data[
                                "Cluster"
                            ].iloc[0]
                        ),
                        "accuracy": 0
                    }

                    teacher_html = create_html_report(
                        student_name,
                        selected_student,
                        str(
                            student_data[
                                "Semester"
                            ].iloc[0]
                        ),
                        teacher_branch,
                        subject_results,
                        ml_results
                    )

                    st.download_button(
                        label="📥 Download Student Progress Report",
                        data=teacher_html,
                        file_name=(
                            f"{selected_student}_"
                            "Teacher_Report.html"
                        ),
                        mime="text/html",
                        width="stretch"
                    )
