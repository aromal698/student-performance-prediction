import streamlit as st
import pandas as pd
import numpy as np
import os
import html

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# FILES
# ============================================================

DATA_FILE = "data/student_performance.csv"

REPORT_FILE = "data/student_reports.csv"


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]


# ============================================================
# BRANCH OPTIONS
# ============================================================

BRANCHES = [
    "Artificial Intelligence and Data Science",
    "Computer Science and Engineering",
    "Computer Science and Engineering (AI)",
    "Computer Science and Engineering (Data Science)",
    "Information Technology",
    "Cyber Security",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electronics and Instrumentation Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Automobile Engineering",
    "Biomedical Engineering",
    "Biotechnology and Biochemical Engineering",
    "Food Technology",
    "Robotics and Automation",
    "Industrial Engineering",
    "Chemical Engineering"
]


# ============================================================
# SUBJECTS FOR EACH BRANCH
# ============================================================

BRANCH_SUBJECTS = {

    "Artificial Intelligence and Data Science": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Artificial Intelligence",
        "Machine Learning",
        "Survey and Geomatics"
    ],

    "Computer Science and Engineering": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Operating Systems",
        "Computer Networks",
        "Software Engineering"
    ],

    "Computer Science and Engineering (AI)": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning"
    ],

    "Computer Science and Engineering (Data Science)": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Data Science",
        "Machine Learning",
        "Statistics"
    ],

    "Information Technology": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Computer Networks",
        "Web Technology",
        "Software Engineering"
    ],

    "Cyber Security": [
        "Mathematics",
        "Data Structures",
        "Computer Networks",
        "Cyber Security",
        "Cryptography",
        "Network Security"
    ],

    "Electronics and Communication Engineering": [
        "Engineering Mathematics",
        "Digital Electronics",
        "Analog Electronics",
        "Signals and Systems",
        "Communication Engineering",
        "Microprocessors"
    ],

    "Electrical and Electronics Engineering": [
        "Engineering Mathematics",
        "Circuit Theory",
        "Electrical Machines",
        "Power Systems",
        "Control Systems",
        "Power Electronics"
    ],

    "Electronics and Instrumentation Engineering": [
        "Engineering Mathematics",
        "Electronic Circuits",
        "Measurements",
        "Control Systems",
        "Instrumentation",
        "Sensors and Transducers"
    ],

    "Mechanical Engineering": [
        "Engineering Mathematics",
        "Engineering Mechanics",
        "Thermodynamics",
        "Fluid Mechanics",
        "Manufacturing Technology",
        "Machine Design"
    ],

    "Civil Engineering": [
        "Engineering Mathematics",
        "Engineering Mechanics",
        "Surveying",
        "Strength of Materials",
        "Fluid Mechanics",
        "Construction Technology"
    ],

    "Automobile Engineering": [
        "Engineering Mathematics",
        "Automobile Engineering",
        "Thermodynamics",
        "Vehicle Dynamics",
        "Automotive Electrical Systems",
        "Manufacturing Technology"
    ],

    "Biomedical Engineering": [
        "Engineering Mathematics",
        "Biomedical Instrumentation",
        "Human Anatomy",
        "Biomaterials",
        "Medical Imaging",
        "Biomedical Signal Processing"
    ],

    "Biotechnology and Biochemical Engineering": [
        "Engineering Mathematics",
        "Biochemistry",
        "Microbiology",
        "Bioprocess Engineering",
        "Genetic Engineering",
        "Biotechnology"
    ],

    "Food Technology": [
        "Engineering Mathematics",
        "Food Chemistry",
        "Food Microbiology",
        "Food Processing",
        "Food Engineering",
        "Food Safety"
    ],

    "Robotics and Automation": [
        "Engineering Mathematics",
        "Robotics",
        "Control Systems",
        "Sensors",
        "Automation",
        "Programming"
    ],

    "Industrial Engineering": [
        "Engineering Mathematics",
        "Operations Research",
        "Production Engineering",
        "Quality Management",
        "Industrial Management",
        "Manufacturing Systems"
    ],

    "Chemical Engineering": [
        "Engineering Mathematics",
        "Chemical Process Calculations",
        "Fluid Mechanics",
        "Heat Transfer",
        "Mass Transfer",
        "Chemical Reaction Engineering"
    ]
}


# ============================================================
# TEACHER ACCOUNTS
# ============================================================
#
# IMPORTANT:
# Edit these usernames/passwords according to your project.
#
# Each teacher is assigned to ONE branch.
#
# Format:
# username:
# {
#     "password": "...",
#     "branch": "..."
# }
# ============================================================

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


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .low-box {
        padding: 16px;
        border: 2px solid #d32f2f;
        border-radius: 12px;
        margin-bottom: 12px;
        background-color: #fff5f5;
    }

    .average-box {
        padding: 16px;
        border: 2px solid #f57c00;
        border-radius: 12px;
        margin-bottom: 12px;
        background-color: #fff8ef;
    }

    .above-box {
        padding: 16px;
        border: 2px solid #e0b400;
        border-radius: 12px;
        margin-bottom: 12px;
        background-color: #fffde7;
    }

    .good-box {
        padding: 16px;
        border: 2px solid #2e7d32;
        border-radius: 12px;
        margin-bottom: 12px;
        background-color: #f1f8f2;
    }

    .student-report {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cccccc;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            "Dataset not found. Please create "
            "data/student_performance.csv"
        )

    return pd.read_csv(DATA_FILE)


# ============================================================
# REPORT STORAGE
# ============================================================

def initialize_report_file():

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(REPORT_FILE):

        columns = [
            "University_ID",
            "Student_Name",
            "DOB_Year",
            "Semester",
            "Branch",
            "Subject",
            "Attendance",
            "Attendance_Mark",
            "Study_Hours",
            "Internal_Mark",
            "Assignment",
            "Previous_Mark",
            "Overall_Percent",
            "Performance_Level",
            "ML_Prediction",
            "Confidence",
            "Cluster"
        ]

        pd.DataFrame(
            columns=columns
        ).to_csv(
            REPORT_FILE,
            index=False
        )


def load_reports():

    initialize_report_file()

    try:

        return pd.read_csv(
            REPORT_FILE
        )

    except Exception:

        initialize_report_file()

        return pd.read_csv(
            REPORT_FILE
        )


def save_student_report(
    student_name,
    university_id,
    dob_year,
    semester,
    branch,
    results
):

    initialize_report_file()

    existing = load_reports()

    # Remove previous report for this student
    # so the newest report is displayed.

    if not existing.empty:

        existing = existing[
            existing["University_ID"].astype(str)
            != str(university_id)
        ]

    rows = []

    for subject, result in results.items():

        rows.append({

            "University_ID":
                university_id,

            "Student_Name":
                student_name,

            "DOB_Year":
                dob_year,

            "Semester":
                semester,

            "Branch":
                branch,

            "Subject":
                subject,

            "Attendance":
                result["attendance"],

            "Attendance_Mark":
                result["attendance_mark"],

            "Study_Hours":
                result["study_hours"],

            "Internal_Mark":
                result["internal"],

            "Assignment":
                result["assignment"],

            "Previous_Mark":
                result["previous"],

            "Overall_Percent":
                result["overall"],

            "Performance_Level":
                result["level"],

            "ML_Prediction":
                result["prediction"],

            "Confidence":
                result["confidence"],

            "Cluster":
                result["cluster"]
        })

    new_data = pd.DataFrame(rows)

    final_data = pd.concat(
        [
            existing,
            new_data
        ],
        ignore_index=True
    )

    final_data.to_csv(
        REPORT_FILE,
        index=False
    )


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    data = load_data()

    required = FEATURES + ["Performance"]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing dataset columns: "
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

    clusters = kmeans.fit_predict(
        X_scaled
    )

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

    y_pred = model.predict(
        X_test
    )

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
# ATTENDANCE MARK
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

    else:
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
        attendance_mark(attendance)
        / 5
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

def get_recommendations(
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
            "Discuss difficult topics with teachers."
        )

    elif level == "Average Performance":

        result.append(
            "Maintain regular revision and improve consistency."
        )

        result.append(
            "Practice additional questions before examinations."
        )

    elif level == "Above Average":

        result.append(
            "Small improvements in attendance and internal marks can improve performance."
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

    return list(
        dict.fromkeys(result)
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

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
# PREDICTION
# ============================================================

def predict_subject(
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
    model,
    scaler,
    kmeans
):

    model_input = prepare_model_input(
        attendance,
        study_hours,
        internal,
        assignment,
        previous
    )

    prediction = model.predict(
        model_input
    )[0]

    probabilities = model.predict_proba(
        model_input
    )[0]

    confidence = max(
        probabilities
    ) * 100

    scaled_input = scaler.transform(
        model_input
    )

    cluster = int(
        kmeans.predict(
            scaled_input
        )[0]
    )

    overall, level, symbol = get_performance_level(
        attendance,
        study_hours,
        internal,
        assignment,
        previous
    )

    suggestions = get_recommendations(
        attendance,
        study_hours,
        internal,
        assignment,
        previous,
        level
    )

    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "cluster":
            cluster,

        "overall":
            overall,

        "level":
            level,

        "symbol":
            symbol,

        "suggestions":
            suggestions,

        "attendance_mark":
            attendance_mark(attendance)
    }


# ============================================================
# CREATE HTML REPORT
# ============================================================

def create_progress_report(
    student_name,
    university_id,
    dob_year,
    semester,
    branch,
    results
):

    student_name = html.escape(
        str(student_name)
    )

    university_id = html.escape(
        str(university_id)
    )

    branch = html.escape(
        str(branch)
    )

    rows = ""

    for subject, result in results.items():

        subject_safe = html.escape(
            subject
        )

        suggestions = "<br>".join(
            [
                "• " + html.escape(item)
                for item in result["suggestions"]
            ]
        )

        rows += f"""

        <tr>

            <td>{subject_safe}</td>

            <td>
                {result["symbol"]}
                {result["level"]}
            </td>

            <td>
                {result["overall"]:.1f}%
            </td>

            <td>
                {result["attendance"]:.0f}%
            </td>

            <td>
                {result["attendance_mark"]}/5
            </td>

            <td>
                {result["internal"]:.0f}/40
            </td>

            <td>
                {result["previous"]:.0f}/60
            </td>

            <td>
                {result["assignment"]:.0f}/15
            </td>

            <td>
                {result["study_hours"]:.1f}/6
            </td>

            <td>
                {html.escape(str(result["prediction"]))}
            </td>

        </tr>

        <tr>

            <td colspan="10">

                <b>Suggestions:</b>

                <br>

                {suggestions}

            </td>

        </tr>
        """

    return f"""
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

.center {{
    text-align: center;
}}

.info {{
    width: 100%;
    border-collapse: collapse;
}}

.info td {{
    border: 1px solid #888;
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
    border: 1px solid #888;
    background: #f5f5f5;
}}

@media print {{

    body {{
        margin: 15px;
    }}

}}

</style>

</head>

<body>

<h1>
STUDENT PERFORMANCE PROGRESS REPORT
</h1>

<p class="center">
Student Performance Prediction System
</p>

<h2>
1. Student Information
</h2>

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
<td>DOB Year</td>
<td>{dob_year}</td>
</tr>

<tr>
<td>Semester</td>
<td>{semester}</td>
</tr>

<tr>
<td>Branch</td>
<td>{branch}</td>
</tr>

</table>


<h2>
2. Subject-wise Performance
</h2>

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


<h2>
3. Attendance Conversion
</h2>

<table class="report">

<tr>
<th>Attendance</th>
<th>Mark</th>
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


<h2>
4. Performance Indicators
</h2>

<table class="report">

<tr>
<th>Indicator</th>
<th>Level</th>
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

<b>Important:</b>

The performance indicator used by this project is a
project-defined academic monitoring scale. It is not
an official KTU grading scale.

The machine-learning prediction is intended to support
academic monitoring and should not replace teacher
evaluation.

</div>

</body>

</html>
"""


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        '<div class="main-title">'
        '🎓 Student Performance Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(
        "Choose Login"
    )

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

    st.title(
        "🎓 Student Login"
    )

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

    if st.button(
        "← Back"
    ):

        st.session_state.login_type = None

        st.rerun()


# ============================================================
# TEACHER LOGIN
# ============================================================

def teacher_login():

    st.title(
        "👨‍🏫 Teacher Login"
    )

    username = st.text_input(
        "Teacher Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        username = username.strip()

        if username in TEACHERS:

            account = TEACHERS[username]

            if password == account["password"]:

                st.session_state.logged_in = True

                st.session_state.role = "teacher"

                st.session_state.teacher_username = username

                st.session_state.teacher_branch = (
                    account["branch"]
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

        else:

            st.error(
                "Invalid username or password."
            )

    if st.button(
        "← Back"
    ):

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

    st.title(
        "🎓 Student Dashboard"
    )

    col1, col2 = st.columns([4, 1])

    with col1:

        st.success(
            "Student login successful."
        )

    with col2:

        if st.button(
            "Logout"
        ):

            st.session_state.clear()

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # STUDENT INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "1. Student Information"
    )

    c1, c2 = st.columns(2)

    with c1:

        student_name = st.text_input(
            "Student Name"
        )

        university_id = st.text_input(
            "University ID"
        )

    with c2:

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

        branch = st.selectbox(
            "Branch",
            BRANCHES
        )

    st.divider()

    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    subjects = BRANCH_SUBJECTS.get(
        branch,
        []
    )

    st.subheader(
        "2. Subject-wise Academic Details"
    )

    st.info(
        f"Subjects for {branch}"
    )

    subject_data = {}

    for subject in subjects:

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

            "attendance":
                attendance,

            "study_hours":
                study_hours,

            "internal":
                internal,

            "assignment":
                assignment,

            "previous":
                previous
        }

        st.divider()

    # --------------------------------------------------------
    # PREDICTION
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

        with st.spinner(
            "Analysing student performance..."
        ):

            for subject, values in subject_data.items():

                prediction = predict_subject(
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

        # ----------------------------------------------------
        # SAVE REPORT
        # ----------------------------------------------------

        save_student_report(
            student_name,
            university_id,
            st.session_state.dob_year,
            semester,
            branch,
            results
        )

        # Clear cached reports after writing
        st.session_state.results = results

        st.session_state.student_name = student_name

        st.session_state.university_id = university_id

        st.session_state.semester = semester

        st.session_state.branch = branch

        st.success(
            "Performance calculated and report saved successfully."
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
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

                <b>Confidence:</b>
                {result["confidence"]:.1f}%

                <br>

                <b>K-Means Cluster:</b>
                {result["cluster"]}

                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # OVERALL TABLE
        # ----------------------------------------------------

        st.subheader(
            "4. Overall Academic Information"
        )

        table = []

        for subject, result in results.items():

            table.append({

                "Subject":
                    subject,

                "Attendance":
                    f'{result["attendance"]:.0f}%',

                "Attendance Mark":
                    f'{result["attendance_mark"]}/5',

                "Internal":
                    f'{result["internal"]:.0f}/40',

                "Previous":
                    f'{result["previous"]:.0f}/60',

                "Assignment":
                    f'{result["assignment"]:.0f}/15',

                "Study Hours":
                    f'{result["study_hours"]:.1f}/6',

                "Prediction":
                    result["prediction"],

                "Performance":
                    result["level"]
            })

        st.dataframe(
            pd.DataFrame(table),
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
                f'{result["symbol"]} {subject} – {result["level"]}'
            ):

                for item in result["suggestions"]:

                    st.write(
                        "• " + item
                    )

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------

        st.subheader(
            "6. Download Progress Report"
        )

        html_report = create_progress_report(
            st.session_state.student_name,
            st.session_state.university_id,
            st.session_state.dob_year,
            st.session_state.semester,
            st.session_state.branch,
            results
        )

        st.download_button(
            "📥 Download Progress Report",
            data=html_report,
            file_name=(
                f'{st.session_state.university_id}'
                '_Progress_Report.html'
            ),
            mime="text/html",
            width="stretch"
        )

        st.info(
            "Open the downloaded report in a browser. "
            "Use Print → Save as PDF to create a PDF copy."
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

    teacher_branch = (
        st.session_state.teacher_branch
    )

    teacher_username = (
        st.session_state.teacher_username
    )

    st.title(
        "👨‍🏫 Teacher Dashboard"
    )

    st.success(
        f"Logged in as: {teacher_username}"
    )

    st.info(
        f"Teaching Branch: {teacher_branch}"
    )

    if st.button(
        "Logout"
    ):

        st.session_state.clear()

        st.rerun()

    st.divider()

    # ========================================================
    # ONLY THIS TEACHER'S BRANCH
    # ========================================================

    reports = load_reports()

    if reports.empty:

        st.warning(
            "No student progress reports have been submitted yet."
        )

    else:

        branch_reports = reports[
            reports["Branch"].astype(str)
            == str(teacher_branch)
        ].copy()

        # ----------------------------------------------------
        # STUDENT LIST
        # ----------------------------------------------------

        st.subheader(
            "👨‍🎓 Students in Your Branch"
        )

        if branch_reports.empty:

            st.info(
                "No student reports are available for your branch."
            )

        else:

            students = (
                branch_reports[
                    [
                        "University_ID",
                        "Student_Name",
                        "Semester",
                        "Branch"
                    ]
                ]
                .drop_duplicates(
                    subset=["University_ID"]
                )
                .sort_values(
                    "Student_Name"
                )
            )

            st.dataframe(
                students,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # SELECT STUDENT
            # ------------------------------------------------

            student_options = (
                students[
                    "University_ID"
                ]
                .astype(str)
                .tolist()
            )

            selected_student = st.selectbox(
                "Select Student",
                student_options
            )

            selected_data = branch_reports[
                branch_reports["University_ID"].astype(str)
                == str(selected_student)
            ].copy()

            if not selected_data.empty:

                student_name = (
                    selected_data.iloc[0]
                    ["Student_Name"]
                )

                semester = (
                    selected_data.iloc[0]
                    ["Semester"]
                )

                branch = (
                    selected_data.iloc[0]
                    ["Branch"]
                )

                # ------------------------------------------------
                # STUDENT PROFILE
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "📋 Student Progress Report"
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "Student",
                        str(student_name)
                    )

                with c2:

                    st.metric(
                        "University ID",
                        str(selected_student)
                    )

                with c3:

                    st.metric(
                        "Semester",
                        str(semester)
                    )

                with c4:

                    st.metric(
                        "Branch",
                        str(branch)
                    )

                # ------------------------------------------------
                # SUBJECT DETAILS
                # ------------------------------------------------

                st.subheader(
                    "📚 Subject-wise Details"
                )

                for _, row in selected_data.iterrows():

                    level = str(
                        row["Performance_Level"]
                    )

                    if level == "Low Performance":

                        symbol = "🔴"

                    elif level == "Average Performance":

                        symbol = "🟠"

                    elif level == "Above Average":

                        symbol = "🟡"

                    else:

                        symbol = "🟢"

                    with st.expander(
                        f'{symbol} {row["Subject"]} – {level}'
                    ):

                        c1, c2, c3 = st.columns(3)

                        with c1:

                            st.write(
                                "**Attendance:**",
                                f'{row["Attendance"]:.0f}%'
                            )

                            st.write(
                                "**Attendance Mark:**",
                                f'{row["Attendance_Mark"]}/5'
                            )

                            st.write(
                                "**Study Hours:**",
                                f'{row["Study_Hours"]:.1f}/6'
                            )

                        with c2:

                            st.write(
                                "**Internal:**",
                                f'{row["Internal_Mark"]:.0f}/40'
                            )

                            st.write(
                                "**Assignment:**",
                                f'{row["Assignment"]:.0f}/15'
                            )

                            st.write(
                                "**Previous Mark:**",
                                f'{row["Previous_Mark"]:.0f}/60'
                            )

                        with c3:

                            st.write(
                                "**Overall:**",
                                f'{row["Overall_Percent"]:.1f}%'
                            )

                            st.write(
                                "**ML Prediction:**",
                                str(row["ML_Prediction"])
                            )

                            st.write(
                                "**Confidence:**",
                                f'{row["Confidence"]:.1f}%'
                            )

                            st.write(
                                "**K-Means Cluster:**",
                                int(row["Cluster"])
                            )

                # ------------------------------------------------
                # FULL REPORT TABLE
                # ------------------------------------------------

                st.subheader(
                    "📊 Complete Student Report"
                )

                display_data = selected_data.copy()

                display_data = display_data[
                    [
                        "Subject",
                        "Attendance",
                        "Attendance_Mark",
                        "Study_Hours",
                        "Internal_Mark",
                        "Assignment",
                        "Previous_Mark",
                        "Overall_Percent",
                        "Performance_Level",
                        "ML_Prediction",
                        "Confidence",
                        "Cluster"
                    ]
                ]

                st.dataframe(
                    display_data,
                    width="stretch",
                    hide_index=True
                )

                # ------------------------------------------------
                # DOWNLOAD STUDENT REPORT
                # ------------------------------------------------

                student_results = {}

                for _, row in selected_data.iterrows():

                    level = str(
                        row["Performance_Level"]
                    )

                    if level == "Low Performance":
                        symbol = "🔴"

                    elif level == "Average Performance":
                        symbol = "🟠"

                    elif level == "Above Average":
                        symbol = "🟡"

                    else:
                        symbol = "🟢"

                    student_results[
                        str(row["Subject"])
                    ] = {

                        "attendance":
                            float(row["Attendance"]),

                        "attendance_mark":
                            int(row["Attendance_Mark"]),

                        "study_hours":
                            float(row["Study_Hours"]),

                        "internal":
                            float(row["Internal_Mark"]),

                        "assignment":
                            float(row["Assignment"]),

                        "previous":
                            float(row["Previous_Mark"]),

                        "overall":
                            float(row["Overall_Percent"]),

                        "level":
                            level,

                        "symbol":
                            symbol,

                        "prediction":
                            str(row["ML_Prediction"]),

                        "confidence":
                            float(row["Confidence"]),

                        "cluster":
                            int(row["Cluster"]),

                        "suggestions":
                            []
                    }

                student_html = create_progress_report(
                    str(student_name),
                    str(selected_student),
                    int(
                        selected_data.iloc[0]
                        ["DOB_Year"]
                    ),
                    str(semester),
                    str(branch),
                    student_results
                )

                st.download_button(
                    "📥 Download Student Progress Report",
                    data=student_html,
                    file_name=(
                        f"{selected_student}"
                        "_Progress_Report.html"
                    ),
                    mime="text/html",
                    width="stretch"
                )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Model Performance"
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

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

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

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

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

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "🌳 Random Forest Feature Importance"
    )

    importance_df = pd.DataFrame({

        "Feature":
            FEATURES,

        "Importance":
            model.feature_importances_
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

    # ========================================================
    # K-MEANS
    # ========================================================

    st.subheader(
        "🔵 K-Means Cluster Distribution"
    )

    cluster_counts = (
        data["Cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_df = pd.DataFrame({

        "Cluster":
            cluster_counts.index.astype(str),

        "Students":
            cluster_counts.values
    })

    st.bar_chart(
        cluster_df.set_index(
            "Cluster"
        )
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
# RUN
# ============================================================

if __name__ == "__main__":

    main()
