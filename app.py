import os
import re
import io
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_TITLE = "STUDENT PERFORMANCE PREDICTION SYSTEM"
APP_SUBTITLE = "AI-Powered Academic Performance Monitoring System"
COURSE_TITLE = "B.Tech – Artificial Intelligence and Data Science"

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "student_records.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# B.TECH BRANCHES
# ============================================================
# Configurable list for the project.
# Update names according to the exact branches offered by
# your college / current KTU scheme.

BRANCHES = [
    "Civil Engineering",
    "Computer Science and Engineering",
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Chemical Engineering",
    "Biotechnology and Biochemical Engineering",
    "Food Technology",
    "Production Engineering",
    "Automobile Engineering",
    "Industrial Engineering",
    "Biomedical Engineering",
    "Aeronautical Engineering",
    "Applied Electronics and Instrumentation Engineering",
    "Electronics and Biomedical Engineering",
    "Electronics and Computer Engineering",
    "Computer Science and Engineering (Artificial Intelligence)",
    "Computer Science and Engineering (Data Science)",
    "Computer Science and Engineering (Cyber Security)",
    "Robotics and Automation",
    "Mechatronics Engineering",
]

# ============================================================
# SEMESTER SUBJECTS
# ============================================================

SEMESTERS = {

    "S1": [
        "Mathematics for Information Science-I",
        "Physics for Information Science",
        "Engineering Graphics",
        "Introduction to Electrical & Electronics Engineering",
        "Introduction to Computing",
        "Health and Wellness",
    ],

    "S2": [
        "Mathematics for Information Science-II",
        "Chemistry for Information Science",
        "Programming in C",
        "Engineering Mechanics",
        "Design Thinking and Product Development",
        "Professional Communication",
    ],

    "S3": [
        "Mathematics for Information Science-III",
        "Foundations of Artificial Intelligence",
        "Data Structures and Algorithms",
        "Introduction to Data Science",
        "Digital Electronics & Logic Design",
        "Economics / Engineering Ethics",
    ],

    "S4": [
        "Mathematics for Information Science-IV",
        "Database Management Systems",
        "Operating Systems",
        "Computer Organization and Architecture",
        "Object Oriented Programming",
        "Constitution of India / Professional Ethics",
    ],

    "S5": [
        "Machine Learning",
        "Computer Networks",
        "Web Technologies",
        "Software Engineering",
        "Artificial Intelligence",
        "Program Elective-I",
    ],

    "S6": [
        "Deep Learning",
        "Big Data Analytics",
        "Data Mining",
        "Cloud Computing",
        "Natural Language Processing",
        "Program Elective-II",
    ],

    "S7": [
        "Major Project Phase-I",
        "Program Elective-III",
        "Program Elective-IV",
        "Open Elective",
        "Seminar",
    ],

    "S8": [
        "Major Project Phase-II",
        "Program Elective-V",
        "Program Elective-VI",
        "Comprehensive Viva",
        "Internship / Industrial Training",
    ],
}

# ============================================================
# TUTOR ACCOUNTS
# ============================================================

TUTOR_ACCOUNTS = {
    "teacher_aids": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Data Science",
    },

    "teacher_cse": {
        "password": "ktucse",
        "branch": "Computer Science and Engineering",
    },

    "teacher_it": {
        "password": "ktuit",
        "branch": "Information Technology",
    },

    "teacher_ece": {
        "password": "ktuece",
        "branch": "Electronics and Communication Engineering",
    },

    "teacher_eee": {
        "password": "ktueee",
        "branch": "Electrical and Electronics Engineering",
    },

    "teacher_me": {
        "password": "ktume",
        "branch": "Mechanical Engineering",
    },

    "teacher_ce": {
        "password": "ktuce",
        "branch": "Civil Engineering",
    },

    "teacher_it2": {
        "password": "ktuit2",
        "branch": "Information Technology",
    },
}

# ============================================================
# DATABASE COLUMNS
# ============================================================

COLUMNS = [
    "Student_Name",
    "Username",
    "University_ID",
    "Branch",
    "Semester",
    "Subject",
    "Attendance",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark",
    "Study_Hours",
    "Performance",
    "Performance_Percent",
    "KMeans_Cluster",
]

# ============================================================
# FUTURISTIC 3D CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 20% 20%,
            rgba(0, 140, 255, 0.20),
            transparent 25%
        ),
        radial-gradient(
            circle at 80% 30%,
            rgba(120, 70, 255, 0.20),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 90%,
            rgba(0, 220, 190, 0.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #020617 0%,
            #07122c 50%,
            #030817 100%
        );

    color: white;
}

/* =========================================================
   ANIMATED GRID
   ========================================================= */

.stApp::before {
    content: "";
    position: fixed;
    left: -20%;
    bottom: -30%;
    width: 140%;
    height: 70%;

    background-image:
        linear-gradient(
            rgba(0, 190, 255, 0.08) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(0, 190, 255, 0.08) 1px,
            transparent 1px
        );

    background-size: 55px 55px;

    transform:
        perspective(450px)
        rotateX(65deg);

    transform-origin: center bottom;

    animation: gridMove 10s linear infinite;

    pointer-events: none;
    z-index: 0;
}

@keyframes gridMove {

    0% {
        transform:
            perspective(450px)
            rotateX(65deg)
            translateY(0);
    }

    100% {
        transform:
            perspective(450px)
            rotateX(65deg)
            translateY(55px);
    }
}

/* =========================================================
   FLOATING LIGHT
   ========================================================= */

.stApp::after {
    content: "";
    position: fixed;

    width: 320px;
    height: 320px;

    top: 8%;
    right: 5%;

    background:
        radial-gradient(
            circle,
            rgba(0, 200, 255, 0.18),
            transparent 65%
        );

    filter: blur(20px);

    animation: floatingLight 6s ease-in-out infinite;

    pointer-events: none;
}

@keyframes floatingLight {

    0%, 100% {
        transform: translateY(0) scale(1);
    }

    50% {
        transform: translateY(30px) scale(1.15);
    }
}

/* =========================================================
   CONTENT
   ========================================================= */

.block-container {
    position: relative;
    z-index: 2;

    padding-top: 2rem;
    padding-bottom: 3rem;

    max-width: 1450px;
}

/* =========================================================
   LOGIN CARD
   ========================================================= */

.login-card {

    position: relative;

    max-width: 900px;

    margin:
        55px auto 30px auto;

    padding:
        55px 35px;

    text-align: center;

    border-radius: 30px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.11),
            rgba(255,255,255,0.035)
        );

    border:
        1px solid rgba(255,255,255,0.16);

    box-shadow:
        0 30px 80px rgba(0,0,0,0.40),
        inset 0 1px rgba(255,255,255,0.12);

    backdrop-filter: blur(18px);

    overflow: hidden;
}

/* glowing border */

.login-card::before {

    content: "";

    position: absolute;

    inset: -2px;

    border-radius: 32px;

    background:
        linear-gradient(
            120deg,
            transparent,
            rgba(0,200,255,0.5),
            transparent,
            rgba(130,80,255,0.5),
            transparent
        );

    opacity: 0.25;

    animation:
        borderGlow 5s linear infinite;

    pointer-events: none;
}

@keyframes borderGlow {

    0% {
        transform: translateX(-70%);
    }

    100% {
        transform: translateX(70%);
    }
}

/* =========================================================
   TITLE
   ========================================================= */

.login-title {

    position: relative;

    font-size: clamp(28px, 4vw, 48px);

    font-weight: 900;

    letter-spacing: 1.5px;

    line-height: 1.2;

    color: white;

    text-shadow:
        0 0 20px rgba(0,200,255,0.30);

    margin-bottom: 15px;
}

/* =========================================================
   SUBTITLE
   ========================================================= */

.login-subtitle {

    position: relative;

    font-size: 17px;

    line-height: 1.6;

    color: #b7c3dd !important;

    margin-top: 5px;

    margin-bottom: 10px;

    font-weight: 400;
}

/* =========================================================
   COURSE
   ========================================================= */

.login-course {

    position: relative;

    color: #7dd3fc !important;

    font-size: 14px;

    font-weight: 700;

    letter-spacing: 0.8px;

    margin-top: 12px;
}

/* =========================================================
   3D GRADUATION ICON
   ========================================================= */

.graduation-icon {

    font-size: 70px;

    display: inline-block;

    margin-bottom: 20px;

    filter:
        drop-shadow(
            0 0 25px
            rgba(0,200,255,0.45)
        );

    animation:
        graduationFloat
        3.5s
        ease-in-out
        infinite;
}

@keyframes graduationFloat {

    0%, 100% {
        transform:
            translateY(0)
            rotateY(0deg);
    }

    50% {
        transform:
            translateY(-14px)
            rotateY(12deg);
    }
}

/* =========================================================
   GLASS CARDS
   ========================================================= */

.glass-card {

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.035)
        );

    border:
        1px solid rgba(255,255,255,0.11);

    border-radius: 22px;

    padding: 25px;

    margin: 12px 0;

    box-shadow:
        0 15px 45px rgba(0,0,0,0.25);

    backdrop-filter: blur(15px);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease;
}

.glass-card:hover {

    transform:
        translateY(-5px);

    border-color:
        rgba(0,200,255,0.35);
}

/* =========================================================
   DASHBOARD
   ========================================================= */

.dashboard-title {

    font-size: 30px;

    font-weight: 800;

    color: white;
}

.small-muted {

    color: #96a4c3 !important;

    font-size: 14px;
}

/* =========================================================
   SUBJECT CARD
   ========================================================= */

.subject-card {

    background:
        rgba(255,255,255,0.055);

    border:
        1px solid rgba(255,255,255,0.09);

    border-radius: 18px;

    padding: 18px;

    margin:
        10px 0;

    box-shadow:
        0 10px 30px
        rgba(0,0,0,0.18);
}

/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {

    border-radius: 13px;

    font-weight: 750;

    border:
        1px solid
        rgba(255,255,255,0.14);

    background:
        rgba(255,255,255,0.07);

    transition:
        all 0.25s ease;
}

.stButton > button:hover {

    transform:
        translateY(-2px);

    border-color:
        rgba(0,200,255,0.45);

    box-shadow:
        0 8px 25px
        rgba(0,180,255,0.18);
}

/* =========================================================
   METRICS
   ========================================================= */

div[data-testid="stMetric"] {

    background:
        rgba(255,255,255,0.055);

    padding: 17px;

    border-radius: 17px;

    border:
        1px solid
        rgba(255,255,255,0.08);
}

/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    text-align: center;

    color: #6f7d9c !important;

    margin-top: 40px;

    font-size: 13px;

    line-height: 1.7;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "home",

    "student_logged_in": False,
    "student_name": "",
    "student_username": "",
    "student_id": "",
    "student_branch": "",

    "tutor_logged_in": False,
    "tutor_username": "",
    "tutor_branch": "",
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATABASE
# ============================================================

def empty_database():

    return pd.DataFrame(
        columns=COLUMNS
    )


def load_data():

    if not os.path.exists(DATA_FILE):

        return empty_database()

    try:

        df = pd.read_csv(DATA_FILE)

    except Exception:

        return empty_database()

    # Add missing columns
    for col in COLUMNS:

        if col not in df.columns:

            if col == "Semester":
                df[col] = "S3"

            elif col == "Branch":
                df[col] = (
                    "Artificial Intelligence "
                    "and Data Science"
                )

            elif col == "Performance":
                df[col] = ""

            elif col == "Performance_Percent":
                df[col] = 0.0

            elif col == "KMeans_Cluster":
                df[col] = -1

            else:
                df[col] = ""

    df = df[COLUMNS]

    df["Semester"] = (
        df["Semester"]
        .fillna("S3")
        .astype(str)
    )

    numeric_cols = [
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
        "Performance_Percent",
        "KMeans_Cluster",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    return df


def save_data(df):

    for col in COLUMNS:

        if col not in df.columns:
            df[col] = ""

    df = df[COLUMNS]

    df.to_csv(
        DATA_FILE,
        index=False,
    )


# ============================================================
# PERFORMANCE CALCULATION
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


def calculate_performance(
    attendance,
    internal,
    assignment,
    previous,
    study_hours,
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

    study_percent = min(
        study_hours / 6 * 100,
        100,
    )

    overall = np.mean(
        [
            attendance_percent,
            internal_percent,
            assignment_percent,
            previous_percent,
            study_percent,
        ]
    )

    if overall < 50:

        level = "Low Performance"

    elif overall < 65:

        level = "Average Performance"

    elif overall < 80:

        level = "Above Average Performance"

    else:

        level = "Good Performance"

    return round(float(overall), 2), level


def performance_color(level):

    if "Low" in level:
        return "#ff4d4d"

    if level == "Average Performance":
        return "#ff9f43"

    if "Above Average" in level:
        return "#ffd93d"

    return "#35d07f"


def performance_emoji(level):

    if "Low" in level:
        return "😟"

    if level == "Average Performance":
        return "🟠"

    if "Above Average" in level:
        return "🟡"

    return "🎉"


# ============================================================
# PASSWORD VALIDATION
# ============================================================

def valid_student_password(password):

    if not isinstance(password, str):
        return False

    if not re.fullmatch(
        r"BTECH(20\d{2})",
        password,
    ):
        return False

    year = int(password[-4:])

    return 2000 <= year <= 2022


# ============================================================
# K-MEANS
# ============================================================

def run_kmeans(df):

    if df.empty:

        return df, None

    features = [
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
    ]

    work = df.copy()

    for col in features:

        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    work = work.dropna(
        subset=features
    )

    if len(work) < 3:

        result = df.copy()

        result["KMeans_Cluster"] = -1

        return result, None

    X = work[features].values

    n_clusters = min(
        3,
        len(work),
    )

    try:

        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X)

        result = df.copy()

        result["KMeans_Cluster"] = -1

        result.loc[
            work.index,
            "KMeans_Cluster",
        ] = labels

        score = None

        if len(set(labels)) > 1:

            score = silhouette_score(
                X,
                labels,
            )

        return result, score

    except Exception:

        result = df.copy()

        result["KMeans_Cluster"] = -1

        return result, None


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    # --------------------------------------------------------
    # CORRECT HTML
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="login-card">

            <div class="graduation-icon">
                🎓
            </div>

            <div class="login-title">
                STUDENT PERFORMANCE<br>
                PREDICTION SYSTEM
            </div>

            <div class="login-subtitle">
                AI-Powered Academic Performance Monitoring System
            </div>

            <div class="login-course">
                B.TECH • ARTIFICIAL INTELLIGENCE AND DATA SCIENCE
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h2 style='text-align:center;'>Choose Login</h2>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        2,
        gap="large",
    )

    # ========================================================
    # STUDENT CARD
    # ========================================================

    with col1:

        st.markdown(
            """
            <div class="glass-card">

                <div style="
                    font-size:45px;
                    text-align:center;
                ">
                    🎓
                </div>

                <h2 style="text-align:center;">
                    Student
                </h2>

                <p style="text-align:center;">
                    View semester-wise and
                    subject-wise academic performance,
                    study progress and reports.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "🎓 STUDENT LOGIN",
            key="student_home_button",
            width="stretch",
        ):

            st.session_state["page"] = (
                "student_login"
            )

            st.rerun()

    # ========================================================
    # TUTOR CARD
    # ========================================================

    with col2:

        st.markdown(
            """
            <div class="glass-card">

                <div style="
                    font-size:45px;
                    text-align:center;
                ">
                    👨‍🏫
                </div>

                <h2 style="text-align:center;">
                    Tutor
                </h2>

                <p style="text-align:center;">
                    Manage student records,
                    analyse performance,
                    apply K-Means clustering
                    and generate PDF reports.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "👨‍🏫 TUTOR LOGIN",
            key="tutor_home_button",
            width="stretch",
        ):

            st.session_state["page"] = (
                "tutor_login"
            )

            st.rerun()

    st.markdown(
        """
        <div class="footer">

            STUDENT PERFORMANCE PREDICTION SYSTEM<br>

            B.Tech – Artificial Intelligence and Data Science<br>

            AI-Based Academic Performance Monitoring Project

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.markdown(
        """
        <div class="login-card">

            <div class="graduation-icon">
                🎓
            </div>

            <div class="login-title">
                STUDENT LOGIN
            </div>

            <div class="login-subtitle">
                Access your academic performance dashboard
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        username = st.text_input(
            "Username",
            key="student_username_input",
        )

        university_id = st.text_input(
            "University ID",
            key="student_id_input",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="student_password_input",
        )

        st.caption(
            "Demo password format: BTECH2000 – BTECH2022"
        )

        if st.button(
            "LOGIN",
            key="student_login_button",
            width="stretch",
        ):

            if not username.strip():

                st.error(
                    "Enter Username."
                )

                return

            if not university_id.strip():

                st.error(
                    "Enter University ID."
                )

                return

            if not valid_student_password(
                password
            ):

                st.error(
                    "Invalid password. "
                    "Use BTECH2000 to BTECH2022."
                )

                return

            df = load_data()

            if df.empty:

                st.error(
                    "No student records available."
                )

                return

            records = df[
                (
                    df["Username"]
                    .astype(str)
                    .str.lower()
                    == username.strip().lower()
                )
                &
                (
                    df["University_ID"]
                    .astype(str)
                    .str.lower()
                    == university_id.strip().lower()
                )
            ]

            if records.empty:

                st.error(
                    "Student record not found."
                )

                return

            row = records.iloc[0]

            st.session_state[
                "student_logged_in"
            ] = True

            st.session_state[
                "student_name"
            ] = str(
                row["Student_Name"]
            )

            st.session_state[
                "student_username"
            ] = str(
                row["Username"]
            )

            st.session_state[
                "student_id"
            ] = str(
                row["University_ID"]
            )

            st.session_state[
                "student_branch"
            ] = str(
                row["Branch"]
            )

            st.session_state["page"] = (
                "student_dashboard"
            )

            st.rerun()

    st.divider()

    if st.button(
        "← Back",
        key="student_back",
    ):

        st.session_state["page"] = "home"

        st.rerun()


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    st.markdown(
        """
        <div class="login-card">

            <div class="graduation-icon">
                👨‍🏫
            </div>

            <div class="login-title">
                TUTOR LOGIN
            </div>

            <div class="login-subtitle">
                Academic data management and analysis
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        username = st.text_input(
            "Tutor Username",
            key="tutor_username_input",
        )

        password = st.text_input(
            "Tutor Password",
            type="password",
            key="tutor_password_input",
        )

        if st.button(
            "LOGIN",
            key="tutor_login_button",
            width="stretch",
        ):

            account = TUTOR_ACCOUNTS.get(
                username.strip()
            )

            if account is None:

                st.error(
                    "Invalid tutor username."
                )

                return

            if password != account["password"]:

                st.error(
                    "Invalid tutor password."
                )

                return

            st.session_state[
                "tutor_logged_in"
            ] = True

            st.session_state[
                "tutor_username"
            ] = username.strip()

            st.session_state[
                "tutor_branch"
            ] = account["branch"]

            st.session_state["page"] = (
                "tutor_dashboard"
            )

            st.rerun()

    st.divider()

    if st.button(
        "← Back",
        key="tutor_back",
    ):

        st.session_state["page"] = "home"

        st.rerun()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    if not st.session_state[
        "student_logged_in"
    ]:

        st.session_state["page"] = (
            "student_login"
        )

        st.rerun()

    df = load_data()

    student_id = st.session_state[
        "student_id"
    ]

    student_name = st.session_state[
        "student_name"
    ]

    branch = st.session_state[
        "student_branch"
    ]

    student_df = df[
        df["University_ID"]
        .astype(str)
        == str(student_id)
    ].copy()

    # --------------------------------------------------------
    # ONLY STUDENT NAME
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="glass-card">

            <div class="dashboard-title">
                🎓 {student_name}
            </div>

            <div class="small-muted">
                Student Dashboard
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        semester = st.selectbox(
            "Select Semester",
            list(SEMESTERS.keys()),
            key="student_semester",
        )

    with col2:

        st.write("")

        if st.button(
            "🚪 Logout",
            key="student_logout",
        ):

            st.session_state[
                "student_logged_in"
            ] = False

            st.session_state["page"] = (
                "home"
            )

            st.rerun()

    semester_df = student_df[
        student_df["Semester"]
        .astype(str)
        == semester
    ].copy()

    if semester_df.empty:

        st.info(
            f"No records available for {semester}."
        )

        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    avg = semester_df[
        "Performance_Percent"
    ].mean()

    if avg < 50:

        overall = "Low Performance"

    elif avg < 65:

        overall = "Average Performance"

    elif avg < 80:

        overall = "Above Average Performance"

    else:

        overall = "Good Performance"

    st.markdown(
        "### 📊 Semester Overview"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Semester",
            semester,
        )

    with c2:
        st.metric(
            "Average",
            f"{avg:.2f}%",
        )

    with c3:
        st.metric(
            "Performance",
            overall,
        )

    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    st.markdown(
        "### 📚 Subject-wise Performance"
    )

    for subject in SEMESTERS[semester]:

        rows = semester_df[
            semester_df["Subject"]
            .astype(str)
            == subject
        ]

        if rows.empty:
            continue

        row = rows.iloc[0]

        level = str(
            row["Performance"]
        )

        color = performance_color(
            level
        )

        emoji = performance_emoji(
            level
        )

        st.markdown(
            f"""
            <div class="subject-card">

                <h3>
                    {subject}
                </h3>

                <span style="
                    color:{color};
                    font-size:24px;
                ">
                    ●
                </span>

                <b>
                    {emoji} {level}
                </b>

                &nbsp;&nbsp;

                <b>
                    {float(row["Performance_Percent"]):.2f}%
                </b>

            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.write(
                f"Attendance: "
                f"{float(row['Attendance']):.1f}%"
            )

        with c2:
            st.write(
                f"Internal: "
                f"{float(row['Internal_Mark']):.1f}/40"
            )

        with c3:
            st.write(
                f"Assignment: "
                f"{float(row['Assignment']):.1f}/15"
            )

        with c4:
            st.write(
                f"Previous: "
                f"{float(row['Previous_Mark']):.1f}/60"
            )

        with c5:
            st.write(
                f"Study: "
                f"{float(row['Study_Hours']):.1f} h"
            )

        st.divider()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    st.markdown(
        "### 📄 Progress Report"
    )

    pdf = create_pdf_report(
        student_name,
        student_id,
        branch,
        semester,
        semester_df,
    )

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        student_name,
    )

    st.download_button(
        "📥 Download Progress Report PDF",
        data=pdf,
        file_name=(
            f"{safe_name}_"
            f"{semester}_"
            "Progress_Report.pdf"
        ),
        mime="application/pdf",
        width="stretch",
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    student_name,
    university_id,
    branch,
    semester,
    df,
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Student Progress Report",
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor(
            "#163a63"
        ),
    )

    subtitle = ParagraphStyle(
        "subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(
            "#555555"
        ),
    )

    heading = ParagraphStyle(
        "heading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor(
            "#163a63"
        ),
    )

    normal = ParagraphStyle(
        "normal",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
    )

    story = []

    # --------------------------------------------------------
    # REPORT HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            APP_TITLE,
            title,
        )
    )

    story.append(
        Paragraph(
            APP_SUBTITLE,
            subtitle,
        )
    )

    story.append(
        Paragraph(
            COURSE_TITLE,
            subtitle,
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # --------------------------------------------------------
    # STUDENT DETAILS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. STUDENT DETAILS",
            heading,
        )
    )

    details = [
        [
            "Student Name",
            student_name,
        ],
        [
            "University ID",
            university_id,
        ],
        [
            "Branch",
            branch,
        ],
        [
            "Semester",
            semester,
        ],
    ]

    table = Table(
        details,
        colWidths=[
            50 * mm,
            120 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#eaf2f8"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(1, 12)
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. OVERALL PERFORMANCE",
            heading,
        )
    )

    average = float(
        df[
            "Performance_Percent"
        ].mean()
    )

    if average < 50:

        level = "Low Performance"

    elif average < 65:

        level = "Average Performance"

    elif average < 80:

        level = "Above Average Performance"

    else:

        level = "Good Performance"

    overall_table = Table(
        [
            [
                "Overall Percentage",
                f"{average:.2f}%",
            ],
            [
                "Overall Performance",
                level,
            ],
        ],
        colWidths=[
            70 * mm,
            100 * mm,
        ],
    )

    overall_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#edf4ff"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        overall_table
    )

    story.append(
        Spacer(1, 12)
    )

    # --------------------------------------------------------
    # SUBJECT REPORT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. SUBJECT-WISE PERFORMANCE",
            heading,
        )
    )

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        subject = str(
            row["Subject"]
        )

        performance = str(
            row["Performance"]
        )

        percentage = float(
            row["Performance_Percent"]
        )

        cluster = str(
            row["KMeans_Cluster"]
        )

        attendance = float(
            row["Attendance"]
        )

        internal = float(
            row["Internal_Mark"]
        )

        assignment = float(
            row["Assignment"]
        )

        previous = float(
            row["Previous_Mark"]
        )

        study = float(
            row["Study_Hours"]
        )

        att_mark = attendance_mark(
            attendance
        )

        subject_table = Table(
            [
                [
                    "Subject",
                    subject,
                ],
                [
                    "Attendance",
                    f"{attendance:.1f}%",
                ],
                [
                    "Attendance Mark",
                    f"{att_mark}/5",
                ],
                [
                    "Study Hours",
                    f"{study:.1f}/6 hours",
                ],
                [
                    "Internal",
                    f"{internal:.1f}/40",
                ],
                [
                    "Assignment",
                    f"{assignment:.1f}/15",
                ],
                [
                    "Previous Mark",
                    f"{previous:.1f}/60",
                ],
                [
                    "Performance",
                    performance,
                ],
                [
                    "Performance %",
                    f"{percentage:.2f}%",
                ],
                [
                    "K-Means Cluster",
                    cluster,
                ],
            ],
            colWidths=[
                65 * mm,
                105 * mm,
            ],
        )

        subject_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.HexColor(
                            "#edf4ff"
                        ),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (0, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8.5,
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(
            Paragraph(
                f"{i}. {subject}",
                heading,
            )
        )

        story.append(
            subject_table
        )

        story.append(
            Spacer(1, 6)
        )

        story.append(
            Paragraph(
                f"<b>Performance Level:</b> "
                f"{performance} "
                f"({percentage:.2f}%)",
                normal,
            )
        )

        if "Low" in performance:

            advice = (
                "Focus on attendance, regular study, "
                "internal examination preparation, "
                "assignments and revision."
            )

        elif "Average" in performance:

            advice = (
                "Maintain consistent study, "
                "improve attendance and "
                "strengthen weak subjects."
            )

        elif "Above Average" in performance:

            advice = (
                "Good progress. Continue regular "
                "revision and improve weaker components."
            )

        else:

            advice = (
                "Excellent performance! "
                "Keep up the good work."
            )

        story.append(
            Paragraph(
                f"<b>Improvement / Guidance:</b> "
                f"{advice}",
                normal,
            )
        )

        story.append(
            Spacer(1, 10)
        )

    # --------------------------------------------------------
    # CONVERSION
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "4. ASSESSMENT CONVERSION",
            heading,
        )
    )

    conversion = Table(
        [
            ["Component", "Maximum"],
            ["Attendance", "100%"],
            ["Attendance Converted Mark", "5"],
            ["Internal Mark", "40"],
            ["Assignment", "15"],
            ["Previous Mark", "60"],
            ["Study Hours", "6 hours/day"],
        ],
        colWidths=[
            100 * mm,
            70 * mm,
        ],
    )

    conversion.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#163a63"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        conversion
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "<b>Attendance conversion:</b> "
            "90–100% = 5 marks, "
            "80–89% = 4 marks, "
            "70–79% = 3 marks, "
            "60–69% = 2 marks, "
            "10–59% = 1 mark, "
            "below 10% = 0 marks.",
            normal,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "<b>Performance classification:</b> "
            "Below 50% = Low, "
            "50–64% = Average, "
            "65–79% = Above Average, "
            "80% and above = Good.",
            normal,
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "This classification is a project-defined "
            "academic monitoring scale and is not "
            "an official KTU grading system.",
            normal,
        )
    )

    story.append(
        Spacer(1, 25)
    )

    story.append(
        Paragraph(
            APP_TITLE,
            subtitle,
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    if not st.session_state[
        "tutor_logged_in"
    ]:

        st.session_state["page"] = (
            "tutor_login"
        )

        st.rerun()

    branch = st.session_state[
        "tutor_branch"
    ]

    st.markdown(
        f"""
        <div class="glass-card">

            <div class="dashboard-title">
                👨‍🏫 Tutor Dashboard
            </div>

            <div class="small-muted">
                Assigned Branch: {branch}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🚪 Logout",
        key="tutor_dashboard_logout",
    ):

        st.session_state[
            "tutor_logged_in"
        ] = False

        st.session_state[
            "page"
        ] = "home"

        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "➕ Add Student",
            "📊 Student Data",
            "🤖 K-Means",
            "📄 Progress Reports",
        ]
    )

    # ========================================================
    # ADD STUDENT
    # ========================================================

    with tab1:

        st.subheader(
            "Add Student Performance"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            name = st.text_input(
                "Student Name",
                key="add_name",
            )

        with c2:

            username = st.text_input(
                "Student Username",
                key="add_username",
            )

        with c3:

            university_id = st.text_input(
                "University ID",
                key="add_id",
            )

        semester = st.selectbox(
            "Select Semester",
            list(SEMESTERS.keys()),
            key="add_semester",
        )

        subjects = SEMESTERS[
            semester
        ]

        st.markdown(
            f"### {semester} – Subject Records"
        )

        # Subject heading
        cols = st.columns(
            len(subjects)
        )

        for i, subject in enumerate(
            subjects
        ):

            with cols[i]:

                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        font-weight:700;
                        min-height:65px;
                    ">
                        {subject}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # INPUT ROWS
        # ----------------------------------------------------

        values = {
            "Attendance": [],
            "Internal": [],
            "Assignment": [],
            "Previous": [],
            "Study": [],
        }

        st.markdown(
            "**Attendance (%)**"
        )

        cols = st.columns(
            len(subjects)
        )

        for i in range(
            len(subjects)
        ):

            with cols[i]:

                values[
                    "Attendance"
                ].append(
                    st.number_input(
                        "Attendance",
                        min_value=0.0,
                        max_value=100.0,
                        value=75.0,
                        step=1.0,
                        key=f"attendance_{semester}_{i}",
                        label_visibility="collapsed",
                    )
                )

        st.markdown(
            "**Internal Mark (/40)**"
        )

        cols = st.columns(
            len(subjects)
        )

        for i in range(
            len(subjects)
        ):

            with cols[i]:

                values[
                    "Internal"
                ].append(
                    st.number_input(
                        "Internal",
                        min_value=0.0,
                        max_value=40.0,
                        value=20.0,
                        step=1.0,
                        key=f"internal_{semester}_{i}",
                        label_visibility="collapsed",
                    )
                )

        st.markdown(
            "**Assignment (/15)**"
        )

        cols = st.columns(
            len(subjects)
        )

        for i in range(
            len(subjects)
        ):

            with cols[i]:

                values[
                    "Assignment"
                ].append(
                    st.number_input(
                        "Assignment",
                        min_value=0.0,
                        max_value=15.0,
                        value=8.0,
                        step=1.0,
                        key=f"assignment_{semester}_{i}",
                        label_visibility="collapsed",
                    )
                )

        st.markdown(
            "**Previous Mark (/60)**"
        )

        cols = st.columns(
            len(subjects)
        )

        for i in range(
            len(subjects)
        ):

            with cols[i]:

                values[
                    "Previous"
                ].append(
                    st.number_input(
                        "Previous",
                        min_value=0.0,
                        max_value=60.0,
                        value=30.0,
                        step=1.0,
                        key=f"previous_{semester}_{i}",
                        label_visibility="collapsed",
                    )
                )

        st.markdown(
            "**Study Hours / Day**"
        )

        cols = st.columns(
            len(subjects)
        )

        for i in range(
            len(subjects)
        ):

            with cols[i]:

                values[
                    "Study"
                ].append(
                    st.number_input(
                        "Study Hours",
                        min_value=0.0,
                        max_value=6.0,
                        value=2.0,
                        step=0.5,
                        key=f"study_{semester}_{i}",
                        label_visibility="collapsed",
                    )
                )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if st.button(
            "💾 SUBMIT ALL SUBJECT RECORDS",
            key="save_student",
            type="primary",
            width="stretch",
        ):

            if not name.strip():

                st.error(
                    "Enter student name."
                )

                st.stop()

            if not username.strip():

                st.error(
                    "Enter student username."
                )

                st.stop()

            if not university_id.strip():

                st.error(
                    "Enter university ID."
                )

                st.stop()

            df = load_data()

            # Replace same student + semester
            df = df[
                ~(
                    (
                        df["University_ID"]
                        .astype(str)
                        == university_id.strip()
                    )
                    &
                    (
                        df["Semester"]
                        .astype(str)
                        == semester
                    )
                )
            ].copy()

            new_rows = []

            for i, subject in enumerate(
                subjects
            ):

                attendance = values[
                    "Attendance"
                ][i]

                internal = values[
                    "Internal"
                ][i]

                assignment = values[
                    "Assignment"
                ][i]

                previous = values[
                    "Previous"
                ][i]

                study = values[
                    "Study"
                ][i]

                percentage, performance = (
                    calculate_performance(
                        attendance,
                        internal,
                        assignment,
                        previous,
                        study,
                    )
                )

                new_rows.append(
                    {
                        "Student_Name":
                            name.strip(),

                        "Username":
                            username.strip(),

                        "University_ID":
                            university_id.strip(),

                        "Branch":
                            branch,

                        "Semester":
                            semester,

                        "Subject":
                            subject,

                        "Attendance":
                            attendance,

                        "Internal_Mark":
                            internal,

                        "Assignment":
                            assignment,

                        "Previous_Mark":
                            previous,

                        "Study_Hours":
                            study,

                        "Performance":
                            performance,

                        "Performance_Percent":
                            percentage,

                        "KMeans_Cluster":
                            -1,
                    }
                )

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        new_rows
                    ),
                ],
                ignore_index=True,
            )

            # K-Means
            clustered, _ = run_kmeans(
                df
            )

            save_data(
                clustered
            )

            st.success(
                f"✅ {name} – {semester} "
                "saved successfully."
            )

            st.balloons()

    # ========================================================
    # STUDENT DATA
    # ========================================================

    with tab2:

        st.subheader(
            "📊 Student Data"
        )

        df = load_data()

        branch_df = df[
            df["Branch"]
            .astype(str)
            == branch
        ].copy()

        if branch_df.empty:

            st.info(
                "No student data available."
            )

        else:

            semester_filter = st.selectbox(
                "Semester",
                [
                    "All Semesters"
                ] + list(
                    SEMESTERS.keys()
                ),
                key="data_filter",
            )

            display_df = branch_df.copy()

            if (
                semester_filter
                != "All Semesters"
            ):

                display_df = display_df[
                    display_df[
                        "Semester"
                    ].astype(str)
                    == semester_filter
                ]

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "📥 Download Branch CSV",
                data=display_df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                ),
                file_name=(
                    "branch_student_data.csv"
                ),
                mime="text/csv",
                width="stretch",
            )

    # ========================================================
    # K-MEANS
    # ========================================================

    with tab3:

        st.subheader(
            "🤖 K-Means Clustering"
        )

        df = load_data()

        branch_df = df[
            df["Branch"]
            .astype(str)
            == branch
        ].copy()

        if len(branch_df) < 3:

            st.warning(
                "At least 3 records are "
                "required for K-Means."
            )

        else:

            semester_filter = st.selectbox(
                "Analysis Semester",
                [
                    "All Semesters"
                ] + list(
                    SEMESTERS.keys()
                ),
                key="analysis_semester",
            )

            analysis = branch_df.copy()

            if (
                semester_filter
                != "All Semesters"
            ):

                analysis = analysis[
                    analysis[
                        "Semester"
                    ].astype(str)
                    == semester_filter
                ]

            clustered, score = run_kmeans(
                analysis
            )

            if score is not None:

                st.metric(
                    "Silhouette Score",
                    f"{score:.3f}",
                )

            st.info(
                "K-Means cluster numbers are "
                "group identifiers, not grades."
            )

            st.dataframe(
                clustered[
                    [
                        "Student_Name",
                        "University_ID",
                        "Semester",
                        "Subject",
                        "Performance",
                        "KMeans_Cluster",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    # ========================================================
    # PROGRESS REPORT
    # ========================================================

    with tab4:

        st.subheader(
            "📄 Student Progress Reports"
        )

        df = load_data()

        branch_df = df[
            df["Branch"]
            .astype(str)
            == branch
        ].copy()

        if branch_df.empty:

            st.info(
                "No student records available."
            )

        else:

            students = (
                branch_df[
                    [
                        "Student_Name",
                        "University_ID",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    "Student_Name"
                )
            )

            options = [
                f"{row['Student_Name']} | "
                f"{row['University_ID']}"
                for _, row
                in students.iterrows()
            ]

            selected = st.selectbox(
                "Select Student",
                options,
                key="report_student",
            )

            selected_id = (
                selected
                .split("|")[-1]
                .strip()
            )

            selected_semester = st.selectbox(
                "Select Semester",
                list(
                    SEMESTERS.keys()
                ),
                key="report_semester",
            )

            report_df = branch_df[
                (
                    branch_df[
                        "University_ID"
                    ].astype(str)
                    == selected_id
                )
                &
                (
                    branch_df[
                        "Semester"
                    ].astype(str)
                    == selected_semester
                )
            ].copy()

            if report_df.empty:

                st.warning(
                    "No records found for "
                    "this student and semester."
                )

            else:

                student_name = str(
                    report_df.iloc[0][
                        "Student_Name"
                    ]
                )

                pdf = create_pdf_report(
                    student_name,
                    selected_id,
                    branch,
                    selected_semester,
                    report_df,
                )

                st.success(
                    "PDF Progress Report is ready."
                )

                safe_name = re.sub(
                    r"[^A-Za-z0-9_-]",
                    "_",
                    student_name,
                )

                st.download_button(
                    "📥 DOWNLOAD PROGRESS REPORT PDF",
                    data=pdf,
                    file_name=(
                        f"{safe_name}_"
                        f"{selected_semester}_"
                        "Progress_Report.pdf"
                    ),
                    mime="application/pdf",
                    width="stretch",
                )

                st.dataframe(
                    report_df,
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# MAIN ROUTER
# ============================================================

if st.session_state["page"] == "home":

    home_page()

elif st.session_state["page"] == "student_login":

    student_login()

elif st.session_state["page"] == "tutor_login":

    tutor_login()

elif st.session_state["page"] == "student_dashboard":

    student_dashboard()

elif st.session_state["page"] == "tutor_dashboard":

    tutor_dashboard()

else:

    st.session_state["page"] = "home"

    st.rerun()
