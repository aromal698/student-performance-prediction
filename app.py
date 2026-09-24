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
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATA FILE
# ============================================================

DATA_FOLDER = "data"
DATA_FILE = os.path.join(
    DATA_FOLDER,
    "student_performance.csv"
)


# ============================================================
# BRANCHES
# ============================================================

BRANCHES = [
    "Artificial Intelligence and Data Science",
    "Computer Science and Engineering",
    "Computer Science and Engineering (Artificial Intelligence)",
    "Computer Science and Engineering (Cyber Security)",
    "Computer Science and Engineering (Data Science)",
    "Artificial Intelligence and Machine Learning",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
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
    "Robotics and Automation",
    "Mechatronics Engineering",
]


# ============================================================
# TUTOR ACCOUNTS
# ============================================================

TUTOR_ACCOUNTS = {

    "Artificial Intelligence and Data Science":
        ("tutor_aids", "aids123"),

    "Computer Science and Engineering":
        ("tutor_cse", "cse123"),

    "Computer Science and Engineering (Artificial Intelligence)":
        ("tutor_cseai", "cseai123"),

    "Computer Science and Engineering (Cyber Security)":
        ("tutor_cyber", "cyber123"),

    "Computer Science and Engineering (Data Science)":
        ("tutor_csedata", "csedata123"),

    "Artificial Intelligence and Machine Learning":
        ("tutor_aiml", "aiml123"),

    "Information Technology":
        ("tutor_it", "it123"),

    "Electronics and Communication Engineering":
        ("tutor_ece", "ece123"),

    "Electrical and Electronics Engineering":
        ("tutor_eee", "eee123"),

    "Electrical Engineering":
        ("tutor_ee", "ee123"),

    "Mechanical Engineering":
        ("tutor_me", "me123"),

    "Civil Engineering":
        ("tutor_civil", "civil123"),

    "Chemical Engineering":
        ("tutor_chemical", "chemical123"),

    "Biotechnology and Biochemical Engineering":
        ("tutor_bio", "bio123"),

    "Food Technology":
        ("tutor_food", "food123"),

    "Production Engineering":
        ("tutor_production", "production123"),

    "Automobile Engineering":
        ("tutor_auto", "auto123"),

    "Industrial Engineering":
        ("tutor_industrial", "industrial123"),

    "Biomedical Engineering":
        ("tutor_biomedical", "biomedical123"),

    "Aeronautical Engineering":
        ("tutor_aero", "aero123"),

    "Applied Electronics and Instrumentation Engineering":
        ("tutor_aei", "aei123"),

    "Electronics and Biomedical Engineering":
        ("tutor_ebm", "ebm123"),

    "Electronics and Computer Engineering":
        ("tutor_ececomp", "ececomp123"),

    "Robotics and Automation":
        ("tutor_robotics", "robotics123"),

    "Mechatronics Engineering":
        ("tutor_mechatronics", "mechatronics123"),
}


# ============================================================
# KTU 2024 AI & DATA SCIENCE CURRICULUM
# ============================================================
#
# This is the main department for your project.
#
# S1/S2 are Group-A common curriculum.
# S3-S8 contain AI & Data Science specific courses.
#
# ============================================================

AI_DS_2024 = {

    "S1": [
        "GAMAT101 - Mathematics for Information Science-I",
        "GAPHT121 - Physics for Information Science",
        "GXCYT122 - Chemistry for Information Science",
        "GMEST103 - Engineering Graphics and Computer Aided Drawing",
        "GXEST104 - Introduction to Electrical & Electronics Engineering",
        "UCEST105 - Algorithmic Thinking with Python",
        "GXESL106 - Basic Electrical and Electronics Engineering Workshop",
        "UCHWT127 - Health and Wellness",
        "UCHUT128 - Life Skills and Professional Communication",
        "UCSEM129 - Digital 101 (NASSCOM)",
    ],

    "S2": [
        "GAMAT201 - Mathematics for Information Science-II",
        "GAPHT121 - Physics for Information Science",
        "GXCYT122 - Chemistry for Information Science",
        "GXEST203 - Foundations of Computing: From Hardware Essentials to Web Design",
        "GXEST204 - Programming in C",
        "PCCST205 - Discrete Mathematics",
        "UCEST206 - Engineering Entrepreneurship & IPR",
        "UCHWT127 - Health and Wellness",
        "UCHUT128 - Life Skills and Professional Communication",
        "GXESL208 - IT Workshop",
        "UCSEM129 - Digital 101 (NASSCOM)",
    ],

    "S3": [
        "GAMAT301 - Mathematics for Information Science-III",
        "PCAIT302 - Foundations of Artificial Intelligence",
        "PCCST303 - Data Structures and Algorithms",
        "PBADT304 - Introduction to Data Science",
        "GAEST305 - Digital Electronics & Logic Design",
        "UCHUT346 - Economics for Engineers",
        "UCHUT347 - Engineering Ethics and Sustainable Development",
        "PCCSL307 - Data Structures Lab",
        "PCCDL308 - Python and Statistical Modelling Lab",
        "Remedial / Minor Course",
    ],

    "S4": [
        "GAMAT401 - Mathematics for Information Science-IV",
        "PCCST402 - Database Management Systems",
        "PCCST403 - Operating Systems",
        "PBCST404 - Computer Organization and Architecture",
        "PEADT41N - Programme Elective-I",
        "UCHUT346 - Economics for Engineers",
        "UCHUT347 - Engineering Ethics and Sustainable Development",
        "PCADL407 - Foundations of AI and Data Science Lab",
        "PCCSL408 - DBMS Lab",
        "Remedial / Minor / Honours Course",
    ],

    "S5": [
        "PCCST501 - Computer Networks",
        "PCADT502 - Robotics and Intelligent Systems",
        "PCCST503 - Machine Learning",
        "PBADT504 - Big Data Analytics",
        "PEADT52N - Programme Elective-II",
        "UCHUM506 - Constitution of India (MOOC)",
        "PCADL507 - Robotics Lab",
        "PCCDL508 - Data Analytics Lab",
        "Remedial / Minor / Honours Course",
        "Industrial Visit / Industrial Training",
    ],

    "S6": [
        "PCADT601 - Deep Learning",
        "PCADT602 - Internet of Things",
        "PEADT63N - Programme Elective-III",
        "PBADT604 - Data Mining and Warehousing",
        "GAEST605 - Design Thinking and Product Development",
        "OEADT61N - Open Elective / Industry Elective-I",
        "PCADL607 - Deep Learning Lab",
        "PCADP608 - Mini Project: Socially Relevant Project",
        "Remedial / Minor / Honours Course",
        "Industrial Visit / Industrial Training",
    ],

    "S7": [
        "PEADT74N - Programme Elective-IV",
        "PEADT75N - Programme Elective-V",
        "OEADT72N - Open Elective / Industry Elective-II",
        "UEHUT704 - HMC Elective",
        "PCADS705 - Seminar",
        "PCADP706 - Major Project / Internship",
        "Remedial / Minor / Honours Course",
    ],

    "S8": [
        "PEADT86N - Programme Elective-VI",
        "OEADT83N - Open Elective / Industry Elective-III",
        "UEHUT803 - Organizational Behavior and Business Communication",
        "PCADP806 - Major Project / Internship",
    ],
}


# ============================================================
# GENERIC 2024 STRUCTURE FOR OTHER DEPARTMENTS
# ============================================================
#
# These names intentionally indicate KTU programme-core slots
# rather than pretending that every branch has identical subjects.
#
# Exact branch curriculum can be added separately.
# ============================================================

GENERIC_2024 = {

    "S1": [
        "Mathematics for Information Science-I",
        "Physics / Chemistry for Information Science",
        "Engineering Graphics and Computer Aided Drawing",
        "Introduction to Electrical & Electronics Engineering",
        "Algorithmic Thinking with Python",
        "Basic Electrical and Electronics Engineering Workshop",
        "Health and Wellness / Life Skills",
        "Digital 101 (NASSCOM)",
    ],

    "S2": [
        "Mathematics for Information Science-II",
        "Physics / Chemistry for Information Science",
        "Branch-specific Engineering Science Course",
        "Programming in C",
        "Discrete Mathematics",
        "Engineering Entrepreneurship & IPR",
        "IT Workshop",
        "Life Skills and Professional Communication",
        "Digital 101 (NASSCOM)",
    ],

    "S3": [
        "Group-specific Mathematics-III",
        "Programme Core-1",
        "Programme Core-2",
        "Programme Core-PBL",
        "Group-specific Engineering Science Course",
        "Engineering Ethics and Sustainable Development / Economics",
        "Programme Core Lab-1",
        "Programme Core Lab-2",
    ],

    "S4": [
        "Group-specific Mathematics-IV",
        "Programme Core-4",
        "Programme Core-5",
        "Programme Core-PBL-2",
        "Programme Elective-I",
        "Engineering Ethics and Sustainable Development / Economics",
        "Programme Core Lab-3",
        "Programme Core Lab-4",
    ],

    "S5": [
        "Programme Core-6",
        "Programme Core-7",
        "Programme Core-8",
        "Programme Core-PBL-3",
        "Programme Elective-II",
        "Constitution of India (MOOC)",
        "Programme Core Lab-5",
        "Programme Core Lab-6",
        "Industrial Visit / Industrial Training",
    ],

    "S6": [
        "Programme Core-9",
        "Programme Core-10",
        "Programme Elective-III",
        "Programme Core-PBL-4",
        "Design Thinking and Product Development",
        "Open Elective / Industry Elective-I",
        "Programme Core Lab-7",
        "Mini Project: Socially Relevant Project",
        "Industrial Visit / Industrial Training",
    ],

    "S7": [
        "Programme Elective-IV",
        "Programme Elective-V",
        "Open Elective / Industry Elective-II",
        "HMC Elective",
        "Seminar",
        "Major Project / Internship",
    ],

    "S8": [
        "Programme Elective-VI",
        "Open Elective / Industry Elective-III",
        "Organizational Behavior and Business Communication",
        "Major Project / Internship",
    ],
}


# ============================================================
# GET SUBJECTS
# ============================================================

def get_subjects(branch, semester):

    if branch == "Artificial Intelligence and Data Science":
        return AI_DS_2024.get(
            semester,
            []
        )

    return GENERIC_2024.get(
        semester,
        []
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* MAIN APP */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0, 200, 255, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(140, 70, 255, 0.18),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #050816,
            #0a1025,
            #050713
        );
}

/* CENTER MAIN CONTENT */

.block-container {
    max-width: 1250px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 2rem !important;
}

/* FRONT PAGE */

.front-page {
    min-height: 82vh;

    display: flex;

    align-items: center;

    justify-content: center;

    text-align: center;
}

.front-box {
    width: min(900px, 95%);

    padding: 55px;

    border-radius: 30px;

    background: rgba(15, 23, 50, 0.80);

    border:
        1px solid
        rgba(255,255,255,0.15);

    box-shadow:
        0 25px 70px
        rgba(0,0,0,0.45);

    backdrop-filter: blur(20px);
}

/* IMPORTANT: MAIN HEADING */

.front-heading {
    font-size: 48px;

    font-weight: 900;

    text-align: center;

    color: white;

    margin-bottom: 35px;
}

/* DASHBOARD */

.dashboard-box {
    width: 100%;

    max-width: 1150px;

    margin-left: auto;

    margin-right: auto;

    text-align: center;

    padding: 28px;

    border-radius: 25px;

    background:
        rgba(12,18,42,0.80);

    border:
        1px solid
        rgba(255,255,255,0.12);

    box-shadow:
        0 20px 60px
        rgba(0,0,0,0.30);
}

/* DASHBOARD HEADING */

.dashboard-heading {
    font-size: 38px;

    font-weight: 900;

    color: white;

    text-align: center;

    margin-bottom: 5px;
}

.dashboard-subtitle {
    text-align: center;

    color: #b9c8df;

    font-size: 16px;
}

/* SECTION */

.section {
    width: 100%;

    max-width: 1150px;

    margin: 20px auto;

    padding: 25px;

    border-radius: 22px;

    background:
        rgba(12,18,42,0.70);

    border:
        1px solid
        rgba(255,255,255,0.10);
}

/* LOGIN */

.login-box {
    max-width: 600px;

    margin: 35px auto;

    padding: 35px;

    border-radius: 25px;

    background:
        rgba(12,18,42,0.82);

    border:
        1px solid
        rgba(255,255,255,0.12);
}

/* SUBJECT */

.subject-box {
    padding: 20px;

    margin-bottom: 15px;

    border-radius: 18px;

    background:
        rgba(255,255,255,0.05);

    border:
        1px solid
        rgba(255,255,255,0.10);
}

.subject-name {
    font-size: 19px;

    font-weight: 800;

    color: white;

    margin-bottom: 10px;
}

/* MOBILE */

@media(max-width:700px) {

    .front-heading {
        font-size: 30px;
    }

    .front-box {
        padding: 30px 20px;
    }

    .dashboard-heading {
        font-size: 28px;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "student_logged" not in st.session_state:
    st.session_state.student_logged = False

if "tutor_logged" not in st.session_state:
    st.session_state.tutor_logged = False

if "student_id" not in st.session_state:
    st.session_state.student_id = ""

if "student_username" not in st.session_state:
    st.session_state.student_username = ""

if "tutor_branch" not in st.session_state:
    st.session_state.tutor_branch = ""

if "tutor_username" not in st.session_state:
    st.session_state.tutor_username = ""


# ============================================================
# CREATE DATA FILE
# ============================================================

def create_data_file():

    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    if not os.path.exists(DATA_FILE):

        columns = [
            "Name",
            "University_ID",
            "Branch",
            "Semester",
            "Subject",
            "Attendance",
            "Study_Hours",
            "Internal",
            "Assignment",
            "Previous_Mark",
        ]

        pd.DataFrame(
            columns=columns
        ).to_csv(
            DATA_FILE,
            index=False
        )


def load_data():

    create_data_file()

    try:
        return pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame()


def save_data(df):

    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    df.to_csv(
        DATA_FILE,
        index=False
    )


# ============================================================
# ATTENDANCE
# ============================================================

def attendance_mark(value):

    value = float(value)

    if value >= 90:
        return 5

    if value >= 80:
        return 4

    if value >= 70:
        return 3

    if value >= 60:
        return 2

    if value >= 10:
        return 1

    return 0


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_performance(row):

    attendance = float(row["Attendance"])
    study = float(row["Study_Hours"])
    internal = float(row["Internal"])
    assignment = float(row["Assignment"])
    previous = float(row["Previous_Mark"])

    att = attendance_mark(
        attendance
    )

    attendance_percent = (
        att / 5
    ) * 100

    study_percent = min(
        study / 6 * 100,
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

    overall = np.mean(
        [
            attendance_percent,
            study_percent,
            internal_percent,
            assignment_percent,
            previous_percent,
        ]
    )

    if overall < 50:

        return (
            round(overall, 2),
            "Low Performance",
            "🔴",
            "#ff4d4d"
        )

    elif overall < 65:

        return (
            round(overall, 2),
            "Average Performance",
            "🟠",
            "#ff9f43"
        )

    elif overall < 80:

        return (
            round(overall, 2),
            "Above Average Performance",
            "🟡",
            "#ffd93d"
        )

    else:

        return (
            round(overall, 2),
            "Good Performance",
            "🟢",
            "#45e36b"
        )


# ============================================================
# PASSWORD
# ============================================================

def valid_student_password(password):

    if not re.fullmatch(
        r"BTECH20\d{2}",
        password
    ):
        return False

    year = int(
        password[-4:]
    )

    return 2000 <= year <= 2022


# ============================================================
# HOME
# ============================================================

def home():

    st.markdown(
        """
        <div class="front-page">

            <div class="front-box">

                <div class="front-heading">
                    🎓 STUDENT PERFORMANCE PREDICTION
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align:center;color:white;'>Select Login</h3>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(
        [1, 2, 2]
    )

    with c2:

        if st.button(
            "🎓 STUDENT LOGIN",
            width="stretch"
        ):

            st.session_state.page = "student_login"

            st.rerun()

    with c3:

        if st.button(
            "👨‍🏫 TUTOR LOGIN",
            width="stretch"
        ):

            st.session_state.page = "tutor_login"

            st.rerun()


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.markdown(
        """
        <div class="dashboard-box">

            <div class="dashboard-heading">
                🎓 Student Login
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-box">',
        unsafe_allow_html=True
    )

    username = st.text_input(
        "Username"
    )

    university_id = st.text_input(
        "University ID"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "🔐 LOGIN",
        width="stretch"
    ):

        if not username or not university_id or not password:

            st.error(
                "Please enter all details."
            )

        elif not valid_student_password(
            password
        ):

            st.error(
                "Password must be BTECH2000 to BTECH2022."
            )

        else:

            df = load_data()

            if df.empty:

                st.error(
                    "No student records available."
                )

            else:

                matched = df[
                    df["University_ID"]
                    .astype(str)
                    .str.strip()
                    ==
                    university_id.strip()
                ]

                if matched.empty:

                    st.error(
                        "University ID not found."
                    )

                else:

                    st.session_state.student_logged = True

                    st.session_state.student_id = (
                        university_id.strip()
                    )

                    st.session_state.student_username = (
                        username
                    )

                    st.session_state.page = (
                        "student_dashboard"
                    )

                    st.rerun()

    if st.button(
        "⬅️ BACK",
        width="stretch"
    ):

        st.session_state.page = "home"

        st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    st.markdown(
        """
        <div class="dashboard-box">

            <div class="dashboard-heading">
                👨‍🏫 Tutor Login
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-box">',
        unsafe_allow_html=True
    )

    branch = st.selectbox(
        "Select Department",
        BRANCHES
    )

    username = st.text_input(
        "Tutor Username"
    )

    password = st.text_input(
        "Tutor Password",
        type="password"
    )

    if st.button(
        "🔐 LOGIN",
        width="stretch"
    ):

        correct_username, correct_password = (
            TUTOR_ACCOUNTS[branch]
        )

        if (
            username == correct_username
            and
            password == correct_password
        ):

            st.session_state.tutor_logged = True

            st.session_state.tutor_branch = (
                branch
            )

            st.session_state.tutor_username = (
                username
            )

            st.session_state.page = (
                "tutor_dashboard"
            )

            st.rerun()

        else:

            st.error(
                "Invalid tutor username or password."
            )

    if st.button(
        "⬅️ BACK",
        width="stretch"
    ):

        st.session_state.page = "home"

        st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    df = load_data()

    student_id = (
        st.session_state.student_id
    )

    student_df = df[
        df["University_ID"]
        .astype(str)
        .str.strip()
        ==
        str(student_id).strip()
    ].copy()

    if student_df.empty:

        st.error(
            "Student data not found."
        )

        return

    student_name = str(
        student_df.iloc[0]["Name"]
    )

    branch = str(
        student_df.iloc[0]["Branch"]
    )

    # --------------------------------------------------------
    # CENTERED HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="dashboard-box">

            <div class="dashboard-heading">
                🎓 {student_name}
            </div>

            <div class="dashboard-subtitle">
                Student Performance Dashboard
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DEPARTMENT
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="section">

            <h3 style="text-align:center;">
                Department
            </h3>

            <p style="text-align:center;font-size:18px;">
                {branch}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SEMESTER
    # --------------------------------------------------------

    semester = st.selectbox(
        "Select Semester",
        [
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
        ]
    )

    subjects = get_subjects(
        branch,
        semester
    )

    if not subjects:

        st.warning(
            "No curriculum has been mapped for this department."
        )

        return

    # --------------------------------------------------------
    # SHOW KTU SUBJECTS
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="section">

            <h2 style="text-align:center;">
                {semester} — KTU Subjects
            </h2>

        </div>
        """,
        unsafe_allow_html=True
    )

    for subject in subjects:

        st.markdown(
            f"""
            <div class="subject-box">

                <div class="subject-name">
                    📘 {subject}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # STUDENT DATA FOR SELECTED SEMESTER
    # --------------------------------------------------------

    current = student_df[
        student_df["Semester"].astype(str)
        == semester
    ].copy()

    if current.empty:

        st.info(
            f"No marks have been entered for {semester} yet."
        )

    else:

        results = []

        for _, row in current.iterrows():

            percentage, level, emoji, color = (
                calculate_performance(row)
            )

            results.append(
                percentage
            )

            st.markdown(
                f"""
                <div class="section">

                    <h3 style="text-align:center;">
                        📚 {row["Subject"]}
                    </h3>

                    <div style="
                        text-align:center;
                        font-size:25px;
                    ">
                        {emoji}
                    </div>

                    <h3 style="text-align:center;">
                        {level}
                    </h3>

                    <h2 style="
                        text-align:center;
                        color:{color};
                    ">
                        {percentage:.2f}%
                    </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

            a, b, c, d, e = st.columns(5)

            with a:
                st.metric(
                    "Attendance",
                    f'{row["Attendance"]}%'
                )

            with b:
                st.metric(
                    "Internal",
                    f'{row["Internal"]}/40'
                )

            with c:
                st.metric(
                    "Assignment",
                    f'{row["Assignment"]}/15'
                )

            with d:
                st.metric(
                    "Previous",
                    f'{row["Previous_Mark"]}/60'
                )

            with e:
                st.metric(
                    "Study Hours",
                    f'{row["Study_Hours"]}'
                )

    st.divider()

    if st.button(
        "🚪 LOGOUT",
        width="stretch"
    ):

        logout()


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    branch = (
        st.session_state.tutor_branch
    )

    st.markdown(
        f"""
        <div class="dashboard-box">

            <div class="dashboard-heading">
                👨‍🏫 Tutor Dashboard
            </div>

            <div class="dashboard-subtitle">
                {branch}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CURRICULUM SELECTOR
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section">

            <h2 style="text-align:center;">
                📚 KTU Curriculum
            </h2>

        </div>
        """,
        unsafe_allow_html=True
    )

    semester = st.selectbox(
        "Select Semester",
        [
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
        ],
        key="tutor_semester"
    )

    subjects = get_subjects(
        branch,
        semester
    )

    if subjects:

        st.markdown(
            f"""
            <div class="section">

                <h3 style="text-align:center;">
                    {branch}
                </h3>

                <h3 style="text-align:center;">
                    {semester} Subjects
                </h3>

            </div>
            """,
            unsafe_allow_html=True
        )

        for i, subject in enumerate(
            subjects,
            start=1
        ):

            st.write(
                f"**{i}.** {subject}"
            )

    st.divider()

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    add_tab, data_tab, cluster_tab, rf_tab = st.tabs(
        [
            "➕ Add Student",
            "📋 Student Data",
            "📊 K-Means",
            "🌲 Random Forest",
        ]
    )

    # ========================================================
    # ADD STUDENT
    # ========================================================

    with add_tab:

        st.header(
            "Add Student Performance"
        )

        name = st.text_input(
            "Student Name",
            key="new_student_name"
        )

        university_id = st.text_input(
            "University ID",
            key="new_student_id"
        )

        st.subheader(
            f"{semester} Subject Records"
        )

        records = []

        for subject in subjects:

            st.markdown(
                f"### 📘 {subject}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                attendance = st.number_input(
                    "Attendance %",
                    0.0,
                    100.0,
                    80.0,
                    key=f"a_{semester}_{subject}"
                )

            with c2:

                study = st.number_input(
                    "Study Hours",
                    0.0,
                    6.0,
                    3.0,
                    key=f"s_{semester}_{subject}"
                )

            with c3:

                internal = st.number_input(
                    "Internal /40",
                    0.0,
                    40.0,
                    25.0,
                    key=f"i_{semester}_{subject}"
                )

            with c4:

                assignment = st.number_input(
                    "Assignment /15",
                    0.0,
                    15.0,
                    10.0,
                    key=f"as_{semester}_{subject}"
                )

            with c5:

                previous = st.number_input(
                    "Previous /60",
                    0.0,
                    60.0,
                    35.0,
                    key=f"p_{semester}_{subject}"
                )

            records.append(
                {
                    "Name": name,
                    "University_ID": university_id,
                    "Branch": branch,
                    "Semester": semester,
                    "Subject": subject,
                    "Attendance": attendance,
                    "Study_Hours": study,
                    "Internal": internal,
                    "Assignment": assignment,
                    "Previous_Mark": previous,
                }
            )

        if st.button(
            "💾 SAVE STUDENT",
            width="stretch"
        ):

            if not name.strip():

                st.error(
                    "Enter student name."
                )

            elif not university_id.strip():

                st.error(
                    "Enter University ID."
                )

            else:

                old_df = load_data()

                if not old_df.empty:

                    old_df = old_df[
                        ~(
                            (
                                old_df[
                                    "University_ID"
                                ].astype(str)
                                ==
                                university_id.strip()
                            )
                            &
                            (
                                old_df[
                                    "Semester"
                                ].astype(str)
                                ==
                                semester
                            )
                        )
                    ]

                new_df = pd.DataFrame(
                    records
                )

                final_df = pd.concat(
                    [
                        old_df,
                        new_df
                    ],
                    ignore_index=True
                )

                save_data(
                    final_df
                )

                st.success(
                    "Student records saved successfully."
                )

                st.rerun()

    # ========================================================
    # DATA TAB
    # ========================================================

    with data_tab:

        st.header(
            "Student Data"
        )

        df = load_data()

        if df.empty:

            st.info(
                "No student data available."
            )

        else:

            branch_df = df[
                df["Branch"].astype(str)
                ==
                branch
            ].copy()

            if branch_df.empty:

                st.info(
                    "No students in this department."
                )

            else:

                st.dataframe(
                    branch_df,
                    width="stretch",
                    hide_index=True
                )

                st.download_button(
                    "⬇️ DOWNLOAD BRANCH CSV",
                    data=branch_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="branch_data.csv",
                    mime="text/csv",
                    width="stretch"
                )

                st.subheader(
                    "Delete Student"
                )

                ids = (
                    branch_df[
                        "University_ID"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_id = st.selectbox(
                    "Select University ID",
                    ids
                )

                if st.button(
                    "🗑️ DELETE STUDENT",
                    width="stretch"
                ):

                    remaining = df[
                        ~(
                            (
                                df[
                                    "University_ID"
                                ].astype(str)
                                ==
                                selected_id
                            )
                            &
                            (
                                df[
                                    "Branch"
                                ].astype(str)
                                ==
                                branch
                            )
                        )
                    ]

                    save_data(
                        remaining
                    )

                    st.success(
                        "Student data deleted."
                    )

                    st.rerun()

    # ========================================================
    # K-MEANS
    # ========================================================

    with cluster_tab:

        st.header(
            "📊 K-Means Clustering"
        )

        df = load_data()

        branch_df = df[
            df["Branch"].astype(str)
            ==
            branch
        ].copy()

        if len(branch_df) < 3:

            st.warning(
                "At least 3 records are required."
            )

        else:

            features = [
                "Attendance",
                "Study_Hours",
                "Internal",
                "Assignment",
                "Previous_Mark",
            ]

            X = branch_df[
                features
            ].apply(
                pd.to_numeric,
                errors="coerce"
            ).dropna()

            if len(X) >= 3:

                model = KMeans(
                    n_clusters=3,
                    random_state=42,
                    n_init=10
                )

                clusters = model.fit_predict(
                    X
                )

                score = silhouette_score(
                    X,
                    clusters
                )

                st.metric(
                    "Silhouette Score",
                    f"{score:.3f}"
                )

                result = branch_df.loc[
                    X.index
                ].copy()

                result[
                    "KMeans_Cluster"
                ] = clusters

                st.dataframe(
                    result[
                        [
                            "Name",
                            "University_ID",
                            "Semester",
                            "Subject",
                            "KMeans_Cluster",
                        ]
                    ],
                    width="stretch",
                    hide_index=True
                )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    with rf_tab:

        st.header(
            "🌲 Random Forest"
        )

        df = load_data()

        branch_df = df[
            df["Branch"].astype(str)
            ==
            branch
        ].copy()

        if len(branch_df) < 10:

            st.warning(
                "At least 10 records are recommended."
            )

        else:

            feature_cols = [
                "Attendance",
                "Study_Hours",
                "Internal",
                "Assignment",
                "Previous_Mark",
            ]

            X = branch_df[
                feature_cols
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            y = []

            valid_rows = []

            for index, row in branch_df.iterrows():

                try:

                    _, level, _, _ = (
                        calculate_performance(
                            row
                        )
                    )

                    y.append(level)

                    valid_rows.append(
                        index
                    )

                except Exception:
                    pass

            X = X.loc[
                valid_rows
            ]

            y = np.array(y)

            X = X.dropna()

            if len(X) >= 10 and len(
                np.unique(y)
            ) >= 2:

                try:

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y[:len(X)],
                            test_size=0.2,
                            random_state=42,
                            stratify=y[:len(X)]
                        )
                    )

                except Exception:

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y[:len(X)],
                            test_size=0.2,
                            random_state=42
                        )
                    )

                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )

                rf.fit(
                    X_train,
                    y_train
                )

                predictions = rf.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

            else:

                st.warning(
                    "Multiple performance classes are required."
                )

    # ========================================================
    # LOGOUT
    # ========================================================

    st.divider()

    if st.button(
        "🚪 LOGOUT",
        width="stretch"
    ):

        logout()


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.page = "home"

    st.session_state.student_logged = False

    st.session_state.tutor_logged = False

    st.session_state.student_id = ""

    st.session_state.student_username = ""

    st.session_state.tutor_branch = ""

    st.session_state.tutor_username = ""

    st.rerun()


# ============================================================
# MAIN
# ============================================================

create_data_file()

if st.session_state.page == "home":

    home()

elif st.session_state.page == "student_login":

    student_login()

elif st.session_state.page == "tutor_login":

    tutor_login()

elif st.session_state.page == "student_dashboard":

    if st.session_state.student_logged:

        student_dashboard()

    else:

        st.session_state.page = "student_login"

        st.rerun()

elif st.session_state.page == "tutor_dashboard":

    if st.session_state.tutor_logged:

        tutor_dashboard()

    else:

        st.session_state.page = "tutor_login"

        st.rerun()

else:

    st.session_state.page = "home"

    st.rerun()
