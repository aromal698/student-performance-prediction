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
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. FILE SETTINGS
# ============================================================

DATA_FOLDER = "data"
DATA_FILE = os.path.join(
    DATA_FOLDER,
    "student_performance.csv"
)


# ============================================================
# 3. DEPARTMENTS
# ============================================================

DEPARTMENTS = [
    "Artificial Intelligence and Data Science",
    "Computer Science and Engineering",
    "Computer Science and Engineering (Artificial Intelligence)",
    "Computer Science and Engineering (Data Science)",
    "Computer Science and Engineering (Cyber Security)",
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
# 4. TUTOR LOGIN DETAILS
# ============================================================

TUTOR_ACCOUNTS = {

    "Artificial Intelligence and Data Science":
        ("tutor_aids", "aids123"),

    "Computer Science and Engineering":
        ("tutor_cse", "cse123"),

    "Computer Science and Engineering (Artificial Intelligence)":
        ("tutor_cseai", "cseai123"),

    "Computer Science and Engineering (Data Science)":
        ("tutor_csedata", "csedata123"),

    "Computer Science and Engineering (Cyber Security)":
        ("tutor_cyber", "cyber123"),

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
# 5. KTU 2024 AI & DATA SCIENCE CURRICULUM
# ============================================================
#
# This is the curriculum section to use for your
# AI & Data Science project.
#
# The branch is selected FIRST.
# Then semester.
# Then only the subjects belonging to that branch/semester
# are displayed.
#
# ============================================================

KTU_AI_DS_2024 = {

    "S1": [
        "Mathematics for Information Science-I",
        "Physics for Information Science",
        "Chemistry for Information Science",
        "Engineering Graphics and Computer Aided Drawing",
        "Introduction to Electrical & Electronics Engineering",
        "Algorithmic Thinking with Python",
        "Basic Electrical and Electronics Engineering Workshop",
        "Health and Wellness",
        "Life Skills and Professional Communication",
        "Digital 101",
    ],

    "S2": [
        "Mathematics for Information Science-II",
        "Foundations of Computing",
        "Programming in C",
        "Discrete Mathematics",
        "Engineering Entrepreneurship & IPR",
        "IT Workshop",
        "Life Skills and Professional Communication",
        "Health and Wellness",
        "Digital 101",
    ],

    "S3": [
        "Mathematics for Information Science-III",
        "Foundations of Artificial Intelligence",
        "Data Structures and Algorithms",
        "Introduction to Data Science",
        "Digital Electronics & Logic Design",
        "Economics for Engineers",
        "Engineering Ethics and Sustainable Development",
        "Data Structures Lab",
        "Python and Statistical Modelling Lab",
        "Remedial / Minor Course",
    ],

    "S4": [
        "Mathematics for Information Science-IV",
        "Database Management Systems",
        "Operating Systems",
        "Computer Organization and Architecture",
        "Programme Elective-I",
        "Economics for Engineers",
        "Engineering Ethics and Sustainable Development",
        "Foundations of AI and Data Science Lab",
        "DBMS Lab",
        "Remedial / Minor / Honours Course",
    ],

    "S5": [
        "Computer Networks",
        "Robotics and Intelligent Systems",
        "Machine Learning",
        "Big Data Analytics",
        "Programme Elective-II",
        "Constitution of India",
        "Robotics Lab",
        "Data Analytics Lab",
        "Remedial / Minor / Honours Course",
        "Industrial Training / Industrial Visit",
    ],

    "S6": [
        "Deep Learning",
        "Internet of Things",
        "Programme Elective-III",
        "Data Mining and Warehousing",
        "Design Thinking and Product Development",
        "Open Elective / Industry Elective-I",
        "Deep Learning Lab",
        "Mini Project",
        "Remedial / Minor / Honours Course",
        "Industrial Training / Industrial Visit",
    ],

    "S7": [
        "Programme Elective-IV",
        "Programme Elective-V",
        "Open Elective / Industry Elective-II",
        "HMC Elective",
        "Seminar",
        "Major Project / Internship",
        "Remedial / Minor / Honours Course",
    ],

    "S8": [
        "Programme Elective-VI",
        "Open Elective / Industry Elective-III",
        "Organizational Behavior and Business Communication",
        "Major Project / Internship",
    ],
}


# ============================================================
# 6. CURRICULUM FUNCTION
# ============================================================

def get_subjects(department, semester):

    if department == "Artificial Intelligence and Data Science":

        return KTU_AI_DS_2024.get(
            semester,
            []
        )

    # Other departments should be mapped using
    # their verified KTU curriculum before deployment.

    return []


# ============================================================
# 7. CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 15% 15%,
            rgba(0, 190, 255, 0.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(150, 80, 255, 0.18),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #050816,
            #0a1025,
            #050713
        );
}

/* Main Streamlit content */

.block-container {
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* ==========================================================
   MAIN TITLE
   ========================================================== */

.main-title {
    text-align: center;

    font-size: 44px;

    font-weight: 900;

    color: #ffffff;

    margin-top: 20px;

    margin-bottom: 10px;
}

.main-title-line {
    width: 120px;

    height: 4px;

    margin: 0 auto 35px auto;

    border-radius: 10px;

    background: linear-gradient(
        90deg,
        #36d1dc,
        #8e54e9
    );
}

/* ==========================================================
   FRONT CARD
   ========================================================== */

.front-card {

    max-width: 900px;

    margin: 15vh auto 40px auto;

    padding: 55px 45px;

    text-align: center;

    border-radius: 30px;

    background:
        rgba(15, 23, 50, 0.82);

    border:
        1px solid
        rgba(255,255,255,0.12);

    box-shadow:
        0 25px 70px
        rgba(0,0,0,0.45);

    backdrop-filter: blur(20px);
}

/* ==========================================================
   DASHBOARD HEADER
   ========================================================== */

.dashboard-card {

    max-width: 1100px;

    margin: 0 auto 25px auto;

    padding: 30px;

    text-align: center;

    border-radius: 25px;

    background:
        rgba(15, 22, 48, 0.80);

    border:
        1px solid
        rgba(255,255,255,0.12);

    box-shadow:
        0 20px 55px
        rgba(0,0,0,0.30);
}

.dashboard-title {

    font-size: 36px;

    font-weight: 900;

    color: white;

    text-align: center;

}

.dashboard-subtitle {

    margin-top: 8px;

    font-size: 16px;

    color: #b8c5dc;

}

/* ==========================================================
   SECTION
   ========================================================== */

.section-card {

    max-width: 1100px;

    margin: 20px auto;

    padding: 25px;

    border-radius: 22px;

    background:
        rgba(15, 22, 48, 0.72);

    border:
        1px solid
        rgba(255,255,255,0.10);

}

/* ==========================================================
   SUBJECT
   ========================================================== */

.subject-card {

    padding: 18px;

    margin: 10px 0;

    border-radius: 15px;

    background:
        rgba(255,255,255,0.05);

    border:
        1px solid
        rgba(255,255,255,0.08);

}

/* ==========================================================
   LOGIN
   ========================================================== */

.login-card {

    max-width: 600px;

    margin: 30px auto;

    padding: 35px;

    border-radius: 25px;

    background:
        rgba(15,22,48,0.82);

    border:
        1px solid
        rgba(255,255,255,0.12);

}

/* ==========================================================
   BUTTON
   ========================================================== */

.stButton > button {

    min-height: 45px;

    border-radius: 13px;

    font-weight: 700;

}

/* ==========================================================
   MOBILE
   ========================================================== */

@media(max-width: 700px) {

    .main-title {
        font-size: 30px;
    }

    .dashboard-title {
        font-size: 27px;
    }

    .front-card {
        margin-top: 8vh;
        padding: 35px 20px;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 8. SESSION STATE
# ============================================================

defaults = {
    "page": "home",
    "student_logged": False,
    "tutor_logged": False,
    "student_id": "",
    "student_username": "",
    "tutor_branch": "",
    "tutor_username": "",
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# 9. DATA FILE
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

        return pd.read_csv(
            DATA_FILE
        )

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
# 10. PERFORMANCE CALCULATION
# ============================================================

def attendance_mark(attendance):

    attendance = float(
        attendance
    )

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


def calculate_performance(row):

    attendance = float(
        row["Attendance"]
    )

    study = float(
        row["Study_Hours"]
    )

    internal = float(
        row["Internal"]
    )

    assignment = float(
        row["Assignment"]
    )

    previous = float(
        row["Previous_Mark"]
    )

    att_mark = attendance_mark(
        attendance
    )

    attendance_percentage = (
        att_mark / 5
    ) * 100

    study_percentage = min(
        study / 6 * 100,
        100
    )

    internal_percentage = (
        internal / 40
    ) * 100

    assignment_percentage = (
        assignment / 15
    ) * 100

    previous_percentage = (
        previous / 60
    ) * 100

    overall = np.mean([
        attendance_percentage,
        study_percentage,
        internal_percentage,
        assignment_percentage,
        previous_percentage,
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
            "Above Average Performance",
            "🟡"
        )

    else:

        return (
            overall,
            "Good Performance",
            "🟢"
        )


# ============================================================
# 11. STUDENT PASSWORD
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
# 12. HOME PAGE
# ============================================================

def home_page():

    st.markdown(
        "<div class='front-card'>",
        unsafe_allow_html=True
    )

    # IMPORTANT:
    # st.title guarantees that the heading is rendered.

    st.markdown(
        "<div class='main-title'>🎓 STUDENT PERFORMANCE PREDICTION</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='main-title-line'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align:center;'>Select Login</h3>",
        unsafe_allow_html=True
    )

    left, student_col, tutor_col, right = st.columns(
        [1, 2, 2, 1]
    )

    with student_col:

        if st.button(
            "🎓 STUDENT LOGIN",
            width="stretch"
        ):

            st.session_state.page = (
                "student_login"
            )

            st.rerun()

    with tutor_col:

        if st.button(
            "👨‍🏫 TUTOR LOGIN",
            width="stretch"
        ):

            st.session_state.page = (
                "tutor_login"
            )

            st.rerun()


# ============================================================
# 13. STUDENT LOGIN
# ============================================================

def student_login():

    st.markdown(
        "<div class='dashboard-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='dashboard-title'>🎓 Student Login</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='login-card'>",
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
                    "No student records are available."
                )

            else:

                found = df[
                    df["University_ID"]
                    .astype(str)
                    .str.strip()
                    ==
                    university_id.strip()
                ]

                if found.empty:

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
# 14. TUTOR LOGIN
# ============================================================

def tutor_login():

    st.markdown(
        "<div class='dashboard-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='dashboard-title'>👨‍🏫 Tutor Login</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='login-card'>",
        unsafe_allow_html=True
    )

    department = st.selectbox(
        "Department",
        DEPARTMENTS
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
            TUTOR_ACCOUNTS[
                department
            ]
        )

        if (
            username == correct_username
            and
            password == correct_password
        ):

            st.session_state.tutor_logged = True

            st.session_state.tutor_branch = (
                department
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
                "Invalid username or password."
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
# 15. STUDENT DASHBOARD
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
        student_id
    ].copy()

    if student_df.empty:

        st.error(
            "Student data not found."
        )

        return

    student_name = str(
        student_df.iloc[0]["Name"]
    )

    department = str(
        student_df.iloc[0]["Branch"]
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        "<div class='dashboard-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='dashboard-title'>🎓 {student_name}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='dashboard-subtitle'>"
        "Student Performance Dashboard"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DEPARTMENT
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="section-card">

            <h3 style="text-align:center;">
                Department
            </h3>

            <p style="
                text-align:center;
                font-size:18px;
            ">
                {department}
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
        department,
        semester
    )

    if not subjects:

        st.warning(
            "The selected department has not yet been mapped "
            "to the verified KTU curriculum in this project."
        )

        return

    # --------------------------------------------------------
    # KTU SUBJECT LIST
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="section-card">

            <h2 style="text-align:center;">
                📚 {semester} — KTU Subjects
            </h2>

        </div>
        """,
        unsafe_allow_html=True
    )

    for number, subject in enumerate(
        subjects,
        start=1
    ):

        st.markdown(
            f"""
            <div class="subject-card">

                <b>
                    {number}. {subject}
                </b>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # STUDENT MARKS
    # --------------------------------------------------------

    current = student_df[
        student_df["Semester"].astype(str)
        ==
        semester
    ].copy()

    if current.empty:

        st.info(
            "Marks have not been entered by the tutor "
            "for this semester."
        )

    else:

        st.markdown(
            f"""
            <div class="section-card">

                <h2 style="text-align:center;">
                    📊 {semester} Performance
                </h2>

            </div>
            """,
            unsafe_allow_html=True
        )

        for _, row in current.iterrows():

            percentage, level, emoji = (
                calculate_performance(
                    row
                )
            )

            st.markdown(
                f"""
                <div class="section-card">

                    <h3 style="text-align:center;">
                        📘 {row["Subject"]}
                    </h3>

                    <h2 style="text-align:center;">
                        {emoji}
                    </h2>

                    <h3 style="text-align:center;">
                        {level}
                    </h3>

                    <h2 style="text-align:center;">
                        {percentage:.2f}%
                    </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric(
                    "Attendance",
                    f'{row["Attendance"]}%'
                )

            with c2:
                st.metric(
                    "Internal",
                    f'{row["Internal"]}/40'
                )

            with c3:
                st.metric(
                    "Assignment",
                    f'{row["Assignment"]}/15'
                )

            with c4:
                st.metric(
                    "Previous",
                    f'{row["Previous_Mark"]}/60'
                )

            with c5:
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
# 16. TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    department = (
        st.session_state.tutor_branch
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        "<div class='dashboard-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='dashboard-title'>"
        "👨‍🏫 Tutor Dashboard"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='dashboard-subtitle'>"
        f"{department}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
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
        ],
        key="tutor_semester"
    )

    subjects = get_subjects(
        department,
        semester
    )

    if not subjects:

        st.warning(
            "This department has not been mapped "
            "to a verified curriculum in this version."
        )

        if st.button(
            "🚪 LOGOUT",
            width="stretch"
        ):
            logout()

        return

    # --------------------------------------------------------
    # CURRICULUM DISPLAY
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="section-card">

            <h2 style="text-align:center;">
                📚 {semester} Subjects
            </h2>

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

    add_tab, data_tab, kmeans_tab, rf_tab = st.tabs(
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
            "Add Student"
        )

        student_name = st.text_input(
            "Student Name"
        )

        university_id = st.text_input(
            "University ID"
        )

        st.subheader(
            "Enter Performance Details"
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
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    key=f"attendance_{semester}_{subject}"
                )

            with c2:

                study_hours = st.number_input(
                    "Study Hours",
                    min_value=0.0,
                    max_value=6.0,
                    value=3.0,
                    step=0.5,
                    key=f"study_{semester}_{subject}"
                )

            with c3:

                internal = st.number_input(
                    "Internal /40",
                    min_value=0.0,
                    max_value=40.0,
                    value=25.0,
                    step=1.0,
                    key=f"internal_{semester}_{subject}"
                )

            with c4:

                assignment = st.number_input(
                    "Assignment /15",
                    min_value=0.0,
                    max_value=15.0,
                    value=10.0,
                    step=1.0,
                    key=f"assignment_{semester}_{subject}"
                )

            with c5:

                previous = st.number_input(
                    "Previous /60",
                    min_value=0.0,
                    max_value=60.0,
                    value=35.0,
                    step=1.0,
                    key=f"previous_{semester}_{subject}"
                )

            records.append(
                {
                    "Name": student_name,
                    "University_ID": university_id,
                    "Branch": department,
                    "Semester": semester,
                    "Subject": subject,
                    "Attendance": attendance,
                    "Study_Hours": study_hours,
                    "Internal": internal,
                    "Assignment": assignment,
                    "Previous_Mark": previous,
                }
            )

        if st.button(
            "💾 SAVE STUDENT RECORDS",
            width="stretch"
        ):

            if not student_name.strip():

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
    # DATA
    # ========================================================

    with data_tab:

        st.header(
            "Student Data"
        )

        df = load_data()

        if df.empty:

            st.info(
                "No data available."
            )

        else:

            branch_df = df[
                df["Branch"].astype(str)
                ==
                department
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
                    file_name="branch_student_data.csv",
                    mime="text/csv",
                    width="stretch"
                )

                st.subheader(
                    "Delete Student"
                )

                student_ids = (
                    branch_df[
                        "University_ID"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_id = st.selectbox(
                    "University ID",
                    student_ids
                )

                if st.button(
                    "🗑️ DELETE COMPLETE HISTORY",
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
                                department
                            )
                        )
                    ]

                    save_data(
                        remaining
                    )

                    st.success(
                        "Student history deleted."
                    )

                    st.rerun()

    # ========================================================
    # K-MEANS
    # ========================================================

    with kmeans_tab:

        st.header(
            "📊 K-Means Clustering"
        )

        df = load_data()

        branch_df = df[
            df["Branch"].astype(str)
            ==
            department
        ].copy()

        features = [
            "Attendance",
            "Study_Hours",
            "Internal",
            "Assignment",
            "Previous_Mark",
        ]

        if len(branch_df) < 3:

            st.warning(
                "At least 3 student records are required."
            )

        else:

            X = branch_df[
                features
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            valid = X.dropna()

            if len(valid) >= 3:

                number_of_clusters = min(
                    3,
                    len(valid)
                )

                model = KMeans(
                    n_clusters=number_of_clusters,
                    random_state=42,
                    n_init=10
                )

                cluster_values = model.fit_predict(
                    valid
                )

                score = silhouette_score(
                    valid,
                    cluster_values
                )

                st.metric(
                    "Silhouette Score",
                    f"{score:.3f}"
                )

                result = branch_df.loc[
                    valid.index
                ].copy()

                result[
                    "KMeans_Cluster"
                ] = cluster_values

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
            department
        ].copy()

        features = [
            "Attendance",
            "Study_Hours",
            "Internal",
            "Assignment",
            "Previous_Mark",
        ]

        if len(branch_df) < 10:

            st.warning(
                "At least 10 records are recommended "
                "for Random Forest."
            )

        else:

            X = branch_df[
                features
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            labels = []

            valid_indices = []

            for index, row in branch_df.iterrows():

                try:

                    _, level, _ = (
                        calculate_performance(
                            row
                        )
                    )

                    labels.append(
                        level
                    )

                    valid_indices.append(
                        index
                    )

                except Exception:
                    continue

            X = X.loc[
                valid_indices
            ]

            y = np.array(
                labels
            )

            valid_rows = ~X.isna().any(
                axis=1
            )

            X = X[
                valid_rows
            ]

            y = y[
                valid_rows.values
            ]

            if (
                len(X) >= 10
                and
                len(np.unique(y)) >= 2
            ):

                try:

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y
                        )
                    )

                except Exception:

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42
                        )
                    )

                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )

                model.fit(
                    X_train,
                    y_train
                )

                prediction = model.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    prediction
                )

                st.metric(
                    "Random Forest Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

            else:

                st.warning(
                    "At least two performance classes "
                    "are required."
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
# 17. LOGOUT
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
# 18. START APPLICATION
# ============================================================

create_data_file()

if st.session_state.page == "home":

    home_page()

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
