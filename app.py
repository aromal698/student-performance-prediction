# ============================================================
# STUDENT PERFORMANCE PREDICTION
# B.Tech S3 - AI & Data Science
# KTU 2024 Scheme - Project Demo
# ============================================================

import os
import re
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

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
    PageBreak,
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
# CONSTANTS
# ============================================================

DATA_DIR = "data"
DATABASE_FILE = os.path.join(DATA_DIR, "student_records.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------
# KTU S3 AI & DATA SCIENCE SUBJECTS
# ------------------------------------------------------------

SUBJECTS = [
    "Mathematics for Information Science-III",
    "Foundations of Artificial Intelligence",
    "Data Structures and Algorithms",
    "Introduction to Data Science",
    "Digital Electronics & Logic Design",
    "Economics / Engineering Ethics",
]

# ------------------------------------------------------------
# MARK LIMITS
# ------------------------------------------------------------

MAX_ATTENDANCE = 100
MAX_INTERNAL = 40
MAX_ASSIGNMENT = 15
MAX_PREVIOUS = 60
MAX_STUDY_HOURS = 6

# Demo tutor account
TUTOR_USERNAME = "teacher_aids"
TUTOR_PASSWORD = "ktutech"

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "page": "home",
    "role": None,
    "logged_in": False,
    "student_username": None,
    "student_id": None,
    "tutor_username": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# CSS - DYNAMIC / 3D FRONT PAGE
# ============================================================

def load_css():

    st.markdown(
        """
        <style>

        /* ==================================================
           GLOBAL
        ================================================== */

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 20%,
                    rgba(50, 100, 255, 0.25),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 90% 15%,
                    rgba(180, 60, 255, 0.20),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 50% 90%,
                    rgba(0, 200, 255, 0.15),
                    transparent 35%
                ),
                #050816;

            color: white;
            overflow-x: hidden;
        }

        /* ==================================================
           3D MOVING GRID
        ================================================== */

        .stApp::before {
            content: "";
            position: fixed;
            left: -50%;
            top: 45%;
            width: 200%;
            height: 120%;

            background-image:
                linear-gradient(
                    rgba(80, 150, 255, 0.08) 1px,
                    transparent 1px
                ),
                linear-gradient(
                    90deg,
                    rgba(80, 150, 255, 0.08) 1px,
                    transparent 1px
                );

            background-size: 55px 55px;

            transform:
                perspective(500px)
                rotateX(60deg);

            animation: gridMove 12s linear infinite;

            pointer-events: none;
            z-index: -5;
        }

        @keyframes gridMove {
            0% {
                transform:
                    perspective(500px)
                    rotateX(60deg)
                    translateY(0);
            }

            100% {
                transform:
                    perspective(500px)
                    rotateX(60deg)
                    translateY(100px);
            }
        }

        /* ==================================================
           FLOATING ORBS
        ================================================== */

        .orb {
            position: fixed;
            border-radius: 50%;
            pointer-events: none;
            z-index: -3;
        }

        .orb1 {
            width: 240px;
            height: 240px;
            left: 4%;
            top: 12%;

            background:
                radial-gradient(
                    circle at 30% 25%,
                    rgba(255,255,255,0.8),
                    rgba(70,110,255,0.35) 25%,
                    transparent 70%
                );

            filter: blur(2px);

            animation: orb1move 8s ease-in-out infinite;
        }

        .orb2 {
            width: 190px;
            height: 190px;
            right: 7%;
            top: 22%;

            background:
                radial-gradient(
                    circle at 30% 25%,
                    rgba(255,255,255,0.7),
                    rgba(190,60,255,0.35) 25%,
                    transparent 70%
                );

            filter: blur(2px);

            animation: orb2move 10s ease-in-out infinite;
        }

        .orb3 {
            width: 150px;
            height: 150px;
            left: 45%;
            bottom: 5%;

            background:
                radial-gradient(
                    circle at 30% 25%,
                    rgba(255,255,255,0.7),
                    rgba(0,210,255,0.3) 25%,
                    transparent 70%
                );

            filter: blur(2px);

            animation: orb3move 7s ease-in-out infinite;
        }

        @keyframes orb1move {
            0%,100% {
                transform: translate(0,0);
            }
            50% {
                transform: translate(90px,60px) scale(1.12);
            }
        }

        @keyframes orb2move {
            0%,100% {
                transform: translate(0,0);
            }
            50% {
                transform: translate(-80px,70px) scale(0.9);
            }
        }

        @keyframes orb3move {
            0%,100% {
                transform: translate(0,0);
            }
            50% {
                transform: translate(50px,-60px);
            }
        }

        /* ==================================================
           FRONT PAGE
        ================================================== */

        .front {
            text-align: center;
            padding: 35px 20px 20px 20px;
        }

        .graduation-icon {
            font-size: 75px;

            animation:
                graduationFloat 3s ease-in-out infinite;

            text-shadow:
                0 0 20px rgba(80,140,255,0.7);
        }

        @keyframes graduationFloat {
            0%,100% {
                transform: translateY(0);
            }

            50% {
                transform: translateY(-10px);
            }
        }

        .main-title {
            font-size: 44px;
            font-weight: 800;
            color: white;

            text-shadow:
                0 0 25px rgba(80,130,255,0.45);
        }

        .subtitle {
            color: rgba(255,255,255,0.68);
            font-size: 18px;
            line-height: 1.6;
            margin-top: 10px;
        }

        .choose-login {
            margin-top: 22px;
            font-size: 20px;
            font-weight: 600;
            color: rgba(255,255,255,0.85);
        }

        /* ==================================================
           GLASS CARDS
        ================================================== */

        .role-card {
            min-height: 235px;

            padding: 30px 20px;

            border-radius: 25px;

            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.14),
                    rgba(255,255,255,0.04)
                );

            border:
                1px solid rgba(255,255,255,0.17);

            backdrop-filter: blur(18px);

            box-shadow:
                inset 0 1px 1px rgba(255,255,255,0.15),
                0 20px 60px rgba(0,0,0,0.35);

            text-align: center;

            transition: 0.35s ease;
        }

        .role-card:hover {
            transform: translateY(-10px);

            border-color:
                rgba(100,170,255,0.5);

            box-shadow:
                0 25px 80px rgba(40,100,255,0.25);
        }

        .role-icon {
            font-size: 52px;
            margin-bottom: 8px;
        }

        .role-title {
            color: white;
            font-size: 29px;
            font-weight: 700;
        }

        .role-description {
            color: rgba(255,255,255,0.65);
            font-size: 16px;
            line-height: 1.7;
            margin-top: 10px;
        }

        /* ==================================================
           BUTTONS
        ================================================== */

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 13px;

            min-height: 45px;

            font-weight: 600;

            border:
                1px solid rgba(255,255,255,0.18);

            background:
                linear-gradient(
                    135deg,
                    rgba(65,105,255,0.85),
                    rgba(125,60,220,0.85)
                );

            color: white;

            transition: 0.25s ease;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            transform: translateY(-3px);

            box-shadow:
                0 10px 30px rgba(80,100,255,0.35);
        }

        /* ==================================================
           DASHBOARD CARDS
        ================================================== */

        .dashboard-card {
            padding: 22px;
            border-radius: 20px;

            background:
                rgba(255,255,255,0.07);

            border:
                1px solid rgba(255,255,255,0.12);

            backdrop-filter: blur(12px);

            margin-bottom: 15px;
        }

        .dashboard-title {
            color: white;
            font-size: 28px;
            font-weight: 700;
        }

        .small-muted {
            color: rgba(255,255,255,0.60);
        }

        /* ==================================================
           PERFORMANCE CIRCLES
        ================================================== */

        .performance-circle {
            width: 15px;
            height: 15px;

            border-radius: 50%;

            display: inline-block;

            margin-right: 7px;

            vertical-align: middle;
        }

        .red {
            background: #ff3b30;
            box-shadow: 0 0 10px rgba(255,59,48,0.55);
        }

        .orange {
            background: #ff9500;
            box-shadow: 0 0 10px rgba(255,149,0,0.55);
        }

        .yellow {
            background: #ffd60a;
            box-shadow: 0 0 10px rgba(255,214,10,0.55);
        }

        .green {
            background: #30d158;
            box-shadow: 0 0 10px rgba(48,209,88,0.55);
        }

        /* ==================================================
           TABLE
        ================================================== */

        .subject-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .subject-table th,
        .subject-table td {
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.12);
            text-align: center;
        }

        .subject-table th {
            background: rgba(80,120,255,0.18);
            color: white;
        }

        .subject-table td {
            background: rgba(255,255,255,0.04);
        }

        </style>

        <div class="orb orb1"></div>
        <div class="orb orb2"></div>
        <div class="orb orb3"></div>
        """,
        unsafe_allow_html=True,
    )


load_css()

# ============================================================
# DATABASE
# ============================================================

COLUMNS = [
    "Student_Name",
    "Username",
    "University_ID",
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


def load_data():

    if not os.path.exists(DATABASE_FILE):
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(DATABASE_FILE)

        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""

        return df[COLUMNS]

    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def save_data(df):

    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(
        DATABASE_FILE,
        index=False
    )


# ============================================================
# VALIDATION
# ============================================================

def valid_student_password(password):

    """
    Required format:
    BTECH + 4 digits
    Last four digits: 2000-2022

    Example:
    BTECH2000
    BTECH2015
    BTECH2022
    """

    if not isinstance(password, str):
        return False

    if not re.fullmatch(r"BTECH\d{4}", password):
        return False

    year = int(password[-4:])

    return 2000 <= year <= 2022


# ============================================================
# ATTENDANCE CONVERSION
# ============================================================

def attendance_mark(attendance):

    attendance = float(attendance)

    if attendance >= 90:
        return 5

    if attendance >= 80:
        return 4

    if attendance >= 70:
        return 3

    if attendance >= 60:
        return 2

    if attendance >= 10:
        return 1

    return 0


# ============================================================
# PERFORMANCE CALCULATION
# ============================================================

def calculate_performance(
    attendance,
    internal,
    assignment,
    previous,
    study_hours,
):

    attendance = float(attendance)
    internal = float(internal)
    assignment = float(assignment)
    previous = float(previous)
    study_hours = float(study_hours)

    attendance_score = (
        attendance_mark(attendance) / 5
    ) * 100

    internal_score = (
        internal / MAX_INTERNAL
    ) * 100

    assignment_score = (
        assignment / MAX_ASSIGNMENT
    ) * 100

    previous_score = (
        previous / MAX_PREVIOUS
    ) * 100

    study_score = (
        min(study_hours, MAX_STUDY_HOURS)
        / MAX_STUDY_HOURS
    ) * 100

    overall = np.mean([
        attendance_score,
        internal_score,
        assignment_score,
        previous_score,
        study_score,
    ])

    if overall < 50:
        level = "Low Performance"
    elif overall < 65:
        level = "Average Performance"
    elif overall < 80:
        level = "Above Average"
    else:
        level = "Good Performance"

    return level, round(float(overall), 2)


# ============================================================
# PERFORMANCE STYLE
# ============================================================

def performance_style(level):

    if level == "Low Performance":
        return "red", "⚠️"

    if level == "Average Performance":
        return "orange", "🟠"

    if level == "Above Average":
        return "yellow", "🟡"

    return "green", "🟢"


# ============================================================
# ADVICE
# ============================================================

def improvement_advice(row):

    level = row["Performance"]

    if level == "Good Performance":

        return (
            "Excellent performance! Keep up the good work. "
            "Maintain your attendance, study routine and assignment performance."
        )

    if level == "Above Average":

        return (
            "Good progress. A small improvement in attendance, "
            "internal marks, assignments and regular study can move "
            "your performance towards the good-performance level."
        )

    if level == "Average Performance":

        return (
            "Maintain a regular study schedule, improve attendance, "
            "complete assignments on time and revise important topics regularly."
        )

    advice = []

    if float(row["Attendance"]) < 70:
        advice.append("Improve attendance")

    if float(row["Internal_Mark"]) < 25:
        advice.append("Improve internal-test preparation")

    if float(row["Assignment"]) < 9:
        advice.append("Complete assignments regularly")

    if float(row["Previous_Mark"]) < 35:
        advice.append("Revise previous topics")

    if float(row["Study_Hours"]) < 3:
        advice.append("Increase daily study time")

    if not advice:
        advice.append("Follow a consistent study schedule")

    return "Needs improvement: " + ", ".join(advice) + "."


# ============================================================
# MODEL TRAINING
# ============================================================

FEATURES = [
    "Attendance",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark",
    "Study_Hours",
]


def prepare_model_data(df):

    if df.empty:
        return None

    temp = df.copy()

    for col in FEATURES:
        temp[col] = pd.to_numeric(
            temp[col],
            errors="coerce"
        )

    temp = temp.dropna(
        subset=FEATURES
    )

    return temp


def run_kmeans(df):

    temp = prepare_model_data(df)

    if temp is None or len(temp) < 2:
        return df.copy()

    scaler = StandardScaler()

    X = scaler.fit_transform(
        temp[FEATURES]
    )

    n_clusters = min(
        3,
        len(temp)
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X)

    result = df.copy()

    result["KMeans_Cluster"] = ""

    result.loc[
        temp.index,
        "KMeans_Cluster"
    ] = labels

    return result


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(df):

    temp = prepare_model_data(df)

    if temp is None or len(temp) < 5:
        return None

    X = temp[FEATURES]

    y = temp["Performance"]

    if y.nunique() < 2:
        return None

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(
        X_scaled,
        y
    )

    return scaler, model


def rf_predict(df, row):

    trained = train_random_forest(df)

    if trained is None:
        return row["Performance"]

    scaler, model = trained

    X = pd.DataFrame([{
        "Attendance": row["Attendance"],
        "Internal_Mark": row["Internal_Mark"],
        "Assignment": row["Assignment"],
        "Previous_Mark": row["Previous_Mark"],
        "Study_Hours": row["Study_Hours"],
    }])

    X_scaled = scaler.transform(X)

    return model.predict(X_scaled)[0]


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf(student_df):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8,
    )

    story = []

    if student_df.empty:
        return None

    student_name = str(
        student_df.iloc[0]["Student_Name"]
    )

    username = str(
        student_df.iloc[0]["Username"]
    )

    university_id = str(
        student_df.iloc[0]["University_ID"]
    )

    story.append(
        Paragraph(
            "Student Performance Progress Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "B.Tech S3 - Artificial Intelligence & Data Science",
            ParagraphStyle(
                "center",
                parent=styles["Normal"],
                alignment=TA_CENTER,
            ),
        )
    )

    story.append(Spacer(1, 15))

    # --------------------------------------------------------
    # STUDENT DETAILS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Student Details",
            heading_style,
        )
    )

    details = [
        ["Student Name", student_name],
        ["Username", username],
        ["University ID", university_id],
        ["Semester", "S3"],
        ["Branch", "Artificial Intelligence & Data Science"],
    ]

    table = Table(
        details,
        colWidths=[150, 350],
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("PADDING", (0,0), (-1,-1), 7),
        ])
    )

    story.append(table)

    # --------------------------------------------------------
    # SUBJECT REPORTS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Subject-wise Performance",
            heading_style,
        )
    )

    for _, row in student_df.iterrows():

        subject = str(row["Subject"])

        level = str(row["Performance"])

        percentage = float(
            row["Performance_Percent"]
        )

        cluster = row["KMeans_Cluster"]

        story.append(
            Paragraph(
                subject,
                ParagraphStyle(
                    "subject",
                    parent=styles["Heading3"],
                    fontSize=12,
                    spaceBefore=10,
                ),
            )
        )

        attendance = float(row["Attendance"])

        attendance_converted = attendance_mark(
            attendance
        )

        subject_data = [
            ["Assessment", "Score", "Maximum"],
            [
                "Attendance",
                f"{attendance:.0f}%",
                "100%",
            ],
            [
                "Attendance Converted",
                str(attendance_converted),
                "5",
            ],
            [
                "Internal Mark",
                f"{float(row['Internal_Mark']):.1f}",
                "40",
            ],
            [
                "Assignment",
                f"{float(row['Assignment']):.1f}",
                "15",
            ],
            [
                "Previous Mark",
                f"{float(row['Previous_Mark']):.1f}",
                "60",
            ],
            [
                "Study Hours",
                f"{float(row['Study_Hours']):.1f}",
                "6 hours",
            ],
        ]

        subject_table = Table(
            subject_data,
            colWidths=[200, 150, 150],
        )

        subject_table.setStyle(
            TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dbe7ff")),
                ("PADDING", (0,0), (-1,-1), 6),
            ])
        )

        story.append(subject_table)

        story.append(Spacer(1, 7))

        story.append(
            Paragraph(
                f"<b>Performance:</b> {level} "
                f"({percentage:.2f}%)",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"<b>K-Means Cluster:</b> {cluster}",
                styles["Normal"],
            )
        )

        advice = improvement_advice(row)

        story.append(
            Paragraph(
                f"<b>Feedback:</b> {advice}",
                styles["Normal"],
            )
        )

        story.append(Spacer(1, 10))

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Overall Performance",
            heading_style,
        )
    )

    overall = student_df[
        "Performance_Percent"
    ].astype(float).mean()

    if overall < 50:
        overall_level = "Low Performance"
    elif overall < 65:
        overall_level = "Average Performance"
    elif overall < 80:
        overall_level = "Above Average"
    else:
        overall_level = "Good Performance"

    overall_table = Table(
        [
            ["Overall Percentage", f"{overall:.2f}%"],
            ["Overall Performance", overall_level],
        ],
        colWidths=[200, 300],
    )

    overall_table.setStyle(
        TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
            ("PADDING", (0,0), (-1,-1), 8),
        ])
    )

    story.append(overall_table)

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Assessment Conversion",
            heading_style,
        )
    )

    conversion_text = """
    Attendance conversion:
    90-100% = 5 marks,
    80-89% = 4 marks,
    70-79% = 3 marks,
    60-69% = 2 marks,
    10-59% = 1 mark,
    below 10% = 0 marks.
    """

    story.append(
        Paragraph(
            conversion_text,
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Note: The performance percentage and performance levels "
            "in this project are project-defined indicators for academic "
            "monitoring and are not official KTU grades.",
            styles["Normal"],
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# FRONT DASHBOARD
# ============================================================

def front_dashboard():

    st.markdown(
        """
        <div class="front">

            <div class="graduation-icon">
                🎓
            </div>

            <div class="main-title">
                Student Performance Prediction
            </div>

            <div class="subtitle">
                AI-powered academic performance<br>
                monitoring system
            </div>

            <div class="choose-login">
                Choose Login
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        st.markdown(
            """
            <div class="role-card">

                <div class="role-icon">
                    🎓
                </div>

                <div class="role-title">
                    Student
                </div>

                <div class="role-description">
                    View academic performance,
                    subject marks and progress report.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "🎓 Student Login",
            key="student_front",
            width="stretch",
        ):

            st.session_state["page"] = "student_login"

            st.rerun()

    with col2:

        st.markdown(
            """
            <div class="role-card">

                <div class="role-icon">
                    👨‍🏫
                </div>

                <div class="role-title">
                    Tutor
                </div>

                <div class="role-description">
                    Manage student data,
                    analyse performance and clustering.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "👨‍🏫 Tutor Login",
            key="tutor_front",
            width="stretch",
        ):

            st.session_state["page"] = "tutor_login"

            st.rerun()

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:35px;
            color:rgba(255,255,255,0.4);
        ">
            B.Tech S3 · Artificial Intelligence & Data Science
            <br>
            KTU · Student Performance Prediction
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    st.markdown(
        "<h1 style='text-align:center;'>👨‍🏫 Tutor Login</h1>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1,2,1]
    )

    with col2:

        username = st.text_input(
            "Tutor Username"
        )

        password = st.text_input(
            "Tutor Password",
            type="password"
        )

        if st.button(
            "Login",
            width="stretch"
        ):

            if (
                username == TUTOR_USERNAME
                and password == TUTOR_PASSWORD
            ):

                st.session_state["role"] = "tutor"
                st.session_state["logged_in"] = True
                st.session_state["tutor_username"] = username
                st.session_state["page"] = "tutor_dashboard"

                st.success("Tutor login successful.")

                st.rerun()

            else:

                st.error(
                    "Invalid tutor username or password."
                )

        if st.button(
            "← Back",
            width="stretch"
        ):

            st.session_state["page"] = "home"

            st.rerun()


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.markdown(
        "<h1 style='text-align:center;'>🎓 Student Login</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="
            text-align:center;
            color:rgba(255,255,255,0.65);
        ">
            Enter your username, University ID and project password.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1,2,1]
    )

    with col2:

        username = st.text_input(
            "Username",
            key="student_username_login"
        )

        university_id = st.text_input(
            "University ID",
            key="student_id_login"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="student_password_login"
        )

        if st.button(
            "View My Result",
            width="stretch"
        ):

            if not username or not university_id:

                st.error(
                    "Enter both username and University ID."
                )

            elif not valid_student_password(password):

                st.error(
                    "Password must be BTECH followed by a year from 2000 to 2022. "
                    "Example: BTECH2007"
                )

            else:

                df = load_data()

                matches = df[
                    (
                        df["Username"].astype(str).str.lower()
                        == username.strip().lower()
                    )
                    &
                    (
                        df["University_ID"].astype(str).str.lower()
                        == university_id.strip().lower()
                    )
                ]

                if matches.empty:

                    st.error(
                        "Student record not found. "
                        "Please check your Username and University ID."
                    )

                else:

                    st.session_state["role"] = "student"
                    st.session_state["logged_in"] = True
                    st.session_state["student_username"] = username.strip()
                    st.session_state["student_id"] = university_id.strip()
                    st.session_state["page"] = "student_dashboard"

                    st.rerun()

        if st.button(
            "← Back",
            width="stretch"
        ):

            st.session_state["page"] = "home"

            st.rerun()


# ============================================================
# TUTOR - STUDENT ENTRY
# ============================================================

def tutor_add_student():

    st.subheader(
        "➕ Add Student Subject Records"
    )

    st.caption(
        "Enter one student's marks for all six S3 subjects. "
        "The complete student record will be saved together."
    )

    # --------------------------------------------------------
    # STUDENT DETAILS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        name = st.text_input(
            "Student Name",
            key="new_student_name"
        )

    with c2:
        username = st.text_input(
            "Student Username",
            key="new_student_username"
        )

    with c3:
        university_id = st.text_input(
            "University ID",
            key="new_student_id"
        )

    st.markdown(
        "### 📊 Subject-wise Assessment"
    )

    # --------------------------------------------------------
    # HORIZONTAL SUBJECT HEADERS
    # --------------------------------------------------------

    cols = st.columns(
        [2] + [1.5] * len(SUBJECTS)
    )

    cols[0].markdown(
        "**Assessment**"
    )

    for i, subject in enumerate(SUBJECTS):

        short_name = subject.replace(
            "Mathematics for Information Science-III",
            "Maths-III"
        ).replace(
            "Foundations of Artificial Intelligence",
            "Foundations of AI"
        ).replace(
            "Data Structures and Algorithms",
            "DSA"
        ).replace(
            "Introduction to Data Science",
            "Intro to Data Science"
        ).replace(
            "Digital Electronics & Logic Design",
            "Digital Electronics"
        ).replace(
            "Economics / Engineering Ethics",
            "Economics / Ethics"
        )

        cols[i + 1].markdown(
            f"**{short_name}**"
        )

    # --------------------------------------------------------
    # ATTENDANCE
    # --------------------------------------------------------

    attendance_values = []

    row = st.columns(
        [2] + [1.5] * len(SUBJECTS)
    )

    row[0].markdown(
        "**Attendance / 100**"
    )

    for i in range(len(SUBJECTS)):

        attendance_values.append(
            row[i + 1].number_input(
                "Attendance",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                key=f"att_{i}",
                label_visibility="collapsed",
            )
        )

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    internal_values = []

    row = st.columns(
        [2] + [1.5] * len(SUBJECTS)
    )

    row[0].markdown(
        "**Internal / 40**"
    )

    for i in range(len(SUBJECTS)):

        internal_values.append(
            row[i + 1].number_input(
                "Internal",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                step=1.0,
                key=f"internal_{i}",
                label_visibility="collapsed",
            )
        )

    # --------------------------------------------------------
    # ASSIGNMENT
    # --------------------------------------------------------

    assignment_values = []

    row = st.columns(
        [2] + [1.5] * len(SUBJECTS)
    )

    row[0].markdown(
        "**Assignment / 15**"
    )

    for i in range(len(SUBJECTS)):

        assignment_values.append(
            row[i + 1].number_input(
                "Assignment",
                min_value=0.0,
                max_value=15.0,
                value=10.0,
                step=1.0,
                key=f"assignment_{i}",
                label_visibility="collapsed",
            )
        )

    # --------------------------------------------------------
    # PREVIOUS MARK
    # --------------------------------------------------------

    previous_values = []

    row = st.columns(
        [2] + [1.5] * len(SUBJECTS)
    )

    row[0].markdown(
        "**Previous Mark / 60**"
    )

    for i in range(len(SUBJECTS)):

        previous_values.append(
            row[i + 1].number_input(
                "Previous",
                min_value=0.0,
                max_value=60.0,
                value=35.0,
                step=1.0,
                key=f"previous_{i}",
                label_visibility="collapsed",
            )
        )

    st.info(
        "Study hours are entered by the student after login. "
        "The tutor does not enter study hours."
    )

    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    if st.button(
        "💾 Submit Student Records",
        width="stretch",
        type="primary"
    ):

        if not name.strip():
            st.error("Enter student name.")

            return

        if not username.strip():
            st.error("Enter student username.")

            return

        if not university_id.strip():
            st.error("Enter University ID.")

            return

        df = load_data()

        # Delete existing records for this student first
        # so submitting again updates the complete student.
        if not df.empty:

            df = df[
                ~(
                    (
                        df["University_ID"].astype(str).str.lower()
                        == university_id.strip().lower()
                    )
                    |
                    (
                        df["Username"].astype(str).str.lower()
                        == username.strip().lower()
                    )
                )
            ]

        new_records = []

        for i, subject in enumerate(SUBJECTS):

            level, percentage = calculate_performance(
                attendance_values[i],
                internal_values[i],
                assignment_values[i],
                previous_values[i],
                0,
            )

            new_records.append({
                "Student_Name": name.strip(),
                "Username": username.strip(),
                "University_ID": university_id.strip(),
                "Subject": subject,
                "Attendance": attendance_values[i],
                "Internal_Mark": internal_values[i],
                "Assignment": assignment_values[i],
                "Previous_Mark": previous_values[i],
                "Study_Hours": 0,
                "Performance": level,
                "Performance_Percent": percentage,
                "KMeans_Cluster": "",
            })

        new_df = pd.DataFrame(
            new_records,
            columns=COLUMNS
        )

        df = pd.concat(
            [df, new_df],
            ignore_index=True
        )

        # Run KMeans on all saved data
        df = run_kmeans(df)

        save_data(df)

        st.success(
            f"Successfully saved {name}'s six subject records."
        )

        st.rerun()


# ============================================================
# TUTOR - STUDENT LIST
# ============================================================

def tutor_student_list():

    df = load_data()

    st.subheader(
        "👥 Saved Students"
    )

    if df.empty:

        st.info(
            "No student records have been saved yet."
        )

        return

    students = (
        df[
            [
                "Student_Name",
                "Username",
                "University_ID",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    display = students.copy()

    display.columns = [
        "Student Name",
        "Username",
        "University ID",
    ]

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# TUTOR - DELETE STUDENT
# ============================================================

def tutor_delete_student():

    df = load_data()

    st.subheader(
        "🗑️ Delete Student"
    )

    if df.empty:

        st.info(
            "No students available."
        )

        return

    students = (
        df[
            [
                "Student_Name",
                "University_ID",
            ]
        ]
        .drop_duplicates()
    )

    choices = [
        f"{row.Student_Name} - {row.University_ID}"
        for _, row in students.iterrows()
    ]

    selected = st.selectbox(
        "Select student to delete",
        choices
    )

    if st.button(
        "🗑️ Delete Selected Student",
        type="secondary",
        width="stretch",
    ):

        selected_id = selected.split(" - ")[-1]

        df = df[
            df["University_ID"].astype(str)
            != selected_id
        ]

        df = run_kmeans(df)

        save_data(df)

        st.success(
            "Complete student record deleted."
        )

        st.rerun()


# ============================================================
# TUTOR - ANALYSIS
# ============================================================

def tutor_analysis():

    df = load_data()

    st.subheader(
        "📊 Student Performance Analysis"
    )

    if df.empty:

        st.info(
            "Add student records first."
        )

        return

    df = run_kmeans(df)

    save_data(df)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    students_count = df[
        "University_ID"
    ].nunique()

    subject_count = df[
        "Subject"
    ].nunique()

    average = df[
        "Performance_Percent"
    ].astype(float).mean()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Students",
        students_count
    )

    c2.metric(
        "Subjects",
        subject_count
    )

    c3.metric(
        "Average Performance",
        f"{average:.2f}%"
    )

    # --------------------------------------------------------
    # KMEANS
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 K-Means Clustering"
    )

    cluster_table = (
        df.groupby("KMeans_Cluster")
        .agg(
            Students=(
                "University_ID",
                "nunique"
            ),
            Average_Performance=(
                "Performance_Percent",
                "mean"
            ),
        )
        .reset_index()
    )

    cluster_table[
        "Average_Performance"
    ] = cluster_table[
        "Average_Performance"
    ].round(2)

    st.dataframe(
        cluster_table,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "K-Means cluster numbers are machine-learning "
        "cluster identifiers and are not official grades."
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    st.markdown(
        "### 🌲 Random Forest"

    )

    trained = train_random_forest(df)

    if trained is None:

        st.info(
            "Random Forest needs sufficient records "
            "with at least two performance classes."
        )

    else:

        scaler, model = trained

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_,
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        st.dataframe(
            importance,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# TUTOR - REPORT DOWNLOAD
# ============================================================

def tutor_reports():

    df = load_data()

    st.subheader(
        "📄 Student Progress Reports"
    )

    if df.empty:

        st.info(
            "No student records available."
        )

        return

    students = (
        df[
            [
                "Student_Name",
                "University_ID",
            ]
        ]
        .drop_duplicates()
    )

    choices = [
        f"{row.Student_Name} - {row.University_ID}"
        for _, row in students.iterrows()
    ]

    selected = st.selectbox(
        "Select student",
        choices,
        key="report_student_select",
    )

    selected_id = selected.split(" - ")[-1]

    student_df = df[
        df["University_ID"].astype(str)
        == selected_id
    ].copy()

    # Ensure KMeans is updated
    student_df = run_kmeans(
        student_df
    )

    pdf = create_pdf(
        student_df
    )

    if pdf:

        st.download_button(
            "📥 Download Progress Report PDF",
            data=pdf,
            file_name=f"{selected_id}_Progress_Report.pdf",
            mime="application/pdf",
            width="stretch",
        )


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    st.markdown(
        """
        <div class="dashboard-card">

            <div class="dashboard-title">
                👨‍🏫 Tutor Dashboard
            </div>

            <div class="small-muted">
                Student data management, performance analysis,
                K-Means clustering and progress reports.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🚪 Logout",
        key="tutor_logout"
    ):

        st.session_state.clear()

        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value

        st.rerun()

    tabs = st.tabs([
        "➕ Add Student",
        "👥 Students",
        "📊 Analysis",
        "📄 Progress Reports",
        "🗑️ Delete Student",
    ])

    with tabs[0]:
        tutor_add_student()

    with tabs[1]:
        tutor_student_list()

    with tabs[2]:
        tutor_analysis()

    with tabs[3]:
        tutor_reports()

    with tabs[4]:
        tutor_delete_student()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    df = load_data()

    username = st.session_state[
        "student_username"
    ]

    university_id = st.session_state[
        "student_id"
    ]

    student_df = df[
        (
            df["Username"].astype(str).str.lower()
            == str(username).lower()
        )
        &
        (
            df["University_ID"].astype(str).str.lower()
            == str(university_id).lower()
        )
    ].copy()

    if student_df.empty:

        st.error(
            "Student record not found."
        )

        return

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    student_name = str(
        student_df.iloc[0]["Student_Name"]
    )

    st.markdown(
        f"""
        <div class="dashboard-card">

            <div class="dashboard-title">
                🎓 Welcome, {student_name}
            </div>

            <div class="small-muted">
                University ID: {university_id}
                <br>
                B.Tech S3 - AI & Data Science
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🚪 Logout",
        key="student_logout"
    ):

        st.session_state.clear()

        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value

        st.rerun()

    # --------------------------------------------------------
    # STUDY HOURS
    # --------------------------------------------------------

    st.subheader(
        "⏱️ Enter Study Hours"
    )

    st.caption(
        "Enter your daily study hours for each subject. "
        "Maximum considered by this project: 6 hours."
    )

    updated_hours = {}

    for idx, row in student_df.iterrows():

        subject = row["Subject"]

        updated_hours[subject] = st.number_input(
            subject,
            min_value=0.0,
            max_value=6.0,
            value=float(
                min(
                    max(
                        float(row["Study_Hours"]),
                        0
                    ),
                    6
                )
            ),
            step=0.5,
            key=f"study_{idx}_{university_id}",
        )

    if st.button(
        "💾 Save Study Hours & Calculate Result",
        type="primary",
        width="stretch",
    ):

        df_all = load_data()

        for subject, hours in updated_hours.items():

            mask = (
                (
                    df_all["University_ID"].astype(str)
                    == str(university_id)
                )
                &
                (
                    df_all["Subject"].astype(str)
                    == str(subject)
                )
            )

            df_all.loc[
                mask,
                "Study_Hours"
            ] = hours

        # Recalculate performance
        for idx in df_all.index:

            try:

                level, percentage = calculate_performance(
                    df_all.loc[idx, "Attendance"],
                    df_all.loc[idx, "Internal_Mark"],
                    df_all.loc[idx, "Assignment"],
                    df_all.loc[idx, "Previous_Mark"],
                    df_all.loc[idx, "Study_Hours"],
                )

                df_all.loc[
                    idx,
                    "Performance"
                ] = level

                df_all.loc[
                    idx,
                    "Performance_Percent"
                ] = percentage

            except Exception:
                pass

        df_all = run_kmeans(
            df_all
        )

        save_data(
            df_all
        )

        st.success(
            "Study hours saved and results updated."
        )

        st.rerun()

    # --------------------------------------------------------
    # REFRESH STUDENT DATA
    # --------------------------------------------------------

    df = load_data()

    student_df = df[
        (
            df["Username"].astype(str).str.lower()
            == str(username).lower()
        )
        &
        (
            df["University_ID"].astype(str).str.lower()
            == str(university_id).lower()
        )
    ].copy()

    # --------------------------------------------------------
    # SUBJECT RESULTS
    # --------------------------------------------------------

    st.subheader(
        "📚 Subject-wise Result"
    )

    for _, row in student_df.iterrows():

        subject = row["Subject"]

        level = row["Performance"]

        percentage = float(
            row["Performance_Percent"]
        )

        css_class, emoji = performance_style(
            level
        )

        advice = improvement_advice(
            row
        )

        st.markdown(
            f"""
            <div class="dashboard-card">

                <h3>
                    {subject}
                </h3>

                <p>
                    <span class="performance-circle {css_class}">
                    </span>

                    <b>{emoji} {level}</b>

                    &nbsp;&nbsp;

                    <b>{percentage:.2f}%</b>
                </p>

                <p>
                    Attendance:
                    <b>{float(row['Attendance']):.0f}%</b>
                    &nbsp; | &nbsp;

                    Internal:
                    <b>{float(row['Internal_Mark']):.1f}/40</b>
                    &nbsp; | &nbsp;

                    Assignment:
                    <b>{float(row['Assignment']):.1f}/15</b>
                    &nbsp; | &nbsp;

                    Previous:
                    <b>{float(row['Previous_Mark']):.1f}/60</b>
                    &nbsp; | &nbsp;

                    Study:
                    <b>{float(row['Study_Hours']):.1f} hr</b>
                </p>

                <p class="small-muted">
                    {advice}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # OVERALL RESULT
    # --------------------------------------------------------

    overall = student_df[
        "Performance_Percent"
    ].astype(float).mean()

    if overall < 50:
        overall_level = "Low Performance"
        overall_emoji = "🔴"

    elif overall < 65:
        overall_level = "Average Performance"
        overall_emoji = "🟠"

    elif overall < 80:
        overall_level = "Above Average"
        overall_emoji = "🟡"

    else:
        overall_level = "Good Performance"
        overall_emoji = "🟢"

    st.markdown(
        f"""
        <div class="dashboard-card">

            <div class="dashboard-title">
                📈 Overall Performance
            </div>

            <h2>
                {overall_emoji}
                {overall_level}
            </h2>

            <h1>
                {overall:.2f}%
            </h1>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    st.subheader(
        "📄 Download Progress Report"
    )

    pdf = create_pdf(
        student_df
    )

    if pdf:

        st.download_button(
            "📥 Download My Progress Report",
            data=pdf,
            file_name=f"{university_id}_Progress_Report.pdf",
            mime="application/pdf",
            width="stretch",
        )


# ============================================================
# MAIN ROUTER
# ============================================================

if st.session_state["page"] == "home":

    front_dashboard()

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
