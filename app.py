import os
import re
import io
import base64
import hashlib

import numpy as np
import pandas as pd
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
DATA_FILE = os.path.join(DATA_DIR, "student_performance.csv")

os.makedirs(DATA_DIR, exist_ok=True)

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
# B.TECH BRANCHES
# ============================================================

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
# DEMO TUTOR ACCOUNTS
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
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION = {
    "page": "home",
    "student_logged_in": False,
    "tutor_logged_in": False,
    "student_username": "",
    "student_id": "",
    "student_name": "",
    "tutor_username": "",
    "tutor_branch": "",
}

for key, value in DEFAULT_SESSION.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

def load_css():
    st.markdown(
        """
        <style>

        /* =====================================================
           GENERAL
        ===================================================== */

        .stApp {
            background: #050816;
            color: white;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        .block-container {
            padding-top: 1rem;
            max-width: 1400px;
        }


        /* =====================================================
           ANIMATED 3D BACKGROUND
        ===================================================== */

        .login-background {
            position: fixed;
            inset: 0;
            overflow: hidden;
            z-index: -10;
            background:
                radial-gradient(
                    circle at 20% 20%,
                    rgba(0, 140, 255, 0.25),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 80% 30%,
                    rgba(140, 60, 255, 0.22),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 50% 90%,
                    rgba(0, 220, 190, 0.16),
                    transparent 35%
                ),
                #03050f;
        }


        /* 3D grid */

        .grid-floor {
            position: absolute;
            width: 180%;
            height: 70%;
            left: -40%;
            bottom: -20%;
            background-image:
                linear-gradient(
                    rgba(0, 180, 255, 0.18) 1px,
                    transparent 1px
                ),
                linear-gradient(
                    90deg,
                    rgba(0, 180, 255, 0.18) 1px,
                    transparent 1px
                );

            background-size: 70px 70px;

            transform:
                perspective(500px)
                rotateX(62deg);

            animation: gridMove 10s linear infinite;
        }

        @keyframes gridMove {
            from {
                transform:
                    perspective(500px)
                    rotateX(62deg)
                    translateY(0);
            }

            to {
                transform:
                    perspective(500px)
                    rotateX(62deg)
                    translateY(70px);
            }
        }


        /* floating orbs */

        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(2px);
            opacity: 0.55;
            animation: floatOrb 9s ease-in-out infinite;
        }

        .orb1 {
            width: 180px;
            height: 180px;
            left: 8%;
            top: 12%;
            background:
                radial-gradient(
                    circle at 30% 30%,
                    #ffffff,
                    #00aaff 15%,
                    transparent 70%
                );
        }

        .orb2 {
            width: 220px;
            height: 220px;
            right: 8%;
            top: 16%;
            background:
                radial-gradient(
                    circle at 40% 35%,
                    #ffffff,
                    #8a3ffc 12%,
                    transparent 70%
                );
            animation-delay: -3s;
        }

        .orb3 {
            width: 150px;
            height: 150px;
            left: 45%;
            bottom: 15%;
            background:
                radial-gradient(
                    circle at 40% 30%,
                    #ffffff,
                    #00e0c0 12%,
                    transparent 70%
                );
            animation-delay: -6s;
        }

        @keyframes floatOrb {
            0%, 100% {
                transform:
                    translate3d(0, 0, 0)
                    scale(1);
            }

            50% {
                transform:
                    translate3d(30px, -45px, 0)
                    scale(1.08);
            }
        }


        /* =====================================================
           LOGIN CARD
        ===================================================== */

        .login-wrapper {
            min-height: 82vh;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .login-card {
            width: min(900px, 94vw);
            padding: 45px 40px;
            border-radius: 30px;

            background:
                linear-gradient(
                    135deg,
                    rgba(255,255,255,0.13),
                    rgba(255,255,255,0.04)
                );

            border: 1px solid rgba(255,255,255,0.18);

            box-shadow:
                0 30px 90px rgba(0,0,0,0.55),
                inset 0 1px 0 rgba(255,255,255,0.15);

            backdrop-filter: blur(22px);

            text-align: center;
        }


        /* =====================================================
           ONLY MAIN HEADING
           ===================================================== */

        .main-heading {
            font-size: clamp(28px, 5vw, 58px);
            font-weight: 900;
            letter-spacing: 2px;
            margin-bottom: 35px;

            background:
                linear-gradient(
                    90deg,
                    #ffffff,
                    #62d9ff,
                    #b07cff,
                    #ffffff
                );

            background-size: 300% auto;

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;

            animation:
                headingGlow 5s linear infinite;
        }

        @keyframes headingGlow {
            0% {
                background-position: 0% center;
            }

            50% {
                background-position: 100% center;
            }

            100% {
                background-position: 0% center;
            }
        }


        /* =====================================================
           LOGIN BUTTON CARDS
        ===================================================== */

        .role-card {
            padding: 30px 20px;
            min-height: 190px;

            border-radius: 24px;

            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.12),
                    rgba(255,255,255,0.035)
                );

            border:
                1px solid rgba(255,255,255,0.15);

            box-shadow:
                0 15px 45px rgba(0,0,0,0.30);

            transition:
                transform 0.3s ease,
                box-shadow 0.3s ease;

            text-align: center;
        }

        .role-card:hover {
            transform:
                translateY(-10px)
                rotateX(3deg);

            box-shadow:
                0 25px 65px rgba(0,150,255,0.25);
        }

        .role-icon {
            font-size: 58px;
            margin-bottom: 12px;
        }

        .role-title {
            font-size: 24px;
            font-weight: 800;
        }

        .role-description {
            color: #b9c7dd;
            margin-top: 8px;
            font-size: 14px;
        }


        /* =====================================================
           DASHBOARD
        ===================================================== */

        .dashboard-title {
            font-size: 34px;
            font-weight: 850;
            margin-bottom: 20px;
        }

        .glass-box {
            padding: 25px;
            border-radius: 20px;

            background:
                rgba(255,255,255,0.06);

            border:
                1px solid rgba(255,255,255,0.12);

            box-shadow:
                0 15px 45px rgba(0,0,0,0.25);

            margin-bottom: 20px;
        }


        /* =====================================================
           PERFORMANCE BADGES
        ===================================================== */

        .performance-box {
            padding: 20px;
            border-radius: 18px;
            text-align: center;
            font-size: 22px;
            font-weight: 800;
            margin: 10px 0;
        }

        .good {
            background: rgba(0, 210, 120, 0.18);
            border: 1px solid rgba(0, 255, 150, 0.35);
        }

        .above {
            background: rgba(255, 220, 0, 0.18);
            border: 1px solid rgba(255, 220, 0, 0.35);
        }

        .average {
            background: rgba(255, 150, 0, 0.18);
            border: 1px solid rgba(255, 150, 0, 0.35);
        }

        .low {
            background: rgba(255, 50, 60, 0.18);
            border: 1px solid rgba(255, 50, 60, 0.35);
        }


        /* =====================================================
           TABLE
        ===================================================== */

        .dataframe {
            border-radius: 15px;
            overflow: hidden;
        }


        /* =====================================================
           MOBILE
        ===================================================== */

        @media (max-width: 700px) {

            .login-card {
                padding: 30px 20px;
            }

            .main-heading {
                font-size: 28px;
            }

            .role-card {
                margin-bottom: 15px;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BACKGROUND
# ============================================================

def animated_background():
    st.markdown(
        """
        <div class="login-background">
            <div class="grid-floor"></div>
            <div class="orb orb1"></div>
            <div class="orb orb2"></div>
            <div class="orb orb3"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA FUNCTIONS
# ============================================================

def empty_dataframe():
    return pd.DataFrame(columns=COLUMNS)


def load_data():

    if not os.path.exists(DATA_FILE):
        return empty_dataframe()

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        return empty_dataframe()

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[COLUMNS]

    numeric_columns = [
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
        "Performance_Percent",
        "KMeans_Cluster",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_data(df):

    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(
        DATA_FILE,
        index=False
    )


# ============================================================
# ATTENDANCE CONVERSION
# ============================================================

def attendance_mark(attendance):

    attendance = float(attendance)

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
# PERFORMANCE CALCULATION
# ============================================================

def calculate_performance(
    attendance,
    internal,
    assignment,
    previous,
    study_hours
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
        100
    )

    overall = np.mean([
        attendance_percent,
        internal_percent,
        assignment_percent,
        previous_percent,
        study_percent,
    ])

    if overall < 50:
        level = "Low Performance"

    elif overall < 65:
        level = "Average Performance"

    elif overall < 80:
        level = "Above Average Performance"

    else:
        level = "Good Performance"

    return round(overall, 2), level


# ============================================================
# PERFORMANCE COLOR
# ============================================================

def performance_class(level):

    if "Low" in level:
        return "low"

    if "Average" in level and "Above" not in level:
        return "average"

    if "Above" in level:
        return "above"

    return "good"


def performance_circle(level):

    if "Low" in level:
        return "🔴"

    if "Average" in level and "Above" not in level:
        return "🟠"

    if "Above" in level:
        return "🟡"

    return "🟢"


# ============================================================
# PASSWORD
# ============================================================

def valid_student_password(password):

    if not isinstance(password, str):
        return False

    if not re.fullmatch(
        r"BTECH(20\d{2})",
        password
    ):
        return False

    year = int(password[-4:])

    return 2000 <= year <= 2022


# ============================================================
# GUIDANCE
# ============================================================

def get_guidance(row):

    advice = []

    attendance = float(row["Attendance"])
    internal = float(row["Internal_Mark"])
    assignment = float(row["Assignment"])
    previous = float(row["Previous_Mark"])
    study = float(row["Study_Hours"])

    if attendance < 75:
        advice.append(
            "Improve class attendance and attend lectures regularly."
        )

    if internal < 20:
        advice.append(
            "Spend more time preparing for internal examinations."
        )

    if assignment < 8:
        advice.append(
            "Complete assignments regularly and submit them on time."
        )

    if previous < 30:
        advice.append(
            "Revise previous topics and strengthen basic concepts."
        )

    if study < 3:
        advice.append(
            "Increase daily study and revision time."
        )

    if not advice:
        advice.append(
            "Excellent performance! Keep up the good work."
        )

    return advice


# ============================================================
# K-MEANS
# ============================================================

def run_kmeans(df):

    if len(df) < 3:
        return df.copy(), None

    features = [
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
    ]

    working = df.copy()

    X = working[features].fillna(0)

    n_clusters = min(
        3,
        len(working)
    )

    if n_clusters < 2:
        return working, None

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    working["KMeans_Cluster"] = model.fit_predict(X)

    score = None

    if len(set(working["KMeans_Cluster"])) > 1:
        score = silhouette_score(
            X,
            working["KMeans_Cluster"]
        )

    return working, score


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(df):

    if len(df) < 10:
        return None, None

    features = [
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
    ]

    working = df.copy()

    working["Performance"] = (
        working["Performance"]
        .astype(str)
    )

    X = working[features].fillna(0)
    y = working["Performance"]

    if y.nunique() < 2:
        return None, None

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

    except ValueError:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy


# ============================================================
# PDF REPORT
# ============================================================

def create_progress_report(
    student_name,
    university_id,
    branch,
    semester,
    student_df
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=15,
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    story = []

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PREDICTION SYSTEM",
            title_style
        )
    )

    story.append(
        Paragraph(
            "STUDENT PROGRESS REPORT",
            heading_style
        )
    )

    details = [
        ["Student Name", student_name],
        ["University ID", university_id],
        ["Branch", branch],
        ["Semester", semester],
    ]

    table = Table(
        details,
        colWidths=[120, 350]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 15))

    # Overall
    if len(student_df) > 0:

        overall_percentage = student_df[
            "Performance_Percent"
        ].mean()

        overall_level = (
            student_df["Performance"]
            .mode()
            .iloc[0]
            if not student_df["Performance"].mode().empty
            else "Not Available"
        )

        story.append(
            Paragraph(
                "OVERALL PERFORMANCE",
                heading_style
            )
        )

        overall_data = [
            ["Overall Percentage",
             f"{overall_percentage:.2f}%"],

            ["Overall Performance",
             overall_level],
        ]

        overall_table = Table(
            overall_data,
            colWidths=[180, 290]
        )

        overall_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ])
        )

        story.append(overall_table)
        story.append(Spacer(1, 15))

    # Subjects
    story.append(
        Paragraph(
            "SUBJECT-WISE PERFORMANCE",
            heading_style
        )
    )

    for _, row in student_df.iterrows():

        attendance = float(row["Attendance"])
        internal = float(row["Internal_Mark"])
        assignment = float(row["Assignment"])
        previous = float(row["Previous_Mark"])
        study = float(row["Study_Hours"])

        level = row["Performance"]

        cluster = row["KMeans_Cluster"]

        subject_data = [
            ["Subject", str(row["Subject"])],

            [
                "Attendance",
                f"{attendance:.1f}%"
            ],

            [
                "Attendance Converted Mark",
                f"{attendance_mark(attendance)}/5"
            ],

            [
                "Study Hours",
                f"{study:.1f}/6 hours"
            ],

            [
                "Internal Mark",
                f"{internal:.1f}/40"
            ],

            [
                "Assignment",
                f"{assignment:.1f}/15"
            ],

            [
                "Previous Mark",
                f"{previous:.1f}/60"
            ],

            [
                "Performance",
                f"{performance_circle(level)} {level}"
            ],

            [
                "Performance Percentage",
                f"{float(row['Performance_Percent']):.2f}%"
            ],

            [
                "K-Means Cluster",
                str(cluster)
            ],
        ]

        subject_table = Table(
            subject_data,
            colWidths=[190, 280]
        )

        subject_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )

        story.append(subject_table)
        story.append(Spacer(1, 12))

        guidance = get_guidance(row)

        story.append(
            Paragraph(
                "<b>Improvement / Guidance:</b>",
                normal_style
            )
        )

        for advice in guidance:

            story.append(
                Paragraph(
                    "• " + advice,
                    normal_style
                )
            )

        story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "ASSESSMENT CONVERSION",
            heading_style
        )
    )

    conversion_data = [
        ["Attendance", "Converted Mark"],
        ["90–100%", "5"],
        ["80–89%", "4"],
        ["70–79%", "3"],
        ["60–69%", "2"],
        ["10–59%", "1"],
        ["Below 10%", "0"],
    ]

    conversion_table = Table(
        conversion_data,
        colWidths=[230, 230]
    )

    conversion_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(conversion_table)
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Note: The performance levels and percentage calculation "
            "used in this project are project-defined assessment rules "
            "and are not official KTU grading rules.",
            normal_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HOME PAGE
# ============================================================

def front_dashboard():

    animated_background()

    st.markdown(
        '<div class="login-wrapper">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-card">',
        unsafe_allow_html=True
    )

    # ONLY MAIN HEADING
    st.markdown(
        """
        <div class="main-heading">
            🎓 STUDENT PERFORMANCE PREDICTION
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="role-card">
                <div class="role-icon">🎓</div>
                <div class="role-title">Student</div>
                <div class="role-description">
                    View academic performance,
                    subjects and progress report.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🎓 Student Login",
            key="student_login_button",
            width="stretch"
        ):
            st.session_state["page"] = "student_login"
            st.rerun()

    with col2:

        st.markdown(
            """
            <div class="role-card">
                <div class="role-icon">👨‍🏫</div>
                <div class="role-title">Tutor</div>
                <div class="role-description">
                    Manage student records,
                    analyse performance and generate reports.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "👨‍🏫 Tutor Login",
            key="tutor_login_button",
            width="stretch"
        ):
            st.session_state["page"] = "tutor_login"
            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    animated_background()

    st.markdown(
        '<div class="login-wrapper">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="main-heading">
            🎓 STUDENT LOGIN
        </div>
        """,
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
        type="password",
        placeholder="Example: BTECH2007"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔐 Login",
            width="stretch"
        ):

            if not username or not university_id or not password:

                st.error(
                    "Please enter all login details."
                )

            elif not valid_student_password(password):

                st.error(
                    "Password must follow BTECH2000 to BTECH2022 format."
                )

            else:

                df = load_data()

                if len(df) == 0:

                    st.error(
                        "No student data found. Ask the tutor to add your records."
                    )

                else:

                    student_rows = df[
                        (
                            df["Username"].astype(str)
                            == username
                        )
                        &
                        (
                            df["University_ID"].astype(str)
                            == university_id
                        )
                    ]

                    if len(student_rows) == 0:

                        st.error(
                            "Student record not found."
                        )

                    else:

                        st.session_state["student_logged_in"] = True

                        st.session_state[
                            "student_username"
                        ] = username

                        st.session_state[
                            "student_id"
                        ] = university_id

                        st.session_state[
                            "student_name"
                        ] = student_rows.iloc[0]["Student_Name"]

                        st.session_state[
                            "page"
                        ] = "student_dashboard"

                        st.rerun()

    with col2:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state["page"] = "home"
            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    animated_background()

    st.markdown(
        '<div class="login-wrapper">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="main-heading">
            👨‍🏫 TUTOR LOGIN
        </div>
        """,
        unsafe_allow_html=True
    )

    username = st.text_input(
        "Tutor Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔐 Login",
            width="stretch"
        ):

            if (
                username in TUTOR_ACCOUNTS
                and
                TUTOR_ACCOUNTS[username]["password"]
                == password
            ):

                st.session_state[
                    "tutor_logged_in"
                ] = True

                st.session_state[
                    "tutor_username"
                ] = username

                st.session_state[
                    "tutor_branch"
                ] = TUTOR_ACCOUNTS[
                    username
                ]["branch"]

                st.session_state[
                    "page"
                ] = "tutor_dashboard"

                st.rerun()

            else:

                st.error(
                    "Invalid tutor username or password."
                )

    with col2:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state["page"] = "home"
            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    if not st.session_state["student_logged_in"]:

        st.session_state["page"] = "student_login"
        st.rerun()

    df = load_data()

    student_name = st.session_state["student_name"]

    student_df = df[
        (
            df["Username"].astype(str)
            == st.session_state["student_username"]
        )
        &
        (
            df["University_ID"].astype(str)
            == st.session_state["student_id"]
        )
    ].copy()

    st.markdown(
        f"""
        <div class="dashboard-title">
            🎓 Welcome, {student_name}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚪 Logout"):

        st.session_state["student_logged_in"] = False
        st.session_state["page"] = "home"
        st.rerun()

    if len(student_df) == 0:

        st.warning(
            "No performance records available."
        )

        return

    semesters = sorted(
        student_df["Semester"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not semesters:
        semesters = list(SEMESTERS.keys())

    semester = st.selectbox(
        "Select Semester",
        semesters
    )

    semester_df = student_df[
        student_df["Semester"].astype(str)
        == semester
    ].copy()

    if len(semester_df) == 0:

        st.info(
            "No records available for this semester."
        )

        return

    # Overall performance
    overall_percentage = semester_df[
        "Performance_Percent"
    ].mean()

    mode = semester_df[
        "Performance"
    ].mode()

    overall_level = (
        mode.iloc[0]
        if not mode.empty
        else "Not Available"
    )

    css_class = performance_class(
        overall_level
    )

    st.markdown(
        f"""
        <div class="performance-box {css_class}">
            {performance_circle(overall_level)}
            Overall Performance:
            {overall_level}
            <br>
            <span style="font-size:17px;">
                {overall_percentage:.2f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "📚 Subject-wise Performance"
    )

    for _, row in semester_df.iterrows():

        level = row["Performance"]

        with st.expander(
            f"{performance_circle(level)} {row['Subject']}"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Attendance",
                    f"{row['Attendance']:.1f}%"
                )

                st.metric(
                    "Internal",
                    f"{row['Internal_Mark']:.1f}/40"
                )

            with c2:

                st.metric(
                    "Assignment",
                    f"{row['Assignment']:.1f}/15"
                )

                st.metric(
                    "Previous Mark",
                    f"{row['Previous_Mark']:.1f}/60"
                )

            with c3:

                st.metric(
                    "Study Hours",
                    f"{row['Study_Hours']:.1f}/6"
                )

                st.metric(
                    "Performance",
                    f"{row['Performance_Percent']:.2f}%"
                )

            st.write(
                f"**Level:** {level}"
            )

            guidance = get_guidance(row)

            st.write(
                "**Guidance:**"
            )

            for advice in guidance:
                st.write(
                    f"• {advice}"
                )

    # PDF
    pdf_data = create_progress_report(
        student_name,
        st.session_state["student_id"],
        str(semester_df.iloc[0]["Branch"]),
        semester,
        semester_df,
    )

    st.download_button(
        "📄 Download My Progress Report",
        data=pdf_data,
        file_name=(
            f"{student_name}_"
            f"{semester}_Progress_Report.pdf"
        ),
        mime="application/pdf",
        width="stretch",
    )


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    if not st.session_state["tutor_logged_in"]:

        st.session_state["page"] = "tutor_login"
        st.rerun()

    branch = st.session_state["tutor_branch"]

    st.markdown(
        """
        <div class="dashboard-title">
            👨‍🏫 Tutor Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        f"Assigned Branch: {branch}"
    )

    if st.button("🚪 Logout"):

        st.session_state["tutor_logged_in"] = False
        st.session_state["page"] = "home"
        st.rerun()

    df = load_data()

    branch_df = df[
        df["Branch"].astype(str)
        == branch
    ].copy()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "➕ Add Student",
            "📊 Student Data",
            "🤖 K-Means",
            "🌲 Random Forest",
            "📄 Progress Reports",
        ]
    )


    # ========================================================
    # TAB 1 - ADD STUDENT
    # ========================================================

    with tab1:

        st.subheader(
            "Add Student Subject Records"
        )

        st.write(
            "Enter one student's information and subject-wise marks."
        )

        student_name = st.text_input(
            "Student Name",
            key="add_student_name"
        )

        username = st.text_input(
            "Student Username",
            key="add_student_username"
        )

        university_id = st.text_input(
            "University ID",
            key="add_student_id"
        )

        semester = st.selectbox(
            "Semester",
            list(SEMESTERS.keys()),
            key="add_semester"
        )

        subjects = SEMESTERS[semester]

        st.markdown(
            "### Subject-wise Details"
        )

        records = []

        for subject in subjects:

            st.markdown(
                f"#### 📘 {subject}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                attendance = st.number_input(
                    "Attendance %",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=1.0,
                    key=f"att_{semester}_{subject}"
                )

            with c2:

                internal = st.number_input(
                    "Internal /40",
                    min_value=0.0,
                    max_value=40.0,
                    value=20.0,
                    step=1.0,
                    key=f"int_{semester}_{subject}"
                )

            with c3:

                assignment = st.number_input(
                    "Assignment /15",
                    min_value=0.0,
                    max_value=15.0,
                    value=8.0,
                    step=1.0,
                    key=f"ass_{semester}_{subject}"
                )

            with c4:

                previous = st.number_input(
                    "Previous /60",
                    min_value=0.0,
                    max_value=60.0,
                    value=30.0,
                    step=1.0,
                    key=f"prev_{semester}_{subject}"
                )

            with c5:

                study = st.number_input(
                    "Study Hours /day",
                    min_value=0.0,
                    max_value=6.0,
                    value=3.0,
                    step=0.5,
                    key=f"study_{semester}_{subject}"
                )

            percentage, performance = calculate_performance(
                attendance,
                internal,
                assignment,
                previous,
                study
            )

            records.append(
                {
                    "Student_Name": student_name,
                    "Username": username,
                    "University_ID": university_id,
                    "Branch": branch,
                    "Semester": semester,
                    "Subject": subject,
                    "Attendance": attendance,
                    "Internal_Mark": internal,
                    "Assignment": assignment,
                    "Previous_Mark": previous,
                    "Study_Hours": study,
                    "Performance": performance,
                    "Performance_Percent": percentage,
                    "KMeans_Cluster": np.nan,
                }
            )

        if st.button(
            "💾 Submit Student Records",
            type="primary",
            width="stretch"
        ):

            if not student_name.strip():

                st.error(
                    "Enter student name."
                )

            elif not username.strip():

                st.error(
                    "Enter student username."
                )

            elif not university_id.strip():

                st.error(
                    "Enter university ID."
                )

            else:

                new_df = pd.DataFrame(
                    records
                )

                # Remove old records of same student + semester
                if len(df) > 0:

                    df = df[
                        ~(
                            (
                                df["University_ID"].astype(str)
                                == str(university_id)
                            )
                            &
                            (
                                df["Semester"].astype(str)
                                == semester
                            )
                            &
                            (
                                df["Branch"].astype(str)
                                == branch
                            )
                        )
                    ]

                final_df = pd.concat(
                    [df, new_df],
                    ignore_index=True
                )

                save_data(final_df)

                st.success(
                    "Student records saved successfully."
                )

                st.rerun()


    # ========================================================
    # TAB 2 - STUDENT DATA
    # ========================================================

    with tab2:

        st.subheader(
            "Branch Student Data"
        )

        if len(branch_df) == 0:

            st.info(
                "No student data available for this branch."
            )

        else:

            st.dataframe(
                branch_df,
                width="stretch",
                hide_index=True
            )

            csv_data = branch_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Branch CSV",
                data=csv_data,
                file_name="branch_student_data.csv",
                mime="text/csv",
                width="stretch"
            )

            st.markdown(
                "### Delete Student"
            )

            student_options = (
                branch_df[
                    [
                        "Student_Name",
                        "University_ID"
                    ]
                ]
                .drop_duplicates()
            )

            selected_student = st.selectbox(
                "Select Student",
                student_options.apply(
                    lambda x:
                    f"{x['Student_Name']} | {x['University_ID']}",
                    axis=1
                ).tolist()
            )

            if st.button(
                "🗑️ Delete Complete Student Data"
            ):

                selected_id = (
                    selected_student.split("|")[-1].strip()
                )

                updated_df = branch_df[
                    branch_df["University_ID"].astype(str)
                    != selected_id
                ]

                # Preserve other branches
                other_df = df[
                    df["Branch"].astype(str)
                    != branch
                ]

                final_df = pd.concat(
                    [other_df, updated_df],
                    ignore_index=True
                )

                save_data(final_df)

                st.success(
                    "Student data deleted successfully."
                )

                st.rerun()


    # ========================================================
    # TAB 3 - K-MEANS
    # ========================================================

    with tab3:

        st.subheader(
            "🤖 K-Means Student Clustering"
        )

        if len(branch_df) < 3:

            st.warning(
                "At least 3 records are required for K-Means."
            )

        else:

            clustered_df, silhouette = run_kmeans(
                branch_df
            )

            if silhouette is not None:

                st.metric(
                    "Silhouette Score",
                    f"{silhouette:.4f}"
                )

            st.dataframe(
                clustered_df[
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
                hide_index=True
            )

            if st.button(
                "💾 Save K-Means Clusters"
            ):

                all_df = load_data()

                for idx in clustered_df.index:

                    all_df.loc[
                        idx,
                        "KMeans_Cluster"
                    ] = clustered_df.loc[
                        idx,
                        "KMeans_Cluster"
                    ]

                save_data(all_df)

                st.success(
                    "K-Means clusters saved."
                )

                st.rerun()


            st.info(
                "K-Means cluster numbers are identifiers only. "
                "They do not directly mean Low, Average or Good."
            )


    # ========================================================
    # TAB 4 - RANDOM FOREST
    # ========================================================

    with tab4:

        st.subheader(
            "🌲 Random Forest Performance Prediction"
        )

        model, accuracy = train_random_forest(
            branch_df
        )

        if model is None:

            st.warning(
                "At least 10 suitable student records "
                "with multiple performance classes are required."
            )

        else:

            st.metric(
                "Random Forest Accuracy",
                f"{accuracy * 100:.2f}%"
            )

            st.success(
                "Random Forest model trained successfully."
            )

            st.write(
                "The model uses Attendance, Internal Mark, "
                "Assignment, Previous Mark and Study Hours."
            )


    # ========================================================
    # TAB 5 - PROGRESS REPORTS
    # ========================================================

    with tab5:

        st.subheader(
            "📄 Student Progress Reports"
        )

        if len(branch_df) == 0:

            st.info(
                "No student records available."
            )

        else:

            students = (
                branch_df[
                    [
                        "Student_Name",
                        "University_ID",
                        "Semester"
                    ]
                ]
                .drop_duplicates()
            )

            selected = st.selectbox(
                "Select Student",
                students.apply(
                    lambda x:
                    f"{x['Student_Name']} | "
                    f"{x['University_ID']} | "
                    f"{x['Semester']}",
                    axis=1
                ).tolist()
            )

            parts = [
                x.strip()
                for x in selected.split("|")
            ]

            selected_name = parts[0]
            selected_id = parts[1]
            selected_semester = parts[2]

            selected_df = branch_df[
                (
                    branch_df["Student_Name"].astype(str)
                    == selected_name
                )
                &
                (
                    branch_df["University_ID"].astype(str)
                    == selected_id
                )
                &
                (
                    branch_df["Semester"].astype(str)
                    == selected_semester
                )
            ].copy()

            st.dataframe(
                selected_df,
                width="stretch",
                hide_index=True
            )

            pdf_data = create_progress_report(
                selected_name,
                selected_id,
                branch,
                selected_semester,
                selected_df
            )

            st.download_button(
                "📄 Download Progress Report PDF",
                data=pdf_data,
                file_name=(
                    f"{selected_name}_"
                    f"{selected_semester}_"
                    f"Progress_Report.pdf"
                ),
                mime="application/pdf",
                width="stretch"
            )


# ============================================================
# MAIN ROUTER
# ============================================================

load_css()

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
