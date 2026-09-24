# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# B.Tech S3 - AI & Data Science
#
# Technologies:
#   - Python
#   - Streamlit
#   - K-Means Clustering
#   - Random Forest
#   - Pandas
#   - ReportLab
#
# File:
#   app.py
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 3. DATA DIRECTORY
# ============================================================

DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_FILE = os.path.join(
    DATA_DIR,
    "student_records.csv"
)


# ============================================================
# 4. BRANCHES
# ============================================================

BRANCHES = [
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning",
    "Computer Science and Engineering",
    "Computer Science and Engineering (AI)",
    "Computer Science and Engineering (Data Science)",
    "Cyber Security",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Robotics and Automation",
    "Mechatronics Engineering",
    "Biomedical Engineering",
    "Aeronautical Engineering",
    "Automobile Engineering",
    "Chemical Engineering",
    "Biotechnology",
    "Food Technology",
    "Industrial Engineering",
    "Production Engineering",
    "Environmental Engineering",
    "Marine Engineering"
]


# ============================================================
# 5. SUBJECTS
# ============================================================

SUBJECTS = {

    "S1": [
        "Mathematics I",
        "Physics",
        "Chemistry",
        "Engineering Graphics",
        "Engineering Mechanics",
        "Programming in C",
        "Basic Electrical Engineering",
        "Basic Electronics",
        "Communication Skills",
        "Engineering Workshop"
    ],

    "S2": [
        "Mathematics II",
        "Physics II",
        "Chemistry II",
        "Programming in Python",
        "Engineering Mechanics",
        "Electrical Engineering",
        "Electronics Engineering",
        "Professional Communication",
        "Engineering Graphics II",
        "Environmental Science"
    ],

    "S3": [
        "Mathematics III",
        "Data Structures",
        "Database Management Systems",
        "Artificial Intelligence",
        "Machine Learning",
        "Computer Organization",
        "Operating Systems",
        "Object Oriented Programming",
        "Probability and Statistics",
        "Digital Electronics"
    ],

    "S4": [
        "Mathematics IV",
        "Computer Networks",
        "Operating Systems",
        "Design and Analysis of Algorithms",
        "Software Engineering",
        "Microprocessors",
        "Web Programming",
        "Data Analytics",
        "Artificial Intelligence II",
        "Professional Ethics"
    ],

    "S5": [
        "Machine Learning",
        "Computer Networks",
        "Compiler Design",
        "Distributed Computing",
        "Data Mining",
        "Cloud Computing",
        "Big Data",
        "Elective I",
        "Mini Project",
        "Seminar"
    ],

    "S6": [
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Cloud Computing",
        "Cyber Security",
        "Data Visualization",
        "Elective II",
        "Elective III",
        "Project Phase I",
        "Seminar"
    ],

    "S7": [
        "Advanced Machine Learning",
        "Artificial Intelligence",
        "Big Data Analytics",
        "Deep Learning",
        "Elective IV",
        "Elective V",
        "Project Phase II",
        "Seminar",
        "Industrial Training",
        "Professional Elective"
    ],

    "S8": [
        "Major Project",
        "Project Presentation",
        "Project Viva",
        "Comprehensive Viva",
        "Industrial Training",
        "Professional Ethics",
        "Open Elective",
        "Technical Seminar",
        "Research Work",
        "Career Development"
    ]
}


# ============================================================
# 6. TUTOR ACCOUNTS
# ============================================================

TEACHERS = {

    "teacher_aids": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Data Science"
    },

    "teacher_aiml": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Machine Learning"
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

    "teacher_cyber": {
        "password": "ktutech",
        "branch": "Cyber Security"
    },

    "teacher_it": {
        "password": "ktutech",
        "branch": "Information Technology"
    },

    "teacher_ece": {
        "password": "ktutech",
        "branch": "Electronics and Communication Engineering"
    },

    "teacher_eee": {
        "password": "ktutech",
        "branch": "Electrical and Electronics Engineering"
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
# 7. DATABASE COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "Username",
    "University_ID",
    "Student_Name",
    "Branch",
    "Semester",
    "Subject",

    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark",

    "Performance",
    "Overall_Percentage",

    "Cluster",

    "RF_Prediction",
    "RF_Confidence",

    "Submitted_By",
    "Submitted_Date"
]


INPUT_COLUMNS = [

    "Username",
    "University_ID",
    "Student_Name",
    "Branch",
    "Semester",
    "Subject",

    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]


# ============================================================
# 8. SESSION STATE
# ============================================================

DEFAULT_SESSION = {

    "logged_in": False,

    "login_mode": None,

    "user_role": None,

    "username": None,

    "university_id": None,

    "branch": None,

    "celebrated": False
}


for key, value in DEFAULT_SESSION.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# 9. LOAD DATABASE
# ============================================================

def load_database():

    if not os.path.exists(DATABASE_FILE):

        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )

    try:

        df = pd.read_csv(
            DATABASE_FILE
        )

        for column in REQUIRED_COLUMNS:

            if column not in df.columns:

                df[column] = ""

        return df[
            REQUIRED_COLUMNS
        ]

    except Exception:

        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )


# ============================================================
# 10. SAVE DATABASE
# ============================================================

def save_database(df):

    df = df.copy()

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:

            df[column] = ""

    df = df[
        REQUIRED_COLUMNS
    ]

    df.to_csv(
        DATABASE_FILE,
        index=False
    )


# ============================================================
# 11. ATTENDANCE CONVERSION
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
# 12. OVERALL PERCENTAGE CALCULATION
# ============================================================

def calculate_overall(
    attendance,
    study_hours,
    internal_mark,
    assignment,
    previous_mark
):

    attendance = np.clip(
        float(attendance),
        0,
        100
    )

    study_hours = np.clip(
        float(study_hours),
        0,
        6
    )

    internal_mark = np.clip(
        float(internal_mark),
        0,
        40
    )

    assignment = np.clip(
        float(assignment),
        0,
        15
    )

    previous_mark = np.clip(
        float(previous_mark),
        0,
        60
    )

    attendance_percentage = (
        attendance_mark(attendance)
        / 5
    ) * 100

    study_percentage = (
        study_hours / 6
    ) * 100

    internal_percentage = (
        internal_mark / 40
    ) * 100

    assignment_percentage = (
        assignment / 15
    ) * 100

    previous_percentage = (
        previous_mark / 60
    ) * 100

    overall = (
        attendance_percentage
        + study_percentage
        + internal_percentage
        + assignment_percentage
        + previous_percentage
    ) / 5

    return round(
        overall,
        2
    )


# ============================================================
# 13. PERFORMANCE CLASSIFICATION
# ============================================================

def performance_from_overall(overall):

    overall = float(overall)

    if overall < 50:

        return "Low Performance"

    elif overall < 65:

        return "Average Performance"

    elif overall < 80:

        return "Above Average"

    else:

        return "Good Performance"


# ============================================================
# 14. PERFORMANCE INFORMATION
# ============================================================

def performance_info(level):

    if level == "Low Performance":

        return {
            "emoji": "🔴",
            "color": "#ff4b4b",
            "message": "Needs improvement"
        }

    elif level == "Average Performance":

        return {
            "emoji": "🟠",
            "color": "#ff9800",
            "message": "Regular improvement required"
        }

    elif level == "Above Average":

        return {
            "emoji": "🟡",
            "color": "#f4c430",
            "message": "Good progress"
        }

    return {
        "emoji": "🟢",
        "color": "#00c853",
        "message": "Excellent performance"
    }


# ============================================================
# 15. IMPROVEMENT SUGGESTIONS
# ============================================================

def improvement_methods(row):

    methods = []

    if float(row["Attendance"]) < 80:

        methods.append(
            "Improve regular class attendance."
        )

    if float(row["Internal_Mark"]) < 26:

        methods.append(
            "Revise class notes and prepare better for internal examinations."
        )

    if float(row["Assignment"]) < 10:

        methods.append(
            "Complete assignments on time."
        )

    if float(row["Previous_Mark"]) < 40:

        methods.append(
            "Revise previous topics and practise more questions."
        )

    if float(row["Study_Hours"]) < 3:

        methods.append(
            "Increase regular study and revision time."
        )

    if not methods:

        methods.append(
            "Excellent performance! Keep up the good work."
        )

    return methods


# ============================================================
# 16. CLEAN DATA
# ============================================================

def clean_input_dataframe(df):

    df = df.copy()

    for column in INPUT_COLUMNS:

        if column not in df.columns:

            df[column] = ""

    df = df[
        INPUT_COLUMNS
    ]

    text_columns = [
        "Username",
        "University_ID",
        "Student_Name",
        "Branch",
        "Semester",
        "Subject"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    numeric_columns = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    df["Attendance"] = df[
        "Attendance"
    ].clip(0, 100)

    df["Study_Hours"] = df[
        "Study_Hours"
    ].clip(0, 6)

    df["Internal_Mark"] = df[
        "Internal_Mark"
    ].clip(0, 40)

    df["Assignment"] = df[
        "Assignment"
    ].clip(0, 15)

    df["Previous_Mark"] = df[
        "Previous_Mark"
    ].clip(0, 60)

    return df


# ============================================================
# 17. ADD PERFORMANCE
# ============================================================

def add_performance_columns(df):

    df = df.copy()

    overall_values = []

    performance_values = []

    for _, row in df.iterrows():

        overall = calculate_overall(
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"]
        )

        level = performance_from_overall(
            overall
        )

        overall_values.append(
            overall
        )

        performance_values.append(
            level
        )

    df["Overall_Percentage"] = (
        overall_values
    )

    df["Performance"] = (
        performance_values
    )

    return df


# ============================================================
# 18. K-MEANS CLUSTERING
# ============================================================

def run_kmeans(df):

    df = df.copy()

    if len(df) == 0:

        return df

    features = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark"
    ]

    X = df[
        features
    ].astype(float)

    if len(df) == 1:

        df["Cluster"] = 0

        return df

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    number_of_clusters = min(
        3,
        len(df)
    )

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(
        X_scaled
    )

    return df


# ============================================================
# 19. RANDOM FOREST
# ============================================================

def run_random_forest(df):

    df = df.copy()

    df["RF_Prediction"] = ""

    df["RF_Confidence"] = 0.0

    if len(df) < 10:

        return df

    if df["Performance"].nunique() < 2:

        return df

    features = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark"
    ]

    X = df[
        features
    ].astype(float)

    y = df[
        "Performance"
    ].astype(str)

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X,
        y
    )

    prediction = model.predict(
        X
    )

    probability = model.predict_proba(
        X
    )

    confidence = (
        probability.max(
            axis=1
        ) * 100
    )

    df["RF_Prediction"] = (
        prediction
    )

    df["RF_Confidence"] = (
        np.round(
            confidence,
            2
        )
    )

    return df


# ============================================================
# 20. PREPARE DATASET
# ============================================================

def prepare_dataset(df):

    if len(df) == 0:

        return df

    df = clean_input_dataframe(
        df
    )

    df = add_performance_columns(
        df
    )

    df = run_kmeans(
        df
    )

    df = run_random_forest(
        df
    )

    return df


# ============================================================
# 21. STUDENT PASSWORD VALIDATION
# ============================================================

def valid_student_password(password):

    if len(password) != 9:

        return False

    if not password.startswith("BTECH"):

        return False

    year = password[5:]

    if not year.isdigit():

        return False

    return (
        2000 <= int(year) <= 2020
    )


# ============================================================
# 22. CSV TEMPLATE
# ============================================================

def create_csv_template():

    template = pd.DataFrame(
        [
            {
                "Username": "student001",
                "University_ID": "KTU001",
                "Student_Name": "Student One",
                "Branch": "Artificial Intelligence and Data Science",
                "Semester": "S3",
                "Subject": "Machine Learning",
                "Attendance": 90,
                "Study_Hours": 4,
                "Internal_Mark": 34,
                "Assignment": 13,
                "Previous_Mark": 50
            },

            {
                "Username": "student001",
                "University_ID": "KTU001",
                "Student_Name": "Student One",
                "Branch": "Artificial Intelligence and Data Science",
                "Semester": "S3",
                "Subject": "Database Management Systems",
                "Attendance": 82,
                "Study_Hours": 3,
                "Internal_Mark": 29,
                "Assignment": 11,
                "Previous_Mark": 45
            }
        ]
    )

    return template.to_csv(
        index=False
    ).encode("utf-8")


# ============================================================
# 23. NEW DYNAMIC 3D LOGIN BACKGROUND
# ============================================================

def show_3d_background():

    st.markdown(
        """
        <style>

        /* ====================================================
           FULL SCREEN BACKGROUND
        ==================================================== */

        .stApp {

            background:
                radial-gradient(
                    circle at 15% 20%,
                    rgba(0, 229, 255, 0.18),
                    transparent 28%
                ),

                radial-gradient(
                    circle at 85% 20%,
                    rgba(139, 92, 246, 0.20),
                    transparent 30%
                ),

                radial-gradient(
                    circle at 50% 90%,
                    rgba(236, 72, 153, 0.13),
                    transparent 30%
                ),

                #030712;

            color: white;
        }


        /* ====================================================
           STREAMLIT CONTENT
        ==================================================== */

        .block-container {

            position: relative;

            z-index: 10;

            padding-top: 2.5rem;
        }


        /* ====================================================
           BACKGROUND SCENE
        ==================================================== */

        .three-d-scene {

            position: fixed;

            inset: 0;

            width: 100vw;

            height: 100vh;

            overflow: hidden;

            pointer-events: none;

            z-index: 0;
        }


        /* ====================================================
           GLOWING BALLS
        ==================================================== */

        .glow-ball {

            position: absolute;

            border-radius: 50%;

            filter: blur(4px);

            opacity: 0.55;

            animation:
                ballFloat 8s ease-in-out infinite;
        }


        .ball-1 {

            width: 260px;
            height: 260px;

            left: -60px;
            top: 10%;

            background:
                radial-gradient(
                    circle,
                    rgba(34,211,238,0.45),
                    transparent 70%
                );
        }


        .ball-2 {

            width: 330px;
            height: 330px;

            right: -100px;
            top: 20%;

            background:
                radial-gradient(
                    circle,
                    rgba(139,92,246,0.45),
                    transparent 70%
                );

            animation-delay: 2s;
        }


        .ball-3 {

            width: 240px;
            height: 240px;

            left: 40%;
            bottom: -100px;

            background:
                radial-gradient(
                    circle,
                    rgba(236,72,153,0.35),
                    transparent 70%
                );

            animation-delay: 4s;
        }


        @keyframes ballFloat {

            0%,
            100% {

                transform:
                    translate3d(0,0,0)
                    scale(1);
            }

            50% {

                transform:
                    translate3d(30px,-35px,0)
                    scale(1.12);
            }
        }


        /* ====================================================
           3D CUBES
        ==================================================== */

        .cube {

            position: absolute;

            width: 100px;

            height: 100px;

            transform-style: preserve-3d;

            animation:
                cubeSpin 14s linear infinite;

            opacity: 0.32;
        }


        .cube div {

            position: absolute;

            width: 100px;

            height: 100px;

            border:
                1px solid
                rgba(103,232,249,0.8);

            background:
                rgba(103,232,249,0.025);

            box-shadow:
                0 0 25px
                rgba(34,211,238,0.10);
        }


        .cube div:nth-child(1) {

            transform:
                translateZ(50px);
        }


        .cube div:nth-child(2) {

            transform:
                rotateY(180deg)
                translateZ(50px);
        }


        .cube div:nth-child(3) {

            transform:
                rotateY(90deg)
                translateZ(50px);
        }


        .cube div:nth-child(4) {

            transform:
                rotateY(-90deg)
                translateZ(50px);
        }


        .cube div:nth-child(5) {

            transform:
                rotateX(90deg)
                translateZ(50px);
        }


        .cube div:nth-child(6) {

            transform:
                rotateX(-90deg)
                translateZ(50px);
        }


        .cube-a {

            left: 7%;
            top: 18%;

            transform:
                scale(1.1);
        }


        .cube-b {

            right: 9%;
            top: 58%;

            transform:
                scale(0.65);

            animation-duration: 10s;

            animation-direction:
                reverse;
        }


        .cube-c {

            right: 25%;
            top: 8%;

            transform:
                scale(0.4);

            animation-duration: 20s;
        }


        .cube-d {

            left: 18%;
            bottom: 8%;

            transform:
                scale(0.55);

            animation-duration: 18s;
        }


        @keyframes cubeSpin {

            0% {

                transform:
                    rotateX(0deg)
                    rotateY(0deg)
                    rotateZ(0deg);
            }

            100% {

                transform:
                    rotateX(360deg)
                    rotateY(360deg)
                    rotateZ(360deg);
            }
        }


        /* ====================================================
           3D FLOOR GRID
        ==================================================== */

        .floor-grid {

            position: absolute;

            left: -20%;

            bottom: -38%;

            width: 140%;

            height: 80%;

            transform:
                perspective(550px)
                rotateX(60deg);

            background-image:

                linear-gradient(
                    rgba(34,211,238,0.14) 1px,
                    transparent 1px
                ),

                linear-gradient(
                    90deg,
                    rgba(34,211,238,0.14) 1px,
                    transparent 1px
                );

            background-size:
                50px 50px;

            animation:
                gridAnimation 5s linear infinite;

            opacity: 0.35;
        }


        @keyframes gridAnimation {

            from {

                background-position:
                    0 0,
                    0 0;
            }

            to {

                background-position:
                    0 50px,
                    50px 0;
            }
        }


        /* ====================================================
           STAR PARTICLES
        ==================================================== */

        .star {

            position: absolute;

            width: 3px;

            height: 3px;

            border-radius: 50%;

            background: white;

            box-shadow:
                0 0 12px
                rgba(103,232,249,0.9);

            animation:
                starMove var(--time)
                linear infinite;

            left: var(--left);

            top: var(--top);
        }


        @keyframes starMove {

            0% {

                transform:
                    translateY(110vh)
                    scale(0.3);

                opacity: 0;
            }

            15% {

                opacity: 1;
            }

            85% {

                opacity: 1;
            }

            100% {

                transform:
                    translateY(-30vh)
                    scale(1.2);

                opacity: 0;
            }
        }


        /* ====================================================
           GLASS LOGIN CARD
        ==================================================== */

        .glass-card {

            max-width: 820px;

            margin:
                25px auto 30px auto;

            padding:
                45px 40px;

            border-radius: 30px;

            background:
                linear-gradient(
                    135deg,
                    rgba(255,255,255,0.12),
                    rgba(255,255,255,0.035)
                );

            border:
                1px solid
                rgba(255,255,255,0.20);

            box-shadow:

                0 30px 100px
                rgba(0,0,0,0.55),

                inset 0 1px 0
                rgba(255,255,255,0.12);

            backdrop-filter:
                blur(22px);

            -webkit-backdrop-filter:
                blur(22px);

            text-align: center;
        }


        /* ====================================================
           TITLE
        ==================================================== */

        .main-title {

            font-size: 43px;

            font-weight: 900;

            background:
                linear-gradient(
                    90deg,
                    #ffffff,
                    #67e8f9,
                    #a78bfa,
                    #f9a8d4,
                    #ffffff
                );

            background-size:
                300% 100%;

            -webkit-background-clip:
                text;

            -webkit-text-fill-color:
                transparent;

            animation:
                titleAnimation 6s
                ease-in-out infinite;
        }


        @keyframes titleAnimation {

            0% {

                background-position:
                    0% 50%;
            }

            50% {

                background-position:
                    100% 50%;
            }

            100% {

                background-position:
                    0% 50%;
            }
        }


        .subtitle {

            color:
                rgba(255,255,255,0.72);

            font-size: 17px;

            margin-top: 12px;
        }


        .graduation-icon {

            font-size: 72px;

            animation:
                iconAnimation 3s
                ease-in-out infinite;

            filter:
                drop-shadow(
                    0 0 25px
                    rgba(34,211,238,0.55)
                );
        }


        @keyframes iconAnimation {

            0%,
            100% {

                transform:
                    translateY(0)
                    rotate(0deg);
            }

            50% {

                transform:
                    translateY(-10px)
                    rotate(2deg);
            }
        }


        /* ====================================================
           BUTTON
        ==================================================== */

        div.stButton > button {

            min-height: 50px;

            border-radius: 15px;

            font-size: 16px;

            font-weight: 700;

            color: white;

            background:
                rgba(255,255,255,0.07);

            border:
                1px solid
                rgba(255,255,255,0.18);

            transition:
                0.25s ease;
        }


        div.stButton > button:hover {

            transform:
                translateY(-4px);

            border-color:
                #67e8f9;

            box-shadow:
                0 12px 35px
                rgba(34,211,238,0.18);
        }


        /* ====================================================
           RESPONSIVE DESIGN
        ==================================================== */

        @media(max-width: 700px) {

            .glass-card {

                margin:
                    10px;

                padding:
                    30px 20px;
            }

            .main-title {

                font-size:
                    29px;
            }

            .subtitle {

                font-size:
                    14px;
            }

            .cube {

                opacity:
                    0.12;
            }
        }

        </style>


        <!-- ==================================================
             3D ANIMATED BACKGROUND
        ================================================== -->

        <div class="three-d-scene">

            <div class="glow-ball ball-1"></div>

            <div class="glow-ball ball-2"></div>

            <div class="glow-ball ball-3"></div>


            <!-- CUBE A -->

            <div class="cube cube-a">

                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>

            </div>


            <!-- CUBE B -->

            <div class="cube cube-b">

                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>

            </div>


            <!-- CUBE C -->

            <div class="cube cube-c">

                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>

            </div>


            <!-- CUBE D -->

            <div class="cube cube-d">

                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
                <div></div>

            </div>


            <!-- FLOOR -->

            <div class="floor-grid"></div>


            <!-- PARTICLES -->

            <div class="star"
                 style="
                    --left:5%;
                    --top:80%;
                    --time:11s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:12%;
                    --top:90%;
                    --time:14s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:20%;
                    --top:70%;
                    --time:12s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:28%;
                    --top:95%;
                    --time:16s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:36%;
                    --top:85%;
                    --time:10s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:45%;
                    --top:92%;
                    --time:15s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:54%;
                    --top:80%;
                    --time:13s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:63%;
                    --top:94%;
                    --time:17s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:72%;
                    --top:75%;
                    --time:12s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:81%;
                    --top:88%;
                    --time:15s;
                 ">
            </div>

            <div class="star"
                 style="
                    --left:90%;
                    --top:70%;
                    --time:11s;
                 ">
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 24. LOGIN PAGE
# ============================================================

def login_page():

    # Show animated background
    show_3d_background()


    # --------------------------------------------------------
    # LOGIN HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="glass-card">

            <div class="graduation-icon">
                🎓
            </div>

            <div class="main-title">
                Student Performance Prediction
            </div>

            <div class="subtitle">
                AI-powered academic performance
                monitoring system
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # ROLE SELECTION
    # ========================================================

    if st.session_state.login_mode is None:

        st.markdown(
            """
            <div style="
                text-align:center;
                font-size:20px;
                font-weight:700;
                margin:25px 0 15px 0;
            ">
                Choose Login
            </div>
            """,
            unsafe_allow_html=True
        )


        col1, col2 = st.columns(
            2,
            gap="large"
        )


        # ----------------------------------------------------
        # STUDENT
        # ----------------------------------------------------

        with col1:

            st.markdown(
                """
                <div class="glass-card"
                     style="
                        margin:0;
                        padding:25px;
                     ">

                    <div style="
                        font-size:45px;
                    ">
                        🎓
                    </div>

                    <h2>
                        Student
                    </h2>

                    <p style="
                        color:rgba(255,255,255,0.65);
                    ">
                        View academic performance,
                        subject marks and progress report.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            if st.button(
                "🎓 Student Login",
                width="stretch"
            ):

                st.session_state.login_mode = (
                    "student"
                )

                st.rerun()


        # ----------------------------------------------------
        # TUTOR
        # ----------------------------------------------------

        with col2:

            st.markdown(
                """
                <div class="glass-card"
                     style="
                        margin:0;
                        padding:25px;
                     ">

                    <div style="
                        font-size:45px;
                    ">
                        👨‍🏫
                    </div>

                    <h2>
                        Tutor
                    </h2>

                    <p style="
                        color:rgba(255,255,255,0.65);
                    ">
                        Manage student data,
                        analyse performance and clustering.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            if st.button(
                "👨‍🏫 Tutor Login",
                width="stretch"
            ):

                st.session_state.login_mode = (
                    "tutor"
                )

                st.rerun()


        return


    # ========================================================
    # STUDENT LOGIN
    # ========================================================

    if st.session_state.login_mode == "student":

        st.markdown(
            "## 🎓 Student Login"
        )

        with st.form(
            "student_login"
        ):

            username = st.text_input(
                "Username",
                placeholder="Enter username"
            )

            university_id = st.text_input(
                "University ID",
                placeholder="Example: KTU001"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Example: BTECH2007"
            )

            login_button = st.form_submit_button(
                "🔐 Login",
                width="stretch"
            )


        if login_button:

            if not username:

                st.error(
                    "Please enter username."
                )

            elif not university_id:

                st.error(
                    "Please enter University ID."
                )

            elif not valid_student_password(
                password
            ):

                st.error(
                    "Invalid password format."
                )

                st.caption(
                    "Demo format: BTECH2007"
                )

            else:

                df = load_database()

                student = df[
                    (
                        df["Username"].astype(str)
                        == username.strip()
                    )
                    &
                    (
                        df["University_ID"].astype(str)
                        == university_id.strip()
                    )
                ]

                if len(student) == 0:

                    st.error(
                        "Student record not found."
                    )

                else:

                    st.session_state.logged_in = True

                    st.session_state.user_role = (
                        "student"
                    )

                    st.session_state.username = (
                        username.strip()
                    )

                    st.session_state.university_id = (
                        university_id.strip()
                    )

                    st.session_state.celebrated = (
                        False
                    )

                    st.rerun()


        if st.button(
            "⬅️ Back to Role Selection"
        ):

            st.session_state.login_mode = None

            st.rerun()


        return


    # ========================================================
    # TUTOR LOGIN
    # ========================================================

    if st.session_state.login_mode == "tutor":

        st.markdown(
            "## 👨‍🏫 Tutor Login"
        )

        with st.form(
            "tutor_login"
        ):

            username = st.text_input(
                "Tutor Username",
                placeholder="Enter tutor username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )

            login_button = st.form_submit_button(
                "🔐 Tutor Login",
                width="stretch"
            )


        if login_button:

            if username not in TEACHERS:

                st.error(
                    "Tutor account not found."
                )

            elif password != TEACHERS[
                username
            ]["password"]:

                st.error(
                    "Incorrect password."
                )

            else:

                st.session_state.logged_in = True

                st.session_state.user_role = (
                    "tutor"
                )

                st.session_state.username = (
                    username
                )

                st.session_state.branch = (
                    TEACHERS[
                        username
                    ]["branch"]
                )

                st.rerun()


        if st.button(
            "⬅️ Back to Role Selection"
        ):

            st.session_state.login_mode = None

            st.rerun()


# ============================================================
# 25. LOGOUT
# ============================================================

def logout():

    st.session_state.logged_in = False

    st.session_state.login_mode = None

    st.session_state.user_role = None

    st.session_state.username = None

    st.session_state.university_id = None

    st.session_state.branch = None

    st.session_state.celebrated = False

    st.rerun()


# ============================================================
# 26. TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    st.title(
        "👨‍🏫 Tutor Dashboard"
    )

    st.caption(
        f"Logged in as: {st.session_state.username}"
    )

    st.info(
        f"Assigned Branch: "
        f"**{st.session_state.branch}**"
    )


    if st.button(
        "🚪 Logout"
    ):

        logout()


    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_database()


    # ========================================================
    # CSV UPLOAD
    # ========================================================

    st.header(
        "📤 Upload Student Data"
    )

    st.download_button(
        "⬇️ Download CSV Template",
        data=create_csv_template(),
        file_name="student_template.csv",
        mime="text/csv",
        width="stretch"
    )


    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            missing_columns = [
                column
                for column in INPUT_COLUMNS
                if column not in uploaded_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing columns: "
                    + ", ".join(
                        missing_columns
                    )
                )

            else:

                st.success(
                    f"{len(uploaded_df)} records found."
                )


                if st.button(
                    "✅ Submit Student Records",
                    width="stretch"
                ):

                    new_df = clean_input_dataframe(
                        uploaded_df
                    )


                    # Tutor branch is fixed
                    new_df["Branch"] = (
                        st.session_state.branch
                    )


                    new_df["Submitted_By"] = (
                        st.session_state.username
                    )


                    new_df["Submitted_Date"] = (
                        datetime.now()
                        .strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )


                    new_df = add_performance_columns(
                        new_df
                    )


                    new_df["Cluster"] = 0

                    new_df["RF_Prediction"] = ""

                    new_df["RF_Confidence"] = 0.0


                    new_df = new_df[
                        REQUIRED_COLUMNS
                    ]


                    combined_df = pd.concat(
                        [
                            df,
                            new_df
                        ],
                        ignore_index=True
                    )


                    combined_df = prepare_dataset(
                        combined_df
                    )


                    save_database(
                        combined_df
                    )


                    st.success(
                        "Student records submitted successfully."
                    )

                    st.rerun()


        except Exception as error:

            st.error(
                f"CSV Error: {error}"
            )


    # ========================================================
    # MANUAL ENTRY
    # ========================================================

    st.header(
        "✍️ Manual Student Entry"
    )


    with st.form(
        "manual_entry"
    ):

        col1, col2 = st.columns(2)


        with col1:

            username = st.text_input(
                "Username"
            )

            university_id = st.text_input(
                "University ID"
            )

            student_name = st.text_input(
                "Student Name"
            )

            semester = st.selectbox(
                "Semester",
                list(
                    SUBJECTS.keys()
                )
            )


        with col2:

            subject = st.selectbox(
                "Subject",
                SUBJECTS[semester]
            )

            attendance = st.number_input(
                "Attendance (%)",
                0.0,
                100.0,
                80.0
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                0.0,
                6.0,
                3.0
            )


        col3, col4, col5 = st.columns(3)


        with col3:

            internal_mark = st.number_input(
                "Internal Mark / 40",
                0.0,
                40.0,
                25.0
            )


        with col4:

            assignment = st.number_input(
                "Assignment / 15",
                0.0,
                15.0,
                10.0
            )


        with col5:

            previous_mark = st.number_input(
                "Previous Mark / 60",
                0.0,
                60.0,
                40.0
            )


        submit = st.form_submit_button(
            "➕ Add Student Record",
            width="stretch"
        )


    if submit:

        if not username:

            st.error(
                "Enter username."
            )

        elif not university_id:

            st.error(
                "Enter University ID."
            )

        elif not student_name:

            st.error(
                "Enter student name."
            )

        else:

            overall = calculate_overall(
                attendance,
                study_hours,
                internal_mark,
                assignment,
                previous_mark
            )

            performance = (
                performance_from_overall(
                    overall
                )
            )


            new_record = pd.DataFrame(
                [
                    {

                        "Username": username,

                        "University_ID":
                            university_id,

                        "Student_Name":
                            student_name,

                        "Branch":
                            st.session_state.branch,

                        "Semester":
                            semester,

                        "Subject":
                            subject,

                        "Attendance":
                            attendance,

                        "Study_Hours":
                            study_hours,

                        "Internal_Mark":
                            internal_mark,

                        "Assignment":
                            assignment,

                        "Previous_Mark":
                            previous_mark,

                        "Performance":
                            performance,

                        "Overall_Percentage":
                            overall,

                        "Cluster":
                            0,

                        "RF_Prediction":
                            "",

                        "RF_Confidence":
                            0.0,

                        "Submitted_By":
                            st.session_state.username,

                        "Submitted_Date":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                    }
                ]
            )


            combined_df = pd.concat(
                [
                    df,
                    new_record
                ],
                ignore_index=True
            )


            combined_df = prepare_dataset(
                combined_df
            )


            save_database(
                combined_df
            )


            st.success(
                "Student record added."
            )

            st.rerun()


    # ========================================================
    # BRANCH DATA
    # ========================================================

    df = load_database()

    branch_df = df[
        df["Branch"].astype(str)
        == str(
            st.session_state.branch
        )
    ].copy()


    # ========================================================
    # ANALYSIS
    # ========================================================

    st.header(
        "📊 Student Performance Analysis"
    )


    if len(branch_df) == 0:

        st.warning(
            "No student records available."
        )

        return


    branch_df = prepare_dataset(
        branch_df
    )


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    total_records = len(
        branch_df
    )

    total_students = (
        branch_df[
            "University_ID"
        ]
        .nunique()
    )

    average_attendance = (
        branch_df[
            "Attendance"
        ]
        .astype(float)
        .mean()
    )

    average_performance = (
        branch_df[
            "Overall_Percentage"
        ]
        .astype(float)
        .mean()
    )


    m1, m2, m3, m4 = st.columns(4)


    m1.metric(
        "Records",
        total_records
    )

    m2.metric(
        "Students",
        total_students
    )

    m3.metric(
        "Avg Attendance",
        f"{average_attendance:.1f}%"
    )

    m4.metric(
        "Avg Performance",
        f"{average_performance:.1f}%"
    )


    # ========================================================
    # PERFORMANCE CHART
    # ========================================================

    st.subheader(
        "📈 Performance Distribution"
    )

    performance_counts = (
        branch_df[
            "Performance"
        ]
        .value_counts()
    )

    st.bar_chart(
        performance_counts
    )


    # ========================================================
    # K-MEANS
    # ========================================================

    st.subheader(
        "🔵 K-Means Clustering"
    )

    cluster_counts = (
        branch_df[
            "Cluster"
        ]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts
    )


    st.caption(
        "Cluster numbers represent machine-learning groups, "
        "not official academic grades."
    )


    # ========================================================
    # STUDENT TABLE
    # ========================================================

    st.subheader(
        "👨‍🎓 Student Summary"
    )


    summary = (
        branch_df
        .groupby(
            [
                "University_ID",
                "Student_Name"
            ]
        )
        .agg(
            Overall_Percentage=(
                "Overall_Percentage",
                "mean"
            ),

            Attendance=(
                "Attendance",
                "mean"
            ),

            Study_Hours=(
                "Study_Hours",
                "mean"
            )
        )
        .reset_index()
    )


    summary["Performance"] = (
        summary[
            "Overall_Percentage"
        ]
        .apply(
            performance_from_overall
        )
    )


    st.dataframe(
        summary.round(2),
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD BRANCH DATA
    # ========================================================

    st.header(
        "⬇️ Download Branch Data"
    )


    branch_csv = branch_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "📥 Download Branch CSV",
        data=branch_csv,
        file_name="branch_student_data.csv",
        mime="text/csv",
        width="stretch"
    )


# ============================================================
# 27. STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    st.title(
        "🎓 Student Dashboard"
    )


    if st.button(
        "🚪 Logout"
    ):

        logout()


    df = load_database()


    student_df = df[
        (
            df["Username"].astype(str)
            == str(
                st.session_state.username
            )
        )
        &
        (
            df["University_ID"].astype(str)
            == str(
                st.session_state.university_id
            )
        )
    ].copy()


    if len(student_df) == 0:

        st.error(
            "No student records found."
        )

        return


    student_df = prepare_dataset(
        student_df
    )


    student_name = (
        student_df[
            "Student_Name"
        ]
        .iloc[0]
    )


    branch = (
        student_df[
            "Branch"
        ]
        .iloc[0]
    )


    # ========================================================
    # STUDENT INFORMATION
    # ========================================================

    st.header(
        "👤 Student Information"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Name",
        student_name
    )

    col2.metric(
        "University ID",
        st.session_state.university_id
    )

    col3.metric(
        "Branch",
        branch
    )


    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    overall = (
        student_df[
            "Overall_Percentage"
        ]
        .astype(float)
        .mean()
    )


    performance = (
        performance_from_overall(
            overall
        )
    )


    info = performance_info(
        performance
    )


    st.header(
        "📊 Overall Performance"
    )


    st.metric(
        "Overall Percentage",
        f"{overall:.2f}%"
    )


    # ========================================================
    # PERFORMANCE MESSAGE
    # ========================================================

    if performance == "Good Performance":

        st.success(
            "🟢 Excellent performance! "
            "Keep up the good work."
        )

        if not st.session_state.celebrated:

            st.balloons()

            st.session_state.celebrated = True


    elif performance == "Above Average":

        st.warning(
            "🟡 Above average performance. "
            "Keep improving."
        )


    elif performance == "Average Performance":

        st.warning(
            "🟠 Average performance. "
            "Regular revision can improve your results."
        )


    else:

        st.error(
            "🔴 Low performance. "
            "Focus on attendance, assignments and regular study."
        )


    # ========================================================
    # SUBJECT-WISE PERFORMANCE
    # ========================================================

    st.header(
        "📚 Subject-wise Performance"
    )


    for subject, group in student_df.groupby(
        "Subject"
    ):

        row = group.iloc[0]


        subject_overall = float(
            group[
                "Overall_Percentage"
            ].mean()
        )


        subject_performance = (
            performance_from_overall(
                subject_overall
            )
        )


        subject_info = performance_info(
            subject_performance
        )


        with st.expander(
            f"{subject}  |  "
            f"{subject_info['emoji']} "
            f"{subject_performance}"
        ):

            c1, c2, c3 = st.columns(3)


            c1.metric(
                "Attendance",
                f"{float(row['Attendance']):.1f}%"
            )


            c2.metric(
                "Attendance Mark",
                f"{attendance_mark(row['Attendance'])}/5"
            )


            c3.metric(
                "Study Hours",
                f"{float(row['Study_Hours']):.1f}/6"
            )


            c4, c5, c6 = st.columns(3)


            c4.metric(
                "Internal",
                f"{float(row['Internal_Mark']):.1f}/40"
            )


            c5.metric(
                "Assignment",
                f"{float(row['Assignment']):.1f}/15"
            )


            c6.metric(
                "Previous Mark",
                f"{float(row['Previous_Mark']):.1f}/60"
            )


            st.markdown(
                f"""
                <div style="
                    margin-top:15px;
                    padding:15px;
                    border-radius:15px;
                    border:1px solid
                        {subject_info['color']};
                    background:
                        {subject_info['color']}18;
                ">

                    <span style="
                        color:{subject_info['color']};
                        font-size:22px;
                    ">
                        ●
                    </span>

                    <b>
                        {subject_performance}
                    </b>

                    &nbsp;&nbsp;

                    Overall:
                    {subject_overall:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )


            st.write(
                "**Improvement / Feedback:**"
            )


            for method in improvement_methods(
                row
            ):

                st.write(
                    "• " + method
                )


    # ========================================================
    # K-MEANS CLUSTER
    # ========================================================

    st.header(
        "🔵 K-Means Cluster"
    )


    clusters = (
        student_df[
            "Cluster"
        ]
        .dropna()
        .unique()
    )


    cluster_text = ", ".join(
        str(int(cluster))
        for cluster in clusters
    )


    st.info(
        f"Your K-Means Cluster: {cluster_text}"
    )


    st.caption(
        "The cluster is a machine-learning grouping "
        "identifier and is not an academic grade."
    )


# ============================================================
# 28. MAIN PROGRAM
# ============================================================

if not st.session_state.logged_in:

    login_page()

else:

    if st.session_state.user_role == "tutor":

        tutor_dashboard()

    elif st.session_state.user_role == "student":

        student_dashboard()
