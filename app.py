import os
import re
import io
import math
import random

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
# FILE SETTINGS
# ============================================================

DATA_DIR = "data"
DATA_FILE = os.path.join(
    DATA_DIR,
    "student_performance.csv"
)

os.makedirs(DATA_DIR, exist_ok=True)


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
# ALL B.TECH DEPARTMENTS
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
# SEMESTERS AND SUBJECTS
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
# DEMO LOGIN
#
# Every department has its own tutor username/password.
# Passwords can be changed here.
# ============================================================

TUTOR_ACCOUNTS = {

    "Civil Engineering": {
        "username": "tutor_civil",
        "password": "civil123",
    },

    "Computer Science and Engineering": {
        "username": "tutor_cse",
        "password": "cse123",
    },

    "Artificial Intelligence and Data Science": {
        "username": "tutor_aids",
        "password": "aids123",
    },

    "Artificial Intelligence and Machine Learning": {
        "username": "tutor_aiml",
        "password": "aiml123",
    },

    "Information Technology": {
        "username": "tutor_it",
        "password": "it123",
    },

    "Electronics and Communication Engineering": {
        "username": "tutor_ece",
        "password": "ece123",
    },

    "Electrical and Electronics Engineering": {
        "username": "tutor_eee",
        "password": "eee123",
    },

    "Electrical Engineering": {
        "username": "tutor_ee",
        "password": "ee123",
    },

    "Mechanical Engineering": {
        "username": "tutor_me",
        "password": "me123",
    },

    "Chemical Engineering": {
        "username": "tutor_chemical",
        "password": "chemical123",
    },

    "Biotechnology and Biochemical Engineering": {
        "username": "tutor_bio",
        "password": "bio123",
    },

    "Food Technology": {
        "username": "tutor_food",
        "password": "food123",
    },

    "Production Engineering": {
        "username": "tutor_production",
        "password": "production123",
    },

    "Automobile Engineering": {
        "username": "tutor_auto",
        "password": "auto123",
    },

    "Industrial Engineering": {
        "username": "tutor_industrial",
        "password": "industrial123",
    },

    "Biomedical Engineering": {
        "username": "tutor_biomedical",
        "password": "biomedical123",
    },

    "Aeronautical Engineering": {
        "username": "tutor_aero",
        "password": "aero123",
    },

    "Applied Electronics and Instrumentation Engineering": {
        "username": "tutor_aei",
        "password": "aei123",
    },

    "Electronics and Biomedical Engineering": {
        "username": "tutor_ebm",
        "password": "ebm123",
    },

    "Electronics and Computer Engineering": {
        "username": "tutor_ececomp",
        "password": "ececomp123",
    },

    "Computer Science and Engineering (Artificial Intelligence)": {
        "username": "tutor_cseai",
        "password": "cseai123",
    },

    "Computer Science and Engineering (Data Science)": {
        "username": "tutor_csedata",
        "password": "csedata123",
    },

    "Computer Science and Engineering (Cyber Security)": {
        "username": "tutor_cyber",
        "password": "cyber123",
    },

    "Robotics and Automation": {
        "username": "tutor_robotics",
        "password": "robotics123",
    },

    "Mechatronics Engineering": {
        "username": "tutor_mechatronics",
        "password": "mechatronics123",
    },
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "page": "home",
    "student_logged_in": False,
    "tutor_logged_in": False,
    "student_username": "",
    "student_id": "",
    "student_name": "",
    "tutor_username": "",
    "tutor_branch": "",
}

for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

def load_css():

    st.markdown(
        """
<style>

/* =========================================================
   MAIN APP
   ========================================================= */

.stApp {

    background:
        radial-gradient(
            circle at 10% 20%,
            rgba(0, 180, 255, .14),
            transparent 25%
        ),

        radial-gradient(
            circle at 90% 15%,
            rgba(145, 70, 255, .16),
            transparent 25%
        ),

        radial-gradient(
            circle at 50% 90%,
            rgba(0, 240, 190, .10),
            transparent 30%
        ),

        #02040d;

    color: white;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   3D WALLPAPER
   ========================================================= */

.wallpaper {

    position: fixed;

    left: 0;
    top: 0;

    width: 100vw;
    height: 100vh;

    overflow: hidden;

    z-index: -100;

    background:

        radial-gradient(
            circle at 50% 40%,
            rgba(20, 90, 180, .13),
            transparent 45%
        ),

        linear-gradient(
            180deg,
            #030713 0%,
            #02040d 100%
        );
}


/* =========================================================
   3D GRID
   ========================================================= */

.grid {

    position: absolute;

    width: 180%;
    height: 75%;

    left: -40%;
    bottom: -20%;

    background-image:

        linear-gradient(
            rgba(0, 190, 255, .18) 1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(0, 190, 255, .18) 1px,
            transparent 1px
        );

    background-size: 65px 65px;

    transform:
        perspective(450px)
        rotateX(63deg);

    animation:
        gridMove 8s linear infinite;
}

@keyframes gridMove {

    from {

        transform:
            perspective(450px)
            rotateX(63deg)
            translateY(0);
    }

    to {

        transform:
            perspective(450px)
            rotateX(63deg)
            translateY(65px);
    }
}


/* =========================================================
   GLOWING ORBS
   ========================================================= */

.orb {

    position: absolute;

    border-radius: 50%;

    filter: blur(2px);

    opacity: .65;

    animation:
        floating 8s ease-in-out infinite;
}

.orb.one {

    width: 180px;
    height: 180px;

    left: 5%;
    top: 15%;

    background:
        radial-gradient(
            circle at 35% 30%,
            white 0%,
            #00aaff 10%,
            transparent 68%
        );
}

.orb.two {

    width: 220px;
    height: 220px;

    right: 4%;
    top: 18%;

    background:
        radial-gradient(
            circle at 40% 30%,
            white 0%,
            #8d4cff 10%,
            transparent 68%
        );

    animation-delay: -3s;
}

.orb.three {

    width: 160px;
    height: 160px;

    left: 43%;
    bottom: 10%;

    background:
        radial-gradient(
            circle at 40% 30%,
            white 0%,
            #00e0bb 10%,
            transparent 68%
        );

    animation-delay: -5s;
}

@keyframes floating {

    0%,100% {

        transform:
            translate3d(0,0,0)
            scale(1);
    }

    50% {

        transform:
            translate3d(35px,-45px,40px)
            scale(1.12);
    }
}


/* =========================================================
   PARTICLES
   ========================================================= */

.particle {

    position: absolute;

    width: 4px;
    height: 4px;

    border-radius: 50%;

    background: #69e7ff;

    box-shadow:
        0 0 15px #69e7ff;

    animation:
        particleMove 7s linear infinite;
}

.p1 { left: 15%; top: 75%; animation-delay: 0s; }
.p2 { left: 25%; top: 25%; animation-delay: 1s; }
.p3 { left: 40%; top: 70%; animation-delay: 2s; }
.p4 { left: 65%; top: 25%; animation-delay: 3s; }
.p5 { left: 80%; top: 70%; animation-delay: 4s; }
.p6 { left: 90%; top: 45%; animation-delay: 5s; }
.p7 { left: 55%; top: 15%; animation-delay: 2s; }
.p8 { left: 10%; top: 45%; animation-delay: 4s; }

@keyframes particleMove {

    0% {
        transform:
            translateY(0)
            scale(.5);
        opacity: .2;
    }

    50% {
        transform:
            translateY(-90px)
            scale(1.5);
        opacity: 1;
    }

    100% {
        transform:
            translateY(-180px)
            scale(.2);
        opacity: 0;
    }
}


/* =========================================================
   FLOATING PICTURE CARDS
   ========================================================= */

.picture {

    position: absolute;

    width: 110px;
    height: 80px;

    border-radius: 18px;

    display: flex;

    align-items: center;
    justify-content: center;

    font-size: 38px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.16),
            rgba(255,255,255,.03)
        );

    border:
        1px solid rgba(255,255,255,.18);

    box-shadow:
        0 20px 50px rgba(0,0,0,.35),
        inset 0 1px 1px rgba(255,255,255,.20);

    backdrop-filter:
        blur(12px);

    transform-style: preserve-3d;

    animation:
        pictureFloat 9s ease-in-out infinite;
}

.picture span {

    transform:
        translateZ(35px);

    filter:
        drop-shadow(
            0 0 12px
            rgba(90,220,255,.65)
        );
}

.picture.p1 {
    left: 8%;
    top: 32%;
    transform: rotateY(25deg);
}

.picture.p2 {
    right: 9%;
    top: 38%;
    transform: rotateY(-25deg);
    animation-delay: -2s;
}

.picture.p3 {
    left: 17%;
    bottom: 12%;
    transform: rotateY(-20deg);
    animation-delay: -4s;
}

.picture.p4 {
    right: 18%;
    bottom: 13%;
    transform: rotateY(20deg);
    animation-delay: -6s;
}

@keyframes pictureFloat {

    0%,100% {
        transform:
            translateY(0)
            rotateY(20deg)
            rotateX(5deg);
    }

    50% {
        transform:
            translateY(-25px)
            rotateY(-20deg)
            rotateX(-5deg);
    }
}


/* =========================================================
   LOGIN CONTAINER
   ========================================================= */

.login-area {

    min-height: 90vh;

    display: flex;

    justify-content: center;

    align-items: center;
}


/* =========================================================
   GLASS LOGIN CARD
   ========================================================= */

.login-card {

    width: min(950px, 94vw);

    padding: 45px;

    border-radius: 32px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.12),
            rgba(255,255,255,.035)
        );

    border:
        1px solid rgba(255,255,255,.17);

    box-shadow:

        0 35px 100px
        rgba(0,0,0,.55),

        inset 0 1px 0
        rgba(255,255,255,.16);

    backdrop-filter:
        blur(25px);

    transform:
        perspective(1000px)
        rotateX(1deg);
}


/* =========================================================
   MAIN HEADING ONLY
   ========================================================= */

.main-heading {

    text-align: center;

    font-size:
        clamp(30px, 5vw, 60px);

    font-weight: 950;

    letter-spacing: 2px;

    margin-bottom: 38px;

    background:

        linear-gradient(
            90deg,
            #ffffff,
            #53dfff,
            #a36cff,
            #ffffff
        );

    background-size: 300% auto;

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    animation:
        headingAnimation 5s linear infinite;

    filter:
        drop-shadow(
            0 0 20px
            rgba(70,210,255,.25)
        );
}

@keyframes headingAnimation {

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


/* =========================================================
   ROLE CARDS
   ========================================================= */

.role-card {

    min-height: 210px;

    padding: 30px 20px;

    border-radius: 25px;

    text-align: center;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.12),
            rgba(255,255,255,.035)
        );

    border:
        1px solid rgba(255,255,255,.16);

    box-shadow:
        0 20px 50px rgba(0,0,0,.30);

    transition:
        all .35s ease;

    transform-style: preserve-3d;
}

.role-card:hover {

    transform:
        translateY(-12px)
        rotateX(5deg)
        rotateY(3deg);

    box-shadow:
        0 30px 70px
        rgba(0,180,255,.20);
}

.role-icon {

    font-size: 62px;

    margin-bottom: 12px;

    transform:
        translateZ(35px);
}

.role-title {

    font-size: 26px;

    font-weight: 900;

    transform:
        translateZ(25px);
}

.role-description {

    color: #b7c7dc;

    font-size: 14px;

    margin-top: 10px;
}


/* =========================================================
   DASHBOARD
   ========================================================= */

.dashboard-title {

    font-size: 36px;

    font-weight: 900;

    margin-bottom: 20px;
}

.glass-box {

    padding: 25px;

    border-radius: 22px;

    background:
        rgba(255,255,255,.06);

    border:
        1px solid rgba(255,255,255,.12);

    box-shadow:
        0 15px 50px
        rgba(0,0,0,.25);
}


/* =========================================================
   PERFORMANCE
   ========================================================= */

.performance {

    padding: 20px;

    border-radius: 20px;

    text-align: center;

    font-size: 22px;

    font-weight: 900;

    margin: 15px 0;
}

.good {

    background:
        rgba(0,220,130,.17);

    border:
        1px solid rgba(0,255,160,.30);
}

.above {

    background:
        rgba(255,220,0,.17);

    border:
        1px solid rgba(255,220,0,.30);
}

.average {

    background:
        rgba(255,150,0,.17);

    border:
        1px solid rgba(255,150,0,.30);
}

.low {

    background:
        rgba(255,50,60,.17);

    border:
        1px solid rgba(255,50,60,.30);
}


/* =========================================================
   MOBILE
   ========================================================= */

@media(max-width:700px) {

    .login-card {
        padding: 25px 18px;
    }

    .picture {
        display: none;
    }

    .main-heading {
        font-size: 29px;
    }

}

</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 3D WALLPAPER HTML
# ============================================================

def show_wallpaper():

    st.markdown(
        """
<div class="wallpaper">

    <div class="grid"></div>

    <div class="orb one"></div>
    <div class="orb two"></div>
    <div class="orb three"></div>

    <div class="particle p1"></div>
    <div class="particle p2"></div>
    <div class="particle p3"></div>
    <div class="particle p4"></div>
    <div class="particle p5"></div>
    <div class="particle p6"></div>
    <div class="particle p7"></div>
    <div class="particle p8"></div>

    <!-- Floating academic picture cards -->

    <div class="picture p1">
        <span>🎓</span>
    </div>

    <div class="picture p2">
        <span>📊</span>
    </div>

    <div class="picture p3">
        <span>🤖</span>
    </div>

    <div class="picture p4">
        <span>💻</span>
    </div>

</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA
# ============================================================

def empty_data():

    return pd.DataFrame(
        columns=COLUMNS
    )


def load_data():

    if not os.path.exists(DATA_FILE):
        return empty_data()

    try:

        df = pd.read_csv(
            DATA_FILE
        )

    except Exception:

        return empty_data()

    for col in COLUMNS:

        if col not in df.columns:
            df[col] = np.nan

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

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


def save_data(df):

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    df.to_csv(
        DATA_FILE,
        index=False
    )


# ============================================================
# ATTENDANCE
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
# PERFORMANCE
# ============================================================

def calculate_performance(
    attendance,
    internal,
    assignment,
    previous,
    study_hours,
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

        performance = "Low Performance"

    elif overall < 65:

        performance = "Average Performance"

    elif overall < 80:

        performance = "Above Average Performance"

    else:

        performance = "Good Performance"

    return (
        round(overall, 2),
        performance
    )


# ============================================================
# PERFORMANCE ICON
# ============================================================

def performance_icon(level):

    if "Low" in level:
        return "🔴"

    if "Above Average" in level:
        return "🟡"

    if "Average" in level:
        return "🟠"

    return "🟢"


def performance_class(level):

    if "Low" in level:
        return "low"

    if "Above Average" in level:
        return "above"

    if "Average" in level:
        return "average"

    return "good"


# ============================================================
# GUIDANCE
# ============================================================

def get_guidance(row):

    advice = []

    if row["Attendance"] < 75:
        advice.append(
            "Improve attendance and attend classes regularly."
        )

    if row["Internal_Mark"] < 20:
        advice.append(
            "Increase preparation for internal examinations."
        )

    if row["Assignment"] < 8:
        advice.append(
            "Complete assignments regularly and on time."
        )

    if row["Previous_Mark"] < 30:
        advice.append(
            "Revise previous topics and strengthen fundamentals."
        )

    if row["Study_Hours"] < 3:
        advice.append(
            "Increase daily study and revision time."
        )

    if not advice:

        advice.append(
            "Excellent performance! Keep up the good work."
        )

    return advice


# ============================================================
# STUDENT PASSWORD
# ============================================================

def valid_student_password(password):

    if not isinstance(
        password,
        str
    ):
        return False

    if not re.fullmatch(
        r"BTECH(20\d{2})",
        password
    ):
        return False

    year = int(
        password[-4:]
    )

    return 2000 <= year <= 2022


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

    work = df.copy()

    X = work[
        features
    ].fillna(0)

    clusters = min(
        3,
        len(work)
    )

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    work[
        "KMeans_Cluster"
    ] = model.fit_predict(X)

    score = None

    if len(
        set(
            work["KMeans_Cluster"]
        )
    ) > 1:

        score = silhouette_score(
            X,
            work["KMeans_Cluster"]
        )

    return work, score


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

    work = df.copy()

    X = work[
        features
    ].fillna(0)

    y = work[
        "Performance"
    ].astype(str)

    if y.nunique() < 2:
        return None, None

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=.20,
            random_state=42,
            stratify=y,
        )

    except ValueError:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=.20,
            random_state=42,
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

    return model, accuracy


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf(
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
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=19,
        leading=23,
        spaceAfter=15,
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    story = []

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PREDICTION",
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
        ["Department", branch],
        ["Semester", semester],
    ]

    table = Table(
        details,
        colWidths=[
            140,
            330
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                .5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            ),
        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 15)
    )

    # Overall
    if len(df) > 0:

        overall = df[
            "Performance_Percent"
        ].mean()

        mode = df[
            "Performance"
        ].mode()

        level = (
            mode.iloc[0]
            if not mode.empty
            else "Not Available"
        )

        story.append(
            Paragraph(
                "OVERALL PERFORMANCE",
                heading_style
            )
        )

        overall_table = Table(
            [
                [
                    "Overall Percentage",
                    f"{overall:.2f}%"
                ],
                [
                    "Overall Performance",
                    level
                ],
            ],
            colWidths=[
                190,
                280
            ]
        )

        overall_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    .5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
            ])
        )

        story.append(
            overall_table
        )

        story.append(
            Spacer(1, 15)
        )

    # Subject records
    story.append(
        Paragraph(
            "SUBJECT-WISE PERFORMANCE",
            heading_style
        )
    )

    for _, row in df.iterrows():

        level = row[
            "Performance"
        ]

        subject_table = Table(
            [
                [
                    "Subject",
                    str(row["Subject"])
                ],
                [
                    "Attendance",
                    f"{row['Attendance']:.1f}%"
                ],
                [
                    "Attendance Mark",
                    f"{attendance_mark(row['Attendance'])}/5"
                ],
                [
                    "Study Hours",
                    f"{row['Study_Hours']:.1f}/6"
                ],
                [
                    "Internal",
                    f"{row['Internal_Mark']:.1f}/40"
                ],
                [
                    "Assignment",
                    f"{row['Assignment']:.1f}/15"
                ],
                [
                    "Previous Mark",
                    f"{row['Previous_Mark']:.1f}/60"
                ],
                [
                    "Performance",
                    f"{performance_icon(level)} {level}"
                ],
                [
                    "Percentage",
                    f"{row['Performance_Percent']:.2f}%"
                ],
                [
                    "K-Means Cluster",
                    str(row["KMeans_Cluster"])
                ],
            ],
            colWidths=[
                190,
                280
            ]
        )

        subject_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    .5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
            ])
        )

        story.append(
            subject_table
        )

        story.append(
            Spacer(1, 8)
        )

        story.append(
            Paragraph(
                "<b>Improvement / Guidance:</b>",
                normal_style
            )
        )

        for advice in get_guidance(row):

            story.append(
                Paragraph(
                    "• " + advice,
                    normal_style
                )
            )

        story.append(
            Spacer(1, 15)
        )

    story.append(
        Paragraph(
            "ATTENDANCE CONVERSION",
            heading_style
        )
    )

    conversion = [
        ["Attendance", "Mark"],
        ["90–100%", "5"],
        ["80–89%", "4"],
        ["70–79%", "3"],
        ["60–69%", "2"],
        ["10–59%", "1"],
        ["Below 10%", "0"],
    ]

    conversion_table = Table(
        conversion,
        colWidths=[
            230,
            230
        ]
    )

    conversion_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                .5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(
        conversion_table
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "Note: Performance levels and percentage calculations "
            "are project-defined assessment rules and are not "
            "official KTU grading rules.",
            normal_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HOME
# ============================================================

def home():

    show_wallpaper()

    st.markdown(
        '<div class="login-area">',
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
                    STUDENT
                </div>

                <div class="role-description">
                    View your subject-wise academic
                    performance and progress report.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🎓 Student Login",
            width="stretch"
        ):

            st.session_state[
                "page"
            ] = "student_login"

            st.rerun()

    with col2:

        st.markdown(
            """
            <div class="role-card">

                <div class="role-icon">
                    👨‍🏫
                </div>

                <div class="role-title">
                    TUTOR
                </div>

                <div class="role-description">
                    Manage student data, analyse
                    performance and generate reports.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "👨‍🏫 Tutor Login",
            width="stretch"
        ):

            st.session_state[
                "page"
            ] = "tutor_login"

            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    show_wallpaper()

    st.markdown(
        '<div class="login-area">',
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

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "🔐 LOGIN",
            width="stretch"
        ):

            if not username or not university_id or not password:

                st.error(
                    "Please enter all details."
                )

            elif not valid_student_password(password):

                st.error(
                    "Password must be BTECH2000 to BTECH2022."
                )

            else:

                df = load_data()

                student = df[
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

                if len(student) == 0:

                    st.error(
                        "Student record not found."
                    )

                else:

                    st.session_state[
                        "student_logged_in"
                    ] = True

                    st.session_state[
                        "student_username"
                    ] = username

                    st.session_state[
                        "student_id"
                    ] = university_id

                    st.session_state[
                        "student_name"
                    ] = student.iloc[0][
                        "Student_Name"
                    ]

                    st.session_state[
                        "page"
                    ] = "student_dashboard"

                    st.rerun()

    with c2:

        if st.button(
            "⬅ BACK",
            width="stretch"
        ):

            st.session_state[
                "page"
            ] = "home"

            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ============================================================
# TUTOR LOGIN - ALL DEPARTMENTS
# ============================================================

def tutor_login():

    show_wallpaper()

    st.markdown(
        '<div class="login-area">',
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

    st.info(
        "Select your B.Tech department and enter the "
        "department tutor account."
    )

    department = st.selectbox(
        "🏫 Select Department",
        BRANCHES
    )

    username = st.text_input(
        "Tutor Username"
    )

    password = st.text_input(
        "Tutor Password",
        type="password"
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "🔐 LOGIN AS TUTOR",
            type="primary",
            width="stretch"
        ):

            account = TUTOR_ACCOUNTS.get(
                department
            )

            if account is None:

                st.error(
                    "Tutor account is not configured."
                )

            elif (
                username == account["username"]
                and
                password == account["password"]
            ):

                st.session_state[
                    "tutor_logged_in"
                ] = True

                st.session_state[
                    "tutor_username"
                ] = username

                st.session_state[
                    "tutor_branch"
                ] = department

                st.session_state[
                    "page"
                ] = "tutor_dashboard"

                st.success(
                    f"Login successful — {department}"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password "
                    "for the selected department."
                )

    with c2:

        if st.button(
            "⬅ BACK",
            width="stretch"
        ):

            st.session_state[
                "page"
            ] = "home"

            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    if not st.session_state[
        "student_logged_in"
    ]:

        st.session_state[
            "page"
        ] = "student_login"

        st.rerun()

    df = load_data()

    name = st.session_state[
        "student_name"
    ]

    student_df = df[
        (
            df["Username"].astype(str)
            ==
            st.session_state[
                "student_username"
            ]
        )
        &
        (
            df["University_ID"].astype(str)
            ==
            st.session_state[
                "student_id"
            ]
        )
    ].copy()

    st.markdown(
        f"""
        <div class="dashboard-title">
            🎓 {name}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚪 Logout"):

        st.session_state[
            "student_logged_in"
        ] = False

        st.session_state[
            "page"
        ] = "home"

        st.rerun()

    if len(student_df) == 0:

        st.warning(
            "No performance data available."
        )

        return

    semesters = sorted(
        student_df[
            "Semester"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    semester = st.selectbox(
        "📚 Select Semester",
        semesters
    )

    semester_df = student_df[
        student_df[
            "Semester"
        ].astype(str)
        == semester
    ].copy()

    if len(semester_df) == 0:

        st.info(
            "No records available."
        )

        return

    overall = semester_df[
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

    css = performance_class(
        overall_level
    )

    st.markdown(
        f"""
        <div class="performance {css}">

            {performance_icon(overall_level)}

            Overall Performance:
            {overall_level}

            <br>

            <span style="font-size:17px;">
                {overall:.2f}%
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "📚 Subject-wise Performance"
    )

    for _, row in semester_df.iterrows():

        level = row[
            "Performance"
        ]

        with st.expander(
            f"{performance_icon(level)} {row['Subject']}"
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
                    "Previous",
                    f"{row['Previous_Mark']:.1f}/60"
                )

            with c3:

                st.metric(
                    "Study Hours",
                    f"{row['Study_Hours']:.1f}/6"
                )

                st.metric(
                    "Percentage",
                    f"{row['Performance_Percent']:.2f}%"
                )

            st.write(
                f"**Performance:** {level}"
            )

            st.write(
                "**Guidance:**"
            )

            for advice in get_guidance(row):

                st.write(
                    f"• {advice}"
                )

    pdf = create_pdf(
        name,
        st.session_state["student_id"],
        str(semester_df.iloc[0]["Branch"]),
        semester,
        semester_df
    )

    st.download_button(
        "📄 Download Progress Report PDF",
        data=pdf,
        file_name=(
            f"{name}_{semester}_Progress_Report.pdf"
        ),
        mime="application/pdf",
        width="stretch"
    )


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    if not st.session_state[
        "tutor_logged_in"
    ]:

        st.session_state[
            "page"
        ] = "tutor_login"

        st.rerun()

    branch = st.session_state[
        "tutor_branch"
    ]

    st.markdown(
        """
        <div class="dashboard-title">
            👨‍🏫 Tutor Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        f"Department: {branch}"
    )

    if st.button(
        "🚪 Logout"
    ):

        st.session_state[
            "tutor_logged_in"
        ] = False

        st.session_state[
            "page"
        ] = "home"

        st.rerun()

    df = load_data()

    branch_df = df[
        df["Branch"].astype(str)
        == branch
    ].copy()

    tabs = st.tabs(
        [
            "➕ Add Student",
            "📊 Student Data",
            "🤖 K-Means",
            "🌲 Random Forest",
            "📄 Reports",
        ]
    )

    # ========================================================
    # ADD STUDENT
    # ========================================================

    with tabs[0]:

        st.subheader(
            "Add Student Records"
        )

        student_name = st.text_input(
            "Student Name",
            key="student_name_input"
        )

        username = st.text_input(
            "Student Username",
            key="username_input"
        )

        university_id = st.text_input(
            "University ID",
            key="university_input"
        )

        semester = st.selectbox(
            "Semester",
            list(SEMESTERS.keys()),
            key="semester_input"
        )

        st.markdown(
            "### Subject-wise Details"
        )

        records = []

        for subject in SEMESTERS[
            semester
        ]:

            st.markdown(
                f"#### 📘 {subject}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                attendance = st.number_input(
                    "Attendance %",
                    0.0,
                    100.0,
                    75.0,
                    1.0,
                    key=f"a_{semester}_{subject}"
                )

            with c2:

                internal = st.number_input(
                    "Internal /40",
                    0.0,
                    40.0,
                    20.0,
                    1.0,
                    key=f"i_{semester}_{subject}"
                )

            with c3:

                assignment = st.number_input(
                    "Assignment /15",
                    0.0,
                    15.0,
                    8.0,
                    1.0,
                    key=f"as_{semester}_{subject}"
                )

            with c4:

                previous = st.number_input(
                    "Previous /60",
                    0.0,
                    60.0,
                    30.0,
                    1.0,
                    key=f"p_{semester}_{subject}"
                )

            with c5:

                study = st.number_input(
                    "Study Hours",
                    0.0,
                    6.0,
                    3.0,
                    .5,
                    key=f"s_{semester}_{subject}"
                )

            percent, level = calculate_performance(
                attendance,
                internal,
                assignment,
                previous,
                study
            )

            records.append({
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
                "Performance": level,
                "Performance_Percent": percent,
                "KMeans_Cluster": np.nan,
            })

        if st.button(
            "💾 SUBMIT STUDENT",
            type="primary",
            width="stretch"
        ):

            if not student_name.strip():

                st.error(
                    "Enter student name."
                )

            elif not username.strip():

                st.error(
                    "Enter username."
                )

            elif not university_id.strip():

                st.error(
                    "Enter University ID."
                )

            else:

                new_df = pd.DataFrame(
                    records
                )

                # Replace existing same student/semester
                if len(df) > 0:

                    df = df[
                        ~(
                            (
                                df["University_ID"].astype(str)
                                == university_id
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
                    [
                        df,
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
    # STUDENT DATA
    # ========================================================

    with tabs[1]:

        st.subheader(
            f"📊 {branch} Student Data"
        )

        if len(branch_df) == 0:

            st.info(
                "No students available."
            )

        else:

            st.dataframe(
                branch_df,
                width="stretch",
                hide_index=True
            )

            csv = branch_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Department CSV",
                csv,
                "department_student_data.csv",
                "text/csv",
                width="stretch"
            )

            st.markdown(
                "### 🗑️ Delete Student"
            )

            students = (
                branch_df[
                    [
                        "Student_Name",
                        "University_ID"
                    ]
                ]
                .drop_duplicates()
            )

            options = students.apply(
                lambda x:
                f"{x['Student_Name']} | "
                f"{x['University_ID']}",
                axis=1
            ).tolist()

            selected = st.selectbox(
                "Select Student",
                options
            )

            if st.button(
                "🗑️ DELETE COMPLETE STUDENT"
            ):

                selected_id = (
                    selected.split("|")[-1]
                    .strip()
                )

                all_df = load_data()

                all_df = all_df[
                    ~(
                        (
                            all_df["Branch"].astype(str)
                            == branch
                        )
                        &
                        (
                            all_df["University_ID"].astype(str)
                            == selected_id
                        )
                    )
                ]

                save_data(
                    all_df
                )

                st.success(
                    "Complete student history deleted."
                )

                st.rerun()


    # ========================================================
    # K-MEANS
    # ========================================================

    with tabs[2]:

        st.subheader(
            "🤖 K-Means Clustering"
        )

        if len(branch_df) < 3:

            st.warning(
                "Minimum 3 records required."
            )

        else:

            clustered, score = run_kmeans(
                branch_df
            )

            if score is not None:

                st.metric(
                    "Silhouette Score",
                    f"{score:.4f}"
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
                hide_index=True
            )

            if st.button(
                "💾 SAVE CLUSTERS"
            ):

                all_data = load_data()

                for idx in clustered.index:

                    all_data.loc[
                        idx,
                        "KMeans_Cluster"
                    ] = clustered.loc[
                        idx,
                        "KMeans_Cluster"
                    ]

                save_data(
                    all_data
                )

                st.success(
                    "Clusters saved."
                )

                st.rerun()

            st.info(
                "K-Means cluster numbers are identifiers, "
                "not official performance grades."
            )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    with tabs[3]:

        st.subheader(
            "🌲 Random Forest"
        )

        model, accuracy = train_random_forest(
            branch_df
        )

        if model is None:

            st.warning(
                "More student records are required "
                "to train Random Forest."
            )

        else:

            st.metric(
                "Model Accuracy",
                f"{accuracy * 100:.2f}%"
            )

            st.success(
                "Random Forest trained successfully."
            )

            st.write(
                """
                Features used:

                • Attendance

                • Internal Mark

                • Assignment Score

                • Previous Mark

                • Study Hours
                """
            )


    # ========================================================
    # REPORTS
    # ========================================================

    with tabs[4]:

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

            options = students.apply(
                lambda x:
                f"{x['Student_Name']} | "
                f"{x['University_ID']} | "
                f"{x['Semester']}",
                axis=1
            ).tolist()

            selected = st.selectbox(
                "Select Student",
                options
            )

            parts = [
                x.strip()
                for x in selected.split("|")
            ]

            name = parts[0]
            university_id = parts[1]
            semester = parts[2]

            student_df = branch_df[
                (
                    branch_df[
                        "Student_Name"
                    ].astype(str)
                    == name
                )
                &
                (
                    branch_df[
                        "University_ID"
                    ].astype(str)
                    == university_id
                )
                &
                (
                    branch_df[
                        "Semester"
                    ].astype(str)
                    == semester
                )
            ].copy()

            st.dataframe(
                student_df,
                width="stretch",
                hide_index=True
            )

            pdf = create_pdf(
                name,
                university_id,
                branch,
                semester,
                student_df
            )

            st.download_button(
                "📄 DOWNLOAD PROGRESS REPORT PDF",
                pdf,
                f"{name}_{semester}_Progress_Report.pdf",
                "application/pdf",
                width="stretch"
            )


# ============================================================
# APPLICATION ROUTER
# ============================================================

load_css()

if st.session_state["page"] == "home":

    home()

elif st.session_state["page"] == "student_login":

    student_login()

elif st.session_state["page"] == "tutor_login":

    tutor_login()

elif st.session_state["page"] == "student_dashboard":

    student_dashboard()

elif st.session_state["page"] == "tutor_dashboard":

    tutor_dashboard()
