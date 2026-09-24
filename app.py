import os
import re
import io

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================
# FILE SETTINGS
# ==========================================================

DATA_DIR = "data"
DATA_FILE = os.path.join(
    DATA_DIR,
    "student_performance.csv"
)

os.makedirs(DATA_DIR, exist_ok=True)


# ==========================================================
# COLUMNS
# ==========================================================

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


# ==========================================================
# DEPARTMENTS
# ==========================================================

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


# ==========================================================
# SEMESTERS
# ==========================================================

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


# ==========================================================
# TUTOR LOGIN ACCOUNTS
# ==========================================================

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


# ==========================================================
# SESSION STATE
# ==========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "student_logged_in" not in st.session_state:
    st.session_state.student_logged_in = False

if "tutor_logged_in" not in st.session_state:
    st.session_state.tutor_logged_in = False

if "student_username" not in st.session_state:
    st.session_state.student_username = ""

if "student_id" not in st.session_state:
    st.session_state.student_id = ""

if "student_name" not in st.session_state:
    st.session_state.student_name = ""

if "tutor_username" not in st.session_state:
    st.session_state.tutor_username = ""

if "tutor_branch" not in st.session_state:
    st.session_state.tutor_branch = ""


# ==========================================================
# CSS
# ==========================================================

def load_css():

    st.markdown(
        """
<style>

/* ===============================
   GLOBAL
   =============================== */

html, body, [data-testid="stAppViewContainer"] {
    margin: 0;
    padding: 0;
}

.stApp {
    background: #02040b;
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


/* ===============================
   3D BACKGROUND
   =============================== */

.background-3d {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    z-index: -999;
    background:
        radial-gradient(
            circle at 50% 40%,
            rgba(0, 150, 255, 0.12),
            transparent 38%
        ),
        radial-gradient(
            circle at 15% 20%,
            rgba(110, 60, 255, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 70%,
            rgba(0, 230, 190, 0.09),
            transparent 30%
        ),
        #02040b;
}


/* ===============================
   GRID
   =============================== */

.grid-3d {
    position: absolute;
    width: 200%;
    height: 80%;
    left: -50%;
    bottom: -25%;

    background-image:
        linear-gradient(
            rgba(0, 190, 255, 0.16) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(0, 190, 255, 0.16) 1px,
            transparent 1px
        );

    background-size: 60px 60px;

    transform:
        perspective(450px)
        rotateX(62deg);

    animation:
        gridAnimation 7s linear infinite;
}

@keyframes gridAnimation {

    from {
        transform:
            perspective(450px)
            rotateX(62deg)
            translateY(0);
    }

    to {
        transform:
            perspective(450px)
            rotateX(62deg)
            translateY(60px);
    }
}


/* ===============================
   GLOWING SPHERES
   =============================== */

.sphere {
    position: absolute;
    border-radius: 50%;
    filter: blur(1px);
    opacity: 0.65;
    animation: sphereFloat 9s ease-in-out infinite;
}

.sphere1 {
    width: 180px;
    height: 180px;
    left: 3%;
    top: 15%;

    background:
        radial-gradient(
            circle at 35% 30%,
            white 0%,
            #00b7ff 8%,
            transparent 68%
        );
}

.sphere2 {
    width: 210px;
    height: 210px;
    right: 4%;
    top: 12%;

    background:
        radial-gradient(
            circle at 35% 30%,
            white 0%,
            #7a43ff 8%,
            transparent 68%
        );

    animation-delay: -3s;
}

.sphere3 {
    width: 150px;
    height: 150px;
    left: 43%;
    bottom: 7%;

    background:
        radial-gradient(
            circle at 35% 30%,
            white 0%,
            #00d8b0 8%,
            transparent 68%
        );

    animation-delay: -6s;
}

@keyframes sphereFloat {

    0%, 100% {
        transform:
            translate3d(0,0,0)
            scale(1);
    }

    50% {
        transform:
            translate3d(30px,-40px,40px)
            scale(1.12);
    }
}


/* ===============================
   FLOATING ACADEMIC OBJECTS
   =============================== */

.float-card {
    position: absolute;

    width: 100px;
    height: 75px;

    display: flex;
    justify-content: center;
    align-items: center;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.13),
            rgba(255,255,255,.035)
        );

    border:
        1px solid rgba(255,255,255,.16);

    backdrop-filter: blur(14px);

    box-shadow:
        0 20px 60px rgba(0,0,0,.45),
        inset 0 1px 1px rgba(255,255,255,.18);

    font-size: 38px;

    animation:
        cardFloat 8s ease-in-out infinite;
}

.float-card span {
    transform: translateZ(35px);
}

.fc1 {
    left: 8%;
    top: 32%;
}

.fc2 {
    right: 8%;
    top: 35%;
    animation-delay: -2s;
}

.fc3 {
    left: 13%;
    bottom: 12%;
    animation-delay: -4s;
}

.fc4 {
    right: 13%;
    bottom: 13%;
    animation-delay: -6s;
}

@keyframes cardFloat {

    0%,100% {
        transform:
            translateY(0)
            rotateY(15deg)
            rotateX(4deg);
    }

    50% {
        transform:
            translateY(-25px)
            rotateY(-15deg)
            rotateX(-4deg);
    }
}


/* ===============================
   PARTICLES
   =============================== */

.particle {
    position: absolute;

    width: 4px;
    height: 4px;

    border-radius: 50%;

    background: #62dcff;

    box-shadow:
        0 0 15px #62dcff;

    animation:
        particleAnimation 8s linear infinite;
}

.particle:nth-child(1) {
    left: 12%;
    top: 70%;
}

.particle:nth-child(2) {
    left: 22%;
    top: 25%;
    animation-delay: 1s;
}

.particle:nth-child(3) {
    left: 35%;
    top: 70%;
    animation-delay: 2s;
}

.particle:nth-child(4) {
    left: 55%;
    top: 20%;
    animation-delay: 3s;
}

.particle:nth-child(5) {
    left: 72%;
    top: 65%;
    animation-delay: 4s;
}

.particle:nth-child(6) {
    left: 90%;
    top: 45%;
    animation-delay: 5s;
}

@keyframes particleAnimation {

    0% {
        transform:
            translateY(0)
            scale(.5);
        opacity: .1;
    }

    50% {
        transform:
            translateY(-100px)
            scale(1.5);
        opacity: 1;
    }

    100% {
        transform:
            translateY(-200px)
            scale(.2);
        opacity: 0;
    }
}


/* ===============================
   CENTERING
   =============================== */

.login-wrapper {

    min-height: 100vh;

    width: 100%;

    display: flex;

    align-items: center;

    justify-content: center;

    padding: 30px;

    box-sizing: border-box;
}


/* ===============================
   LOGIN CARD
   =============================== */

.login-card {

    width: min(900px, 94vw);

    margin: auto;

    padding: 42px;

    border-radius: 32px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.12),
            rgba(255,255,255,.035)
        );

    border:
        1px solid rgba(255,255,255,.18);

    backdrop-filter: blur(25px);

    box-shadow:
        0 35px 100px rgba(0,0,0,.60),
        inset 0 1px 1px rgba(255,255,255,.18);

    box-sizing: border-box;
}


/* ===============================
   MAIN HEADING
   =============================== */

.main-heading {

    width: 100%;

    text-align: center;

    font-size:
        clamp(30px, 5vw, 58px);

    font-weight: 950;

    letter-spacing: 2px;

    line-height: 1.15;

    margin:
        0 0 35px 0;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #50ddff,
            #a56cff,
            #ffffff
        );

    background-size: 300% auto;

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    animation:
        headingGlow 5s linear infinite;

    filter:
        drop-shadow(
            0 0 20px
            rgba(60,200,255,.25)
        );
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


/* ===============================
   ROLE CARDS
   =============================== */

.role-card {

    min-height: 210px;

    display: flex;

    flex-direction: column;

    justify-content: center;

    align-items: center;

    text-align: center;

    padding: 25px;

    border-radius: 25px;

    background:
        rgba(255,255,255,.055);

    border:
        1px solid rgba(255,255,255,.14);

    transition:
        .35s ease;

    box-shadow:
        0 15px 50px rgba(0,0,0,.30);
}

.role-card:hover {

    transform:
        translateY(-10px)
        scale(1.02);

    border-color:
        rgba(80,220,255,.45);

    box-shadow:
        0 25px 70px
        rgba(0,180,255,.18);
}

.role-icon {
    font-size: 60px;
    margin-bottom: 12px;
}

.role-title {
    font-size: 25px;
    font-weight: 900;
}

.role-description {
    color: #b8c7d9;
    font-size: 14px;
    margin-top: 8px;
}


/* ===============================
   INPUTS
   =============================== */

.stTextInput input,
.stSelectbox div[data-baseweb="select"] {

    background:
        rgba(255,255,255,.07) !important;

    border:
        1px solid rgba(255,255,255,.15) !important;

    color: white !important;

    border-radius: 12px !important;
}


/* ===============================
   BUTTONS
   =============================== */

.stButton > button,
.stDownloadButton > button {

    border-radius: 12px !important;

    min-height: 45px;

    font-weight: 800 !important;

    border:
        1px solid rgba(255,255,255,.16) !important;

    background:
        linear-gradient(
            135deg,
            rgba(0,180,255,.20),
            rgba(130,70,255,.20)
        ) !important;

    color: white !important;

    transition:
        .25s ease !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {

    transform:
        translateY(-2px);

    border-color:
        rgba(80,220,255,.65) !important;

    box-shadow:
        0 10px 30px
        rgba(0,180,255,.20);
}


/* ===============================
   DASHBOARD
   =============================== */

.dashboard-container {
    padding: 25px;
}

.dashboard-title {

    font-size: 36px;

    font-weight: 900;

    margin-bottom: 20px;
}


/* ===============================
   MOBILE
   =============================== */

@media(max-width:700px) {

    .login-wrapper {
        padding: 15px;
    }

    .login-card {
        padding: 25px 18px;
    }

    .float-card {
        display: none;
    }

    .main-heading {
        font-size: 30px;
    }
}

</style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# BACKGROUND
# ==========================================================

def background():

    st.markdown(
        """
<div class="background-3d">

    <div class="grid-3d"></div>

    <div class="sphere sphere1"></div>
    <div class="sphere sphere2"></div>
    <div class="sphere sphere3"></div>

    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>

    <div class="float-card fc1">
        <span>🎓</span>
    </div>

    <div class="float-card fc2">
        <span>📊</span>
    </div>

    <div class="float-card fc3">
        <span>🤖</span>
    </div>

    <div class="float-card fc4">
        <span>💻</span>
    </div>

</div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# DATA FUNCTIONS
# ==========================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:

        if col not in df.columns:
            df[col] = np.nan

    return df[COLUMNS]


def save_data(df):

    df.to_csv(
        DATA_FILE,
        index=False
    )


# ==========================================================
# ATTENDANCE
# ==========================================================

def attendance_mark(attendance):

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


# ==========================================================
# PERFORMANCE
# ==========================================================

def calculate_performance(
    attendance,
    internal,
    assignment,
    previous,
    study,
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
        study / 6 * 100,
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


# ==========================================================
# PASSWORD
# ==========================================================

def valid_student_password(password):

    return bool(
        re.fullmatch(
            r"BTECH(20\d{2})",
            password
        )
        and
        2000 <= int(password[-4:]) <= 2022
    )


# ==========================================================
# GUIDANCE
# ==========================================================

def guidance(row):

    result = []

    if row["Attendance"] < 75:
        result.append(
            "Improve attendance and attend classes regularly."
        )

    if row["Internal_Mark"] < 20:
        result.append(
            "Increase preparation for internal examinations."
        )

    if row["Assignment"] < 8:
        result.append(
            "Complete assignments regularly."
        )

    if row["Previous_Mark"] < 30:
        result.append(
            "Revise previous topics and strengthen fundamentals."
        )

    if row["Study_Hours"] < 3:
        result.append(
            "Increase daily study and revision time."
        )

    if not result:
        result.append(
            "Excellent performance! Keep up the good work."
        )

    return result


# ==========================================================
# K-MEANS
# ==========================================================

def run_kmeans(df):

    features = [
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
    ]

    if len(df) < 3:
        return df.copy(), None

    work = df.copy()

    X = work[
        features
    ].fillna(0)

    k = min(
        3,
        len(work)
    )

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    work[
        "KMeans_Cluster"
    ] = model.fit_predict(X)

    score = None

    if work[
        "KMeans_Cluster"
    ].nunique() > 1:

        score = silhouette_score(
            X,
            work["KMeans_Cluster"]
        )

    return work, score


# ==========================================================
# RANDOM FOREST
# ==========================================================

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

    X = df[
        features
    ].fillna(0)

    y = df[
        "Performance"
    ].astype(str)

    if y.nunique() < 2:
        return None, None

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=.2,
            random_state=42,
            stratify=y,
        )

    except ValueError:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=.2,
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


# ==========================================================
# PDF
# ==========================================================

def make_pdf(
    name,
    university_id,
    branch,
    semester,
    data,
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

    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=15,
    )

    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=8,
    )

    normal = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    story = []

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PREDICTION",
            title
        )
    )

    story.append(
        Paragraph(
            "STUDENT PROGRESS REPORT",
            heading
        )
    )

    details = Table(
        [
            ["Student Name", name],
            ["University ID", university_id],
            ["Department", branch],
            ["Semester", semester],
        ],
        colWidths=[150, 320]
    )

    details.setStyle(
        TableStyle([
            ("GRID", (0,0), (-1,-1), .5, colors.grey),
            ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("PADDING", (0,0), (-1,-1), 7),
        ])
    )

    story.append(details)

    story.append(
        Spacer(1, 15)
    )

    if len(data) > 0:

        overall = data[
            "Performance_Percent"
        ].mean()

        story.append(
            Paragraph(
                f"<b>Overall Performance:</b> "
                f"{overall:.2f}%",
                heading
            )
        )

    story.append(
        Paragraph(
            "SUBJECT-WISE PERFORMANCE",
            heading
        )
    )

    for _, row in data.iterrows():

        table = Table(
            [
                ["Subject", str(row["Subject"])],
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
                    str(row["Performance"])
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
            colWidths=[170, 300]
        )

        table.setStyle(
            TableStyle([
                ("GRID", (0,0), (-1,-1), .5, colors.grey),
                ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
                ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                ("PADDING", (0,0), (-1,-1), 6),
            ])
        )

        story.append(table)

        story.append(
            Paragraph(
                "<b>Improvement / Guidance</b>",
                normal
            )
        )

        for item in guidance(row):

            story.append(
                Paragraph(
                    "• " + item,
                    normal
                )
            )

        story.append(
            Spacer(1, 12)
        )

    story.append(
        Paragraph(
            "Attendance Conversion: "
            "90–100%=5, 80–89%=4, 70–79%=3, "
            "60–69%=2, 10–59%=1, below 10%=0.",
            normal
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================================
# HOME PAGE
# ==========================================================

def home():

    background()

    st.markdown(
        '<div class="login-wrapper">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-card">',
        unsafe_allow_html=True
    )

    # MAIN HEADING ONLY
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

                <div class="role-icon">🎓</div>

                <div class="role-title">
                    STUDENT
                </div>

                <div class="role-description">
                    Login to view your academic
                    performance and progress report.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🎓 STUDENT LOGIN",
            width="stretch"
        ):

            st.session_state.page = "student_login"

            st.rerun()

    with col2:

        st.markdown(
            """
            <div class="role-card">

                <div class="role-icon">👨‍🏫</div>

                <div class="role-title">
                    TUTOR
                </div>

                <div class="role-description">
                    Login according to your department
                    and manage student performance.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "👨‍🏫 TUTOR LOGIN",
            width="stretch"
        ):

            st.session_state.page = "tutor_login"

            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ==========================================================
# STUDENT LOGIN
# ==========================================================

def student_login():

    background()

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
        "Username",
        placeholder="Enter your username"
    )

    university_id = st.text_input(
        "University ID",
        placeholder="Enter your University ID"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Example: BTECH2007"
    )

    st.caption(
        "Demo password format: BTECH2000 – BTECH2022"
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "🔐 LOGIN",
            type="primary",
            width="stretch"
        ):

            if not username.strip():

                st.error(
                    "Please enter username."
                )

            elif not university_id.strip():

                st.error(
                    "Please enter University ID."
                )

            elif not valid_student_password(
                password
            ):

                st.error(
                    "Invalid password format. "
                    "Use BTECH2000–BTECH2022."
                )

            else:

                df = load_data()

                student = df[
                    (
                        df["Username"]
                        .astype(str)
                        .str.strip()
                        == username.strip()
                    )
                    &
                    (
                        df["University_ID"]
                        .astype(str)
                        .str.strip()
                        == university_id.strip()
                    )
                ]

                if student.empty:

                    st.error(
                        "Student username or University ID "
                        "was not found."
                    )

                else:

                    st.session_state.student_logged_in = True

                    st.session_state.student_username = (
                        username.strip()
                    )

                    st.session_state.student_id = (
                        university_id.strip()
                    )

                    st.session_state.student_name = (
                        str(
                            student.iloc[0][
                                "Student_Name"
                            ]
                        )
                    )

                    st.session_state.page = (
                        "student_dashboard"
                    )

                    st.rerun()

    with c2:

        if st.button(
            "⬅ BACK",
            width="stretch"
        ):

            st.session_state.page = "home"

            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ==========================================================
# TUTOR LOGIN
# ==========================================================

def tutor_login():

    background()

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

    department = st.selectbox(
        "🏫 Department",
        BRANCHES
    )

    username = st.text_input(
        "Tutor Username",
        placeholder="Enter tutor username"
    )

    password = st.text_input(
        "Tutor Password",
        type="password",
        placeholder="Enter tutor password"
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "🔐 TUTOR LOGIN",
            type="primary",
            width="stretch"
        ):

            account = TUTOR_ACCOUNTS[
                department
            ]

            entered_user = username.strip()

            entered_password = password

            if not entered_user:

                st.error(
                    "Please enter tutor username."
                )

            elif not entered_password:

                st.error(
                    "Please enter tutor password."
                )

            elif (
                entered_user
                != account["username"]
            ):

                st.error(
                    "Incorrect username for this department."
                )

            elif (
                entered_password
                != account["password"]
            ):

                st.error(
                    "Incorrect tutor password."
                )

            else:

                st.session_state.tutor_logged_in = True

                st.session_state.tutor_username = (
                    entered_user
                )

                st.session_state.tutor_branch = (
                    department
                )

                st.session_state.page = (
                    "tutor_dashboard"
                )

                st.rerun()

    with c2:

        if st.button(
            "⬅ BACK",
            width="stretch"
        ):

            st.session_state.page = "home"

            st.rerun()

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True
    )


# ==========================================================
# STUDENT DASHBOARD
# ==========================================================

def student_dashboard():

    if not st.session_state.student_logged_in:

        st.session_state.page = "student_login"

        st.rerun()

    df = load_data()

    username = st.session_state.student_username
    university_id = st.session_state.student_id
    student_name = st.session_state.student_name

    student_df = df[
        (
            df["Username"].astype(str).str.strip()
            == username
        )
        &
        (
            df["University_ID"].astype(str).str.strip()
            == university_id
        )
    ].copy()

    st.markdown(
        f"""
        <div class="dashboard-title">
            🎓 {student_name}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚪 Logout"):

        st.session_state.student_logged_in = False
        st.session_state.page = "home"

        st.rerun()

    if student_df.empty:

        st.warning(
            "No performance data found."
        )

        return

    semester_list = list(
        student_df["Semester"]
        .astype(str)
        .unique()
    )

    semester = st.selectbox(
        "📚 Select Semester",
        semester_list
    )

    data = student_df[
        student_df["Semester"].astype(str)
        == semester
    ].copy()

    overall = data[
        "Performance_Percent"
    ].mean()

    st.metric(
        "Overall Performance",
        f"{overall:.2f}%"
    )

    st.subheader(
        "📚 Subject-wise Performance"
    )

    for _, row in data.iterrows():

        with st.expander(
            f"📘 {row['Subject']}"
        ):

            st.write(
                f"**Performance:** "
                f"{row['Performance']}"
            )

            st.write(
                f"**Percentage:** "
                f"{row['Performance_Percent']:.2f}%"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Attendance",
                f"{row['Attendance']:.1f}%"
            )

            c2.metric(
                "Internal",
                f"{row['Internal_Mark']:.1f}/40"
            )

            c3.metric(
                "Assignment",
                f"{row['Assignment']:.1f}/15"
            )

            c1.metric(
                "Previous Mark",
                f"{row['Previous_Mark']:.1f}/60"
            )

            c2.metric(
                "Study Hours",
                f"{row['Study_Hours']:.1f}/6"
            )

            st.write(
                "**Improvement / Guidance:**"
            )

            for item in guidance(row):

                st.write(
                    f"• {item}"
                )

    pdf = make_pdf(
        student_name,
        university_id,
        str(data.iloc[0]["Branch"]),
        semester,
        data
    )

    st.download_button(
        "📄 DOWNLOAD MY PROGRESS REPORT",
        pdf,
        f"{student_name}_{semester}_Report.pdf",
        "application/pdf",
        width="stretch"
    )


# ==========================================================
# TUTOR DASHBOARD
# ==========================================================

def tutor_dashboard():

    if not st.session_state.tutor_logged_in:

        st.session_state.page = "tutor_login"

        st.rerun()

    branch = st.session_state.tutor_branch

    st.markdown(
        """
        <div class="dashboard-title">
            👨‍🏫 TUTOR DASHBOARD
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        f"Logged in department: {branch}"
    )

    if st.button("🚪 Logout"):

        st.session_state.tutor_logged_in = False
        st.session_state.page = "home"

        st.rerun()

    df = load_data()

    branch_df = df[
        df["Branch"].astype(str)
        == branch
    ].copy()

    tabs = st.tabs([
        "➕ Add Student",
        "📊 Student Data",
        "🤖 K-Means",
        "🌲 Random Forest",
        "📄 Progress Reports",
    ])


    # ======================================================
    # ADD STUDENT
    # ======================================================

    with tabs[0]:

        st.subheader(
            "➕ Add Student"
        )

        name = st.text_input(
            "Student Name"
        )

        username = st.text_input(
            "Student Username"
        )

        university_id = st.text_input(
            "University ID"
        )

        semester = st.selectbox(
            "Semester",
            list(SEMESTERS.keys())
        )

        records = []

        st.markdown(
            "### Subject-wise Details"
        )

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
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    key=f"attendance_{subject}"
                )

            with c2:

                internal = st.number_input(
                    "Internal /40",
                    min_value=0.0,
                    max_value=40.0,
                    value=20.0,
                    key=f"internal_{subject}"
                )

            with c3:

                assignment = st.number_input(
                    "Assignment /15",
                    min_value=0.0,
                    max_value=15.0,
                    value=8.0,
                    key=f"assignment_{subject}"
                )

            with c4:

                previous = st.number_input(
                    "Previous /60",
                    min_value=0.0,
                    max_value=60.0,
                    value=30.0,
                    key=f"previous_{subject}"
                )

            with c5:

                study = st.number_input(
                    "Study Hours /6",
                    min_value=0.0,
                    max_value=6.0,
                    value=3.0,
                    step=.5,
                    key=f"study_{subject}"
                )

            percentage, performance = calculate_performance(
                attendance,
                internal,
                assignment,
                previous,
                study
            )

            records.append({
                "Student_Name": name,
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
            })

        if st.button(
            "💾 SAVE STUDENT",
            type="primary",
            width="stretch"
        ):

            if not name.strip():

                st.error(
                    "Enter student name."
                )

            elif not username.strip():

                st.error(
                    "Enter student username."
                )

            elif not university_id.strip():

                st.error(
                    "Enter University ID."
                )

            else:

                new_data = pd.DataFrame(
                    records
                )

                old_data = load_data()

                # Replace the same student semester
                old_data = old_data[
                    ~(
                        (
                            old_data[
                                "University_ID"
                            ].astype(str)
                            == university_id
                        )
                        &
                        (
                            old_data[
                                "Semester"
                            ].astype(str)
                            == semester
                        )
                        &
                        (
                            old_data[
                                "Branch"
                            ].astype(str)
                            == branch
                        )
                    )
                ]

                final_data = pd.concat(
                    [
                        old_data,
                        new_data
                    ],
                    ignore_index=True
                )

                save_data(
                    final_data
                )

                st.success(
                    "Student saved successfully."
                )

                st.rerun()


    # ======================================================
    # STUDENT DATA
    # ======================================================

    with tabs[1]:

        st.subheader(
            "📊 Department Student Data"
        )

        if branch_df.empty:

            st.info(
                "No student records available."
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
                "⬇️ DOWNLOAD DEPARTMENT CSV",
                csv_data,
                "department_data.csv",
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

            choices = [
                f"{row.Student_Name} | "
                f"{row.University_ID}"
                for row in students.itertuples()
            ]

            selected = st.selectbox(
                "Select student",
                choices
            )

            if st.button(
                "🗑️ DELETE STUDENT"
            ):

                selected_id = (
                    selected.split("|")[1].strip()
                )

                all_data = load_data()

                all_data = all_data[
                    ~(
                        (
                            all_data[
                                "Branch"
                            ].astype(str)
                            == branch
                        )
                        &
                        (
                            all_data[
                                "University_ID"
                            ].astype(str)
                            == selected_id
                        )
                    )
                ]

                save_data(
                    all_data
                )

                st.success(
                    "Student history deleted."
                )

                st.rerun()


    # ======================================================
    # K-MEANS
    # ======================================================

    with tabs[2]:

        st.subheader(
            "🤖 K-Means Clustering"
        )

        if len(branch_df) < 3:

            st.warning(
                "At least 3 records are required."
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
                        "Subject",
                        "Performance",
                        "KMeans_Cluster"
                    ]
                ],
                width="stretch",
                hide_index=True
            )

            st.info(
                "K-Means cluster numbers are identifiers, "
                "not official grades."
            )


    # ======================================================
    # RANDOM FOREST
    # ======================================================

    with tabs[3]:

        st.subheader(
            "🌲 Random Forest"
        )

        model, accuracy = train_random_forest(
            branch_df
        )

        if model is None:

            st.warning(
                "At least 10 suitable records are "
                "recommended for model training."
            )

        else:

            st.metric(
                "Random Forest Accuracy",
                f"{accuracy * 100:.2f}%"
            )

            st.success(
                "Random Forest model trained successfully."
            )


    # ======================================================
    # REPORTS
    # ======================================================

    with tabs[4]:

        st.subheader(
            "📄 Progress Reports"
        )

        if branch_df.empty:

            st.info(
                "No students available."
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

            choices = [
                f"{row.Student_Name} | "
                f"{row.University_ID} | "
                f"{row.Semester}"
                for row in students.itertuples()
            ]

            selected = st.selectbox(
                "Select Student",
                choices
            )

            name, university_id, semester = [
                x.strip()
                for x in selected.split("|")
            ]

            report_data = branch_df[
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
                report_data,
                width="stretch",
                hide_index=True
            )

            pdf = make_pdf(
                name,
                university_id,
                branch,
                semester,
                report_data
            )

            st.download_button(
                "📄 DOWNLOAD PROGRESS REPORT",
                pdf,
                f"{name}_{semester}_Progress_Report.pdf",
                "application/pdf",
                width="stretch"
            )


# ==========================================================
# ROUTER
# ==========================================================

load_css()

if st.session_state.page == "home":

    home()

elif st.session_state.page == "student_login":

    student_login()

elif st.session_state.page == "tutor_login":

    tutor_login()

elif st.session_state.page == "student_dashboard":

    student_dashboard()

elif st.session_state.page == "tutor_dashboard":

    tutor_dashboard()
