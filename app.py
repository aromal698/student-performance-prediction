import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# FILE / DATABASE SETTINGS
# ============================================================

DATA_DIR = "data"
DATABASE_FILE = os.path.join(DATA_DIR, "student_records.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# BRANCHES
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
]

# ============================================================
# SUBJECTS
# ============================================================

SUBJECTS = {
    "S1": [
        "Engineering Mathematics I",
        "Engineering Physics",
        "Engineering Chemistry",
        "Engineering Graphics",
        "Programming in C",
        "Life Skills",
    ],

    "S2": [
        "Engineering Mathematics II",
        "Data Structures",
        "Object Oriented Programming",
        "Digital Electronics",
        "Engineering Mechanics",
        "Database Fundamentals",
    ],

    "S3": [
        "Engineering Mathematics III",
        "Data Structures and Algorithms",
        "Database Management Systems",
        "Operating Systems",
        "Computer Organization",
        "Object Oriented Programming",
    ],

    "S4": [
        "Engineering Mathematics IV",
        "Computer Networks",
        "Operating Systems",
        "Design and Analysis of Algorithms",
        "Software Engineering",
        "Database Systems",
    ],

    "S5": [
        "Machine Learning",
        "Cloud Computing",
        "Data Mining",
        "Cyber Security",
        "Internet of Things",
        "Distributed Systems",
    ],

    "S6": [
        "Artificial Intelligence",
        "Machine Learning",
        "Cloud Computing",
        "Computer Security",
        "Data Analytics",
        "Internet of Things",
    ],

    "S7": [
        "Advanced Computing",
        "Artificial Intelligence",
        "Cloud Applications",
        "Cyber Security",
        "Advanced Data Analytics",
        "Project Phase I",
    ],

    "S8": [
        "Major Project",
        "Project Phase II",
        "Professional Elective VII",
        "Professional Elective VIII",
        "Industry Internship",
        "Technical Seminar",
    ],
}

# ============================================================
# TUTOR ACCOUNTS
# ============================================================

TUTORS = {
    "teacher_aids": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Data Science",
    },

    "teacher_cse": {
        "password": "ktutech",
        "branch": "Computer Science and Engineering",
    },

    "teacher_ds": {
        "password": "ktutech",
        "branch": "Computer Science and Engineering (Data Science)",
    },

    "teacher_it": {
        "password": "ktutech",
        "branch": "Information Technology",
    },

    "teacher_ece": {
        "password": "ktutech",
        "branch": "Electronics and Communication Engineering",
    },

    "teacher_eee": {
        "password": "ktutech",
        "branch": "Electrical and Electronics Engineering",
    },

    "teacher_me": {
        "password": "ktutech",
        "branch": "Mechanical Engineering",
    },

    "teacher_ce": {
        "password": "ktutech",
        "branch": "Civil Engineering",
    },
}

# ============================================================
# DATA COLUMNS
# ============================================================

COLUMNS = [
    "Student_Name",
    "Username",
    "University_ID",
    "Semester",
    "Branch",
    "Subject",
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark",
    "Performance_Level",
    "Overall_Percent",
    "KMeans_Cluster",
    "Created_At",
]

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark",
]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 20%,
            rgba(66, 133, 244, 0.25),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(0, 220, 255, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 90%,
            rgba(150, 70, 255, 0.20),
            transparent 32%
        ),
        #050816;
    color: white;
}

/* Moving futuristic grid */

.stApp::before {
    content: "";
    position: fixed;
    inset: -30%;
    z-index: -10;

    background:
        linear-gradient(
            90deg,
            transparent 48%,
            rgba(0, 220, 255, 0.12) 49%,
            transparent 50%
        ),
        linear-gradient(
            0deg,
            transparent 48%,
            rgba(100, 80, 255, 0.12) 49%,
            transparent 50%
        );

    background-size: 100px 100px;

    transform:
        perspective(700px)
        rotateX(58deg)
        scale(1.5);

    animation: gridAnimation 16s linear infinite;
}

@keyframes gridAnimation {

    from {
        background-position: 0 0;
    }

    to {
        background-position: 100px 100px;
    }
}

/* 3D floating orbs */

.orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: -5;
    filter: blur(2px);
}

.orb1 {

    width: 260px;
    height: 260px;

    left: 4%;
    top: 8%;

    background:
        radial-gradient(
            circle at 30% 30%,
            #73e8ff,
            #5046e5 45%,
            transparent 72%
        );

    animation: orbOne 9s ease-in-out infinite;
}

.orb2 {

    width: 320px;
    height: 320px;

    right: 2%;
    top: 40%;

    background:
        radial-gradient(
            circle at 30% 30%,
            #d09cff,
            #6821b8 45%,
            transparent 72%
        );

    animation: orbTwo 12s ease-in-out infinite;
}

.orb3 {

    width: 180px;
    height: 180px;

    left: 42%;
    bottom: 3%;

    background:
        radial-gradient(
            circle at 30% 30%,
            #75f4ff,
            #0786a0 45%,
            transparent 72%
        );

    animation: orbThree 10s ease-in-out infinite;
}

@keyframes orbOne {

    0%, 100% {
        transform: translate3d(0, 0, 0);
    }

    50% {
        transform: translate3d(100px, 50px, 100px);
    }
}

@keyframes orbTwo {

    0%, 100% {
        transform: translate3d(0, 0, 0);
    }

    50% {
        transform: translate3d(-100px, -50px, 80px);
    }
}

@keyframes orbThree {

    0%, 100% {
        transform: translate3d(0, 0, 0);
    }

    50% {
        transform: translate3d(50px, -70px, 100px);
    }
}

/* Login card */

.login-container {

    min-height: 70vh;

    display: flex;

    justify-content: center;

    align-items: center;

    perspective: 1200px;
}

.login-card {

    width: min(850px, 95vw);

    padding: 50px 40px;

    border-radius: 30px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.14),
            rgba(255,255,255,0.035)
        );

    border:
        1px solid
        rgba(255,255,255,0.18);

    box-shadow:
        0 35px 100px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.25);

    backdrop-filter: blur(25px);

    animation: loginFloat 6s ease-in-out infinite;
}

@keyframes loginFloat {

    0%,100% {
        transform:
            translateY(0)
            rotateX(2deg)
            rotateY(-2deg);
    }

    50% {
        transform:
            translateY(-10px)
            rotateX(1deg)
            rotateY(2deg);
    }
}

.login-title {

    text-align: center;

    font-size:
        clamp(30px, 5vw, 52px);

    font-weight: 800;

    letter-spacing: -1px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #76e7ff,
            #b78cff,
            #ffffff
        );

    background-size: 250% auto;

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    animation: titleGlow 5s linear infinite;
}

@keyframes titleGlow {

    from {
        background-position: 0% center;
    }

    to {
        background-position: 250% center;
    }
}

.login-subtitle {

    text-align: center;

    margin-top: 12px;

    color: #b9c5df;

    font-size: 16px;
}

.login-badge {

    width: fit-content;

    margin: 20px auto 0;

    padding: 8px 18px;

    border-radius: 50px;

    background:
        rgba(80, 220, 255, 0.10);

    border:
        1px solid
        rgba(80, 220, 255, 0.25);

    color: #8cecff;

    font-size: 13px;
}

/* Dashboard */

.dashboard-title {

    font-size: 35px;

    font-weight: 800;

    margin-top: 15px;
}

.dashboard-sub {

    color: #aebbd5;

    margin-bottom: 25px;
}

/* Cards */

.info-card {

    padding: 18px;

    border-radius: 18px;

    background:
        rgba(255,255,255,0.06);

    border:
        1px solid
        rgba(255,255,255,0.10);
}

/* Performance circles */

.performance-circle {

    display: inline-flex;

    align-items: center;

    justify-content: center;

    width: 14px;

    height: 14px;

    border-radius: 50%;

    margin-right: 8px;

    box-shadow:
        0 0 8px currentColor;
}

.red {
    background: #ff3333;
    color: #ff3333;
}

.orange {
    background: #ff9800;
    color: #ff9800;
}

.yellow {
    background: #ffd400;
    color: #ffd400;
}

.green {
    background: #25d366;
    color: #25d366;
}

</style>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""",
    unsafe_allow_html=True,
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

if "student_username" not in st.session_state:
    st.session_state.student_username = ""

if "student_id" not in st.session_state:
    st.session_state.student_id = ""

if "tutor_username" not in st.session_state:
    st.session_state.tutor_username = ""

# ============================================================
# DATABASE FUNCTIONS
# ============================================================

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

    else:
        return 0


# ============================================================
# PERFORMANCE CALCULATION
# ============================================================

def calculate_performance(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    attendance_score = (
        attendance_mark(attendance) / 5
    ) * 100

    study_score = (
        min(float(study_hours), 6) / 6
    ) * 100

    internal_score = (
        float(internal) / 40
    ) * 100

    assignment_score = (
        float(assignment) / 15
    ) * 100

    previous_score = (
        float(previous) / 60
    ) * 100

    overall = np.mean([
        attendance_score,
        study_score,
        internal_score,
        assignment_score,
        previous_score
    ])

    if overall < 50:

        return (
            "Low Performance",
            "🔴",
            "red",
            overall
        )

    elif overall < 65:

        return (
            "Average Performance",
            "🟠",
            "orange",
            overall
        )

    elif overall < 80:

        return (
            "Above Average Performance",
            "🟡",
            "yellow",
            overall
        )

    else:

        return (
            "Good Performance",
            "🟢",
            "green",
            overall
        )


# ============================================================
# IMPROVEMENT / APPRECIATION
# ============================================================

def get_advice(
    attendance,
    study,
    internal,
    assignment,
    previous,
    level
):

    advice = []

    if level == "Good Performance":

        return [
            "Excellent performance! Keep up the good work.",
            "Continue regular attendance and consistent study.",
            "Keep submitting assignments on time."
        ]

    if attendance < 80:

        advice.append(
            "Improve attendance by attending classes regularly."
        )

    if study < 3:

        advice.append(
            "Increase daily study hours gradually."
        )

    if internal < 20:

        advice.append(
            "Improve internal marks through regular revision and practice."
        )

    if assignment < 8:

        advice.append(
            "Complete and submit assignments regularly."
        )

    if previous < 30:

        advice.append(
            "Revise previous topics and practise more questions."
        )

    if level == "Low Performance":

        advice.append(
            "Focus first on the weakest areas and prepare a weekly improvement plan."
        )

    elif level == "Average Performance":

        advice.append(
            "Maintain consistency in attendance, study and assignments."
        )

    elif level == "Above Average Performance":

        advice.append(
            "Small improvements in attendance, internal marks and assignments can help reach Good Performance."
        )

    return advice


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    df = df.copy()

    numeric = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark"
    ]

    for col in numeric:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["Attendance"] = df["Attendance"].clip(
        0, 100
    )

    df["Study_Hours"] = df["Study_Hours"].clip(
        0, 6
    )

    df["Internal_Mark"] = df["Internal_Mark"].clip(
        0, 40
    )

    df["Assignment"] = df["Assignment"].clip(
        0, 15
    )

    df["Previous_Mark"] = df["Previous_Mark"].clip(
        0, 60
    )

    return df


# ============================================================
# ADD PERFORMANCE TO DATA
# ============================================================

def add_performance(df):

    if df.empty:
        return df

    df = clean_data(df)

    levels = []
    percentages = []

    for _, row in df.iterrows():

        level, _, _, percentage = calculate_performance(
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"]
        )

        levels.append(level)
        percentages.append(
            round(percentage, 2)
        )

    df["Performance_Level"] = levels
    df["Overall_Percent"] = percentages

    return df


# ============================================================
# MACHINE LEARNING DATA
# ============================================================

def make_ml_features(df):

    return pd.DataFrame({

        "Attendance":
            df["Attendance"],

        "Study_Hours":
            df["Study_Hours"] / 6 * 100,

        "Internal_Mark":
            df["Internal_Mark"] / 40 * 100,

        "Assignment":
            df["Assignment"] / 15 * 100,

        "Previous_Mark":
            df["Previous_Mark"] / 60 * 100
    })


# ============================================================
# TRAIN K-MEANS + RANDOM FOREST
# ============================================================

def train_models(df):

    if len(df) < 6:

        return None

    df = add_performance(df)

    df = df.dropna(
        subset=FEATURES
    )

    if len(df) < 6:

        return None

    X = make_ml_features(df)

    y = df["Performance_Level"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    k = min(
        3,
        len(df)
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        X_scaled
    )

    if len(set(clusters)) > 1:

        silhouette = silhouette_score(
            X_scaled,
            clusters
        )

    else:

        silhouette = 0

    rf = None
    accuracy = None

    if y.nunique() >= 2 and len(df) >= 10:

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

        rf = RandomForestClassifier(
            n_estimators=120,
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

    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "random_forest": rf,
        "accuracy": accuracy,
        "silhouette": silhouette
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_student(models, row):

    temp = pd.DataFrame([row])

    X = make_ml_features(
        temp
    )

    X_scaled = models["scaler"].transform(
        X
    )

    cluster = int(
        models["kmeans"].predict(
            X_scaled
        )[0]
    )

    rf = models["random_forest"]

    if rf is not None:

        prediction = str(
            rf.predict(X)[0]
        )

        probability = float(
            np.max(
                rf.predict_proba(X)
            ) * 100
        )

    else:

        prediction = str(
            row["Performance_Level"]
        )

        probability = 100.0

    return (
        prediction,
        probability,
        cluster
    )


# ============================================================
# PROGRESS REPORT PDF
# ============================================================

def create_progress_report(
    student_name,
    username,
    university_id,
    semester,
    branch,
    df,
    ml_info=None
):

    df = add_performance(
        df.copy()
    )

    # Overall
    overall_percentage = (
        df["Overall_Percent"].mean()
    )

    overall_level, overall_icon, _, _ = calculate_performance(
        df["Attendance"].mean(),
        df["Study_Hours"].mean(),
        df["Internal_Mark"].mean(),
        df["Assignment"].mean(),
        df["Previous_Mark"].mean()
    )

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11
    )

    small_style = ParagraphStyle(
        "SmallStyle",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Powered Academic Performance Monitoring System",
            body_style
        )
    )

    story.append(
        Spacer(1, 8)
    )

    # --------------------------------------------------------
    # STUDENT INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Student Information",
            heading_style
        )
    )

    student_info = Table(
        [
            [
                "Student Name",
                student_name,
                "University ID",
                university_id
            ],

            [
                "Username",
                username,
                "Semester",
                semester
            ],

            [
                "Branch",
                branch,
                "Generated",
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )
            ]
        ],
        colWidths=[
            32 * mm,
            55 * mm,
            32 * mm,
            65 * mm
        ]
    )

    student_info.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#EAF2FF")
            ),

            (
                "BACKGROUND",
                (2, 0),
                (2, -1),
                colors.HexColor("#EAF2FF")
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            )
        ])
    )

    story.append(
        student_info
    )

    story.append(
        Spacer(1, 8)
    )

    # --------------------------------------------------------
    # OVERALL INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Overall Information",
            heading_style
        )
    )

    overall_table = Table(
        [
            [
                "Overall Percentage",
                f"{overall_percentage:.2f}%"
            ],

            [
                "Overall Performance",
                f"{overall_icon} {overall_level}"
            ],

            [
                "Average Attendance",
                f"{df['Attendance'].mean():.1f}%"
            ],

            [
                "Average Internal",
                f"{df['Internal_Mark'].mean():.1f} / 40"
            ],

            [
                "Average Previous Mark",
                f"{df['Previous_Mark'].mean():.1f} / 60"
            ],

            [
                "Average Study Hours",
                f"{df['Study_Hours'].mean():.1f} / 6 hr"
            ],

            [
                "Average Assignment",
                f"{df['Assignment'].mean():.1f} / 15"
            ]
        ],
        colWidths=[
            70 * mm,
            90 * mm
        ]
    )

    overall_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F0F3F8")
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8.5
            )
        ])
    )

    story.append(
        overall_table
    )

    story.append(
        Spacer(1, 8)
    )

    # --------------------------------------------------------
    # SUBJECT-WISE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Subject-wise Performance",
            heading_style
        )
    )

    subject_table = [
        [
            "Subject",
            "Attend.\n/100",
            "Attend.\n/5",
            "Study\n/6",
            "Internal\n/40",
            "Assign.\n/15",
            "Previous\n/60",
            "Performance"
        ]
    ]

    for _, row in df.iterrows():

        level, icon, _, percentage = calculate_performance(
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"]
        )

        subject_table.append(
            [
                str(row["Subject"]),

                f"{row['Attendance']:.0f}%",

                str(
                    attendance_mark(
                        row["Attendance"]
                    )
                ),

                f"{row['Study_Hours']:.1f}",

                f"{row['Internal_Mark']:.0f}",

                f"{row['Assignment']:.0f}",

                f"{row['Previous_Mark']:.0f}",

                f"{icon} {level}\n{percentage:.1f}%"
            ]
        )

    table = Table(
        subject_table,
        repeatRows=1,
        colWidths=[
            42 * mm,
            16 * mm,
            17 * mm,
            16 * mm,
            18 * mm,
            17 * mm,
            18 * mm,
            35 * mm
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#DDEBFF")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.grey
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                6.7
            ),

            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            )
        ])
    )

    story.append(
        table
    )

    story.append(
        Spacer(1, 8)
    )

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Improvement Methods / Appreciation",
            heading_style
        )
    )

    for _, row in df.iterrows():

        level, icon, _, percentage = calculate_performance(
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"]
        )

        story.append(
            Paragraph(
                f"<b>{icon} {row['Subject']}</b> — "
                f"{level} ({percentage:.1f}%)",
                body_style
            )
        )

        advice_list = get_advice(
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"],
            level
        )

        for advice in advice_list:

            story.append(
                Paragraph(
                    "• " + advice,
                    small_style
                )
            )

        story.append(
            Spacer(1, 3)
        )

    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Machine Learning Analysis",
            heading_style
        )
    )

    if ml_info is not None:

        ml_table = Table(
            [
                ["Item", "Result"],

                [
                    "K-Means Cluster",
                    str(
                        ml_info["cluster"]
                    )
                ],

                [
                    "Random Forest Prediction",
                    ml_info["prediction"]
                ],

                [
                    "Prediction Confidence",
                    f"{ml_info['confidence']:.2f}%"
                ],

                [
                    "Random Forest Accuracy",
                    (
                        f"{ml_info['accuracy'] * 100:.2f}%"
                        if ml_info["accuracy"] is not None
                        else "Not enough data"
                    )
                ],

                [
                    "K-Means Silhouette Score",
                    f"{ml_info['silhouette']:.4f}"
                ]
            ],
            colWidths=[
                75 * mm,
                75 * mm
            ]
        )

        ml_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#EEF1F7")
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ])
        )

        story.append(
            ml_table
        )

    # --------------------------------------------------------
    # CONVERSION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. Assessment Conversion",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "Attendance conversion: "
            "90–100% = 5 marks; "
            "80–89% = 4 marks; "
            "70–79% = 3 marks; "
            "60–69% = 2 marks; "
            "10–59% = 1 mark; "
            "below 10% = 0 marks.",
            small_style
        )
    )

    story.append(
        Paragraph(
            "Maximum values used: Attendance = 100%, "
            "Internal Mark = 40, Previous Mark = 60, "
            "Study Hours = 6 hours/day, Assignment = 15.",
            small_style
        )
    )

    story.append(
        Paragraph(
            "Performance categories are project-defined indicators "
            "for academic monitoring and are not official KTU grades. "
            "K-Means cluster numbers are identifiers and do not represent grades.",
            small_style
        )
    )

    document.build(
        story
    )

    return buffer.getvalue()


# ============================================================
# CSV TEMPLATE
# ============================================================

def get_template():

    return pd.DataFrame(
        [
            {
                "Student_Name": "Example Student",
                "Username": "student01",
                "University_ID": "KTU2026001",
                "Semester": "S3",
                "Branch":
                    "Artificial Intelligence and Data Science",
                "Subject":
                    "Database Management Systems",
                "Attendance": 88,
                "Study_Hours": 4,
                "Internal_Mark": 30,
                "Assignment": 12,
                "Previous_Mark": 45,
            }
        ]
    )


# ============================================================
# CSV VALIDATION
# ============================================================

def validate_uploaded_data(df):

    required = [
        "Student_Name",
        "Username",
        "University_ID",
        "Semester",
        "Branch",
        "Subject",
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    numeric_columns = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():

        raise ValueError(
            "Some numeric values are invalid."
        )

    if (df["Attendance"] < 0).any() or (
        df["Attendance"] > 100
    ).any():

        raise ValueError(
            "Attendance must be between 0 and 100."
        )

    if (df["Study_Hours"] < 0).any() or (
        df["Study_Hours"] > 6
    ).any():

        raise ValueError(
            "Study Hours must be between 0 and 6."
        )

    if (df["Internal_Mark"] < 0).any() or (
        df["Internal_Mark"] > 40
    ).any():

        raise ValueError(
            "Internal Mark must be between 0 and 40."
        )

    if (df["Assignment"] < 0).any() or (
        df["Assignment"] > 15
    ).any():

        raise ValueError(
            "Assignment must be between 0 and 15."
        )

    if (df["Previous_Mark"] < 0).any() or (
        df["Previous_Mark"] > 60
    ).any():

        raise ValueError(
            "Previous Mark must be between 0 and 60."
        )

    return add_performance(
        df
    )


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.markdown(
        """
<div class="login-container">

<div class="login-card">

<div class="login-title">
🎓 Student Performance Prediction
</div>

<div class="login-subtitle">
AI-powered academic performance monitoring and
progress reporting system
</div>

<div class="login-badge">
K-Means • Random Forest • Smart Progress Report
</div>

</div>

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        "## Choose Login"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student Login",
            width="stretch"
        ):

            st.session_state.page = "student_login"

            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Tutor Login",
            width="stretch"
        ):

            st.session_state.page = "tutor_login"

            st.rerun()


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.title(
        "🎓 Student Login"
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

    st.caption(
        "Demo password format: BTECH2007"
    )

    if st.button(
        "🔐 Login",
        width="stretch"
    ):

        password_valid = (
            len(password) == 9
            and password.upper().startswith("BTECH")
            and password[5:].isdigit()
        )

        data = load_data()

        student = data[
            (
                data["Username"]
                .astype(str)
                .str.lower()
                ==
                username.strip().lower()
            )
            &
            (
                data["University_ID"]
                .astype(str)
                .str.lower()
                ==
                university_id.strip().lower()
            )
        ]

        if not password_valid:

            st.error(
                "Invalid password format. "
                "Example: BTECH2007"
            )

        elif student.empty:

            st.error(
                "Username or University ID not found."
            )

        else:

            st.session_state.student_logged = True

            st.session_state.student_username = (
                username.strip()
            )

            st.session_state.student_id = (
                university_id.strip()
            )

            st.session_state.page = "student"

            st.rerun()

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.rerun()


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    st.title(
        "👨‍🏫 Tutor Login"
    )

    username = st.text_input(
        "Tutor Username"
    )

    password = st.text_input(
        "Tutor Password",
        type="password"
    )

    if st.button(
        "🔐 Login",
        width="stretch"
    ):

        if (
            username in TUTORS
            and
            password
            ==
            TUTORS[username]["password"]
        ):

            st.session_state.tutor_logged = True

            st.session_state.tutor_username = (
                username
            )

            st.session_state.page = "tutor"

            st.rerun()

        else:

            st.error(
                "Invalid tutor username or password."
            )

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.rerun()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    if not st.session_state.student_logged:

        st.session_state.page = "home"

        st.rerun()

    data = load_data()

    username = (
        st.session_state.student_username
    )

    university_id = (
        st.session_state.student_id
    )

    df = data[
        (
            data["Username"]
            .astype(str)
            .str.lower()
            ==
            username.lower()
        )
        &
        (
            data["University_ID"]
            .astype(str)
            .str.lower()
            ==
            university_id.lower()
        )
    ].copy()

    st.markdown(
        '<div class="dashboard-title">'
        '🎓 Student Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-sub">'
        'View your academic performance and download your progress report.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🚪 Logout"
    ):

        st.session_state.student_logged = False

        st.session_state.page = "home"

        st.rerun()

    if df.empty:

        st.warning(
            "No student records found."
        )

        return

    df = add_performance(
        df
    )

    student_name = str(
        df.iloc[0]["Student_Name"]
    )

    branch = str(
        df.iloc[0]["Branch"]
    )

    semester = str(
        df.iloc[0]["Semester"]
    )

    st.success(
        f"Welcome {student_name}"
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Overall",
        f"{df['Overall_Percent'].mean():.1f}%"
    )

    c2.metric(
        "Attendance",
        f"{df['Attendance'].mean():.1f}%"
    )

    c3.metric(
        "Internal",
        f"{df['Internal_Mark'].mean():.1f}/40"
    )

    c4.metric(
        "Previous",
        f"{df['Previous_Mark'].mean():.1f}/60"
    )

    c5.metric(
        "Assignment",
        f"{df['Assignment'].mean():.1f}/15"
    )

    # --------------------------------------------------------
    # SUBJECT PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "📚 Subject-wise Performance"
    )

    for _, row in df.iterrows():

        level, icon, colour, percentage = (
            calculate_performance(
                row["Attendance"],
                row["Study_Hours"],
                row["Internal_Mark"],
                row["Assignment"],
                row["Previous_Mark"]
            )
        )

        st.markdown(
            f"""
<div class="info-card">

<h4>
{icon} {row["Subject"]}
</h4>

<p>
<b>Attendance:</b> {row["Attendance"]:.0f}% |
<b>Attendance Mark:</b>
{attendance_mark(row["Attendance"])} / 5
</p>

<p>
<b>Study Hours:</b>
{row["Study_Hours"]:.1f} / 6 hr |
<b>Internal:</b>
{row["Internal_Mark"]:.0f} / 40
</p>

<p>
<b>Assignment:</b>
{row["Assignment"]:.0f} / 15 |
<b>Previous:</b>
{row["Previous_Mark"]:.0f} / 60
</p>

<p>
<span class="performance-circle {colour}"></span>
<b>{level}</b> — {percentage:.1f}%
</p>

</div>

<br>
""",
            unsafe_allow_html=True
        )

        advice = get_advice(
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"],
            level
        )

        with st.expander(
            f"💡 {row['Subject']} — Improvement / Appreciation"
        ):

            for item in advice:

                st.write(
                    "• " + item
                )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    models = train_models(
        df
    )

    ml_info = None

    if models:

        prediction, confidence, cluster = (
            predict_student(
                models,
                df.iloc[0].to_dict()
            )
        )

        ml_info = {
            "prediction": prediction,
            "confidence": confidence,
            "cluster": cluster,
            "accuracy": models["accuracy"],
            "silhouette": models["silhouette"]
        }

        st.info(
            f"""
🤖 **Random Forest Prediction:** {prediction}

🎯 **Confidence:** {confidence:.1f}%

🔵 **K-Means Cluster:** {cluster}
"""
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf = create_progress_report(
        student_name,
        username,
        university_id,
        semester,
        branch,
        df,
        ml_info
    )

    st.download_button(
        label="📥 Download My Progress Report",
        data=pdf,
        file_name=f"{university_id}_Progress_Report.pdf",
        mime="application/pdf",
        width="stretch"
    )


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    if not st.session_state.tutor_logged:

        st.session_state.page = "home"

        st.rerun()

    tutor = (
        st.session_state.tutor_username
    )

    assigned_branch = (
        TUTORS[tutor]["branch"]
    )

    st.markdown(
        '<div class="dashboard-title">'
        '👨‍🏫 Tutor Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
<div class="dashboard-sub">
Assigned Branch:
<b>{assigned_branch}</b>
</div>
""",
        unsafe_allow_html=True
    )

    if st.button(
        "🚪 Logout"
    ):

        st.session_state.tutor_logged = False

        st.session_state.page = "home"

        st.rerun()

    # ========================================================
    # CSV TEMPLATE
    # ========================================================

    st.header(
        "1. 📤 Upload Student Details / Marks"
    )

    template = get_template()

    st.download_button(
        label="⬇️ Download CSV Template",
        data=template.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="student_records_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        if st.button(
            "➕ Add Uploaded Student Records"
        ):

            try:

                uploaded_df = pd.read_csv(
                    uploaded_file
                )

                uploaded_df = (
                    validate_uploaded_data(
                        uploaded_df
                    )
                )

                if not (
                    uploaded_df["Branch"]
                    ==
                    assigned_branch
                ).all():

                    st.error(
                        "CSV contains students from another branch."
                    )

                else:

                    old_data = load_data()

                    uploaded_df[
                        "KMeans_Cluster"
                    ] = ""

                    uploaded_df[
                        "Created_At"
                    ] = datetime.now().isoformat(
                        timespec="seconds"
                    )

                    final_data = pd.concat(
                        [
                            old_data,
                            uploaded_df
                        ],
                        ignore_index=True
                    )

                    save_data(
                        final_data
                    )

                    st.success(
                        "Student records added successfully."
                    )

                    st.rerun()

            except Exception as error:

                st.error(
                    f"Upload error: {error}"
                )

    # ========================================================
    # MANUAL ENTRY
    # ========================================================

    st.header(
        "2. ✍️ Add Student Subject Record"
    )

    with st.form(
        "student_record_form"
    ):

        col1, col2 = st.columns(2)

        student_name = col1.text_input(
            "Student Name"
        )

        username = col2.text_input(
            "Username"
        )

        university_id = col1.text_input(
            "University ID"
        )

        semester = col2.selectbox(
            "Semester",
            list(SUBJECTS.keys())
        )

        subject = col1.selectbox(
            "Subject",
            SUBJECTS[semester]
        )

        st.info(
            f"Branch: {assigned_branch}"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        attendance = c1.number_input(
            "Attendance %",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

        study = c2.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=6.0,
            value=3.0,
            step=0.5
        )

        internal = c3.number_input(
            "Internal /40",
            min_value=0.0,
            max_value=40.0,
            value=25.0,
            step=1.0
        )

        assignment = c4.number_input(
            "Assignment /15",
            min_value=0.0,
            max_value=15.0,
            value=10.0,
            step=1.0
        )

        previous = c5.number_input(
            "Previous /60",
            min_value=0.0,
            max_value=60.0,
            value=40.0,
            step=1.0
        )

        submitted = st.form_submit_button(
            "💾 Save Record",
            width="stretch"
        )

    if submitted:

        if (
            not student_name.strip()
            or not username.strip()
            or not university_id.strip()
        ):

            st.error(
                "Student Name, Username and University ID are required."
            )

        else:

            level, icon, colour, percentage = (
                calculate_performance(
                    attendance,
                    study,
                    internal,
                    assignment,
                    previous
                )
            )

            new_record = pd.DataFrame(
                [
                    {
                        "Student_Name":
                            student_name.strip(),

                        "Username":
                            username.strip(),

                        "University_ID":
                            university_id.strip(),

                        "Semester":
                            semester,

                        "Branch":
                            assigned_branch,

                        "Subject":
                            subject,

                        "Attendance":
                            attendance,

                        "Study_Hours":
                            study,

                        "Internal_Mark":
                            internal,

                        "Assignment":
                            assignment,

                        "Previous_Mark":
                            previous,

                        "Performance_Level":
                            level,

                        "Overall_Percent":
                            round(
                                percentage,
                                2
                            ),

                        "KMeans_Cluster":
                            "",

                        "Created_At":
                            datetime.now().isoformat(
                                timespec="seconds"
                            )
                    }
                ]
            )

            old_data = load_data()

            save_data(
                pd.concat(
                    [
                        old_data,
                        new_record
                    ],
                    ignore_index=True
                )
            )

            st.success(
                "Student subject record saved successfully."
            )

    # ========================================================
    # BRANCH DATA
    # ========================================================

    data = load_data()

    branch_data = data[
        data["Branch"].astype(str)
        ==
        assigned_branch
    ].copy()

    if branch_data.empty:

        st.warning(
            "No student data available for this branch."
        )

        return

    branch_data = add_performance(
        branch_data
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    st.header(
        "3. 📊 Overall Student Data Analysis"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Total Students",
        branch_data[
            "University_ID"
        ].nunique()
    )

    m2.metric(
        "Total Records",
        len(branch_data)
    )

    m3.metric(
        "Average Attendance",
        f"{branch_data['Attendance'].mean():.1f}%"
    )

    m4.metric(
        "Average Performance",
        f"{branch_data['Overall_Percent'].mean():.1f}%"
    )

    st.dataframe(
        branch_data[
            [
                "Student_Name",
                "Username",
                "University_ID",
                "Semester",
                "Subject",
                "Attendance",
                "Study_Hours",
                "Internal_Mark",
                "Assignment",
                "Previous_Mark",
                "Performance_Level",
                "Overall_Percent"
            ]
        ],
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # K-MEANS + RANDOM FOREST
    # ========================================================

    st.subheader(
        "🤖 K-Means + Random Forest Analysis"
    )

    models = train_models(
        branch_data
    )

    ml_info = None

    if models:

        cluster_labels = (
            models["kmeans"].labels_
        )

        cluster_summary = (
            pd.DataFrame(
                {
                    "Cluster":
                        cluster_labels
                }
            )
            .value_counts()
            .reset_index(
                name="Records"
            )
        )

        st.write(
            "### K-Means Cluster Summary"
        )

        st.dataframe(
            cluster_summary,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "K-Means cluster numbers are identifiers only; "
            "they are not grade levels."
        )

        if models["accuracy"] is not None:

            st.success(
                "Random Forest Evaluation Accuracy: "
                f"{models['accuracy'] * 100:.2f}%"
            )

        else:

            st.info(
                "More labelled records are required for Random Forest evaluation accuracy."
            )

        st.write(
            "K-Means Silhouette Score: "
            f"**{models['silhouette']:.4f}**"
        )

    # ========================================================
    # PROGRESS REPORT DOWNLOAD
    # ========================================================

    st.header(
        "4. 📄 Download Progress Report for Students"
    )

    student_ids = sorted(
        branch_data[
            "University_ID"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    selected_id = st.selectbox(
        "Select Student",
        student_ids
    )

    selected_student = branch_data[
        branch_data[
            "University_ID"
        ].astype(str)
        ==
        str(selected_id)
    ].copy()

    if not selected_student.empty:

        selected_name = str(
            selected_student.iloc[0][
                "Student_Name"
            ]
        )

        selected_username = str(
            selected_student.iloc[0][
                "Username"
            ]
        )

        selected_semester = str(
            selected_student.iloc[0][
                "Semester"
            ]
        )

        ml_info = None

        if models:

            prediction, confidence, cluster = (
                predict_student(
                    models,
                    selected_student.iloc[0].to_dict()
                )
            )

            ml_info = {
                "prediction":
                    prediction,

                "confidence":
                    confidence,

                "cluster":
                    cluster,

                "accuracy":
                    models["accuracy"],

                "silhouette":
                    models["silhouette"]
            }

        pdf = create_progress_report(
            selected_name,
            selected_username,
            selected_id,
            selected_semester,
            assigned_branch,
            selected_student,
            ml_info
        )

        st.download_button(
            label="📥 Download Student Progress Report PDF",
            data=pdf,
            file_name=f"{selected_id}_Progress_Report.pdf",
            mime="application/pdf",
            width="stretch"
        )

    # ========================================================
    # DELETE STUDENT
    # ========================================================

    st.header(
        "5. 🗑️ Delete Student Data"
    )

    st.warning(
        "Deleting a student removes all subject records "
        "for that University ID from this branch."
    )

    delete_id = st.selectbox(
        "Select Student to Delete",
        student_ids,
        key="delete_student"
    )

    if st.button(
        "🗑️ Delete Selected Student",
        type="primary",
        width="stretch"
    ):

        current_data = load_data()

        mask = (
            (
                current_data["Branch"]
                .astype(str)
                ==
                assigned_branch
            )
            &
            (
                current_data["University_ID"]
                .astype(str)
                ==
                str(delete_id)
            )
        )

        deleted_count = int(
            mask.sum()
        )

        current_data = current_data[
            ~mask
        ]

        save_data(
            current_data
        )

        st.success(
            f"Deleted {deleted_count} record(s) "
            f"for {delete_id}."
        )

        st.rerun()

    # ========================================================
    # DOWNLOAD BRANCH DATA
    # ========================================================

    st.header(
        "6. 📥 Download Branch Data"
    )

    st.download_button(
        label="📥 Download Branch Student CSV",
        data=branch_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            assigned_branch
            .replace(" ", "_")
            + "_Student_Data.csv"
        ),
        mime="text/csv",
        width="stretch"
    )


# ============================================================
# MAIN ROUTER
# ============================================================

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "student_login":

    student_login()

elif st.session_state.page == "tutor_login":

    tutor_login()

elif st.session_state.page == "student":

    student_dashboard()

elif st.session_state.page == "tutor":

    tutor_dashboard()

else:

    st.session_state.page = "home"

    st.rerun()
