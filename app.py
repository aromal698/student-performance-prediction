# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# B.Tech S3 - AI & Data Science
# K-Means Clustering + Random Forest
# Streamlit Application
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import io

from datetime import datetime

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
    TableStyle
)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 3. FILE / DATA CONFIGURATION
# ============================================================

DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

RECORD_FILE = os.path.join(
    DATA_DIR,
    "student_records.csv"
)


# ============================================================
# 4. BRANCH LIST
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
# 5. SUBJECT LIST
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
    },

    "teacher_robotics": {
        "password": "ktutech",
        "branch": "Robotics and Automation"
    },

    "teacher_mechatronics": {
        "password": "ktutech",
        "branch": "Mechatronics Engineering"
    }
}


# ============================================================
# 7. REQUIRED DATA COLUMNS
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

if "login_mode" not in st.session_state:
    st.session_state.login_mode = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "username" not in st.session_state:
    st.session_state.username = None

if "branch" not in st.session_state:
    st.session_state.branch = None

if "celebrated" not in st.session_state:
    st.session_state.celebrated = False


# ============================================================
# 9. LOAD DATABASE
# ============================================================

def load_database():

    if os.path.exists(RECORD_FILE):

        try:
            df = pd.read_csv(RECORD_FILE)

            for col in REQUIRED_COLUMNS:

                if col not in df.columns:
                    df[col] = ""

            return df[REQUIRED_COLUMNS]

        except Exception:

            return pd.DataFrame(
                columns=REQUIRED_COLUMNS
            )

    return pd.DataFrame(
        columns=REQUIRED_COLUMNS
    )


def save_database(df):

    df.to_csv(
        RECORD_FILE,
        index=False
    )


# ============================================================
# 10. PERFORMANCE CALCULATION
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

    attendance_score = (
        attendance_mark(attendance) / 5
    ) * 100

    study_score = (
        study_hours / 6
    ) * 100

    internal_score = (
        internal_mark / 40
    ) * 100

    assignment_score = (
        assignment / 15
    ) * 100

    previous_score = (
        previous_mark / 60
    ) * 100

    overall = (
        attendance_score +
        study_score +
        internal_score +
        assignment_score +
        previous_score
    ) / 5

    return round(overall, 2)


# ============================================================
# 11. PERFORMANCE LEVEL
# ============================================================

def performance_from_overall(overall):

    overall = float(overall)

    if overall < 50:
        return "Low Performance"

    elif overall < 65:
        return "Average Performance"

    elif overall < 80:
        return "Above Average"

    return "Good Performance"


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
# 12. IMPROVEMENT METHODS
# ============================================================

def improvement_methods(row):

    methods = []

    attendance = float(row["Attendance"])
    study = float(row["Study_Hours"])
    internal = float(row["Internal_Mark"])
    assignment = float(row["Assignment"])
    previous = float(row["Previous_Mark"])

    if attendance < 80:
        methods.append(
            "Improve regular class attendance."
        )

    if internal < 26:
        methods.append(
            "Revise class notes and prepare better for internal examinations."
        )

    if assignment < 10:
        methods.append(
            "Complete assignments on time and improve assignment quality."
        )

    if previous < 40:
        methods.append(
            "Revise previous topics and practise more questions."
        )

    if study < 3:
        methods.append(
            "Increase regular study and revision time."
        )

    if not methods:

        methods.append(
            "Excellent performance! Keep up the good work."
        )

    return methods


# ============================================================
# 13. DATA CLEANING
# ============================================================

def clean_input_dataframe(df):

    df = df.copy()

    for col in INPUT_COLUMNS:

        if col not in df.columns:
            df[col] = ""

    df = df[INPUT_COLUMNS]

    text_columns = [
        "Username",
        "University_ID",
        "Student_Name",
        "Branch",
        "Semester",
        "Subject"
    ]

    for col in text_columns:

        df[col] = (
            df[col]
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

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    df["Attendance"] = df["Attendance"].clip(0, 100)
    df["Study_Hours"] = df["Study_Hours"].clip(0, 6)
    df["Internal_Mark"] = df["Internal_Mark"].clip(0, 40)
    df["Assignment"] = df["Assignment"].clip(0, 15)
    df["Previous_Mark"] = df["Previous_Mark"].clip(0, 60)

    return df


# ============================================================
# 14. ADD CALCULATED PERFORMANCE
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

        performance = performance_from_overall(
            overall
        )

        overall_values.append(overall)
        performance_values.append(performance)

    df["Overall_Percentage"] = overall_values

    df["Performance"] = performance_values

    return df


# ============================================================
# 15. K-MEANS CLUSTERING
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

    X = df[features].astype(float)

    if len(df) == 1:

        df["Cluster"] = 0
        return df

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    n_clusters = min(
        3,
        len(df)
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(
        X_scaled
    )

    return df


# ============================================================
# 16. RANDOM FOREST
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

    X = df[features].astype(float)

    y = df["Performance"].astype(str)

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    confidence = probabilities.max(
        axis=1
    ) * 100

    df["RF_Prediction"] = predictions

    df["RF_Confidence"] = np.round(
        confidence,
        2
    )

    return df


# ============================================================
# 17. PREPARE COMPLETE DATASET
# ============================================================

def prepare_dataset(df):

    if len(df) == 0:
        return df

    df = clean_input_dataframe(df)

    df = add_performance_columns(df)

    df = run_kmeans(df)

    df = run_random_forest(df)

    return df


# ============================================================
# 18. PDF PROGRESS REPORT
# ============================================================

def generate_progress_report(
    student_df,
    student_name,
    university_id,
    branch
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13
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
            "B.Tech – AI & Data Science",
            normal_style
        )
    )

    story.append(Spacer(1, 12))

    # --------------------------------------------------------
    # STUDENT DETAILS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Student Details",
            heading_style
        )
    )

    details = [

        ["Student Name", student_name],

        ["University ID", university_id],

        ["Branch", branch],

        [
            "Semester",
            ", ".join(
                sorted(
                    student_df["Semester"]
                    .astype(str)
                    .unique()
                )
            )
        ],

        [
            "Report Date",
            datetime.now().strftime(
                "%d-%m-%Y"
            )
        ]
    ]

    detail_table = Table(
        details,
        colWidths=[140, 350]
    )

    detail_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(detail_table)

    story.append(Spacer(1, 15))

    # --------------------------------------------------------
    # OVERALL PERFORMANCE
    # --------------------------------------------------------

    overall = student_df[
        "Overall_Percentage"
    ].astype(float).mean()

    overall_level = performance_from_overall(
        overall
    )

    info = performance_info(
        overall_level
    )

    story.append(
        Paragraph(
            "2. Overall Performance",
            heading_style
        )
    )

    overall_table = Table(
        [
            ["Overall Percentage", f"{overall:.2f}%"],
            ["Performance", overall_level],
            ["Status", info["message"]]
        ],
        colWidths=[180, 310]
    )

    overall_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(overall_table)

    story.append(Spacer(1, 15))

    # --------------------------------------------------------
    # SUBJECT-WISE REPORT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Subject-wise Performance",
            heading_style
        )
    )

    for subject, group in student_df.groupby(
        "Subject"
    ):

        row = group.iloc[0]

        subject_overall = float(
            group["Overall_Percentage"].mean()
        )

        subject_level = performance_from_overall(
            subject_overall
        )

        subject_info = performance_info(
            subject_level
        )

        story.append(
            Paragraph(
                f"<b>{subject}</b>",
                normal_style
            )
        )

        subject_data = [

            [
                "Attendance",
                f"{float(row['Attendance']):.1f}%"
            ],

            [
                "Attendance Mark",
                f"{attendance_mark(row['Attendance'])}/5"
            ],

            [
                "Study Hours",
                f"{float(row['Study_Hours']):.1f}/6"
            ],

            [
                "Internal Mark",
                f"{float(row['Internal_Mark']):.1f}/40"
            ],

            [
                "Assignment",
                f"{float(row['Assignment']):.1f}/15"
            ],

            [
                "Previous Mark",
                f"{float(row['Previous_Mark']):.1f}/60"
            ],

            [
                "Overall",
                f"{subject_overall:.2f}%"
            ],

            [
                "Performance",
                subject_info["emoji"] +
                " " +
                subject_level
            ]
        ]

        subject_table = Table(
            subject_data,
            colWidths=[180, 310]
        )

        subject_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.whitesmoke
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                ("PADDING", (0, 0), (-1, -1), 5)
            ])
        )

        story.append(subject_table)

        story.append(
            Spacer(1, 5)
        )

        # Improvement methods

        methods = improvement_methods(
            row
        )

        story.append(
            Paragraph(
                "<b>Improvement / Feedback:</b>",
                normal_style
            )
        )

        for method in methods:

            story.append(
                Paragraph(
                    "• " + method,
                    normal_style
                )
            )

        story.append(
            Spacer(1, 10)
        )

    # --------------------------------------------------------
    # K-MEANS INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. K-Means Clustering",
            heading_style
        )
    )

    clusters = (
        student_df["Cluster"]
        .dropna()
        .astype(str)
        .unique()
    )

    cluster_text = ", ".join(
        clusters
    )

    story.append(
        Paragraph(
            f"Student Cluster: {cluster_text}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            "K-Means cluster numbers are machine-learning group identifiers. "
            "They are not official grades or academic ranks.",
            normal_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # --------------------------------------------------------
    # ASSESSMENT CONVERSION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Assessment Conversion",
            heading_style
        )
    )

    conversion = [

        ["Component", "Maximum"],

        ["Attendance", "100%"],

        ["Attendance Converted Mark", "5"],

        ["Study Hours", "6 hours/day"],

        ["Internal Mark", "40"],

        ["Assignment", "15"],

        ["Previous Mark", "60"]
    ]

    conversion_table = Table(
        conversion,
        colWidths=[260, 230]
    )

    conversion_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
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
            ("PADDING", (0, 0), (-1, -1), 5)
        ])
    )

    story.append(
        conversion_table
    )

    story.append(
        Spacer(1, 12)
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Note:</b> The performance categories used in this project "
            "are project-defined thresholds for demonstration and are not "
            "official KTU grading rules. Machine-learning predictions are "
            "intended to support academic monitoring and should not replace "
            "teacher evaluation.",
            normal_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# 19. CSV TEMPLATE
# ============================================================

def create_template():

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
# 20. DYNAMIC 3D LOGIN BACKGROUND
# ============================================================

def login_background():

    st.markdown(
        """
        <style>

        /* ====================================================
           MAIN APP BACKGROUND
        ==================================================== */

        .stApp {
            background:
                radial-gradient(
                    circle at 20% 20%,
                    rgba(0, 229, 255, 0.12),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 80% 30%,
                    rgba(124, 58, 237, 0.15),
                    transparent 35%
                ),
                radial-gradient(
                    circle at 50% 90%,
                    rgba(236, 72, 153, 0.10),
                    transparent 30%
                ),
                #050816;

            color: white;
        }


        /* ====================================================
           REMOVE NORMAL STREAMLIT TOP SPACE
        ==================================================== */

        .block-container {
            padding-top: 3rem;
            position: relative;
            z-index: 5;
        }


        /* ====================================================
           3D SCENE
        ==================================================== */

        .scene {

            position: fixed;

            top: 0;
            left: 0;

            width: 100vw;
            height: 100vh;

            overflow: hidden;

            pointer-events: none;

            z-index: 0;
        }


        /* ====================================================
           GLOWING ORBS
        ==================================================== */

        .orb {

            position: absolute;

            border-radius: 50%;

            filter: blur(2px);

            opacity: 0.65;

            animation:
                floatOrb 9s ease-in-out infinite;
        }


        .orb-one {

            width: 230px;
            height: 230px;

            top: 5%;
            left: 8%;

            background:
                radial-gradient(
                    circle,
                    rgba(0, 229, 255, 0.40),
                    transparent 70%
                );
        }


        .orb-two {

            width: 300px;
            height: 300px;

            right: 5%;
            top: 18%;

            background:
                radial-gradient(
                    circle,
                    rgba(124, 58, 237, 0.40),
                    transparent 70%
                );

            animation-delay: 2s;
        }


        .orb-three {

            width: 260px;
            height: 260px;

            bottom: 5%;
            left: 35%;

            background:
                radial-gradient(
                    circle,
                    rgba(236, 72, 153, 0.30),
                    transparent 70%
                );

            animation-delay: 4s;
        }


        /* ====================================================
           3D CUBE
        ==================================================== */

        .cube {

            position: absolute;

            width: 90px;
            height: 90px;

            transform-style: preserve-3d;

            animation:
                cubeRotate 16s linear infinite;

            opacity: 0.35;
        }


        .cube span {

            position: absolute;

            width: 90px;
            height: 90px;

            border:
                1px solid
                rgba(0, 229, 255, 0.65);

            background:
                rgba(0, 229, 255, 0.035);

            box-shadow:
                0 0 25px
                rgba(0, 229, 255, 0.10);
        }


        .cube span:nth-child(1) {
            transform: translateZ(45px);
        }

        .cube span:nth-child(2) {
            transform:
                rotateY(180deg)
                translateZ(45px);
        }

        .cube span:nth-child(3) {
            transform:
                rotateY(90deg)
                translateZ(45px);
        }

        .cube span:nth-child(4) {
            transform:
                rotateY(-90deg)
                translateZ(45px);
        }

        .cube span:nth-child(5) {
            transform:
                rotateX(90deg)
                translateZ(45px);
        }

        .cube span:nth-child(6) {
            transform:
                rotateX(-90deg)
                translateZ(45px);
        }


        .cube-one {

            left: 7%;
            top: 20%;

            transform: scale(1.2);

        }


        .cube-two {

            right: 12%;
            top: 55%;

            transform: scale(0.7);

            animation-duration: 12s;
            animation-direction: reverse;
        }


        .cube-three {

            left: 70%;
            top: 8%;

            transform: scale(0.45);

            animation-duration: 20s;
        }


        .cube-four {

            left: 18%;
            bottom: 8%;

            transform: scale(0.55);

            animation-duration: 18s;
        }


        /* ====================================================
           MOVING GRID
        ==================================================== */

        .grid {

            position: absolute;

            left: -20%;
            bottom: -35%;

            width: 140%;
            height: 75%;

            transform:
                perspective(500px)
                rotateX(60deg);

            background-image:

                linear-gradient(
                    rgba(0, 229, 255, 0.13) 1px,
                    transparent 1px
                ),

                linear-gradient(
                    90deg,
                    rgba(0, 229, 255, 0.13) 1px,
                    transparent 1px
                );

            background-size:
                55px 55px;

            animation:
                gridMove 5s linear infinite;

            opacity: 0.30;
        }


        /* ====================================================
           FLOATING PARTICLES
        ==================================================== */

        .particles {

            position: absolute;

            width: 100%;
            height: 100%;
        }


        .particle {

            position: absolute;

            width: 4px;
            height: 4px;

            border-radius: 50%;

            background: #ffffff;

            box-shadow:
                0 0 12px
                rgba(0, 229, 255, 0.9);

            animation:
                particleFloat
                var(--duration)
                linear infinite;

            left: var(--left);
            top: var(--top);
        }


        /* ====================================================
           ANIMATIONS
        ==================================================== */

        @keyframes cubeRotate {

            from {
                transform:
                    rotateX(0deg)
                    rotateY(0deg)
                    rotateZ(0deg);
            }

            to {
                transform:
                    rotateX(360deg)
                    rotateY(360deg)
                    rotateZ(360deg);
            }
        }


        @keyframes floatOrb {

            0%, 100% {
                transform:
                    translate3d(0, 0, 0)
                    scale(1);
            }

            50% {
                transform:
                    translate3d(30px, -35px, 0)
                    scale(1.12);
            }
        }


        @keyframes gridMove {

            from {
                background-position:
                    0 0,
                    0 0;
            }

            to {
                background-position:
                    0 55px,
                    55px 0;
            }
        }


        @keyframes particleFloat {

            0% {
                transform:
                    translateY(100vh)
                    scale(0.4);

                opacity: 0;
            }

            15% {
                opacity: 0.8;
            }

            85% {
                opacity: 0.8;
            }

            100% {
                transform:
                    translateY(-20vh)
                    scale(1.2);

                opacity: 0;
            }
        }


        /* ====================================================
           LOGIN CARD
        ==================================================== */

        .login-card {

            max-width: 760px;

            margin:
                40px auto 20px auto;

            padding:
                45px 45px 35px 45px;

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

                0 30px 80px
                rgba(0,0,0,0.45),

                inset 0 0 40px
                rgba(255,255,255,0.025);

            backdrop-filter:
                blur(18px);

            -webkit-backdrop-filter:
                blur(18px);

            text-align: center;
        }


        /* ====================================================
           LOGIN TITLE
        ==================================================== */

        .login-title {

            font-size: 42px;

            font-weight: 800;

            letter-spacing: -1px;

            background:
                linear-gradient(
                    90deg,
                    #ffffff,
                    #67e8f9,
                    #c4b5fd,
                    #ffffff
                );

            background-size: 300% 100%;

            -webkit-background-clip: text;

            -webkit-text-fill-color:
                transparent;

            animation:
                titleGlow 6s ease infinite;

            margin-bottom: 12px;
        }


        .login-subtitle {

            color:
                rgba(255,255,255,0.72);

            font-size: 17px;

            margin-bottom: 30px;
        }


        @keyframes titleGlow {

            0%, 100% {
                background-position: 0% 50%;
            }

            50% {
                background-position: 100% 50%;
            }
        }


        /* ====================================================
           LOGIN ICON
        ==================================================== */

        .login-icon {

            font-size: 70px;

            margin-bottom: 5px;

            filter:
                drop-shadow(
                    0 0 20px
                    rgba(0,229,255,0.45)
                );

            animation:
                iconFloat 3s ease-in-out infinite;
        }


        @keyframes iconFloat {

            0%, 100% {
                transform:
                    translateY(0);
            }

            50% {
                transform:
                    translateY(-10px);
            }
        }


        /* ====================================================
           ROLE CARDS
        ==================================================== */

        .role-box {

            padding: 20px;

            border-radius: 20px;

            background:
                rgba(255,255,255,0.05);

            border:
                1px solid
                rgba(255,255,255,0.12);

            margin-top: 15px;

            transition:
                transform 0.3s ease;
        }


        .role-box:hover {

            transform:
                translateY(-5px);

            background:
                rgba(255,255,255,0.08);
        }


        /* ====================================================
           STREAMLIT BUTTON STYLE
        ==================================================== */

        div.stButton > button {

            border-radius: 14px;

            min-height: 48px;

            font-size: 16px;

            font-weight: 700;

            border:
                1px solid
                rgba(255,255,255,0.18);

            background:
                rgba(255,255,255,0.08);

            color: white;

            transition:
                all 0.25s ease;
        }


        div.stButton > button:hover {

            transform:
                translateY(-3px);

            border-color:
                rgba(103,232,249,0.7);

            box-shadow:
                0 10px 30px
                rgba(0,229,255,0.15);
        }


        /* ====================================================
           MOBILE RESPONSIVE
        ==================================================== */

        @media (max-width: 700px) {

            .login-card {

                margin:
                    20px 10px;

                padding:
                    30px 20px;
            }

            .login-title {

                font-size: 28px;
            }

            .login-subtitle {

                font-size: 14px;
            }

            .cube {

                opacity: 0.15;
            }

        }

        </style>


        <!-- ==================================================
             3D BACKGROUND SCENE
        ================================================== -->

        <div class="scene">

            <div class="orb orb-one"></div>

            <div class="orb orb-two"></div>

            <div class="orb orb-three"></div>


            <!-- 3D CUBE 1 -->

            <div class="cube cube-one">

                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>

            </div>


            <!-- 3D CUBE 2 -->

            <div class="cube cube-two">

                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>

            </div>


            <!-- 3D CUBE 3 -->

            <div class="cube cube-three">

                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>

            </div>


            <!-- 3D CUBE 4 -->

            <div class="cube cube-four">

                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>

            </div>


            <!-- FLOOR GRID -->

            <div class="grid"></div>


            <!-- PARTICLES -->

            <div class="particles">

                <div class="particle"
                     style="--left:5%;--top:90%;--duration:12s;">
                </div>

                <div class="particle"
                     style="--left:12%;--top:80%;--duration:15s;">
                </div>

                <div class="particle"
                     style="--left:20%;--top:70%;--duration:10s;">
                </div>

                <div class="particle"
                     style="--left:28%;--top:95%;--duration:14s;">
                </div>

                <div class="particle"
                     style="--left:35%;--top:85%;--duration:11s;">
                </div>

                <div class="particle"
                     style="--left:43%;--top:92%;--duration:16s;">
                </div>

                <div class="particle"
                     style="--left:52%;--top:75%;--duration:13s;">
                </div>

                <div class="particle"
                     style="--left:61%;--top:88%;--duration:17s;">
                </div>

                <div class="particle"
                     style="--left:70%;--top:80%;--duration:12s;">
                </div>

                <div class="particle"
                     style="--left:78%;--top:93%;--duration:15s;">
                </div>

                <div class="particle"
                     style="--left:87%;--top:72%;--duration:11s;">
                </div>

                <div class="particle"
                     style="--left:94%;--top:85%;--duration:14s;">
                </div>

            </div>

        </div>

        """,
        unsafe_allow_html=True
    )


# ============================================================
# 21. LOGIN PAGE
# ============================================================

def login_page():

    login_background()

    st.markdown(
        """
        <div class="login-card">

            <div class="login-icon">
                🎓
            </div>

            <div class="login-title">
                Student Performance Prediction
            </div>

            <div class="login-subtitle">
                AI-powered academic performance monitoring system
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ROLE SELECTION
    # --------------------------------------------------------

    if st.session_state.login_mode is None:

        st.markdown(
            """
            <div style="
                text-align:center;
                color:rgba(255,255,255,0.8);
                font-size:18px;
                margin-top:25px;
                margin-bottom:15px;
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

        with col1:

            st.markdown(
                """
                <div class="role-box">
                    <h2 style="text-align:center;">
                        🎓 Student
                    </h2>

                    <p style="text-align:center;
                              color:rgba(255,255,255,0.65);">
                        View your academic performance
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "🎓 Student Login",
                width="stretch"
            ):

                st.session_state.login_mode = "student"

                st.rerun()


        with col2:

            st.markdown(
                """
                <div class="role-box">
                    <h2 style="text-align:center;">
                        👨‍🏫 Tutor
                    </h2>

                    <p style="text-align:center;
                              color:rgba(255,255,255,0.65);">
                        Manage and analyze student data
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "👨‍🏫 Tutor Login",
                width="stretch"
            ):

                st.session_state.login_mode = "tutor"

                st.rerun()


    # --------------------------------------------------------
    # STUDENT LOGIN
    # --------------------------------------------------------

    elif st.session_state.login_mode == "student":

        st.markdown(
            "## 🎓 Student Login"
        )

        with st.form(
            "student_login_form"
        ):

            username = st.text_input(
                "Username"
            )

            university_id = st.text_input(
                "University ID"
            )

            password = st.text_input(
                "Password",
                type="password",
                help="Example: BTECH2007"
            )

            submitted = st.form_submit_button(
                "🔐 Login",
                width="stretch"
            )

        if submitted:

            if not username or not university_id:

                st.error(
                    "Please enter Username and University ID."
                )

            elif not valid_student_password(password):

                st.error(
                    "Invalid password format. Example: BTECH2007"
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
                    st.session_state.user_role = "student"
                    st.session_state.username = username.strip()
                    st.session_state.university_id = university_id.strip()

                    st.rerun()


        if st.button(
            "⬅️ Back"
        ):

            st.session_state.login_mode = None

            st.rerun()


    # --------------------------------------------------------
    # TUTOR LOGIN
    # --------------------------------------------------------

    elif st.session_state.login_mode == "tutor":

        st.markdown(
            "## 👨‍🏫 Tutor Login"
        )

        with st.form(
            "tutor_login_form"
        ):

            username = st.text_input(
                "Tutor Username"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            submitted = st.form_submit_button(
                "🔐 Tutor Login",
                width="stretch"
            )

        if submitted:

            if username in TEACHERS:

                teacher = TEACHERS[
                    username
                ]

                if password == teacher["password"]:

                    st.session_state.logged_in = True
                    st.session_state.user_role = "tutor"
                    st.session_state.username = username
                    st.session_state.branch = teacher["branch"]

                    st.rerun()

                else:

                    st.error(
                        "Incorrect password."
                    )

            else:

                st.error(
                    "Tutor account not found."
                )


        if st.button(
            "⬅️ Back"
        ):

            st.session_state.login_mode = None

            st.rerun()


# ============================================================
# 22. STUDENT PASSWORD VALIDATION
# ============================================================

def valid_student_password(password):

    if len(password) != 9:
        return False

    if not password.startswith("BTECH"):
        return False

    year = password[5:]

    if not year.isdigit():
        return False

    return 2000 <= int(year) <= 2020


# ============================================================
# 23. LOGOUT FUNCTION
# ============================================================

def logout():

    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.session_state.branch = None
    st.session_state.login_mode = None
    st.session_state.celebrated = False

    st.rerun()


# ============================================================
# 24. TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    df = load_database()

    teacher_username = st.session_state.username

    teacher_branch = st.session_state.branch


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "👨‍🏫 Tutor Dashboard"
    )

    st.caption(
        f"Logged in as: {teacher_username}"
    )

    st.info(
        f"Assigned Branch: **{teacher_branch}**"
    )


    if st.button(
        "🚪 Logout"
    ):

        logout()


    # ========================================================
    # UPLOAD SECTION
    # ========================================================

    st.header(
        "📤 Upload Student Data"
    )

    st.download_button(
        "⬇️ Download CSV Template",
        data=create_template(),
        file_name="student_template.csv",
        mime="text/csv",
        width="stretch"
    )

    uploaded_file = st.file_uploader(
        "Upload student CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            missing = [
                col
                for col in INPUT_COLUMNS
                if col not in uploaded_df.columns
            ]

            if missing:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing)
                )

            else:

                st.success(
                    f"{len(uploaded_df)} records loaded."
                )

                if st.button(
                    "✅ Submit Student Records",
                    width="stretch"
                ):

                    new_df = clean_input_dataframe(
                        uploaded_df
                    )

                    # Tutor's branch is authoritative
                    new_df["Branch"] = teacher_branch

                    new_df["Submitted_By"] = (
                        teacher_username
                    )

                    new_df["Submitted_Date"] = (
                        datetime.now()
                        .strftime("%Y-%m-%d %H:%M:%S")
                    )

                    new_df = add_performance_columns(
                        new_df
                    )

                    # Add empty model columns
                    new_df["Cluster"] = 0
                    new_df["RF_Prediction"] = ""
                    new_df["RF_Confidence"] = 0.0

                    new_df = new_df[
                        REQUIRED_COLUMNS
                    ]

                    combined = pd.concat(
                        [
                            df,
                            new_df
                        ],
                        ignore_index=True
                    )

                    combined = prepare_dataset(
                        combined
                    )

                    save_database(
                        combined
                    )

                    st.success(
                        "Student records submitted successfully."
                    )

                    st.rerun()

        except Exception as e:

            st.error(
                f"Error reading CSV: {e}"
            )


    # ========================================================
    # MANUAL ENTRY
    # ========================================================

    st.header(
        "✍️ Manual Student Record Entry"
    )

    with st.form(
        "manual_student_entry"
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
                list(SUBJECTS.keys())
            )

        with col2:

            subject = st.selectbox(
                "Subject",
                SUBJECTS[semester]
            )

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=6.0,
                value=3.0
            )

        col3, col4, col5 = st.columns(3)

        with col3:

            internal = st.number_input(
                "Internal Mark / 40",
                min_value=0.0,
                max_value=40.0,
                value=25.0
            )

        with col4:

            assignment = st.number_input(
                "Assignment / 15",
                min_value=0.0,
                max_value=15.0,
                value=10.0
            )

        with col5:

            previous = st.number_input(
                "Previous Mark / 60",
                min_value=0.0,
                max_value=60.0,
                value=40.0
            )

        submit_manual = st.form_submit_button(
            "➕ Add Student Record",
            width="stretch"
        )


    if submit_manual:

        if not username or not university_id or not student_name:

            st.error(
                "Please fill all student details."
            )

        else:

            overall = calculate_overall(
                attendance,
                study_hours,
                internal,
                assignment,
                previous
            )

            performance = performance_from_overall(
                overall
            )

            new_record = pd.DataFrame(
                [
                    {
                        "Username": username,
                        "University_ID": university_id,
                        "Student_Name": student_name,
                        "Branch": teacher_branch,
                        "Semester": semester,
                        "Subject": subject,
                        "Attendance": attendance,
                        "Study_Hours": study_hours,
                        "Internal_Mark": internal,
                        "Assignment": assignment,
                        "Previous_Mark": previous,
                        "Performance": performance,
                        "Overall_Percentage": overall,
                        "Cluster": 0,
                        "RF_Prediction": "",
                        "RF_Confidence": 0,
                        "Submitted_By": teacher_username,
                        "Submitted_Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    }
                ]
            )

            combined = pd.concat(
                [
                    df,
                    new_record
                ],
                ignore_index=True
            )

            combined = prepare_dataset(
                combined
            )

            save_database(
                combined
            )

            st.success(
                "Student record added successfully."
            )

            st.rerun()


    # ========================================================
    # BRANCH FILTER
    # ========================================================

    df = load_database()

    branch_df = df[
        df["Branch"].astype(str)
        == teacher_branch
    ].copy()


    # ========================================================
    # ANALYSIS
    # ========================================================

    st.header(
        "📊 Student Data Analysis"
    )

    if len(branch_df) == 0:

        st.warning(
            "No student records available for this branch."
        )

        return


    branch_df = prepare_dataset(
        branch_df
    )


    total_records = len(
        branch_df
    )

    total_students = (
        branch_df["University_ID"]
        .nunique()
    )

    avg_attendance = (
        branch_df["Attendance"]
        .astype(float)
        .mean()
    )

    avg_overall = (
        branch_df["Overall_Percentage"]
        .astype(float)
        .mean()
    )


    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Total Records",
        total_records
    )

    m2.metric(
        "Students",
        total_students
    )

    m3.metric(
        "Average Attendance",
        f"{avg_attendance:.1f}%"
    )

    m4.metric(
        "Average Performance",
        f"{avg_overall:.1f}%"
    )


    # ========================================================
    # PERFORMANCE DISTRIBUTION
    # ========================================================

    st.subheader(
        "📈 Performance Distribution"
    )

    performance_counts = (
        branch_df["Performance"]
        .value_counts()
    )

    st.bar_chart(
        performance_counts
    )


    # ========================================================
    # K-MEANS CLUSTERING
    # ========================================================

    st.subheader(
        "🔵 K-Means Clustering"
    )

    cluster_counts = (
        branch_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts
    )

    st.caption(
        "Cluster numbers are machine-learning group identifiers "
        "and are not official grades."
    )


    # ========================================================
    # STUDENT SUMMARY
    # ========================================================

    st.subheader(
        "👨‍🎓 Student Summary"
    )

    student_summary = (
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

    student_summary[
        "Performance"
    ] = student_summary[
        "Overall_Percentage"
    ].apply(
        performance_from_overall
    )

    st.dataframe(
        student_summary,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # SUBJECT ANALYSIS
    # ========================================================

    st.subheader(
        "📚 Subject-wise Analysis"
    )

    subject_summary = (
        branch_df
        .groupby("Subject")
        .agg(
            Average_Attendance=(
                "Attendance",
                "mean"
            ),
            Average_Study_Hours=(
                "Study_Hours",
                "mean"
            ),
            Average_Internal=(
                "Internal_Mark",
                "mean"
            ),
            Average_Assignment=(
                "Assignment",
                "mean"
            ),
            Average_Previous=(
                "Previous_Mark",
                "mean"
            ),
            Average_Percentage=(
                "Overall_Percentage",
                "mean"
            )
        )
        .reset_index()
    )

    st.dataframe(
        subject_summary.round(2),
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # PROGRESS REPORT DOWNLOAD
    # ========================================================

    st.header(
        "📄 Student Progress Report"
    )

    students = (
        branch_df[
            [
                "University_ID",
                "Student_Name"
            ]
        ]
        .drop_duplicates()
        .sort_values("Student_Name")
    )

    student_options = {

        f"{row['University_ID']} | {row['Student_Name']}":
        (
            row["University_ID"],
            row["Student_Name"]
        )

        for _, row in students.iterrows()
    }


    if student_options:

        selected_student = st.selectbox(
            "Select Student",
            list(student_options.keys())
        )

        selected_id, selected_name = (
            student_options[selected_student]
        )

        selected_df = branch_df[
            branch_df["University_ID"].astype(str)
            == str(selected_id)
        ].copy()

        st.dataframe(
            selected_df[
                [
                    "Semester",
                    "Subject",
                    "Attendance",
                    "Study_Hours",
                    "Internal_Mark",
                    "Assignment",
                    "Previous_Mark",
                    "Performance",
                    "Overall_Percentage",
                    "Cluster"
                ]
            ],
            width="stretch",
            hide_index=True
        )

        pdf_data = generate_progress_report(
            selected_df,
            selected_name,
            selected_id,
            teacher_branch
        )

        st.download_button(
            "📥 Download Student Progress Report",
            data=pdf_data,
            file_name=f"{selected_id}_Progress_Report.pdf",
            mime="application/pdf",
            width="stretch"
        )


    # ========================================================
    # DELETE STUDENT DATA
    # ========================================================

    st.header(
        "🗑️ Delete Student Data"
    )

    delete_options = {

        f"{row['University_ID']} | {row['Student_Name']}":
        row["University_ID"]

        for _, row in students.iterrows()
    }

    if delete_options:

        delete_student = st.selectbox(
            "Select student to delete",
            list(delete_options.keys()),
            key="delete_student"
        )

        delete_id = delete_options[
            delete_student
        ]

        confirm_delete = st.checkbox(
            "I confirm that I want to delete this student's complete history."
        )

        if st.button(
            "🗑️ Delete Student",
            width="stretch"
        ):

            if not confirm_delete:

                st.warning(
                    "Please confirm deletion first."
                )

            else:

                all_df = load_database()

                mask = ~(
                    (
                        all_df["Branch"].astype(str)
                        == teacher_branch
                    )
                    &
                    (
                        all_df["University_ID"].astype(str)
                        == str(delete_id)
                    )
                )

                all_df = all_df[
                    mask
                ]

                save_database(
                    all_df
                )

                st.success(
                    "Student data deleted successfully."
                )

                st.rerun()


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
# 25. STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    df = load_database()

    username = st.session_state.username

    university_id = (
        st.session_state.university_id
    )

    student_df = df[
        (
            df["Username"].astype(str)
            == str(username)
        )
        &
        (
            df["University_ID"].astype(str)
            == str(university_id)
        )
    ].copy()


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "🎓 Student Dashboard"
    )

    if st.button(
        "🚪 Logout"
    ):

        logout()


    if len(student_df) == 0:

        st.error(
            "No academic records found."
        )

        return


    student_df = prepare_dataset(
        student_df
    )


    student_name = (
        student_df["Student_Name"]
        .iloc[0]
    )

    branch = (
        student_df["Branch"]
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
        "Student Name",
        student_name
    )

    col2.metric(
        "University ID",
        university_id
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

    level = performance_from_overall(
        overall
    )

    info = performance_info(
        level
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

    if level == "Good Performance":

        st.success(
            "🟢 Excellent performance! Keep up the good work."
        )

        if not st.session_state.celebrated:

            st.balloons()

            st.session_state.celebrated = True


    elif level == "Above Average":

        st.warning(
            "🟡 Above average performance. "
            "Small improvements can help you reach the next level."
        )


    elif level == "Average Performance":

        st.warning(
            "🟠 Average performance. "
            "Regular revision and consistency can improve your results."
        )


    else:

        st.error(
            "🔴 Low performance. "
            "Focus on attendance, assignments, internal marks and regular study."
        )


    # ========================================================
    # SUBJECT-WISE REPORT
    # ========================================================

    st.header(
        "📚 Subject-wise Performance"
    )


    for subject, group in student_df.groupby(
        "Subject"
    ):

        row = group.iloc[0]

        subject_overall = float(
            group["Overall_Percentage"]
            .mean()
        )

        subject_level = (
            performance_from_overall(
                subject_overall
            )
        )

        subject_info = performance_info(
            subject_level
        )


        with st.expander(
            f"{subject}  |  {subject_info['emoji']} {subject_level}"
        ):

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Attendance",
                f"{float(row['Attendance']):.1f}%"
            )

            col2.metric(
                "Attendance Mark",
                f"{attendance_mark(row['Attendance'])}/5"
            )

            col3.metric(
                "Study Hours",
                f"{float(row['Study_Hours']):.1f}/6"
            )


            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Internal",
                f"{float(row['Internal_Mark']):.1f}/40"
            )

            col5.metric(
                "Assignment",
                f"{float(row['Assignment']):.1f}/15"
            )

            col6.metric(
                "Previous Mark",
                f"{float(row['Previous_Mark']):.1f}/60"
            )


            st.markdown(
                f"""
                <div style="
                    padding:15px;
                    border-radius:15px;
                    background:{subject_info['color']}18;
                    border:1px solid {subject_info['color']};
                    margin-top:10px;
                ">

                    <span style="
                        color:{subject_info['color']};
                        font-size:20px;
                    ">
                        ●
                    </span>

                    <b>
                        {subject_level}
                    </b>

                    <br>

                    Overall:
                    {subject_overall:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
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

    student_clusters = (
        student_df["Cluster"]
        .dropna()
        .unique()
    )

    st.info(
        "Your cluster: "
        + ", ".join(
            str(int(x))
            for x in student_clusters
        )
    )

    st.caption(
        "The K-Means cluster is a machine-learning grouping "
        "identifier and is not an academic grade."
    )


    # ========================================================
    # DOWNLOAD PROGRESS REPORT
    # ========================================================

    st.header(
        "📄 My Progress Report"
    )

    pdf_data = generate_progress_report(
        student_df,
        student_name,
        university_id,
        branch
    )

    st.download_button(
        "📥 Download My Progress Report",
        data=pdf_data,
        file_name=f"{university_id}_Progress_Report.pdf",
        mime="application/pdf",
        width="stretch"
    )


# ============================================================
# 26. MAIN APPLICATION ROUTER
# ============================================================

if not st.session_state.logged_in:

    login_page()

else:

    if st.session_state.user_role == "tutor":

        tutor_dashboard()

    elif st.session_state.user_role == "student":

        student_dashboard()
