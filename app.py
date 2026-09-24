import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
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
    TableStyle,
    PageBreak
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ==========================================================
# DATA FOLDER
# ==========================================================

os.makedirs("data", exist_ok=True)

REPORT_FILE = "data/student_reports.csv"


# ==========================================================
# B.TECH BRANCH OPTIONS
# ==========================================================

BRANCHES = [
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning",
    "Computer Science and Engineering",
    "Computer Science and Engineering (AI)",
    "Computer Science and Engineering (Data Science)",
    "Computer Science and Engineering (Cyber Security)",
    "Information Technology",
    "Cyber Security",
    "Data Science",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electronics and Instrumentation Engineering",
    "Electronics Engineering",
    "Instrumentation Engineering",
    "Biomedical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Automobile Engineering",
    "Aeronautical Engineering",
    "Food Technology",
    "Biotechnology",
    "Industrial Engineering",
    "Production Engineering",
    "Mechatronics Engineering",
    "Robotics and Automation",
    "Environmental Engineering",
    "Computer Engineering"
]


# ==========================================================
# S1-S8 SUBJECT OPTIONS
# ==========================================================

SUBJECTS = {

    "S1": [
        "Mathematics I",
        "Physics",
        "Chemistry",
        "Engineering Graphics",
        "Programming in C",
        "Engineering Mechanics",
        "Basic Electrical Engineering",
        "Basic Civil Engineering",
        "Life Skills",
        "Workshop Practice",
        "Design and Engineering"
    ],

    "S2": [
        "Mathematics II",
        "Engineering Physics",
        "Engineering Chemistry",
        "Python Programming",
        "Engineering Mechanics",
        "Basic Electronics",
        "Professional Communication",
        "Environmental Science",
        "Electrical Engineering",
        "Design and Engineering",
        "Programming Lab"
    ],

    "S3": [
        "Mathematics III",
        "Data Structures",
        "Database Management Systems",
        "Artificial Intelligence",
        "Machine Learning",
        "Computer Organization",
        "Object Oriented Programming",
        "Digital Electronics",
        "Probability and Statistics",
        "Operating Systems",
        "Data Structures Lab"
    ],

    "S4": [
        "Mathematics IV",
        "Operating Systems",
        "Computer Networks",
        "Design and Analysis of Algorithms",
        "Artificial Intelligence",
        "Machine Learning",
        "Software Engineering",
        "Microprocessors",
        "Web Programming",
        "Theory of Computation",
        "Database Lab"
    ],

    "S5": [
        "Compiler Design",
        "Computer Networks",
        "Distributed Computing",
        "Data Mining",
        "Deep Learning",
        "Cloud Computing",
        "Cyber Security",
        "Software Testing",
        "Computer Graphics",
        "Data Analytics",
        "Elective I"
    ],

    "S6": [
        "Big Data Analytics",
        "Natural Language Processing",
        "Computer Vision",
        "Internet of Things",
        "Cloud Computing",
        "Information Security",
        "Data Science",
        "Mobile Computing",
        "Data Mining Lab",
        "Elective II",
        "Project Phase I"
    ],

    "S7": [
        "Advanced Machine Learning",
        "Deep Learning",
        "Blockchain",
        "Artificial Intelligence Applications",
        "Data Analytics",
        "Elective III",
        "Elective IV",
        "Seminar",
        "Project Phase II",
        "Professional Elective",
        "Major Project"
    ],

    "S8": [
        "Project",
        "Project Viva",
        "Comprehensive Viva",
        "Elective V",
        "Elective VI",
        "Industrial Training",
        "Advanced Artificial Intelligence",
        "Advanced Data Science",
        "Professional Elective",
        "Open Elective",
        "Technical Seminar"
    ]
}


# ==========================================================
# TEACHER ACCOUNTS
# ==========================================================

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

    "teacher_eie": {
        "password": "ktutech",
        "branch": "Electronics and Instrumentation Engineering"
    },

    "teacher_me": {
        "password": "ktutech",
        "branch": "Mechanical Engineering"
    },

    "teacher_ce": {
        "password": "ktutech",
        "branch": "Civil Engineering"
    },

    "teacher_auto": {
        "password": "ktutech",
        "branch": "Automobile Engineering"
    },

    "teacher_rob": {
        "password": "ktutech",
        "branch": "Robotics and Automation"
    },

    "teacher_mec": {
        "password": "ktutech",
        "branch": "Mechatronics Engineering"
    }
}


# ==========================================================
# SESSION STATE
# ==========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

if "branch" not in st.session_state:
    st.session_state.branch = None


# ==========================================================
# ATTENDANCE CONVERSION
# ==========================================================

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

    else:
        return 0


# ==========================================================
# PERFORMANCE ANALYSIS
# ==========================================================

def analyse_performance(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    # Attendance is converted according to the requested table
    attendance_score = (
        attendance_mark(attendance) / 5
    ) * 100

    study_score = min(
        (study_hours / 6) * 100,
        100
    )

    internal_score = (
        internal / 40
    ) * 100

    assignment_score = (
        assignment / 15
    ) * 100

    previous_score = (
        previous / 60
    ) * 100

    overall = np.mean([
        attendance_score,
        study_score,
        internal_score,
        assignment_score,
        previous_score
    ])

    # Project-defined performance bands
    if overall < 50:

        return (
            "Low Performance",
            "🔴",
            overall
        )

    elif overall < 65:

        return (
            "Average Performance",
            "🟠",
            overall
        )

    elif overall < 80:

        return (
            "Above Average",
            "🟡",
            overall
        )

    else:

        return (
            "Good Performance",
            "🟢",
            overall
        )


# ==========================================================
# SUBJECT IMPROVEMENT / COMPLEMENT
# ==========================================================

def get_subject_feedback(
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
    level
):

    feedback = []

    # ---------------- LOW ----------------

    if level == "Low Performance":

        feedback.append(
            "⚠️ This subject needs attention and regular improvement."
        )

        if attendance < 75:
            feedback.append(
                "Improve attendance and attend classes regularly."
            )

        if study_hours < 2:
            feedback.append(
                "Increase daily study time gradually."
            )

        if internal < 20:
            feedback.append(
                "Improve internal examination preparation."
            )

        if assignment < 8:
            feedback.append(
                "Complete assignments on time."
            )

        if previous < 30:
            feedback.append(
                "Revise previous topics and strengthen fundamentals."
            )

    # ---------------- AVERAGE ----------------

    elif level == "Average Performance":

        feedback.append(
            "You are making reasonable progress, but consistency can be improved."
        )

        if attendance < 85:
            feedback.append(
                "Improve attendance slightly."
            )

        if internal < 28:
            feedback.append(
                "Try to increase internal marks through regular revision."
            )

        if study_hours < 3:
            feedback.append(
                "Add a little more daily study time."
            )

        if assignment < 11:
            feedback.append(
                "Try to score better in assignments."
            )

    # ---------------- ABOVE AVERAGE ----------------

    elif level == "Above Average":

        feedback.append(
            "Good progress. Small incremental improvements can help you reach Good Performance."
        )

        if attendance < 90:
            feedback.append(
                "A small improvement in attendance can strengthen the overall score."
            )

        if internal < 32:
            feedback.append(
                "A few additional internal marks can improve the overall performance."
            )

        if assignment < 13:
            feedback.append(
                "A small improvement in assignment marks can help."
            )

        if study_hours < 4:
            feedback.append(
                "Slightly increasing focused study time can improve consistency."
            )

    # ---------------- GOOD ----------------

    else:

        feedback.append(
            "🌟 Excellent work! Your performance in this subject is very good."
        )

        feedback.append(
            "👏 Keep up the consistency and maintain your current standard."
        )

        feedback.append(
            "🏆 Continue regular revision and active participation."
        )

    return feedback


# ==========================================================
# DATASET
# ==========================================================

def create_fallback_dataset():

    rng = np.random.default_rng(42)

    n = 300

    attendance = rng.uniform(40, 100, n)
    study = rng.uniform(0, 6, n)
    internal = rng.uniform(5, 40, n)
    assignment = rng.uniform(3, 15, n)
    previous = rng.uniform(10, 60, n)

    score = (
        attendance * 0.25
        + (study / 6 * 100) * 0.15
        + (internal / 40 * 100) * 0.25
        + (assignment / 15 * 100) * 0.15
        + (previous / 60 * 100) * 0.20
    )

    performance = []

    for value in score:

        if value < 50:
            performance.append("Low")

        elif value < 70:
            performance.append("Medium")

        else:
            performance.append("High")

    return pd.DataFrame({
        "Attendance": attendance,
        "Study_Hours": study,
        "Internal_Mark": internal,
        "Assignment": assignment,
        "Previous_Mark": previous,
        "Performance": performance
    })


def load_dataset():

    path = "data/student_performance.csv"

    if os.path.exists(path):

        try:

            df = pd.read_csv(path)

            required = [
                "Attendance",
                "Study_Hours",
                "Internal_Mark",
                "Assignment",
                "Previous_Mark",
                "Performance"
            ]

            if all(
                column in df.columns
                for column in required
            ):

                return df

        except Exception:
            pass

    return create_fallback_dataset()


# ==========================================================
# PREPARE DATA FOR MACHINE LEARNING
# ==========================================================

def prepare_dataset(df):

    data = df.copy()

    data["Attendance_Model"] = pd.to_numeric(
        data["Attendance"],
        errors="coerce"
    ).fillna(0)

    study = pd.to_numeric(
        data["Study_Hours"],
        errors="coerce"
    ).fillna(0)

    data["Study_Model"] = (
        study / 6
    ) * 100

    internal = pd.to_numeric(
        data["Internal_Mark"],
        errors="coerce"
    ).fillna(0)

    data["Internal_Model"] = (
        internal / 40
    ) * 100

    assignment = pd.to_numeric(
        data["Assignment"],
        errors="coerce"
    ).fillna(0)

    data["Assignment_Model"] = (
        assignment / 15
    ) * 100

    previous = pd.to_numeric(
        data["Previous_Mark"],
        errors="coerce"
    ).fillna(0)

    data["Previous_Model"] = (
        previous / 60
    ) * 100

    return data


# ==========================================================
# K-MEANS + RANDOM FOREST
# ==========================================================

@st.cache_resource
def train_models():

    df = load_dataset()

    df = prepare_dataset(df)

    features = [
        "Attendance_Model",
        "Study_Model",
        "Internal_Model",
        "Assignment_Model",
        "Previous_Model"
    ]

    X = df[features].fillna(0)

    y = df["Performance"].astype(str)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------
    # K-MEANS CLUSTERING
    # ------------------------------------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(
        X_scaled
    )

    # ------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    random_forest.fit(
        X_scaled,
        y
    )

    return (
        random_forest,
        kmeans,
        scaler
    )


# ==========================================================
# PREDICT
# ==========================================================

def predict_student(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    random_forest, kmeans, scaler = train_models()

    values = np.array([[
        attendance,
        (study_hours / 6) * 100,
        (internal / 40) * 100,
        (assignment / 15) * 100,
        (previous / 60) * 100
    ]])

    scaled = scaler.transform(values)

    # K-MEANS
    cluster = int(
        kmeans.predict(scaled)[0]
    )

    # RANDOM FOREST
    prediction = random_forest.predict(
        scaled
    )[0]

    probabilities = random_forest.predict_proba(
        scaled
    )[0]

    confidence = float(
        np.max(probabilities) * 100
    )

    return (
        prediction,
        confidence,
        cluster
    )


# ==========================================================
# REPORT DATABASE
# ==========================================================

REPORT_COLUMNS = [
    "Student_Name",
    "University_ID",
    "Semester",
    "Branch",
    "Subjects_JSON",
    "Created_Time"
]


def load_reports():

    if not os.path.exists(REPORT_FILE):

        return pd.DataFrame(
            columns=REPORT_COLUMNS
        )

    try:

        return pd.read_csv(
            REPORT_FILE
        )

    except Exception:

        return pd.DataFrame(
            columns=REPORT_COLUMNS
        )


def save_reports(df):

    df.to_csv(
        REPORT_FILE,
        index=False
    )


# ==========================================================
# PDF COLOURS
# ==========================================================

PDF_COLOURS = {

    "Low Performance":
        colors.HexColor("#F8CCCC"),

    "Average Performance":
        colors.HexColor("#FFD8A8"),

    "Above Average":
        colors.HexColor("#FFF0A6"),

    "Good Performance":
        colors.HexColor("#BFE8C2")
}


# ==========================================================
# COLOURED PDF REPORT
# ==========================================================

def generate_pdf(student):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25,
        leftMargin=25,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=12
    )

    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=8
    )

    normal = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12
    )

    story = []

    # ======================================================
    # TITLE
    # ======================================================

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            title
        )
    )

    story.append(
        Paragraph(
            "AI-Based Student Performance Prediction System",
            normal
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # ======================================================
    # OVERALL STUDENT INFORMATION
    # ======================================================

    story.append(
        Paragraph(
            "1. Student Information",
            heading
        )
    )

    information = [
        ["Student Name", str(student["Student_Name"])],
        ["University ID", str(student["University_ID"])],
        ["Semester", str(student["Semester"])],
        ["Branch", str(student["Branch"])],
        ["Report Date", str(student["Created_Time"])]
    ]

    info_table = Table(
        information,
        colWidths=[120, 390]
    )

    info_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#E8E8E8")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(info_table)

    story.append(Spacer(1, 15))

    # ======================================================
    # MARKING SCHEME
    # ======================================================

    story.append(
        Paragraph(
            "2. Assessment Information",
            heading
        )
    )

    marking_data = [

        ["Component", "Maximum"],

        ["Attendance", "100%"],

        ["Internal Mark", "40"],

        ["Previous Mark", "60"],

        ["Study Hours", "6 hours/day"],

        ["Assignment Score", "15"]
    ]

    marking_table = Table(
        marking_data,
        colWidths=[300, 210]
    )

    marking_table.setStyle(
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
                colors.HexColor("#D9EAF7")
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(marking_table)

    story.append(Spacer(1, 15))

    # ======================================================
    # ATTENDANCE CONVERSION
    # ======================================================

    story.append(
        Paragraph(
            "3. Attendance Conversion",
            heading
        )
    )

    attendance_data = [

        ["Attendance", "Converted Mark"],

        ["90% - 100%", "5"],

        ["80% - 89%", "4"],

        ["70% - 79%", "3"],

        ["60% - 69%", "2"],

        ["10% - 59%", "1"],

        ["Below 10%", "0"]
    ]

    attendance_table = Table(
        attendance_data,
        colWidths=[300, 210]
    )

    attendance_table.setStyle(
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
                colors.HexColor("#D9EAF7")
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            )
        ])
    )

    story.append(
        attendance_table
    )

    story.append(
        Spacer(1, 15)
    )

    # ======================================================
    # SUBJECT DATA
    # ======================================================

    subjects = json.loads(
        student["Subjects_JSON"]
    )

    # ======================================================
    # OVERALL INFORMATION
    # ======================================================

    story.append(
        Paragraph(
            "4. Overall Student Performance",
            heading
        )
    )

    overall_scores = []

    for subject in subjects:
        overall_scores.append(
            subject["Overall"]
        )

    overall_average = np.mean(
        overall_scores
    )

    if overall_average < 50:

        overall_level = "Low Performance"
        overall_circle = "RED"

    elif overall_average < 65:

        overall_level = "Average Performance"
        overall_circle = "ORANGE"

    elif overall_average < 80:

        overall_level = "Above Average"
        overall_circle = "YELLOW"

    else:

        overall_level = "Good Performance"
        overall_circle = "GREEN"

    overall_data = [

        ["Overall Average",
         f"{overall_average:.2f}%"],

        ["Overall Performance",
         overall_level],

        ["Indicator",
         overall_circle],

        ["Number of Subjects",
         str(len(subjects))]
    ]

    overall_table = Table(
        overall_data,
        colWidths=[200, 310]
    )

    overall_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#E8E8E8")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "BACKGROUND",
                (1, 1),
                (1, 1),
                PDF_COLOURS[overall_level]
            )
        ])
    )

    story.append(
        overall_table
    )

    story.append(
        Spacer(1, 18)
    )

    # ======================================================
    # EACH SUBJECT SEPARATELY
    # ======================================================

    story.append(
        Paragraph(
            "5. Subject-wise Performance",
            heading
        )
    )

    summary_data = [[
        "Subject",
        "Attendance",
        "Attend. Mark",
        "Internal",
        "Assignment",
        "Previous",
        "Study",
        "Performance"
    ]]

    summary_levels = []

    for subject in subjects:

        summary_data.append([

            subject["Subject"],

            f'{subject["Attendance"]:.0f}%',

            f'{subject["Attendance_Mark"]}/5',

            f'{subject["Internal"]:.0f}/40',

            f'{subject["Assignment"]:.0f}/15',

            f'{subject["Previous"]:.0f}/60',

            f'{subject["Study_Hours"]:.1f} hr',

            subject["Level"]
        ])

        summary_levels.append(
            subject["Level"]
        )

    summary_table = Table(
        summary_data,
        repeatRows=1,
        colWidths=[
            105,
            48,
            55,
            50,
            55,
            55,
            48,
            90
        ]
    )

    commands = [

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
            colors.HexColor("#D9EAF7")
        ),

        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            6.5
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE"
        )
    ]

    # Each subject gets its own colour
    for row, level in enumerate(
        summary_levels,
        start=1
    ):

        commands.append(
            (
                "BACKGROUND",
                (0, row),
                (-1, row),
                PDF_COLOURS[level]
            )
        )

    summary_table.setStyle(
        TableStyle(commands)
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(1, 18)
    )

    # ======================================================
    # DETAILED SUBJECT REPORT
    # ======================================================

    story.append(
        Paragraph(
            "6. Detailed Subject Reports",
            heading
        )
    )

    for number, subject in enumerate(
        subjects,
        start=1
    ):

        level = subject["Level"]

        # --------------------------------------------------
        # SUBJECT TITLE WITH COLOUR
        # --------------------------------------------------

        subject_title = Table([[
            f"{number}. {subject['Subject']}",
            subject["Icon"] + " " + level
        ]], colWidths=[360, 150])

        subject_title.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, 0),
                    colors.HexColor("#E8E8E8")
                ),
                (
                    "BACKGROUND",
                    (1, 0),
                    (1, 0),
                    PDF_COLOURS[level]
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica-Bold"
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
            subject_title
        )

        story.append(
            Spacer(1, 5)
        )

        # --------------------------------------------------
        # SUBJECT MARKS
        # --------------------------------------------------

        subject_marks = [

            ["Attendance",
             f'{subject["Attendance"]:.0f}%'],

            ["Attendance Converted Mark",
             f'{subject["Attendance_Mark"]}/5'],

            ["Study Hours",
             f'{subject["Study_Hours"]:.1f} / 6 hours'],

            ["Internal Mark",
             f'{subject["Internal"]:.0f}/40'],

            ["Assignment",
             f'{subject["Assignment"]:.0f}/15'],

            ["Previous Mark",
             f'{subject["Previous"]:.0f}/60'],

            ["Overall Subject Score",
             f'{subject["Overall"]:.2f}%'],

            ["Performance",
             level]
        ]

        subject_marks_table = Table(
            subject_marks,
            colWidths=[230, 280]
        )

        subject_commands = [

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
                colors.HexColor("#F0F0F0")
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            )
        ]

        subject_commands.append(
            (
                "BACKGROUND",
                (1, 7),
                (1, 7),
                PDF_COLOURS[level]
            )
        )

        subject_marks_table.setStyle(
            TableStyle(subject_commands)
        )

        story.append(
            subject_marks_table
        )

        story.append(
            Spacer(1, 6)
        )

        # --------------------------------------------------
        # ML INFORMATION
        # --------------------------------------------------

        if "ML_Prediction" in subject:

            ml_data = [

                [
                    "K-Means Cluster",
                    f'Cluster {subject["Cluster"]}'
                ],

                [
                    "Random Forest Prediction",
                    subject["ML_Prediction"]
                ],

                [
                    "Prediction Confidence",
                    f'{subject["Confidence"]:.2f}%'
                ]
            ]

            ml_table = Table(
                ml_data,
                colWidths=[230, 280]
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
                        (0, -1),
                        colors.HexColor("#E8F1FA")
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (0, -1),
                        "Helvetica-Bold"
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

            story.append(
                Spacer(1, 6)
            )

        # --------------------------------------------------
        # FEEDBACK
        # --------------------------------------------------

        story.append(
            Paragraph(
                "<b>Improvement / Complement:</b>",
                normal
            )
        )

        for message in subject[
            "Recommendations"
        ]:

            story.append(
                Paragraph(
                    "• " + message,
                    normal
                )
            )

        story.append(
            Spacer(1, 12)
        )

    # ======================================================
    # INDICATOR LEGEND
    # ======================================================

    story.append(
        Paragraph(
            "7. Performance Indicator Legend",
            heading
        )
    )

    legend = [
        ["🔴", "Low Performance", "Needs improvement"],
        ["🟠", "Average Performance", "Needs consistency"],
        ["🟡", "Above Average", "Small incremental improvement"],
        ["🟢", "Good Performance", "Excellent – maintain consistency"]
    ]

    legend_table = Table(
        legend,
        colWidths=[40, 160, 310]
    )

    legend_commands = [
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.4,
            colors.grey
        ),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8
        )
    ]

    for row, (_, level, _) in enumerate(legend):

        level_map = {
            "Low Performance": "Low Performance",
            "Average Performance": "Average Performance",
            "Above Average": "Above Average",
            "Good Performance": "Good Performance"
        }

        legend_commands.append(
            (
                "BACKGROUND",
                (0, row),
                (-1, row),
                PDF_COLOURS[level_map[level]]
            )
        )

    legend_table.setStyle(
        TableStyle(legend_commands)
    )

    story.append(
        legend_table
    )

    story.append(
        Spacer(1, 15)
    )

    # ======================================================
    # FINAL NOTE
    # ======================================================

    story.append(
        Paragraph(
            "Note: The performance categories and overall score "
            "used in this project are project-defined indicators "
            "for student monitoring. K-Means cluster numbers are "
            "data-driven group identifiers and do not themselves "
            "represent good or poor performance. Machine-learning "
            "predictions should support, not replace, teacher "
            "evaluation.",
            normal
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================================
# HOME
# ==========================================================

def home():

    st.title(
        "🎓 Student Performance Prediction"
    )

    st.subheader(
        "Choose Login"
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "🎓 Student Login",
            width="stretch"
        ):

            st.session_state.page = "student_login"

            st.rerun()

    with c2:

        if st.button(
            "👨‍🏫 Teacher Login",
            width="stretch"
        ):

            st.session_state.page = "teacher_login"

            st.rerun()


# ==========================================================
# STUDENT LOGIN
# ==========================================================

def student_login():

    st.title(
        "🎓 Student Login"
    )

    password = st.text_input(
        "Password",
        type="password",
        max_chars=9
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "🔐 Login",
            width="stretch"
        ):

            valid = False

            if len(password) == 9:

                if password.startswith("BTECH"):

                    year = password[5:]

                    if year.isdigit():

                        year = int(year)

                        if 2000 <= year <= 2020:

                            valid = True

            if valid:

                st.session_state.logged_in = True
                st.session_state.role = "student"
                st.session_state.username = password
                st.session_state.page = "student_dashboard"

                st.success(
                    "🔔 Login successful! Welcome Student."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid login."
                )

    with c2:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state.page = "home"

            st.rerun()


# ==========================================================
# TEACHER LOGIN
# ==========================================================

def teacher_login():

    st.title(
        "👨‍🏫 Teacher Login"
    )

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "🔐 Login",
            width="stretch"
        ):

            if (
                username in TEACHERS
                and
                TEACHERS[username]["password"] == password
            ):

                st.session_state.logged_in = True
                st.session_state.role = "teacher"
                st.session_state.username = username
                st.session_state.branch = (
                    TEACHERS[username]["branch"]
                )
                st.session_state.page = "teacher_dashboard"

                st.success(
                    "🔔 Login successful! Welcome Teacher."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    with c2:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state.page = "home"

            st.rerun()


# ==========================================================
# STUDENT DASHBOARD
# ==========================================================

def student_dashboard():

    st.title(
        "🎓 Student Dashboard"
    )

    st.success(
        "🔔 Login successful. Enter your academic details."
    )

    if st.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.page = "home"

        st.rerun()

    st.divider()

    # ------------------------------------------------------
    # STUDENT DETAILS
    # ------------------------------------------------------

    st.subheader(
        "👤 Student Information"
    )

    c1, c2 = st.columns(2)

    with c1:

        student_name = st.text_input(
            "Student Name"
        )

        university_id = st.text_input(
            "University ID"
        )

    with c2:

        semester = st.selectbox(
            "Semester",
            list(SUBJECTS.keys())
        )

        branch = st.selectbox(
            "B.Tech Branch",
            BRANCHES
        )

    # ------------------------------------------------------
    # SUBJECT SELECTION
    # ------------------------------------------------------

    st.subheader(
        "📚 Select Exactly 6 Subjects"
    )

    selected_subjects = st.multiselect(
        f"Subjects for {semester}",
        SUBJECTS[semester],
        max_selections=6
    )

    st.write(
        f"Selected subjects: "
        f"**{len(selected_subjects)} / 6**"
    )

    if len(selected_subjects) != 6:

        st.warning(
            "⚠️ Please select exactly 6 subjects."
        )

        return

    # ------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------

    subject_results = []

    st.divider()

    st.subheader(
        "📊 Subject-wise Marks"
    )

    for i, subject in enumerate(
        selected_subjects
    ):

        st.markdown(
            f"### {i + 1}. {subject}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            attendance = st.number_input(
                "Attendance (%)",
                0.0,
                100.0,
                75.0,
                1.0,
                key=f"attendance_{i}"
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                0.0,
                6.0,
                2.0,
                0.5,
                key=f"study_{i}"
            )

        with c2:

            internal = st.number_input(
                "Internal Mark / 40",
                0.0,
                40.0,
                20.0,
                1.0,
                key=f"internal_{i}"
            )

            assignment = st.number_input(
                "Assignment Score / 15",
                0.0,
                15.0,
                8.0,
                1.0,
                key=f"assignment_{i}"
            )

        with c3:

            previous = st.number_input(
                "Previous Mark / 60",
                0.0,
                60.0,
                30.0,
                1.0,
                key=f"previous_{i}"
            )

            level, icon, overall = analyse_performance(
                attendance,
                study_hours,
                internal,
                assignment,
                previous
            )

            st.metric(
                "Performance",
                f"{icon} {level}"
            )

        feedback = get_subject_feedback(
            attendance,
            study_hours,
            internal,
            assignment,
            previous,
            level
        )

        subject_results.append({

            "Subject": subject,

            "Attendance": attendance,

            "Attendance_Mark":
                attendance_mark(attendance),

            "Study_Hours":
                study_hours,

            "Internal":
                internal,

            "Assignment":
                assignment,

            "Previous":
                previous,

            "Level":
                level,

            "Icon":
                icon,

            "Overall":
                overall,

            "Recommendations":
                feedback
        })

        st.divider()

    # ------------------------------------------------------
    # RUN ML
    # ------------------------------------------------------

    if st.button(
        "🤖 Run K-Means + Random Forest",
        type="primary",
        width="stretch"
    ):

        if not student_name.strip():

            st.error(
                "Enter Student Name."
            )

            return

        if not university_id.strip():

            st.error(
                "Enter University ID."
            )

            return

        # --------------------------------------------------
        # PREDICT EACH SUBJECT
        # --------------------------------------------------

        for subject in subject_results:

            prediction, confidence, cluster = predict_student(

                subject["Attendance"],

                subject["Study_Hours"],

                subject["Internal"],

                subject["Assignment"],

                subject["Previous"]
            )

            subject["ML_Prediction"] = prediction

            subject["Confidence"] = confidence

            subject["Cluster"] = cluster

        # --------------------------------------------------
        # SAVE REPORT
        # --------------------------------------------------

        reports = load_reports()

        # Replace old report of same student
        reports = reports[
            reports["University_ID"].astype(str)
            != str(university_id)
        ].copy()

        created = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        new_report = {

            "Student_Name":
                student_name.strip(),

            "University_ID":
                university_id.strip(),

            "Semester":
                semester,

            "Branch":
                branch,

            "Subjects_JSON":
                json.dumps(subject_results),

            "Created_Time":
                created
        }

        reports = pd.concat(
            [
                reports,
                pd.DataFrame([new_report])
            ],
            ignore_index=True
        )

        save_reports(reports)

        # --------------------------------------------------
        # PDF
        # --------------------------------------------------

        pdf = generate_pdf(
            new_report
        )

        st.success(
            "🎉 Progress analysis completed successfully!"
        )

        # --------------------------------------------------
        # RESULT TABLE
        # --------------------------------------------------

        result_rows = []

        for subject in subject_results:

            result_rows.append([

                subject["Subject"],

                subject["Icon"],

                subject["Level"],

                f'{subject["Overall"]:.2f}%',

                f'{subject["Attendance_Mark"]}/5',

                subject["ML_Prediction"],

                f'{subject["Confidence"]:.1f}%',

                f'Cluster {subject["Cluster"]}'
            ])

        result_df = pd.DataFrame(

            result_rows,

            columns=[

                "Subject",

                "Indicator",

                "Performance",

                "Overall",

                "Attendance Mark",

                "Random Forest",

                "Confidence",

                "K-Means"
            ]
        )

        st.dataframe(
            result_df,
            width="stretch"
        )

        # --------------------------------------------------
        # PDF DOWNLOAD
        # --------------------------------------------------

        st.download_button(

            "📥 Download Coloured Progress Report PDF",

            data=pdf,

            file_name=
                f"{university_id}_Progress_Report.pdf",

            mime="application/pdf",

            width="stretch"
        )


# ==========================================================
# TEACHER DASHBOARD
# ==========================================================

def teacher_dashboard():

    teacher_branch = st.session_state.branch

    st.title(
        "👨‍🏫 Teacher Dashboard"
    )

    st.success(
        f"🔔 Logged in successfully. "
        f"Branch: {teacher_branch}"
    )

    if st.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.branch = None
        st.session_state.page = "home"

        st.rerun()

    reports = load_reports()

    # ------------------------------------------------------
    # ONLY TEACHER'S BRANCH
    # ------------------------------------------------------

    branch_reports = reports[
        reports["Branch"].astype(str)
        == str(teacher_branch)
    ].copy()

    st.subheader(
        "👨‍🎓 Students in Your Branch"
    )

    if branch_reports.empty:

        st.info(
            "No student reports found."
        )

        return

    st.dataframe(
        branch_reports[
            [
                "Student_Name",
                "University_ID",
                "Semester",
                "Branch",
                "Created_Time"
            ]
        ],
        width="stretch"
    )

    # ------------------------------------------------------
    # SELECT STUDENT
    # ------------------------------------------------------

    student_ids = (
        branch_reports[
            "University_ID"
        ]
        .astype(str)
        .tolist()
    )

    selected_id = st.selectbox(
        "Select Student",
        student_ids
    )

    selected = branch_reports[
        branch_reports[
            "University_ID"
        ].astype(str)
        == str(selected_id)
    ]

    if selected.empty:
        return

    student = selected.iloc[0]

    st.markdown(
        f"### 👤 {student['Student_Name']}"
    )

    st.write(
        f"University ID: **{student['University_ID']}**"
    )

    st.write(
        f"Semester: **{student['Semester']}**"
    )

    subjects = json.loads(
        student["Subjects_JSON"]
    )

    teacher_rows = []

    for subject in subjects:

        teacher_rows.append([

            subject["Subject"],

            f'{subject["Attendance"]:.0f}%',

            f'{subject["Attendance_Mark"]}/5',

            f'{subject["Study_Hours"]:.1f}',

            f'{subject["Internal"]:.0f}/40',

            f'{subject["Assignment"]:.0f}/15',

            f'{subject["Previous"]:.0f}/60',

            subject["Icon"],

            subject["Level"],

            f'Cluster {subject.get("Cluster", "-")}'
        ])

    teacher_df = pd.DataFrame(

        teacher_rows,

        columns=[

            "Subject",
            "Attendance",
            "Attend Mark",
            "Study Hrs",
            "Internal",
            "Assignment",
            "Previous",
            "Indicator",
            "Performance",
            "K-Means"
        ]
    )

    st.dataframe(
        teacher_df,
        width="stretch"
    )

    # ------------------------------------------------------
    # PDF
    # ------------------------------------------------------

    pdf = generate_pdf(
        student.to_dict()
    )

    st.download_button(

        "📥 Download Selected Student PDF",

        data=pdf,

        file_name=
            f"{selected_id}_Progress_Report.pdf",

        mime="application/pdf",

        width="stretch"
    )

    st.divider()

    # ------------------------------------------------------
    # DELETE
    # ------------------------------------------------------

    st.subheader(
        "🗑️ Delete Student Details and Marks"
    )

    st.warning(
        "This deletes the selected student's saved "
        "student information and all 6 subject records."
    )

    confirm = st.checkbox(
        "I confirm that I want to delete this student."
    )

    if st.button(
        "🗑️ Delete Selected Student",
        width="stretch"
    ):

        if not confirm:

            st.error(
                "Please confirm deletion."
            )

        else:

            latest = load_reports()

            delete_mask = (

                (
                    latest["University_ID"]
                    .astype(str)
                    == str(selected_id)
                )

                &

                (
                    latest["Branch"]
                    .astype(str)
                    == str(teacher_branch)
                )
            )

            latest = latest[
                ~delete_mask
            ].copy()

            save_reports(latest)

            st.success(
                "🗑️ Student details and marks deleted successfully."
            )

            st.rerun()


# ==========================================================
# MAIN ROUTER
# ==========================================================

if st.session_state.page == "home":

    home()

elif st.session_state.page == "student_login":

    student_login()

elif st.session_state.page == "teacher_login":

    teacher_login()

elif (
    st.session_state.page == "student_dashboard"
    and st.session_state.logged_in
    and st.session_state.role == "student"
):

    student_dashboard()

elif (
    st.session_state.page == "teacher_dashboard"
    and st.session_state.logged_in
    and st.session_state.role == "teacher"
):

    teacher_dashboard()

else:

    st.session_state.page = "home"

    st.rerun()
