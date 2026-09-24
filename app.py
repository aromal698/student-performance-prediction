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
    PageBreak,
    KeepTogether
)
from reportlab.graphics.shapes import Drawing, Circle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# FILES
# ============================================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

REPORT_FILE = os.path.join(
    DATA_DIR,
    "student_reports.csv"
)


# ============================================================
# B.TECH BRANCH OPTIONS
# ============================================================

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
    "Computer Engineering",
    "Software Engineering",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electronics and Instrumentation Engineering",
    "Electronics Engineering",
    "Instrumentation Engineering",
    "Biomedical Engineering",
    "Mechanical Engineering",
    "Automobile Engineering",
    "Aeronautical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Food Technology",
    "Biotechnology",
    "Industrial Engineering",
    "Production Engineering",
    "Mechatronics Engineering",
    "Robotics and Automation",
    "Environmental Engineering",
    "Marine Engineering",
    "Metallurgical Engineering",
    "Mining Engineering",
    "Textile Engineering",
    "Agricultural Engineering"
]


# ============================================================
# SUBJECT OPTIONS
# ============================================================

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
        "Design and Engineering",
        "Engineering Drawing"
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
        "Programming Lab",
        "Constitution of India"
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
        "Data Structures Lab",
        "Microprocessors"
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
        "Database Lab",
        "Computer Networks Lab"
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
        "Elective I",
        "Mini Project"
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
        "Project Phase I",
        "Seminar"
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
        "Major Project",
        "Industrial Training"
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
        "Technical Seminar",
        "Major Project Viva"
    ]
}


# ============================================================
# TEACHER ACCOUNTS
# ============================================================
# Demo passwords. Change these for actual deployment.
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

    "teacher_aero": {
        "password": "ktutech",
        "branch": "Aeronautical Engineering"
    },

    "teacher_bio": {
        "password": "ktutech",
        "branch": "Biotechnology"
    },

    "teacher_biomed": {
        "password": "ktutech",
        "branch": "Biomedical Engineering"
    },

    "teacher_robotics": {
        "password": "ktutech",
        "branch": "Robotics and Automation"
    },

    "teacher_mechatronics": {
        "password": "ktutech",
        "branch": "Mechatronics Engineering"
    },

    "teacher_chemical": {
        "password": "ktutech",
        "branch": "Chemical Engineering"
    },

    "teacher_food": {
        "password": "ktutech",
        "branch": "Food Technology"
    }
}


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

if "teacher_branch" not in st.session_state:
    st.session_state.teacher_branch = None


# ============================================================
# PERFORMANCE COLOURS
# ============================================================

PERFORMANCE_COLORS = {
    "Low Performance": "#F4CCCC",
    "Average Performance": "#FCE5CD",
    "Above Average": "#FFF2CC",
    "Good Performance": "#D9EAD3"
}

PERFORMANCE_TEXT_COLORS = {
    "Low Performance": "#CC0000",
    "Average Performance": "#E69138",
    "Above Average": "#C9A227",
    "Good Performance": "#38761D"
}


# ============================================================
# COLOURED CIRCLE FOR PDF
# ============================================================

def performance_circle(level):

    hex_colour = PERFORMANCE_TEXT_COLORS.get(
        level,
        "#777777"
    )

    drawing = Drawing(
        16,
        16
    )

    drawing.add(
        Circle(
            8,
            8,
            5,
            fillColor=colors.HexColor(hex_colour),
            strokeColor=colors.HexColor(hex_colour)
        )
    )

    return drawing


# ============================================================
# ATTENDANCE CONVERSION
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

    else:
        return 0


def attendance_range(attendance):

    if attendance >= 90:
        return "90–100% → 5 marks"

    elif attendance >= 80:
        return "80–89% → 4 marks"

    elif attendance >= 70:
        return "70–79% → 3 marks"

    elif attendance >= 60:
        return "60–69% → 2 marks"

    elif attendance >= 10:
        return "10–59% → 1 mark"

    else:
        return "Below 10% → 0 marks"


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

    # Attendance converted to 5 marks
    attendance_score = (
        attendance_mark(attendance) / 5
    ) * 100

    # Study hours maximum 6
    study_score = min(
        (study_hours / 6) * 100,
        100
    )

    # Internal /40
    internal_score = (
        internal / 40
    ) * 100

    # Assignment /15
    assignment_score = (
        assignment / 15
    ) * 100

    # Previous /60
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

    # Project-defined performance categories
    if overall < 50:

        level = "Low Performance"

    elif overall < 65:

        level = "Average Performance"

    elif overall < 80:

        level = "Above Average"

    else:

        level = "Good Performance"

    return (
        level,
        overall
    )


# ============================================================
# RECOMMENDATIONS / COMPLEMENT
# ============================================================

def get_feedback(
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
    level
):

    methods = []

    # Low performance
    if level == "Low Performance":

        if attendance < 75:
            methods.append(
                "Improve attendance by attending classes regularly."
            )

        if study_hours < 2:
            methods.append(
                "Increase daily study time gradually."
            )

        if internal < 20:
            methods.append(
                "Prepare more consistently for internal examinations."
            )

        if assignment < 8:
            methods.append(
                "Complete assignments on time."
            )

        if previous < 30:
            methods.append(
                "Revise previous topics and strengthen fundamentals."
            )

        if not methods:
            methods.append(
                "Follow a regular study timetable and revise weak topics."
            )

    # Average
    elif level == "Average Performance":

        methods.append(
            "Maintain regular revision and improve consistency."
        )

        if attendance < 80:
            methods.append(
                "Try to improve attendance."
            )

        if internal < 25:
            methods.append(
                "Give additional attention to internal examinations."
            )

    # Above average
    elif level == "Above Average":

        methods.append(
            "Good progress. Small improvements in marks and internal assessment can improve the overall result."
        )

        if internal < 32:
            methods.append(
                "Improve internal marks through regular revision."
            )

        if assignment < 12:
            methods.append(
                "Try to improve assignment scores."
            )

    # Good
    else:

        methods.append(
            "Excellent performance! Keep up the good work."
        )

        methods.append(
            "Maintain your current study habits and consistency."
        )

    return methods


# ============================================================
# DATASET
# ============================================================

def load_dataset():

    files = [
        "data/student_performance.csv",
        "student_performance.csv"
    ]

    for file in files:

        if os.path.exists(file):

            try:

                df = pd.read_csv(file)

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

    return None


# ============================================================
# FALLBACK TRAINING DATA
# ============================================================

def create_training_data():

    rng = np.random.default_rng(42)

    n = 300

    attendance = rng.uniform(
        40,
        100,
        n
    )

    study = rng.uniform(
        0,
        6,
        n
    )

    internal = rng.uniform(
        8,
        40,
        n
    )

    assignment = rng.uniform(
        3,
        15,
        n
    )

    previous = rng.uniform(
        15,
        60,
        n
    )

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


# ============================================================
# PREPARE MODEL DATA
# ============================================================

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


# ============================================================
# TRAIN K-MEANS + RANDOM FOREST
# ============================================================

@st.cache_resource
def train_models():

    df = load_dataset()

    if df is None:
        df = create_training_data()

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

    # Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # K-MEANS CLUSTERING
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(
        X_scaled
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

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
        scaler,
        df
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_student(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    (
        random_forest,
        kmeans,
        scaler,
        dataset
    ) = train_models()

    student = np.array([[
        attendance,
        (study_hours / 6) * 100,
        (internal / 40) * 100,
        (assignment / 15) * 100,
        (previous / 60) * 100
    ]])

    student_scaled = scaler.transform(
        student
    )

    # K-MEANS
    cluster = int(
        kmeans.predict(
            student_scaled
        )[0]
    )

    # RANDOM FOREST
    prediction = random_forest.predict(
        student_scaled
    )[0]

    probabilities = random_forest.predict_proba(
        student_scaled
    )[0]

    confidence = float(
        np.max(probabilities) * 100
    )

    return (
        prediction,
        confidence,
        cluster
    )


# ============================================================
# REPORT STORAGE
# ============================================================

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

        df = pd.read_csv(
            REPORT_FILE
        )

        for column in REPORT_COLUMNS:

            if column not in df.columns:

                df[column] = ""

        return df

    except Exception:

        return pd.DataFrame(
            columns=REPORT_COLUMNS
        )


def save_reports(df):

    df.to_csv(
        REPORT_FILE,
        index=False
    )


# ============================================================
# PDF STYLES
# ============================================================

def create_pdf_styles():

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=10
    )

    heading = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8
    )

    subheading = ParagraphStyle(
        "CustomSubHeading",
        parent=styles["Heading3"],
        fontSize=10,
        spaceBefore=8,
        spaceAfter=5
    )

    normal = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11
    )

    small = ParagraphStyle(
        "CustomSmall",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10
    )

    return (
        title,
        heading,
        subheading,
        normal,
        small
    )


# ============================================================
# COLOURED STUDENT PDF
# ============================================================

def generate_student_pdf(student):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28
    )

    (
        title_style,
        heading_style,
        subheading_style,
        normal_style,
        small_style
    ) = create_pdf_styles()

    story = []

    # ========================================================
    # 1. TITLE
    # ========================================================

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Based Student Performance Prediction System",
            normal_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # ========================================================
    # 2. STUDENT DETAILS
    # ========================================================

    story.append(
        Paragraph(
            "1. Student Details",
            heading_style
        )
    )

    student_details = [

        [
            "Student Name",
            str(student["Student_Name"])
        ],

        [
            "University ID",
            str(student["University_ID"])
        ],

        [
            "Semester",
            str(student["Semester"])
        ],

        [
            "Branch",
            str(student["Branch"])
        ],

        [
            "Report Generated",
            str(student["Created_Time"])
        ]
    ]

    student_table = Table(
        student_details,
        colWidths=[125, 385]
    )

    student_table.setStyle(
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
                colors.HexColor("#E8EEF7")
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
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            )
        ])
    )

    story.append(
        student_table
    )

    story.append(
        Spacer(1, 15)
    )

    subjects = json.loads(
        student["Subjects_JSON"]
    )

    # ========================================================
    # 3. OVERALL INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "2. Overall Information and Marks",
            heading_style
        )
    )

    total_internal = sum(
        float(x["Internal"])
        for x in subjects
    )

    total_assignment = sum(
        float(x["Assignment"])
        for x in subjects
    )

    total_previous = sum(
        float(x["Previous"])
        for x in subjects
    )

    total_attendance_mark = sum(
        int(x["Attendance_Mark"])
        for x in subjects
    )

    total_possible_internal = (
        len(subjects) * 40
    )

    total_possible_assignment = (
        len(subjects) * 15
    )

    total_possible_previous = (
        len(subjects) * 60
    )

    average_overall = np.mean([
        x["Overall"]
        for x in subjects
    ])

    overall_data = [

        [
            "Number of Subjects",
            str(len(subjects))
        ],

        [
            "Total Internal",
            f"{total_internal:.0f} / "
            f"{total_possible_internal}"
        ],

        [
            "Total Assignment",
            f"{total_assignment:.0f} / "
            f"{total_possible_assignment}"
        ],

        [
            "Total Previous Mark",
            f"{total_previous:.0f} / "
            f"{total_possible_previous}"
        ],

        [
            "Attendance Converted Total",
            f"{total_attendance_mark} / "
            f"{len(subjects) * 5}"
        ],

        [
            "Overall Performance Score",
            f"{average_overall:.2f}%"
        ]
    ]

    overall_table = Table(
        overall_data,
        colWidths=[230, 280]
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
                colors.HexColor("#EAF4EA")
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
                8.5
            )
        ])
    )

    story.append(
        overall_table
    )

    story.append(
        Spacer(1, 15)
    )

    # ========================================================
    # 4. SUBJECT-WISE PERFORMANCE
    # ========================================================

    story.append(
        Paragraph(
            "3. Subject-wise Performance",
            heading_style
        )
    )

    subject_table_data = [

        [
            "Subject",
            "Indicator",
            "Attendance",
            "Study",
            "Internal",
            "Assignment",
            "Previous",
            "Score"
        ]
    ]

    subject_table_objects = []

    for subject in subjects:

        circle = performance_circle(
            subject["Level"]
        )

        subject_table_data.append([

            subject["Subject"],

            circle,

            f'{subject["Attendance"]:.0f}%',

            f'{subject["Study_Hours"]:.1f}/6',

            f'{subject["Internal"]:.0f}/40',

            f'{subject["Assignment"]:.0f}/15',

            f'{subject["Previous"]:.0f}/60',

            f'{subject["Overall"]:.1f}%'
        ])

        subject_table_objects.append(
            subject
        )

    subject_table = Table(
        subject_table_data,
        repeatRows=1,
        colWidths=[
            105,
            30,
            50,
            42,
            55,
            58,
            55,
            55
        ]
    )

    table_commands = [

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
            6.8
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE"
        )
    ]

    for row_index, subject in enumerate(
        subject_table_objects,
        start=1
    ):

        table_commands.append(
            (
                "BACKGROUND",
                (0, row_index),
                (-1, row_index),
                colors.HexColor(
                    PERFORMANCE_COLORS[
                        subject["Level"]
                    ]
                )
            )
        )

    subject_table.setStyle(
        TableStyle(table_commands)
    )

    story.append(
        subject_table
    )

    story.append(
        Spacer(1, 15)
    )

    # ========================================================
    # 5. ASSESSMENT INFORMATION & CONVERSION
    # ========================================================

    story.append(
        Paragraph(
            "4. Assessment Information and Conversion",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "Assessment limits used in this project:",
            normal_style
        )
    )

    assessment_info = [

        ["Assessment", "Maximum"],

        ["Attendance", "100%"],

        ["Study Hours", "6 hours/day"],

        ["Internal Mark", "40"],

        ["Assignment Score", "15"],

        ["Previous Mark", "60"]
    ]

    assessment_table = Table(
        assessment_info,
        colWidths=[250, 260]
    )

    assessment_table.setStyle(
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

    story.append(
        assessment_table
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "Attendance Conversion",
            subheading_style
        )
    )

    conversion_table = Table(

        [
            ["Attendance", "Converted Mark"],

            ["90–100%", "5"],

            ["80–89%", "4"],

            ["70–79%", "3"],

            ["60–69%", "2"],

            ["10–59%", "1"],

            ["Below 10%", "0"]
        ],

        colWidths=[250, 260]
    )

    conversion_table.setStyle(
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

    story.append(
        conversion_table
    )

    story.append(
        Spacer(1, 15)
    )

    # ========================================================
    # 6. EACH SUBJECT SEPARATELY
    # ========================================================

    story.append(
        Paragraph(
            "5. Detailed Subject-wise Assessment",
            heading_style
        )
    )

    for index, subject in enumerate(
        subjects,
        start=1
    ):

        level = subject["Level"]

        background = colors.HexColor(
            PERFORMANCE_COLORS[level]
        )

        subject_header = Table(

            [[

                f"{index}. {subject['Subject']}",

                performance_circle(level),

                level

            ]],

            colWidths=[330, 35, 145]
        )

        subject_header.setStyle(
            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    background
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
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
                    9
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
            subject_header
        )

        story.append(
            Spacer(1, 4)
        )

        details = [

            [
                "Attendance",
                f'{subject["Attendance"]:.0f}%'
            ],

            [
                "Attendance Conversion",
                f'{subject["Attendance_Mark"]}/5'
            ],

            [
                "Study Hours",
                f'{subject["Study_Hours"]:.1f}/6 hours'
            ],

            [
                "Internal Mark",
                f'{subject["Internal"]:.0f}/40'
            ],

            [
                "Assignment",
                f'{subject["Assignment"]:.0f}/15'
            ],

            [
                "Previous Mark",
                f'{subject["Previous"]:.0f}/60'
            ],

            [
                "Overall Subject Score",
                f'{subject["Overall"]:.2f}%'
            ]
        ]

        details_table = Table(
            details,
            colWidths=[220, 290]
        )

        details_table.setStyle(
            TableStyle([

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.grey
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#F4F4F4")
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
            details_table
        )

        story.append(
            Spacer(1, 5)
        )

        # ----------------------------------------------------
        # K-MEANS
        # ----------------------------------------------------

        if "Cluster" in subject:

            story.append(
                Paragraph(
                    f'<b>K-Means Cluster:</b> '
                    f'{subject["Cluster"]}',
                    normal_style
                )
            )

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        if "ML_Prediction" in subject:

            story.append(
                Paragraph(
                    f'<b>Random Forest Prediction:</b> '
                    f'{subject["ML_Prediction"]}',
                    normal_style
                )
            )

            story.append(
                Paragraph(
                    f'<b>Prediction Confidence:</b> '
                    f'{subject["Confidence"]:.2f}%',
                    normal_style
                )
            )

        story.append(
            Spacer(1, 4)
        )

        # ----------------------------------------------------
        # FEEDBACK
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "<b>Feedback / Improvement Method:</b>",
                normal_style
            )
        )

        for method in subject[
            "Recommendations"
        ]:

            story.append(
                Paragraph(
                    "• " + method,
                    small_style
                )
            )

        story.append(
            Spacer(1, 10)
        )

    # ========================================================
    # 7. PERFORMANCE LEGEND
    # ========================================================

    story.append(
        Paragraph(
            "6. Performance Indicator",
            heading_style
        )
    )

    legend_rows = [

        [
            performance_circle(
                "Low Performance"
            ),
            "Low Performance",
            "Needs improvement"
        ],

        [
            performance_circle(
                "Average Performance"
            ),
            "Average Performance",
            "Regular improvement required"
        ],

        [
            performance_circle(
                "Above Average"
            ),
            "Above Average",
            "Good progress; incremental improvement"
        ],

        [
            performance_circle(
                "Good Performance"
            ),
            "Good Performance",
            "Excellent; maintain consistency"
        ]
    ]

    legend_table = Table(
        legend_rows,
        colWidths=[35, 170, 305]
    )

    legend_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.grey
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
        legend_table
    )

    story.append(
        Spacer(1, 15)
    )

    # ========================================================
    # 8. NOTE
    # ========================================================

    story.append(
        Paragraph(
            "Note: The performance categories in this project "
            "are project-defined indicators. K-Means cluster "
            "numbers are group identifiers and do not themselves "
            "represent a performance grade. Machine learning "
            "results are intended to support academic monitoring "
            "and should not replace teacher evaluation.",
            small_style
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

def home_page():

    st.title(
        "🎓 Student Performance Prediction"
    )

    st.subheader(
        "Choose Login"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student Login",
            width="stretch"
        ):

            st.session_state.page = (
                "student_login"
            )

            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Teacher Login",
            width="stretch"
        ):

            st.session_state.page = (
                "teacher_login"
            )

            st.rerun()


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.title(
        "🎓 Student Login"
    )

    password = st.text_input(
        "Password",
        type="password",
        max_chars=9
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔐 Login",
            width="stretch"
        ):

            valid = False

            if len(password) == 9:

                if password[:5] == "BTECH":

                    year_text = password[5:]

                    if year_text.isdigit():

                        year = int(
                            year_text
                        )

                        if 2000 <= year <= 2020:

                            valid = True

            if valid:

                st.session_state.logged_in = True

                st.session_state.role = "student"

                st.session_state.username = password

                st.session_state.page = (
                    "student_dashboard"
                )

                st.success(
                    "🔔 Login successful! Welcome to your Student Dashboard."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid student login."
                )

    with col2:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state.page = "home"

            st.rerun()


# ============================================================
# TEACHER LOGIN
# ============================================================

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

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔐 Login",
            width="stretch"
        ):

            if (
                username in TEACHERS
                and TEACHERS[
                    username
                ]["password"] == password
            ):

                st.session_state.logged_in = True

                st.session_state.role = "teacher"

                st.session_state.username = username

                st.session_state.teacher_branch = (
                    TEACHERS[
                        username
                    ]["branch"]
                )

                st.session_state.page = (
                    "teacher_dashboard"
                )

                st.success(
                    "🔔 Login successful! Teacher Dashboard opened."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid teacher username or password."
                )

    with col2:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state.page = "home"

            st.rerun()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    st.title(
        "🎓 Student Dashboard"
    )

    st.success(
        "🔔 Login successful. Enter your academic details below."
    )

    if st.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.page = "home"

        st.rerun()

    st.divider()

    # ========================================================
    # STUDENT DETAILS
    # ========================================================

    st.header(
        "1. Student Details"
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

    # ========================================================
    # SUBJECT SELECTION
    # ========================================================

    st.header(
        "2. Subject Selection"
    )

    st.info(
        "Select exactly 6 subjects."
    )

    selected_subjects = st.multiselect(

        f"Select subjects for {semester}",

        SUBJECTS[semester],

        max_selections=6
    )

    st.write(
        f"Selected subjects: "
        f"**{len(selected_subjects)} / 6**"
    )

    if len(selected_subjects) != 6:

        st.warning(
            "⚠️ You must select exactly 6 subjects."
        )

        return

    # ========================================================
    # SUBJECT INPUT
    # ========================================================

    st.header(
        "3. Subject-wise Assessment Information"
    )

    subject_results = []

    for i, subject in enumerate(
        selected_subjects
    ):

        st.subheader(
            f"{i + 1}. {subject}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                key=f"attendance_{i}"
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=6.0,
                value=2.0,
                step=0.5,
                key=f"study_{i}"
            )

        with c2:

            internal = st.number_input(
                "Internal Mark / 40",
                min_value=0.0,
                max_value=40.0,
                value=20.0,
                step=1.0,
                key=f"internal_{i}"
            )

            assignment = st.number_input(
                "Assignment Score / 15",
                min_value=0.0,
                max_value=15.0,
                value=8.0,
                step=1.0,
                key=f"assignment_{i}"
            )

        with c3:

            previous = st.number_input(
                "Previous Mark / 60",
                min_value=0.0,
                max_value=60.0,
                value=30.0,
                step=1.0,
                key=f"previous_{i}"
            )

            level, overall = calculate_performance(

                attendance,

                study_hours,

                internal,

                assignment,

                previous
            )

            st.write(
                f"Performance: **{level}**"
            )

        feedback = get_feedback(

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

            "Study_Hours": study_hours,

            "Internal": internal,

            "Assignment": assignment,

            "Previous": previous,

            "Level": level,

            "Overall": overall,

            "Attendance_Mark":
                attendance_mark(attendance),

            "Recommendations":
                feedback
        })

        st.divider()

    # ========================================================
    # RUN ML
    # ========================================================

    if st.button(

        "🤖 Run K-Means + Random Forest and Generate Progress Report",

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

        # ----------------------------------------------------
        # MODEL PREDICTIONS
        # ----------------------------------------------------

        for subject in subject_results:

            (
                prediction,
                confidence,
                cluster
            ) = predict_student(

                subject["Attendance"],

                subject["Study_Hours"],

                subject["Internal"],

                subject["Assignment"],

                subject["Previous"]
            )

            subject["ML_Prediction"] = prediction

            subject["Confidence"] = confidence

            subject["Cluster"] = cluster

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        reports = load_reports()

        # Replace previous report of the same student
        reports = reports[
            reports[
                "University_ID"
            ].astype(str)
            != str(university_id)
        ].copy()

        created_time = datetime.now().strftime(
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
                json.dumps(
                    subject_results
                ),

            "Created_Time":
                created_time
        }

        reports = pd.concat(
            [
                reports,
                pd.DataFrame(
                    [new_report]
                )
            ],
            ignore_index=True
        )

        save_reports(
            reports
        )

        # ----------------------------------------------------
        # GENERATE PDF
        # ----------------------------------------------------

        pdf = generate_student_pdf(
            new_report
        )

        st.success(
            "🎉 Progress analysis completed successfully!"
        )

        st.subheader(
            "📊 Overall Result"
        )

        overall_score = np.mean([
            x["Overall"]
            for x in subject_results
        ])

        st.metric(
            "Overall Performance Score",
            f"{overall_score:.2f}%"
        )

        # ----------------------------------------------------
        # RESULT TABLE
        # ----------------------------------------------------

        result_rows = []

        for subject in subject_results:

            result_rows.append([

                subject["Subject"],

                subject["Level"],

                f'{subject["Overall"]:.2f}%',

                subject["ML_Prediction"],

                f'{subject["Confidence"]:.1f}%',

                f'Cluster {subject["Cluster"]}'
            ])

        result_df = pd.DataFrame(

            result_rows,

            columns=[
                "Subject",
                "Performance",
                "Score",
                "Random Forest",
                "Confidence",
                "K-Means"
            ]
        )

        st.dataframe(
            result_df,
            width="stretch"
        )

        st.download_button(

            "📥 Download Complete Coloured Progress Report PDF",

            data=pdf,

            file_name=
                f"{university_id}_Progress_Report.pdf",

            mime="application/pdf",

            width="stretch"
        )


# ============================================================
# TEACHER DASHBOARD
# ============================================================

def teacher_dashboard():

    teacher_branch = (
        st.session_state.teacher_branch
    )

    st.title(
        "👨‍🏫 Teacher Dashboard"
    )

    st.success(
        f"🔔 Logged in successfully. "
        f"Assigned Branch: {teacher_branch}"
    )

    if st.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.teacher_branch = None
        st.session_state.page = "home"

        st.rerun()

    st.divider()

    reports = load_reports()

    # ========================================================
    # BRANCH FILTER
    # ========================================================

    branch_reports = reports[
        reports[
            "Branch"
        ].astype(str)
        == str(teacher_branch)
    ].copy()

    st.header(
        "Student History"
    )

    if branch_reports.empty:

        st.info(
            "No students have submitted reports for this branch."
        )

        return

    st.write(
        f"Total students: **{len(branch_reports)}**"
    )

    # ========================================================
    # STUDENT HISTORY TABLE
    # ========================================================

    display = branch_reports[
        [
            "Student_Name",
            "University_ID",
            "Semester",
            "Created_Time"
        ]
    ].copy()

    st.dataframe(
        display,
        width="stretch"
    )

    # ========================================================
    # SELECT ONE STUDENT
    # ========================================================

    st.subheader(
        "Select Student"
    )

    student_ids = (
        branch_reports[
            "University_ID"
        ]
        .astype(str)
        .tolist()
    )

    selected_id = st.selectbox(
        "University ID",
        student_ids
    )

    selected_rows = branch_reports[
        branch_reports[
            "University_ID"
        ].astype(str)
        == str(selected_id)
    ]

    if selected_rows.empty:
        return

    student = selected_rows.iloc[0]

    st.markdown(
        f"### 👤 {student['Student_Name']}"
    )

    st.write(
        f"**University ID:** {student['University_ID']}"
    )

    st.write(
        f"**Semester:** {student['Semester']}"
    )

    st.write(
        f"**Branch:** {student['Branch']}"
    )

    # ========================================================
    # SUBJECT HISTORY
    # ========================================================

    subjects = json.loads(
        student["Subjects_JSON"]
    )

    rows = []

    for subject in subjects:

        rows.append([

            subject["Subject"],

            subject["Attendance"],

            subject["Study_Hours"],

            subject["Internal"],

            subject["Assignment"],

            subject["Previous"],

            subject["Level"],

            subject.get(
                "ML_Prediction",
                ""
            ),

            subject.get(
                "Cluster",
                ""
            )
        ])

    student_subject_df = pd.DataFrame(

        rows,

        columns=[

            "Subject",

            "Attendance %",

            "Study Hours",

            "Internal /40",

            "Assignment /15",

            "Previous /60",

            "Performance",

            "Random Forest",

            "K-Means Cluster"
        ]
    )

    st.dataframe(
        student_subject_df,
        width="stretch"
    )

    # ========================================================
    # DOWNLOAD PDF
    # ========================================================

    pdf = generate_student_pdf(
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

    # ========================================================
    # DELETE ONE STUDENT HISTORY
    # ========================================================

    st.subheader(
        "🗑️ Delete Student History"
    )

    st.warning(
        f"This deletes the complete saved history of "
        f"{student['Student_Name']} "
        f"({selected_id}), including all 6 subject marks."
    )

    confirm = st.checkbox(
        "I confirm that I want to permanently delete this student's history.",
        key=f"delete_{selected_id}"
    )

    if st.button(
        "🗑️ Delete This Student",
        type="secondary",
        width="stretch"
    ):

        if not confirm:

            st.error(
                "Please confirm deletion."
            )

        else:

            latest = load_reports()

            # ------------------------------------------------
            # Delete ONLY selected student from this branch
            # ------------------------------------------------

            delete_mask = (

                (
                    latest[
                        "University_ID"
                    ].astype(str)
                    == str(selected_id)
                )

                &

                (
                    latest[
                        "Branch"
                    ].astype(str)
                    == str(teacher_branch)
                )
            )

            deleted = int(
                delete_mask.sum()
            )

            latest = latest[
                ~delete_mask
            ].copy()

            save_reports(
                latest
            )

            if deleted > 0:

                st.success(
                    "🗑️ Student history deleted successfully."
                )

                st.toast(
                    "Student history deleted.",
                    icon="🗑️"
                )

                st.rerun()

            else:

                st.error(
                    "Student history was not found."
                )

    # ========================================================
    # BRANCH PDF
    # ========================================================

    st.divider()

    st.subheader(
        "📑 Branch Reports"
    )

    # Individual reports combined into one PDF
    branch_pdf = generate_branch_pdf(
        branch_reports
    ) if "generate_branch_pdf" in globals() else None

    if branch_pdf:

        st.download_button(

            "📥 Download All Branch Reports PDF",

            data=branch_pdf,

            file_name=
                "All_Branch_Student_Reports.pdf",

            mime="application/pdf",

            width="stretch"
        )


# ============================================================
# OPTIONAL BRANCH PDF
# ============================================================

def generate_branch_pdf(reports):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28
    )

    (
        title_style,
        heading_style,
        subheading_style,
        normal_style,
        small_style
    ) = create_pdf_styles()

    story = []

    story.append(
        Paragraph(
            "BRANCH STUDENT PERFORMANCE REPORTS",
            title_style
        )
    )

    for index, (_, student) in enumerate(
        reports.iterrows()
    ):

        story.append(
            Paragraph(
                f"Student: {student['Student_Name']}",
                heading_style
            )
        )

        story.append(
            Paragraph(
                f"University ID: {student['University_ID']}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Semester: {student['Semester']}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Branch: {student['Branch']}",
                normal_style
            )
        )

        subjects = json.loads(
            student["Subjects_JSON"]
        )

        data = [[
            "Subject",
            "Indicator",
            "Attendance",
            "Internal",
            "Assignment",
            "Previous",
            "Performance"
        ]]

        levels = []

        for subject in subjects:

            data.append([
                subject["Subject"],
                performance_circle(
                    subject["Level"]
                ),
                f'{subject["Attendance"]:.0f}%',
                f'{subject["Internal"]:.0f}/40',
                f'{subject["Assignment"]:.0f}/15',
                f'{subject["Previous"]:.0f}/60',
                subject["Level"]
            ])

            levels.append(
                subject["Level"]
            )

        table = Table(
            data,
            repeatRows=1,
            colWidths=[
                120,
                30,
                55,
                65,
                70,
                65,
                105
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
                7
            )
        ]

        for row, level in enumerate(
            levels,
            start=1
        ):

            commands.append(
                (
                    "BACKGROUND",
                    (0, row),
                    (-1, row),
                    colors.HexColor(
                        PERFORMANCE_COLORS[level]
                    )
                )
            )

        table.setStyle(
            TableStyle(commands)
        )

        story.append(
            table
        )

        if index < len(reports) - 1:

            story.append(
                PageBreak()
            )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# MAIN ROUTER
# ============================================================

if st.session_state.page == "home":

    home_page()

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
