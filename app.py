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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# FOLDERS / FILES
# =========================================================

DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

REPORT_FILE = os.path.join(
    DATA_DIR,
    "student_reports.csv"
)


# =========================================================
# B.TECH BRANCHES
# =========================================================

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

    "Biomedical Engineering",

    "Mechanical Engineering",

    "Civil Engineering",

    "Chemical Engineering",

    "Automobile Engineering",

    "Aeronautical Engineering",

    "Food Technology",

    "Biotechnology",

    "Industrial Engineering",

    "Robotics and Automation",

    "Mechatronics Engineering",

    "Production Engineering",

    "Environmental Engineering",

    "Electronics Engineering",

    "Instrumentation Engineering",

    "Computer Engineering"
]


# =========================================================
# SEMESTER SUBJECTS
# =========================================================

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


# =========================================================
# TEACHER ACCOUNTS
# =========================================================
#
# Demo credentials.
# Change these before real use.
#
# =========================================================

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
    }
}


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {

    "page": "home",

    "logged_in": False,

    "role": None,

    "username": None,

    "branch": None
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# REPORT DATABASE COLUMNS
# =========================================================

REPORT_COLUMNS = [

    "Student_Name",

    "University_ID",

    "Semester",

    "Branch",

    "Subjects_JSON",

    "Created_Time"
]


# =========================================================
# LOAD STUDENT DATASET
# =========================================================

def load_dataset():

    files = [

        "data/student_performance.csv",

        "student_performance.csv"
    ]

    for file in files:

        if os.path.exists(file):

            try:

                df = pd.read_csv(file)

                required_columns = [

                    "Attendance",
                    "Study_Hours",
                    "Internal_Mark",
                    "Assignment",
                    "Previous_Mark",
                    "Performance"
                ]

                if all(
                    column in df.columns
                    for column in required_columns
                ):

                    return df

            except Exception:

                pass

    return None


# =========================================================
# CREATE FALLBACK DATASET
# =========================================================

def create_fallback_dataset():

    rng = np.random.default_rng(42)

    number_of_students = 300

    attendance = rng.uniform(
        40,
        100,
        number_of_students
    )

    study_hours = rng.uniform(
        0,
        6,
        number_of_students
    )

    internal = rng.uniform(
        8,
        40,
        number_of_students
    )

    assignment = rng.uniform(
        3,
        15,
        number_of_students
    )

    previous = rng.uniform(
        15,
        60,
        number_of_students
    )

    score = (

        attendance * 0.25

        + (study_hours / 6 * 100) * 0.15

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

        "Study_Hours": study_hours,

        "Internal_Mark": internal,

        "Assignment": assignment,

        "Previous_Mark": previous,

        "Performance": performance
    })


# =========================================================
# PREPARE DATASET
# =========================================================

def prepare_dataset(df):

    result = df.copy()

    result["Attendance_Model"] = pd.to_numeric(
        result["Attendance"],
        errors="coerce"
    ).fillna(0)

    study = pd.to_numeric(
        result["Study_Hours"],
        errors="coerce"
    ).fillna(0)

    if study.max() <= 10:

        result["Study_Model"] = (
            study / 6
        ) * 100

    else:

        result["Study_Model"] = study

    internal = pd.to_numeric(
        result["Internal_Mark"],
        errors="coerce"
    ).fillna(0)

    if internal.max() <= 40:

        result["Internal_Model"] = (
            internal / 40
        ) * 100

    else:

        result["Internal_Model"] = internal

    assignment = pd.to_numeric(
        result["Assignment"],
        errors="coerce"
    ).fillna(0)

    if assignment.max() <= 15:

        result["Assignment_Model"] = (
            assignment / 15
        ) * 100

    else:

        result["Assignment_Model"] = assignment

    previous = pd.to_numeric(
        result["Previous_Mark"],
        errors="coerce"
    ).fillna(0)

    if previous.max() <= 60:

        result["Previous_Model"] = (
            previous / 60
        ) * 100

    else:

        result["Previous_Model"] = previous

    return result


# =========================================================
# TRAIN K-MEANS + RANDOM FOREST
# =========================================================

@st.cache_resource
def train_models():

    dataset = load_dataset()

    if dataset is None:

        dataset = create_fallback_dataset()

    dataset = prepare_dataset(dataset)

    feature_columns = [

        "Attendance_Model",

        "Study_Model",

        "Internal_Model",

        "Assignment_Model",

        "Previous_Model"
    ]

    X = dataset[
        feature_columns
    ].fillna(0)

    y = dataset[
        "Performance"
    ].astype(str)

    # -----------------------------------------------------
    # STANDARD SCALING
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # K-MEANS CLUSTERING
    # -----------------------------------------------------

    kmeans = KMeans(

        n_clusters=3,

        random_state=42,

        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        X_scaled
    )

    dataset["Cluster"] = cluster_labels

    # -----------------------------------------------------
    # RANDOM FOREST CLASSIFIER
    # -----------------------------------------------------

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
        dataset
    )


# =========================================================
# K-MEANS + RANDOM FOREST PREDICTION
# =========================================================

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


# =========================================================
# ATTENDANCE CONVERSION
# =========================================================

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


# =========================================================
# PERFORMANCE ANALYSIS
# =========================================================

def analyse_performance(

    attendance,

    study_hours,

    internal,

    assignment,

    previous

):

    attendance_score = (
        attendance_mark(attendance)
        / 5
    ) * 100

    study_score = min(
        study_hours / 6 * 100,
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

    if overall < 50:

        level = "Low Performance"

        indicator = "LOW"

    elif overall < 65:

        level = "Average Performance"

        indicator = "AVERAGE"

    elif overall < 80:

        level = "Above Average"

        indicator = "ABOVE AVERAGE"

    else:

        level = "Good Performance"

        indicator = "GOOD"

    return (

        level,

        indicator,

        overall
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

def get_recommendations(

    attendance,

    study_hours,

    internal,

    assignment,

    previous,

    level

):

    recommendations = []

    if attendance < 75:

        recommendations.append(
            "Improve attendance and attend classes regularly."
        )

    if study_hours < 2:

        recommendations.append(
            "Increase daily study time and follow a regular timetable."
        )

    if internal < 20:

        recommendations.append(
            "Focus more on internal examination preparation."
        )

    if assignment < 8:

        recommendations.append(
            "Complete assignments regularly and improve assignment quality."
        )

    if previous < 30:

        recommendations.append(
            "Revise previous topics and strengthen fundamental concepts."
        )

    if not recommendations:

        if level == "Good Performance":

            recommendations.append(
                "Excellent performance. Maintain your current consistency."
            )

        elif level == "Above Average":

            recommendations.append(
                "Good progress. Small improvements can help reach the good-performance level."
            )

        elif level == "Average Performance":

            recommendations.append(
                "Maintain regular revision and improve consistency."
            )

        else:

            recommendations.append(
                "Create a regular study plan and focus on weak areas."
            )

    return recommendations


# =========================================================
# LOAD REPORTS
# =========================================================

def load_reports():

    if not os.path.exists(
        REPORT_FILE
    ):

        return pd.DataFrame(
            columns=REPORT_COLUMNS
        )

    try:

        reports = pd.read_csv(
            REPORT_FILE
        )

        for column in REPORT_COLUMNS:

            if column not in reports.columns:

                reports[column] = ""

        return reports

    except Exception:

        return pd.DataFrame(
            columns=REPORT_COLUMNS
        )


# =========================================================
# SAVE REPORTS
# =========================================================

def save_reports(reports):

    reports.to_csv(
        REPORT_FILE,
        index=False
    )


# =========================================================
# PDF COLOUR SETTINGS
# =========================================================

PDF_COLOURS = {

    "Low Performance":
        colors.HexColor("#FFCCCC"),

    "Average Performance":
        colors.HexColor("#FFDAB9"),

    "Above Average":
        colors.HexColor("#FFF2A8"),

    "Good Performance":
        colors.HexColor("#C6EFCE")
}


# =========================================================
# GENERATE COLOURED STUDENT PDF
# =========================================================

def generate_student_pdf(student):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=30,

        leftMargin=30,

        topMargin=30,

        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=18,

        spaceAfter=12
    )

    heading_style = ParagraphStyle(

        "ReportHeading",

        parent=styles["Heading2"],

        fontSize=13,

        spaceBefore=12,

        spaceAfter=8
    )

    normal_style = ParagraphStyle(

        "ReportNormal",

        parent=styles["Normal"],

        fontSize=8.5,

        leading=11
    )

    story = []

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------------

    information = [

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
            "Generated Time",
            str(student["Created_Time"])
        ]
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
        info_table
    )

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # K-MEANS EXPLANATION
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "K-Means Clustering Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "K-Means clustering groups students with similar "
            "academic-input patterns based on attendance, study "
            "hours, internal marks, assignment score and previous marks.",
            normal_style
        )
    )

    # -----------------------------------------------------
    # SUBJECTS
    # -----------------------------------------------------

    subjects = json.loads(
        student["Subjects_JSON"]
    )

    story.append(
        Paragraph(
            "Subject-wise Performance",
            heading_style
        )
    )

    table_data = [

        [
            "Subject",
            "Attend.",
            "Study",
            "Internal",
            "Assign.",
            "Previous",
            "Performance"
        ]
    ]

    row_levels = []

    for subject in subjects:

        table_data.append([

            subject["Subject"],

            f'{subject["Attendance"]:.0f}%',

            f'{subject["Study_Hours"]:.1f}',

            f'{subject["Internal"]:.0f}/40',

            f'{subject["Assignment"]:.0f}/15',

            f'{subject["Previous"]:.0f}/60',

            subject["Level"]
        ])

        row_levels.append(
            subject["Level"]
        )

    subject_table = Table(

        table_data,

        repeatRows=1,

        colWidths=[130, 48, 45, 55, 55, 55, 100]
    )

    style_commands = [

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
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE"
        )
    ]

    # -----------------------------------------------------
    # COLOUR EACH PERFORMANCE ROW
    # -----------------------------------------------------

    for row_number, level in enumerate(
        row_levels,
        start=1
    ):

        row_colour = PDF_COLOURS.get(

            level,

            colors.white
        )

        style_commands.append(

            (
                "BACKGROUND",

                (0, row_number),

                (-1, row_number),

                row_colour
            )
        )

    subject_table.setStyle(
        TableStyle(
            style_commands
        )
    )

    story.append(
        subject_table
    )

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # DETAILED ANALYSIS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Detailed Subject Analysis",
            heading_style
        )
    )

    for subject in subjects:

        level = subject["Level"]

        level_colour = PDF_COLOURS.get(
            level,
            colors.white
        )

        # Subject heading box

        heading_table = Table([

            [
                subject["Subject"],
                level
            ]

        ], colWidths=[360, 130])

        heading_table.setStyle(
            TableStyle([

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
                    level_colour
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica-Bold"
                ),

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
            ])
        )

        story.append(
            heading_table
        )

        story.append(
            Spacer(1, 4)
        )

        story.append(
            Paragraph(
                f'Overall Score: {subject["Overall"]:.2f}%',
                normal_style
            )
        )

        story.append(
            Paragraph(
                f'Attendance Conversion: '
                f'{subject["Attendance_Mark"]}/5',
                normal_style
            )
        )

        if "ML_Prediction" in subject:

            story.append(
                Paragraph(
                    f'Machine Learning Prediction: '
                    f'{subject["ML_Prediction"]}',
                    normal_style
                )
            )

            story.append(
                Paragraph(
                    f'Random Forest Confidence: '
                    f'{subject["Confidence"]:.2f}%',
                    normal_style
                )
            )

            story.append(
                Paragraph(
                    f'K-Means Cluster: '
                    f'{subject["Cluster"]}',
                    normal_style
                )
            )

        story.append(
            Paragraph(
                "<b>Recommendations:</b>",
                normal_style
            )
        )

        for recommendation in subject[
            "Recommendations"
        ]:

            story.append(
                Paragraph(
                    "• " + recommendation,
                    normal_style
                )
            )

        story.append(
            Spacer(1, 8)
        )

    # -----------------------------------------------------
    # LEGEND
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Performance Indicator",
            heading_style
        )
    )

    legend = [

        [
            "LOW",
            "Low Performance"
        ],

        [
            "AVERAGE",
            "Average Performance"
        ],

        [
            "ABOVE AVERAGE",
            "Above Average"
        ],

        [
            "GOOD",
            "Good Performance"
        ]
    ]

    legend_table = Table(

        legend,

        colWidths=[100, 200]
    )

    legend_style = [

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.4,
            colors.grey
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

    for index, (_, level_name) in enumerate(
        legend
    ):

        legend_style.append(

            (
                "BACKGROUND",
                (0, index),
                (-1, index),
                PDF_COLOURS[level_name]
            )
        )

    legend_table.setStyle(
        TableStyle(
            legend_style
        )
    )

    story.append(
        legend_table
    )

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # NOTE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Note: The performance levels and score ranges used "
            "in this project are project-defined indicators. "
            "The ML prediction is intended to support academic "
            "monitoring and should not replace teacher evaluation.",
            normal_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# GENERATE ALL BRANCH REPORTS PDF
# =========================================================

def generate_all_reports_pdf(
    reports
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=30,

        leftMargin=30,

        topMargin=30,

        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(

        "AllTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=17,

        spaceAfter=15
    )

    normal_style = ParagraphStyle(

        "AllNormal",

        parent=styles["Normal"],

        fontSize=8
    )

    story = []

    story.append(
        Paragraph(
            "BRANCH STUDENT PROGRESS REPORTS",
            title_style
        )
    )

    for index, (_, student) in enumerate(
        reports.iterrows()
    ):

        story.append(
            Paragraph(
                f'<b>Student:</b> '
                f'{student["Student_Name"]}',
                normal_style
            )
        )

        story.append(
            Paragraph(
                f'<b>University ID:</b> '
                f'{student["University_ID"]}',
                normal_style
            )
        )

        story.append(
            Paragraph(
                f'<b>Semester:</b> '
                f'{student["Semester"]}',
                normal_style
            )
        )

        story.append(
            Paragraph(
                f'<b>Branch:</b> '
                f'{student["Branch"]}',
                normal_style
            )
        )

        story.append(
            Spacer(1, 8)
        )

        subjects = json.loads(
            student["Subjects_JSON"]
        )

        data = [

            [
                "Subject",
                "Attend.",
                "Internal",
                "Assign.",
                "Previous",
                "Performance"
            ]
        ]

        levels = []

        for subject in subjects:

            data.append([

                subject["Subject"],

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
                155,
                55,
                60,
                60,
                60,
                110
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
                    PDF_COLOURS.get(
                        level,
                        colors.white
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


# =========================================================
# HOME PAGE
# =========================================================

def home_page():

    st.title(
        "🎓 Student Performance Prediction"
    )

    st.subheader(
        "Choose Login"
    )

    left, right = st.columns(2)

    with left:

        if st.button(
            "🎓 Student Login",
            width="stretch"
        ):

            st.session_state.page = (
                "student_login"
            )

            st.rerun()

    with right:

        if st.button(
            "👨‍🏫 Teacher Login",
            width="stretch"
        ):

            st.session_state.page = (
                "teacher_login"
            )

            st.rerun()


# =========================================================
# STUDENT LOGIN
# =========================================================

def student_login():

    st.title(
        "🎓 Student Login"
    )

    password = st.text_input(

        "Password",

        type="password",

        max_chars=9
    )

    left, right = st.columns(2)

    with left:

        if st.button(
            "🔐 Login",
            width="stretch"
        ):

            valid = False

            if len(password) == 9:

                if password.startswith(
                    "BTECH"
                ):

                    year_text = password[5:]

                    if year_text.isdigit():

                        year = int(
                            year_text
                        )

                        if (
                            2000
                            <= year
                            <= 2020
                        ):

                            valid = True

            if valid:

                st.session_state.logged_in = True

                st.session_state.role = (
                    "student"
                )

                st.session_state.username = (
                    password
                )

                st.session_state.page = (
                    "student_dashboard"
                )

                st.success(
                    "🔔 Login successful! Welcome Student."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid login."
                )

    with right:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state.page = (
                "home"
            )

            st.rerun()


# =========================================================
# TEACHER LOGIN
# =========================================================

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

    left, right = st.columns(2)

    with left:

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

                st.session_state.role = (
                    "teacher"
                )

                st.session_state.username = (
                    username
                )

                st.session_state.branch = (
                    TEACHERS[
                        username
                    ]["branch"]
                )

                st.session_state.page = (
                    "teacher_dashboard"
                )

                st.success(
                    "🔔 Login successful! Welcome Teacher."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    with right:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state.page = (
                "home"
            )

            st.rerun()


# =========================================================
# STUDENT DASHBOARD
# =========================================================

def student_dashboard():

    st.title(
        "🎓 Student Dashboard"
    )

    st.success(
        "🔔 You are logged in successfully."
    )

    if st.button(
        "🚪 Logout"
    ):

        st.session_state.logged_in = False

        st.session_state.role = None

        st.session_state.username = None

        st.session_state.page = "home"

        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    st.subheader(
        "👤 Student Details"
    )

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input(
            "Student Name"
        )

        university_id = st.text_input(
            "University ID"
        )

    with col2:

        semester = st.selectbox(

            "Semester",

            [
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
                "S6",
                "S7",
                "S8"
            ]
        )

        branch = st.selectbox(
            "B.Tech Branch",
            BRANCHES
        )

    # -----------------------------------------------------
    # SUBJECT SELECTION
    # -----------------------------------------------------

    st.subheader(
        "📚 Subject Selection"
    )

    st.info(
        "Select exactly 6 subjects."
    )

    selected_subjects = st.multiselect(

        f"Subjects for {semester}",

        SUBJECTS[semester],

        max_selections=6
    )

    st.write(
        f"Selected: {len(selected_subjects)} / 6"
    )

    if len(selected_subjects) != 6:

        st.warning(
            "⚠️ Exactly 6 subjects must be selected."
        )

        return

    # -----------------------------------------------------
    # SUBJECT DETAILS
    # -----------------------------------------------------

    subject_results = []

    st.divider()

    st.subheader(
        "📊 Subject-wise Academic Details"
    )

    for index, subject in enumerate(
        selected_subjects
    ):

        st.markdown(
            f"### {index + 1}. {subject}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            attendance = st.number_input(

                "Attendance (%)",

                min_value=0.0,

                max_value=100.0,

                value=75.0,

                step=1.0,

                key=f"attendance_{index}"
            )

            study_hours = st.number_input(

                "Study Hours / Day",

                min_value=0.0,

                max_value=6.0,

                value=2.0,

                step=0.5,

                key=f"study_{index}"
            )

        with c2:

            internal = st.number_input(

                "Internal Mark / 40",

                min_value=0.0,

                max_value=40.0,

                value=20.0,

                step=1.0,

                key=f"internal_{index}"
            )

            assignment = st.number_input(

                "Assignment Score / 15",

                min_value=0.0,

                max_value=15.0,

                value=8.0,

                step=1.0,

                key=f"assignment_{index}"
            )

        with c3:

            previous = st.number_input(

                "Previous Mark / 60",

                min_value=0.0,

                max_value=60.0,

                value=30.0,

                step=1.0,

                key=f"previous_{index}"
            )

            level, indicator, overall = (
                analyse_performance(
                    attendance,
                    study_hours,
                    internal,
                    assignment,
                    previous
                )
            )

            st.metric(
                "Current Level",
                level
            )

        recommendations = get_recommendations(

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

            "Indicator": indicator,

            "Overall": overall,

            "Attendance_Mark":
                attendance_mark(attendance),

            "Recommendations":
                recommendations
        })

        st.divider()

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(

        "🤖 Run K-Means + Random Forest & Generate PDF",

        type="primary",

        width="stretch"
    ):

        if not student_name.strip():

            st.error(
                "Please enter Student Name."
            )

            return

        if not university_id.strip():

            st.error(
                "Please enter University ID."
            )

            return

        # -------------------------------------------------
        # ML
        # -------------------------------------------------

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

            subject["ML_Prediction"] = (
                prediction
            )

            subject["Confidence"] = (
                confidence
            )

            subject["Cluster"] = (
                cluster
            )

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        reports = load_reports()

        # Replace existing report for same University ID
        reports = reports[
            reports["University_ID"].astype(str)
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

        # -------------------------------------------------
        # PDF
        # -------------------------------------------------

        pdf = generate_student_pdf(
            new_report
        )

        st.success(
            "🔔 Analysis completed and coloured PDF generated successfully!"
        )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.subheader(
            "📊 Final Results"
        )

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

        results_df = pd.DataFrame(

            result_rows,

            columns=[

                "Subject",

                "Performance",

                "Score",

                "Random Forest",

                "Confidence",

                "K-Means Cluster"
            ]
        )

        st.dataframe(
            results_df,
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


# =========================================================
# TEACHER DASHBOARD
# =========================================================

def teacher_dashboard():

    teacher_branch = (
        st.session_state.branch
    )

    st.title(
        "👨‍🏫 Teacher Dashboard"
    )

    st.success(
        f"🔔 Login successful. "
        f"Branch: {teacher_branch}"
    )

    if st.button(
        "🚪 Logout"
    ):

        st.session_state.logged_in = False

        st.session_state.role = None

        st.session_state.username = None

        st.session_state.branch = None

        st.session_state.page = "home"

        st.rerun()

    st.divider()

    reports = load_reports()

    # -----------------------------------------------------
    # BRANCH FILTER
    # -----------------------------------------------------

    branch_reports = reports[
        reports["Branch"].astype(str)
        == str(teacher_branch)
    ].copy()

    st.subheader(
        "📊 Students in Your Branch"
    )

    if branch_reports.empty:

        st.info(
            "No student reports are currently available."
        )

        return

    st.write(
        f"Total students: **{len(branch_reports)}**"
    )

    # -----------------------------------------------------
    # STUDENT TABLE
    # -----------------------------------------------------

    student_table = branch_reports[
        [
            "Student_Name",
            "University_ID",
            "Semester",
            "Branch",
            "Created_Time"
        ]
    ].copy()

    st.dataframe(
        student_table,
        width="stretch"
    )

    # -----------------------------------------------------
    # SELECT STUDENT
    # -----------------------------------------------------

    st.subheader(
        "👤 Select Student"
    )

    student_ids = (

        branch_reports[
            "University_ID"
        ]

        .astype(str)

        .tolist()
    )

    selected_id = st.selectbox(

        "Select University ID",

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
        f"**University ID:** "
        f"{student['University_ID']}"
    )

    st.write(
        f"**Semester:** "
        f"{student['Semester']}"
    )

    st.write(
        f"**Branch:** "
        f"{student['Branch']}"
    )

    # -----------------------------------------------------
    # SUBJECT DETAILS
    # -----------------------------------------------------

    subjects = json.loads(
        student["Subjects_JSON"]
    )

    subject_rows = []

    for subject in subjects:

        subject_rows.append([

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

    subject_df = pd.DataFrame(

        subject_rows,

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
        subject_df,
        width="stretch"
    )

    # -----------------------------------------------------
    # DOWNLOAD INDIVIDUAL PDF
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # DELETE STUDENT
    # -----------------------------------------------------

    st.subheader(
        "🗑️ Delete Selected Student"
    )

    st.warning(
        "This permanently removes the selected student's "
        "saved details, marks and progress report from "
        "the application's report database."
    )

    confirm_delete = st.checkbox(

        "I confirm that I want to delete this student's details and marks.",

        key=f"delete_confirm_{selected_id}"
    )

    if st.button(

        "🗑️ Delete Selected Student",

        type="secondary",

        width="stretch"
    ):

        if not confirm_delete:

            st.error(
                "Please confirm the deletion first."
            )

        else:

            latest_reports = load_reports()

            # -------------------------------------------------
            # SECURITY:
            # Delete only selected student from teacher's branch
            # -------------------------------------------------

            delete_mask = (

                (
                    latest_reports[
                        "University_ID"
                    ].astype(str)
                    == str(selected_id)
                )

                &

                (
                    latest_reports[
                        "Branch"
                    ].astype(str)
                    == str(teacher_branch)
                )
            )

            deleted_count = int(
                delete_mask.sum()
            )

            latest_reports = latest_reports[
                ~delete_mask
            ].copy()

            save_reports(
                latest_reports
            )

            if deleted_count > 0:

                st.success(
                    "🗑️ Student details and marks deleted successfully."
                )

                st.toast(
                    "Student deleted.",
                    icon="🗑️"
                )

                st.rerun()

            else:

                st.error(
                    "Student could not be deleted."
                )

    # -----------------------------------------------------
    # ALL BRANCH REPORTS
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "📑 Branch Progress Reports"
    )

    all_pdf = generate_all_reports_pdf(
        branch_reports
    )

    st.download_button(

        "📥 Download All Branch Students PDF",

        data=all_pdf,

        file_name=
            "Branch_All_Student_Progress_Reports.pdf",

        mime="application/pdf",

        width="stretch"
    )


# =========================================================
# MAIN APPLICATION ROUTER
# =========================================================

if st.session_state.page == "home":

    home_page()


elif st.session_state.page == "student_login":

    student_login()


elif st.session_state.page == "teacher_login":

    teacher_login()


elif (

    st.session_state.page
    == "student_dashboard"

    and st.session_state.logged_in

    and st.session_state.role
    == "student"

):

    student_dashboard()


elif (

    st.session_state.page
    == "teacher_dashboard"

    and st.session_state.logged_in

    and st.session_state.role
    == "teacher"

):

    teacher_dashboard()


else:

    st.session_state.page = "home"

    st.rerun()
