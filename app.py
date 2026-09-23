import os
import html
import io
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
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
# BRANCHES
# =========================================================

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
    "Civil Engineering"
]


# =========================================================
# SUBJECT DATABASE
#
# This is a project catalogue.
# Update names to exactly match your college's KTU scheme
# if necessary.
# =========================================================

SUBJECTS = {

    # =====================================================
    # AI & DATA SCIENCE
    # =====================================================

    "Artificial Intelligence and Data Science": {

        "S1": [
            "Mathematics for Information Science I",
            "Physics for Information Science",
            "Engineering Graphics",
            "Introduction to Computing",
            "Life Skills",
            "Programming in C",
            "Engineering Chemistry",
            "Basics of Electrical Engineering",
            "Basics of Electronics Engineering",
            "Engineering Workshop",
            "Health and Wellness",
            "Professional Communication"
        ],

        "S2": [
            "Mathematics for Information Science II",
            "Data Structures",
            "Object Oriented Programming",
            "Digital Electronics",
            "Engineering Mechanics",
            "Professional Communication",
            "Database Fundamentals",
            "Environmental Science",
            "Discrete Mathematics",
            "Computer Organization",
            "Python Programming",
            "Design Thinking"
        ],

        "S3": [
            "Mathematics for Information Science III",
            "Data Structures and Algorithms",
            "Database Management Systems",
            "Artificial Intelligence",
            "Machine Learning",
            "Digital Signal Processing",
            "Computer Organization and Architecture",
            "Probability and Statistics",
            "Operating Systems",
            "Object Oriented Programming",
            "Data Analytics",
            "Professional Ethics"
        ],

        "S4": [
            "Operating Systems",
            "Computer Networks",
            "Machine Learning",
            "Data Mining",
            "Web Programming",
            "Probability and Statistics",
            "Design and Analysis of Algorithms",
            "Software Engineering",
            "Artificial Neural Networks",
            "Database Systems",
            "Research Methodology",
            "Constitution of India"
        ],

        "S5": [
            "Deep Learning",
            "Big Data Analytics",
            "Data Visualization",
            "Natural Language Processing",
            "Cloud Computing",
            "Computer Vision",
            "Artificial Intelligence Laboratory",
            "Machine Learning Laboratory",
            "Data Science Laboratory",
            "Professional Elective I",
            "Open Elective I",
            "Entrepreneurship"
        ],

        "S6": [
            "Reinforcement Learning",
            "Big Data Technologies",
            "Cloud Computing",
            "Computer Vision",
            "Natural Language Processing",
            "Data Engineering",
            "AI Laboratory",
            "Data Analytics Laboratory",
            "Professional Elective II",
            "Professional Elective III",
            "Open Elective II",
            "Mini Project"
        ],

        "S7": [
            "Advanced Machine Learning",
            "Generative AI",
            "MLOps",
            "Advanced Data Analytics",
            "Distributed Computing",
            "Cyber Security",
            "Project Phase I",
            "Professional Elective IV",
            "Professional Elective V",
            "Open Elective III",
            "Seminar",
            "Industrial Training"
        ],

        "S8": [
            "Project Phase II",
            "Major Project",
            "Professional Elective VI",
            "Professional Elective VII",
            "Open Elective IV",
            "Artificial Intelligence Applications",
            "Data Science Applications",
            "Industry Internship",
            "Technical Seminar",
            "Comprehensive Viva"
        ]
    },


    # =====================================================
    # CSE
    # =====================================================

    "Computer Science and Engineering": {

        "S1": [
            "Mathematics I",
            "Physics",
            "Engineering Graphics",
            "Programming in C",
            "Life Skills",
            "Engineering Chemistry",
            "Electrical Engineering",
            "Electronics Engineering",
            "Engineering Workshop",
            "Professional Communication",
            "Computer Fundamentals",
            "Health and Wellness"
        ],

        "S2": [
            "Mathematics II",
            "Data Structures",
            "Object Oriented Programming",
            "Digital Electronics",
            "Engineering Mechanics",
            "Database Fundamentals",
            "Discrete Mathematics",
            "Computer Organization",
            "Python Programming",
            "Environmental Science",
            "Communication Skills",
            "Design Thinking"
        ],

        "S3": [
            "Mathematics III",
            "Data Structures and Algorithms",
            "Database Management Systems",
            "Object Oriented Programming",
            "Digital Logic",
            "Computer Organization",
            "Operating Systems",
            "Probability and Statistics",
            "Software Engineering",
            "Artificial Intelligence",
            "Computer Architecture",
            "Professional Ethics"
        ],

        "S4": [
            "Operating Systems",
            "Computer Networks",
            "Design and Analysis of Algorithms",
            "Microprocessors",
            "Database Systems",
            "Software Engineering",
            "Web Programming",
            "Theory of Computation",
            "Artificial Intelligence",
            "Computer Graphics",
            "Probability and Statistics",
            "Constitution of India"
        ],

        "S5": [
            "Compiler Design",
            "Computer Networks",
            "Distributed Systems",
            "Cloud Computing",
            "Machine Learning",
            "Web Technologies",
            "Cyber Security",
            "Software Testing",
            "Professional Elective I",
            "Professional Elective II",
            "Open Elective I",
            "Mini Project"
        ],

        "S6": [
            "Machine Learning",
            "Cloud Computing",
            "Computer Security",
            "Distributed Computing",
            "Mobile Computing",
            "Internet of Things",
            "Data Mining",
            "Professional Elective III",
            "Professional Elective IV",
            "Open Elective II",
            "Seminar",
            "Mini Project"
        ],

        "S7": [
            "Deep Learning",
            "Blockchain",
            "Natural Language Processing",
            "Advanced Computer Networks",
            "Cyber Security",
            "MLOps",
            "Project Phase I",
            "Professional Elective V",
            "Professional Elective VI",
            "Open Elective III",
            "Seminar",
            "Industrial Training"
        ],

        "S8": [
            "Major Project",
            "Project Phase II",
            "Professional Elective VII",
            "Professional Elective VIII",
            "Open Elective IV",
            "Advanced AI",
            "Cloud Applications",
            "Industry Internship",
            "Technical Seminar",
            "Comprehensive Viva"
        ]
    },


    # =====================================================
    # CSE AI
    # =====================================================

    "Computer Science and Engineering (AI)": {

        "S1": [
            "Mathematics I",
            "Physics",
            "Engineering Graphics",
            "Programming in C",
            "Engineering Chemistry",
            "Life Skills",
            "Electrical Engineering",
            "Electronics Engineering",
            "Engineering Workshop",
            "Professional Communication"
        ],

        "S2": [
            "Mathematics II",
            "Data Structures",
            "Object Oriented Programming",
            "Digital Electronics",
            "Discrete Mathematics",
            "Database Fundamentals",
            "Computer Organization",
            "Python Programming",
            "Engineering Mechanics",
            "Environmental Science"
        ],

        "S3": [
            "Mathematics III",
            "Data Structures and Algorithms",
            "Database Management Systems",
            "Artificial Intelligence",
            "Machine Learning",
            "Computer Organization",
            "Probability and Statistics",
            "Operating Systems",
            "Object Oriented Programming",
            "Data Analytics"
        ],

        "S4": [
            "Machine Learning",
            "Artificial Intelligence",
            "Computer Networks",
            "Operating Systems",
            "Design and Analysis of Algorithms",
            "Database Systems",
            "Web Programming",
            "Computer Vision",
            "Natural Language Processing",
            "Software Engineering"
        ],

        "S5": [
            "Deep Learning",
            "Computer Vision",
            "Natural Language Processing",
            "Reinforcement Learning",
            "Cloud Computing",
            "Data Mining",
            "AI Laboratory",
            "Machine Learning Laboratory",
            "Professional Elective I",
            "Professional Elective II",
            "Open Elective I"
        ],

        "S6": [
            "Advanced Machine Learning",
            "Generative AI",
            "Robotics",
            "Computer Vision",
            "NLP",
            "MLOps",
            "AI Laboratory",
            "Data Analytics",
            "Professional Elective III",
            "Professional Elective IV",
            "Open Elective II",
            "Mini Project"
        ],

        "S7": [
            "Generative AI",
            "Advanced Deep Learning",
            "MLOps",
            "AI Security",
            "Distributed AI",
            "Project Phase I",
            "Professional Elective V",
            "Professional Elective VI",
            "Open Elective III",
            "Seminar"
        ],

        "S8": [
            "Major Project",
            "Project Phase II",
            "Advanced AI Applications",
            "Professional Elective VII",
            "Professional Elective VIII",
            "Open Elective IV",
            "Industry Internship",
            "Technical Seminar",
            "Comprehensive Viva"
        ]
    }
}


# =========================================================
# COPY SUBJECTS FOR OTHER BRANCHES
# =========================================================

COMMON_BRANCHES = [
    "Computer Science and Engineering (Data Science)",
    "Information Technology",
    "Cyber Security",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electronics and Instrumentation Engineering",
    "Mechanical Engineering",
    "Civil Engineering"
]

for branch in COMMON_BRANCHES:

    SUBJECTS[branch] = {

        "S1": [
            "Engineering Mathematics I",
            "Engineering Physics",
            "Engineering Chemistry",
            "Engineering Graphics",
            "Programming in C",
            "Life Skills",
            "Electrical Engineering",
            "Electronics Engineering",
            "Engineering Workshop",
            "Professional Communication",
            "Environmental Science",
            "Health and Wellness"
        ],

        "S2": [
            "Engineering Mathematics II",
            "Data Structures",
            "Object Oriented Programming",
            "Digital Electronics",
            "Engineering Mechanics",
            "Database Fundamentals",
            "Discrete Mathematics",
            "Python Programming",
            "Professional Communication",
            "Environmental Science",
            "Computer Organization",
            "Design Thinking"
        ],

        "S3": [
            "Engineering Mathematics III",
            "Data Structures and Algorithms",
            "Database Management Systems",
            "Operating Systems",
            "Computer Organization",
            "Object Oriented Programming",
            "Probability and Statistics",
            "Computer Networks",
            "Digital Systems",
            "Professional Ethics",
            "Programming Laboratory",
            "Branch Core I"
        ],

        "S4": [
            "Engineering Mathematics IV",
            "Computer Networks",
            "Operating Systems",
            "Design and Analysis of Algorithms",
            "Software Engineering",
            "Database Systems",
            "Web Programming",
            "Microprocessors",
            "Theory of Computation",
            "Branch Core II",
            "Laboratory",
            "Professional Elective I"
        ],

        "S5": [
            "Machine Learning",
            "Cloud Computing",
            "Data Mining",
            "Cyber Security",
            "Internet of Things",
            "Distributed Systems",
            "Web Technologies",
            "Professional Elective I",
            "Professional Elective II",
            "Open Elective I",
            "Laboratory",
            "Mini Project"
        ],

        "S6": [
            "Artificial Intelligence",
            "Machine Learning",
            "Cloud Computing",
            "Computer Security",
            "Data Analytics",
            "Internet of Things",
            "Distributed Computing",
            "Professional Elective III",
            "Professional Elective IV",
            "Open Elective II",
            "Seminar",
            "Mini Project"
        ],

        "S7": [
            "Advanced Computing",
            "Artificial Intelligence",
            "Cloud Applications",
            "Cyber Security",
            "Advanced Data Analytics",
            "Distributed Computing",
            "Project Phase I",
            "Professional Elective V",
            "Professional Elective VI",
            "Open Elective III",
            "Seminar",
            "Industrial Training"
        ],

        "S8": [
            "Major Project",
            "Project Phase II",
            "Professional Elective VII",
            "Professional Elective VIII",
            "Open Elective IV",
            "Advanced Engineering Applications",
            "Industry Internship",
            "Technical Seminar",
            "Comprehensive Viva"
        ]
    }


# =========================================================
# TEACHER ACCOUNTS
# =========================================================

TEACHERS = {
    "teacher_aids": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Data Science"
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

    "teacher_it": {
        "password": "ktutech",
        "branch": "Information Technology"
    },

    "teacher_cyber": {
        "password": "ktutech",
        "branch": "Cyber Security"
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
# FILES
# =========================================================

DATA_DIR = "data"

DATASET_PATH = os.path.join(
    DATA_DIR,
    "student_performance.csv"
)

REPORT_PATH = os.path.join(
    DATA_DIR,
    "student_reports.csv"
)

os.makedirs(DATA_DIR, exist_ok=True)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "home",
    "student_logged_in": False,
    "teacher_logged_in": False,
    "teacher_username": "",
    "student_password": ""
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# TITLE
# =========================================================

st.title("🎓 Student Performance Prediction")


# =========================================================
# ATTENDANCE MARK
# =========================================================

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


# =========================================================
# PERFORMANCE LEVEL
# =========================================================

def performance_analysis(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    att_mark = attendance_mark(attendance)

    att_percent = att_mark / 5 * 100
    study_percent = study_hours / 6 * 100
    internal_percent = internal / 40 * 100
    assignment_percent = assignment / 15 * 100
    previous_percent = previous / 60 * 100

    overall = np.mean([
        att_percent,
        study_percent,
        internal_percent,
        assignment_percent,
        previous_percent
    ])

    if overall < 50:
        level = "Low Performance"
        symbol = "🔴"

    elif overall < 65:
        level = "Average Performance"
        symbol = "🟠"

    elif overall < 80:
        level = "Above Average Performance"
        symbol = "🟡"

    else:
        level = "Good Performance"
        symbol = "🟢"

    return {
        "attendance_mark": att_mark,
        "overall": overall,
        "level": level,
        "symbol": symbol
    }


# =========================================================
# RECOMMENDATIONS
# =========================================================

def recommendations(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    result = []

    if attendance < 80:
        result.append(
            "Improve attendance and attend classes regularly."
        )

    if study_hours < 3:
        result.append(
            "Increase daily study time gradually."
        )

    if internal < 20:
        result.append(
            "Give more attention to internal examination preparation."
        )

    if assignment < 8:
        result.append(
            "Complete assignments regularly and submit them on time."
        )

    if previous < 30:
        result.append(
            "Revise previous topics and strengthen basic concepts."
        )

    if not result:
        result.append(
            "Good progress. Maintain consistency and continue revision."
        )

    return result


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_PATH):
        return None

    try:
        return pd.read_csv(DATASET_PATH)

    except Exception:
        return None


# =========================================================
# TRAIN MODEL
#
# All model features are converted to percentage scale.
# This allows the UI to use:
#
# Attendance 0-100
# Study Hours 0-6
# Internal 0-40
# Assignment 0-15
# Previous 0-60
# =========================================================

@st.cache_resource
def train_models():

    data = load_dataset()

    if data is None:
        return None

    required = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Performance"
    ]

    for column in required:

        if column not in data.columns:
            return None

    data = data.copy()

    numeric_columns = required[:-1]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=required
    )

    if len(data) < 10:
        return None

    # Convert every feature to 0-100 scale

    X = pd.DataFrame()

    X["Attendance"] = (
        data["Attendance"].clip(0, 100)
    )

    X["Study_Hours"] = (
        data["Study_Hours"].clip(0, 6)
        / 6 * 100
    )

    X["Internal_Mark"] = (
        data["Internal_Mark"].clip(0, 40)
        / 40 * 100
    )

    X["Assignment"] = (
        data["Assignment"].clip(0, 15)
        / 15 * 100
    )

    X["Previous_Mark"] = (
        data["Previous_Mark"].clip(0, 60)
        / 60 * 100
    )

    y = data["Performance"].astype(str)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    cluster_values = kmeans.fit_predict(
        X_scaled
    )

    try:

        silhouette = silhouette_score(
            X_scaled,
            cluster_values
        )

    except Exception:

        silhouette = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return {
        "model": model,
        "kmeans": kmeans,
        "scaler": scaler,
        "accuracy": accuracy,
        "silhouette": silhouette
    }


# =========================================================
# SAVE STUDENT REPORT
# =========================================================

def save_student_report(record):

    new_data = pd.DataFrame(
        [record]
    )

    if os.path.exists(REPORT_PATH):

        try:

            old_data = pd.read_csv(
                REPORT_PATH
            )

            final_data = pd.concat(
                [old_data, new_data],
                ignore_index=True
            )

        except Exception:

            final_data = new_data

    else:

        final_data = new_data

    final_data.to_csv(
        REPORT_PATH,
        index=False
    )


# =========================================================
# CREATE PDF REPORT
# =========================================================

def create_pdf_report(
    student_name,
    university_id,
    semester,
    branch,
    subject_results,
    ml_prediction,
    ml_confidence,
    cluster,
    model_accuracy
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=25,
        leftMargin=25,
        topMargin=25,
        bottomMargin=25
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=15
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
        parent=styles["Normal"],
        fontSize=9,
        leading=12
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
            "AI & Data Science - Machine Learning Based Academic Analysis",
            normal_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # -----------------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "1. Student Information",
            heading_style
        )
    )

    student_table = Table(
        [
            [
                "Student Name",
                student_name,
                "University ID",
                university_id
            ],
            [
                "Semester",
                semester,
                "Branch",
                branch
            ]
        ],
        colWidths=[
            90, 180, 90, 350
        ]
    )

    student_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("BACKGROUND", (2, 0), (2, -1), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8)
        ])
    )

    story.append(
        student_table
    )

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # SUBJECT TABLE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "2. Subject-wise Academic Performance",
            heading_style
        )
    )

    table_data = [
        [
            "Subject",
            "Attendance",
            "Study\nHours",
            "Internal\n/40",
            "Assignment\n/15",
            "Previous\n/60",
            "Level",
            "Overall"
        ]
    ]

    for result in subject_results:

        table_data.append([
            result["subject"],
            f'{result["attendance"]}%',
            f'{result["study_hours"]:.1f}',
            f'{result["internal"]}/40',
            f'{result["assignment"]}/15',
            f'{result["previous"]}/60',
            f'{result["symbol"]} {result["level"]}',
            f'{result["overall"]:.1f}%'
        ])

    subject_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            155,
            65,
            55,
            65,
            75,
            75,
            145,
            60
        ]
    )

    subject_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#D9EAF7")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.black
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
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
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        subject_table
    )

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # PERFORMANCE INDICATORS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "3. Performance Indicator",
            heading_style
        )
    )

    indicator_data = [
        ["Indicator", "Meaning"],
        ["🔴 Low Performance", "Needs improvement"],
        ["🟠 Average Performance", "Moderate performance"],
        ["🟡 Above Average Performance", "Good progress with scope for improvement"],
        ["🟢 Good Performance", "Good academic performance"]
    ]

    indicator_table = Table(
        indicator_data,
        colWidths=[200, 450]
    )

    indicator_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
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
        indicator_table
    )

    story.append(
        Spacer(1, 15)
    )

    # -----------------------------------------------------
    # MACHINE LEARNING RESULT
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "4. Machine Learning Result",
            heading_style
        )
    )

    ml_table = Table(
        [
            ["Parameter", "Result"],
            [
                "Random Forest Prediction",
                str(ml_prediction)
            ],
            [
                "Model Confidence",
                f"{ml_confidence:.2f}%"
            ],
            [
                "K-Means Cluster",
                str(cluster)
            ],
            [
                "Random Forest Test Accuracy",
                f"{model_accuracy * 100:.2f}%"
            ]
        ],
        colWidths=[250, 300]
    )

    ml_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(
        ml_table
    )

    story.append(
        PageBreak()
    )

    # -----------------------------------------------------
    # SUBJECT RECOMMENDATIONS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "5. Subject-wise Improvement Plan",
            heading_style
        )
    )

    for result in subject_results:

        story.append(
            Paragraph(
                f'{result["symbol"]} '
                f'<b>{html.escape(result["subject"])}</b>',
                normal_style
            )
        )

        story.append(
            Paragraph(
                f'Performance Level: '
                f'<b>{html.escape(result["level"])}</b>',
                normal_style
            )
        )

        story.append(
            Paragraph(
                f'Overall Score: '
                f'{result["overall"]:.1f}%',
                normal_style
            )
        )

        for recommendation in result[
            "recommendations"
        ]:

            story.append(
                Paragraph(
                    "• " +
                    html.escape(
                        recommendation
                    ),
                    normal_style
                )
            )

        story.append(
            Spacer(1, 8)
        )

    # -----------------------------------------------------
    # ATTENDANCE CONVERSION
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "6. Attendance Mark Conversion",
            heading_style
        )
    )

    attendance_table = Table(
        [
            ["Attendance", "Converted Mark"],
            ["90% - 100%", "5"],
            ["80% - 89%", "4"],
            ["70% - 79%", "3"],
            ["60% - 69%", "2"],
            ["10% - 59%", "1"],
            ["Below 10%", "0"]
        ],
        colWidths=[180, 150]
    )

    attendance_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "ALIGN",
                (1, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(
        attendance_table
    )

    story.append(
        Spacer(1, 20)
    )

    # -----------------------------------------------------
    # NOTE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "<b>Note:</b> This report is generated by the "
            "Student Performance Prediction project using "
            "K-Means clustering and Random Forest classification. "
            "The performance categories used by this project are "
            "project-defined indicators and are not official KTU grades. "
            "The report is intended to support academic monitoring "
            "and should not replace teacher or institutional evaluation.",
            normal_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "home":

    st.subheader("Choose Login")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):

            st.session_state.page = "student_login"
            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):

            st.session_state.page = "teacher_login"
            st.rerun()


# =========================================================
# STUDENT LOGIN
# =========================================================

elif st.session_state.page == "student_login":

    st.subheader("🎓 Student Login")

    password = st.text_input(
        "Enter Student Password",
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        valid = False

        if len(password) == 9:

            if password[:5].upper() == "BTECH":

                year = password[5:]

                if year.isdigit():

                    year_value = int(year)

                    if 2000 <= year_value <= 2020:
                        valid = True

        if valid:

            st.session_state.student_logged_in = True
            st.session_state.student_password = password
            st.session_state.page = "student_dashboard"

            st.rerun()

        else:

            st.error(
                "Invalid student password."
            )

    if st.button("⬅ Back"):

        st.session_state.page = "home"
        st.rerun()


# =========================================================
# TEACHER LOGIN
# =========================================================

elif st.session_state.page == "teacher_login":

    st.subheader("👨‍🏫 Teacher Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        if (
            username in TEACHERS
            and
            password == TEACHERS[username]["password"]
        ):

            st.session_state.teacher_logged_in = True
            st.session_state.teacher_username = username
            st.session_state.page = "teacher_dashboard"

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    if st.button("⬅ Back"):

        st.session_state.page = "home"
        st.rerun()


# =========================================================
# STUDENT DASHBOARD
# =========================================================

elif st.session_state.page == "student_dashboard":

    if not st.session_state.student_logged_in:

        st.session_state.page = "home"
        st.rerun()

    st.subheader("🎓 Student Dashboard")

    if st.button("Logout"):

        st.session_state.student_logged_in = False
        st.session_state.page = "home"

        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # STUDENT DETAILS
    # -----------------------------------------------------

    st.header("Student Information")

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
            "Branch",
            BRANCHES
        )

    # -----------------------------------------------------
    # SUBJECT SELECTION
    # -----------------------------------------------------

    st.header(
        f"📚 {semester} Subject Selection"
    )

    available_subjects = SUBJECTS[
        branch
    ][semester]

    st.write(
        f"Available subjects: "
        f"**{len(available_subjects)}**"
    )

    st.info(
        "Select exactly 6 subjects."
    )

    selected_subjects = st.multiselect(
        "Choose Subjects",
        available_subjects,
        max_selections=6
    )

    if len(selected_subjects) < 6:

        st.warning(
            f"Please select "
            f"{6 - len(selected_subjects)} "
            f"more subject(s)."
        )

    elif len(selected_subjects) > 6:

        st.error(
            "You can select only 6 subjects."
        )

    else:

        st.success(
            "Exactly 6 subjects selected."
        )

    # -----------------------------------------------------
    # ACADEMIC DATA
    # -----------------------------------------------------

    subject_inputs = {}

    if len(selected_subjects) == 6:

        st.header(
            "📊 Academic Information"
        )

        for index, subject in enumerate(
            selected_subjects,
            start=1
        ):

            st.subheader(
                f"{index}. {subject}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                attendance = st.number_input(
                    "Attendance (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    key=f"att_{subject}"
                )

            with c2:

                study_hours = st.number_input(
                    "Study Hours / Day",
                    min_value=0.0,
                    max_value=6.0,
                    value=3.0,
                    step=0.5,
                    key=f"study_{subject}"
                )

            with c3:

                internal = st.number_input(
                    "Internal Mark / 40",
                    min_value=0.0,
                    max_value=40.0,
                    value=25.0,
                    step=1.0,
                    key=f"internal_{subject}"
                )

            with c4:

                assignment = st.number_input(
                    "Assignment / 15",
                    min_value=0.0,
                    max_value=15.0,
                    value=10.0,
                    step=1.0,
                    key=f"assignment_{subject}"
                )

            with c5:

                previous = st.number_input(
                    "Previous Mark / 60",
                    min_value=0.0,
                    max_value=60.0,
                    value=40.0,
                    step=1.0,
                    key=f"previous_{subject}"
                )

            subject_inputs[subject] = {
                "attendance": attendance,
                "study_hours": study_hours,
                "internal": internal,
                "assignment": assignment,
                "previous": previous
            }

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if len(selected_subjects) == 6:

        if st.button(
            "🔍 Predict Performance",
            width="stretch"
        ):

            if not student_name.strip():

                st.error(
                    "Please enter Student Name."
                )

                st.stop()

            if not university_id.strip():

                st.error(
                    "Please enter University ID."
                )

                st.stop()

            models = train_models()

            if models is None:

                st.error(
                    "Model could not be trained. "
                    "Please check data/student_performance.csv "
                    "and make sure it contains the required columns."
                )

                st.stop()

            subject_results = []

            predictions = []
            confidences = []
            clusters = []

            for subject in selected_subjects:

                values = subject_inputs[subject]

                analysis = performance_analysis(
                    values["attendance"],
                    values["study_hours"],
                    values["internal"],
                    values["assignment"],
                    values["previous"]
                )

                recs = recommendations(
                    values["attendance"],
                    values["study_hours"],
                    values["internal"],
                    values["assignment"],
                    values["previous"]
                )

                # Convert to 0-100 scale
                model_input = pd.DataFrame(
                    [[
                        values["attendance"],
                        values["study_hours"] / 6 * 100,
                        values["internal"] / 40 * 100,
                        values["assignment"] / 15 * 100,
                        values["previous"] / 60 * 100
                    ]],
                    columns=[
                        "Attendance",
                        "Study_Hours",
                        "Internal_Mark",
                        "Assignment",
                        "Previous_Mark"
                    ]
                )

                # Random Forest
                prediction = models[
                    "model"
                ].predict(
                    model_input
                )[0]

                probability = models[
                    "model"
                ].predict_proba(
                    model_input
                )[0]

                confidence = (
                    max(probability) * 100
                )

                # K-Means
                scaled = models[
                    "scaler"
                ].transform(
                    model_input
                )

                cluster = int(
                    models["kmeans"].predict(
                        scaled
                    )[0]
                )

                predictions.append(
                    str(prediction)
                )

                confidences.append(
                    confidence
                )

                clusters.append(
                    cluster
                )

                subject_results.append({
                    "subject": subject,
                    "attendance": values["attendance"],
                    "study_hours": values["study_hours"],
                    "internal": values["internal"],
                    "assignment": values["assignment"],
                    "previous": values["previous"],
                    "level": analysis["level"],
                    "symbol": analysis["symbol"],
                    "overall": analysis["overall"],
                    "recommendations": recs
                })

            # -------------------------------------------------
            # OVERALL PREDICTION
            # -------------------------------------------------

            prediction_counts = pd.Series(
                predictions
            ).value_counts()

            overall_prediction = (
                prediction_counts.index[0]
            )

            overall_confidence = float(
                np.mean(confidences)
            )

            overall_cluster = int(
                round(np.mean(clusters))
            )

            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------

            st.divider()

            st.header(
                "📈 Overall Prediction"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Random Forest Prediction",
                    overall_prediction
                )

            with c2:

                st.metric(
                    "Confidence",
                    f"{overall_confidence:.1f}%"
                )

            with c3:

                st.metric(
                    "K-Means Cluster",
                    overall_cluster
                )

            # -------------------------------------------------
            # SUBJECT RESULTS
            # -------------------------------------------------

            st.header(
                "📚 Subject-wise Result"
            )

            for result in subject_results:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        f'{result["symbol"]} '
                        f'{result["subject"]}'
                    )

                    st.write(
                        f'**Performance:** '
                        f'{result["level"]}'
                    )

                    st.write(
                        f'**Overall:** '
                        f'{result["overall"]:.1f}%'
                    )

                    for recommendation in result[
                        "recommendations"
                    ]:

                        st.write(
                            f"• {recommendation}"
                        )

            # -------------------------------------------------
            # SAVE TO DATABASE/CSV
            # -------------------------------------------------

            for result in subject_results:

                save_student_report({
                    "Student_Name": student_name,
                    "University_ID": university_id,
                    "Semester": semester,
                    "Branch": branch,
                    "Subject": result["subject"],
                    "Attendance": result["attendance"],
                    "Study_Hours": result["study_hours"],
                    "Internal_Mark": result["internal"],
                    "Assignment": result["assignment"],
                    "Previous_Mark": result["previous"],
                    "Performance_Level": result["level"],
                    "Overall_Percent": round(
                        result["overall"],
                        2
                    ),
                    "ML_Prediction": overall_prediction,
                    "ML_Confidence": round(
                        overall_confidence,
                        2
                    ),
                    "Cluster": overall_cluster
                })

            # -------------------------------------------------
            # PDF
            # -------------------------------------------------

            pdf_data = create_pdf_report(
                student_name,
                university_id,
                semester,
                branch,
                subject_results,
                overall_prediction,
                overall_confidence,
                overall_cluster,
                models["accuracy"]
            )

            st.download_button(
                label="📥 Download Complete Progress Report PDF",
                data=pdf_data,
                file_name=(
                    f"{university_id}_"
                    f"{semester}_"
                    "Progress_Report.pdf"
                ),
                mime="application/pdf",
                width="stretch"
            )

            st.success(
                "Complete PDF progress report generated successfully."
            )


# =========================================================
# TEACHER DASHBOARD
# =========================================================

elif st.session_state.page == "teacher_dashboard":

    if not st.session_state.teacher_logged_in:

        st.session_state.page = "home"
        st.rerun()

    username = (
        st.session_state.teacher_username
    )

    teacher_branch = TEACHERS[
        username
    ]["branch"]

    st.subheader(
        "👨‍🏫 Teacher Dashboard"
    )

    st.write(
        f"**Assigned Branch:** {teacher_branch}"
    )

    if st.button("Logout"):

        st.session_state.teacher_logged_in = False
        st.session_state.teacher_username = ""
        st.session_state.page = "home"

        st.rerun()

    st.divider()

    if not os.path.exists(REPORT_PATH):

        st.info(
            "No student reports available."
        )

    else:

        try:

            reports = pd.read_csv(
                REPORT_PATH
            )

        except Exception:

            reports = pd.DataFrame()

        if reports.empty:

            st.info(
                "No student reports available."
            )

        else:

            # -------------------------------------------------
            # BRANCH SECURITY FILTER
            # -------------------------------------------------

            branch_reports = reports[
                reports["Branch"].astype(str)
                == str(teacher_branch)
            ].copy()

            st.header(
                "📋 Students in Your Branch"
            )

            if branch_reports.empty:

                st.info(
                    "No student reports found for this branch."
                )

            else:

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Report Records",
                        len(branch_reports)
                    )

                with col2:

                    st.metric(
                        "Students",
                        branch_reports[
                            "University_ID"
                        ].nunique()
                    )

                with col3:

                    average = branch_reports[
                        "Overall_Percent"
                    ].mean()

                    st.metric(
                        "Average Performance",
                        f"{average:.1f}%"
                    )

                # -------------------------------------------------
                # STUDENT
                # -------------------------------------------------

                student_ids = sorted(
                    branch_reports[
                        "University_ID"
                    ].astype(str).unique()
                )

                selected_student = st.selectbox(
                    "Select Student",
                    student_ids
                )

                student_data = branch_reports[
                    branch_reports[
                        "University_ID"
                    ].astype(str)
                    == str(selected_student)
                ].copy()

                if not student_data.empty:

                    student_name = str(
                        student_data[
                            "Student_Name"
                        ].iloc[0]
                    )

                    semester = str(
                        student_data[
                            "Semester"
                        ].iloc[0]
                    )

                    st.header(
                        f"👤 {student_name}"
                    )

                    st.write(
                        f"**University ID:** "
                        f"{selected_student}"
                    )

                    st.write(
                        f"**Semester:** {semester}"
                    )

                    st.write(
                        f"**Branch:** {teacher_branch}"
                    )

                    # -------------------------------------------------
                    # TABLE
                    # -------------------------------------------------

                    display_columns = [
                        "Subject",
                        "Attendance",
                        "Study_Hours",
                        "Internal_Mark",
                        "Assignment",
                        "Previous_Mark",
                        "Performance_Level",
                        "Overall_Percent",
                        "ML_Prediction",
                        "ML_Confidence",
                        "Cluster"
                    ]

                    st.dataframe(
                        student_data[
                            display_columns
                        ],
                        width="stretch"
                    )

                    # -------------------------------------------------
                    # CREATE TEACHER PDF
                    # -------------------------------------------------

                    subject_results = []

                    for _, row in student_data.iterrows():

                        level = str(
                            row[
                                "Performance_Level"
                            ]
                        )

                        if "Low" in level:

                            symbol = "🔴"

                        elif "Average" in level:

                            symbol = "🟠"

                        elif "Above" in level:

                            symbol = "🟡"

                        else:

                            symbol = "🟢"

                        recs = recommendations(
                            float(
                                row["Attendance"]
                            ),
                            float(
                                row["Study_Hours"]
                            ),
                            float(
                                row["Internal_Mark"]
                            ),
                            float(
                                row["Assignment"]
                            ),
                            float(
                                row["Previous_Mark"]
                            )
                        )

                        subject_results.append({
                            "subject": str(
                                row["Subject"]
                            ),
                            "attendance": float(
                                row["Attendance"]
                            ),
                            "study_hours": float(
                                row["Study_Hours"]
                            ),
                            "internal": float(
                                row["Internal_Mark"]
                            ),
                            "assignment": float(
                                row["Assignment"]
                            ),
                            "previous": float(
                                row["Previous_Mark"]
                            ),
                            "level": level,
                            "symbol": symbol,
                            "overall": float(
                                row["Overall_Percent"]
                            ),
                            "recommendations": recs
                        })

                    teacher_pdf = create_pdf_report(
                        student_name,
                        selected_student,
                        semester,
                        teacher_branch,
                        subject_results,
                        str(
                            student_data[
                                "ML_Prediction"
                            ].iloc[0]
                        ),
                        float(
                            student_data[
                                "ML_Confidence"
                            ].iloc[0]
                        ),
                        int(
                            student_data[
                                "Cluster"
                            ].iloc[0]
                        ),
                        0
                    )

                    st.download_button(
                        label="📥 Download Student Progress Report PDF",
                        data=teacher_pdf,
                        file_name=(
                            f"{selected_student}_"
                            f"{semester}_Teacher_Report.pdf"
                        ),
                        mime="application/pdf",
                        width="stretch"
                    )
