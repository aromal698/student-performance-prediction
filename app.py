import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import html
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
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# FOLDERS
# =========================================================

DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

REPORT_FILE = os.path.join(
    DATA_DIR,
    "student_reports.csv"
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
# SUBJECTS
# S1-S8
# =========================================================

SUBJECTS = {

    "S1": [
        "Mathematics I",
        "Physics",
        "Chemistry",
        "Engineering Graphics",
        "Programming in C",
        "Engineering Mechanics",
        "Life Skills",
        "Electrical Engineering",
        "Basic Civil Engineering",
        "Workshop Practice"
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
        "Design and Engineering"
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
        "Operating Systems"
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
        "Theory of Computation"
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
        "Professional Elective"
    ],

    "S8": [
        "Project",
        "Project Viva",
        "Comprehensive Viva",
        "Elective V",
        "Elective VI",
        "Industrial Training",
        "Advanced AI",
        "Advanced Data Science",
        "Professional Elective",
        "Open Elective"
    ]
}


# =========================================================
# TEACHER ACCOUNTS
# DEMO ACCOUNTS
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
# SESSION STATE
# =========================================================

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


# =========================================================
# DATASET
# =========================================================

def load_dataset():

    possible_files = [
        "data/student_performance.csv",
        "student_performance.csv"
    ]

    for file in possible_files:

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

                if all(col in df.columns for col in required):
                    return df

            except Exception:
                pass

    return None


# =========================================================
# CONVERT DATASET INTO COMMON 0-100 SCALE
# =========================================================

def convert_dataset_features(df):

    result = df.copy()

    # Attendance
    result["Attendance_Model"] = pd.to_numeric(
        result["Attendance"],
        errors="coerce"
    ).fillna(0)

    # Study hours
    study = pd.to_numeric(
        result["Study_Hours"],
        errors="coerce"
    ).fillna(0)

    if study.max() <= 10:
        result["Study_Model"] = (study / 6) * 100
    else:
        result["Study_Model"] = study

    # Internal
    internal = pd.to_numeric(
        result["Internal_Mark"],
        errors="coerce"
    ).fillna(0)

    if internal.max() <= 40:
        result["Internal_Model"] = (internal / 40) * 100
    else:
        result["Internal_Model"] = internal

    # Assignment
    assignment = pd.to_numeric(
        result["Assignment"],
        errors="coerce"
    ).fillna(0)

    if assignment.max() <= 15:
        result["Assignment_Model"] = (assignment / 15) * 100
    else:
        result["Assignment_Model"] = assignment

    # Previous mark
    previous = pd.to_numeric(
        result["Previous_Mark"],
        errors="coerce"
    ).fillna(0)

    if previous.max() <= 60:
        result["Previous_Model"] = (previous / 60) * 100
    else:
        result["Previous_Model"] = previous

    return result


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models():

    df = load_dataset()

    if df is None:

        # Backup synthetic training data
        rng = np.random.default_rng(42)

        n = 300

        attendance = rng.uniform(40, 100, n)
        study = rng.uniform(0, 6, n)
        internal = rng.uniform(10, 40, n)
        assignment = rng.uniform(4, 15, n)
        previous = rng.uniform(20, 60, n)

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

        df = pd.DataFrame({
            "Attendance": attendance,
            "Study_Hours": study,
            "Internal_Mark": internal,
            "Assignment": assignment,
            "Previous_Mark": previous,
            "Performance": performance
        })

    df2 = convert_dataset_features(df)

    feature_columns = [
        "Attendance_Model",
        "Study_Model",
        "Internal_Model",
        "Assignment_Model",
        "Previous_Model"
    ]

    X = df2[feature_columns].fillna(0)

    y = df2["Performance"].astype(str)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    rf.fit(X_scaled, y)

    return rf, kmeans, scaler


# =========================================================
# MODEL PREDICTION
# =========================================================

def predict_performance(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    rf, kmeans, scaler = train_models()

    row = np.array([[
        attendance,
        (study_hours / 6) * 100,
        (internal / 40) * 100,
        (assignment / 15) * 100,
        (previous / 60) * 100
    ]])

    scaled = scaler.transform(row)

    prediction = rf.predict(scaled)[0]

    probabilities = rf.predict_proba(scaled)[0]

    confidence = float(max(probabilities) * 100)

    cluster = int(kmeans.predict(scaled)[0])

    return prediction, confidence, cluster


# =========================================================
# ATTENDANCE MARK
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
# PERFORMANCE LEVEL
# =========================================================

def performance_level(
    attendance,
    study,
    internal,
    assignment,
    previous
):

    att = (attendance_mark(attendance) / 5) * 100

    study_p = min((study / 6) * 100, 100)

    internal_p = (internal / 40) * 100

    assignment_p = (assignment / 15) * 100

    previous_p = (previous / 60) * 100

    overall = np.mean([
        att,
        study_p,
        internal_p,
        assignment_p,
        previous_p
    ])

    if overall < 50:
        return "Low Performance", "🔴", overall

    elif overall < 65:
        return "Average Performance", "🟠", overall

    elif overall < 80:
        return "Above Average", "🟡", overall

    else:
        return "Good Performance", "🟢", overall


# =========================================================
# RECOMMENDATIONS
# =========================================================

def recommendations(
    attendance,
    study,
    internal,
    assignment,
    previous,
    level
):

    rec = []

    if attendance < 75:
        rec.append(
            "Improve attendance and attend classes regularly."
        )

    if study < 2:
        rec.append(
            "Increase daily study time with a consistent timetable."
        )

    if internal < 20:
        rec.append(
            "Focus on internal examinations and class revision."
        )

    if assignment < 8:
        rec.append(
            "Complete assignments on time and improve assignment quality."
        )

    if previous < 30:
        rec.append(
            "Revise previous topics and strengthen basic concepts."
        )

    if not rec:

        if level == "Good Performance":
            rec.append(
                "Excellent performance. Maintain your current consistency."
            )

        elif level == "Above Average":
            rec.append(
                "Good progress. Small improvements can help you reach the good-performance level."
            )

        else:
            rec.append(
                "Continue regular study and improve consistency."
            )

    return rec


# =========================================================
# REPORT DATABASE
# =========================================================

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
        return pd.DataFrame(columns=REPORT_COLUMNS)

    try:

        df = pd.read_csv(REPORT_FILE)

        for col in REPORT_COLUMNS:

            if col not in df.columns:
                df[col] = ""

        return df

    except Exception:

        return pd.DataFrame(columns=REPORT_COLUMNS)


def save_reports(df):

    df.to_csv(
        REPORT_FILE,
        index=False
    )


# =========================================================
# PDF REPORT
# =========================================================

def generate_student_pdf(student_data):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=12
    )

    story = []

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Student Performance Prediction System",
            normal_style
        )
    )

    story.append(Spacer(1, 12))

    info = [
        ["Student Name", student_data["Student_Name"]],
        ["University ID", student_data["University_ID"]],
        ["Semester", student_data["Semester"]],
        ["Branch", student_data["Branch"]],
        ["Report Generated", student_data["Created_Time"]]
    ]

    info_table = Table(
        info,
        colWidths=[120, 390]
    )

    info_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ])
    )

    story.append(info_table)

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Subject-wise Performance",
            heading_style
        )
    )

    subjects = json.loads(
        student_data["Subjects_JSON"]
    )

    subject_table = [
        [
            "Subject",
            "Attendance",
            "Study Hrs",
            "Internal",
            "Assignment",
            "Previous",
            "Level"
        ]
    ]

    for subject in subjects:

        subject_table.append([
            subject["Subject"],
            f'{subject["Attendance"]}%',
            str(subject["Study_Hours"]),
            f'{subject["Internal"]}/40',
            f'{subject["Assignment"]}/15',
            f'{subject["Previous"]}/60',
            f'{subject["Icon"]} {subject["Level"]}'
        ])

    table = Table(
        subject_table,
        repeatRows=1,
        colWidths=[
            115,
            55,
            55,
            55,
            60,
            55,
            100
        ]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ])
    )

    story.append(table)

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Detailed Subject Analysis",
            heading_style
        )
    )

    for subject in subjects:

        story.append(
            Paragraph(
                html.escape(
                    f'{subject["Icon"]} {subject["Subject"]}'
                ),
                styles["Heading3"]
            )
        )

        story.append(
            Paragraph(
                f'Performance Level: {html.escape(subject["Level"])}',
                normal_style
            )
        )

        story.append(
            Paragraph(
                f'Overall Score: {subject["Overall"]:.2f}%',
                normal_style
            )
        )

        story.append(
            Paragraph(
                f'Attendance Conversion: {subject["Attendance_Mark"]}/5',
                normal_style
            )
        )

        story.append(
            Paragraph(
                "<b>Improvement / Recommendation:</b>",
                normal_style
            )
        )

        for recommendation in subject["Recommendations"]:

            story.append(
                Paragraph(
                    "• " + html.escape(recommendation),
                    normal_style
                )
            )

        story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Performance Indicator",
            heading_style
        )
    )

    legend = [
        ["🔴", "Low Performance"],
        ["🟠", "Average Performance"],
        ["🟡", "Above Average"],
        ["🟢", "Good Performance"]
    ]

    legend_table = Table(
        legend,
        colWidths=[40, 200]
    )

    legend_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9)
        ])
    )

    story.append(legend_table)

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Note: This system provides academic performance "
            "prediction and recommendations for educational support. "
            "The results should support teacher evaluation and should "
            "not replace academic judgment.",
            normal_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# ALL STUDENTS PDF
# =========================================================

def generate_all_reports_pdf(reports):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleAll",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=17
    )

    heading = ParagraphStyle(
        "HeadingAll",
        parent=styles["Heading2"],
        fontSize=12
    )

    normal = ParagraphStyle(
        "NormalAll",
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

    story.append(Spacer(1, 15))

    for index, (_, row) in enumerate(reports.iterrows()):

        story.append(
            Paragraph(
                f'Student: {html.escape(str(row["Student_Name"]))}',
                heading
            )
        )

        story.append(
            Paragraph(
                f'University ID: {html.escape(str(row["University_ID"]))}',
                normal
            )
        )

        story.append(
            Paragraph(
                f'Semester: {html.escape(str(row["Semester"]))}',
                normal
            )
        )

        story.append(
            Paragraph(
                f'Branch: {html.escape(str(row["Branch"]))}',
                normal
            )
        )

        subjects = json.loads(
            row["Subjects_JSON"]
        )

        table_data = [
            [
                "Subject",
                "Attendance",
                "Internal",
                "Assignment",
                "Previous",
                "Level"
            ]
        ]

        for subject in subjects:

            table_data.append([
                subject["Subject"],
                f'{subject["Attendance"]}%',
                f'{subject["Internal"]}/40',
                f'{subject["Assignment"]}/15',
                f'{subject["Previous"]}/60',
                f'{subject["Icon"]} {subject["Level"]}'
            ])

        table = Table(
            table_data,
            repeatRows=1,
            colWidths=[
                145,
                65,
                65,
                70,
                65,
                90
            ]
        )

        table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7)
            ])
        )

        story.append(table)

        if index < len(reports) - 1:
            story.append(PageBreak())

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# HOME PAGE
# =========================================================

def home_page():

    st.title("🎓 Student Performance Prediction")

    st.markdown(
        "### Choose Login"
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
            "👨‍🏫 Teacher Login",
            width="stretch"
        ):

            st.session_state.page = "teacher_login"
            st.rerun()


# =========================================================
# STUDENT LOGIN
# =========================================================

def student_login():

    st.title("🎓 Student Login")

    st.info(
        "Enter your password in the format BTECH + your birth year."
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

                prefix = password[:5]
                year_text = password[5:]

                if (
                    prefix == "BTECH"
                    and year_text.isdigit()
                ):

                    year = int(year_text)

                    if 2000 <= year <= 2020:
                        valid = True

            if valid:

                st.session_state.logged_in = True
                st.session_state.role = "student"
                st.session_state.username = password
                st.session_state.page = "student_dashboard"

                st.success(
                    "🔔 Login successful! Welcome to the Student Dashboard."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid password."
                )

    with col2:

        if st.button(
            "⬅ Back",
            width="stretch"
        ):

            st.session_state.page = "home"
            st.rerun()


# =========================================================
# TEACHER LOGIN
# =========================================================

def teacher_login():

    st.title("👨‍🏫 Teacher Login")

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
            "🔐 Teacher Login",
            width="stretch"
        ):

            if (
                username in TEACHERS
                and TEACHERS[username]["password"] == password
            ):

                st.session_state.logged_in = True
                st.session_state.role = "teacher"
                st.session_state.username = username
                st.session_state.branch = TEACHERS[
                    username
                ]["branch"]

                st.session_state.page = "teacher_dashboard"

                st.success(
                    "🔔 Login successful! Welcome Teacher."
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


# =========================================================
# STUDENT DASHBOARD
# =========================================================

def student_dashboard():

    st.title("🎓 Student Dashboard")

    st.success(
        "🔔 You are logged in successfully. Please complete your student details."
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
    # STUDENT DETAILS
    # -----------------------------------------------------

    st.subheader("👤 Student Details")

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

    st.subheader(
        "📚 Select Subjects"
    )

    st.info(
        "Select exactly 6 subjects for the progress report."
    )

    selected_subjects = st.multiselect(
        f"Select subjects for {semester}",
        SUBJECTS[semester],
        max_selections=6
    )

    st.write(
        f"Selected: {len(selected_subjects)} / 6"
    )

    if len(selected_subjects) != 6:

        st.warning(
            "⚠️ Please select exactly 6 subjects."
        )

        return

    # -----------------------------------------------------
    # SUBJECT INPUTS
    # -----------------------------------------------------

    subject_results = []

    st.divider()

    st.subheader(
        "📊 Enter Subject-wise Academic Details"
    )

    for i, subject in enumerate(selected_subjects):

        st.markdown(
            f"### {i + 1}. {subject}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                key=f"att_{i}"
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

            st.write("")

            level, icon, overall = performance_level(
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

        recs = recommendations(
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
            "Icon": icon,
            "Overall": overall,
            "Attendance_Mark": attendance_mark(attendance),
            "Recommendations": recs
        })

        st.divider()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button(
        "🤖 Predict Performance & Generate Progress Report",
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

        predictions = []

        for subject in subject_results:

            prediction, confidence, cluster = predict_performance(
                subject["Attendance"],
                subject["Study_Hours"],
                subject["Internal"],
                subject["Assignment"],
                subject["Previous"]
            )

            subject["ML_Prediction"] = prediction
            subject["Confidence"] = confidence
            subject["Cluster"] = cluster

            predictions.append(prediction)

        # -------------------------------------------------
        # SAVE REPORT
        # -------------------------------------------------

        reports = load_reports()

        # Delete previous report for same student ID
        reports = reports[
            reports["University_ID"].astype(str)
            != str(university_id)
        ]

        now = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        new_report = {
            "Student_Name": student_name.strip(),
            "University_ID": university_id.strip(),
            "Semester": semester,
            "Branch": branch,
            "Subjects_JSON": json.dumps(
                subject_results
            ),
            "Created_Time": now
        }

        reports = pd.concat(
            [
                reports,
                pd.DataFrame([new_report])
            ],
            ignore_index=True
        )

        save_reports(reports)

        # -------------------------------------------------
        # PDF
        # -------------------------------------------------

        student_data = new_report

        pdf_bytes = generate_student_pdf(
            student_data
        )

        st.success(
            "🔔 Prediction completed! Your progress report has been generated successfully."
        )

        st.subheader(
            "📋 Overall Subject Results"
        )

        display_data = []

        for subject in subject_results:

            display_data.append([
                subject["Subject"],
                subject["Icon"],
                subject["Level"],
                f'{subject["Overall"]:.2f}%',
                subject["ML_Prediction"],
                f'{subject["Confidence"]:.1f}%'
            ])

        result_df = pd.DataFrame(
            display_data,
            columns=[
                "Subject",
                "Indicator",
                "Performance",
                "Score",
                "ML Prediction",
                "Confidence"
            ]
        )

        st.dataframe(
            result_df,
            width="stretch"
        )

        st.download_button(
            label="📥 Download Complete Progress Report PDF",
            data=pdf_bytes,
            file_name=f"{university_id}_Progress_Report.pdf",
            mime="application/pdf",
            width="stretch"
        )

        st.info(
            "Your PDF contains all 6 selected subjects, "
            "marks, attendance conversion, performance level, "
            "recommendations and ML prediction."
        )


# =========================================================
# TEACHER DASHBOARD
# =========================================================

def teacher_dashboard():

    teacher_branch = st.session_state.branch

    st.title("👨‍🏫 Teacher Dashboard")

    st.success(
        f"🔔 Login successful. You are viewing the "
        f"{teacher_branch} branch."
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
    # FILTER BRANCH
    # -----------------------------------------------------

    branch_reports = reports[
        reports["Branch"].astype(str)
        == str(teacher_branch)
    ].copy()

    st.subheader(
        "📊 Branch Student Reports"
    )

    if branch_reports.empty:

        st.info(
            "No student progress reports are available for your branch."
        )

        return

    st.write(
        f"Total student reports: **{len(branch_reports)}**"
    )

    # -----------------------------------------------------
    # STUDENT LIST
    # -----------------------------------------------------

    display = branch_reports[
        [
            "Student_Name",
            "University_ID",
            "Semester",
            "Branch",
            "Created_Time"
        ]
    ].copy()

    st.dataframe(
        display,
        width="stretch"
    )

    # -----------------------------------------------------
    # SELECT STUDENT
    # -----------------------------------------------------

    st.subheader(
        "🔎 View Student Report"
    )

    student_options = (
        branch_reports["University_ID"]
        .astype(str)
        .tolist()
    )

    selected_id = st.selectbox(
        "Select University ID",
        student_options
    )

    selected_rows = branch_reports[
        branch_reports["University_ID"].astype(str)
        == str(selected_id)
    ]

    if selected_rows.empty:
        return

    selected_row = selected_rows.iloc[0]

    st.markdown(
        f"### 👤 {selected_row['Student_Name']}"
    )

    st.write(
        f"**University ID:** {selected_row['University_ID']}"
    )

    st.write(
        f"**Semester:** {selected_row['Semester']}"
    )

    st.write(
        f"**Branch:** {selected_row['Branch']}"
    )

    subjects = json.loads(
        selected_row["Subjects_JSON"]
    )

    # -----------------------------------------------------
    # SUBJECT REPORT
    # -----------------------------------------------------

    teacher_table = []

    for subject in subjects:

        teacher_table.append([
            subject["Subject"],
            subject["Attendance"],
            subject["Study_Hours"],
            subject["Internal"],
            subject["Assignment"],
            subject["Previous"],
            subject["Icon"],
            subject["Level"]
        ])

    teacher_df = pd.DataFrame(
        teacher_table,
        columns=[
            "Subject",
            "Attendance %",
            "Study Hours",
            "Internal /40",
            "Assignment /15",
            "Previous /60",
            "Indicator",
            "Performance"
        ]
    )

    st.dataframe(
        teacher_df,
        width="stretch"
    )

    # -----------------------------------------------------
    # DOWNLOAD STUDENT PDF
    # -----------------------------------------------------

    pdf_bytes = generate_student_pdf(
        selected_row.to_dict()
    )

    st.download_button(
        "📥 Download Student Progress Report PDF",
        data=pdf_bytes,
        file_name=f"{selected_id}_Progress_Report.pdf",
        mime="application/pdf",
        width="stretch"
    )

    st.divider()

    # -----------------------------------------------------
    # DELETE STUDENT
    # -----------------------------------------------------

    st.subheader(
        "🗑️ Delete Student Details and Marks"
    )

    st.warning(
        "Deleting a student removes the saved student report "
        "and all subject-wise marks from this system."
    )

    confirm_delete = st.checkbox(
        "I confirm that I want to delete this student's details and marks.",
        key=f"confirm_delete_{selected_id}"
    )

    if st.button(
        "🗑️ Delete Selected Student",
        type="secondary"
    ):

        if not confirm_delete:

            st.error(
                "Please confirm deletion first."
            )

        else:

            # Reload latest data
            latest_reports = load_reports()

            # IMPORTANT:
            # Only delete within this teacher's branch
            mask = (
                (latest_reports["University_ID"].astype(str)
                 == str(selected_id))
                &
                (latest_reports["Branch"].astype(str)
                 == str(teacher_branch))
            )

            deleted_count = int(mask.sum())

            latest_reports = latest_reports[
                ~mask
            ].copy()

            save_reports(
                latest_reports
            )

            if deleted_count > 0:

                st.success(
                    "🗑️ Student details and marks were deleted successfully."
                )

                st.toast(
                    "Student report deleted.",
                    icon="🗑️"
                )

                st.rerun()

            else:

                st.error(
                    "Student report was not found."
                )

    # -----------------------------------------------------
    # DOWNLOAD ALL BRANCH REPORTS
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "📑 Branch Reports"
    )

    all_pdf = generate_all_reports_pdf(
        branch_reports
    )

    st.download_button(
        "📥 Download All Branch Progress Reports PDF",
        data=all_pdf,
        file_name="Branch_All_Student_Progress_Reports.pdf",
        mime="application/pdf",
        width="stretch"
    )


# =========================================================
# MAIN ROUTER
# =========================================================

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
