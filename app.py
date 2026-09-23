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
    TableStyle,
    PageBreak
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "data/student_performance.csv"
REPORT_FILE = "data/student_reports.csv"


FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]


# ============================================================
# BRANCHES AND SUBJECTS
# ============================================================

BRANCH_SUBJECTS = {

    "Artificial Intelligence and Data Science": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Artificial Intelligence",
        "Machine Learning",
        "Computer Networks"
    ],

    "Computer Science and Engineering": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Operating Systems",
        "Computer Networks",
        "Object Oriented Programming"
    ],

    "Computer Science and Engineering (AI)": [
        "Mathematics",
        "Data Structures",
        "Artificial Intelligence",
        "Machine Learning",
        "Database Management System",
        "Computer Networks"
    ],

    "Computer Science and Engineering (Data Science)": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Data Analytics",
        "Machine Learning",
        "Statistics"
    ],

    "Information Technology": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Computer Networks",
        "Operating Systems",
        "Web Programming"
    ],

    "Cyber Security": [
        "Mathematics",
        "Data Structures",
        "Database Management System",
        "Cyber Security",
        "Computer Networks",
        "Cryptography"
    ],

    "Electronics and Communication Engineering": [
        "Mathematics",
        "Digital Electronics",
        "Electronic Devices",
        "Signals and Systems",
        "Communication Engineering",
        "Microprocessors"
    ],

    "Electrical and Electronics Engineering": [
        "Mathematics",
        "Circuit Theory",
        "Electrical Machines",
        "Power Systems",
        "Digital Electronics",
        "Control Systems"
    ],

    "Electronics and Instrumentation Engineering": [
        "Mathematics",
        "Electronic Devices",
        "Measurements",
        "Digital Electronics",
        "Control Systems",
        "Instrumentation"
    ],

    "Mechanical Engineering": [
        "Mathematics",
        "Engineering Mechanics",
        "Thermodynamics",
        "Fluid Mechanics",
        "Manufacturing Process",
        "Material Science"
    ],

    "Civil Engineering": [
        "Mathematics",
        "Engineering Mechanics",
        "Surveying",
        "Fluid Mechanics",
        "Structural Engineering",
        "Construction Technology"
    ]
}


BRANCHES = list(BRANCH_SUBJECTS.keys())


# ============================================================
# TEACHER ACCOUNTS
# ============================================================

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


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "login"

if "login_type" not in st.session_state:
    st.session_state.login_type = None

if "student_logged" not in st.session_state:
    st.session_state.student_logged = False

if "teacher_logged" not in st.session_state:
    st.session_state.teacher_logged = False

if "student_dob" not in st.session_state:
    st.session_state.student_dob = None

if "teacher_username" not in st.session_state:
    st.session_state.teacher_username = None

if "teacher_branch" not in st.session_state:
    st.session_state.teacher_branch = None


# ============================================================
# CREATE DATA DIRECTORY
# ============================================================

os.makedirs("data", exist_ok=True)


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_models():

    if not os.path.exists(DATA_FILE):
        return None, None, None, None

    data = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + ["Performance"]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        st.error(
            "Missing columns in student_performance.csv: "
            + ", ".join(missing)
        )
        return None, None, None, None

    X = data[FEATURES].copy()
    y = data["Performance"].copy()

    # -----------------------------
    # Standardization for K-Means
    # -----------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # K-Means
    # -----------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    # -----------------------------
    # Random Forest
    # -----------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model, kmeans, scaler, data


model, kmeans, scaler, training_data = train_models()


# ============================================================
# PERFORMANCE CALCULATION
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


def calculate_performance(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    internal_percent = (internal / 40) * 100

    previous_percent = (previous / 60) * 100

    assignment_percent = (assignment / 15) * 100

    study_percent = min(
        (study_hours / 6) * 100,
        100
    )

    attendance_percent = (
        attendance_mark(attendance) / 5
    ) * 100

    overall_percent = np.mean([
        attendance_percent,
        study_percent,
        internal_percent,
        assignment_percent,
        previous_percent
    ])

    if overall_percent < 50:

        level = "Low Performance"
        indicator = "RED"
        emoji = "🔴"

    elif overall_percent < 65:

        level = "Average Performance"
        indicator = "ORANGE"
        emoji = "🟠"

    elif overall_percent < 80:

        level = "Above Average"
        indicator = "YELLOW"
        emoji = "🟡"

    else:

        level = "Good Performance"
        indicator = "GREEN"
        emoji = "🟢"

    return (
        overall_percent,
        level,
        indicator,
        emoji
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def get_recommendation(
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
            "Increase daily study time gradually."
        )

    if internal < 20:
        recommendations.append(
            "Focus on internal examinations and regular revision."
        )

    if assignment < 8:
        recommendations.append(
            "Complete assignments on time and improve assignment quality."
        )

    if previous < 30:
        recommendations.append(
            "Revise previous topics and strengthen basic concepts."
        )

    if level == "Good Performance":

        recommendations.append(
            "Excellent progress. Continue the same consistency."
        )

    elif level == "Above Average":

        recommendations.append(
            "Small improvements can help you reach good performance."
        )

    elif level == "Average Performance":

        recommendations.append(
            "Follow a regular study timetable and revise weekly."
        )

    elif level == "Low Performance":

        recommendations.append(
            "Give special attention to weak areas and seek teacher guidance."
        )

    return recommendations


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

    if model is None or kmeans is None or scaler is None:

        return "Model unavailable", 0, 0

    # ---------------------------------------------------------
    # IMPORTANT:
    # Model was trained using percentage-like values.
    # Convert the UI values into compatible model values.
    # ---------------------------------------------------------

    study_model = (
        study_hours / 6
    ) * 100

    internal_model = (
        internal / 40
    ) * 100

    assignment_model = (
        assignment / 15
    ) * 100

    previous_model = (
        previous / 60
    ) * 100

    input_data = pd.DataFrame(
        [[
            attendance,
            study_model,
            internal_model,
            assignment_model,
            previous_model
        ]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = float(
        max(probabilities) * 100
    )

    scaled_input = scaler.transform(input_data)

    cluster = int(
        kmeans.predict(scaled_input)[0]
    )

    return prediction, cluster, confidence


# ============================================================
# SAVE STUDENT REPORT
# ============================================================

def save_student_report(report_data):

    new_data = pd.DataFrame([report_data])

    if os.path.exists(REPORT_FILE):

        old_data = pd.read_csv(REPORT_FILE)

        all_data = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )

    else:

        all_data = new_data

    all_data.to_csv(
        REPORT_FILE,
        index=False
    )


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf_report(
    student_name,
    university_id,
    semester,
    branch,
    subjects_data,
    overall_level,
    overall_percentage
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
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13
    )

    elements = []

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            title_style
        )
    )

    elements.append(
        Paragraph(
            "Student Performance Prediction System",
            ParagraphStyle(
                "subtitle",
                parent=normal_style,
                alignment=TA_CENTER,
                fontSize=10
            )
        )
    )

    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # STUDENT DETAILS
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "1. Student Information",
            heading_style
        )
    )

    student_table = Table(
        [
            ["Student Name", student_name],
            ["University ID", university_id],
            ["Semester", semester],
            ["Branch", branch],
            [
                "Report Date",
                datetime.now().strftime("%d-%m-%Y")
            ]
        ],
        colWidths=[130, 360]
    )

    student_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    elements.append(student_table)

    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # OVERALL RESULT
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "2. Overall Performance",
            heading_style
        )
    )

    overall_table = Table(
        [
            ["Overall Percentage", f"{overall_percentage:.2f}%"],
            ["Performance Level", overall_level]
        ],
        colWidths=[180, 310]
    )

    overall_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    elements.append(overall_table)

    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # SUBJECT-WISE REPORT
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "3. Subject-wise Performance",
            heading_style
        )
    )

    table_data = [
        [
            "Subject",
            "Attendance",
            "Study Hours",
            "Internal /40",
            "Assignment /15",
            "Previous /60",
            "Level"
        ]
    ]

    for item in subjects_data:

        table_data.append([
            item["Subject"],
            f'{item["Attendance"]:.0f}%',
            f'{item["Study Hours"]:.1f}',
            f'{item["Internal"]:.0f}',
            f'{item["Assignment"]:.0f}',
            f'{item["Previous"]:.0f}',
            item["Level"]
        ])

    subject_table = Table(
        table_data,
        colWidths=[
            100,
            55,
            55,
            55,
            65,
            60,
            80
        ],
        repeatRows=1
    )

    subject_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    elements.append(subject_table)

    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # DETAILED SUBJECT INFORMATION
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "4. Subject-wise Analysis and Recommendations",
            heading_style
        )
    )

    for item in subjects_data:

        elements.append(
            Paragraph(
                item["Subject"],
                ParagraphStyle(
                    "SubjectHeading",
                    parent=normal_style,
                    fontSize=11,
                    leading=14,
                    spaceBefore=8,
                    spaceAfter=5,
                    fontName="Helvetica-Bold"
                )
            )
        )

        detail_table = Table(
            [
                ["Performance Level", item["Level"]],
                ["Performance Percentage",
                 f'{item["Percentage"]:.2f}%'],
                ["Attendance",
                 f'{item["Attendance"]:.0f}%'],
                ["Attendance Mark",
                 f'{item["Attendance Mark"]}/5'],
                ["Study Hours",
                 f'{item["Study Hours"]:.1f} hours/day'],
                ["Internal Mark",
                 f'{item["Internal"]:.0f}/40'],
                ["Assignment Score",
                 f'{item["Assignment"]:.0f}/15'],
                ["Previous Mark",
                 f'{item["Previous"]:.0f}/60']
            ],
            colWidths=[180, 310]
        )

        detail_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )

        elements.append(detail_table)

        elements.append(Spacer(1, 5))

        elements.append(
            Paragraph(
                "<b>Improvement / Recommendation:</b> "
                + item["Recommendation"],
                normal_style
            )
        )

        elements.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # ATTENDANCE CONVERSION
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "5. Attendance Mark Conversion",
            heading_style
        )
    )

    attendance_table = Table(
        [
            ["Attendance", "Mark"],
            ["90% - 100%", "5"],
            ["80% - 89%", "4"],
            ["70% - 79%", "3"],
            ["60% - 69%", "2"],
            ["10% - 59%", "1"],
            ["Below 10%", "0"]
        ],
        colWidths=[250, 240]
    )

    attendance_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    elements.append(attendance_table)

    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # PERFORMANCE LEVELS
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "6. Performance Indicators",
            heading_style
        )
    )

    indicator_table = Table(
        [
            ["Indicator", "Performance Level"],
            ["RED", "Low Performance"],
            ["ORANGE", "Average Performance"],
            ["YELLOW", "Above Average"],
            ["GREEN", "Good Performance"]
        ],
        colWidths=[180, 310]
    )

    indicator_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    elements.append(indicator_table)

    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # NOTE
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Note:</b> This report is generated using a machine "
            "learning-based student performance prediction system. "
            "The prediction is intended to support academic monitoring "
            "and should be used together with teacher evaluation and "
            "academic records.",
            normal_style
        )
    )

    document.build(elements)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.page = "login"
    st.session_state.login_type = None
    st.session_state.student_logged = False
    st.session_state.teacher_logged = False
    st.session_state.student_dob = None
    st.session_state.teacher_username = None
    st.session_state.teacher_branch = None

    st.rerun()


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.title("🎓 Student Performance Prediction")

    st.write("")

    st.subheader("Choose Login")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):

            st.session_state.login_type = "student"
            st.session_state.page = "student_login"

            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):

            st.session_state.login_type = "teacher"
            st.session_state.page = "teacher_login"

            st.rerun()


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.title("🎓 Student Login")

    st.write(
        "Enter your student password."
    )

    password = st.text_input(
        "Password",
        type="password",
        max_chars=9
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        # Password must be exactly:
        # BTECH + four digit year

        valid = False
        year = None

        if len(password) == 9:

            if password[:5].upper() == "BTECH":

                year_text = password[5:]

                if year_text.isdigit():

                    year = int(year_text)

                    if 2000 <= year <= 2020:

                        valid = True

        if valid:

            st.session_state.student_logged = True
            st.session_state.student_dob = year
            st.session_state.page = "student_dashboard"

            st.success("Student login successful.")

            st.rerun()

        else:

            st.error(
                "Invalid password. Password must contain "
                "BTECH followed by a valid birth year from 2000 to 2020."
            )

    st.write("")

    if st.button("⬅ Back"):

        st.session_state.page = "login"

        st.rerun()


# ============================================================
# TEACHER LOGIN
# ============================================================

def teacher_login():

    st.title("👨‍🏫 Teacher Login")

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

        if username in TEACHERS:

            teacher = TEACHERS[username]

            if password == teacher["password"]:

                st.session_state.teacher_logged = True
                st.session_state.teacher_username = username
                st.session_state.teacher_branch = teacher["branch"]
                st.session_state.page = "teacher_dashboard"

                st.success(
                    "Teacher login successful."
                )

                st.rerun()

            else:

                st.error(
                    "Incorrect username or password."
                )

        else:

            st.error(
                "Incorrect username or password."
            )

    st.write("")

    if st.button("⬅ Back"):

        st.session_state.page = "login"

        st.rerun()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    st.title("🎓 Student Dashboard")

    if st.button("Logout"):

        logout()

    st.divider()

    # ---------------------------------------------------------
    # STUDENT INFORMATION
    # ---------------------------------------------------------

    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input(
            "Student Name"
        )

        university_id = st.text_input(
            "University ID"
        )

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

    with col2:

        branch = st.selectbox(
            "Branch",
            BRANCHES
        )

        st.info(
            "Student DOB Year: "
            + str(st.session_state.student_dob)
        )

    # ---------------------------------------------------------
    # SUBJECTS
    # ---------------------------------------------------------

    st.divider()

    st.subheader("Select 6 Subjects")

    subjects = BRANCH_SUBJECTS[branch]

    selected_subjects = st.multiselect(
        "Subjects",
        subjects,
        default=subjects[:6],
        max_selections=6
    )

    if len(selected_subjects) != 6:

        st.warning(
            "Please select exactly 6 subjects."
        )

        return

    # ---------------------------------------------------------
    # INPUTS
    # ---------------------------------------------------------

    st.divider()

    st.subheader(
        "Academic Information"
    )

    st.caption(
        "Maximum values: Attendance 100%, "
        "Study Hours 6 hours/day, "
        "Internal 40, Assignment 15, Previous Mark 60."
    )

    subject_results = []

    for index, subject in enumerate(
        selected_subjects,
        start=1
    ):

        st.markdown(
            f"### {index}. {subject}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            attendance = st.number_input(
                f"Attendance (%) - {subject}",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                key=f"attendance_{index}"
            )

            study_hours = st.number_input(
                f"Study Hours/day - {subject}",
                min_value=0.0,
                max_value=6.0,
                value=3.0,
                step=0.5,
                key=f"study_{index}"
            )

        with c2:

            internal = st.number_input(
                f"Internal Mark /40 - {subject}",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                step=1.0,
                key=f"internal_{index}"
            )

            assignment = st.number_input(
                f"Assignment /15 - {subject}",
                min_value=0.0,
                max_value=15.0,
                value=10.0,
                step=1.0,
                key=f"assignment_{index}"
            )

        with c3:

            previous = st.number_input(
                f"Previous Mark /60 - {subject}",
                min_value=0.0,
                max_value=60.0,
                value=35.0,
                step=1.0,
                key=f"previous_{index}"
            )

        subject_results.append({
            "Subject": subject,
            "Attendance": attendance,
            "Study Hours": study_hours,
            "Internal": internal,
            "Assignment": assignment,
            "Previous": previous
        })

    # ---------------------------------------------------------
    # PREDICTION BUTTON
    # ---------------------------------------------------------

    st.divider()

    if st.button(
        "🔮 Predict Performance",
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

        final_results = []

        for item in subject_results:

            prediction, cluster, confidence = predict_student(
                item["Attendance"],
                item["Study Hours"],
                item["Internal"],
                item["Assignment"],
                item["Previous"]
            )

            percentage, level, indicator, emoji = calculate_performance(
                item["Attendance"],
                item["Study Hours"],
                item["Internal"],
                item["Assignment"],
                item["Previous"]
            )

            recommendations = get_recommendation(
                item["Attendance"],
                item["Study Hours"],
                item["Internal"],
                item["Assignment"],
                item["Previous"],
                level
            )

            item["Prediction"] = prediction
            item["Cluster"] = cluster
            item["Confidence"] = confidence
            item["Percentage"] = percentage
            item["Level"] = level
            item["Indicator"] = indicator
            item["Emoji"] = emoji
            item["Attendance Mark"] = attendance_mark(
                item["Attendance"]
            )
            item["Recommendation"] = " ".join(
                recommendations
            )

            final_results.append(item)

        # -----------------------------------------------------
        # OVERALL PERFORMANCE
        # -----------------------------------------------------

        overall_percentage = np.mean([
            item["Percentage"]
            for item in final_results
        ])

        if overall_percentage < 50:

            overall_level = "Low Performance"
            overall_emoji = "🔴"

        elif overall_percentage < 65:

            overall_level = "Average Performance"
            overall_emoji = "🟠"

        elif overall_percentage < 80:

            overall_level = "Above Average"
            overall_emoji = "🟡"

        else:

            overall_level = "Good Performance"
            overall_emoji = "🟢"

        # -----------------------------------------------------
        # STORE IN SESSION
        # -----------------------------------------------------

        st.session_state.student_results = final_results
        st.session_state.student_name = student_name
        st.session_state.university_id = university_id
        st.session_state.semester = semester
        st.session_state.branch = branch
        st.session_state.overall_percentage = overall_percentage
        st.session_state.overall_level = overall_level
        st.session_state.overall_emoji = overall_emoji

        # -----------------------------------------------------
        # SAVE EACH SUBJECT
        # -----------------------------------------------------

        for item in final_results:

            report_row = {
                "Student_Name": student_name,
                "University_ID": university_id,
                "Semester": semester,
                "Branch": branch,
                "Subject": item["Subject"],
                "Attendance": item["Attendance"],
                "Study_Hours": item["Study Hours"],
                "Internal_Mark": item["Internal"],
                "Assignment": item["Assignment"],
                "Previous_Mark": item["Previous"],
                "Performance": item["Prediction"],
                "Performance_Level": item["Level"],
                "Performance_Percentage": item["Percentage"],
                "Cluster": item["Cluster"],
                "Confidence": item["Confidence"],
                "Report_Date": datetime.now().strftime(
                    "%Y-%m-%d"
                )
            }

            save_student_report(report_row)

        st.success(
            "Performance prediction completed successfully."
        )

    # =========================================================
    # DISPLAY RESULTS
    # =========================================================

    if "student_results" not in st.session_state:

        return

    results = st.session_state.student_results

    st.divider()

    st.header(
        "📊 Overall Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Overall Percentage",
            f'{st.session_state.overall_percentage:.2f}%'
        )

    with col2:

        st.metric(
            "Performance Level",
            st.session_state.overall_level
        )

    with col3:

        st.metric(
            "Status",
            st.session_state.overall_emoji
        )

    # ---------------------------------------------------------
    # SUBJECT RESULTS
    # ---------------------------------------------------------

    st.subheader(
        "Subject-wise Performance"
    )

    for item in results:

        st.markdown(
            f"### {item['Subject']}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Level",
                f"{item['Emoji']} {item['Level']}"
            )

        with col2:

            st.metric(
                "Percentage",
                f"{item['Percentage']:.2f}%"
            )

        with col3:

            st.metric(
                "ML Prediction",
                str(item["Prediction"])
            )

        with col4:

            st.metric(
                "Confidence",
                f"{item['Confidence']:.1f}%"
            )

        st.write(
            "**Attendance:**",
            f'{item["Attendance"]:.0f}%'
        )

        st.write(
            "**Attendance Mark:**",
            f'{item["Attendance Mark"]}/5'
        )

        st.write(
            "**Study Hours:**",
            f'{item["Study Hours"]:.1f} hours/day'
        )

        st.write(
            "**Internal:**",
            f'{item["Internal"]:.0f}/40'
        )

        st.write(
            "**Assignment:**",
            f'{item["Assignment"]:.0f}/15'
        )

        st.write(
            "**Previous Mark:**",
            f'{item["Previous"]:.0f}/60'
        )

        st.info(
            "💡 " + item["Recommendation"]
        )

        st.divider()

    # =========================================================
    # PDF DOWNLOAD
    # =========================================================

    st.subheader(
        "📄 Download Progress Report"
    )

    pdf_bytes = generate_pdf_report(
        st.session_state.student_name,
        st.session_state.university_id,
        st.session_state.semester,
        st.session_state.branch,
        results,
        st.session_state.overall_level,
        st.session_state.overall_percentage
    )

    filename = (
        st.session_state.university_id
        + "_Progress_Report.pdf"
    )

    st.download_button(
        label="📥 Download Progress Report PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        width="stretch"
    )


# ============================================================
# TEACHER DASHBOARD
# ============================================================

def teacher_dashboard():

    st.title("👨‍🏫 Teacher Dashboard")

    teacher_branch = st.session_state.teacher_branch

    st.info(
        f"Branch: {teacher_branch}"
    )

    if st.button("Logout"):

        logout()

    st.divider()

    # ---------------------------------------------------------
    # CHECK REPORT FILE
    # ---------------------------------------------------------

    if not os.path.exists(REPORT_FILE):

        st.warning(
            "No student reports are available yet."
        )

        return

    reports = pd.read_csv(
        REPORT_FILE
    )

    if reports.empty:

        st.warning(
            "No student reports are available yet."
        )

        return

    # ---------------------------------------------------------
    # BRANCH FILTER
    # ---------------------------------------------------------

    branch_reports = reports[
        reports["Branch"].astype(str)
        == str(teacher_branch)
    ].copy()

    if branch_reports.empty:

        st.warning(
            "No student reports are available "
            "for your assigned branch."
        )

        return

    # ---------------------------------------------------------
    # STUDENT LIST
    # ---------------------------------------------------------

    students = (
        branch_reports[
            [
                "Student_Name",
                "University_ID",
                "Semester"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    st.subheader(
        "Students in Your Branch"
    )

    st.dataframe(
        students,
        width="stretch"
    )

    student_ids = students[
        "University_ID"
    ].astype(str).tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_ids
    )

    student_report = branch_reports[
        branch_reports[
            "University_ID"
        ].astype(str)
        == str(selected_id)
    ].copy()

    if student_report.empty:

        return

    # ---------------------------------------------------------
    # STUDENT INFORMATION
    # ---------------------------------------------------------

    first = student_report.iloc[0]

    st.divider()

    st.subheader(
        "Student Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.write(
            "**Name:**",
            first["Student_Name"]
        )

    with col2:

        st.write(
            "**University ID:**",
            first["University_ID"]
        )

    with col3:

        st.write(
            "**Semester:**",
            first["Semester"]
        )

    with col4:

        st.write(
            "**Branch:**",
            first["Branch"]
        )

    # ---------------------------------------------------------
    # SUBJECT DETAILS
    # ---------------------------------------------------------

    st.subheader(
        "Subject-wise Report"
    )

    for _, row in student_report.iterrows():

        st.markdown(
            f"### {row['Subject']}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Performance",
                str(row["Performance_Level"])
            )

        with col2:

            st.metric(
                "Percentage",
                f'{float(row["Performance_Percentage"]):.2f}%'
            )

        with col3:

            st.metric(
                "ML Prediction",
                str(row["Performance"])
            )

        with col4:

            st.metric(
                "Confidence",
                f'{float(row["Confidence"]):.1f}%'
            )

        st.write(
            f"Attendance: {float(row['Attendance']):.0f}%"
        )

        st.write(
            f"Study Hours: {float(row['Study_Hours']):.1f}"
        )

        st.write(
            f"Internal: {float(row['Internal_Mark']):.0f}/40"
        )

        st.write(
            f"Assignment: {float(row['Assignment']):.0f}/15"
        )

        st.write(
            f"Previous Mark: {float(row['Previous_Mark']):.0f}/60"
        )

        st.divider()

    # ---------------------------------------------------------
    # TEACHER PDF
    # ---------------------------------------------------------

    st.subheader(
        "📄 Download Student Progress Report"
    )

    pdf_subjects = []

    for _, row in student_report.iterrows():

        percentage = float(
            row["Performance_Percentage"]
        )

        if percentage < 50:
            level = "Low Performance"
            indicator = "RED"

        elif percentage < 65:
            level = "Average Performance"
            indicator = "ORANGE"

        elif percentage < 80:
            level = "Above Average"
            indicator = "YELLOW"

        else:
            level = "Good Performance"
            indicator = "GREEN"

        recommendation = get_recommendation(
            float(row["Attendance"]),
            float(row["Study_Hours"]),
            float(row["Internal_Mark"]),
            float(row["Assignment"]),
            float(row["Previous_Mark"]),
            level
        )

        pdf_subjects.append({

            "Subject": row["Subject"],

            "Attendance":
                float(row["Attendance"]),

            "Study Hours":
                float(row["Study_Hours"]),

            "Internal":
                float(row["Internal_Mark"]),

            "Assignment":
                float(row["Assignment"]),

            "Previous":
                float(row["Previous_Mark"]),

            "Level":
                level,

            "Percentage":
                percentage,

            "Attendance Mark":
                attendance_mark(
                    float(row["Attendance"])
                ),

            "Recommendation":
                " ".join(recommendation)
        })

    overall_percentage = np.mean([
        x["Percentage"]
        for x in pdf_subjects
    ])

    if overall_percentage < 50:

        overall_level = "Low Performance"

    elif overall_percentage < 65:

        overall_level = "Average Performance"

    elif overall_percentage < 80:

        overall_level = "Above Average"

    else:

        overall_level = "Good Performance"

    teacher_pdf = generate_pdf_report(
        str(first["Student_Name"]),
        str(first["University_ID"]),
        str(first["Semester"]),
        str(first["Branch"]),
        pdf_subjects,
        overall_level,
        overall_percentage
    )

    st.download_button(
        label="📥 Download Student PDF Report",
        data=teacher_pdf,
        file_name=f"{selected_id}_Progress_Report.pdf",
        mime="application/pdf",
        width="stretch"
    )


# ============================================================
# MAIN ROUTER
# ============================================================

if st.session_state.page == "login":

    login_page()

elif st.session_state.page == "student_login":

    student_login()

elif st.session_state.page == "teacher_login":

    teacher_login()

elif st.session_state.page == "student_dashboard":

    if st.session_state.student_logged:

        student_dashboard()

    else:

        logout()

elif st.session_state.page == "teacher_dashboard":

    if st.session_state.teacher_logged:

        teacher_dashboard()

    else:

        logout()

else:

    login_page()
