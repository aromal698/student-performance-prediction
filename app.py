import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
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
# FILE PATHS
# =========================================================

DATA_FOLDER = "data"

DATASET_FILE = os.path.join(
    DATA_FOLDER,
    "student_performance.csv"
)

REPORT_FILE = os.path.join(
    DATA_FOLDER,
    "student_reports.csv"
)

os.makedirs(DATA_FOLDER, exist_ok=True)

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
# SUBJECT LIST
# S1-S8
# =========================================================

SUBJECTS = {

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
# MODEL FEATURES
# =========================================================

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "student_logged_in" not in st.session_state:
    st.session_state.student_logged_in = False

if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False

if "teacher_username" not in st.session_state:
    st.session_state.teacher_username = ""

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

    else:
        return 0


# =========================================================
# PERFORMANCE CALCULATION
# =========================================================

def calculate_performance(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    attendance_percentage = (
        attendance_mark(attendance) / 5
    ) * 100

    study_percentage = (
        study_hours / 6
    ) * 100

    internal_percentage = (
        internal / 40
    ) * 100

    assignment_percentage = (
        assignment / 15
    ) * 100

    previous_percentage = (
        previous / 60
    ) * 100

    overall = np.mean([
        attendance_percentage,
        study_percentage,
        internal_percentage,
        assignment_percentage,
        previous_percentage
    ])

    if overall < 50:
        level = "Low Performance"
        indicator = "🔴"

    elif overall < 65:
        level = "Average Performance"
        indicator = "🟠"

    elif overall < 80:
        level = "Above Average Performance"
        indicator = "🟡"

    else:
        level = "Good Performance"
        indicator = "🟢"

    return level, indicator, overall


# =========================================================
# RECOMMENDATIONS
# =========================================================

def get_recommendations(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    recommendations = []

    if attendance < 80:
        recommendations.append(
            "Improve attendance and attend classes regularly."
        )

    if study_hours < 3:
        recommendations.append(
            "Increase daily study time gradually."
        )

    if internal < 20:
        recommendations.append(
            "Focus more on internal examination preparation."
        )

    if assignment < 8:
        recommendations.append(
            "Complete assignments regularly and submit them on time."
        )

    if previous < 30:
        recommendations.append(
            "Revise previous topics and strengthen basic concepts."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Excellent progress. Maintain your current consistency."
        )

    return recommendations


# =========================================================
# DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_FILE):
        return None

    try:
        return pd.read_csv(DATASET_FILE)

    except Exception:
        return None


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model():

    data = load_dataset()

    if data is None:
        return None, "Dataset not found."

    required_columns = FEATURES + ["Performance"]

    missing = []

    for column in required_columns:

        if column not in data.columns:
            missing.append(column)

    if missing:

        return None, (
            "Missing dataset columns: "
            + ", ".join(missing)
        )

    data = data.copy()

    for column in FEATURES:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=required_columns
    )

    if len(data) < 10:

        return None, (
            "Dataset should contain at least "
            "10 valid records."
        )

    # Convert all inputs to common 0-100 scale

    X = pd.DataFrame({

        "Attendance":
            data["Attendance"].clip(0, 100),

        "Study_Hours":
            data["Study_Hours"].clip(0, 6)
            / 6 * 100,

        "Internal_Mark":
            data["Internal_Mark"].clip(0, 40)
            / 40 * 100,

        "Assignment":
            data["Assignment"].clip(0, 15)
            / 15 * 100,

        "Previous_Mark":
            data["Previous_Mark"].clip(0, 60)
            / 60 * 100

    })

    y = data["Performance"].astype(str)

    # Scaling

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # K-Means

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        X_scaled
    )

    try:

        silhouette = silhouette_score(
            X_scaled,
            clusters
        )

    except Exception:

        silhouette = 0.0

    # Train/Test split

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

    # Random Forest

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    random_forest.fit(
        X_train,
        y_train
    )

    predictions = random_forest.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return {

        "model": random_forest,

        "scaler": scaler,

        "kmeans": kmeans,

        "accuracy": accuracy,

        "silhouette": silhouette,

        "rows": len(data)

    }, None


# =========================================================
# PREDICTION
# =========================================================

def predict_student(models, values):

    input_data = pd.DataFrame([{

        "Attendance":
            values["attendance"],

        "Study_Hours":
            values["study_hours"] / 6 * 100,

        "Internal_Mark":
            values["internal"] / 40 * 100,

        "Assignment":
            values["assignment"] / 15 * 100,

        "Previous_Mark":
            values["previous"] / 60 * 100

    }])

    model = models["model"]

    prediction = model.predict(
        input_data
    )[0]

    probabilities = model.predict_proba(
        input_data
    )[0]

    confidence = (
        np.max(probabilities) * 100
    )

    scaled = models["scaler"].transform(
        input_data
    )

    cluster = models["kmeans"].predict(
        scaled
    )[0]

    return (
        prediction,
        confidence,
        cluster
    )


# =========================================================
# SAVE STUDENT REPORT
# =========================================================

def save_student_report(records):

    new_data = pd.DataFrame(records)

    if os.path.exists(REPORT_FILE):

        try:

            old_data = pd.read_csv(
                REPORT_FILE
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
        REPORT_FILE,
        index=False
    )


# =========================================================
# PDF REPORT
# =========================================================

def create_pdf_report(
    student_name,
    university_id,
    semester,
    branch,
    subject_results,
    prediction,
    confidence,
    cluster,
    accuracy,
    silhouette
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=landscape(A4),

        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(

        "TitleStyle",

        parent=styles["Title"],

        fontSize=18,

        alignment=TA_CENTER,

        spaceAfter=10
    )

    heading_style = ParagraphStyle(

        "HeadingStyle",

        parent=styles["Heading2"],

        fontSize=12,

        spaceBefore=8,

        spaceAfter=6
    )

    normal_style = ParagraphStyle(

        "NormalStyle",

        parent=styles["BodyText"],

        fontSize=8,

        leading=10
    )

    small_style = ParagraphStyle(

        "SmallStyle",

        parent=styles["BodyText"],

        fontSize=7,

        leading=9
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
            "Student Performance Prediction",
            normal_style
        )
    )

    story.append(
        Spacer(1, 8)
    )

    # -----------------------------------------------------
    # STUDENT DETAILS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "1. Student Information",
            heading_style
        )
    )

    student_table = Table([

        [
            "Student Name",
            str(student_name),

            "University ID",
            str(university_id)
        ],

        [
            "Semester",
            str(semester),

            "Branch",
            str(branch)
        ],

        [
            "Report Generated",
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),

            "",
            ""
        ]

    ], colWidths=[
        30 * mm,
        75 * mm,
        30 * mm,
        125 * mm
    ])

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
                colors.lightgrey
            ),

            (
                "BACKGROUND",
                (2, 0),
                (2, 1),
                colors.lightgrey
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

    story.append(student_table)

    story.append(
        Spacer(1, 8)
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

    subject_table_data = [[

        "Subject",

        "Attendance",

        "Study Hours",

        "Internal /40",

        "Assignment /15",

        "Previous /60",

        "Indicator",

        "Overall"

    ]]

    for result in subject_results:

        subject_table_data.append([

            result["subject"],

            f'{result["attendance"]:.0f}%',

            f'{result["study_hours"]:.1f}',

            f'{result["internal"]:.0f}',

            f'{result["assignment"]:.0f}',

            f'{result["previous"]:.0f}',

            f'{result["indicator"]} {result["level"]}',

            f'{result["overall"]:.1f}%'

        ])

    subject_table = Table(

        subject_table_data,

        repeatRows=1,

        colWidths=[

            55 * mm,
            20 * mm,
            20 * mm,
            22 * mm,
            25 * mm,
            22 * mm,
            55 * mm,
            22 * mm

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
                7
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

    story.append(subject_table)

    story.append(
        Spacer(1, 10)
    )

    # -----------------------------------------------------
    # ML RESULT
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "3. Machine Learning Result",
            heading_style
        )
    )

    ml_table = Table([

        ["Parameter", "Result"],

        [
            "Random Forest Prediction",
            str(prediction)
        ],

        [
            "Average Confidence",
            f"{confidence:.2f}%"
        ],

        [
            "K-Means Cluster",
            str(cluster)
        ],

        [
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%"
        ],

        [
            "K-Means Silhouette Score",
            f"{silhouette:.4f}"
        ]

    ], colWidths=[
        75 * mm,
        60 * mm
    ])

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

    story.append(ml_table)

    story.append(
        PageBreak()
    )

    # -----------------------------------------------------
    # IMPROVEMENT PLAN
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "4. Subject-wise Improvement Plan",
            heading_style
        )
    )

    for result in subject_results:

        story.append(

            Paragraph(

                f'<b>{result["indicator"]} '
                f'{result["subject"]}</b> '
                f'- {result["level"]} '
                f'- Overall: '
                f'{result["overall"]:.1f}%',

                normal_style

            )

        )

        for recommendation in result["recommendations"]:

            story.append(

                Paragraph(

                    "• " + recommendation,

                    small_style

                )

            )

        story.append(
            Spacer(1, 5)
        )

    # -----------------------------------------------------
    # ATTENDANCE CONVERSION
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "5. Attendance Mark Conversion",
            heading_style
        )
    )

    attendance_table = Table([

        ["Attendance", "Converted Mark"],

        ["90% - 100%", "5"],

        ["80% - 89%", "4"],

        ["70% - 79%", "3"],

        ["60% - 69%", "2"],

        ["10% - 59%", "1"],

        ["Below 10%", "0"]

    ], colWidths=[
        55 * mm,
        45 * mm
    ])

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
                0.4,
                colors.grey
            ),

            (
                "ALIGN",
                (1, 0),
                (1, -1),
                "CENTER"
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
        attendance_table
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(

        Paragraph(

            "<b>Note:</b> The performance indicators "
            "used in this project are project-defined "
            "categories and are not official KTU grades. "
            "The prediction should support academic "
            "monitoring and should not replace teacher "
            "evaluation.",

            small_style

        )

    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.title(
        "🎓 Student Performance Prediction"
    )

    st.subheader(
        "Choose Login"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):

            st.session_state.page = (
                "student_login"
            )

            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):

            st.session_state.page = (
                "teacher_login"
            )

            st.rerun()


# =========================================================
# STUDENT LOGIN
# =========================================================

elif st.session_state.page == "student_login":

    st.title(
        "🎓 Student Login"
    )

    password = st.text_input(
        "Student Password",
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

                if (
                    year.isdigit()
                    and
                    2000 <= int(year) <= 2020
                ):

                    valid = True

        if valid:

            st.session_state.student_logged_in = True

            st.session_state.page = (
                "student_dashboard"
            )

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

    if st.button(
        "Login",
        width="stretch"
    ):

        if (
            username in TEACHERS
            and
            password
            == TEACHERS[username]["password"]
        ):

            st.session_state.teacher_logged_in = True

            st.session_state.teacher_username = (
                username
            )

            st.session_state.page = (
                "teacher_dashboard"
            )

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

    st.title(
        "🎓 Student Dashboard"
    )

    if st.button("Logout"):

        st.session_state.student_logged_in = False

        st.session_state.page = "home"

        st.rerun()

    st.divider()

    # Student information

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
        semester
    ]

    st.write(
        f"Available subjects: "
        f"**{len(available_subjects)}**"
    )

    st.info(
        "Select exactly 6 subjects."
    )

    selected_subjects = st.multiselect(

        "Select Subjects",

        available_subjects,

        max_selections=6

    )

    st.write(
        f"Selected: "
        f"**{len(selected_subjects)}/6**"
    )

    if len(selected_subjects) < 6:

        st.warning(
            "Please select 6 subjects."
        )

    elif len(selected_subjects) == 6:

        st.success(
            "Exactly 6 subjects selected."
        )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    subject_values = {}

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

                    key=f"attendance_{index}"

                )

            with c2:

                study_hours = st.number_input(

                    "Study Hours / Day",

                    min_value=0.0,

                    max_value=6.0,

                    value=3.0,

                    step=0.5,

                    key=f"study_{index}"

                )

            with c3:

                internal = st.number_input(

                    "Internal / 40",

                    min_value=0.0,

                    max_value=40.0,

                    value=25.0,

                    step=1.0,

                    key=f"internal_{index}"

                )

            with c4:

                assignment = st.number_input(

                    "Assignment / 15",

                    min_value=0.0,

                    max_value=15.0,

                    value=10.0,

                    step=1.0,

                    key=f"assignment_{index}"

                )

            with c5:

                previous = st.number_input(

                    "Previous Mark / 60",

                    min_value=0.0,

                    max_value=60.0,

                    value=40.0,

                    step=1.0,

                    key=f"previous_{index}"

                )

            subject_values[subject] = {

                "attendance": attendance,

                "study_hours": study_hours,

                "internal": internal,

                "assignment": assignment,

                "previous": previous

            }

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if len(selected_subjects) == 6:

        if st.button(

            "🔍 Predict Performance",

            width="stretch"

        ):

            if not student_name.strip():

                st.error(
                    "Enter Student Name."
                )

                st.stop()

            if not university_id.strip():

                st.error(
                    "Enter University ID."
                )

                st.stop()

            models, error = train_model()

            if models is None:

                st.error(error)

                st.info(
                    "Dataset must contain: "
                    "Attendance, Study_Hours, "
                    "Internal_Mark, Assignment, "
                    "Previous_Mark, Performance"
                )

                st.stop()

            results = []

            predictions = []

            confidences = []

            clusters = []

            # -------------------------------------------------
            # PROCESS EACH SUBJECT
            # -------------------------------------------------

            for subject in selected_subjects:

                values = subject_values[
                    subject
                ]

                level, indicator, overall = (
                    calculate_performance(
                        values["attendance"],
                        values["study_hours"],
                        values["internal"],
                        values["assignment"],
                        values["previous"]
                    )
                )

                recommendations = (
                    get_recommendations(
                        values["attendance"],
                        values["study_hours"],
                        values["internal"],
                        values["assignment"],
                        values["previous"]
                    )
                )

                prediction, confidence, cluster = (
                    predict_student(
                        models,
                        values
                    )
                )

                predictions.append(
                    prediction
                )

                confidences.append(
                    confidence
                )

                clusters.append(
                    cluster
                )

                results.append({

                    "subject": subject,

                    "attendance":
                        values["attendance"],

                    "study_hours":
                        values["study_hours"],

                    "internal":
                        values["internal"],

                    "assignment":
                        values["assignment"],

                    "previous":
                        values["previous"],

                    "level": level,

                    "indicator": indicator,

                    "overall": overall,

                    "recommendations":
                        recommendations

                })

            # -------------------------------------------------
            # OVERALL ML RESULT
            # -------------------------------------------------

            overall_prediction = (
                pd.Series(
                    predictions
                ).mode().iloc[0]
            )

            overall_confidence = (
                np.mean(
                    confidences
                )
            )

            overall_cluster = int(
                round(
                    np.mean(
                        clusters
                    )
                )
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
                    "Average Confidence",
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
                "📚 Subject-wise Performance"
            )

            for result in results:

                with st.container(
                    border=True
                ):

                    st.subheader(

                        f'{result["indicator"]} '
                        f'{result["subject"]}'

                    )

                    st.write(
                        f'**Level:** '
                        f'{result["level"]}'
                    )

                    st.write(
                        f'**Overall:** '
                        f'{result["overall"]:.1f}%'
                    )

                    st.write(
                        f'Attendance: '
                        f'{result["attendance"]:.0f}%'
                    )

                    st.write(
                        f'Study Hours: '
                        f'{result["study_hours"]:.1f}'
                    )

                    st.write(
                        f'Internal: '
                        f'{result["internal"]:.0f}/40'
                    )

                    st.write(
                        f'Assignment: '
                        f'{result["assignment"]:.0f}/15'
                    )

                    st.write(
                        f'Previous Mark: '
                        f'{result["previous"]:.0f}/60'
                    )

                    st.write(
                        "**Improvement / Recommendation:**"
                    )

                    for recommendation in (
                        result["recommendations"]
                    ):

                        st.write(
                            "• " + recommendation
                        )

            # -------------------------------------------------
            # SAVE REPORT
            # -------------------------------------------------

            records = []

            for result in results:

                records.append({

                    "Student_Name":
                        student_name,

                    "University_ID":
                        university_id,

                    "Semester":
                        semester,

                    "Branch":
                        branch,

                    "Subject":
                        result["subject"],

                    "Attendance":
                        result["attendance"],

                    "Study_Hours":
                        result["study_hours"],

                    "Internal_Mark":
                        result["internal"],

                    "Assignment":
                        result["assignment"],

                    "Previous_Mark":
                        result["previous"],

                    "Performance_Level":
                        result["level"],

                    "Overall_Percent":
                        round(
                            result["overall"],
                            2
                        ),

                    "ML_Prediction":
                        overall_prediction,

                    "ML_Confidence":
                        round(
                            overall_confidence,
                            2
                        ),

                    "Cluster":
                        overall_cluster

                })

            save_student_report(
                records
            )

            # -------------------------------------------------
            # CREATE PDF
            # -------------------------------------------------

            pdf_data = create_pdf_report(

                student_name,

                university_id,

                semester,

                branch,

                results,

                overall_prediction,

                overall_confidence,

                overall_cluster,

                models["accuracy"],

                models["silhouette"]

            )

            st.success(
                "Progress report created successfully."
            )

            st.download_button(

                label=(
                    "📥 Download Complete "
                    "Progress Report PDF"
                ),

                data=pdf_data,

                file_name=(
                    f"{university_id}_"
                    f"{semester}_"
                    f"Progress_Report.pdf"
                ),

                mime="application/pdf",

                width="stretch"

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

    teacher_branch = (
        TEACHERS[username]["branch"]
    )

    st.title(
        "👨‍🏫 Teacher Dashboard"
    )

    st.write(
        f"**Assigned Branch:** "
        f"{teacher_branch}"
    )

    if st.button("Logout"):

        st.session_state.teacher_logged_in = False

        st.session_state.teacher_username = ""

        st.session_state.page = "home"

        st.rerun()

    st.divider()

    if not os.path.exists(
        REPORT_FILE
    ):

        st.info(
            "No student reports available."
        )

    else:

        try:

            reports = pd.read_csv(
                REPORT_FILE
            )

        except Exception:

            reports = pd.DataFrame()

        if reports.empty:

            st.info(
                "No student reports available."
            )

        else:

            # IMPORTANT:
            # Teacher can only see assigned branch.

            branch_reports = reports[
                reports["Branch"].astype(str)
                ==
                str(teacher_branch)
            ].copy()

            st.header(
                "📋 Branch Student Reports"
            )

            if branch_reports.empty:

                st.info(
                    "No reports found "
                    "for your branch."
                )

            else:

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Report Records",
                        len(branch_reports)
                    )

                with c2:

                    st.metric(

                        "Students",

                        branch_reports[
                            "University_ID"
                        ]
                        .astype(str)
                        .nunique()

                    )

                with c3:

                    average = (
                        branch_reports[
                            "Overall_Percent"
                        ].mean()
                    )

                    st.metric(

                        "Average Performance",

                        f"{average:.1f}%"

                    )

                # Student selection

                students = sorted(

                    branch_reports[
                        "University_ID"
                    ]
                    .astype(str)
                    .unique()

                )

                selected_student = st.selectbox(

                    "Select Student",

                    students

                )

                student_data = (
                    branch_reports[
                        branch_reports[
                            "University_ID"
                        ].astype(str)
                        ==
                        str(selected_student)
                    ]
                    .copy()
                )

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
                        f"**Semester:** "
                        f"{semester}"
                    )

                    st.write(
                        f"**Branch:** "
                        f"{teacher_branch}"
                    )

                    columns = [

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
                            columns
                        ],

                        width="stretch"

                    )

                    # Build PDF data

                    teacher_results = []

                    for _, row in (
                        student_data.iterrows()
                    ):

                        level = str(
                            row[
                                "Performance_Level"
                            ]
                        )

                        if "Low" in level:
                            indicator = "🔴"

                        elif "Average" in level:
                            indicator = "🟠"

                        elif "Above" in level:
                            indicator = "🟡"

                        else:
                            indicator = "🟢"

                        teacher_results.append({

                            "subject":
                                str(row["Subject"]),

                            "attendance":
                                float(
                                    row["Attendance"]
                                ),

                            "study_hours":
                                float(
                                    row["Study_Hours"]
                                ),

                            "internal":
                                float(
                                    row["Internal_Mark"]
                                ),

                            "assignment":
                                float(
                                    row["Assignment"]
                                ),

                            "previous":
                                float(
                                    row["Previous_Mark"]
                                ),

                            "level":
                                level,

                            "indicator":
                                indicator,

                            "overall":
                                float(
                                    row["Overall_Percent"]
                                ),

                            "recommendations":
                                get_recommendations(

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

                        })

                    # Create teacher PDF

                    pdf_data = create_pdf_report(

                        student_name,

                        selected_student,

                        semester,

                        teacher_branch,

                        teacher_results,

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

                        0.0,

                        0.0

                    )

                    st.download_button(

                        label=(
                            "📥 Download "
                            "Student Progress Report PDF"
                        ),

                        data=pdf_data,

                        file_name=(
                            f"{selected_student}_"
                            f"{semester}_"
                            "Teacher_Report.pdf"
                        ),

                        mime="application/pdf",

                        width="stretch"

                    )
