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
DATASET_FILE = os.path.join(DATA_DIR, "student_performance.csv")
REPORT_FILE = os.path.join(DATA_DIR, "student_reports.csv")

os.makedirs(DATA_DIR, exist_ok=True)

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
    "Civil Engineering",
]

# =========================================================
# SUBJECTS S1-S8
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
        "Health and Wellness",
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
        "Design Thinking",
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
        "Branch Core I",
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
        "Professional Elective I",
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
        "Mini Project",
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
        "Mini Project",
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
        "Industrial Training",
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
        "Comprehensive Viva",
    ],
}

# =========================================================
# TEACHER LOGIN
# =========================================================

TEACHERS = {
    "teacher_aids": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Data Science",
    },

    "teacher_cse": {
        "password": "ktutech",
        "branch": "Computer Science and Engineering",
    },

    "teacher_cseai": {
        "password": "ktutech",
        "branch": "Computer Science and Engineering (AI)",
    },

    "teacher_ds": {
        "password": "ktutech",
        "branch": "Computer Science and Engineering (Data Science)",
    },

    "teacher_it": {
        "password": "ktutech",
        "branch": "Information Technology",
    },

    "teacher_cyber": {
        "password": "ktutech",
        "branch": "Cyber Security",
    },

    "teacher_ece": {
        "password": "ktutech",
        "branch": "Electronics and Communication Engineering",
    },

    "teacher_eee": {
        "password": "ktutech",
        "branch": "Electrical and Electronics Engineering",
    },

    "teacher_eie": {
        "password": "ktutech",
        "branch": "Electronics and Instrumentation Engineering",
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

# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "home",
    "student_logged_in": False,
    "teacher_logged_in": False,
    "teacher_username": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

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
# PERFORMANCE LEVEL
# =========================================================

def calculate_performance(
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
):
    attendance_score = (
        attendance_mark(attendance) / 5
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
        previous_score,
    ])

    if overall < 50:
        return "Low Performance", "🔴", overall

    if overall < 65:
        return "Average Performance", "🟠", overall

    if overall < 80:
        return "Above Average Performance", "🟡", overall

    return "Good Performance", "🟢", overall


# =========================================================
# RECOMMENDATIONS
# =========================================================

def recommendations(
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
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
            "Complete and submit assignments regularly."
        )

    if previous < 30:
        result.append(
            "Revise previous topics and strengthen basic concepts."
        )

    if not result:
        result.append(
            "Good progress. Maintain your current consistency."
        )

    return result


# =========================================================
# LOAD DATASET
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
# TRAIN ML MODEL
# =========================================================

@st.cache_resource
def train_models():

    data = load_dataset()

    if data is None:
        return None, "Dataset not found."

    required = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Performance",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        return None, (
            "Missing columns: "
            + ", ".join(missing)
        )

    for column in required[:-1]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=required
    )

    if len(data) < 10:
        return None, (
            "At least 10 valid dataset rows are required."
        )

    # Convert all features to 0-100 scale
    X = pd.DataFrame({
        "Attendance":
            data["Attendance"].clip(0, 100),

        "Study_Hours":
            data["Study_Hours"].clip(0, 6) / 6 * 100,

        "Internal_Mark":
            data["Internal_Mark"].clip(0, 40) / 40 * 100,

        "Assignment":
            data["Assignment"].clip(0, 15) / 15 * 100,

        "Previous_Mark":
            data["Previous_Mark"].clip(0, 60) / 60 * 100,
    })

    y = data["Performance"].astype(str)

    # -------------------------------
    # K-MEANS
    # -------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    clusters = kmeans.fit_predict(X_scaled)

    try:
        silhouette = silhouette_score(
            X_scaled,
            clusters
        )
    except Exception:
        silhouette = 0.0

    # -------------------------------
    # RANDOM FOREST
    # -------------------------------

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
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

    return {
        "model": model,
        "scaler": scaler,
        "kmeans": kmeans,
        "accuracy": accuracy,
        "silhouette": silhouette,
    }, None


# =========================================================
# PREDICT
# =========================================================

def predict(
    models,
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
):
    input_data = pd.DataFrame([{
        "Attendance": attendance,
        "Study_Hours": study_hours / 6 * 100,
        "Internal_Mark": internal / 40 * 100,
        "Assignment": assignment / 15 * 100,
        "Previous_Mark": previous / 60 * 100,
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
        int(cluster),
    )


# =========================================================
# SAVE REPORT
# =========================================================

def save_reports(records):

    new_data = pd.DataFrame(records)

    if os.path.exists(REPORT_FILE):

        try:
            old_data = pd.read_csv(
                REPORT_FILE
            )

            final = pd.concat(
                [old_data, new_data],
                ignore_index=True
            )

        except Exception:
            final = new_data

    else:
        final = new_data

    final.to_csv(
        REPORT_FILE,
        index=False
    )


# =========================================================
# PDF GENERATOR
# =========================================================

def generate_pdf(
    student_name,
    university_id,
    semester,
    branch,
    results,
    overall_prediction,
    confidence,
    cluster,
    accuracy,
    silhouette,
):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=7,
        spaceAfter=5,
    )

    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=7,
        leading=9,
    )

    story = []

    # =====================================================
    # TITLE
    # =====================================================

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Student Performance Prediction",
            normal_style,
        )
    )

    story.append(Spacer(1, 8))

    # =====================================================
    # STUDENT INFORMATION
    # =====================================================

    story.append(
        Paragraph(
            "1. Student Information",
            heading_style,
        )
    )

    details = [
        [
            "Student Name",
            student_name,
            "University ID",
            university_id,
        ],
        [
            "Semester",
            semester,
            "Branch",
            branch,
        ],
        [
            "Report Generated",
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),
            "",
            "",
        ],
    ]

    details_table = Table(
        details,
        colWidths=[
            32 * mm,
            70 * mm,
            32 * mm,
            120 * mm,
        ],
    )

    details_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey,
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey,
            ),
            (
                "BACKGROUND",
                (2, 0),
                (2, 1),
                colors.lightgrey,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica",
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8,
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
        ])
    )

    story.append(details_table)
    story.append(Spacer(1, 8))

    # =====================================================
    # SUBJECT TABLE
    # =====================================================

    story.append(
        Paragraph(
            "2. Subject-wise Academic Performance",
            heading_style,
        )
    )

    subject_data = [[
        "Subject",
        "Attendance",
        "Study Hours",
        "Internal /40",
        "Assignment /15",
        "Previous /60",
        "Performance",
        "Overall",
    ]]

    for item in results:

        subject_data.append([
            item["subject"],
            f'{item["attendance"]:.0f}%',
            f'{item["study_hours"]:.1f}',
            f'{item["internal"]:.0f}',
            f'{item["assignment"]:.0f}',
            f'{item["previous"]:.0f}',
            f'{item["indicator"]} {item["level"]}',
            f'{item["overall"]:.1f}%',
        ])

    subject_table = Table(
        subject_data,
        repeatRows=1,
        colWidths=[
            58 * mm,
            21 * mm,
            22 * mm,
            22 * mm,
            25 * mm,
            23 * mm,
            57 * mm,
            22 * mm,
        ],
    )

    subject_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#D9EAF7"),
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.grey,
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                7,
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER",
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
        ])
    )

    story.append(subject_table)
    story.append(Spacer(1, 10))

    # =====================================================
    # MACHINE LEARNING RESULT
    # =====================================================

    story.append(
        Paragraph(
            "3. Machine Learning Analysis",
            heading_style,
        )
    )

    ml_data = [
        ["Parameter", "Result"],
        [
            "Random Forest Prediction",
            str(overall_prediction),
        ],
        [
            "Average Confidence",
            f"{confidence:.2f}%",
        ],
        [
            "K-Means Cluster",
            str(cluster),
        ],
        [
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%",
        ],
        [
            "K-Means Silhouette Score",
            f"{silhouette:.4f}",
        ],
    ]

    ml_table = Table(
        ml_data,
        colWidths=[
            75 * mm,
            65 * mm,
        ],
    )

    ml_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.grey,
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8,
            ),
        ])
    )

    story.append(ml_table)

    story.append(PageBreak())

    # =====================================================
    # IMPROVEMENT PLAN
    # =====================================================

    story.append(
        Paragraph(
            "4. Subject-wise Improvement Plan",
            heading_style,
        )
    )

    for item in results:

        story.append(
            Paragraph(
                f'<b>{item["indicator"]} '
                f'{item["subject"]}</b> — '
                f'{item["level"]} — '
                f'Overall: {item["overall"]:.1f}%',
                normal_style,
            )
        )

        for recommendation in item["recommendations"]:

            story.append(
                Paragraph(
                    "• " + recommendation,
                    small_style,
                )
            )

        story.append(Spacer(1, 5))

    # =====================================================
    # ATTENDANCE CONVERSION
    # =====================================================

    story.append(
        Paragraph(
            "5. Attendance Mark Conversion",
            heading_style,
        )
    )

    attendance_data = [
        ["Attendance", "Converted Mark"],
        ["90% - 100%", "5"],
        ["80% - 89%", "4"],
        ["70% - 79%", "3"],
        ["60% - 69%", "2"],
        ["10% - 59%", "1"],
        ["Below 10%", "0"],
    ]

    attendance_table = Table(
        attendance_data,
        colWidths=[
            55 * mm,
            45 * mm,
        ],
    )

    attendance_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.grey,
            ),
            (
                "ALIGN",
                (1, 0),
                (1, -1),
                "CENTER",
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8,
            ),
        ])
    )

    story.append(attendance_table)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "<b>Note:</b> The performance categories "
            "used in this project are project-defined "
            "indicators and are not official KTU grades. "
            "The ML prediction is intended to support "
            "academic monitoring and should not replace "
            "teacher evaluation.",
            small_style,
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# HOME
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
            width="stretch",
        ):
            st.session_state.page = "student_login"
            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch",
        ):
            st.session_state.page = "teacher_login"
            st.rerun()


# =========================================================
# STUDENT LOGIN
# =========================================================

elif st.session_state.page == "student_login":

    st.title("🎓 Student Login")

    password = st.text_input(
        "Student Password",
        type="password",
    )

    if st.button(
        "Login",
        width="stretch",
    ):

        valid = False

        if len(password) == 9:

            if password[:5].upper() == "BTECH":

                year = password[5:]

                if (
                    year.isdigit()
                    and 2000 <= int(year) <= 2020
                ):
                    valid = True

        if valid:

            st.session_state.student_logged_in = True
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

    st.title("👨‍🏫 Teacher Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password",
    )

    if st.button(
        "Login",
        width="stretch",
    ):

        if (
            username in TEACHERS
            and password
            == TEACHERS[username]["password"]
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

    st.title("🎓 Student Dashboard")

    if st.button("Logout"):

        st.session_state.student_logged_in = False
        st.session_state.page = "home"

        st.rerun()

    st.divider()

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
                "S8",
            ],
        )

        branch = st.selectbox(
            "Branch",
            BRANCHES,
        )

    # =====================================================
    # SUBJECT SELECTION
    # =====================================================

    st.header(
        f"📚 {semester} Subject Selection"
    )

    selected_subjects = st.multiselect(
        "Select exactly 6 subjects",
        SUBJECTS[semester],
        max_selections=6,
    )

    st.write(
        f"Selected subjects: "
        f"**{len(selected_subjects)}/6**"
    )

    if len(selected_subjects) != 6:

        st.warning(
            "You must select exactly 6 subjects."
        )

    # =====================================================
    # ACADEMIC INPUTS
    # =====================================================

    values = {}

    if len(selected_subjects) == 6:

        st.header(
            "📊 Academic Information"
        )

        for i, subject in enumerate(
            selected_subjects,
            start=1,
        ):

            st.subheader(
                f"{i}. {subject}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                attendance = st.number_input(
                    "Attendance (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    key=f"attendance_{i}",
                )

            with c2:

                study_hours = st.number_input(
                    "Study Hours / Day",
                    min_value=0.0,
                    max_value=6.0,
                    value=3.0,
                    step=0.5,
                    key=f"study_{i}",
                )

            with c3:

                internal = st.number_input(
                    "Internal / 40",
                    min_value=0.0,
                    max_value=40.0,
                    value=25.0,
                    step=1.0,
                    key=f"internal_{i}",
                )

            with c4:

                assignment = st.number_input(
                    "Assignment / 15",
                    min_value=0.0,
                    max_value=15.0,
                    value=10.0,
                    step=1.0,
                    key=f"assignment_{i}",
                )

            with c5:

                previous = st.number_input(
                    "Previous Mark / 60",
                    min_value=0.0,
                    max_value=60.0,
                    value=40.0,
                    step=1.0,
                    key=f"previous_{i}",
                )

            values[subject] = {
                "attendance": attendance,
                "study_hours": study_hours,
                "internal": internal,
                "assignment": assignment,
                "previous": previous,
            }

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if len(selected_subjects) == 6:

        if st.button(
            "🔍 Predict Performance & Generate PDF",
            width="stretch",
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

            models, error = train_models()

            if models is None:

                st.error(error)

                st.info(
                    "Required dataset columns: "
                    "Attendance, Study_Hours, "
                    "Internal_Mark, Assignment, "
                    "Previous_Mark, Performance"
                )

                st.stop()

            results = []

            predictions = []
            confidences = []
            clusters = []

            # =================================================
            # EACH SUBJECT
            # =================================================

            for subject in selected_subjects:

                v = values[subject]

                level, indicator, overall = (
                    calculate_performance(
                        v["attendance"],
                        v["study_hours"],
                        v["internal"],
                        v["assignment"],
                        v["previous"],
                    )
                )

                recs = recommendations(
                    v["attendance"],
                    v["study_hours"],
                    v["internal"],
                    v["assignment"],
                    v["previous"],
                )

                pred, conf, cluster = predict(
                    models,
                    v["attendance"],
                    v["study_hours"],
                    v["internal"],
                    v["assignment"],
                    v["previous"],
                )

                predictions.append(pred)
                confidences.append(conf)
                clusters.append(cluster)

                results.append({
                    "subject": subject,
                    "attendance": v["attendance"],
                    "study_hours": v["study_hours"],
                    "internal": v["internal"],
                    "assignment": v["assignment"],
                    "previous": v["previous"],
                    "level": level,
                    "indicator": indicator,
                    "overall": overall,
                    "recommendations": recs,
                })

            # =================================================
            # OVERALL RESULT
            # =================================================

            overall_prediction = (
                pd.Series(predictions)
                .mode()
                .iloc[0]
            )

            average_confidence = float(
                np.mean(confidences)
            )

            overall_cluster = int(
                round(np.mean(clusters))
            )

            # =================================================
            # DISPLAY
            # =================================================

            st.divider()

            st.header(
                "📈 Overall Prediction"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Random Forest Prediction",
                    overall_prediction,
                )

            with c2:
                st.metric(
                    "Average Confidence",
                    f"{average_confidence:.2f}%",
                )

            with c3:
                st.metric(
                    "K-Means Cluster",
                    overall_cluster,
                )

            # =================================================
            # SUBJECT RESULTS
            # =================================================

            st.header(
                "📚 Subject-wise Results"
            )

            for result in results:

                with st.container(border=True):

                    st.subheader(
                        f'{result["indicator"]} '
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
                        "**Recommendations:**"
                    )

                    for rec in result["recommendations"]:

                        st.write(
                            "• " + rec
                        )

            # =================================================
            # SAVE REPORT DATA
            # =================================================

            records = []

            for result in results:

                records.append({
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
                        average_confidence,
                        2
                    ),
                    "Cluster": overall_cluster,
                })

            save_reports(records)

            # =================================================
            # PDF
            # =================================================

            pdf = generate_pdf(
                student_name,
                university_id,
                semester,
                branch,
                results,
                overall_prediction,
                average_confidence,
                overall_cluster,
                models["accuracy"],
                models["silhouette"],
            )

            st.success(
                "✅ Complete PDF progress report generated."
            )

            st.download_button(
                "📥 Download Progress Report PDF",
                data=pdf,
                file_name=(
                    f"{university_id}_"
                    f"{semester}_"
                    f"Progress_Report.pdf"
                ),
                mime="application/pdf",
                width="stretch",
            )


# =========================================================
# TEACHER DASHBOARD
# =========================================================

elif st.session_state.page == "teacher_dashboard":

    if not st.session_state.teacher_logged_in:

        st.session_state.page = "home"
        st.rerun()

    username = st.session_state.teacher_username

    teacher_branch = TEACHERS[
        username
    ]["branch"]

    st.title(
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

    if not os.path.exists(REPORT_FILE):

        st.info(
            "No student reports available yet."
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

            # Only teacher's branch
            branch_reports = reports[
                reports["Branch"].astype(str)
                == str(teacher_branch)
            ].copy()

            if branch_reports.empty:

                st.info(
                    "No reports found for your branch."
                )

            else:

                st.header(
                    "📋 Student Reports"
                )

                students = sorted(
                    branch_reports[
                        "University_ID"
                    ]
                    .astype(str)
                    .unique()
                )

                selected_student = st.selectbox(
                    "Select Student",
                    students,
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

                    st.subheader(
                        f"👤 {student_name}"
                    )

                    st.write(
                        f"University ID: "
                        f"**{selected_student}**"
                    )

                    st.write(
                        f"Semester: **{semester}**"
                    )

                    st.write(
                        f"Branch: **{teacher_branch}**"
                    )

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
                        "Cluster",
                    ]

                    st.dataframe(
                        student_data[
                            display_columns
                        ],
                        width="stretch",
                    )

                    # -----------------------------
                    # CREATE TEACHER PDF
                    # -----------------------------

                    teacher_results = []

                    for _, row in student_data.iterrows():

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
                                float(row["Attendance"]),

                            "study_hours":
                                float(row["Study_Hours"]),

                            "internal":
                                float(row["Internal_Mark"]),

                            "assignment":
                                float(row["Assignment"]),

                            "previous":
                                float(row["Previous_Mark"]),

                            "level": level,

                            "indicator": indicator,

                            "overall":
                                float(
                                    row[
                                        "Overall_Percent"
                                    ]
                                ),

                            "recommendations":
                                recommendations(
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
                                    ),
                                ),
                        })

                    teacher_pdf = generate_pdf(
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
                        0.0,
                    )

                    st.download_button(
                        "📥 Download Student Progress Report PDF",
                        data=teacher_pdf,
                        file_name=(
                            f"{selected_student}_"
                            f"{semester}_"
                            "Teacher_Report.pdf"
                        ),
                        mime="application/pdf",
                        width="stretch",
                    )
