import os
from io import BytesIO
import html

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)

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
)
from reportlab.lib.units import cm


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

DATA_FILE = "data/student_performance.csv"
REPORT_FILE = "data/student_reports.csv"


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
# SUBJECTS FOR EACH BRANCH
# =========================================================

BRANCH_SUBJECTS = {

    "Artificial Intelligence and Data Science": [
        "Mathematics",
        "Data Structures",
        "Database Management Systems",
        "Artificial Intelligence",
        "Machine Learning",
        "Digital Electronics",
        "Computer Networks",
        "Operating Systems",
        "Python Programming",
        "Data Science",
    ],

    "Computer Science and Engineering": [
        "Mathematics",
        "Data Structures",
        "Database Management Systems",
        "Operating Systems",
        "Computer Networks",
        "Software Engineering",
        "Computer Organization",
        "Theory of Computation",
        "Compiler Design",
        "Programming",
    ],

    "Computer Science and Engineering (AI)": [
        "Mathematics",
        "Data Structures",
        "Artificial Intelligence",
        "Machine Learning",
        "Database Management Systems",
        "Computer Networks",
        "Operating Systems",
        "Deep Learning",
        "Python Programming",
        "Data Mining",
    ],

    "Computer Science and Engineering (Data Science)": [
        "Mathematics",
        "Data Structures",
        "Database Management Systems",
        "Data Science",
        "Machine Learning",
        "Statistics",
        "Python Programming",
        "Data Mining",
        "Computer Networks",
        "Artificial Intelligence",
    ],

    "Information Technology": [
        "Mathematics",
        "Data Structures",
        "Database Management Systems",
        "Computer Networks",
        "Operating Systems",
        "Web Technology",
        "Software Engineering",
        "Cyber Security",
        "Programming",
        "Data Analytics",
    ],

    "Cyber Security": [
        "Mathematics",
        "Data Structures",
        "Database Management Systems",
        "Computer Networks",
        "Cyber Security",
        "Cryptography",
        "Ethical Hacking",
        "Operating Systems",
        "Digital Forensics",
        "Network Security",
    ],

    "Electronics and Communication Engineering": [
        "Mathematics",
        "Digital Electronics",
        "Analog Electronics",
        "Signals and Systems",
        "Communication Systems",
        "Microprocessors",
        "Control Systems",
        "Electromagnetic Theory",
        "VLSI",
        "Embedded Systems",
    ],

    "Electrical and Electronics Engineering": [
        "Mathematics",
        "Circuit Theory",
        "Electrical Machines",
        "Power Systems",
        "Power Electronics",
        "Digital Electronics",
        "Control Systems",
        "Electrical Measurements",
        "Signals and Systems",
        "Microprocessors",
    ],

    "Electronics and Instrumentation Engineering": [
        "Mathematics",
        "Electronic Devices",
        "Digital Electronics",
        "Measurements",
        "Control Systems",
        "Instrumentation",
        "Signals and Systems",
        "Microprocessors",
        "Industrial Instrumentation",
        "Embedded Systems",
    ],

    "Mechanical Engineering": [
        "Mathematics",
        "Engineering Mechanics",
        "Thermodynamics",
        "Fluid Mechanics",
        "Manufacturing Processes",
        "Machine Design",
        "Heat Transfer",
        "Dynamics",
        "Material Science",
        "CAD",
    ],

    "Civil Engineering": [
        "Mathematics",
        "Engineering Mechanics",
        "Surveying",
        "Strength of Materials",
        "Fluid Mechanics",
        "Structural Engineering",
        "Geotechnical Engineering",
        "Transportation Engineering",
        "Environmental Engineering",
        "Construction Technology",
    ],
}


# =========================================================
# TEACHER LOGIN ACCOUNTS
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
    },
}


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "logged_in": False,
    "role": None,
    "student_year": None,
    "teacher_username": None,
    "teacher_branch": None,
    "last_results": None,
    "last_student_name": None,
    "last_university_id": None,
    "last_semester": None,
    "last_branch": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CREATE REPORT FILE
# =========================================================

def initialize_report_file():

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(REPORT_FILE):

        columns = [
            "Student_Name",
            "University_ID",
            "Semester",
            "Branch",
            "Subject",
            "Attendance",
            "Attendance_Mark",
            "Study_Hours",
            "Internal_Mark",
            "Assignment",
            "Previous_Mark",
            "Performance",
            "Cluster",
            "Confidence",
            "Overall_Percentage",
        ]

        pd.DataFrame(
            columns=columns
        ).to_csv(
            REPORT_FILE,
            index=False
        )


initialize_report_file()


# =========================================================
# LOAD REPORTS
# =========================================================

def load_reports():

    initialize_report_file()

    try:
        return pd.read_csv(REPORT_FILE)

    except Exception:
        initialize_report_file()
        return pd.read_csv(REPORT_FILE)


# =========================================================
# SAVE REPORT
# =========================================================

def save_student_report(
    student_name,
    university_id,
    semester,
    branch,
    subject,
    attendance,
    attendance_mark_value,
    study_hours,
    internal_mark,
    assignment,
    previous_mark,
    performance,
    cluster,
    confidence,
    overall_percentage,
):

    old_data = load_reports()

    if not old_data.empty:

        old_data = old_data[
            ~(
                (old_data["University_ID"].astype(str)
                 == str(university_id))
                &
                (old_data["Subject"].astype(str)
                 == str(subject))
            )
        ]

    new_row = pd.DataFrame([{
        "Student_Name": student_name,
        "University_ID": university_id,
        "Semester": semester,
        "Branch": branch,
        "Subject": subject,
        "Attendance": attendance,
        "Attendance_Mark": attendance_mark_value,
        "Study_Hours": study_hours,
        "Internal_Mark": internal_mark,
        "Assignment": assignment,
        "Previous_Mark": previous_mark,
        "Performance": performance,
        "Cluster": cluster,
        "Confidence": confidence,
        "Overall_Percentage": overall_percentage,
    }])

    final_data = pd.concat(
        [old_data, new_row],
        ignore_index=True
    )

    final_data.to_csv(
        REPORT_FILE,
        index=False
    )


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_models():

    if not os.path.exists(DATA_FILE):
        return None

    data = pd.read_csv(DATA_FILE)

    features = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
    ]

    required = features + ["Performance"]

    for column in required:

        if column not in data.columns:

            st.error(
                f"Missing column in dataset: {column}"
            )

            return None

    X = data[features].copy()
    y = data["Performance"].astype(str)

    # -----------------------------------------------------
    # K-MEANS
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    try:

        silhouette = silhouette_score(
            X_scaled,
            clusters
        )

    except Exception:

        silhouette = 0.0

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

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

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(y.unique())
    )

    return {
        "model": model,
        "kmeans": kmeans,
        "scaler": scaler,
        "accuracy": accuracy,
        "silhouette": silhouette,
        "report": report,
        "matrix": matrix,
        "labels": sorted(y.unique()),
        "features": features,
    }


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
# PERFORMANCE CALCULATION
# =========================================================

def calculate_performance(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    attendance_percent = (
        attendance_mark(attendance) / 5
    ) * 100

    study_percent = (
        study_hours / 6
    ) * 100

    internal_percent = (
        internal / 40
    ) * 100

    assignment_percent = (
        assignment / 15
    ) * 100

    previous_percent = (
        previous / 60
    ) * 100

    overall = (
        attendance_percent
        + study_percent
        + internal_percent
        + assignment_percent
        + previous_percent
    ) / 5

    if overall < 50:

        return (
            "Low Performance",
            "RED",
            "Focus on regular attendance, revision and assignment completion.",
            overall
        )

    elif overall < 65:

        return (
            "Average Performance",
            "ORANGE",
            "Improve consistency in attendance, study time and internal preparation.",
            overall
        )

    elif overall < 80:

        return (
            "Above Average",
            "YELLOW",
            "Small improvements in attendance, internal marks and assignments can improve performance.",
            overall
        )

    else:

        return (
            "Good Performance",
            "GREEN",
            "Excellent performance. Maintain your current study habits.",
            overall
        )


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

def prepare_model_input(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    # Convert UI values to the scale used by
    # the original training dataset.

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

    return pd.DataFrame([{
        "Attendance": attendance,
        "Study_Hours": study_model,
        "Internal_Mark": internal_model,
        "Assignment": assignment_model,
        "Previous_Mark": previous_model,
    }])


# =========================================================
# PREDICT ONE SUBJECT
# =========================================================

def predict_subject(
    model_data,
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    model = model_data["model"]
    kmeans = model_data["kmeans"]
    scaler = model_data["scaler"]

    X_input = prepare_model_input(
        attendance,
        study_hours,
        internal,
        assignment,
        previous
    )

    prediction = model.predict(
        X_input
    )[0]

    probabilities = model.predict_proba(
        X_input
    )[0]

    confidence = max(
        probabilities
    ) * 100

    X_scaled = scaler.transform(
        X_input
    )

    cluster = int(
        kmeans.predict(X_scaled)[0]
    )

    level, indicator, recommendation, overall = (
        calculate_performance(
            attendance,
            study_hours,
            internal,
            assignment,
            previous
        )
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "cluster": cluster,
        "level": level,
        "indicator": indicator,
        "recommendation": recommendation,
        "overall": overall,
    }


# =========================================================
# CREATE PDF
# =========================================================

def create_progress_report_pdf(
    student_name,
    university_id,
    semester,
    branch,
    results
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=17,
        leading=21,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=7,
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
    )

    small_style = ParagraphStyle(
        "SmallStyle",
        parent=styles["Normal"],
        fontSize=7,
        leading=9,
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
            "Student Performance Prediction System",
            subtitle_style
        )
    )

    # -----------------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------------

    info = [
        [
            Paragraph("<b>Student Name</b>", normal_style),
            Paragraph(
                html.escape(str(student_name)),
                normal_style
            ),
            Paragraph("<b>University ID</b>", normal_style),
            Paragraph(
                html.escape(str(university_id)),
                normal_style
            ),
        ],
        [
            Paragraph("<b>Semester</b>", normal_style),
            Paragraph(
                html.escape(str(semester)),
                normal_style
            ),
            Paragraph("<b>Branch</b>", normal_style),
            Paragraph(
                html.escape(str(branch)),
                small_style
            ),
        ],
    ]

    info_table = Table(
        info,
        colWidths=[
            3 * cm,
            5 * cm,
            3 * cm,
            5 * cm
        ]
    )

    info_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("BACKGROUND", (2, 0), (2, -1), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    story.append(info_table)
    story.append(Spacer(1, 12))

    # -----------------------------------------------------
    # SUBJECT TABLE
    # -----------------------------------------------------

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
            "Att./5",
            "Study",
            "Internal",
            "Assign.",
            "Previous",
            "Level",
            "Overall"
        ]
    ]

    for result in results:

        table_data.append([
            Paragraph(
                html.escape(str(result["subject"])),
                small_style
            ),
            f'{result["attendance"]:.0f}%',
            f'{result["attendance_mark"]}/5',
            f'{result["study_hours"]:.1f}/6',
            f'{result["internal"]:.0f}/40',
            f'{result["assignment"]:.0f}/15',
            f'{result["previous"]:.0f}/60',
            result["level"],
            f'{result["overall"]:.1f}%'
        ])

    subject_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            3.1 * cm,
            1.35 * cm,
            1.25 * cm,
            1.2 * cm,
            1.35 * cm,
            1.35 * cm,
            1.35 * cm,
            2.15 * cm,
            1.3 * cm
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
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
        ])
    )

    for row, result in enumerate(results, start=1):

        if result["indicator"] == "RED":
            background = colors.HexColor("#F8D7DA")

        elif result["indicator"] == "ORANGE":
            background = colors.HexColor("#FFE5B4")

        elif result["indicator"] == "YELLOW":
            background = colors.HexColor("#FFF3CD")

        else:
            background = colors.HexColor("#D4EDDA")

        subject_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (7, row),
                    (7, row),
                    background
                )
            ])
        )

    story.append(subject_table)

    # -----------------------------------------------------
    # ATTENDANCE CONVERSION
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Attendance Mark Conversion",
            heading_style
        )
    )

    attendance_table = Table([
        ["Attendance", "Mark"],
        ["90% - 100%", "5 / 5"],
        ["80% - 89%", "4 / 5"],
        ["70% - 79%", "3 / 5"],
        ["60% - 69%", "2 / 5"],
        ["10% - 59%", "1 / 5"],
        ["Below 10%", "0 / 5"],
    ], colWidths=[6 * cm, 4 * cm])

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
                8
            ),
        ])
    )

    story.append(attendance_table)

    # -----------------------------------------------------
    # PERFORMANCE INDICATOR
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Performance Indicator",
            heading_style
        )
    )

    indicator_table = Table([
        ["Indicator", "Performance", "Recommended Action"],
        ["RED", "Low Performance", "Immediate improvement required"],
        ["ORANGE", "Average Performance", "Improve consistency"],
        ["YELLOW", "Above Average", "Small improvements recommended"],
        ["GREEN", "Good Performance", "Maintain current performance"],
    ], colWidths=[3 * cm, 5 * cm, 7 * cm])

    indicator_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "BACKGROUND",
                (0, 1),
                (0, 1),
                colors.HexColor("#F8D7DA")
            ),
            (
                "BACKGROUND",
                (0, 2),
                (0, 2),
                colors.HexColor("#FFE5B4")
            ),
            (
                "BACKGROUND",
                (0, 3),
                (0, 3),
                colors.HexColor("#FFF3CD")
            ),
            (
                "BACKGROUND",
                (0, 4),
                (0, 4),
                colors.HexColor("#D4EDDA")
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
            ),
        ])
    )

    story.append(indicator_table)

    # -----------------------------------------------------
    # PAGE 2
    # -----------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Subject-wise Analysis and Recommendations",
            heading_style
        )
    )

    for result in results:

        story.append(
            Paragraph(
                f"<b>{html.escape(str(result['subject']))}</b>",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Performance Level: "
                f"{html.escape(str(result['level']))}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Overall Score: "
                f"{result['overall']:.1f}%",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"ML Prediction: "
                f"{html.escape(str(result['prediction']))}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"ML Confidence: "
                f"{result['confidence']:.1f}%",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Cluster: "
                f"{result['cluster']}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Recommendation:</b> "
                f"{html.escape(str(result['recommendation']))}",
                normal_style
            )
        )

        story.append(
            Spacer(1, 8)
        )

    # -----------------------------------------------------
    # FINAL NOTE
    # -----------------------------------------------------

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "<b>Note:</b> The performance prediction is generated "
            "using machine learning and is intended to support "
            "academic monitoring. It should be considered together "
            "with teacher evaluation and other academic information.",
            small_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer


# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    st.title("🎓 Student Performance Prediction")

    st.write("### Choose Login")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):

            st.session_state.login_choice = "student"

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):

            st.session_state.login_choice = "teacher"

    if "login_choice" not in st.session_state:
        return

    st.divider()

    if st.session_state.login_choice == "student":

        student_login()

    else:

        teacher_login()


# =========================================================
# STUDENT LOGIN
# =========================================================

def student_login():

    st.subheader("🎓 Student Login")

    password = st.text_input(
        "Enter Password",
        type="password",
        max_chars=9
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        if len(password) != 9:

            st.error(
                "Password must contain exactly 9 characters."
            )

            return

        if not password.startswith("BTECH"):

            st.error(
                "Invalid password."
            )

            return

        year = password[5:]

        if not year.isdigit():

            st.error(
                "Invalid password."
            )

            return

        year = int(year)

        if year < 2000 or year > 2020:

            st.error(
                "Invalid password."
            )

            return

        st.session_state.logged_in = True
        st.session_state.role = "student"
        st.session_state.student_year = year

        st.rerun()


# =========================================================
# TEACHER LOGIN
# =========================================================

def teacher_login():

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

        if username in TEACHERS:

            account = TEACHERS[username]

            if password == account["password"]:

                st.session_state.logged_in = True
                st.session_state.role = "teacher"
                st.session_state.teacher_username = username
                st.session_state.teacher_branch = account["branch"]

                st.rerun()

        st.error(
            "Invalid username or password."
        )


# =========================================================
# STUDENT DASHBOARD
# =========================================================

def student_dashboard():

    st.title("🎓 Student Dashboard")

    st.caption(
        "Enter your academic information and select exactly 6 subjects."
    )

    # -----------------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------------

    st.subheader("Student Information")

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
            ],
            index=2
        )

        branch = st.selectbox(
            "Branch",
            BRANCHES
        )

    # -----------------------------------------------------
    # SIX SUBJECTS
    # -----------------------------------------------------

    st.subheader("📚 Select Exactly 6 Subjects")

    subjects = BRANCH_SUBJECTS[branch]

    selected_subjects = st.multiselect(
        "Subjects",
        subjects,
        max_selections=6
    )

    st.write(
        f"**Selected: {len(selected_subjects)} / 6**"
    )

    if len(selected_subjects) != 6:

        st.warning(
            "You must select exactly 6 subjects."
        )

        if st.button(
            "Logout",
            width="stretch"
        ):
            logout()

        return

    st.success(
        "Exactly 6 subjects selected."
    )

    # -----------------------------------------------------
    # SUBJECT DATA
    # -----------------------------------------------------

    subject_data = {}

    for i, subject in enumerate(
        selected_subjects,
        start=1
    ):

        st.divider()

        st.subheader(
            f"{i}. {subject}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
                key=f"att_{i}"
            )

            study = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=6.0,
                value=3.0,
                step=0.5,
                key=f"study_{i}"
            )

        with c2:

            internal = st.number_input(
                "Internal Mark / 40",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                step=1.0,
                key=f"internal_{i}"
            )

            assignment = st.number_input(
                "Assignment Score / 15",
                min_value=0.0,
                max_value=15.0,
                value=10.0,
                step=1.0,
                key=f"assignment_{i}"
            )

        with c3:

            previous = st.number_input(
                "Previous Mark / 60",
                min_value=0.0,
                max_value=60.0,
                value=35.0,
                step=1.0,
                key=f"previous_{i}"
            )

            st.metric(
                "Attendance Converted Mark",
                f"{attendance_mark(attendance)} / 5"
            )

        subject_data[subject] = {
            "attendance": attendance,
            "study": study,
            "internal": internal,
            "assignment": assignment,
            "previous": previous,
        }

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "🔮 Predict Performance",
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

        model_data = train_models()

        if model_data is None:

            st.error(
                "Model training failed. "
                "Check data/student_performance.csv."
            )

            return

        results = []

        progress = st.progress(0)

        for i, subject in enumerate(
            selected_subjects
        ):

            values = subject_data[subject]

            prediction = predict_subject(
                model_data,
                values["attendance"],
                values["study"],
                values["internal"],
                values["assignment"],
                values["previous"]
            )

            result = {
                "subject": subject,
                "attendance": values["attendance"],
                "attendance_mark": attendance_mark(
                    values["attendance"]
                ),
                "study_hours": values["study"],
                "internal": values["internal"],
                "assignment": values["assignment"],
                "previous": values["previous"],
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "cluster": prediction["cluster"],
                "level": prediction["level"],
                "indicator": prediction["indicator"],
                "recommendation": prediction["recommendation"],
                "overall": prediction["overall"],
            }

            results.append(result)

            save_student_report(
                student_name,
                university_id,
                semester,
                branch,
                subject,
                values["attendance"],
                attendance_mark(
                    values["attendance"]
                ),
                values["study"],
                values["internal"],
                values["assignment"],
                values["previous"],
                prediction["level"],
                prediction["cluster"],
                prediction["confidence"],
                prediction["overall"],
            )

            progress.progress(
                int(((i + 1) / 6) * 100)
            )

        st.session_state.last_results = results
        st.session_state.last_student_name = student_name
        st.session_state.last_university_id = university_id
        st.session_state.last_semester = semester
        st.session_state.last_branch = branch

        st.success(
            "Prediction completed successfully."
        )

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    if st.session_state.last_results:

        results = st.session_state.last_results

        st.divider()

        st.subheader(
            "📊 Progress Summary"
        )

        rows = []

        for result in results:

            if result["indicator"] == "RED":
                icon = "🔴"

            elif result["indicator"] == "ORANGE":
                icon = "🟠"

            elif result["indicator"] == "YELLOW":
                icon = "🟡"

            else:
                icon = "🟢"

            rows.append([
                result["subject"],
                icon,
                result["level"],
                f'{result["overall"]:.1f}%',
                result["prediction"],
                f'{result["confidence"]:.1f}%'
            ])

        result_df = pd.DataFrame(
            rows,
            columns=[
                "Subject",
                "Indicator",
                "Performance",
                "Overall",
                "ML Prediction",
                "Confidence"
            ]
        )

        st.dataframe(
            result_df,
            width="stretch",
            hide_index=True
        )

        # -------------------------------------------------
        # PDF DOWNLOAD
        # -------------------------------------------------

        pdf = create_progress_report_pdf(
            st.session_state.last_student_name,
            st.session_state.last_university_id,
            st.session_state.last_semester,
            st.session_state.last_branch,
            results
        )

        st.download_button(
            "📥 Download Progress Report PDF",
            data=pdf.getvalue(),
            file_name=(
                f"{st.session_state.last_university_id}"
                "_Progress_Report.pdf"
            ),
            mime="application/pdf",
            width="stretch"
        )

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    with st.expander("📈 Model Information"):

        model_data = train_models()

        if model_data:

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Random Forest Accuracy",
                    f"{model_data['accuracy'] * 100:.2f}%"
                )

            with c2:

                st.metric(
                    "K-Means Silhouette Score",
                    f"{model_data['silhouette']:.3f}"
                )

    # -----------------------------------------------------
    # LOGOUT
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "Logout",
        width="stretch"
    ):

        logout()


# =========================================================
# TEACHER DASHBOARD
# =========================================================

def teacher_dashboard():

    teacher_branch = (
        st.session_state.teacher_branch
    )

    st.title("👨‍🏫 Teacher Dashboard")

    st.write(
        f"Assigned Branch: **{teacher_branch}**"
    )

    reports = load_reports()

    if reports.empty:

        st.info(
            "No student reports available."
        )

        if st.button(
            "Logout",
            width="stretch"
        ):
            logout()

        return

    # -----------------------------------------------------
    # ONLY TEACHER'S BRANCH
    # -----------------------------------------------------

    branch_reports = reports[
        reports["Branch"].astype(str)
        == str(teacher_branch)
    ].copy()

    if branch_reports.empty:

        st.warning(
            "No students from your branch have submitted reports."
        )

        if st.button(
            "Logout",
            width="stretch"
        ):
            logout()

        return

    # -----------------------------------------------------
    # STUDENT LIST
    # -----------------------------------------------------

    students = (
        branch_reports[
            [
                "Student_Name",
                "University_ID",
                "Semester",
                "Branch"
            ]
        ]
        .drop_duplicates()
        .sort_values("Student_Name")
    )

    st.subheader(
        "Students in Your Branch"
    )

    st.dataframe(
        students,
        width="stretch",
        hide_index=True
    )

    # -----------------------------------------------------
    # SELECT STUDENT
    # -----------------------------------------------------

    student_ids = (
        students["University_ID"]
        .astype(str)
        .tolist()
    )

    selected_id = st.selectbox(
        "Select Student",
        student_ids
    )

    student_report = branch_reports[
        branch_reports["University_ID"].astype(str)
        == str(selected_id)
    ].copy()

    if student_report.empty:
        return

    student_name = student_report.iloc[0][
        "Student_Name"
    ]

    semester = student_report.iloc[0][
        "Semester"
    ]

    branch = student_report.iloc[0][
        "Branch"
    ]

    st.divider()

    st.subheader(
        f"📋 {student_name}"
    )

    st.write(
        f"**University ID:** {selected_id}"
    )

    st.write(
        f"**Semester:** {semester}"
    )

    st.write(
        f"**Branch:** {branch}"
    )

    # -----------------------------------------------------
    # SUBJECT DETAILS
    # -----------------------------------------------------

    for _, row in student_report.iterrows():

        st.markdown(
            f"### {row['Subject']}"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Attendance",
                f"{row['Attendance']}%"
            )

            st.metric(
                "Attendance Mark",
                f"{row['Attendance_Mark']}/5"
            )

        with c2:

            st.metric(
                "Study Hours",
                f"{row['Study_Hours']}/6"
            )

            st.metric(
                "Internal",
                f"{row['Internal_Mark']}/40"
            )

        with c3:

            st.metric(
                "Assignment",
                f"{row['Assignment']}/15"
            )

            st.metric(
                "Previous Mark",
                f"{row['Previous_Mark']}/60"
            )

        with c4:

            performance = str(
                row["Performance"]
            )

            if performance == "Low Performance":

                st.error(
                    "🔴 Low Performance"
                )

            elif performance == "Average Performance":

                st.warning(
                    "🟠 Average Performance"
                )

            elif performance == "Above Average":

                st.info(
                    "🟡 Above Average"
                )

            else:

                st.success(
                    "🟢 Good Performance"
                )

            st.write(
                f"Overall: "
                f"{float(row['Overall_Percentage']):.1f}%"
            )

        st.divider()

    # -----------------------------------------------------
    # PREPARE TEACHER PDF
    # -----------------------------------------------------

    teacher_results = []

    for _, row in student_report.iterrows():

        performance = str(
            row["Performance"]
        )

        if performance == "Low Performance":

            indicator = "RED"

            recommendation = (
                "Focus on regular attendance, "
                "revision and assignment completion."
            )

        elif performance == "Average Performance":

            indicator = "ORANGE"

            recommendation = (
                "Improve consistency in attendance, "
                "study time and internal preparation."
            )

        elif performance == "Above Average":

            indicator = "YELLOW"

            recommendation = (
                "Small improvements in attendance, "
                "internal marks and assignments can "
                "improve performance."
            )

        else:

            indicator = "GREEN"

            recommendation = (
                "Excellent performance. Maintain "
                "current study habits."
            )

        teacher_results.append({
            "subject": row["Subject"],
            "attendance": float(row["Attendance"]),
            "attendance_mark": int(row["Attendance_Mark"]),
            "study_hours": float(row["Study_Hours"]),
            "internal": float(row["Internal_Mark"]),
            "assignment": float(row["Assignment"]),
            "previous": float(row["Previous_Mark"]),
            "prediction": performance,
            "confidence": float(row["Confidence"]),
            "cluster": int(row["Cluster"]),
            "level": performance,
            "indicator": indicator,
            "recommendation": recommendation,
            "overall": float(row["Overall_Percentage"]),
        })

    teacher_pdf = create_progress_report_pdf(
        student_name,
        selected_id,
        semester,
        branch,
        teacher_results
    )

    st.download_button(
        "📥 Download Student Progress Report PDF",
        data=teacher_pdf.getvalue(),
        file_name=(
            f"{selected_id}_Progress_Report.pdf"
        ),
        mime="application/pdf",
        width="stretch"
    )

    # -----------------------------------------------------
    # LOGOUT
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "Logout",
        width="stretch"
    ):

        logout()


# =========================================================
# LOGOUT
# =========================================================

def logout():

    for key in list(st.session_state.keys()):

        del st.session_state[key]

    st.rerun()


# =========================================================
# MAIN
# =========================================================

def main():

    if not st.session_state.logged_in:

        login_page()

    elif st.session_state.role == "student":

        student_dashboard()

    elif st.session_state.role == "teacher":

        teacher_dashboard()


main()
