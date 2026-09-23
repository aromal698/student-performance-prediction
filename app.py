import os
import html
from io import BytesIO

import joblib
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


# ============================================================
# BRANCHES
# ============================================================

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


# ============================================================
# SUBJECTS BY BRANCH
# ============================================================

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


# ============================================================
# TEACHER ACCOUNTS
# DEMO LOGIN DETAILS
# ============================================================

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


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "logged_in": False,
    "role": None,
    "student_year": None,
    "teacher_username": None,
    "teacher_branch": None,
    "student_name": None,
    "university_id": None,
    "semester": None,
    "branch": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CREATE REPORT FILE
# ============================================================

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

        pd.DataFrame(columns=columns).to_csv(
            REPORT_FILE,
            index=False
        )


initialize_report_file()


# ============================================================
# LOAD SAVED REPORTS
# ============================================================

def load_reports():

    if not os.path.exists(REPORT_FILE):
        initialize_report_file()

    try:
        return pd.read_csv(REPORT_FILE)
    except Exception:
        initialize_report_file()
        return pd.read_csv(REPORT_FILE)


# ============================================================
# SAVE STUDENT REPORT
# ============================================================

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

    initialize_report_file()

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

    old_data = load_reports()

    # Remove previous record for same student + subject
    if not old_data.empty:
        old_data = old_data[
            ~(
                (old_data["University_ID"].astype(str) == str(university_id))
                &
                (old_data["Subject"].astype(str) == str(subject))
            )
        ]

    final_data = pd.concat(
        [old_data, new_row],
        ignore_index=True
    )

    final_data.to_csv(
        REPORT_FILE,
        index=False
    )


# ============================================================
# MODEL TRAINING
# ============================================================

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

    required_columns = features + ["Performance"]

    for column in required_columns:
        if column not in data.columns:
            st.error(
                f"Required column '{column}' is missing from "
                f"{DATA_FILE}"
            )
            return None

    X = data[features].copy()
    y = data["Performance"].astype(str)

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = kmeans.fit_predict(X_scaled)

    try:
        silhouette = silhouette_score(
            X_scaled,
            data["Cluster"]
        )
    except Exception:
        silhouette = 0.0

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

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
        "confusion_matrix": matrix,
        "labels": sorted(y.unique()),
        "features": features,
    }


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


# ============================================================
# PERFORMANCE LEVEL
# ============================================================

def get_performance_level(
    attendance,
    study_hours,
    internal_mark,
    assignment,
    previous_mark
):

    att_mark = attendance_mark(attendance)

    attendance_percent = (att_mark / 5) * 100

    internal_percent = (
        internal_mark / 40
    ) * 100

    previous_percent = (
        previous_mark / 60
    ) * 100

    assignment_percent = (
        assignment / 15
    ) * 100

    study_percent = min(
        (study_hours / 6) * 100,
        100
    )

    overall = (
        attendance_percent
        + internal_percent
        + previous_percent
        + assignment_percent
        + study_percent
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
            "Excellent progress. Maintain your current consistency and study habits.",
            overall
        )


# ============================================================
# MODEL INPUT
# ============================================================

def prepare_model_input(
    attendance,
    study_hours,
    internal_mark,
    assignment,
    previous_mark
):

    # The original trained model uses the dataset scale.
    # These values are converted to the model-compatible scale.

    model_study = (
        study_hours / 6
    ) * 100

    model_internal = (
        internal_mark / 40
    ) * 100

    model_assignment = (
        assignment / 15
    ) * 100

    model_previous = (
        previous_mark / 60
    ) * 100

    return pd.DataFrame([{
        "Attendance": attendance,
        "Study_Hours": model_study,
        "Internal_Mark": model_internal,
        "Assignment": model_assignment,
        "Previous_Mark": model_previous,
    }])


# ============================================================
# SUBJECT PREDICTION
# ============================================================

def predict_subject(
    model_data,
    attendance,
    study_hours,
    internal_mark,
    assignment,
    previous_mark
):

    model = model_data["model"]
    kmeans = model_data["kmeans"]
    scaler = model_data["scaler"]

    X_input = prepare_model_input(
        attendance,
        study_hours,
        internal_mark,
        assignment,
        previous_mark
    )

    prediction = model.predict(
        X_input
    )[0]

    probabilities = model.predict_proba(
        X_input
    )[0]

    confidence = max(probabilities) * 100

    X_scaled = scaler.transform(
        X_input
    )

    cluster = int(
        kmeans.predict(X_scaled)[0]
    )

    level, color_name, recommendation, overall = (
        get_performance_level(
            attendance,
            study_hours,
            internal_mark,
            assignment,
            previous_mark
        )
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "cluster": cluster,
        "level": level,
        "color": color_name,
        "recommendation": recommendation,
        "overall": overall,
    }


# ============================================================
# PDF GENERATION
# ============================================================

def create_progress_report_pdf(
    student_name,
    university_id,
    semester,
    branch,
    subject_results,
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        spaceAfter=15,
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
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
            "Student Performance Prediction System",
            subtitle_style
        )
    )

    # --------------------------------------------------------
    # STUDENT DETAILS
    # --------------------------------------------------------

    student_details = [
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

    detail_table = Table(
        student_details,
        colWidths=[
            3.0 * cm,
            5.2 * cm,
            3.0 * cm,
            5.2 * cm,
        ]
    )

    detail_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("BACKGROUND", (2, 0), (2, -1), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(detail_table)
    story.append(Spacer(1, 12))

    # --------------------------------------------------------
    # OVERALL TABLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Subject-wise Performance",
            heading_style
        )
    )

    table_data = [
        [
            "Subject",
            "Attendance",
            "Att. Mark",
            "Study",
            "Internal",
            "Assignment",
            "Previous",
            "Level",
            "Overall",
        ]
    ]

    for result in subject_results:

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
            f'{result["overall"]:.1f}%',
        ])

    subject_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            3.2 * cm,
            1.5 * cm,
            1.3 * cm,
            1.3 * cm,
            1.4 * cm,
            1.5 * cm,
            1.4 * cm,
            2.1 * cm,
            1.5 * cm,
        ],
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
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                3
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                3
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
        ])
    )

    # Add level background colors
    for row_index, result in enumerate(
        subject_results,
        start=1
    ):

        if result["color"] == "RED":
            bg = colors.HexColor("#F8D7DA")

        elif result["color"] == "ORANGE":
            bg = colors.HexColor("#FFE5B4")

        elif result["color"] == "YELLOW":
            bg = colors.HexColor("#FFF3CD")

        else:
            bg = colors.HexColor("#D4EDDA")

        subject_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (7, row_index),
                    (7, row_index),
                    bg
                )
            ])
        )

    story.append(subject_table)

    # --------------------------------------------------------
    # ATTENDANCE CONVERSION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Attendance Mark Conversion",
            heading_style
        )
    )

    attendance_data = [
        ["Attendance", "Converted Mark"],
        ["90% – 100%", "5 / 5"],
        ["80% – 89%", "4 / 5"],
        ["70% – 79%", "3 / 5"],
        ["60% – 69%", "2 / 5"],
        ["10% – 59%", "1 / 5"],
        ["Below 10%", "0 / 5"],
    ]

    attendance_table = Table(
        attendance_data,
        colWidths=[
            7 * cm,
            5 * cm
        ]
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
                8
            ),
        ])
    )

    story.append(attendance_table)

    # --------------------------------------------------------
    # PERFORMANCE LEGEND
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Performance Indicator",
            heading_style
        )
    )

    legend_data = [
        ["Indicator", "Meaning", "Action"],
        ["RED", "Low Performance", "Immediate improvement required"],
        ["ORANGE", "Average Performance", "Improve consistency"],
        ["YELLOW", "Above Average", "Small improvements recommended"],
        ["GREEN", "Good Performance", "Maintain current performance"],
    ]

    legend_table = Table(
        legend_data,
        colWidths=[
            3.5 * cm,
            5.0 * cm,
            7.5 * cm
        ]
    )

    legend_table.setStyle(
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

    story.append(legend_table)

    # --------------------------------------------------------
    # SUBJECT RECOMMENDATIONS
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Subject-wise Analysis and Recommendations",
            heading_style
        )
    )

    for result in subject_results:

        story.append(
            Paragraph(
                f"<b>{html.escape(str(result['subject']))}</b>",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Performance: {html.escape(str(result['level']))}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Overall Score: {result['overall']:.1f}%",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Model Prediction: "
                f"{html.escape(str(result['prediction']))}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Model Confidence: "
                f"{result['confidence']:.1f}%",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"Recommendation: "
                f"{html.escape(str(result['recommendation']))}",
                normal_style
            )
        )

        story.append(
            Spacer(1, 8)
        )

    # --------------------------------------------------------
    # GENERAL NOTE
    # --------------------------------------------------------

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "<b>Note:</b> This report is generated using a machine "
            "learning based student performance prediction system. "
            "The predictions are intended to support academic "
            "monitoring and should be considered together with "
            "teacher evaluation and other academic information.",
            small_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.title("🎓 Student Performance Prediction")

    st.write("Choose Login")

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


# ============================================================
# STUDENT LOGIN
# ============================================================

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

        year_text = password[5:]

        if not year_text.isdigit():
            st.error(
                "Invalid password."
            )
            return

        year = int(year_text)

        if year < 2000 or year > 2020:
            st.error(
                "Invalid password."
            )
            return

        st.session_state.logged_in = True
        st.session_state.role = "student"
        st.session_state.student_year = year

        st.rerun()


# ============================================================
# TEACHER LOGIN
# ============================================================

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


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    st.title("🎓 Student Dashboard")

    st.caption(
        "Select exactly 6 subjects and enter the academic details."
    )

    # --------------------------------------------------------
    # BASIC DETAILS
    # --------------------------------------------------------

    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input(
            "Student Name",
            value=st.session_state.student_name or ""
        )

        university_id = st.text_input(
            "University ID",
            value=st.session_state.university_id or ""
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
            index=2
        )

        branch = st.selectbox(
            "Branch",
            BRANCHES
        )

    # --------------------------------------------------------
    # SUBJECT SELECTION
    # --------------------------------------------------------

    st.subheader("📚 Select 6 Subjects")

    available_subjects = BRANCH_SUBJECTS.get(
        branch,
        []
    )

    selected_subjects = st.multiselect(
        "Choose exactly 6 subjects",
        available_subjects,
        max_selections=6
    )

    st.info(
        f"Selected: {len(selected_subjects)} / 6 subjects"
    )

    if len(selected_subjects) != 6:

        st.warning(
            "Please select exactly 6 subjects to continue."
        )

        if st.button(
            "Logout",
            width="stretch"
        ):
            logout()

        return

    st.success(
        "Six subjects selected successfully."
    )

    # --------------------------------------------------------
    # SUBJECT INPUTS
    # --------------------------------------------------------

    subject_inputs = {}

    for index, subject in enumerate(
        selected_subjects,
        start=1
    ):

        st.divider()

        st.subheader(
            f"Subject {index}: {subject}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
                key=f"attendance_{index}"
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=6.0,
                value=3.0,
                step=0.5,
                key=f"study_{index}"
            )

        with col2:

            internal = st.number_input(
                "Internal Mark / 40",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                step=1.0,
                key=f"internal_{index}"
            )

            assignment = st.number_input(
                "Assignment / 15",
                min_value=0.0,
                max_value=15.0,
                value=10.0,
                step=1.0,
                key=f"assignment_{index}"
            )

        with col3:

            previous = st.number_input(
                "Previous Mark / 60",
                min_value=0.0,
                max_value=60.0,
                value=35.0,
                step=1.0,
                key=f"previous_{index}"
            )

            att_mark = attendance_mark(
                attendance
            )

            st.metric(
                "Attendance Converted Mark",
                f"{att_mark} / 5"
            )

        subject_inputs[subject] = {
            "attendance": attendance,
            "study_hours": study_hours,
            "internal": internal,
            "assignment": assignment,
            "previous": previous,
        }

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🔮 Predict Performance & Generate Progress Report",
        width="stretch"
    )

    if predict_button:

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
                "Model could not be trained. "
                "Please check data/student_performance.csv."
            )

            return

        results = []

        progress_bar = st.progress(0)

        for i, subject in enumerate(
            selected_subjects
        ):

            values = subject_inputs[subject]

            prediction = predict_subject(
                model_data,
                values["attendance"],
                values["study_hours"],
                values["internal"],
                values["assignment"],
                values["previous"]
            )

            att_mark = attendance_mark(
                values["attendance"]
            )

            result = {
                "subject": subject,
                "attendance": values["attendance"],
                "attendance_mark": att_mark,
                "study_hours": values["study_hours"],
                "internal": values["internal"],
                "assignment": values["assignment"],
                "previous": values["previous"],
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "cluster": prediction["cluster"],
                "level": prediction["level"],
                "color": prediction["color"],
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
                att_mark,
                values["study_hours"],
                values["internal"],
                values["assignment"],
                values["previous"],
                prediction["level"],
                prediction["cluster"],
                prediction["confidence"],
                prediction["overall"],
            )

            progress_bar.progress(
                int(((i + 1) / 6) * 100)
            )

        st.session_state.last_results = results
        st.session_state.last_student_name = student_name
        st.session_state.last_university_id = university_id
        st.session_state.last_semester = semester
        st.session_state.last_branch = branch

        st.success(
            "Performance prediction completed successfully!"
        )

    # --------------------------------------------------------
    # DISPLAY LAST RESULTS
    # --------------------------------------------------------

    if "last_results" in st.session_state:

        results = st.session_state.last_results

        st.divider()

        st.subheader(
            "📊 Performance Summary"
        )

        summary_data = []

        for result in results:

            if result["color"] == "RED":
                indicator = "🔴"

            elif result["color"] == "ORANGE":
                indicator = "🟠"

            elif result["color"] == "YELLOW":
                indicator = "🟡"

            else:
                indicator = "🟢"

            summary_data.append([
                result["subject"],
                indicator,
                result["level"],
                f'{result["overall"]:.1f}%',
                result["prediction"],
                f'{result["confidence"]:.1f}%'
            ])

        summary_df = pd.DataFrame(
            summary_data,
            columns=[
                "Subject",
                "Indicator",
                "Performance Level",
                "Overall",
                "ML Prediction",
                "Confidence",
            ]
        )

        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        pdf_file = create_progress_report_pdf(
            st.session_state.last_student_name,
            st.session_state.last_university_id,
            st.session_state.last_semester,
            st.session_state.last_branch,
            results
        )

        st.download_button(
            label="📥 Download Progress Report as PDF",
            data=pdf_file.getvalue(),
            file_name=(
                f"{st.session_state.last_university_id}"
                "_Progress_Report.pdf"
            ),
            mime="application/pdf",
            width="stretch"
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "📈 Model Information"
    ):

        model_data = train_models()

        if model_data:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Random Forest Accuracy",
                    f'{model_data["accuracy"] * 100:.2f}%'
                )

            with col2:

                st.metric(
                    "K-Means Silhouette Score",
                    f'{model_data["silhouette"]:.3f}'
                )

    # --------------------------------------------------------
    # LOGOUT
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "Logout",
        width="stretch"
    ):
        logout()


# ============================================================
# TEACHER DASHBOARD
# ============================================================

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
            "No student progress reports are available yet."
        )

        if st.button(
            "Logout",
            width="stretch"
        ):
            logout()

        return

    # --------------------------------------------------------
    # BRANCH FILTER
    # --------------------------------------------------------

    branch_reports = reports[
        reports["Branch"].astype(str)
        == str(teacher_branch)
    ].copy()

    if branch_reports.empty:

        st.warning(
            "No students from your assigned branch "
            "have submitted reports yet."
        )

        if st.button(
            "Logout",
            width="stretch"
        ):
            logout()

        return

    # --------------------------------------------------------
    # STUDENT LIST
    # --------------------------------------------------------

    students = (
        branch_reports[
            [
                "Student_Name",
                "University_ID",
                "Semester",
                "Branch",
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

    # --------------------------------------------------------
    # SELECT STUDENT
    # --------------------------------------------------------

    student_ids = students[
        "University_ID"
    ].astype(str).tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_ids
    )

    selected_student = branch_reports[
        branch_reports["University_ID"].astype(str)
        == str(selected_id)
    ].copy()

    if selected_student.empty:
        return

    student_name = selected_student.iloc[0][
        "Student_Name"
    ]

    semester = selected_student.iloc[0][
        "Semester"
    ]

    branch = selected_student.iloc[0][
        "Branch"
    ]

    st.divider()

    st.subheader(
        f"📋 Progress Report: {student_name}"
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

    # --------------------------------------------------------
    # DISPLAY SUBJECT DETAILS
    # --------------------------------------------------------

    for _, row in selected_student.iterrows():

        st.markdown(
            f"### {row['Subject']}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Attendance",
                f"{row['Attendance']}%"
            )

            st.metric(
                "Attendance Mark",
                f"{row['Attendance_Mark']}/5"
            )

        with col2:

            st.metric(
                "Study Hours",
                f"{row['Study_Hours']}/6"
            )

            st.metric(
                "Internal",
                f"{row['Internal_Mark']}/40"
            )

        with col3:

            st.metric(
                "Assignment",
                f"{row['Assignment']}/15"
            )

            st.metric(
                "Previous Mark",
                f"{row['Previous_Mark']}/60"
            )

        with col4:

            performance = row["Performance"]

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
                f"Overall: {float(row['Overall_Percentage']):.1f}%"
            )

        st.divider()

    # --------------------------------------------------------
    # GENERATE PDF FROM TEACHER DASHBOARD
    # --------------------------------------------------------

    teacher_results = []

    for _, row in selected_student.iterrows():

        level = row["Performance"]

        if level == "Low Performance":

            color_name = "RED"

            recommendation = (
                "Focus on regular attendance, "
                "revision and assignment completion."
            )

        elif level == "Average Performance":

            color_name = "ORANGE"

            recommendation = (
                "Improve consistency in attendance, "
                "study time and internal preparation."
            )

        elif level == "Above Average":

            color_name = "YELLOW"

            recommendation = (
                "Small improvements in attendance, "
                "internal marks and assignments can "
                "improve performance."
            )

        else:

            color_name = "GREEN"

            recommendation = (
                "Excellent progress. Maintain your "
                "current consistency and study habits."
            )

        teacher_results.append({
            "subject": row["Subject"],
            "attendance": float(row["Attendance"]),
            "attendance_mark": int(row["Attendance_Mark"]),
            "study_hours": float(row["Study_Hours"]),
            "internal": float(row["Internal_Mark"]),
            "assignment": float(row["Assignment"]),
            "previous": float(row["Previous_Mark"]),
            "prediction": row["Performance"],
            "confidence": float(row["Confidence"]),
            "cluster": int(row["Cluster"]),
            "level": level,
            "color": color_name,
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
        label="📥 Download Student Progress Report PDF",
        data=teacher_pdf.getvalue(),
        file_name=(
            f"{selected_id}_Progress_Report.pdf"
        ),
        mime="application/pdf",
        width="stretch"
    )

    # --------------------------------------------------------
    # LOGOUT
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "Logout",
        width="stretch"
    ):
        logout()


# ============================================================
# LOGOUT
# ============================================================

def logout():

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()


# ============================================================
# MAIN
# ============================================================

def main():

    if not st.session_state.logged_in:

        login_page()

        return

    if st.session_state.role == "student":

        student_dashboard()

    elif st.session_state.role == "teacher":

        teacher_dashboard()


main()
