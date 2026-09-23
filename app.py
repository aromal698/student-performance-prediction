import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
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
    PageBreak
)

from io import BytesIO


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

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]

SUBJECTS = [
    "Mathematics",
    "Data Structures",
    "Database Management System",
    "Artificial Intelligence",
    "Machine Learning",
    "Survey and Geomatics"
]

TEACHER_PASSWORD = "*ktutech"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .login-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    .good-card {
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #2e7d32;
        background-color: #f1f8f2;
        margin-bottom: 10px;
    }

    .average-card {
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #f57c00;
        background-color: #fff8ef;
        margin-bottom: 10px;
    }

    .above-card {
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #fbc02d;
        background-color: #fffde7;
        margin-bottom: 10px;
    }

    .low-card {
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #d32f2f;
        background-color: #fff5f5;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)

    return data


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    data = load_data()

    # Check required columns
    required_columns = FEATURES + ["Performance"]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    X = data[FEATURES].copy()
    y = data["Performance"].copy()

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    data["Cluster"] = clusters

    # Silhouette score
    try:
        silhouette = silhouette_score(
            X_scaled,
            clusters
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
        output_dict=True
    )

    report_text = classification_report(
        y_test,
        y_pred
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    return (
        data,
        scaler,
        kmeans,
        model,
        accuracy,
        silhouette,
        report,
        report_text,
        matrix
    )


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

def calculate_performance_level(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    attendance_score = attendance_mark(attendance)

    attendance_percent = (
        attendance_score / 5
    ) * 100

    internal_percent = (
        internal / 40
    ) * 100

    previous_percent = (
        previous / 60
    ) * 100

    assignment_percent = (
        assignment / 15
    ) * 100

    study_percent = (
        study_hours / 6
    ) * 100

    study_percent = min(
        study_percent,
        100
    )

    overall_percent = np.mean([
        attendance_percent,
        internal_percent,
        previous_percent,
        assignment_percent,
        study_percent
    ])

    if overall_percent < 50:

        level = "Low Performance"
        indicator = "RED"
        symbol = "🔴"

    elif overall_percent < 65:

        level = "Average Performance"
        indicator = "ORANGE"
        symbol = "🟠"

    elif overall_percent < 80:

        level = "Above Average"
        indicator = "YELLOW"
        symbol = "🟡"

    else:

        level = "Good Performance"
        indicator = "GREEN"
        symbol = "🟢"

    return (
        overall_percent,
        level,
        indicator,
        symbol
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
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
            "Improve class attendance and attend lectures regularly."
        )

    if study_hours < 2:
        recommendations.append(
            "Increase daily study time gradually."
        )

    if internal < 20:
        recommendations.append(
            "Focus more on internal examinations and regular revision."
        )

    if assignment < 8:
        recommendations.append(
            "Complete assignments on time and improve assignment quality."
        )

    if previous < 30:
        recommendations.append(
            "Revise previous topics and strengthen basic concepts."
        )

    if level == "Low Performance":

        recommendations.append(
            "Prepare a weekly study timetable and follow it consistently."
        )

        recommendations.append(
            "Discuss difficult topics with teachers or classmates."
        )

    elif level == "Average Performance":

        recommendations.append(
            "Maintain regular revision and improve consistency."
        )

        recommendations.append(
            "Try additional practice questions before examinations."
        )

    elif level == "Above Average":

        recommendations.append(
            "Small improvements in attendance, internal marks and assignments can further improve performance."
        )

        recommendations.append(
            "Continue regular revision and practice."
        )

    else:

        recommendations.append(
            "Excellent performance. Maintain the same level of consistency."
        )

        recommendations.append(
            "Continue regular study, attendance and assignment submission."
        )

    # Remove duplicates
    recommendations = list(
        dict.fromkeys(recommendations)
    )

    return recommendations


# ============================================================
# NORMALIZE STUDENT INPUT FOR MODEL
# ============================================================

def prepare_model_input(
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    # Dataset model expects marks in approximately 0-100 scale.
    # Student UI uses the requested real maximums.

    internal_model = (
        internal / 40
    ) * 100

    assignment_model = (
        assignment / 15
    ) * 100

    previous_model = (
        previous / 60
    ) * 100

    study_model = (
        study_hours / 6
    ) * 100

    model_input = pd.DataFrame(
        [[
            attendance,
            study_model,
            internal_model,
            assignment_model,
            previous_model
        ]],
        columns=FEATURES
    )

    return model_input


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_subject(
    attendance,
    study_hours,
    internal,
    assignment,
    previous,
    model,
    scaler,
    kmeans
):

    model_input = prepare_model_input(
        attendance,
        study_hours,
        internal,
        assignment,
        previous
    )

    # Random Forest prediction
    prediction = model.predict(
        model_input
    )[0]

    probabilities = model.predict_proba(
        model_input
    )[0]

    confidence = max(
        probabilities
    ) * 100

    # K-Means prediction
    scaled_input = scaler.transform(
        model_input
    )

    cluster = int(
        kmeans.predict(
            scaled_input
        )[0]
    )

    # Project-defined reporting level
    (
        overall_percent,
        level,
        indicator,
        symbol
    ) = calculate_performance_level(
        attendance,
        study_hours,
        internal,
        assignment,
        previous
    )

    recommendations = generate_recommendations(
        attendance,
        study_hours,
        internal,
        assignment,
        previous,
        level
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "cluster": cluster,
        "overall_percent": overall_percent,
        "level": level,
        "indicator": indicator,
        "symbol": symbol,
        "recommendations": recommendations
    }


# ============================================================
# PDF REPORT GENERATION
# ============================================================

def generate_pdf(
    student_name,
    university_id,
    semester,
    branch,
    dob_year,
    results
):

    buffer = BytesIO()

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
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=11
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
            "B.Tech S3 – Artificial Intelligence & Data Science",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "K-Means Clustering + Random Forest Classification",
            subtitle_style
        )
    )

    # --------------------------------------------------------
    # STUDENT INFORMATION
    # --------------------------------------------------------

    story.append(
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
            ["DOB Year", str(dob_year)]
        ],
        colWidths=[150, 350]
    )

    student_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(student_table)

    story.append(Spacer(1, 15))

    # --------------------------------------------------------
    # OVERALL SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Overall Performance Summary",
            heading_style
        )
    )

    summary_data = [
        [
            "Subject",
            "Level",
            "Overall %",
            "Attendance",
            "Internal",
            "Previous",
            "Assignment"
        ]
    ]

    for subject, result in results.items():

        summary_data.append([
            subject,
            result["level"],
            f'{result["overall_percent"]:.1f}%',
            f'{result["attendance"]:.0f}%',
            f'{result["internal"]:.0f}/40',
            f'{result["previous"]:.0f}/60',
            f'{result["assignment"]:.0f}/15'
        ])

    summary_table = Table(
        summary_data,
        repeatRows=1,
        colWidths=[
            105,
            80,
            55,
            55,
            55,
            55,
            60
        ]
    )

    summary_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (2, 1), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
        ])
    )

    story.append(summary_table)

    story.append(Spacer(1, 15))

    # --------------------------------------------------------
    # SUBJECT DETAILS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Subject-wise Detailed Performance",
            heading_style
        )
    )

    for subject, result in results.items():

        story.append(
            Paragraph(
                subject,
                ParagraphStyle(
                    "SubjectHeading",
                    parent=styles["Heading3"],
                    fontSize=12,
                    spaceBefore=10,
                    spaceAfter=6
                )
            )
        )

        detail_data = [
            ["Parameter", "Value"],
            ["Performance Level", result["level"]],
            ["Indicator", result["indicator"]],
            ["Overall Indicator %", f'{result["overall_percent"]:.2f}%'],
            ["ML Prediction", result["prediction"]],
            ["Model Confidence", f'{result["confidence"]:.2f}%'],
            ["K-Means Cluster", str(result["cluster"])],
            ["Attendance", f'{result["attendance"]:.0f}%'],
            [
                "Attendance Converted Mark",
                f'{result["attendance_mark"]}/5'
            ],
            ["Internal Mark", f'{result["internal"]:.0f}/40'],
            ["Previous Mark", f'{result["previous"]:.0f}/60'],
            ["Assignment Score", f'{result["assignment"]:.0f}/15'],
            ["Study Hours", f'{result["study_hours"]:.1f}/6 hours/day']
        ]

        detail_table = Table(
            detail_data,
            colWidths=[200, 300]
        )

        detail_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
            ])
        )

        story.append(detail_table)

        story.append(Spacer(1, 8))

        story.append(
            Paragraph(
                "<b>Improvement / Maintenance Suggestions:</b>",
                normal_style
            )
        )

        for recommendation in result["recommendations"]:

            story.append(
                Paragraph(
                    "• " + recommendation,
                    normal_style
                )
            )

        story.append(Spacer(1, 10))

    # --------------------------------------------------------
    # ATTENDANCE CONVERSION
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "4. Attendance Mark Conversion",
            heading_style
        )
    )

    attendance_table = Table(
        [
            ["Attendance", "Converted Mark"],
            ["90–100%", "5 / 5"],
            ["80–89%", "4 / 5"],
            ["70–79%", "3 / 5"],
            ["60–69%", "2 / 5"],
            ["10–59%", "1 / 5"],
            ["Below 10%", "0 / 5"]
        ],
        colWidths=[200, 200]
    )

    attendance_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(attendance_table)

    story.append(Spacer(1, 15))

    # --------------------------------------------------------
    # PERFORMANCE LEVEL
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Performance Indicator",
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
        colWidths=[150, 250]
    )

    indicator_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(indicator_table)

    story.append(Spacer(1, 20))

    # --------------------------------------------------------
    # NOTE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Note:</b> The performance indicator scale used in this "
            "project is a project-defined monitoring scale. It is not "
            "an official KTU grading standard. The machine-learning "
            "prediction is intended to support academic monitoring and "
            "should not replace teacher evaluation.",
            small_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# LOGIN SCREEN
# ============================================================

def login_screen():

    st.markdown(
        '<div class="main-title">🎓 Student Performance Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">B.Tech S3 – Artificial Intelligence & Data Science</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Choose Login")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):

            st.session_state.login_type = "student"
            st.rerun()

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):

            st.session_state.login_type = "teacher"
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
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        valid = False

        if len(password) == 9:

            if password[:5].upper() == "BTECH":

                try:

                    year = int(
                        password[5:]
                    )

                    if 2000 <= year <= 2020:
                        valid = True

                except ValueError:
                    valid = False

        if valid:

            st.session_state.logged_in = True
            st.session_state.role = "student"
            st.session_state.dob_year = int(
                password[5:]
            )

            st.rerun()

        else:

            st.error(
                "Invalid student password."
            )

    if st.button(
        "← Back"
    ):

        st.session_state.login_type = None
        st.rerun()


# ============================================================
# TEACHER LOGIN
# ============================================================

def teacher_login():

    st.title("👨‍🏫 Teacher Login")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        width="stretch"
    ):

        if password == TEACHER_PASSWORD:

            st.session_state.logged_in = True
            st.session_state.role = "teacher"

            st.rerun()

        else:

            st.error(
                "Invalid teacher password."
            )

    if st.button(
        "← Back"
    ):

        st.session_state.login_type = None
        st.rerun()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    (
        data,
        scaler,
        kmeans,
        model,
        accuracy,
        silhouette,
        report,
        report_text,
        matrix
    ) = train_models()

    st.title(
        "🎓 Student Dashboard"
    )

    st.caption(
        "Student Performance Prediction using K-Means and Random Forest"
    )

    # --------------------------------------------------------
    # TOP BAR
    # --------------------------------------------------------

    col1, col2 = st.columns([4, 1])

    with col1:

        st.success(
            "Student Login Successful"
        )

    with col2:

        if st.button(
            "Logout"
        ):

            st.session_state.clear()
            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # STUDENT INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "1. Student Information"
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
            ],
            index=2
        )

        branch = st.text_input(
            "Branch",
            value="B.Tech AI & Data Science"
        )

    st.divider()

    # --------------------------------------------------------
    # SUBJECT INPUTS
    # --------------------------------------------------------

    st.subheader(
        "2. Subject-wise Academic Details"
    )

    st.info(
        "Enter the academic details separately for each subject."
    )

    subject_inputs = {}

    for subject in SUBJECTS:

        st.markdown(
            f"### 📘 {subject}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
                key=f"{subject}_attendance"
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=6.0,
                value=3.0,
                step=0.5,
                key=f"{subject}_study"
            )

        with col2:

            internal = st.number_input(
                "Internal Mark / 40",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                step=1.0,
                key=f"{subject}_internal"
            )

            assignment = st.number_input(
                "Assignment / 15",
                min_value=0.0,
                max_value=15.0,
                value=10.0,
                step=1.0,
                key=f"{subject}_assignment"
            )

        with col3:

            previous = st.number_input(
                "Previous Mark / 60",
                min_value=0.0,
                max_value=60.0,
                value=35.0,
                step=1.0,
                key=f"{subject}_previous"
            )

            st.write(
                f"Attendance converted mark: "
                f"**{attendance_mark(attendance)}/5**"
            )

        subject_inputs[subject] = {
            "attendance": attendance,
            "study_hours": study_hours,
            "internal": internal,
            "assignment": assignment,
            "previous": previous
        }

        st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Student Performance",
        type="primary",
        width="stretch"
    ):

        if not student_name.strip():

            st.warning(
                "Please enter the Student Name."
            )

            return

        if not university_id.strip():

            st.warning(
                "Please enter the University ID."
            )

            return

        results = {}

        with st.spinner(
            "Analysing student performance..."
        ):

            for subject, values in subject_inputs.items():

                prediction_result = predict_subject(
                    values["attendance"],
                    values["study_hours"],
                    values["internal"],
                    values["assignment"],
                    values["previous"],
                    model,
                    scaler,
                    kmeans
                )

                results[subject] = {
                    **values,
                    **prediction_result,
                    "attendance_mark": attendance_mark(
                        values["attendance"]
                    )
                }

        st.session_state.student_results = results
        st.session_state.student_name = student_name
        st.session_state.university_id = university_id
        st.session_state.semester = semester
        st.session_state.branch = branch

    # --------------------------------------------------------
    # SHOW RESULTS
    # --------------------------------------------------------

    if "student_results" in st.session_state:

        results = st.session_state.student_results

        st.divider()

        st.subheader(
            "3. Subject-wise Performance"
        )

        for subject, result in results.items():

            level = result["level"]

            if level == "Low Performance":
                card_class = "low-card"

            elif level == "Average Performance":
                card_class = "average-card"

            elif level == "Above Average":
                card_class = "above-card"

            else:
                card_class = "good-card"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <h3>{result["symbol"]} {subject}</h3>
                    <b>Performance:</b> {result["level"]}<br>
                    <b>Overall Indicator:</b>
                    {result["overall_percent"]:.1f}%<br>
                    <b>ML Prediction:</b>
                    {result["prediction"]}<br>
                    <b>Confidence:</b>
                    {result["confidence"]:.1f}%<br>
                    <b>K-Means Cluster:</b>
                    {result["cluster"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # OVERALL TABLE
        # ----------------------------------------------------

        st.subheader(
            "4. Overall Academic Information"
        )

        table_rows = []

        for subject, result in results.items():

            table_rows.append({
                "Subject": subject,
                "Attendance": f'{result["attendance"]:.0f}%',
                "Attendance Mark": f'{result["attendance_mark"]}/5',
                "Internal": f'{result["internal"]:.0f}/40',
                "Previous": f'{result["previous"]:.0f}/60',
                "Assignment": f'{result["assignment"]:.0f}/15',
                "Study Hours": f'{result["study_hours"]:.1f}/6',
                "ML Prediction": result["prediction"],
                "Performance": result["level"]
            })

        df_results = pd.DataFrame(
            table_rows
        )

        st.dataframe(
            df_results,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # SUBJECT DETAILS
        # ----------------------------------------------------

        st.subheader(
            "5. Improvement / Maintenance Suggestions"
        )

        for subject, result in results.items():

            with st.expander(
                f'{result["symbol"]} {subject} – {result["level"]}'
            ):

                for recommendation in result["recommendations"]:

                    st.write(
                        "• " + recommendation
                    )

        # ----------------------------------------------------
        # DOWNLOAD PDF
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "6. Download Progress Report"
        )

        pdf_file = generate_pdf(
            st.session_state.student_name,
            st.session_state.university_id,
            st.session_state.semester,
            st.session_state.branch,
            st.session_state.dob_year,
            results
        )

        st.download_button(
            label="📥 Download Progress Report (PDF)",
            data=pdf_file,
            file_name=(
                f'{st.session_state.university_id}'
                "_Progress_Report.pdf"
            ),
            mime="application/pdf",
            width="stretch"
        )

        st.caption(
            "The report contains student information, "
            "subject-wise marks, performance indicators, "
            "ML predictions, clusters and recommendations."
        )


# ============================================================
# TEACHER DASHBOARD
# ============================================================

def teacher_dashboard():

    (
        data,
        scaler,
        kmeans,
        model,
        accuracy,
        silhouette,
        report,
        report_text,
        matrix
    ) = train_models()

    st.title(
        "👨‍🏫 Teacher Dashboard"
    )

    st.caption(
        "Student Performance Prediction – Model Monitoring"
    )

    if st.button(
        "Logout"
    ):

        st.session_state.clear()
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "📊 Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "K-Means Silhouette Score",
            f"{silhouette:.4f}"
        )

    with col3:

        st.metric(
            "Dataset Records",
            len(data)
        )

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "📋 Classification Report"
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        width="stretch"
    )

    with st.expander(
        "View Classification Report Text"
    ):

        st.code(
            report_text
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "🔢 Confusion Matrix"
    )

    labels = sorted(
        data["Performance"].unique()
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    st.dataframe(
        matrix_df,
        width="stretch"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🌳 Random Forest Feature Importance"
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )

    st.dataframe(
        importance_df.round(4),
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # K-MEANS CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "🔵 K-Means Cluster Distribution"
    )

    cluster_counts = (
        data["Cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_df = pd.DataFrame({
        "Cluster": cluster_counts.index.astype(str),
        "Students": cluster_counts.values
    })

    st.bar_chart(
        cluster_df.set_index(
            "Cluster"
        )
    )

    # --------------------------------------------------------
    # PERFORMANCE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📈 Performance Distribution"
    )

    performance_counts = (
        data["Performance"]
        .value_counts()
    )

    performance_df = pd.DataFrame({
        "Performance": performance_counts.index,
        "Students": performance_counts.values
    })

    st.bar_chart(
        performance_df.set_index(
            "Performance"
        )
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader(
        "📁 Student Dataset"
    )

    st.dataframe(
        data,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD DATASET
    # --------------------------------------------------------

    csv_data = data.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Dataset",
        data=csv_data,
        file_name="student_performance_dataset.csv",
        mime="text/csv",
        width="stretch"
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🤖 Machine Learning Information"
    )

    st.write(
        """
        **K-Means Clustering**

        K-Means is an unsupervised machine learning algorithm.
        It groups students into three clusters based on their
        academic characteristics.

        **Random Forest Classification**

        Random Forest is a supervised machine learning algorithm.
        It uses multiple decision trees to predict the student's
        performance category.

        **Features used:**

        - Attendance
        - Study Hours
        - Internal Mark
        - Assignment Score
        - Previous Mark

        **Performance classes:**

        - High
        - Medium
        - Low
        """
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # INITIAL SESSION STATE
    # --------------------------------------------------------

    if "login_type" not in st.session_state:

        st.session_state.login_type = None

    if "logged_in" not in st.session_state:

        st.session_state.logged_in = False

    # --------------------------------------------------------
    # LOGGED-IN USER
    # --------------------------------------------------------

    if st.session_state.logged_in:

        if st.session_state.role == "student":

            student_dashboard()

        elif st.session_state.role == "teacher":

            teacher_dashboard()

        return

    # --------------------------------------------------------
    # LOGIN SELECTION
    # --------------------------------------------------------

    if st.session_state.login_type is None:

        login_screen()

    elif st.session_state.login_type == "student":

        student_login()

    elif st.session_state.login_type == "teacher":

        teacher_login()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
