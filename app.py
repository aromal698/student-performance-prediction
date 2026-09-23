import streamlit as st
import pandas as pd
import io

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
# TITLE
# =========================================================

st.title("🎓 Student Performance Prediction")


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/student_performance.csv")


try:
    data = load_data()

except FileNotFoundError:
    st.error(
        "Dataset not found.\n\n"
        "Please make sure the file is located at:\n"
        "data/student_performance.csv"
    )
    st.stop()


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

REQUIRED_COLUMNS = FEATURES + ["Performance"]


missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in data.columns
]

if missing_columns:
    st.error(
        f"Missing columns in dataset: {missing_columns}"
    )
    st.stop()


# =========================================================
# K-MEANS
# =========================================================

X = data[FEATURES]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(
    X_scaled
)


# =========================================================
# RANDOM FOREST
# =========================================================

y = data["Performance"]


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


y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


# =========================================================
# SESSION STATE
# =========================================================

if "login_type" not in st.session_state:
    st.session_state.login_type = None

if "student_logged_in" not in st.session_state:
    st.session_state.student_logged_in = False

if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False


# =========================================================
# LOGIN PAGE
# =========================================================

if (
    not st.session_state.student_logged_in
    and not st.session_state.teacher_logged_in
):

    st.subheader("Choose Login")

    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # STUDENT
    # -----------------------------------------------------

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):

            st.session_state.login_type = "student"
            st.rerun()


    # -----------------------------------------------------
    # TEACHER
    # -----------------------------------------------------

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):

            st.session_state.login_type = "teacher"
            st.rerun()


    # =====================================================
    # STUDENT LOGIN
    # =====================================================

    if st.session_state.login_type == "student":

        st.divider()

        st.header("🔐 Student Login")

        st.info(
            "Student password must contain BTECH "
            "followed by a 4-digit DOB year."
        )


        student_password = st.text_input(
            "Student Password",
            type="password",
            max_chars=9,
            placeholder="BTECH2003"
        )


        if st.button(
            "🔓 Student Login",
            type="primary",
            width="stretch"
        ):

            password = (
                student_password
                .strip()
                .upper()
            )


            valid_password = (
                len(password) == 9
                and password[:5] == "BTECH"
                and password[5:].isdigit()
                and 2000 <= int(password[5:]) <= 2020
            )


            if valid_password:

                st.session_state.student_logged_in = True

                st.session_state.dob_year = password[5:]

                st.session_state.login_type = None

                st.rerun()

            else:

                st.error(
                    "❌ Invalid student password."
                )

                st.warning(
                    "Password must be exactly 9 characters."
                )


    # =====================================================
    # TEACHER LOGIN
    # =====================================================

    if st.session_state.login_type == "teacher":

        st.divider()

        st.header("🔐 Teacher Login")


        teacher_password = st.text_input(
            "Teacher Password",
            type="password"
        )


        if st.button(
            "🔓 Teacher Login",
            type="primary",
            width="stretch"
        ):

            if teacher_password == "*ktutech":

                st.session_state.teacher_logged_in = True

                st.session_state.login_type = None

                st.rerun()

            else:

                st.error(
                    "❌ Invalid teacher password."
                )


    st.stop()


# =========================================================
# ATTENDANCE MARK CALCULATION
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

def get_performance_level(
    internal_mark,
    previous_mark,
    assignment,
    attendance,
    study_hours
):

    attendance_marks = attendance_mark(
        attendance
    )


    # Normalize academic marks to 100
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
        study_hours / 6 * 100,
        100
    )


    attendance_percent = (
        attendance_marks / 5
    ) * 100


    overall_percent = (
        attendance_percent
        + internal_percent
        + previous_percent
        + assignment_percent
        + study_percent
    ) / 5


    if overall_percent < 50:

        return (
            "Low Performance",
            "🔴",
            overall_percent
        )


    elif overall_percent < 65:

        return (
            "Average Performance",
            "🟠",
            overall_percent
        )


    elif overall_percent < 80:

        return (
            "Above Average",
            "🟡",
            overall_percent
        )


    else:

        return (
            "Good Performance",
            "🟢",
            overall_percent
        )


# =========================================================
# RECOMMENDATION
# =========================================================

def get_recommendation(
    level,
    attendance,
    internal,
    previous,
    assignment,
    study_hours
):

    if level == "Low Performance":

        advice = []

        if attendance < 80:
            advice.append(
                "Improve attendance and attend classes regularly."
            )

        if internal < 24:
            advice.append(
                "Revise important topics regularly to improve internal marks."
            )

        if previous < 36:
            advice.append(
                "Practice previous questions and strengthen basic concepts."
            )

        if assignment < 9:
            advice.append(
                "Complete assignments on time and review corrections."
            )

        if study_hours < 3:
            advice.append(
                "Increase focused study time gradually."
            )

        if not advice:
            advice.append(
                "Follow a consistent study timetable and revise regularly."
            )

        return " ".join(advice)


    elif level == "Average Performance":

        return (
            "Performance is satisfactory. "
            "Improve regular revision, attendance, assignments "
            "and study consistency to move to the next level."
        )


    elif level == "Above Average":

        return (
            "Good progress. Continue regular study and revision. "
            "Small improvements in internal marks, attendance "
            "and assignment performance can lead to good performance."
        )


    else:

        return (
            "Excellent work! Maintain regular attendance, "
            "consistent study habits and timely assignment submission."
        )


# =========================================================
# PDF GENERATION
# =========================================================

def create_progress_report(
    student_info,
    results
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
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=15
    )


    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8
    )


    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13
    )


    story = []


    # =====================================================
    # TITLE
    # =====================================================

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


    story.append(
        Spacer(1, 12)
    )


    # =====================================================
    # STUDENT INFORMATION
    # =====================================================

    story.append(
        Paragraph(
            "1. Student Information",
            heading_style
        )
    )


    student_table = Table(
        [
            ["Student Name", student_info["name"]],
            ["University ID", student_info["id"]],
            ["Semester", student_info["semester"]],
            ["Branch", student_info["branch"]],
            ["DOB Year", student_info["dob_year"]]
        ],
        colWidths=[140, 350]
    )


    student_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9)
            ]
        )
    )


    story.append(
        student_table
    )


    story.append(
        Spacer(1, 15)
    )


    # =====================================================
    # OVERALL SUMMARY
    # =====================================================

    story.append(
        Paragraph(
            "2. Overall Performance Summary",
            heading_style
        )
    )


    overall_rows = [
        [
            "Subject",
            "Performance",
            "Overall %",
            "Attendance",
            "Internal",
            "Previous",
            "Assignment"
        ]
    ]


    total_percent = 0


    for r in results:

        total_percent += r["Overall %"]

        overall_rows.append(
            [
                r["Subject"],
                f'{r["Circle"]} {r["Level"]}',
                f'{r["Overall %"]:.1f}%',
                f'{r["Attendance"]:.0f}%',
                f'{r["Internal"]:.0f}/40',
                f'{r["Previous"]:.0f}/60',
                f'{r["Assignment"]:.0f}/15'
            ]
        )


    average_percent = (
        total_percent / len(results)
        if results
        else 0
    )


    overall_rows.append(
        [
            "OVERALL",
            "",
            f"{average_percent:.1f}%",
            "",
            "",
            "",
            ""
        ]
    )


    overall_table = Table(
        overall_rows,
        repeatRows=1,
        colWidths=[
            70,
            100,
            55,
            55,
            55,
            55,
            55
        ]
    )


    overall_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
            ]
        )
    )


    story.append(
        overall_table
    )


    story.append(
        Spacer(1, 20)
    )


    # =====================================================
    # EACH SUBJECT
    # =====================================================

    story.append(
        Paragraph(
            "3. Subject-wise Detailed Progress",
            heading_style
        )
    )


    for index, r in enumerate(results):

        story.append(
            Paragraph(
                f'{r["Circle"]} {r["Subject"]}',
                heading_style
            )
        )


        details = [
            ["Performance Level", f'{r["Circle"]} {r["Level"]}'],
            ["Overall Percentage", f'{r["Overall %"]:.1f}%'],
            ["Prediction", r["Prediction"]],
            ["Model Confidence", f'{r["Confidence"]:.1f}%'],
            ["K-Means Cluster", str(r["Cluster"])],
            ["Attendance", f'{r["Attendance"]:.0f}%'],
            ["Attendance Mark", f'{r["Attendance Mark"]}/5'],
            ["Internal Mark", f'{r["Internal"]:.0f}/40'],
            ["Previous Mark", f'{r["Previous"]:.0f}/60'],
            ["Assignment Score", f'{r["Assignment"]:.0f}/15'],
            ["Study Hours", f'{r["Study Hours"]:.1f} hours/day']
        ]


        detail_table = Table(
            details,
            colWidths=[170, 320]
        )


        detail_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.lightgrey
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
                ]
            )
        )


        story.append(
            detail_table
        )


        story.append(
            Spacer(1, 8)
        )


        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

        story.append(
            Paragraph(
                "<b>Improvement / Appreciation:</b>",
                normal_style
            )
        )


        story.append(
            Paragraph(
                r["Recommendation"],
                normal_style
            )
        )


        story.append(
            Spacer(1, 15)
        )


        # Page break after every 2 subjects

        if (index + 1) % 2 == 0 and index != len(results) - 1:

            story.append(
                PageBreak()
            )


    # =====================================================
    # GRADING INFORMATION
    # =====================================================

    story.append(
        Paragraph(
            "4. Attendance Mark Conversion",
            heading_style
        )
    )


    attendance_table = Table(
        [
            ["Attendance", "Attendance Mark"],
            ["90–100%", "5 / 5"],
            ["80–89%", "4 / 5"],
            ["70–79%", "3 / 5"],
            ["60–69%", "2 / 5"],
            ["10–59%", "1 / 5"]
        ],
        colWidths=[250, 240]
    )


    attendance_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (1, -1), "CENTER")
            ]
        )
    )


    story.append(
        attendance_table
    )


    story.append(
        Spacer(1, 15)
    )


    # =====================================================
    # PERFORMANCE LEGEND
    # =====================================================

    story.append(
        Paragraph(
            "5. Performance Indicator",
            heading_style
        )
    )


    legend = Table(
        [
            ["Indicator", "Meaning"],
            ["🔴", "Low Performance"],
            ["🟠", "Average Performance"],
            ["🟡", "Above Average"],
            ["🟢", "Good Performance"]
        ],
        colWidths=[100, 390]
    )


    legend.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 1), (0, -1), "CENTER")
            ]
        )
    )


    story.append(
        legend
    )


    story.append(
        Spacer(1, 15)
    )


    story.append(
        Paragraph(
            "<b>Note:</b> This progress report is generated "
            "using the student inputs and machine-learning "
            "prediction system. It is intended to support "
            "academic monitoring and should be considered "
            "along with teacher evaluation.",
            normal_style
        )
    )


    document.build(
        story
    )


    buffer.seek(0)

    return buffer


# =========================================================
# STUDENT DASHBOARD
# =========================================================

if st.session_state.student_logged_in:

    st.header("🎓 Student Dashboard")

    st.success(
        "Student Login Successful"
    )


    # =====================================================
    # STUDENT INFORMATION
    # =====================================================

    st.subheader(
        "👤 Student Information"
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
            "Branch",
            [
                "AI & Data Science",
                "Computer Science",
                "Information Technology",
                "Electronics & Communication",
                "Electrical & Electronics",
                "Mechanical Engineering",
                "Civil Engineering"
            ]
        )


    st.divider()


    # =====================================================
    # SUBJECT NAMES
    # =====================================================

    st.subheader(
        "📚 Six Subjects"
    )


    subject_names = []


    subject_columns = st.columns(3)


    for i in range(6):

        with subject_columns[i % 3]:

            subject = st.text_input(
                f"Subject {i + 1}",
                placeholder=f"Enter Subject {i + 1}",
                key=f"subject_name_{i}"
            )

            subject_names.append(
                subject
            )


    st.divider()


    # =====================================================
    # ACADEMIC INPUT
    # =====================================================

    st.subheader(
        "📊 Academic Information"
    )


    st.write(
        "Enter marks separately for each subject."
    )


    subject_inputs = []


    for i in range(6):

        subject = subject_names[i].strip()


        if subject == "":

            subject = f"Subject {i + 1}"


        st.markdown(
            f"### 📘 {subject}"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=10.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
                key=f"attendance_{i}"
            )


            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=6.0,
                value=3.0,
                step=0.5,
                key=f"study_{i}"
            )


        with col2:

            internal = st.number_input(
                "Internal Mark (Max 40)",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                step=1.0,
                key=f"internal_{i}"
            )


            assignment = st.number_input(
                "Assignment Score (Max 15)",
                min_value=0.0,
                max_value=15.0,
                value=10.0,
                step=1.0,
                key=f"assignment_{i}"
            )


        with col3:

            previous = st.number_input(
                "Previous Mark (Max 60)",
                min_value=0.0,
                max_value=60.0,
                value=35.0,
                step=1.0,
                key=f"previous_{i}"
            )


        subject_inputs.append(
            {
                "Subject": subject,
                "Attendance": attendance,
                "Study_Hours": study_hours,
                "Internal": internal,
                "Assignment": assignment,
                "Previous": previous
            }
        )


        st.divider()


    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button(
        "🔮 Predict Performance",
        type="primary",
        width="stretch"
    ):


        if student_name.strip() == "":

            st.error(
                "Please enter student name."
            )

            st.stop()


        if university_id.strip() == "":

            st.error(
                "Please enter university ID."
            )

            st.stop()


        results = []


        # =================================================
        # EACH SUBJECT
        # =================================================

        for subject_data in subject_inputs:


            # ---------------------------------------------
            # Convert user marks to model scale
            # ---------------------------------------------

            model_internal = (
                subject_data["Internal"] / 40
            ) * 100


            model_previous = (
                subject_data["Previous"] / 60
            ) * 100


            model_assignment = (
                subject_data["Assignment"] / 15
            ) * 100


            input_data = pd.DataFrame(
                [
                    {
                        "Attendance":
                            subject_data["Attendance"],

                        "Study_Hours":
                            subject_data["Study_Hours"],

                        "Internal_Mark":
                            model_internal,

                        "Assignment":
                            model_assignment,

                        "Previous_Mark":
                            model_previous
                    }
                ]
            )


            # ---------------------------------------------
            # RANDOM FOREST
            # ---------------------------------------------

            prediction = model.predict(
                input_data
            )[0]


            probabilities = model.predict_proba(
                input_data
            )[0]


            confidence = (
                probabilities.max() * 100
            )


            # ---------------------------------------------
            # K-MEANS
            # ---------------------------------------------

            scaled_input = scaler.transform(
                input_data
            )


            cluster = int(
                kmeans.predict(
                    scaled_input
                )[0]
            )


            # ---------------------------------------------
            # ATTENDANCE MARK
            # ---------------------------------------------

            att_mark = attendance_mark(
                subject_data["Attendance"]
            )


            # ---------------------------------------------
            # PERFORMANCE LEVEL
            # ---------------------------------------------

            level, circle, overall_percent = (
                get_performance_level(
                    subject_data["Internal"],
                    subject_data["Previous"],
                    subject_data["Assignment"],
                    subject_data["Attendance"],
                    subject_data["Study_Hours"]
                )
            )


            # ---------------------------------------------
            # RECOMMENDATION
            # ---------------------------------------------

            recommendation = get_recommendation(
                level,
                subject_data["Attendance"],
                subject_data["Internal"],
                subject_data["Previous"],
                subject_data["Assignment"],
                subject_data["Study_Hours"]
            )


            results.append(
                {
                    "Subject":
                        subject_data["Subject"],

                    "Prediction":
                        prediction,

                    "Confidence":
                        confidence,

                    "Cluster":
                        cluster,

                    "Attendance":
                        subject_data["Attendance"],

                    "Attendance Mark":
                        att_mark,

                    "Internal":
                        subject_data["Internal"],

                    "Previous":
                        subject_data["Previous"],

                    "Assignment":
                        subject_data["Assignment"],

                    "Study Hours":
                        subject_data["Study_Hours"],

                    "Level":
                        level,

                    "Circle":
                        circle,

                    "Overall %":
                        overall_percent,

                    "Recommendation":
                        recommendation
                }
            )


        # =================================================
        # SAVE RESULTS
        # =================================================

        st.session_state["results"] = results

        st.session_state["student_info"] = {
            "name": student_name,
            "id": university_id,
            "semester": semester,
            "branch": branch,
            "dob_year":
                st.session_state.get(
                    "dob_year",
                    ""
                )
        }


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    if "results" in st.session_state:

        results = st.session_state["results"]


        st.divider()


        st.subheader(
            "📊 Subject-wise Performance"
        )


        # -------------------------------------------------
        # SUBJECT CARDS
        # -------------------------------------------------

        for r in results:

            if r["Level"] == "Low Performance":

                st.error(
                    f"{r['Circle']}  "
                    f"**{r['Subject']}** — "
                    f"{r['Level']} | "
                    f"{r['Overall %']:.1f}%"
                )

            elif r["Level"] == "Average Performance":

                st.warning(
                    f"{r['Circle']}  "
                    f"**{r['Subject']}** — "
                    f"{r['Level']} | "
                    f"{r['Overall %']:.1f}%"
                )

            elif r["Level"] == "Above Average":

                st.info(
                    f"{r['Circle']}  "
                    f"**{r['Subject']}** — "
                    f"{r['Level']} | "
                    f"{r['Overall %']:.1f}%"
                )

            else:

                st.success(
                    f"{r['Circle']}  "
                    f"**{r['Subject']}** — "
                    f"{r['Level']} | "
                    f"{r['Overall %']:.1f}%"
                )


        # -------------------------------------------------
        # DETAILED TABLE
        # -------------------------------------------------

        st.subheader(
            "📋 Overall Information"
        )


        result_table = []


        for r in results:

            result_table.append(
                {
                    "Subject":
                        r["Subject"],

                    "Attendance":
                        f'{r["Attendance"]:.0f}%',

                    "Attendance Mark":
                        f'{r["Attendance Mark"]}/5',

                    "Internal":
                        f'{r["Internal"]:.0f}/40',

                    "Previous":
                        f'{r["Previous"]:.0f}/60',

                    "Assignment":
                        f'{r["Assignment"]:.0f}/15',

                    "Study Hours":
                        f'{r["Study Hours"]:.1f}',

                    "Performance":
                        f'{r["Circle"]} {r["Level"]}',

                    "Overall %":
                        f'{r["Overall %"]:.1f}%'
                }
            )


        st.dataframe(
            pd.DataFrame(result_table),
            width="stretch",
            hide_index=True
        )


        # -------------------------------------------------
        # SUBJECT DETAILS
        # -------------------------------------------------

        st.subheader(
            "📚 Subject-wise Details"
        )


        for r in results:

            with st.expander(
                f'{r["Circle"]} {r["Subject"]}'
            ):

                col1, col2 = st.columns(2)


                with col1:

                    st.write(
                        f'**Attendance:** '
                        f'{r["Attendance"]:.0f}%'
                    )

                    st.write(
                        f'**Attendance Mark:** '
                        f'{r["Attendance Mark"]}/5'
                    )

                    st.write(
                        f'**Internal Mark:** '
                        f'{r["Internal"]:.0f}/40'
                    )

                    st.write(
                        f'**Previous Mark:** '
                        f'{r["Previous"]:.0f}/60'
                    )


                with col2:

                    st.write(
                        f'**Assignment:** '
                        f'{r["Assignment"]:.0f}/15'
                    )

                    st.write(
                        f'**Study Hours:** '
                        f'{r["Study Hours"]:.1f} hours/day'
                    )

                    st.write(
                        f'**ML Prediction:** '
                        f'{r["Prediction"]}'
                    )

                    st.write(
                        f'**Confidence:** '
                        f'{r["Confidence"]:.1f}%'
                    )


                st.markdown(
                    f'### {r["Circle"]} {r["Level"]}'
                )


                st.write(
                    f'**Overall Score:** '
                    f'{r["Overall %"]:.1f}%'
                )


                if r["Level"] == "Good Performance":

                    st.success(
                        "🌟 Excellent work! "
                        "Keep maintaining this performance."
                    )


                elif r["Level"] == "Above Average":

                    st.info(
                        "👍 Good progress! "
                        "Small improvements in marks and "
                        "internal performance can move you "
                        "to the good-performance level."
                    )


                elif r["Level"] == "Average Performance":

                    st.warning(
                        "📈 Your performance is satisfactory. "
                        "Regular revision and improvement "
                        "in attendance, internal marks and "
                        "assignments can help you progress."
                    )


                else:

                    st.error(
                        "⚠️ Improvement is needed. "
                        "Focus on attendance, regular study, "
                        "assignments and revision."
                    )


                st.write(
                    f'**Recommendation:** '
                    f'{r["Recommendation"]}'
                )


        # =================================================
        # DOWNLOAD PDF
        # =================================================

        st.divider()


        st.subheader(
            "📄 Progress Report"
        )


        student_info = st.session_state[
            "student_info"
        ]


        pdf_file = create_progress_report(
            student_info,
            results
        )


        filename = (
            f'{student_info["id"]}_'
            f'Progress_Report.pdf'
        )


        st.download_button(
            label="📥 Download Progress Report (PDF)",
            data=pdf_file,
            file_name=filename,
            mime="application/pdf",
            width="stretch"
        )


    # =====================================================
    # LOGOUT
    # =====================================================

    st.divider()


    if st.button(
        "🚪 Student Logout",
        width="stretch"
    ):

        st.session_state.student_logged_in = False

        st.session_state.login_type = None

        if "results" in st.session_state:
            del st.session_state["results"]

        if "student_info" in st.session_state:
            del st.session_state["student_info"]

        st.rerun()


    st.stop()


# =========================================================
# TEACHER DASHBOARD
# =========================================================

if st.session_state.teacher_logged_in:

    st.header("👨‍🏫 Teacher Dashboard")

    st.success(
        "Teacher Login Successful"
    )


    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.subheader(
        "📊 Dataset Overview"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Students",
            len(data)
        )


    with col2:

        st.metric(
            "Features",
            len(FEATURES)
        )


    with col3:

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with col4:

        st.metric(
            "Clusters",
            3
        )


    st.divider()


    # =====================================================
    # DATASET
    # =====================================================

    st.subheader(
        "📁 Student Dataset"
    )


    st.dataframe(
        data,
        width="stretch",
        hide_index=True
    )


    st.divider()


    # =====================================================
    # PERFORMANCE DISTRIBUTION
    # =====================================================

    st.subheader(
        "📈 Performance Distribution"
    )


    performance_counts = (
        data["Performance"]
        .value_counts()
    )


    st.bar_chart(
        performance_counts
    )


    st.divider()


    # =====================================================
    # CLUSTER DISTRIBUTION
    # =====================================================

    st.subheader(
        "🔵 K-Means Cluster Distribution"
    )


    cluster_counts = (
        data["Cluster"]
        .value_counts()
        .sort_index()
    )


    st.bar_chart(
        cluster_counts
    )


    st.divider()


    # =====================================================
    # MODEL ACCURACY
    # =====================================================

    st.subheader(
        "🤖 Random Forest Model"
    )


    st.write(
        f"**Model Accuracy:** "
        f"{accuracy * 100:.2f}%"
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader(
        "📋 Classification Report"
    )


    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    st.dataframe(
        report_df.round(2),
        width="stretch"
    )


    st.divider()


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader(
        "📌 Feature Importance"
    )


    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance":
                model.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    )


    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )


    st.divider()


    # =====================================================
    # TEACHER LOGOUT
    # =====================================================

    if st.button(
        "🚪 Teacher Logout",
        width="stretch"
    ):

        st.session_state.teacher_logged_in = False

        st.session_state.login_type = None

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Student Performance Prediction System"
)
