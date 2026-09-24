import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from io import BytesIO

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
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
# FILE STORAGE
# ============================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "student_records.csv"

# ============================================================
# DEPARTMENTS
# ============================================================

DEPARTMENTS = [
    "Artificial Intelligence and Data Science",
    "Computer Science and Engineering",
    "Computer Science and Engineering (Artificial Intelligence)",
    "Computer Science and Engineering (Data Science)",
    "Computer Science and Engineering (Cyber Security)",
    "Artificial Intelligence and Machine Learning",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Biotechnology and Biochemical Engineering",
    "Food Technology",
    "Production Engineering",
    "Automobile Engineering",
    "Industrial Engineering",
    "Biomedical Engineering",
    "Aeronautical Engineering",
    "Applied Electronics and Instrumentation Engineering",
    "Electronics and Biomedical Engineering",
    "Electronics and Computer Engineering",
    "Robotics and Automation",
    "Mechatronics Engineering"
]

SEMESTERS = [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8"
]

# ============================================================
# SUBJECT DATABASE
#
# IMPORTANT:
# This is a PROJECT STRUCTURE.
# Replace the subject names with your college's exact
# KTU curriculum if your department provides different
# subjects/electives.
# ============================================================

COMMON_SUBJECTS = {

    "S1": [
        "Mathematics I",
        "Physics",
        "Engineering Graphics",
        "Engineering Mechanics",
        "Programming Fundamentals",
        "Life Skills"
    ],

    "S2": [
        "Mathematics II",
        "Chemistry",
        "Basic Electrical Engineering",
        "Engineering Workshop",
        "Programming",
        "Professional Communication"
    ],

    "S3": [
        "Mathematics / Probability and Statistics",
        "Data Structures",
        "Database Management Systems",
        "Object Oriented Programming",
        "Digital Electronics / Computer Organization",
        "Department Elective"
    ],

    "S4": [
        "Design and Analysis of Algorithms",
        "Operating Systems",
        "Computer Networks",
        "Software Engineering",
        "Mathematics / Statistics",
        "Department Elective"
    ],

    "S5": [
        "Machine Learning",
        "Artificial Intelligence",
        "Department Core I",
        "Department Core II",
        "Department Elective I",
        "Open Elective"
    ],

    "S6": [
        "Data Analytics",
        "Artificial Intelligence / Data Science",
        "Department Core III",
        "Department Core IV",
        "Department Elective II",
        "Mini Project / Seminar"
    ],

    "S7": [
        "Department Core V",
        "Department Core VI",
        "Department Elective III",
        "Open Elective",
        "Seminar",
        "Project Phase I"
    ],

    "S8": [
        "Department Elective IV",
        "Open Elective",
        "Comprehensive Viva",
        "Project Phase II"
    ]
}

# Every department gets its own copy.
# This prevents accidental sharing/modification.
CURRICULUM = {
    department: {
        semester: list(subjects)
        for semester, subjects in COMMON_SUBJECTS.items()
    }
    for department in DEPARTMENTS
}

# Example department-specific S3 structure.
# Edit this section according to your exact college curriculum.
CURRICULUM[
    "Artificial Intelligence and Data Science"
]["S3"] = [
    "Mathematics / Probability and Statistics",
    "Data Structures",
    "Database Management Systems",
    "Object Oriented Programming",
    "Artificial Intelligence / Machine Learning Fundamentals",
    "Department Elective"
]

# ============================================================
# TUTOR ACCOUNTS
# ============================================================

TUTOR_ACCOUNTS = {}

for i, department in enumerate(DEPARTMENTS, start=1):
    TUTOR_ACCOUNTS[department] = (
        f"tutor{i}",
        f"pass{i:03d}"
    )

# ============================================================
# DATA COLUMNS
# ============================================================

COLUMNS = [
    "Name",
    "University_ID",
    "Department",
    "Semester",
    "Subject",
    "Attendance",
    "Study_Hours",
    "Internal",
    "Assignment",
    "Previous_Mark",
    "Performance",
    "Percentage",
    "Cluster"
]

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:

        df = pd.read_csv(DATA_FILE)

        for column in COLUMNS:

            if column not in df.columns:
                df[column] = ""

        return df[COLUMNS]

    except Exception:

        return pd.DataFrame(columns=COLUMNS)


# ============================================================
# SAVE DATA
# ============================================================

def save_data(df):

    DATA_DIR.mkdir(exist_ok=True)

    df.to_csv(
        DATA_FILE,
        index=False
    )


# ============================================================
# ATTENDANCE MARK
# ============================================================

def attendance_mark(attendance):

    attendance = float(attendance)

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
# PERFORMANCE CALCULATION
# ============================================================

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

    study_percentage = min(
        (float(study_hours) / 6) * 100,
        100
    )

    internal_percentage = (
        float(internal) / 40
    ) * 100

    assignment_percentage = (
        float(assignment) / 15
    ) * 100

    previous_percentage = (
        float(previous) / 60
    ) * 100

    overall = np.mean([
        attendance_percentage,
        study_percentage,
        internal_percentage,
        assignment_percentage,
        previous_percentage
    ])

    overall = round(float(overall), 2)

    if overall < 50:

        level = "Low Performance"

    elif overall < 65:

        level = "Average Performance"

    elif overall < 80:

        level = "Above Average Performance"

    else:

        level = "Good Performance"

    return overall, level


# ============================================================
# PERFORMANCE ICON
# ============================================================

def performance_icon(level):

    if level == "Low Performance":
        return "🔴"

    if level == "Average Performance":
        return "🟠"

    if level == "Above Average Performance":
        return "🟡"

    return "🟢"


# ============================================================
# PERFORMANCE ADVICE
# ============================================================

def performance_advice(row):

    level = str(row["Performance"])

    if level == "Low Performance":

        return (
            "Focus on the weak areas. Improve attendance, "
            "complete assignments regularly and increase "
            "daily study time."
        )

    if level == "Average Performance":

        return (
            "Maintain regular revision, improve attendance "
            "and assignment completion, and study consistently."
        )

    if level == "Above Average Performance":

        return (
            "Good progress. Small improvements in attendance, "
            "internal marks and assignments can improve the result."
        )

    return (
        "Excellent performance! Keep up the good work."
    )


# ============================================================
# K-MEANS
# ============================================================

def perform_kmeans(df):

    numeric_columns = [
        "Attendance",
        "Study_Hours",
        "Internal",
        "Assignment",
        "Previous_Mark"
    ]

    work = df.copy()

    for column in numeric_columns:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce"
        ).fillna(0)

    if len(work) < 3:

        return None, None

    scaler = StandardScaler()

    X = scaler.fit_transform(
        work[numeric_columns]
    )

    number_of_clusters = min(
        3,
        len(work)
    )

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X)

    return model, clusters


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(df):

    numeric_columns = [
        "Attendance",
        "Study_Hours",
        "Internal",
        "Assignment",
        "Previous_Mark"
    ]

    work = df.copy()

    for column in numeric_columns:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce"
        ).fillna(0)

    if work["Performance"].nunique() < 2:

        return None, None

    X = work[numeric_columns]

    y = work["Performance"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    return model, accuracy


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf(
    student_df,
    student_name,
    university_id,
    department,
    semester
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            f"<b>Student Name:</b> {student_name}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Department:</b> {department}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Semester:</b> {semester}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    table_data = [[
        "Subject",
        "Attendance",
        "Internal",
        "Assignment",
        "Previous",
        "Study",
        "Performance"
    ]]

    for _, row in student_df.iterrows():

        table_data.append([
            str(row["Subject"]),
            f'{float(row["Attendance"]):.0f}%',
            f'{float(row["Internal"]):.0f}/40',
            f'{float(row["Assignment"]):.0f}/15',
            f'{float(row["Previous_Mark"]):.0f}/60',
            f'{float(row["Study_Hours"]):.1f}',
            f'{performance_icon(row["Performance"])} '
            f'{row["Performance"]}'
        ])

    table = Table(
        table_data,
        repeatRows=1
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1f2937")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
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
                7
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            )
        ])
    )

    story.append(table)

    story.append(Spacer(1, 15))

    percentages = pd.to_numeric(
        student_df["Percentage"],
        errors="coerce"
    )

    overall = percentages.mean()

    story.append(
        Paragraph(
            f"<b>Overall Percentage:</b> {overall:.2f}%",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 10))

    for _, row in student_df.iterrows():

        story.append(
            Paragraph(
                f"<b>{row['Subject']}</b> - "
                f"{performance_icon(row['Performance'])} "
                f"{row['Performance']}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                performance_advice(row),
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 5))

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Attendance conversion: "
            "90-100%=5, 80-89%=4, 70-79%=3, "
            "60-69%=2, 10-59%=1, below 10%=0.",
            styles["Normal"]
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "home"

if "role" not in st.session_state:

    st.session_state.role = None

if "department" not in st.session_state:

    st.session_state.department = None

if "student_name" not in st.session_state:

    st.session_state.student_name = None

if "university_id" not in st.session_state:

    st.session_state.university_id = None


df = load_data()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 40px;
    }

    .login-box {
        max-width: 800px;
        margin: auto;
        padding: 30px;
        border-radius: 25px;
        border: 1px solid #dddddd;
        box-shadow: 0 10px 35px rgba(0,0,0,0.10);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">'
        '🎓 STUDENT PERFORMANCE PREDICTION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-box">',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎓 Student")

        if st.button(
            "Student Login",
            width="stretch"
        ):

            st.session_state.page = "student_login"

            st.rerun()

    with col2:

        st.subheader("👨‍🏫 Tutor")

        if st.button(
            "Tutor Login",
            width="stretch"
        ):

            st.session_state.page = "tutor_login"

            st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# STUDENT LOGIN
# ============================================================

if st.session_state.page == "student_login":

    st.title("🎓 Student Login")

    name = st.text_input(
        "Student Name"
    )

    university_id = st.text_input(
        "University ID"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    st.caption(
        "Demo password format: BTECH2000 to BTECH2022"
    )

    if st.button(
        "Login",
        type="primary",
        width="stretch"
    ):

        valid_password = re.fullmatch(
            r"BTECH20\d{2}",
            password
        )

        if not valid_password:

            st.error(
                "Password must be like BTECH2007."
            )

        elif not (
            2000 <= int(password[-4:]) <= 2022
        ):

            st.error(
                "Password year must be between 2000 and 2022."
            )

        else:

            student_rows = df[
                df["University_ID"]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                university_id
                .strip()
                .lower()
            ]

            if student_rows.empty:

                st.error(
                    "University ID not found."
                )

            elif (
                str(student_rows.iloc[0]["Name"])
                .strip()
                .lower()
                !=
                name.strip().lower()
            ):

                st.error(
                    "Student name does not match."
                )

            else:

                st.session_state.role = "student"

                st.session_state.student_name = (
                    str(student_rows.iloc[0]["Name"])
                )

                st.session_state.university_id = (
                    university_id
                )

                st.session_state.department = (
                    str(student_rows.iloc[0]["Department"])
                )

                st.session_state.page = (
                    "student_dashboard"
                )

                st.rerun()

    if st.button("← Back"):

        st.session_state.page = "home"

        st.rerun()

    st.stop()


# ============================================================
# TUTOR LOGIN
# ============================================================

if st.session_state.page == "tutor_login":

    st.title("👨‍🏫 Tutor Login")

    department = st.selectbox(
        "Department",
        DEPARTMENTS
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
        type="primary",
        width="stretch"
    ):

        correct_username, correct_password = (
            TUTOR_ACCOUNTS[department]
        )

        if (
            username == correct_username
            and
            password == correct_password
        ):

            st.session_state.role = "tutor"

            st.session_state.department = department

            st.session_state.page = (
                "tutor_dashboard"
            )

            st.rerun()

        else:

            st.error(
                "Incorrect username or password."
            )

    with st.expander(
        "Demo tutor credentials"
    ):

        st.write(
            "Select a department and use its "
            "generated tutor account."
        )

        st.code(
            "\n".join(
                f"{department}: "
                f"{TUTOR_ACCOUNTS[department][0]} / "
                f"{TUTOR_ACCOUNTS[department][1]}"
                for department in DEPARTMENTS
            )
        )

    if st.button("← Back"):

        st.session_state.page = "home"

        st.rerun()

    st.stop()


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.clear()

    st.session_state.page = "home"


# ============================================================
# TUTOR DASHBOARD
# ============================================================

if st.session_state.page == "tutor_dashboard":

    department = st.session_state.department

    st.title("👨‍🏫 Tutor Dashboard")

    st.caption(
        f"Department: {department}"
    )

    if st.button("Logout"):

        logout()

        st.rerun()

    tabs = st.tabs([
        "➕ Add Student",
        "📋 Student Data",
        "📊 K-Means",
        "🤖 Random Forest",
        "📥 Download Data"
    ])

    # --------------------------------------------------------
    # ADD STUDENT
    # --------------------------------------------------------

    with tabs[0]:

        st.subheader(
            "Add Student Performance"
        )

        student_name = st.text_input(
            "Student Name",
            key="student_name_input"
        )

        university_id = st.text_input(
            "University ID",
            key="university_id_input"
        )

        semester = st.selectbox(
            "Semester",
            SEMESTERS,
            key="tutor_semester"
        )

        subjects = CURRICULUM[
            department
        ][semester]

        st.info(
            f"{semester} - {len(subjects)} subjects"
        )

        records = []

        for index, subject in enumerate(
            subjects,
            start=1
        ):

            st.markdown(
                f"### Subject {index}: {subject}"
            )

            col1, col2, col3, col4, col5 = (
                st.columns(5)
            )

            with col1:

                attendance = st.number_input(
                    "Attendance %",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    key=f"attendance_{semester}_{index}"
                )

            with col2:

                study_hours = st.number_input(
                    "Study Hours",
                    min_value=0.0,
                    max_value=6.0,
                    value=2.0,
                    key=f"study_{semester}_{index}"
                )

            with col3:

                internal = st.number_input(
                    "Internal /40",
                    min_value=0.0,
                    max_value=40.0,
                    value=20.0,
                    key=f"internal_{semester}_{index}"
                )

            with col4:

                assignment = st.number_input(
                    "Assignment /15",
                    min_value=0.0,
                    max_value=15.0,
                    value=8.0,
                    key=f"assignment_{semester}_{index}"
                )

            with col5:

                previous = st.number_input(
                    "Previous /60",
                    min_value=0.0,
                    max_value=60.0,
                    value=30.0,
                    key=f"previous_{semester}_{index}"
                )

            percentage, performance = (
                calculate_performance(
                    attendance,
                    study_hours,
                    internal,
                    assignment,
                    previous
                )
            )

            st.write(
                f"{performance_icon(performance)} "
                f"{performance} — {percentage:.2f}%"
            )

            records.append({
                "Name": student_name.strip(),
                "University_ID": university_id.strip(),
                "Department": department,
                "Semester": semester,
                "Subject": subject,
                "Attendance": attendance,
                "Study_Hours": study_hours,
                "Internal": internal,
                "Assignment": assignment,
                "Previous_Mark": previous,
                "Performance": performance,
                "Percentage": percentage,
                "Cluster": -1
            })

        if st.button(
            "💾 Save Student",
            type="primary",
            width="stretch"
        ):

            if (
                not student_name.strip()
                or
                not university_id.strip()
            ):

                st.error(
                    "Enter Student Name and University ID."
                )

            else:

                new_records = pd.DataFrame(
                    records
                )

                # Remove old data for the same
                # student + department + semester.

                keep = ~(
                    (
                        df["University_ID"]
                        .astype(str)
                        .str.lower()
                        ==
                        university_id
                        .strip()
                        .lower()
                    )
                    &
                    (
                        df["Department"]
                        ==
                        department
                    )
                    &
                    (
                        df["Semester"]
                        ==
                        semester
                    )
                )

                df = pd.concat(
                    [
                        df[keep],
                        new_records
                    ],
                    ignore_index=True
                )

                save_data(df)

                st.success(
                    "Student data saved successfully."
                )

                st.rerun()

    # --------------------------------------------------------
    # STUDENT DATA
    # --------------------------------------------------------

    with tabs[1]:

        department_df = df[
            df["Department"]
            ==
            department
        ].copy()

        if department_df.empty:

            st.info(
                "No student data available."
            )

        else:

            st.subheader(
                "Student Records"
            )

            student_list = (
                department_df[
                    [
                        "Name",
                        "University_ID",
                        "Semester"
                    ]
                ]
                .drop_duplicates()
                .sort_values("Name")
            )

            st.dataframe(
                student_list,
                width="stretch",
                hide_index=True
            )

            selected_id = st.selectbox(
                "Select University ID",
                sorted(
                    department_df[
                        "University_ID"
                    ]
                    .astype(str)
                    .unique()
                )
            )

            selected_student = department_df[
                department_df[
                    "University_ID"
                ].astype(str)
                ==
                selected_id
            ]

            st.dataframe(
                selected_student,
                width="stretch",
                hide_index=True
            )

            selected_semester = st.selectbox(
                "Semester for Progress Report",
                sorted(
                    selected_student[
                        "Semester"
                    ].astype(str).unique()
                )
            )

            report_df = selected_student[
                selected_student["Semester"]
                ==
                selected_semester
            ]

            if not report_df.empty:

                pdf = create_pdf(
                    report_df,
                    str(
                        report_df.iloc[0]["Name"]
                    ),
                    selected_id,
                    department,
                    selected_semester
                )

                st.download_button(
                    "📄 Download Progress Report",
                    data=pdf,
                    file_name=(
                        f"{selected_id}_"
                        f"{selected_semester}_"
                        "progress_report.pdf"
                    ),
                    mime="application/pdf",
                    width="stretch"
                )

            st.divider()

            if st.button(
                "🗑️ Delete Complete Student History"
            ):

                df = df[
                    ~(
                        (
                            df["University_ID"]
                            .astype(str)
                            ==
                            selected_id
                        )
                        &
                        (
                            df["Department"]
                            ==
                            department
                        )
                    )
                ]

                save_data(df)

                st.success(
                    "Student history deleted."
                )

                st.rerun()

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    with tabs[2]:

        st.subheader(
            "📊 K-Means Clustering"
        )

        department_df = df[
            df["Department"]
            ==
            department
        ].copy()

        if len(department_df) < 3:

            st.info(
                "Add at least 3 records to perform K-Means."
            )

        else:

            model, clusters = perform_kmeans(
                department_df
            )

            if model is not None:

                department_df[
                    "Cluster"
                ] = clusters

                # Match rows safely using index.

                for index, cluster in zip(
                    department_df.index,
                    clusters
                ):

                    df.loc[
                        index,
                        "Cluster"
                    ] = int(cluster)

                save_data(df)

                summary = (
                    department_df
                    .groupby("Cluster")
                    .size()
                    .reset_index(
                        name="Number of Records"
                    )
                )

                st.dataframe(
                    summary,
                    width="stretch",
                    hide_index=True
                )

                st.caption(
                    "Cluster numbers are machine-generated "
                    "group identifiers. They are not official "
                    "KTU grades."
                )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    with tabs[3]:

        st.subheader(
            "🤖 Random Forest"
        )

        department_df = df[
            df["Department"]
            ==
            department
        ].copy()

        if len(department_df) < 5:

            st.info(
                "Add more student records before running Random Forest."
            )

        else:

            model, accuracy = train_random_forest(
                department_df
            )

            if model is None:

                st.warning(
                    "At least two performance classes "
                    "are required."
                )

            else:

                st.metric(
                    "Training Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

                st.caption(
                    "This is a project demonstration metric. "
                    "It should not be presented as independent "
                    "test accuracy."
                )

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

    with tabs[4]:

        department_df = df[
            df["Department"]
            ==
            department
        ].copy()

        if department_df.empty:

            st.info(
                "No data available."
            )

        else:

            csv_data = department_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 Download Department CSV",
                data=csv_data,
                file_name="department_student_data.csv",
                mime="text/csv",
                width="stretch"
            )

    st.stop()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

if st.session_state.page == "student_dashboard":

    student_name = (
        st.session_state.student_name
    )

    university_id = (
        st.session_state.university_id
    )

    department = (
        st.session_state.department
    )

    # Only student's name is shown.
    st.title(
        f"🎓 {student_name}"
    )

    if st.button("Logout"):

        logout()

        st.rerun()

    student_data = df[
        (
            df["University_ID"]
            .astype(str)
            .str.lower()
            ==
            university_id
            .strip()
            .lower()
        )
        &
        (
            df["Department"]
            ==
            department
        )
    ].copy()

    if student_data.empty:

        st.warning(
            "No performance data found."
        )

        st.stop()

    semester = st.selectbox(
        "Select Semester",
        SEMESTERS
    )

    semester_data = student_data[
        student_data["Semester"]
        ==
        semester
    ].copy()

    st.subheader(
        f"{semester} Performance"
    )

    if semester_data.empty:

        st.info(
            "No data has been entered by your tutor "
            "for this semester."
        )

    else:

        st.dataframe(
            semester_data[
                [
                    "Subject",
                    "Attendance",
                    "Internal",
                    "Assignment",
                    "Previous_Mark",
                    "Study_Hours",
                    "Performance",
                    "Percentage"
                ]
            ],
            width="stretch",
            hide_index=True
        )

        st.subheader(
            "📚 Subject-wise Performance"
        )

        for _, row in semester_data.iterrows():

            level = str(
                row["Performance"]
            )

            icon = performance_icon(
                level
            )

            if level == "Good Performance":

                st.success(
                    f"{icon} {row['Subject']} — {level}"
                )

            elif level == "Low Performance":

                st.error(
                    f"{icon} {row['Subject']} — {level}"
                )

            else:

                st.warning(
                    f"{icon} {row['Subject']} — {level}"
                )

            st.write(
                f"Attendance: "
                f"{float(row['Attendance']):.0f}%"
            )

            st.write(
                f"Internal: "
                f"{float(row['Internal']):.0f}/40"
            )

            st.write(
                f"Assignment: "
                f"{float(row['Assignment']):.0f}/15"
            )

            st.write(
                f"Previous Mark: "
                f"{float(row['Previous_Mark']):.0f}/60"
            )

            st.write(
                f"Study Hours: "
                f"{float(row['Study_Hours']):.1f} hours/day"
            )

            st.write(
                f"Performance: "
                f"**{float(row['Percentage']):.2f}%**"
            )

            st.caption(
                performance_advice(row)
            )

            st.divider()

        pdf = create_pdf(
            semester_data,
            student_name,
            university_id,
            department,
            semester
        )

        st.download_button(
            "📄 Download My Progress Report",
            data=pdf,
            file_name=(
                f"{university_id}_"
                f"{semester}_progress_report.pdf"
            ),
            mime="application/pdf",
            width="stretch"
        )

    st.stop()
