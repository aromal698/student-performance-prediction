import os
import re
import io
import hashlib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

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

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CONSTANTS
# ============================================================

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "student_performance.csv")

BRANCHES = [
    "Civil Engineering",
    "Computer Science and Engineering",
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
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
    "Computer Science and Engineering (Artificial Intelligence)",
    "Computer Science and Engineering (Data Science)",
    "Computer Science and Engineering (Cyber Security)",
    "Robotics and Automation",
    "Mechatronics Engineering",
]

# ============================================================
# TUTOR USERNAMES / PASSWORDS
# ============================================================

TUTOR_ACCOUNTS = {
    "Civil Engineering": ("tutor_civil", "civil123"),
    "Computer Science and Engineering": ("tutor_cse", "cse123"),
    "Artificial Intelligence and Data Science": ("tutor_aids", "aids123"),
    "Artificial Intelligence and Machine Learning": ("tutor_aiml", "aiml123"),
    "Information Technology": ("tutor_it", "it123"),
    "Electronics and Communication Engineering": ("tutor_ece", "ece123"),
    "Electrical and Electronics Engineering": ("tutor_eee", "eee123"),
    "Electrical Engineering": ("tutor_ee", "ee123"),
    "Mechanical Engineering": ("tutor_me", "me123"),
    "Chemical Engineering": ("tutor_chemical", "chemical123"),
    "Biotechnology and Biochemical Engineering": ("tutor_bio", "bio123"),
    "Food Technology": ("tutor_food", "food123"),
    "Production Engineering": ("tutor_production", "production123"),
    "Automobile Engineering": ("tutor_auto", "auto123"),
    "Industrial Engineering": ("tutor_industrial", "industrial123"),
    "Biomedical Engineering": ("tutor_biomedical", "biomedical123"),
    "Aeronautical Engineering": ("tutor_aero", "aero123"),
    "Applied Electronics and Instrumentation Engineering": (
        "tutor_aei",
        "aei123",
    ),
    "Electronics and Biomedical Engineering": ("tutor_ebm", "ebm123"),
    "Electronics and Computer Engineering": ("tutor_ececomp", "ececomp123"),
    "Computer Science and Engineering (Artificial Intelligence)": (
        "tutor_cseai",
        "cseai123",
    ),
    "Computer Science and Engineering (Data Science)": (
        "tutor_csedata",
        "csedata123",
    ),
    "Computer Science and Engineering (Cyber Security)": (
        "tutor_cyber",
        "cyber123",
    ),
    "Robotics and Automation": ("tutor_robotics", "robotics123"),
    "Mechatronics Engineering": ("tutor_mechatronics", "mechatronics123"),
}

# ============================================================
# SEMESTER SUBJECTS
# ============================================================

SEMESTERS = {
    "S1": [
        "Mathematics for Information Science-I",
        "Physics for Information Science",
        "Engineering Graphics",
        "Introduction to Electrical & Electronics Engineering",
        "Introduction to Computing",
        "Health and Wellness",
    ],
    "S2": [
        "Mathematics for Information Science-II",
        "Chemistry for Information Science",
        "Programming in C",
        "Engineering Mechanics",
        "Design Thinking and Product Development",
        "Professional Communication",
    ],
    "S3": [
        "Mathematics for Information Science-III",
        "Foundations of Artificial Intelligence",
        "Data Structures and Algorithms",
        "Introduction to Data Science",
        "Digital Electronics & Logic Design",
        "Engineering Economics / Ethics",
    ],
    "S4": [
        "Mathematics for Information Science-IV",
        "Database Management Systems",
        "Operating Systems",
        "Computer Organization and Architecture",
        "Object Oriented Programming",
        "Professional Ethics",
    ],
    "S5": [
        "Machine Learning",
        "Computer Networks",
        "Web Technologies",
        "Software Engineering",
        "Artificial Intelligence",
        "Program Elective-I",
    ],
    "S6": [
        "Deep Learning",
        "Big Data Analytics",
        "Data Mining",
        "Cloud Computing",
        "Natural Language Processing",
        "Program Elective-II",
    ],
    "S7": [
        "Major Project Phase-I",
        "Program Elective-III",
        "Program Elective-IV",
        "Open Elective",
        "Seminar",
        "Mini Project",
    ],
    "S8": [
        "Major Project Phase-II",
        "Program Elective-V",
        "Program Elective-VI",
        "Comprehensive Viva",
        "Internship / Industrial Training",
        "Project Viva",
    ],
}

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ---------------- GLOBAL ---------------- */

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(0, 200, 255, 0.18), transparent 28%),
        radial-gradient(circle at 85% 25%, rgba(160, 70, 255, 0.20), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(0, 255, 170, 0.12), transparent 35%),
        linear-gradient(135deg, #050816 0%, #0b1024 50%, #050713 100%);
    color: white;
}

/* Hide Streamlit default menu */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* ---------------- CENTERED CONTENT ---------------- */

.block-container {
    max-width: 1250px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* ---------------- FRONT PAGE ---------------- */

.front-wrapper {
    min-height: 88vh;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.front-card {
    width: min(900px, 94%);
    padding: 55px 50px;
    border-radius: 32px;

    background: rgba(12, 18, 45, 0.72);
    border: 1px solid rgba(255,255,255,0.14);

    box-shadow:
        0 30px 80px rgba(0,0,0,0.45),
        0 0 60px rgba(0,190,255,0.08);

    backdrop-filter: blur(18px);

    text-align: center;
}

/* MAIN HEADING */

.main-heading {
    text-align: center;
    font-size: clamp(30px, 5vw, 58px);
    font-weight: 900;
    letter-spacing: 2px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #58d9ff,
        #b77cff,
        #ffffff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 45px;
}

/* LOGIN ROLE CARDS */

.role-card {
    padding: 28px 20px;
    border-radius: 22px;

    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);

    transition: all 0.3s ease;
}

.role-card:hover {
    transform: translateY(-8px);
    background: rgba(255,255,255,0.10);
    box-shadow: 0 15px 40px rgba(0,200,255,0.12);
}

.role-icon {
    font-size: 55px;
    margin-bottom: 10px;
}

.role-title {
    font-size: 24px;
    font-weight: 800;
}

/* ---------------- DASHBOARD HEADER ---------------- */

.dashboard-header {
    width: 100%;
    max-width: 1150px;
    margin: 0 auto 25px auto;

    text-align: center;

    padding: 30px 25px;

    border-radius: 25px;

    background: rgba(13, 20, 48, 0.75);
    border: 1px solid rgba(255,255,255,0.12);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35),
        0 0 40px rgba(70,150,255,0.07);

    backdrop-filter: blur(15px);
}

.dashboard-title {
    font-size: clamp(28px, 4vw, 45px);
    font-weight: 900;

    background: linear-gradient(
        90deg,
        #ffffff,
        #5bd8ff,
        #b878ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.dashboard-subtitle {
    color: #b8c6df;
    font-size: 16px;
    margin-top: 8px;
}

/* ---------------- SECTION CARDS ---------------- */

.section-card {
    max-width: 1150px;
    margin: 0 auto 25px auto;

    padding: 28px;

    border-radius: 24px;

    background: rgba(12,18,42,0.70);

    border: 1px solid rgba(255,255,255,0.10);

    box-shadow: 0 18px 45px rgba(0,0,0,0.25);

    backdrop-filter: blur(14px);
}

/* ---------------- METRIC CARDS ---------------- */

.metric-card {
    text-align: center;

    padding: 22px;

    border-radius: 20px;

    background: rgba(255,255,255,0.06);

    border: 1px solid rgba(255,255,255,0.10);
}

.metric-number {
    font-size: 32px;
    font-weight: 900;
}

.metric-label {
    color: #aab9d0;
    font-size: 14px;
}

/* ---------------- PERFORMANCE CIRCLE ---------------- */

.performance-circle {
    width: 105px;
    height: 105px;

    border-radius: 50%;

    margin: 15px auto;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 35px;

    border: 8px solid currentColor;
}

/* ---------------- SUBJECT CARD ---------------- */

.subject-card {
    padding: 22px;

    border-radius: 20px;

    background: rgba(255,255,255,0.045);

    border: 1px solid rgba(255,255,255,0.09);

    margin-bottom: 18px;
}

.subject-title {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 15px;
}

/* ---------------- LOGIN BOX ---------------- */

.login-box {
    max-width: 620px;
    margin: 25px auto;

    padding: 35px;

    border-radius: 25px;

    background: rgba(10,16,40,0.82);

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow: 0 20px 60px rgba(0,0,0,0.4);

    backdrop-filter: blur(18px);
}

/* ---------------- BUTTONS ---------------- */

.stButton > button {
    width: 100%;

    border-radius: 14px;

    min-height: 45px;

    font-weight: 700;

    border: 1px solid rgba(255,255,255,0.15);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,190,255,0.18);
}

/* ---------------- TABS ---------------- */

.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    border-radius: 12px;
}

/* ---------------- DATAFRAME ---------------- */

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

/* ---------------- MOBILE ---------------- */

@media (max-width: 700px) {

    .front-card {
        padding: 35px 20px;
    }

    .main-heading {
        font-size: 30px;
    }

    .dashboard-header {
        padding: 22px 15px;
    }

    .section-card {
        padding: 18px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "home",
    "role": None,
    "student_logged": False,
    "tutor_logged": False,
    "student_username": "",
    "student_id": "",
    "tutor_branch": "",
    "tutor_username": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATA FUNCTIONS
# ============================================================

def ensure_data_file():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        columns = [
            "Name",
            "University_ID",
            "Branch",
            "Semester",
            "Subject",
            "Attendance",
            "Study_Hours",
            "Internal",
            "Assignment",
            "Previous_Mark",
        ]

        pd.DataFrame(columns=columns).to_csv(DATA_FILE, index=False)


def load_data():
    ensure_data_file()

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        df = pd.DataFrame()

    return df


def save_data(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)


# ============================================================
# PERFORMANCE FUNCTIONS
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


def calculate_performance(row):
    attendance = float(row["Attendance"])
    study = float(row["Study_Hours"])
    internal = float(row["Internal"])
    assignment = float(row["Assignment"])
    previous = float(row["Previous_Mark"])

    att_mark = attendance_mark(attendance)

    attendance_percent = (att_mark / 5) * 100
    study_percent = min((study / 6) * 100, 100)
    internal_percent = (internal / 40) * 100
    assignment_percent = (assignment / 15) * 100
    previous_percent = (previous / 60) * 100

    overall = np.mean([
        attendance_percent,
        study_percent,
        internal_percent,
        assignment_percent,
        previous_percent,
    ])

    if overall < 50:
        level = "Low Performance"
        emoji = "🔴"
        color = "#ff4d4d"
    elif overall < 65:
        level = "Average Performance"
        emoji = "🟠"
        color = "#ff9f43"
    elif overall < 80:
        level = "Above Average Performance"
        emoji = "🟡"
        color = "#ffd93d"
    else:
        level = "Good Performance"
        emoji = "🟢"
        color = "#45e36b"

    return {
        "Attendance Mark": att_mark,
        "Overall Percentage": round(overall, 2),
        "Performance": level,
        "Emoji": emoji,
        "Color": color,
    }


def add_performance_columns(df):
    if df.empty:
        return df

    result = df.copy()

    performance_values = result.apply(calculate_performance, axis=1)

    result["Attendance_Mark"] = performance_values.apply(
        lambda x: x["Attendance Mark"]
    )

    result["Overall_Percentage"] = performance_values.apply(
        lambda x: x["Overall Percentage"]
    )

    result["Performance"] = performance_values.apply(
        lambda x: x["Performance"]
    )

    return result


# ============================================================
# K-MEANS
# ============================================================

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal",
    "Assignment",
    "Previous_Mark",
]


def run_kmeans(df):
    if len(df) < 3:
        return df, None, None

    clean = df.copy()

    for col in FEATURES:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=FEATURES)

    if len(clean) < 3:
        return df, None, None

    X = clean[FEATURES].values

    n_clusters = min(3, len(clean))

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    clusters = model.fit_predict(X)

    clean["KMeans_Cluster"] = clusters

    score = None

    if len(set(clusters)) > 1:
        score = silhouette_score(X, clusters)

    return clean, model, score


# ============================================================
# RANDOM FOREST
# ============================================================

def create_rf_model(df):
    if len(df) < 10:
        return None, None, None

    clean = df.copy()

    for col in FEATURES:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=FEATURES)

    if len(clean) < 10:
        return None, None, None

    clean = add_performance_columns(clean)

    X = clean[FEATURES]

    y = clean["Performance"]

    if len(y.unique()) < 2:
        return None, None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    return model, clean, accuracy


# ============================================================
# PASSWORD VALIDATION
# ============================================================

def valid_student_password(password):

    pattern = r"BTECH(20\d{2})"

    match = re.fullmatch(pattern, password)

    if not match:
        return False

    year = int(password[-4:])

    return 2000 <= year <= 2022


# ============================================================
# PDF REPORT
# ============================================================

def create_student_pdf(student_df):

    if student_df.empty:
        return None

    first = student_df.iloc[0]

    name = str(first["Name"])
    university_id = str(first["University_ID"])
    branch = str(first["Branch"])

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    story = []

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Student Performance Prediction System",
            ParagraphStyle(
                "Center",
                parent=styles["Normal"],
                alignment=TA_CENTER,
                fontSize=11,
            ),
        )
    )

    story.append(Spacer(1, 18))

    student_details = [
        ["Student Name", name],
        ["University ID", university_id],
        ["Branch", branch],
    ]

    table = Table(
        student_details,
        colWidths=[150, 330],
    )

    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 18))

    student_df = add_performance_columns(student_df)

    overall = student_df["Overall_Percentage"].mean()

    if overall < 50:
        overall_level = "Low Performance"
    elif overall < 65:
        overall_level = "Average Performance"
    elif overall < 80:
        overall_level = "Above Average Performance"
    else:
        overall_level = "Good Performance"

    story.append(
        Paragraph(
            "Overall Performance",
            heading_style,
        )
    )

    overall_table = Table(
        [
            ["Overall Percentage", f"{overall:.2f}%"],
            ["Performance Level", overall_level],
        ],
        colWidths=[200, 280],
    )

    overall_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(overall_table)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Subject-wise Performance",
            heading_style,
        )
    )

    data = [
        [
            "Subject",
            "Attendance",
            "Att. Mark",
            "Study Hrs",
            "Internal",
            "Assignment",
            "Previous",
            "Performance",
        ]
    ]

    for _, row in student_df.iterrows():

        performance = calculate_performance(row)

        data.append(
            [
                str(row["Subject"])[:25],
                f'{float(row["Attendance"]):.1f}%',
                str(performance["Attendance Mark"]),
                f'{float(row["Study_Hours"]):.1f}',
                f'{float(row["Internal"]):.1f}/40',
                f'{float(row["Assignment"]):.1f}/15',
                f'{float(row["Previous_Mark"]):.1f}/60',
                performance["Performance"].replace(
                    " Performance", ""
                ),
            ]
        )

    subject_table = Table(
        data,
        repeatRows=1,
        colWidths=[
            105,
            58,
            48,
            50,
            55,
            58,
            55,
            75,
        ],
    )

    subject_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(subject_table)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Assessment Conversion",
            heading_style,
        )
    )

    conversion_text = """
    Attendance: 90–100% = 5 marks, 80–89% = 4 marks,
    70–79% = 3 marks, 60–69% = 2 marks,
    10–59% = 1 mark, below 10% = 0 marks.
    <br/><br/>
    Internal maximum = 40 marks.
    Assignment maximum = 15 marks.
    Previous mark maximum = 60 marks.
    Study hours are normalized using 6 hours/day as the project maximum.
    """

    story.append(
        Paragraph(
            conversion_text,
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Improvement / Feedback",
            heading_style,
        )
    )

    if overall < 50:
        advice = (
            "Focus on regular study, attendance, internal assessments, "
            "assignments and revision of previous topics."
        )
    elif overall < 65:
        advice = (
            "Maintain regular study habits and improve attendance, "
            "assignment completion and internal assessment performance."
        )
    elif overall < 80:
        advice = (
            "Good progress. Small improvements in attendance, internal "
            "marks, assignments and study consistency can further improve performance."
        )
    else:
        advice = (
            "Excellent performance! Keep up the good work and maintain consistency."
        )

    story.append(
        Paragraph(
            advice,
            styles["BodyText"],
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.markdown(
        """
        <div class="front-wrapper">
            <div class="front-card">
                <div class="main-heading">
                    🎓 STUDENT PERFORMANCE PREDICTION
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 2])

    with col2:

        st.markdown(
            """
            <div class="role-card">
                <div class="role-icon">🎓</div>
                <div class="role-title">Student</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "🎓 Student Login",
            key="student_role",
            width="stretch",
        ):
            st.session_state.role = "student"
            st.session_state.page = "student_login"
            st.rerun()

    with col3:

        st.markdown(
            """
            <div class="role-card">
                <div class="role-icon">👨‍🏫</div>
                <div class="role-title">Tutor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "👨‍🏫 Tutor Login",
            key="tutor_role",
            width="stretch",
        ):
            st.session_state.role = "tutor"
            st.session_state.page = "tutor_login"
            st.rerun()


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.markdown(
        """
        <div class="dashboard-header">
            <div class="dashboard-title">
                🎓 Student Login
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    username = st.text_input(
        "Username",
        placeholder="Enter username",
    )

    university_id = st.text_input(
        "University ID",
        placeholder="Enter University ID",
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Example: BTECH2007",
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔐 Login",
            width="stretch",
        ):

            if not username or not university_id or not password:
                st.error("Please fill all fields.")

            elif not valid_student_password(password):
                st.error(
                    "Password must be in format BTECH2000 to BTECH2022."
                )

            else:

                df = load_data()

                if df.empty:
                    st.error(
                        "No student records found. Please contact your tutor."
                    )

                else:

                    matched = df[
                        (
                            df["University_ID"]
                            .astype(str)
                            .str.strip()
                            == university_id.strip()
                        )
                    ]

                    if matched.empty:
                        st.error("University ID not found.")

                    else:

                        student_name = str(
                            matched.iloc[0]["Name"]
                        )

                        st.session_state.student_logged = True
                        st.session_state.student_username = username
                        st.session_state.student_id = university_id
                        st.session_state.page = "student_dashboard"

                        st.rerun()

    with col2:

        if st.button(
            "⬅️ Back",
            width="stretch",
        ):
            st.session_state.page = "home"
            st.session_state.role = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "Demo password format: BTECH2000 to BTECH2022"
    )


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    st.markdown(
        """
        <div class="dashboard-header">
            <div class="dashboard-title">
                👨‍🏫 Tutor Login
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    branch = st.selectbox(
        "Department / Branch",
        BRANCHES,
    )

    username = st.text_input(
        "Tutor Username",
        placeholder="Enter tutor username",
    )

    password = st.text_input(
        "Tutor Password",
        type="password",
        placeholder="Enter tutor password",
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔐 Tutor Login",
            width="stretch",
        ):

            correct_username, correct_password = TUTOR_ACCOUNTS[
                branch
            ]

            if (
                username == correct_username
                and password == correct_password
            ):

                st.session_state.tutor_logged = True
                st.session_state.tutor_branch = branch
                st.session_state.tutor_username = username
                st.session_state.page = "tutor_dashboard"

                st.rerun()

            else:
                st.error(
                    "Invalid tutor username or password."
                )

    with col2:

        if st.button(
            "⬅️ Back",
            width="stretch",
        ):
            st.session_state.page = "home"
            st.session_state.role = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    df = load_data()

    student_id = st.session_state.student_id

    student_df = df[
        df["University_ID"].astype(str).str.strip()
        == str(student_id).strip()
    ].copy()

    if student_df.empty:

        st.error("Student data not found.")

        if st.button("Logout"):
            logout()

        return

    student_name = str(student_df.iloc[0]["Name"])
    branch = str(student_df.iloc[0]["Branch"])

    # ---------------- HEADER ----------------

    st.markdown(
        f"""
        <div class="dashboard-header">
            <div class="dashboard-title">
                🎓 {student_name}
            </div>
            <div class="dashboard-subtitle">
                Student Performance Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- SEMESTER ----------------

    selected_semester = st.selectbox(
        "Select Semester",
        list(SEMESTERS.keys()),
    )

    subjects = SEMESTERS[selected_semester]

    semester_df = student_df[
        student_df["Semester"].astype(str)
        == selected_semester
    ].copy()

    if semester_df.empty:

        st.warning(
            f"No tutor-entered records are available for {selected_semester}."
        )

        st.info(
            "The tutor must add the student's subject records for this semester."
        )

    else:

        semester_df = add_performance_columns(
            semester_df
        )

        # ---------------- OVERALL ----------------

        overall = semester_df[
            "Overall_Percentage"
        ].mean()

        if overall < 50:
            level = "Low Performance"
            emoji = "🔴"
        elif overall < 65:
            level = "Average Performance"
            emoji = "🟠"
        elif overall < 80:
            level = "Above Average Performance"
            emoji = "🟡"
        else:
            level = "Good Performance"
            emoji = "🟢"

        st.markdown(
            f"""
            <div class="section-card">
                <h2 style="text-align:center;">
                    Overall Performance
                </h2>

                <div class="performance-circle">
                    {emoji}
                </div>

                <h2 style="text-align:center;">
                    {overall:.2f}%
                </h2>

                <p style="text-align:center;font-size:18px;">
                    {level}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------- SUBJECTS ----------------

        st.markdown(
            """
            <div class="section-card">
                <h2 style="text-align:center;">
                    📚 Subject-wise Performance
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for _, row in semester_df.iterrows():

            result = calculate_performance(row)

            st.markdown(
                f"""
                <div class="subject-card">

                    <div class="subject-title">
                        📘 {row["Subject"]}
                    </div>

                    <div style="text-align:center;">
                        <div class="performance-circle"
                             style="color:{result["Color"]};">
                            {result["Emoji"]}
                        </div>

                        <h3>
                            {result["Performance"]}
                        </h3>

                        <p>
                            Overall: {result["Overall Percentage"]:.2f}%
                        </p>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric(
                    "Attendance",
                    f'{float(row["Attendance"]):.1f}%',
                )

            with c2:
                st.metric(
                    "Internal",
                    f'{float(row["Internal"]):.1f}/40',
                )

            with c3:
                st.metric(
                    "Assignment",
                    f'{float(row["Assignment"]):.1f}/15',
                )

            with c4:
                st.metric(
                    "Previous",
                    f'{float(row["Previous_Mark"]):.1f}/60',
                )

            with c5:
                st.metric(
                    "Study Hours",
                    f'{float(row["Study_Hours"]):.1f}',
                )

        # ---------------- DOWNLOAD PDF ----------------

        pdf = create_student_pdf(
            semester_df
        )

        if pdf:

            st.download_button(
                label="📄 Download My Progress Report",
                data=pdf,
                file_name=f"{student_name}_Progress_Report.pdf",
                mime="application/pdf",
                width="stretch",
            )

    st.divider()

    if st.button(
        "🚪 Logout",
        width="stretch",
    ):
        logout()


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    branch = st.session_state.tutor_branch

    st.markdown(
        f"""
        <div class="dashboard-header">

            <div class="dashboard-title">
                👨‍🏫 Tutor Dashboard
            </div>

            <div class="dashboard-subtitle">
                {branch}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    # Only branch students
    if not df.empty and "Branch" in df.columns:

        branch_df = df[
            df["Branch"].astype(str).str.strip()
            == branch.strip()
        ].copy()

    else:

        branch_df = pd.DataFrame()

    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(
        [
            "➕ Add Student",
            "📋 Student Data",
            "📊 K-Means",
            "🌲 Random Forest",
            "📄 Progress Reports",
        ]
    )

    # ========================================================
    # ADD STUDENT
    # ========================================================

    with tabs[0]:

        st.markdown(
            """
            <div class="section-card">
                <h2 style="text-align:center;">
                    Add Student Performance
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            student_name = st.text_input(
                "Student Name",
                key="add_name",
            )

        with col2:

            university_id = st.text_input(
                "University ID",
                key="add_uid",
            )

        semester = st.selectbox(
            "Semester",
            list(SEMESTERS.keys()),
            key="add_semester",
        )

        st.markdown(
            "### Enter Subject-wise Marks"
        )

        selected_subjects = SEMESTERS[
            semester
        ]

        records = []

        for subject in selected_subjects:

            st.markdown(
                f"#### 📘 {subject}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                attendance = st.number_input(
                    "Attendance %",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    key=f"att_{semester}_{subject}",
                )

            with c2:

                study_hours = st.number_input(
                    "Study Hours",
                    min_value=0.0,
                    max_value=6.0,
                    value=3.0,
                    step=0.5,
                    key=f"study_{semester}_{subject}",
                )

            with c3:

                internal = st.number_input(
                    "Internal /40",
                    min_value=0.0,
                    max_value=40.0,
                    value=25.0,
                    step=1.0,
                    key=f"internal_{semester}_{subject}",
                )

            with c4:

                assignment = st.number_input(
                    "Assignment /15",
                    min_value=0.0,
                    max_value=15.0,
                    value=10.0,
                    step=1.0,
                    key=f"assignment_{semester}_{subject}",
                )

            with c5:

                previous = st.number_input(
                    "Previous /60",
                    min_value=0.0,
                    max_value=60.0,
                    value=35.0,
                    step=1.0,
                    key=f"previous_{semester}_{subject}",
                )

            records.append(
                {
                    "Name": student_name,
                    "University_ID": university_id,
                    "Branch": branch,
                    "Semester": semester,
                    "Subject": subject,
                    "Attendance": attendance,
                    "Study_Hours": study_hours,
                    "Internal": internal,
                    "Assignment": assignment,
                    "Previous_Mark": previous,
                }
            )

        if st.button(
            "💾 Save Student Records",
            width="stretch",
        ):

            if not student_name.strip():
                st.error("Enter student name.")

            elif not university_id.strip():
                st.error("Enter University ID.")

            else:

                new_df = pd.DataFrame(records)

                old_df = load_data()

                # Remove existing same student + same semester
                if not old_df.empty:

                    old_df = old_df[
                        ~(
                            (old_df["University_ID"].astype(str)
                             == university_id.strip())
                            &
                            (old_df["Semester"].astype(str)
                             == semester)
                        )
                    ]

                final_df = pd.concat(
                    [old_df, new_df],
                    ignore_index=True,
                )

                save_data(final_df)

                st.success(
                    f"Saved {student_name}'s {semester} records successfully."
                )

                st.rerun()

    # ========================================================
    # STUDENT DATA
    # ========================================================

    with tabs[1]:

        st.markdown(
            """
            <div class="section-card">
                <h2 style="text-align:center;">
                    📋 Student Data
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if branch_df.empty:

            st.info(
                "No student records available for this department."
            )

        else:

            display_df = add_performance_columns(
                branch_df
            )

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download Branch CSV",
                data=display_df.to_csv(index=False).encode(
                    "utf-8"
                ),
                file_name="branch_student_data.csv",
                mime="text/csv",
                width="stretch",
            )

            st.markdown("### 🗑️ Delete Student")

            student_ids = (
                branch_df["University_ID"]
                .astype(str)
                .unique()
                .tolist()
            )

            selected_delete = st.selectbox(
                "Select University ID",
                student_ids,
            )

            if st.button(
                "🗑️ Delete Complete Student History",
                width="stretch",
            ):

                remaining = branch_df[
                    branch_df["University_ID"].astype(str)
                    != selected_delete
                ]

                # Keep records from other branches
                other_branches = df[
                    df["Branch"].astype(str).str.strip()
                    != branch.strip()
                ]

                final_df = pd.concat(
                    [
                        other_branches,
                        remaining,
                    ],
                    ignore_index=True,
                )

                save_data(final_df)

                st.success(
                    "Student history deleted successfully."
                )

                st.rerun()

    # ========================================================
    # K-MEANS
    # ========================================================

    with tabs[2]:

        st.markdown(
            """
            <div class="section-card">
                <h2 style="text-align:center;">
                    📊 K-Means Student Clustering
                </h2>

                <p style="text-align:center;color:#aebbd0;">
                    Clusters are groups identified from student performance features.
                    Cluster numbers are identifiers, not official grades.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if branch_df.empty:

            st.info("No data available for K-Means.")

        else:

            clustered, model, score = run_kmeans(
                branch_df
            )

            if model is None:

                st.warning(
                    "At least 3 valid records are required."
                )

            else:

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Number of Clusters",
                        model.n_clusters,
                    )

                with c2:

                    if score is not None:

                        st.metric(
                            "Silhouette Score",
                            f"{score:.3f}",
                        )

                    else:

                        st.metric(
                            "Silhouette Score",
                            "N/A",
                        )

                st.dataframe(
                    clustered[
                        [
                            "Name",
                            "University_ID",
                            "Subject",
                            "KMeans_Cluster",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    with tabs[3]:

        st.markdown(
            """
            <div class="section-card">
                <h2 style="text-align:center;">
                    🌲 Random Forest Prediction
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if branch_df.empty:

            st.info(
                "Add student data before running Random Forest."
            )

        else:

            rf_model, rf_data, accuracy = create_rf_model(
                branch_df
            )

            if rf_model is None:

                st.warning(
                    "More student records with multiple performance classes are required."
                )

            else:

                st.metric(
                    "Random Forest Accuracy",
                    f"{accuracy * 100:.2f}%",
                )

                st.dataframe(
                    rf_data[
                        [
                            "Name",
                            "University_ID",
                            "Subject",
                            "Performance",
                            "Overall_Percentage",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

                st.info(
                    "This project uses Random Forest as a demonstration "
                    "of supervised performance classification."
                )

    # ========================================================
    # PROGRESS REPORTS
    # ========================================================

    with tabs[4]:

        st.markdown(
            """
            <div class="section-card">
                <h2 style="text-align:center;">
                    📄 Student Progress Reports
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if branch_df.empty:

            st.info(
                "No students available."
            )

        else:

            student_options = (
                branch_df[
                    [
                        "University_ID",
                        "Name",
                    ]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            selected = st.selectbox(
                "Select Student",
                student_options.index,
                format_func=lambda x:
                    f"{student_options.loc[x, 'Name']} — "
                    f"{student_options.loc[x, 'University_ID']}",
            )

            selected_id = str(
                student_options.loc[
                    selected,
                    "University_ID",
                ]
            )

            selected_student = branch_df[
                branch_df["University_ID"].astype(str)
                == selected_id
            ].copy()

            pdf = create_student_pdf(
                selected_student
            )

            if pdf:

                st.download_button(
                    "📄 Download Student Progress Report",
                    data=pdf,
                    file_name=(
                        f"{selected_student.iloc[0]['Name']}"
                        "_Progress_Report.pdf"
                    ),
                    mime="application/pdf",
                    width="stretch",
                )

                st.markdown(
                    "### Preview"
                )

                preview = add_performance_columns(
                    selected_student
                )

                st.dataframe(
                    preview[
                        [
                            "Semester",
                            "Subject",
                            "Attendance",
                            "Internal",
                            "Assignment",
                            "Previous_Mark",
                            "Overall_Percentage",
                            "Performance",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    st.divider()

    if st.button(
        "🚪 Logout",
        width="stretch",
    ):
        logout()


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.page = "home"
    st.session_state.role = None
    st.session_state.student_logged = False
    st.session_state.tutor_logged = False
    st.session_state.student_username = ""
    st.session_state.student_id = ""
    st.session_state.tutor_branch = ""
    st.session_state.tutor_username = ""

    st.rerun()


# ============================================================
# MAIN ROUTER
# ============================================================

ensure_data_file()

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "student_login":

    student_login()

elif st.session_state.page == "tutor_login":

    tutor_login()

elif st.session_state.page == "student_dashboard":

    if st.session_state.student_logged:
        student_dashboard()
    else:
        st.session_state.page = "student_login"
        st.rerun()

elif st.session_state.page == "tutor_dashboard":

    if st.session_state.tutor_logged:
        tutor_dashboard()
    else:
        st.session_state.page = "tutor_login"
        st.rerun()

else:

    st.session_state.page = "home"
    st.rerun()
