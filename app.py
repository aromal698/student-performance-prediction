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
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PROJECT TITLE
# ============================================================

APP_TITLE = "STUDENT PERFORMANCE PREDICTION SYSTEM"
APP_SUBTITLE = "AI-Powered Academic Performance Monitoring System"

# ============================================================
# FOLDERS / DATABASE
# ============================================================

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "student_records.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# SEMESTER SUBJECTS
# ============================================================
# These are configurable project subjects.
# If your college's official KTU subject names differ,
# edit only this dictionary.

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
        "Economics / Engineering Ethics",
    ],

    "S4": [
        "Mathematics for Information Science-IV",
        "Database Management Systems",
        "Operating Systems",
        "Computer Organization and Architecture",
        "Object Oriented Programming",
        "Constitution of India / Professional Ethics",
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
    ],

    "S8": [
        "Major Project Phase-II",
        "Program Elective-V",
        "Program Elective-VI",
        "Comprehensive Viva",
        "Internship / Industrial Training",
    ],
}

# ============================================================
# BRANCHES
# ============================================================

BRANCHES = [
    "Artificial Intelligence and Data Science",
    "Computer Science and Engineering",
    "Information Technology",
    "Electronics and Communication Engineering",
]

# ============================================================
# TUTOR ACCOUNTS
# ============================================================

TUTOR_ACCOUNTS = {
    "teacher_aids": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Data Science",
    },

    "teacher_cse": {
        "password": "ktucse",
        "branch": "Computer Science and Engineering",
    },

    "teacher_it": {
        "password": "ktuit",
        "branch": "Information Technology",
    },

    "teacher_ece": {
        "password": "ktuece",
        "branch": "Electronics and Communication Engineering",
    },
}

# ============================================================
# DATABASE COLUMNS
# ============================================================

COLUMNS = [
    "Student_Name",
    "Username",
    "University_ID",
    "Branch",
    "Semester",
    "Subject",
    "Attendance",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark",
    "Study_Hours",
    "Performance",
    "Performance_Percent",
    "KMeans_Cluster",
]

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(0, 160, 255, 0.18), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(150, 70, 255, 0.16), transparent 25%),
        linear-gradient(135deg, #050816, #0b1024, #071329);
    color: white;
}

.main {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3, h4 {
    color: white !important;
}

p, label, .stMarkdown {
    color: #e6e9f2 !important;
}

.title-main {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

.subtitle-main {
    text-align: center;
    color: #9da8c4 !important;
    font-size: 16px;
    margin-bottom: 30px;
}

.glass-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 25px;
    margin: 10px 0;
    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
}

.login-card {
    max-width: 650px;
    margin: 60px auto 20px auto;
    padding: 45px;
    text-align: center;
    border-radius: 30px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

.login-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.login-subtitle {
    color: #aeb8d0 !important;
    margin-top: 10px;
    font-size: 17px;
}

.dashboard-title {
    font-size: 30px;
    font-weight: 800;
}

.small-muted {
    color: #9da8c4 !important;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.10);
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
}

.metric-label {
    color: #9da8c4 !important;
    font-size: 14px;
}

.subject-card {
    background: rgba(255,255,255,0.055);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.10);
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
}

.footer {
    text-align: center;
    color: #76809b !important;
    margin-top: 40px;
    font-size: 13px;
}

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION = {
    "page": "home",
    "student_logged_in": False,
    "tutor_logged_in": False,
    "student_name": "",
    "student_username": "",
    "student_id": "",
    "student_branch": "",
    "tutor_username": "",
    "tutor_branch": "",
}

for key, value in DEFAULT_SESSION.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def empty_database():
    return pd.DataFrame(columns=COLUMNS)


def load_data():
    if not os.path.exists(DATA_FILE):
        return empty_database()

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        return empty_database()

    # Add missing columns safely
    for col in COLUMNS:
        if col not in df.columns:
            if col == "Semester":
                df[col] = "S3"
            elif col == "Branch":
                df[col] = "Artificial Intelligence and Data Science"
            elif col == "Performance":
                df[col] = ""
            elif col == "Performance_Percent":
                df[col] = 0.0
            elif col == "KMeans_Cluster":
                df[col] = -1
            else:
                df[col] = ""

    # Keep required columns in correct order
    df = df[COLUMNS]

    # Clean Semester
    df["Semester"] = (
        df["Semester"]
        .fillna("S3")
        .astype(str)
        .replace("", "S3")
    )

    # Numeric columns
    numeric_cols = [
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
        "Performance_Percent",
        "KMeans_Cluster",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def save_data(df):
    df = df.copy()

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[COLUMNS]
    df.to_csv(DATA_FILE, index=False)


# ============================================================
# VALIDATION / CALCULATION
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


def calculate_performance(
    attendance,
    internal,
    assignment,
    previous,
    study_hours,
):
    attendance_score = (attendance_mark(attendance) / 5) * 100

    internal_percent = (internal / 40) * 100
    assignment_percent = (assignment / 15) * 100
    previous_percent = (previous / 60) * 100
    study_percent = min((study_hours / 6) * 100, 100)

    overall = np.mean([
        attendance_score,
        internal_percent,
        assignment_percent,
        previous_percent,
        study_percent,
    ])

    if overall < 50:
        level = "Low Performance"
    elif overall < 65:
        level = "Average Performance"
    elif overall < 80:
        level = "Above Average Performance"
    else:
        level = "Good Performance"

    return round(float(overall), 2), level


def performance_color(level):
    if "Low" in level:
        return "#ff4d4d"

    if "Average Performance" == level:
        return "#ff9f43"

    if "Above Average" in level:
        return "#ffd93d"

    return "#35d07f"


def performance_emoji(level):
    if "Low" in level:
        return "😟"

    if "Average Performance" == level:
        return "🟠"

    if "Above Average" in level:
        return "🟡"

    return "🎉"


def improvement_advice(row):
    advice = []

    attendance = float(row["Attendance"])
    internal = float(row["Internal_Mark"])
    assignment = float(row["Assignment"])
    previous = float(row["Previous_Mark"])
    study = float(row["Study_Hours"])

    if attendance < 75:
        advice.append("Improve class attendance and maintain regular participation.")

    if internal < 24:
        advice.append("Give more attention to internal examinations and class preparation.")

    if assignment < 9:
        advice.append("Complete assignments regularly and submit them on time.")

    if previous < 36:
        advice.append("Revise previous topics and strengthen basic concepts.")

    if study < 3:
        advice.append("Increase regular study and revision time.")

    if not advice:
        if "Good" in str(row["Performance"]):
            advice.append("Excellent performance! Keep up the good work.")
        elif "Above Average" in str(row["Performance"]):
            advice.append("Good progress. Small improvements can help reach the next level.")
        else:
            advice.append("Maintain consistency and continue regular revision.")

    return advice


# ============================================================
# PASSWORD
# ============================================================

def valid_student_password(password):
    """
    Required demo format:
    BTECH + four digits from 2000 to 2022
    Example: BTECH2007
    """

    if not isinstance(password, str):
        return False

    if not re.fullmatch(r"BTECH(20\d{2})", password):
        return False

    year = int(password[-4:])

    return 2000 <= year <= 2022


# ============================================================
# K-MEANS
# ============================================================

def run_kmeans(df):
    if df.empty:
        return df, None

    feature_cols = [
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
    ]

    work = df.copy()

    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=feature_cols)

    if len(work) < 3:
        result = df.copy()
        result["KMeans_Cluster"] = -1
        return result, None

    X = work[feature_cols].values

    # Maximum of 3 clusters and minimum possible
    n_clusters = min(3, len(work))

    try:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X)

        result = df.copy()
        result["KMeans_Cluster"] = -1

        result.loc[work.index, "KMeans_Cluster"] = labels

        score = None

        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)

        return result, score

    except Exception:
        result = df.copy()
        result["KMeans_Cluster"] = -1
        return result, None


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(df):
    if df.empty:
        return None, None

    features = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
    ]

    work = df.copy()

    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=features + ["Performance"])

    if len(work) < 10:
        return None, None

    if work["Performance"].nunique() < 2:
        return None, None

    X = work[features]
    y = work["Performance"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)

        return model, accuracy

    except Exception:
        return None, None


# ============================================================
# PDF REPORT
# ============================================================

def build_progress_report(
    student_name,
    university_id,
    branch,
    semester,
    student_df,
):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Student Progress Report",
        author="Student Performance Prediction System",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#163a63"),
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=10,
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#163a63"),
        spaceBefore=8,
        spaceAfter=7,
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
    )

    story = []

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PREDICTION SYSTEM",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "AI-Powered Academic Performance Monitoring System",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "B.Tech – Artificial Intelligence and Data Science",
            subtitle_style,
        )
    )

    story.append(Spacer(1, 5))

    # --------------------------------------------------------
    # STUDENT DETAILS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. STUDENT DETAILS",
            heading_style,
        )
    )

    details = [
        ["Student Name", str(student_name)],
        ["University ID", str(university_id)],
        ["Branch", str(branch)],
        ["Semester", str(semester)],
        ["Number of Subjects", str(len(student_df))],
    ]

    detail_table = Table(
        details,
        colWidths=[45 * mm, 125 * mm],
    )

    detail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2f8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(detail_table)
    story.append(Spacer(1, 10))

    # --------------------------------------------------------
    # OVERALL SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. OVERALL PERFORMANCE SUMMARY",
            heading_style,
        )
    )

    if student_df.empty:
        story.append(
            Paragraph(
                "No performance data is available for this semester.",
                normal_style,
            )
        )
    else:
        avg_percentage = float(
            pd.to_numeric(
                student_df["Performance_Percent"],
                errors="coerce",
            ).mean()
        )

        if avg_percentage < 50:
            overall_level = "Low Performance"
        elif avg_percentage < 65:
            overall_level = "Average Performance"
        elif avg_percentage < 80:
            overall_level = "Above Average Performance"
        else:
            overall_level = "Good Performance"

        cluster_values = student_df["KMeans_Cluster"].tolist()
        valid_clusters = [
            int(x)
            for x in cluster_values
            if str(x) not in ["-1", "nan", ""]
        ]

        cluster_text = (
            str(max(set(valid_clusters), key=valid_clusters.count))
            if valid_clusters
            else "Not available"
        )

        summary = [
            ["Overall Percentage", f"{avg_percentage:.2f}%"],
            ["Overall Performance", overall_level],
            ["K-Means Cluster", cluster_text],
        ]

        summary_table = Table(
            summary,
            colWidths=[55 * mm, 115 * mm],
        )

        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf4ff")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(summary_table)
        story.append(Spacer(1, 10))

    # --------------------------------------------------------
    # SUBJECT-WISE REPORT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. SUBJECT-WISE PERFORMANCE",
            heading_style,
        )
    )

    for index, row in student_df.reset_index(drop=True).iterrows():

        subject = str(row["Subject"])
        attendance = float(row["Attendance"])
        internal = float(row["Internal_Mark"])
        assignment = float(row["Assignment"])
        previous = float(row["Previous_Mark"])
        study = float(row["Study_Hours"])
        performance = str(row["Performance"])
        percentage = float(row["Performance_Percent"])
        cluster = str(row["KMeans_Cluster"])

        att_mark = attendance_mark(attendance)

        advice = improvement_advice(row)

        story.append(
            Paragraph(
                f"<b>{index + 1}. {subject}</b>",
                heading_style,
            )
        )

        subject_data = [
            ["Assessment", "Score / Maximum"],
            ["Attendance", f"{attendance:.1f}% / 100%"],
            ["Attendance Converted Mark", f"{att_mark} / 5"],
            ["Study Hours", f"{study:.1f} / 6 hours/day"],
            ["Internal Mark", f"{internal:.1f} / 40"],
            ["Assignment", f"{assignment:.1f} / 15"],
            ["Previous Mark", f"{previous:.1f} / 60"],
            ["Performance", performance],
            ["Performance Percentage", f"{percentage:.2f}%"],
            ["K-Means Cluster", cluster],
        ]

        subject_table = Table(
            subject_data,
            colWidths=[70 * mm, 100 * mm],
            repeatRows=1,
        )

        subject_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163a63")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        story.append(subject_table)
        story.append(Spacer(1, 5))

        # Performance circle
        circle_color = performance_color(performance)

        circle_table = Table(
            [
                [
                    Paragraph(
                        f'<font color="{circle_color}">●</font>',
                        ParagraphStyle(
                            "Circle",
                            parent=normal_style,
                            fontSize=20,
                        ),
                    ),
                    Paragraph(
                        f"<b>{performance}</b> — {percentage:.2f}%",
                        normal_style,
                    ),
                ]
            ],
            colWidths=[12 * mm, 150 * mm],
        )

        story.append(circle_table)

        story.append(
            Paragraph(
                "<b>Improvement / Guidance:</b>",
                normal_style,
            )
        )

        for item in advice:
            story.append(
                Paragraph(
                    f"• {item}",
                    normal_style,
                )
            )

        story.append(Spacer(1, 10))

    # --------------------------------------------------------
    # ASSESSMENT CONVERSION
    # --------------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "4. ASSESSMENT CONVERSION",
            heading_style,
        )
    )

    conversion_data = [
        ["Component", "Maximum"],
        ["Attendance", "100%"],
        ["Attendance Converted Mark", "5"],
        ["Internal Mark", "40"],
        ["Assignment", "15"],
        ["Previous Mark", "60"],
        ["Study Hours", "6 hours/day"],
    ]

    conversion_table = Table(
        conversion_data,
        colWidths=[100 * mm, 70 * mm],
    )

    conversion_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163a63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(conversion_table)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "<b>Attendance conversion:</b> 90–100 = 5 marks, "
            "80–89 = 4 marks, 70–79 = 3 marks, "
            "60–69 = 2 marks, 10–59 = 1 mark, below 10 = 0 marks.",
            normal_style,
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "<b>Performance classification:</b> "
            "Below 50% = Low Performance, "
            "50–64% = Average Performance, "
            "65–79% = Above Average Performance, "
            "80% and above = Good Performance.",
            normal_style,
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "This classification is a project-defined academic monitoring scale "
            "and is not an official KTU grading system.",
            normal_style,
        )
    )

    story.append(Spacer(1, 25))

    story.append(
        Paragraph(
            "Generated by Student Performance Prediction System",
            subtitle_style,
        )
    )

    doc.build(story)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.markdown(
        """
        <div class="login-card">
            <div class="login-title">
                🎓 STUDENT PERFORMANCE PREDICTION SYSTEM
            </div>

            <div class="login-subtitle">
                AI-Powered Academic Performance Monitoring System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Choose Login",
        unsafe_allow_html=False,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h2>🎓 Student</h2>
                <p>View your semester-wise and subject-wise academic performance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "🎓 Student Login",
            key="go_student_login",
            width="stretch",
        ):
            st.session_state["page"] = "student_login"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h2>👨‍🏫 Tutor</h2>
                <p>Manage student records, analyse performance and generate reports.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "👨‍🏫 Tutor Login",
            key="go_tutor_login",
            width="stretch",
        ):
            st.session_state["page"] = "tutor_login"
            st.rerun()

    st.markdown(
        """
        <div class="footer">
            B.Tech – Artificial Intelligence and Data Science<br>
            Student Academic Analytics Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.markdown(
        """
        <div class="login-card">
            <div class="login-title">🎓 Student Login</div>
            <div class="login-subtitle">
                Access your academic performance
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        username = st.text_input(
            "Username",
            key="student_login_username",
        )

        university_id = st.text_input(
            "University ID",
            key="student_login_id",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="student_login_password",
        )

        st.caption(
            "Demo password format: BTECH + year from 2000 to 2022. "
            "Example: BTECH2007"
        )

        if st.button(
            "Login",
            key="student_login_button",
            width="stretch",
        ):

            if not username.strip():
                st.error("Please enter your username.")
                return

            if not university_id.strip():
                st.error("Please enter your University ID.")
                return

            if not valid_student_password(password):
                st.error(
                    "Invalid password format. Use BTECH2000 to BTECH2022."
                )
                return

            df = load_data()

            if df.empty:
                st.error("No student records are available.")
                return

            student_rows = df[
                (
                    df["Username"].astype(str).str.lower()
                    == username.strip().lower()
                )
                &
                (
                    df["University_ID"].astype(str).str.lower()
                    == university_id.strip().lower()
                )
            ]

            if student_rows.empty:
                st.error(
                    "Student record not found. "
                    "Please check Username and University ID."
                )
                return

            row = student_rows.iloc[0]

            st.session_state["student_logged_in"] = True
            st.session_state["student_name"] = str(row["Student_Name"])
            st.session_state["student_username"] = str(row["Username"])
            st.session_state["student_id"] = str(row["University_ID"])
            st.session_state["student_branch"] = str(row["Branch"])
            st.session_state["page"] = "student_dashboard"

            st.rerun()

    st.divider()

    if st.button(
        "← Back",
        key="student_login_back",
    ):
        st.session_state["page"] = "home"
        st.rerun()


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    st.markdown(
        """
        <div class="login-card">
            <div class="login-title">👨‍🏫 Tutor Login</div>
            <div class="login-subtitle">
                Student academic data management
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        username = st.text_input(
            "Tutor Username",
            key="tutor_login_username",
        )

        password = st.text_input(
            "Tutor Password",
            type="password",
            key="tutor_login_password",
        )

        if st.button(
            "Login",
            key="tutor_login_button",
            width="stretch",
        ):

            account = TUTOR_ACCOUNTS.get(username.strip())

            if account is None:
                st.error("Invalid tutor username.")
                return

            if password != account["password"]:
                st.error("Invalid tutor password.")
                return

            st.session_state["tutor_logged_in"] = True
            st.session_state["tutor_username"] = username.strip()
            st.session_state["tutor_branch"] = account["branch"]
            st.session_state["page"] = "tutor_dashboard"

            st.rerun()

    st.divider()

    if st.button(
        "← Back",
        key="tutor_login_back",
    ):
        st.session_state["page"] = "home"
        st.rerun()


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    if not st.session_state["student_logged_in"]:
        st.session_state["page"] = "student_login"
        st.rerun()

    df = load_data()

    student_id = st.session_state["student_id"]
    student_name = st.session_state["student_name"]
    branch = st.session_state["student_branch"]

    student_df = df[
        df["University_ID"].astype(str) == str(student_id)
    ].copy()

    # --------------------------------------------------------
    # HEADER - ONLY STUDENT NAME
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="dashboard-title">
                🎓 {student_name}
            </div>
            <div class="small-muted">
                Student Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TOP CONTROLS
    # --------------------------------------------------------

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_semester = st.selectbox(
            "Select Semester",
            list(SEMESTERS.keys()),
            key="student_selected_semester",
        )

    with col2:
        st.write("")
        st.write("")

        if st.button(
            "🚪 Logout",
            key="student_logout",
        ):
            st.session_state["student_logged_in"] = False
            st.session_state["page"] = "home"
            st.rerun()

    # --------------------------------------------------------
    # SEMESTER FILTER
    # --------------------------------------------------------

    semester_df = student_df[
        student_df["Semester"].astype(str) == selected_semester
    ].copy()

    subjects = SEMESTERS[selected_semester]

    if semester_df.empty:

        st.info(
            f"No records available for {selected_semester}. "
            "Please contact your tutor."
        )

        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    avg_percentage = float(
        pd.to_numeric(
            semester_df["Performance_Percent"],
            errors="coerce",
        ).mean()
    )

    if avg_percentage < 50:
        overall_level = "Low Performance"
    elif avg_percentage < 65:
        overall_level = "Average Performance"
    elif avg_percentage < 80:
        overall_level = "Above Average Performance"
    else:
        overall_level = "Good Performance"

    st.markdown("### 📊 Semester Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Semester",
            selected_semester,
        )

    with c2:
        st.metric(
            "Overall Percentage",
            f"{avg_percentage:.2f}%",
        )

    with c3:
        st.metric(
            "Performance",
            overall_level,
        )

    # --------------------------------------------------------
    # SUBJECT RESULTS
    # --------------------------------------------------------

    st.markdown("### 📚 Subject-wise Performance")

    for subject in subjects:

        subject_rows = semester_df[
            semester_df["Subject"].astype(str) == subject
        ]

        if subject_rows.empty:
            continue

        row = subject_rows.iloc[0]

        percentage = float(row["Performance_Percent"])
        performance = str(row["Performance"])

        color = performance_color(performance)
        emoji = performance_emoji(performance)

        st.markdown(
            f"""
            <div class="subject-card">
                <h3>{subject}</h3>
                <p>
                    <span style="
                    color:{color};
                    font-size:24px;
                    ">●</span>
                    <b>{emoji} {performance}</b>
                    &nbsp;&nbsp;
                    <b>{percentage:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.write(
                f"Attendance: {float(row['Attendance']):.1f}%"
            )

        with col2:
            st.write(
                f"Internal: {float(row['Internal_Mark']):.1f}/40"
            )

        with col3:
            st.write(
                f"Assignment: {float(row['Assignment']):.1f}/15"
            )

        with col4:
            st.write(
                f"Previous: {float(row['Previous_Mark']):.1f}/60"
            )

        with col5:
            st.write(
                f"Study: {float(row['Study_Hours']):.1f} hrs"
            )

        # ----------------------------------------------------
        # STUDY HOURS
        # ----------------------------------------------------

        st.markdown("**Study Hours Entry**")

        study_key = (
            f"student_study_{selected_semester}_"
            f"{student_id}_{subject}"
        )

        study_value = st.number_input(
            f"Study hours for {subject}",
            min_value=0.0,
            max_value=6.0,
            value=float(row["Study_Hours"]),
            step=0.5,
            key=study_key,
        )

        if st.button(
            f"Calculate Result – {subject}",
            key=f"calc_{selected_semester}_{subject}",
        ):

            new_percentage, new_level = calculate_performance(
                float(row["Attendance"]),
                float(row["Internal_Mark"]),
                float(row["Assignment"]),
                float(row["Previous_Mark"]),
                float(study_value),
            )

            st.success(
                f"{emoji} {subject}: "
                f"{new_level} — {new_percentage:.2f}%"
            )

        st.divider()

    # --------------------------------------------------------
    # PDF DOWNLOAD
    # --------------------------------------------------------

    st.markdown("### 📄 Progress Report")

    pdf_bytes = build_progress_report(
        student_name=student_name,
        university_id=student_id,
        branch=branch,
        semester=selected_semester,
        student_df=semester_df,
    )

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        student_name,
    )

    st.download_button(
        "📥 Download Progress Report PDF",
        data=pdf_bytes,
        file_name=(
            f"{safe_name}_{selected_semester}"
            "_Progress_Report.pdf"
        ),
        mime="application/pdf",
        width="stretch",
    )


# ============================================================
# TUTOR: SAVE STUDENT
# ============================================================

def save_student_records(
    student_name,
    username,
    university_id,
    branch,
    semester,
    records,
):
    df = load_data()

    # Remove existing records for same student + semester
    df = df[
        ~(
            (df["University_ID"].astype(str) == str(university_id))
            &
            (df["Semester"].astype(str) == str(semester))
        )
    ].copy()

    new_rows = []

    for record in records:

        attendance = float(record["Attendance"])
        internal = float(record["Internal_Mark"])
        assignment = float(record["Assignment"])
        previous = float(record["Previous_Mark"])
        study_hours = float(record["Study_Hours"])

        percentage, performance = calculate_performance(
            attendance,
            internal,
            assignment,
            previous,
            study_hours,
        )

        new_rows.append(
            {
                "Student_Name": student_name,
                "Username": username,
                "University_ID": university_id,
                "Branch": branch,
                "Semester": semester,
                "Subject": record["Subject"],
                "Attendance": attendance,
                "Internal_Mark": internal,
                "Assignment": assignment,
                "Previous_Mark": previous,
                "Study_Hours": study_hours,
                "Performance": performance,
                "Performance_Percent": percentage,
                "KMeans_Cluster": -1,
            }
        )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat(
            [df, new_df],
            ignore_index=True,
        )

    save_data(df)

    # Run K-Means after saving
    updated_df = load_data()
    updated_df, _ = run_kmeans(updated_df)

    save_data(updated_df)


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    if not st.session_state["tutor_logged_in"]:
        st.session_state["page"] = "tutor_login"
        st.rerun()

    tutor_branch = st.session_state["tutor_branch"]
    tutor_username = st.session_state["tutor_username"]

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="dashboard-title">
                👨‍🏫 Tutor Dashboard
            </div>

            <div class="small-muted">
                Branch: {tutor_branch}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # LOGOUT
    # --------------------------------------------------------

    if st.button(
        "🚪 Logout",
        key="tutor_logout",
    ):
        st.session_state["tutor_logged_in"] = False
        st.session_state["page"] = "home"
        st.rerun()

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "➕ Add Student",
            "📊 Student Data",
            "🤖 K-Means Analysis",
            "📄 Progress Reports",
            "🗑️ Delete Student",
        ]
    )

    # ========================================================
    # TAB 1 - ADD STUDENT
    # ========================================================

    with tab1:

        st.subheader("Add / Update Student Performance")

        col1, col2 = st.columns(2)

        with col1:
            student_name = st.text_input(
                "Student Name",
                key="tutor_student_name",
            )

        with col2:
            university_id = st.text_input(
                "University ID",
                key="tutor_university_id",
            )

        col1, col2 = st.columns(2)

        with col1:
            username = st.text_input(
                "Student Username",
                key="tutor_student_username",
            )

        with col2:
            selected_semester = st.selectbox(
                "Semester",
                list(SEMESTERS.keys()),
                key="tutor_selected_semester",
            )

        subjects = SEMESTERS[selected_semester]

        st.info(
            f"Enter marks for all {len(subjects)} subjects "
            f"of {selected_semester}."
        )

        # ----------------------------------------------------
        # HORIZONTAL SUBJECT TABLE
        # ----------------------------------------------------

        subject_cols = st.columns(len(subjects))

        for i, subject in enumerate(subjects):
            with subject_cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        font-weight:700;
                        min-height:80px;
                    ">
                        {subject}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        attendance_values = []
        internal_values = []
        assignment_values = []
        previous_values = []
        study_values = []

        # Attendance row
        st.markdown("**Attendance (%)**")

        cols = st.columns(len(subjects))

        for i, subject in enumerate(subjects):
            with cols[i]:
                value = st.number_input(
                    subject,
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=1.0,
                    key=f"att_{selected_semester}_{i}",
                    label_visibility="collapsed",
                )
                attendance_values.append(value)

        # Internal
        st.markdown("**Internal Mark (/40)**")

        cols = st.columns(len(subjects))

        for i, subject in enumerate(subjects):
            with cols[i]:
                value = st.number_input(
                    subject,
                    min_value=0.0,
                    max_value=40.0,
                    value=20.0,
                    step=1.0,
                    key=f"internal_{selected_semester}_{i}",
                    label_visibility="collapsed",
                )
                internal_values.append(value)

        # Assignment
        st.markdown("**Assignment (/15)**")

        cols = st.columns(len(subjects))

        for i, subject in enumerate(subjects):
            with cols[i]:
                value = st.number_input(
                    subject,
                    min_value=0.0,
                    max_value=15.0,
                    value=8.0,
                    step=1.0,
                    key=f"assignment_{selected_semester}_{i}",
                    label_visibility="collapsed",
                )
                assignment_values.append(value)

        # Previous
        st.markdown("**Previous Mark (/60)**")

        cols = st.columns(len(subjects))

        for i, subject in enumerate(subjects):
            with cols[i]:
                value = st.number_input(
                    subject,
                    min_value=0.0,
                    max_value=60.0,
                    value=30.0,
                    step=1.0,
                    key=f"previous_{selected_semester}_{i}",
                    label_visibility="collapsed",
                )
                previous_values.append(value)

        # Study
        st.markdown("**Study Hours / Day**")

        cols = st.columns(len(subjects))

        for i, subject in enumerate(subjects):
            with cols[i]:
                value = st.number_input(
                    subject,
                    min_value=0.0,
                    max_value=6.0,
                    value=2.0,
                    step=0.5,
                    key=f"study_{selected_semester}_{i}",
                    label_visibility="collapsed",
                )
                study_values.append(value)

        st.markdown("---")

        if st.button(
            "💾 Submit All Subject Records",
            key="submit_all_student_records",
            type="primary",
            width="stretch",
        ):

            if not student_name.strip():
                st.error("Enter Student Name.")
                st.stop()

            if not username.strip():
                st.error("Enter Student Username.")
                st.stop()

            if not university_id.strip():
                st.error("Enter University ID.")
                st.stop()

            records = []

            for i, subject in enumerate(subjects):

                records.append(
                    {
                        "Subject": subject,
                        "Attendance": attendance_values[i],
                        "Internal_Mark": internal_values[i],
                        "Assignment": assignment_values[i],
                        "Previous_Mark": previous_values[i],
                        "Study_Hours": study_values[i],
                    }
                )

            save_student_records(
                student_name=student_name.strip(),
                username=username.strip(),
                university_id=university_id.strip(),
                branch=tutor_branch,
                semester=selected_semester,
                records=records,
            )

            st.success(
                f"✅ {student_name} — {selected_semester} "
                "records saved successfully."
            )

            st.balloons()

    # ========================================================
    # TAB 2 - STUDENT DATA
    # ========================================================

    with tab2:

        st.subheader("📊 Student Data")

        df = load_data()

        branch_df = df[
            df["Branch"].astype(str) == tutor_branch
        ].copy()

        if branch_df.empty:
            st.info("No students have been added yet.")
        else:

            selected_semester = st.selectbox(
                "Filter Semester",
                ["All Semesters"] + list(SEMESTERS.keys()),
                key="tutor_data_semester",
            )

            if selected_semester != "All Semesters":
                branch_df = branch_df[
                    branch_df["Semester"].astype(str)
                    == selected_semester
                ]

            if branch_df.empty:
                st.info(
                    f"No student records found for {selected_semester}."
                )
            else:

                st.dataframe(
                    branch_df,
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "📥 Download Branch Data CSV",
                    data=branch_df.to_csv(index=False).encode("utf-8"),
                    file_name="branch_student_data.csv",
                    mime="text/csv",
                    width="stretch",
                )

    # ========================================================
    # TAB 3 - KMEANS
    # ========================================================

    with tab3:

        st.subheader("🤖 K-Means Student Clustering")

        df = load_data()

        branch_df = df[
            df["Branch"].astype(str) == tutor_branch
        ].copy()

        if branch_df.empty:
            st.info("No student data available for analysis.")
        else:

            selected_semester = st.selectbox(
                "Analysis Semester",
                ["All Semesters"] + list(SEMESTERS.keys()),
                key="kmeans_semester",
            )

            analysis_df = branch_df.copy()

            if selected_semester != "All Semesters":
                analysis_df = analysis_df[
                    analysis_df["Semester"].astype(str)
                    == selected_semester
                ]

            if len(analysis_df) < 3:
                st.warning(
                    "At least 3 valid records are required "
                    "for K-Means clustering."
                )
            else:

                clustered_df, silhouette = run_kmeans(
                    analysis_df
                )

                if silhouette is not None:
                    st.metric(
                        "Silhouette Score",
                        f"{silhouette:.3f}",
                    )

                st.info(
                    "K-Means cluster numbers are group identifiers. "
                    "They are not official grades."
                )

                st.dataframe(
                    clustered_df[
                        [
                            "Student_Name",
                            "University_ID",
                            "Semester",
                            "Subject",
                            "Performance",
                            "KMeans_Cluster",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

                if st.button(
                    "💾 Save K-Means Results",
                    key="save_kmeans",
                ):

                    full_df = load_data()

                    for idx in clustered_df.index:
                        full_df.loc[
                            idx,
                            "KMeans_Cluster"
                        ] = clustered_df.loc[
                            idx,
                            "KMeans_Cluster"
                        ]

                    save_data(full_df)

                    st.success(
                        "K-Means cluster assignments saved."
                    )

    # ========================================================
    # TAB 4 - REPORTS
    # ========================================================

    with tab4:

        st.subheader("📄 Student Progress Reports")

        df = load_data()

        branch_df = df[
            df["Branch"].astype(str) == tutor_branch
        ].copy()

        if branch_df.empty:
            st.info("No student records available.")
        else:

            student_list = (
                branch_df[
                    [
                        "Student_Name",
                        "University_ID",
                    ]
                ]
                .drop_duplicates()
                .sort_values("Student_Name")
            )

            student_options = []

            for _, row in student_list.iterrows():
                student_options.append(
                    f"{row['Student_Name']} | "
                    f"{row['University_ID']}"
                )

            selected_student = st.selectbox(
                "Select Student",
                student_options,
                key="report_student",
            )

            selected_id = selected_student.split("|")[-1].strip()

            selected_semester = st.selectbox(
                "Select Semester",
                list(SEMESTERS.keys()),
                key="report_semester",
            )

            report_df = branch_df[
                (
                    branch_df["University_ID"].astype(str)
                    == selected_id
                )
                &
                (
                    branch_df["Semester"].astype(str)
                    == selected_semester
                )
            ].copy()

            if report_df.empty:

                st.warning(
                    "No records are available for this student "
                    "in the selected semester."
                )

            else:

                report_name = str(
                    report_df.iloc[0]["Student_Name"]
                )

                report_pdf = build_progress_report(
                    student_name=report_name,
                    university_id=selected_id,
                    branch=tutor_branch,
                    semester=selected_semester,
                    student_df=report_df,
                )

                st.success(
                    f"Progress report ready for "
                    f"{report_name} — {selected_semester}"
                )

                safe_name = re.sub(
                    r"[^A-Za-z0-9_-]",
                    "_",
                    report_name,
                )

                st.download_button(
                    "📥 Download Progress Report PDF",
                    data=report_pdf,
                    file_name=(
                        f"{safe_name}_"
                        f"{selected_semester}_"
                        "Progress_Report.pdf"
                    ),
                    mime="application/pdf",
                    width="stretch",
                )

                st.markdown("### Preview")

                preview_cols = st.columns(4)

                avg = report_df[
                    "Performance_Percent"
                ].mean()

                with preview_cols[0]:
                    st.metric(
                        "Student",
                        report_name,
                    )

                with preview_cols[1]:
                    st.metric(
                        "Semester",
                        selected_semester,
                    )

                with preview_cols[2]:
                    st.metric(
                        "Average",
                        f"{avg:.2f}%",
                    )

                with preview_cols[3]:
                    st.metric(
                        "Subjects",
                        len(report_df),
                    )

                st.dataframe(
                    report_df[
                        [
                            "Subject",
                            "Attendance",
                            "Internal_Mark",
                            "Assignment",
                            "Previous_Mark",
                            "Study_Hours",
                            "Performance",
                            "Performance_Percent",
                            "KMeans_Cluster",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    # ========================================================
    # TAB 5 - DELETE
    # ========================================================

    with tab5:

        st.subheader("🗑️ Delete Student")

        df = load_data()

        branch_df = df[
            df["Branch"].astype(str) == tutor_branch
        ].copy()

        if branch_df.empty:
            st.info("No student records available.")
        else:

            student_list = (
                branch_df[
                    [
                        "Student_Name",
                        "University_ID",
                    ]
                ]
                .drop_duplicates()
                .sort_values("Student_Name")
            )

            options = []

            for _, row in student_list.iterrows():
                options.append(
                    f"{row['Student_Name']} | "
                    f"{row['University_ID']}"
                )

            delete_student = st.selectbox(
                "Select Student to Delete",
                options,
                key="delete_student_select",
            )

            delete_id = delete_student.split("|")[-1].strip()

            st.warning(
                "Deleting a student removes all semester and "
                "subject records belonging to that University ID."
            )

            confirm = st.checkbox(
                "I confirm that I want to permanently delete this student.",
                key="delete_confirmation",
            )

            if st.button(
                "🗑️ Delete Student Completely",
                key="delete_student_button",
                type="primary",
            ):

                if not confirm:
                    st.error(
                        "Please confirm deletion first."
                    )

                else:

                    before = len(df)

                    df = df[
                        ~(
                            (
                                df["University_ID"]
                                .astype(str)
                                == delete_id
                            )
                            &
                            (
                                df["Branch"]
                                .astype(str)
                                == tutor_branch
                            )
                        )
                    ].copy()

                    after = len(df)

                    save_data(df)

                    st.success(
                        f"Student deleted successfully. "
                        f"{before - after} records removed."
                    )

                    st.rerun()


# ============================================================
# CSV UPLOAD PAGE
# ============================================================

def csv_upload_section():

    st.markdown("### 📤 Upload Existing Student CSV")

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="tutor_csv_upload",
    )

    if uploaded is None:
        return

    try:
        uploaded_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    required = [
        "Student_Name",
        "Username",
        "University_ID",
        "Semester",
        "Subject",
        "Attendance",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark",
        "Study_Hours",
    ]

    missing = [
        col for col in required
        if col not in uploaded_df.columns
    ]

    if missing:
        st.error(
            "Missing columns: "
            + ", ".join(missing)
        )
        return

    st.dataframe(
        uploaded_df,
        width="stretch",
        hide_index=True,
    )

    if st.button(
        "📥 Import CSV",
        key="import_csv_button",
    ):

        current_df = load_data()

        imported = uploaded_df.copy()

        imported["Branch"] = st.session_state["tutor_branch"]

        new_rows = []

        for _, row in imported.iterrows():

            percentage, performance = calculate_performance(
                float(row["Attendance"]),
                float(row["Internal_Mark"]),
                float(row["Assignment"]),
                float(row["Previous_Mark"]),
                float(row["Study_Hours"]),
            )

            new_rows.append(
                {
                    "Student_Name": str(row["Student_Name"]),
                    "Username": str(row["Username"]),
                    "University_ID": str(row["University_ID"]),
                    "Branch": st.session_state["tutor_branch"],
                    "Semester": str(row["Semester"]),
                    "Subject": str(row["Subject"]),
                    "Attendance": float(row["Attendance"]),
                    "Internal_Mark": float(row["Internal_Mark"]),
                    "Assignment": float(row["Assignment"]),
                    "Previous_Mark": float(row["Previous_Mark"]),
                    "Study_Hours": float(row["Study_Hours"]),
                    "Performance": performance,
                    "Performance_Percent": percentage,
                    "KMeans_Cluster": -1,
                }
            )

        imported_final = pd.DataFrame(new_rows)

        current_df = pd.concat(
            [current_df, imported_final],
            ignore_index=True,
        )

        save_data(current_df)

        st.success(
            "CSV imported successfully."
        )

        st.rerun()


# ============================================================
# MAIN ROUTER
# ============================================================

page = st.session_state["page"]

if page == "home":
    home_page()

elif page == "student_login":
    student_login()

elif page == "tutor_login":
    tutor_login()

elif page == "student_dashboard":
    student_dashboard()

elif page == "tutor_dashboard":
    tutor_dashboard()

else:
    st.session_state["page"] = "home"
    st.rerun()
