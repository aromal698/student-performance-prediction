import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import hashlib
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
from reportlab.graphics.shapes import Drawing, Circle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM 3D / ANIMATED LOGIN STYLE
# ============================================================

def add_custom_css():

    st.markdown(
        """
        <style>

        /* Main background */
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, rgba(70,130,180,0.25), transparent 25%),
                radial-gradient(circle at 85% 80%, rgba(138,43,226,0.25), transparent 25%),
                linear-gradient(135deg, #07111f, #10243d, #08111d);
        }

        /* Hide Streamlit menu */
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        /* 3D animated background */
        .login-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
            z-index: 0;
        }

        .cube {
            position: absolute;
            width: 80px;
            height: 80px;
            border: 1px solid rgba(255,255,255,0.18);
            transform-style: preserve-3d;
            animation: rotateCube 15s infinite linear;
        }

        .cube:nth-child(1) {
            left: 10%;
            top: 20%;
            animation-duration: 18s;
        }

        .cube:nth-child(2) {
            right: 12%;
            top: 15%;
            width: 120px;
            height: 120px;
            animation-duration: 22s;
        }

        .cube:nth-child(3) {
            left: 20%;
            bottom: 15%;
            width: 100px;
            height: 100px;
            animation-duration: 20s;
        }

        .cube:nth-child(4) {
            right: 25%;
            bottom: 10%;
            width: 60px;
            height: 60px;
            animation-duration: 12s;
        }

        @keyframes rotateCube {
            0% {
                transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg);
            }

            50% {
                transform: rotateX(180deg) rotateY(180deg) rotateZ(90deg);
            }

            100% {
                transform: rotateX(360deg) rotateY(360deg) rotateZ(360deg);
            }
        }

        /* Login card */
        .login-card {
            position: relative;
            z-index: 2;
            padding: 35px;
            border-radius: 25px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.20);
            backdrop-filter: blur(18px);
            box-shadow:
                0 20px 60px rgba(0,0,0,0.45),
                inset 0 0 30px rgba(255,255,255,0.03);
            text-align: center;
        }

        .login-title {
            font-size: 42px;
            font-weight: 800;
            color: white;
            text-shadow: 0 5px 20px rgba(0,0,0,0.4);
        }

        .login-subtitle {
            font-size: 17px;
            color: #c9d8ea;
            margin-bottom: 25px;
        }

        .role-box {
            padding: 18px;
            border-radius: 18px;
            background: rgba(255,255,255,0.06);
            color: white;
            margin-top: 10px;
        }

        /* Dashboard cards */
        .dashboard-card {
            padding: 20px;
            border-radius: 18px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 15px;
        }

        .good-card {
            background: rgba(46, 204, 113, 0.15);
            border: 1px solid rgba(46,204,113,0.35);
        }

        .bad-card {
            background: rgba(231, 76, 60, 0.15);
            border: 1px solid rgba(231,76,60,0.35);
        }

        </style>

        <div class="login-background">
            <div class="cube"></div>
            <div class="cube"></div>
            <div class="cube"></div>
            <div class="cube"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


add_custom_css()


# ============================================================
# FILES
# ============================================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

REPORT_FILE = os.path.join(DATA_DIR, "student_records.csv")


# ============================================================
# BRANCHES
# ============================================================

BRANCHES = [
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning",
    "Computer Science and Engineering",
    "Computer Science and Engineering (AI)",
    "Computer Science and Engineering (Data Science)",
    "Computer Science and Engineering (Cyber Security)",
    "Information Technology",
    "Cyber Security",
    "Data Science",
    "Computer Engineering",
    "Software Engineering",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electronics and Instrumentation Engineering",
    "Electronics Engineering",
    "Instrumentation Engineering",
    "Biomedical Engineering",
    "Mechanical Engineering",
    "Automobile Engineering",
    "Aeronautical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Food Technology",
    "Biotechnology",
    "Industrial Engineering",
    "Production Engineering",
    "Mechatronics Engineering",
    "Robotics and Automation",
    "Environmental Engineering",
    "Marine Engineering",
    "Metallurgical Engineering",
    "Mining Engineering",
    "Textile Engineering",
    "Agricultural Engineering"
]


# ============================================================
# SUBJECT LISTS
# ============================================================

SUBJECTS = {

    "S1": [
        "Mathematics I",
        "Physics",
        "Chemistry",
        "Engineering Graphics",
        "Programming in C",
        "Engineering Mechanics",
        "Basic Electrical Engineering",
        "Basic Civil Engineering",
        "Life Skills",
        "Workshop Practice"
    ],

    "S2": [
        "Mathematics II",
        "Engineering Physics",
        "Engineering Chemistry",
        "Python Programming",
        "Engineering Mechanics",
        "Basic Electronics",
        "Professional Communication",
        "Design and Engineering",
        "Life Skills",
        "Workshop"
    ],

    "S3": [
        "Mathematics III",
        "Data Structures",
        "Database Management Systems",
        "Artificial Intelligence",
        "Machine Learning",
        "Computer Organization",
        "Operating Systems",
        "Object Oriented Programming",
        "Probability and Statistics",
        "Digital Electronics"
    ],

    "S4": [
        "Mathematics IV",
        "Computer Networks",
        "Operating Systems",
        "Software Engineering",
        "Design and Analysis of Algorithms",
        "Data Analytics",
        "Web Programming",
        "Microprocessors",
        "Theory of Computation",
        "Elective I"
    ],

    "S5": [
        "Machine Learning",
        "Artificial Intelligence",
        "Cloud Computing",
        "Computer Networks",
        "Cyber Security",
        "Data Mining",
        "Deep Learning",
        "Big Data Analytics",
        "Elective II",
        "Mini Project"
    ],

    "S6": [
        "Natural Language Processing",
        "Computer Vision",
        "Internet of Things",
        "Big Data",
        "Cloud Computing",
        "Advanced Machine Learning",
        "Elective III",
        "Mini Project",
        "Seminar",
        "Professional Ethics"
    ],

    "S7": [
        "Deep Learning",
        "Generative AI",
        "Reinforcement Learning",
        "Data Science",
        "Blockchain",
        "IoT Analytics",
        "Elective IV",
        "Project Phase I",
        "Seminar",
        "Industrial Training"
    ],

    "S8": [
        "Advanced AI",
        "Advanced Data Science",
        "Explainable AI",
        "AI Ethics",
        "Elective V",
        "Project Phase II",
        "Viva Voce",
        "Technical Seminar",
        "Professional Practice",
        "Comprehensive Viva"
    ]
}


# ============================================================
# DEMO TUTOR ACCOUNTS
# ============================================================

TEACHERS = {

    "teacher_aids": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Data Science"
    },

    "teacher_aiml": {
        "password": "ktutech",
        "branch": "Artificial Intelligence and Machine Learning"
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

    "teacher_cyber": {
        "password": "ktutech",
        "branch": "Cyber Security"
    },

    "teacher_it": {
        "password": "ktutech",
        "branch": "Information Technology"
    },

    "teacher_ece": {
        "password": "ktutech",
        "branch": "Electronics and Communication Engineering"
    },

    "teacher_eee": {
        "password": "ktutech",
        "branch": "Electrical and Electronics Engineering"
    },

    "teacher_me": {
        "password": "ktutech",
        "branch": "Mechanical Engineering"
    },

    "teacher_ce": {
        "password": "ktutech",
        "branch": "Civil Engineering"
    },

    "teacher_robotics": {
        "password": "ktutech",
        "branch": "Robotics and Automation"
    },

    "teacher_mechatronics": {
        "password": "ktutech",
        "branch": "Mechatronics Engineering"
    }
}


# ============================================================
# PERFORMANCE COLORS
# ============================================================

PERFORMANCE_COLORS = {
    "Low Performance": "#F4CCCC",
    "Average Performance": "#FCE5CD",
    "Above Average": "#FFF2CC",
    "Good Performance": "#D9EAD3"
}

PERFORMANCE_TEXT_COLORS = {
    "Low Performance": "#CC0000",
    "Average Performance": "#E69138",
    "Above Average": "#C9A227",
    "Good Performance": "#38761D"
}


# ============================================================
# LOAD / SAVE DATABASE
# ============================================================

REQUIRED_COLUMNS = [
    "Username",
    "University_ID",
    "Student_Name",
    "Branch",
    "Semester",
    "Subject",
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark",
    "Performance",
    "Overall_Percentage",
    "Cluster",
    "RF_Prediction",
    "RF_Confidence",
    "Submitted_By",
    "Submitted_Date"
]


def load_records():

    if os.path.exists(REPORT_FILE):

        try:
            df = pd.read_csv(REPORT_FILE)

            for col in REQUIRED_COLUMNS:

                if col not in df.columns:
                    df[col] = ""

            return df[REQUIRED_COLUMNS]

        except Exception:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def save_records(df):

    df.to_csv(
        REPORT_FILE,
        index=False
    )


# ============================================================
# ATTENDANCE CONVERSION
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

    attendance_score = (
        attendance_mark(attendance) / 5
    ) * 100

    study_score = (
        min(float(study_hours), 6) / 6
    ) * 100

    internal_score = (
        float(internal) / 40
    ) * 100

    assignment_score = (
        float(assignment) / 15
    ) * 100

    previous_score = (
        float(previous) / 60
    ) * 100

    overall = np.mean([
        attendance_score,
        study_score,
        internal_score,
        assignment_score,
        previous_score
    ])

    if overall < 50:

        level = "Low Performance"

    elif overall < 65:

        level = "Average Performance"

    elif overall < 80:

        level = "Above Average"

    else:

        level = "Good Performance"

    return level, round(float(overall), 2)


# ============================================================
# PERFORMANCE FEEDBACK
# ============================================================

def performance_feedback(level):

    if level == "Low Performance":

        return (
            "⚠️ Performance needs improvement. "
            "Increase attendance, revise regularly, complete assignments "
            "and spend more consistent study time."
        )

    elif level == "Average Performance":

        return (
            "🟠 You are making progress. "
            "Regular revision, better attendance and consistent assignments "
            "can improve your performance."
        )

    elif level == "Above Average":

        return (
            "🟡 Good progress. "
            "Small improvements in internal marks, attendance and assignments "
            "can help you reach the Good Performance level."
        )

    return (
        "🟢 Excellent performance! "
        "Keep up the good work and maintain your consistency."
    )


# ============================================================
# PERFORMANCE CIRCLE FOR PDF
# ============================================================

def performance_circle(level):

    colour = PERFORMANCE_TEXT_COLORS.get(
        level,
        "#777777"
    )

    drawing = Drawing(
        18,
        18
    )

    drawing.add(
        Circle(
            9,
            9,
            6,
            fillColor=colors.HexColor(colour),
            strokeColor=colors.HexColor(colour)
        )
    )

    return drawing


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

MODEL_FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]


def run_kmeans(df):

    if df.empty:

        return df

    working = df.copy()

    numeric_columns = MODEL_FEATURES

    for col in numeric_columns:

        working[col] = pd.to_numeric(
            working[col],
            errors="coerce"
        ).fillna(0)

    if len(working) < 2:

        working["Cluster"] = 0

        return working

    X = working[numeric_columns].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    n_clusters = min(
        3,
        len(working)
    )

    if n_clusters < 2:

        working["Cluster"] = 0

        return working

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    working["Cluster"] = model.fit_predict(
        X_scaled
    )

    return working


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(df):

    if df.empty:

        return None

    working = df.copy()

    for col in MODEL_FEATURES:

        working[col] = pd.to_numeric(
            working[col],
            errors="coerce"
        ).fillna(0)

    X = working[MODEL_FEATURES]

    y = working["Performance"].astype(str)

    if len(working) < 10:

        return None

    if y.nunique() < 2:

        return None

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X,
        y
    )

    return model


def predict_with_random_forest(
    model,
    attendance,
    study_hours,
    internal,
    assignment,
    previous
):

    if model is None:

        return None, None

    X = pd.DataFrame(
        [[
            attendance,
            study_hours,
            internal,
            assignment,
            previous
        ]],
        columns=MODEL_FEATURES
    )

    prediction = model.predict(X)[0]

    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[0]

        confidence = round(
            float(max(probabilities)) * 100,
            2
        )

    return prediction, confidence


# ============================================================
# PDF REPORT
# ============================================================

def generate_student_pdf(
    student_df,
    student_name,
    username,
    university_id,
    branch,
    semester
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
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13
    )

    story = []

    story.append(
        Paragraph(
            "🎓 STUDENT PERFORMANCE PROGRESS REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Based Student Performance Prediction System",
            ParagraphStyle(
                "center",
                parent=normal_style,
                alignment=TA_CENTER
            )
        )
    )

    story.append(Spacer(1, 15))

    # --------------------------------------------------------
    # STUDENT DETAILS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Student Details",
            heading_style
        )
    )

    details = [
        ["Student Name", str(student_name)],
        ["Username", str(username)],
        ["University ID", str(university_id)],
        ["Branch", str(branch)],
        ["Semester", str(semester)],
        [
            "Report Date",
            datetime.now().strftime("%d-%m-%Y %H:%M")
        ]
    ]

    table = Table(
        details,
        colWidths=[1.8 * 72, 4.8 * 72]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EAF2F8")),
            ("VALIGN", (0,0), (-1,-1), "TOP")
        ])
    )

    story.append(table)

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall = student_df["Overall_Percentage"].astype(float).mean()

    if overall < 50:

        overall_level = "Low Performance"

    elif overall < 65:

        overall_level = "Average Performance"

    elif overall < 80:

        overall_level = "Above Average"

    else:

        overall_level = "Good Performance"

    story.append(
        Paragraph(
            "2. Overall Information and Marks",
            heading_style
        )
    )

    overall_table = Table(
        [
            ["Overall Percentage", f"{overall:.2f}%"],
            ["Overall Performance", overall_level],
            ["Number of Subjects", str(len(student_df))]
        ],
        colWidths=[2.5 * 72, 4.1 * 72]
    )

    overall_table.setStyle(
        TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            (
                "BACKGROUND",
                (0,0),
                (0,-1),
                colors.HexColor("#E8F8F5")
            )
        ])
    )

    story.append(overall_table)

    # --------------------------------------------------------
    # SUBJECT PERFORMANCE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Subject-wise Performance",
            heading_style
        )
    )

    subject_table_data = [
        [
            "Subject",
            "Attendance",
            "Internal",
            "Assignment",
            "Previous",
            "Study Hr",
            "Performance"
        ]
    ]

    for _, row in student_df.iterrows():

        subject_table_data.append(
            [
                str(row["Subject"]),
                f'{float(row["Attendance"]):.0f}%',
                f'{float(row["Internal_Mark"]):.1f}/40',
                f'{float(row["Assignment"]):.1f}/15',
                f'{float(row["Previous_Mark"]):.1f}/60',
                f'{float(row["Study_Hours"]):.1f}',
                str(row["Performance"])
            ]
        )

    subject_table = Table(
        subject_table_data,
        repeatRows=1,
        colWidths=[
            1.45*72,
            0.7*72,
            0.7*72,
            0.8*72,
            0.7*72,
            0.65*72,
            1.0*72
        ]
    )

    style_commands = [
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9EAF7")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]

    for row_index, (_, row) in enumerate(
        student_df.iterrows(),
        start=1
    ):

        level = str(row["Performance"])

        background = PERFORMANCE_COLORS.get(
            level,
            "#FFFFFF"
        )

        text_colour = PERFORMANCE_TEXT_COLORS.get(
            level,
            "#000000"
        )

        style_commands.extend([
            (
                "BACKGROUND",
                (0,row_index),
                (-1,row_index),
                colors.HexColor(background)
            ),
            (
                "TEXTCOLOR",
                (-1,row_index),
                (-1,row_index),
                colors.HexColor(text_colour)
            )
        ])

    subject_table.setStyle(
        TableStyle(style_commands)
    )

    story.append(subject_table)

    # --------------------------------------------------------
    # ASSESSMENT INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Assessment Information and Conversion",
            heading_style
        )
    )

    conversion = [
        ["Assessment", "Maximum"],
        ["Attendance", "100%"],
        ["Internal Mark", "40"],
        ["Assignment", "15"],
        ["Previous Mark", "60"],
        ["Study Hours", "6 hours/day"]
    ]

    conversion_table = Table(
        conversion,
        colWidths=[3.0*72, 3.0*72]
    )

    conversion_table.setStyle(
        TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#FCE5CD")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9)
        ])
    )

    story.append(conversion_table)

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "<b>Attendance conversion:</b> "
            "90–100% = 5 marks, "
            "80–89% = 4 marks, "
            "70–79% = 3 marks, "
            "60–69% = 2 marks, "
            "10–59% = 1 mark, "
            "below 10% = 0 marks.",
            normal_style
        )
    )

    # --------------------------------------------------------
    # DETAILED SUBJECT ANALYSIS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Detailed Subject-wise Assessment",
            heading_style
        )
    )

    for _, row in student_df.iterrows():

        level = str(row["Performance"])

        feedback = performance_feedback(level)

        cluster = row.get(
            "Cluster",
            0
        )

        prediction = row.get(
            "RF_Prediction",
            level
        )

        confidence = row.get(
            "RF_Confidence",
            ""
        )

        if pd.isna(confidence):
            confidence = ""

        story.append(
            Paragraph(
                f"<b>{row['Subject']}</b>",
                ParagraphStyle(
                    "subject",
                    parent=normal_style,
                    fontSize=11
                )
            )
        )

        detail_data = [
            [
                "Performance",
                f"{level}"
            ],
            [
                "Indicator",
                performance_circle(level)
            ],
            [
                "Overall Score",
                f"{float(row['Overall_Percentage']):.2f}%"
            ],
            [
                "K-Means Cluster",
                str(cluster)
            ],
            [
                "Random Forest",
                str(prediction)
            ],
            [
                "RF Confidence",
                f"{confidence}%"
                if confidence != ""
                else "Not available"
            ],
            [
                "Feedback",
                feedback
            ]
        ]

        detail_table = Table(
            detail_data,
            colWidths=[
                2.0*72,
                4.8*72
            ]
        )

        detail_table.setStyle(
            TableStyle([
                ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 8.5),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                (
                    "BACKGROUND",
                    (0,0),
                    (0,-1),
                    colors.HexColor("#F2F3F4")
                )
            ])
        )

        story.append(detail_table)

        story.append(
            Spacer(1, 8)
        )

    # --------------------------------------------------------
    # K-MEANS EXPLANATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. K-Means Clustering Information",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "K-Means clustering groups students with similar academic "
            "patterns using attendance, study hours, internal marks, "
            "assignment marks and previous marks. Cluster numbers are "
            "identifiers generated by the algorithm and do not themselves "
            "represent a grade.",
            normal_style
        )
    )

    # --------------------------------------------------------
    # IMPROVEMENT METHODS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "7. Improvement Methods",
            heading_style
        )
    )

    improvement_data = [
        [
            "Level",
            "Recommended Action"
        ],
        [
            "🔴 Low",
            "Improve attendance, revise weak topics, complete assignments "
            "on time and increase consistent study hours."
        ],
        [
            "🟠 Average",
            "Maintain regular revision, improve internal preparation and "
            "avoid missing assignments."
        ],
        [
            "🟡 Above Average",
            "Make incremental improvements in internal marks, attendance "
            "and assignment performance."
        ],
        [
            "🟢 Good",
            "Excellent work. Maintain consistency and continue the same "
            "study habits."
        ]
    ]

    improvement_table = Table(
        improvement_data,
        colWidths=[
            1.5*72,
            5.1*72
        ]
    )

    improvement_table.setStyle(
        TableStyle([
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9EAD3")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8.5),
            ("VALIGN", (0,0), (-1,-1), "TOP")
        ])
    )

    story.append(improvement_table)

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "This report is generated from the academic data submitted "
            "by the tutor. Machine-learning results are intended to support "
            "academic monitoring and should not replace teacher judgment.",
            normal_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# CSV TEMPLATE
# ============================================================

def csv_template():

    template = pd.DataFrame([
        {
            "Username": "student001",
            "University_ID": "KTU001",
            "Student_Name": "Student One",
            "Branch": "Artificial Intelligence and Data Science",
            "Semester": "S3",
            "Subject": "Machine Learning",
            "Attendance": 90,
            "Study_Hours": 4,
            "Internal_Mark": 34,
            "Assignment": 13,
            "Previous_Mark": 50
        },
        {
            "Username": "student001",
            "University_ID": "KTU001",
            "Student_Name": "Student One",
            "Branch": "Artificial Intelligence and Data Science",
            "Semester": "S3",
            "Subject": "Database Management Systems",
            "Attendance": 82,
            "Study_Hours": 3,
            "Internal_Mark": 29,
            "Assignment": 11,
            "Previous_Mark": 45
        }
    ])

    return template


# ============================================================
# PROCESS UPLOADED CSV
# ============================================================

def process_uploaded_csv(uploaded_file, tutor_username, tutor_branch):

    try:

        uploaded = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(
            f"Unable to read CSV: {e}"
        )

        return None

    required = [
        "Username",
        "University_ID",
        "Student_Name",
        "Branch",
        "Semester",
        "Subject",
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark"
    ]

    missing = [
        col
        for col in required
        if col not in uploaded.columns
    ]

    if missing:

        st.error(
            "Missing columns: "
            + ", ".join(missing)
        )

        return None

    uploaded = uploaded.copy()

    # Tutor can only upload records belonging to assigned branch
    uploaded["Branch"] = tutor_branch

    numeric_cols = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark"
    ]

    for col in numeric_cols:

        uploaded[col] = pd.to_numeric(
            uploaded[col],
            errors="coerce"
        )

    uploaded = uploaded.dropna(
        subset=numeric_cols
    )

    # Limits
    uploaded["Attendance"] = uploaded[
        "Attendance"
    ].clip(0, 100)

    uploaded["Study_Hours"] = uploaded[
        "Study_Hours"
    ].clip(0, 6)

    uploaded["Internal_Mark"] = uploaded[
        "Internal_Mark"
    ].clip(0, 40)

    uploaded["Assignment"] = uploaded[
        "Assignment"
    ].clip(0, 15)

    uploaded["Previous_Mark"] = uploaded[
        "Previous_Mark"
    ].clip(0, 60)

    # Calculate performance
    performances = []
    percentages = []

    for _, row in uploaded.iterrows():

        level, percentage = calculate_performance(
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"]
        )

        performances.append(level)
        percentages.append(percentage)

    uploaded["Performance"] = performances

    uploaded["Overall_Percentage"] = percentages

    uploaded["Submitted_By"] = tutor_username

    uploaded["Submitted_Date"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # RF
    model = train_random_forest(
        uploaded
    )

    rf_predictions = []
    rf_confidence = []

    for _, row in uploaded.iterrows():

        prediction, confidence = predict_with_random_forest(
            model,
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"]
        )

        rf_predictions.append(
            prediction if prediction is not None
            else row["Performance"]
        )

        rf_confidence.append(
            confidence if confidence is not None
            else ""
        )

    uploaded["RF_Prediction"] = rf_predictions

    uploaded["RF_Confidence"] = rf_confidence

    # KMeans
    uploaded = run_kmeans(
        uploaded
    )

    for col in REQUIRED_COLUMNS:

        if col not in uploaded.columns:
            uploaded[col] = ""

    return uploaded[REQUIRED_COLUMNS]


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    st.markdown(
        """
        <div class="login-card">
            <div class="login-title">👨‍🏫 Tutor Login</div>
            <div class="login-subtitle">
                Academic Management Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        username = st.text_input(
            "Tutor Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button(
            "🔐 Tutor Login",
            use_container_width=True
        ):

            if username in TEACHERS:

                if password == TEACHERS[username]["password"]:

                    st.session_state.logged_in = True
                    st.session_state.role = "tutor"
                    st.session_state.username = username
                    st.session_state.branch = TEACHERS[
                        username
                    ]["branch"]

                    st.success(
                        "Tutor login successful."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Incorrect password."
                    )

            else:

                st.error(
                    "Tutor username not found."
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
                View your academic performance and progress report
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        username = st.text_input(
            "Student Username"
        )

        university_id = st.text_input(
            "University ID"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        st.info(
            "Example student password: BTECH2007"
        )

        if st.button(
            "🎓 Student Login",
            use_container_width=True
        ):

            if len(password) != 9:

                st.error(
                    "Password must contain exactly 9 characters."
                )

                return

            if not password[:5].upper() == "BTECH":

                st.error(
                    "Password must start with BTECH."
                )

                return

            year = password[5:]

            if not year.isdigit():

                st.error(
                    "The last 4 characters must be the birth year."
                )

                return

            year_value = int(year)

            if year_value < 2000 or year_value > 2020:

                st.error(
                    "Birth year must be between 2000 and 2020."
                )

                return

            records = load_records()

            matched = records[
                (
                    records["Username"].astype(str).str.lower()
                    == username.strip().lower()
                )
                &
                (
                    records["University_ID"].astype(str).str.lower()
                    == university_id.strip().lower()
                )
            ]

            if matched.empty:

                st.error(
                    "Student record not found. "
                    "Please check your Username and University ID."
                )

                return

            st.session_state.logged_in = True
            st.session_state.role = "student"
            st.session_state.username = username.strip()
            st.session_state.university_id = university_id.strip()

            st.success(
                "Student login successful."
            )

            st.rerun()


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        """
        <div class="login-card">

            <div class="login-title">
                🎓 Student Performance Prediction
            </div>

            <div class="login-subtitle">
                AI-powered academic performance monitoring system
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="role-box">
                <h2>👨‍🏫 Tutor</h2>
                <p>Upload student academic details and monitor performance.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "👨‍🏫 Tutor Login",
            use_container_width=True
        ):

            st.session_state.login_mode = "tutor"
            st.rerun()

    with col2:

        st.markdown(
            """
            <div class="role-box">
                <h2>🎓 Student</h2>
                <p>View marks, performance and download your progress report.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🎓 Student Login",
            use_container_width=True
        ):

            st.session_state.login_mode = "student"
            st.rerun()


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    tutor_username = st.session_state.username

    tutor_branch = st.session_state.branch

    st.title("👨‍🏫 Tutor Dashboard")

    st.success(
        f"Logged in as {tutor_username} | Branch: {tutor_branch}"
    )

    if st.button("🚪 Logout"):

        st.session_state.clear()

        st.rerun()

    st.divider()

    records = load_records()

    branch_records = records[
        records["Branch"].astype(str)
        == str(tutor_branch)
    ].copy()

    # ========================================================
    # CSV TEMPLATE
    # ========================================================

    st.subheader(
        "📥 Step 1 — Download CSV Template"
    )

    template = csv_template()

    st.download_button(
        "Download Student CSV Template",
        data=template.to_csv(index=False),
        file_name="student_marks_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    st.subheader(
        "📤 Step 2 — Upload Student Details and Marks"
    )

    st.caption(
        "The tutor's assigned branch is automatically applied to uploaded records."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        st.write(
            "Preview:"
        )

        preview = pd.read_csv(
            uploaded_file
        )

        st.dataframe(
            preview,
            use_container_width=True
        )

        if st.button(
            "📤 Submit Student Records",
            type="primary",
            use_container_width=True
        ):

            uploaded_file.seek(0)

            processed = process_uploaded_csv(
                uploaded_file,
                tutor_username,
                tutor_branch
            )

            if processed is not None and not processed.empty:

                existing = load_records()

                # Remove duplicate records for same
                # Username + University ID + Subject
                if not existing.empty:

                    keys = [
                        "Username",
                        "University_ID",
                        "Subject"
                    ]

                    uploaded_keys = set(
                        tuple(x)
                        for x in processed[keys].astype(str).values
                    )

                    existing = existing[
                        ~existing[keys].astype(str).apply(
                            tuple,
                            axis=1
                        ).isin(uploaded_keys)
                    ]

                combined = pd.concat(
                    [
                        existing,
                        processed
                    ],
                    ignore_index=True
                )

                # Recalculate KMeans on complete branch
                # so cluster records are consistent.
                combined_branch = combined[
                    combined["Branch"].astype(str)
                    == str(tutor_branch)
                ].copy()

                combined_branch = run_kmeans(
                    combined_branch
                )

                combined.loc[
                    combined["Branch"].astype(str)
                    == str(tutor_branch),
                    "Cluster"
                ] = combined_branch[
                    "Cluster"
                ].values

                save_records(
                    combined
                )

                st.success(
                    f"Successfully submitted {len(processed)} records."
                )

                st.rerun()

    st.divider()

    # ========================================================
    # MANUAL ENTRY
    # ========================================================

    st.subheader(
        "✍️ Manual Student Entry"
    )

    with st.expander(
        "Open Manual Entry Form"
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            username = st.text_input(
                "Student Username",
                key="manual_username"
            )

            university_id = st.text_input(
                "University ID",
                key="manual_id"
            )

            student_name = st.text_input(
                "Student Name",
                key="manual_name"
            )

        with c2:

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
                key="manual_sem"
            )

            subject = st.selectbox(
                "Subject",
                SUBJECTS[semester],
                key="manual_subject"
            )

        with c3:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=6.0,
                value=3.0,
                step=0.5
            )

        c4, c5, c6 = st.columns(3)

        with c4:

            internal = st.number_input(
                "Internal Mark / 40",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                step=1.0
            )

        with c5:

            assignment = st.number_input(
                "Assignment / 15",
                min_value=0.0,
                max_value=15.0,
                value=10.0,
                step=1.0
            )

        with c6:

            previous = st.number_input(
                "Previous Mark / 60",
                min_value=0.0,
                max_value=60.0,
                value=35.0,
                step=1.0
            )

        if st.button(
            "✅ Submit Student Mark",
            use_container_width=True
        ):

            level, percentage = calculate_performance(
                attendance,
                study_hours,
                internal,
                assignment,
                previous
            )

            record = pd.DataFrame(
                [{
                    "Username": username.strip(),
                    "University_ID": university_id.strip(),
                    "Student_Name": student_name.strip(),
                    "Branch": tutor_branch,
                    "Semester": semester,
                    "Subject": subject,
                    "Attendance": attendance,
                    "Study_Hours": study_hours,
                    "Internal_Mark": internal,
                    "Assignment": assignment,
                    "Previous_Mark": previous,
                    "Performance": level,
                    "Overall_Percentage": percentage,
                    "Cluster": 0,
                    "RF_Prediction": level,
                    "RF_Confidence": "",
                    "Submitted_By": tutor_username,
                    "Submitted_Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }]
            )

            existing = load_records()

            # Remove same student/subject previous record
            if not existing.empty:

                duplicate = (
                    (existing["Username"].astype(str)
                     == username.strip())
                    &
                    (existing["University_ID"].astype(str)
                     == university_id.strip())
                    &
                    (existing["Subject"].astype(str)
                     == subject)
                    &
                    (existing["Branch"].astype(str)
                     == tutor_branch)
                )

                existing = existing[
                    ~duplicate
                ]

            combined = pd.concat(
                [
                    existing,
                    record
                ],
                ignore_index=True
            )

            branch_data = combined[
                combined["Branch"].astype(str)
                == tutor_branch
            ].copy()

            # KMeans
            branch_data = run_kmeans(
                branch_data
            )

            combined.loc[
                combined["Branch"].astype(str)
                == tutor_branch,
                "Cluster"
            ] = branch_data[
                "Cluster"
            ].values

            # RF
            model = train_random_forest(
                branch_data
            )

            for index in combined.index:

                if str(combined.loc[index, "Branch"]) != tutor_branch:
                    continue

                prediction, confidence = predict_with_random_forest(
                    model,
                    combined.loc[index, "Attendance"],
                    combined.loc[index, "Study_Hours"],
                    combined.loc[index, "Internal_Mark"],
                    combined.loc[index, "Assignment"],
                    combined.loc[index, "Previous_Mark"]
                )

                if prediction is not None:

                    combined.loc[
                        index,
                        "RF_Prediction"
                    ] = prediction

                    combined.loc[
                        index,
                        "RF_Confidence"
                    ] = confidence

            save_records(
                combined
            )

            st.success(
                "Student mark submitted successfully."
            )

            st.rerun()

    st.divider()

    # ========================================================
    # STUDENT RECORDS
    # ========================================================

    st.subheader(
        "📊 Branch Student Records"
    )

    records = load_records()

    branch_records = records[
        records["Branch"].astype(str)
        == tutor_branch
    ].copy()

    if branch_records.empty:

        st.info(
            "No student records available for this branch yet."
        )

        return

    st.dataframe(
        branch_records[
            [
                "Username",
                "University_ID",
                "Student_Name",
                "Semester",
                "Subject",
                "Performance",
                "Overall_Percentage",
                "Cluster"
            ]
        ],
        use_container_width=True
    )

    # ========================================================
    # KMEANS RECORDS
    # ========================================================

    st.subheader(
        "📊 K-Means Clustering Records"
    )

    cluster_data = branch_records.copy()

    cluster_summary = (
        cluster_data
        .groupby("Cluster")
        .agg(
            Students=(
                "University_ID",
                "nunique"
            ),
            Average_Performance=(
                "Overall_Percentage",
                "mean"
            ),
            Average_Attendance=(
                "Attendance",
                "mean"
            ),
            Average_Study_Hours=(
                "Study_Hours",
                "mean"
            )
        )
        .reset_index()
    )

    cluster_summary[
        "Average_Performance"
    ] = cluster_summary[
        "Average_Performance"
    ].round(2)

    cluster_summary[
        "Average_Attendance"
    ] = cluster_summary[
        "Average_Attendance"
    ].round(2)

    cluster_summary[
        "Average_Study_Hours"
    ] = cluster_summary[
        "Average_Study_Hours"
    ].round(2)

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    st.caption(
        "K-Means cluster numbers are algorithm-generated group identifiers; "
        "a larger cluster number does not automatically mean better performance."
    )

    # ========================================================
    # SELECT STUDENT
    # ========================================================

    st.subheader(
        "👤 Student History"
    )

    students = (
        branch_records[
            [
                "Username",
                "University_ID",
                "Student_Name"
            ]
        ]
        .drop_duplicates()
    )

    student_options = [
        f"{row.Username} | {row.University_ID} | {row.Student_Name}"
        for row in students.itertuples()
    ]

    selected = st.selectbox(
        "Select Student",
        student_options
    )

    selected_id = selected.split("|")[1].strip()

    selected_student = branch_records[
        branch_records["University_ID"].astype(str)
        == selected_id
    ].copy()

    st.dataframe(
        selected_student,
        use_container_width=True
    )

    # ========================================================
    # DOWNLOAD SELECTED PDF
    # ========================================================

    first = selected_student.iloc[0]

    pdf_data = generate_student_pdf(
        selected_student,
        first["Student_Name"],
        first["Username"],
        first["University_ID"],
        first["Branch"],
        first["Semester"]
    )

    st.download_button(
        "📄 Download Selected Student Progress Report",
        data=pdf_data,
        file_name=f"{first['University_ID']}_progress_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # ========================================================
    # DELETE STUDENT
    # ========================================================

    st.subheader(
        "🗑️ Delete Student History"
    )

    st.warning(
        "This removes all stored subject records for the selected student "
        "from this branch."
    )

    confirm_delete = st.checkbox(
        "I confirm that I want to delete this student's complete history."
    )

    if confirm_delete:

        if st.button(
            "🗑️ Delete Selected Student History",
            type="secondary",
            use_container_width=True
        ):

            all_records = load_records()

            delete_mask = (
                (all_records["University_ID"].astype(str)
                 == selected_id)
                &
                (all_records["Branch"].astype(str)
                 == tutor_branch)
            )

            all_records = all_records[
                ~delete_mask
            ]

            save_records(
                all_records
            )

            st.success(
                "Student history deleted successfully."
            )

            st.rerun()

    # ========================================================
    # DOWNLOAD BRANCH DATA
    # ========================================================

    st.subheader(
        "📥 Branch Data"
    )

    st.download_button(
        "Download Branch Student Records",
        data=branch_records.to_csv(index=False),
        file_name="branch_student_records.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    username = st.session_state.username

    university_id = st.session_state.university_id

    st.title(
        "🎓 Student Dashboard"
    )

    if st.button("🚪 Logout"):

        st.session_state.clear()

        st.rerun()

    records = load_records()

    student_records = records[
        (
            records["Username"].astype(str).str.lower()
            == username.lower()
        )
        &
        (
            records["University_ID"].astype(str).str.lower()
            == university_id.lower()
        )
    ].copy()

    if student_records.empty:

        st.warning(
            "No academic records are available yet."
        )

        return

    student_name = student_records.iloc[0][
        "Student_Name"
    ]

    branch = student_records.iloc[0][
        "Branch"
    ]

    semester = student_records.iloc[0][
        "Semester"
    ]

    st.success(
        f"Welcome, {student_name}!"
    )

    # ========================================================
    # STUDENT DETAILS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Student",
        student_name
    )

    c2.metric(
        "University ID",
        university_id
    )

    c3.metric(
        "Branch",
        branch
    )

    c4.metric(
        "Semester",
        semester
    )

    st.divider()

    # ========================================================
    # OVERALL
    # ========================================================

    overall = student_records[
        "Overall_Percentage"
    ].astype(float).mean()

    if overall < 50:

        overall_level = "Low Performance"

    elif overall < 65:

        overall_level = "Average Performance"

    elif overall < 80:

        overall_level = "Above Average"

    else:

        overall_level = "Good Performance"

    st.subheader(
        "📈 Overall Performance"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Overall Score",
        f"{overall:.2f}%"
    )

    c2.metric(
        "Performance",
        overall_level
    )

    # ========================================================
    # POPUP / MESSAGE
    # ========================================================

    if overall_level == "Good Performance":

        st.balloons()

        st.success(
            "🎉 Excellent! Your performance is good. Keep up the great work!"
        )

    elif overall_level == "Low Performance":

        st.error(
            "😟 Your performance needs improvement. "
            "Don't give up — follow the improvement suggestions below."
        )

    elif overall_level == "Average Performance":

        st.warning(
            "🟠 Your performance is average. "
            "Regular revision and consistency can improve it."
        )

    else:

        st.info(
            "🟡 You are above average. "
            "A little additional effort can improve your performance further."
        )

    st.divider()

    # ========================================================
    # SUBJECT PERFORMANCE
    # ========================================================

    st.subheader(
        "📚 Subject-wise Performance"
    )

    for _, row in student_records.iterrows():

        level = str(
            row["Performance"]
        )

        if level == "Good Performance":

            icon = "🟢"

        elif level == "Above Average":

            icon = "🟡"

        elif level == "Average Performance":

            icon = "🟠"

        else:

            icon = "🔴"

        with st.expander(
            f"{icon} {row['Subject']} — {level}"
        ):

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Overall",
                f"{float(row['Overall_Percentage']):.2f}%"
            )

            c2.metric(
                "Attendance",
                f"{float(row['Attendance']):.0f}%"
            )

            c3.metric(
                "K-Means Cluster",
                str(row["Cluster"])
            )

            marks = pd.DataFrame(
                {
                    "Assessment": [
                        "Internal Mark",
                        "Assignment",
                        "Previous Mark",
                        "Study Hours"
                    ],
                    "Score": [
                        f'{row["Internal_Mark"]}/40',
                        f'{row["Assignment"]}/15',
                        f'{row["Previous_Mark"]}/60',
                        f'{row["Study_Hours"]} hours/day'
                    ]
                }
            )

            st.table(
                marks
            )

            st.markdown(
                f"**Feedback:** {performance_feedback(level)}"
            )

    st.divider()

    # ========================================================
    # DOWNLOAD PDF
    # ========================================================

    st.subheader(
        "📄 Progress Report"
    )

    pdf_data = generate_student_pdf(
        student_records,
        student_name,
        username,
        university_id,
        branch,
        semester
    )

    st.download_button(
        "📥 Download My Progress Report PDF",
        data=pdf_data,
        file_name=f"{university_id}_Progress_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False

if "role" not in st.session_state:

    st.session_state.role = None

if "login_mode" not in st.session_state:

    st.session_state.login_mode = None


# ============================================================
# MAIN ROUTER
# ============================================================

if not st.session_state.logged_in:

    if st.session_state.login_mode == "tutor":

        tutor_login()

        if st.button("⬅️ Back"):

            st.session_state.login_mode = None

            st.rerun()

    elif st.session_state.login_mode == "student":

        student_login()

        if st.button("⬅️ Back"):

            st.session_state.login_mode = None

            st.rerun()

    else:

        login_page()

else:

    if st.session_state.role == "tutor":

        tutor_dashboard()

    elif st.session_state.role == "student":

        student_dashboard()
