import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.graphics.shapes import Drawing, Circle


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# DYNAMIC 3D LOGIN DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 15%,
            rgba(0, 180, 255, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 80%,
            rgba(160, 70, 255, 0.22),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #020617,
            #071a33,
            #0b1025,
            #020617
        );
    color: white;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ===============================
   ANIMATED BACKGROUND
   =============================== */

.login-bg {
    position: fixed;
    inset: 0;
    overflow: hidden;
    z-index: -1;
    pointer-events: none;
}

.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(2px);
    opacity: 0.45;
    animation: floatOrb 12s infinite ease-in-out;
}

.orb1 {
    width: 220px;
    height: 220px;
    background: #00c6ff;
    top: 8%;
    left: 8%;
}

.orb2 {
    width: 180px;
    height: 180px;
    background: #9b5cff;
    top: 60%;
    right: 10%;
    animation-delay: 2s;
}

.orb3 {
    width: 130px;
    height: 130px;
    background: #00e5a8;
    bottom: 8%;
    left: 25%;
    animation-delay: 4s;
}

@keyframes floatOrb {

    0% {
        transform:
            translate3d(0,0,0)
            scale(1);
    }

    50% {
        transform:
            translate3d(40px,-35px,0)
            scale(1.12);
    }

    100% {
        transform:
            translate3d(0,0,0)
            scale(1);
    }
}


/* ===============================
   LOGIN CARD
   =============================== */

.login-card {
    max-width: 850px;
    margin: 35px auto 25px auto;
    padding: 42px 35px;
    border-radius: 30px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.15),
            rgba(255,255,255,0.05)
        );

    border:
        1px solid rgba(255,255,255,0.22);

    backdrop-filter: blur(22px);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.12);

    text-align: center;

    animation:
        cardFloat 5s ease-in-out infinite;
}

@keyframes cardFloat {

    0%,100% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-8px);
    }
}

.login-title {
    font-size: 44px;
    font-weight: 900;
    color: white;
    letter-spacing: 0.5px;
}

.login-subtitle {
    margin-top: 12px;
    font-size: 18px;
    color: #c8d8ee;
}


/* ===============================
   ROLE CARDS
   =============================== */

.role-card {
    min-height: 210px;
    padding: 30px 20px;
    border-radius: 25px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.13),
            rgba(255,255,255,0.04)
        );

    border:
        1px solid rgba(255,255,255,0.16);

    backdrop-filter: blur(18px);

    box-shadow:
        0 15px 40px rgba(0,0,0,0.30);

    text-align: center;

    transition:
        transform 0.3s,
        box-shadow 0.3s;
}

.role-card:hover {
    transform:
        translateY(-10px)
        rotateX(3deg);

    box-shadow:
        0 25px 60px rgba(0,0,0,0.45);
}

.role-icon {
    font-size: 55px;
}


/* ===============================
   DASHBOARD CARDS
   =============================== */

.dashboard-card {
    padding: 22px;
    border-radius: 18px;

    background:
        rgba(255,255,255,0.07);

    border:
        1px solid rgba(255,255,255,0.12);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.20);
}


/* ===============================
   PERFORMANCE INDICATOR
   =============================== */

.performance-box {
    padding: 18px;
    border-radius: 16px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.12);
}

</style>


<div class="login-bg">

    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="orb orb3"></div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DATA STORAGE
# ============================================================

DATA_DIR = "data"

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

RECORD_FILE = os.path.join(
    DATA_DIR,
    "student_records.csv"
)


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
    "Biomedical Engineering",
    "Mechanical Engineering",
    "Automobile Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Biotechnology",
    "Food Technology",
    "Mechatronics Engineering",
    "Robotics and Automation"
]


# ============================================================
# SUBJECTS
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
# TUTOR ACCOUNTS
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
# DATA COLUMNS
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


FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]


# ============================================================
# PERFORMANCE DESIGN
# ============================================================

PERFORMANCE_INFO = {

    "Low Performance": {
        "circle": "🔴",
        "colour": "#FF4B4B",
        "message":
            "Performance needs improvement.",
        "advice":
            "Improve attendance, revise weak topics, "
            "complete assignments regularly and increase "
            "daily study consistency."
    },

    "Average Performance": {
        "circle": "🟠",
        "colour": "#FF9800",
        "message":
            "You are making reasonable progress.",
        "advice":
            "Maintain regular revision, improve internal "
            "marks and complete assignments on time."
    },

    "Above Average": {
        "circle": "🟡",
        "colour": "#F4C20D",
        "message":
            "Good progress.",
        "advice":
            "Small incremental improvements in marks, "
            "internal assessment and attendance can help "
            "you reach the good-performance level."
    },

    "Good Performance": {
        "circle": "🟢",
        "colour": "#21C55D",
        "message":
            "Excellent performance!",
        "advice":
            "Excellent work! Keep up the consistency "
            "and continue maintaining your performance."
    }
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(RECORD_FILE):

        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )

    try:

        df = pd.read_csv(
            RECORD_FILE
        )

        for column in REQUIRED_COLUMNS:

            if column not in df.columns:

                df[column] = ""

        return df[
            REQUIRED_COLUMNS
        ]

    except Exception:

        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )


# ============================================================
# SAVE DATA
# ============================================================

def save_data(df):

    df.to_csv(
        RECORD_FILE,
        index=False
    )


# ============================================================
# ATTENDANCE CONVERSION
# ============================================================

def attendance_to_marks(attendance):

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

    attendance_marks = attendance_to_marks(
        attendance
    )

    attendance_percent = (
        attendance_marks / 5
    ) * 100

    study_percent = (
        min(float(study_hours), 6) / 6
    ) * 100

    internal_percent = (
        float(internal) / 40
    ) * 100

    assignment_percent = (
        float(assignment) / 15
    ) * 100

    previous_percent = (
        float(previous) / 60
    ) * 100

    overall = np.mean([
        attendance_percent,
        study_percent,
        internal_percent,
        assignment_percent,
        previous_percent
    ])

    # Project-defined performance levels
    if overall < 50:

        level = "Low Performance"

    elif overall < 65:

        level = "Average Performance"

    elif overall < 80:

        level = "Above Average"

    else:

        level = "Good Performance"

    return (
        level,
        round(float(overall), 2)
    )


# ============================================================
# PERFORMANCE FEEDBACK
# ============================================================

def get_feedback(level):

    info = PERFORMANCE_INFO.get(
        level,
        PERFORMANCE_INFO["Average Performance"]
    )

    return (
        f"{info['circle']} "
        f"**{info['message']}**\n\n"
        f"{info['advice']}"
    )


# ============================================================
# K-MEANS
# ============================================================

def apply_kmeans(df):

    if df.empty:

        return df

    result = df.copy()

    for feature in FEATURES:

        result[feature] = pd.to_numeric(
            result[feature],
            errors="coerce"
        ).fillna(0)

    if len(result) < 2:

        result["Cluster"] = 0

        return result

    scaler = StandardScaler()

    X = scaler.fit_transform(
        result[FEATURES]
    )

    n_clusters = min(
        3,
        len(result)
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    result["Cluster"] = (
        model.fit_predict(X)
    )

    return result


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(df):

    if len(df) < 10:

        return None

    data = df.copy()

    for feature in FEATURES:

        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce"
        ).fillna(0)

    y = data["Performance"].astype(str)

    if y.nunique() < 2:

        return None

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        data[FEATURES],
        y
    )

    return model


def predict_rf(
    model,
    row
):

    if model is None:

        return (
            row["Performance"],
            ""
        )

    X = pd.DataFrame(
        [[
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"]
        ]],
        columns=FEATURES
    )

    prediction = model.predict(X)[0]

    confidence = ""

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = model.predict_proba(X)[0]

        confidence = round(
            float(
                np.max(probability)
            ) * 100,
            2
        )

    return (
        prediction,
        confidence
    )


# ============================================================
# PERFORMANCE CIRCLE FOR PDF
# ============================================================

def make_circle(level):

    colour = PERFORMANCE_INFO.get(
        level,
        PERFORMANCE_INFO["Average Performance"]
    )["colour"]

    drawing = Drawing(
        20,
        20
    )

    drawing.add(
        Circle(
            10,
            10,
            6,
            fillColor=colors.HexColor(
                colour
            ),
            strokeColor=colors.HexColor(
                colour
            )
        )
    )

    return drawing


# ============================================================
# PDF REPORT
# ============================================================

def generate_progress_report(
    student_df
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=8
    )

    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.grey
    )

    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=8
    )

    normal = ParagraphStyle(
        "Normal2",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12
    )

    story = []

    first = student_df.iloc[0]

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE PROGRESS REPORT",
            title
        )
    )

    story.append(
        Paragraph(
            "AI-Based Student Performance Prediction System",
            subtitle
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # STUDENT INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Student Information",
            heading
        )
    )

    student_info = [
        ["Student Name", str(first["Student_Name"])],
        ["Username", str(first["Username"])],
        ["University ID", str(first["University_ID"])],
        ["Branch", str(first["Branch"])],
        ["Semester", str(first["Semester"])],
        [
            "Report Date",
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            )
        ]
    ]

    table = Table(
        student_info,
        colWidths=[
            2.1 * 72,
            4.5 * 72
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "GRID",
                (0,0),
                (-1,-1),
                0.5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0,0),
                (0,-1),
                colors.HexColor("#E8F1F8")
            ),
            (
                "FONTNAME",
                (0,0),
                (0,-1),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0,0),
                (-1,-1),
                8.5
            )
        ])
    )

    story.append(table)

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Overall Academic Information",
            heading
        )
    )

    overall = student_df[
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

    overall_table = Table(
        [
            [
                "Overall Percentage",
                f"{overall:.2f}%"
            ],
            [
                "Overall Performance",
                overall_level
            ],
            [
                "Total Subjects",
                str(len(student_df))
            ]
        ],
        colWidths=[
            2.5 * 72,
            4.1 * 72
        ]
    )

    overall_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0,0),
                (-1,-1),
                0.5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0,0),
                (0,-1),
                colors.HexColor("#EAF6EA")
            ),
            (
                "FONTNAME",
                (0,0),
                (0,-1),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0,0),
                (-1,-1),
                9
            )
        ])
    )

    story.append(
        overall_table
    )

    # --------------------------------------------------------
    # MAXIMUM MARKS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Assessment Maximums",
            heading
        )
    )

    maximums = [
        ["Assessment", "Maximum"],
        ["Attendance", "100%"],
        ["Attendance Converted Mark", "5"],
        ["Internal Mark", "40"],
        ["Previous Mark", "60"],
        ["Assignment Score", "15"],
        ["Study Hours", "6 hours/day"]
    ]

    max_table = Table(
        maximums,
        colWidths=[
            3.5 * 72,
            3 * 72
        ]
    )

    max_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0,0),
                (-1,-1),
                0.5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0,0),
                (-1,0),
                colors.HexColor("#D9EAF7")
            ),
            (
                "FONTNAME",
                (0,0),
                (-1,0),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0,0),
                (-1,-1),
                8.5
            )
        ])
    )

    story.append(
        max_table
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        Paragraph(
            "<b>Attendance conversion:</b> "
            "90–100% = 5 marks; "
            "80–89% = 4 marks; "
            "70–79% = 3 marks; "
            "60–69% = 2 marks; "
            "10–59% = 1 mark; "
            "below 10% = 0 marks.",
            normal
        )
    )

    # --------------------------------------------------------
    # SUBJECT-WISE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Subject-wise Performance",
            heading
        )
    )

    subject_data = [[
        "Subject",
        "Att.",
        "Att. Mark",
        "Internal",
        "Assignment",
        "Previous",
        "Study",
        "Overall",
        "Level"
    ]]

    for _, row in student_df.iterrows():

        subject_data.append([
            str(row["Subject"]),
            f'{float(row["Attendance"]):.0f}%',
            str(
                attendance_to_marks(
                    row["Attendance"]
                )
            ),
            f'{float(row["Internal_Mark"]):.0f}/40',
            f'{float(row["Assignment"]):.0f}/15',
            f'{float(row["Previous_Mark"]):.0f}/60',
            f'{float(row["Study_Hours"]):.1f}',
            f'{float(row["Overall_Percentage"]):.1f}%',
            str(row["Performance"])
        ])

    subject_table = Table(
        subject_data,
        repeatRows=1,
        colWidths=[
            1.25*72,
            .48*72,
            .58*72,
            .65*72,
            .7*72,
            .68*72,
            .48*72,
            .62*72,
            .9*72
        ]
    )

    styles_list = [
        (
            "GRID",
            (0,0),
            (-1,-1),
            0.35,
            colors.grey
        ),
        (
            "BACKGROUND",
            (0,0),
            (-1,0),
            colors.HexColor("#D9EAF7")
        ),
        (
            "FONTNAME",
            (0,0),
            (-1,0),
            "Helvetica-Bold"
        ),
        (
            "FONTSIZE",
            (0,0),
            (-1,-1),
            6.5
        ),
        (
            "VALIGN",
            (0,0),
            (-1,-1),
            "MIDDLE"
        )
    ]

    for i, (_, row) in enumerate(
        student_df.iterrows(),
        start=1
    ):

        level = str(
            row["Performance"]
        )

        if level == "Low Performance":

            bg = "#F4CCCC"

        elif level == "Average Performance":

            bg = "#FCE5CD"

        elif level == "Above Average":

            bg = "#FFF2CC"

        else:

            bg = "#D9EAD3"

        styles_list.append(
            (
                "BACKGROUND",
                (0,i),
                (-1,i),
                colors.HexColor(bg)
            )
        )

    subject_table.setStyle(
        TableStyle(styles_list)
    )

    story.append(
        subject_table
    )

    # --------------------------------------------------------
    # EACH SUBJECT SEPARATELY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Detailed Subject Reports",
            heading
        )
    )

    for _, row in student_df.iterrows():

        level = str(
            row["Performance"]
        )

        info = PERFORMANCE_INFO[
            level
        ]

        detail_data = [
            [
                "Subject",
                str(row["Subject"])
            ],
            [
                "Performance Indicator",
                make_circle(level)
            ],
            [
                "Performance Level",
                f"{info['circle']} {level}"
            ],
            [
                "Attendance",
                f'{float(row["Attendance"]):.0f}% '
                f'→ {attendance_to_marks(row["Attendance"])} / 5'
            ],
            [
                "Internal Mark",
                f'{float(row["Internal_Mark"]):.0f} / 40'
            ],
            [
                "Previous Mark",
                f'{float(row["Previous_Mark"]):.0f} / 60'
            ],
            [
                "Assignment Score",
                f'{float(row["Assignment"]):.0f} / 15'
            ],
            [
                "Study Hours",
                f'{float(row["Study_Hours"]):.1f} / 6 hours'
            ],
            [
                "Overall Score",
                f'{float(row["Overall_Percentage"]):.2f}%'
            ],
            [
                "K-Means Cluster",
                str(row["Cluster"])
            ],
            [
                "Random Forest Prediction",
                str(row["RF_Prediction"])
            ],
            [
                "RF Confidence",
                (
                    f'{row["RF_Confidence"]}%'
                    if str(row["RF_Confidence"]) != ""
                    else "Not available"
                )
            ]
        ]

        detail_table = Table(
            detail_data,
            colWidths=[
                2.4*72,
                4.2*72
            ]
        )

        detail_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0,0),
                    (-1,-1),
                    0.4,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0,0),
                    (0,-1),
                    colors.HexColor("#F2F4F5")
                ),
                (
                    "FONTNAME",
                    (0,0),
                    (0,-1),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0,0),
                    (-1,-1),
                    8
                ),
                (
                    "VALIGN",
                    (0,0),
                    (-1,-1),
                    "MIDDLE"
                )
            ])
        )

        story.append(
            Paragraph(
                f"{info['circle']} "
                f"{row['Subject']}",
                heading
            )
        )

        story.append(
            detail_table
        )

        story.append(
            Spacer(1, 5)
        )

        story.append(
            Paragraph(
                f"<b>Feedback:</b> {info['message']}",
                normal
            )
        )

        story.append(
            Paragraph(
                f"<b>Improvement / Recommendation:</b> "
                f"{info['advice']}",
                normal
            )
        )

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. K-Means Clustering Analysis",
            heading
        )
    )

    story.append(
        Paragraph(
            "K-Means clustering groups records according to "
            "similar academic patterns using attendance, study "
            "hours, internal marks, assignment scores and previous "
            "marks. The cluster number is only an algorithmic "
            "identifier and is not itself a grade.",
            normal
        )
    )

    # --------------------------------------------------------
    # FINAL NOTE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "7. Final Academic Note",
            heading
        )
    )

    story.append(
        Paragraph(
            "This report is intended to help tutors and students "
            "monitor academic progress. Machine-learning predictions "
            "should be used as supporting information and should "
            "not replace teacher evaluation or academic judgment.",
            normal
        )
    )

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# CSV TEMPLATE
# ============================================================

def create_csv_template():

    template = pd.DataFrame([
        {
            "Username": "student001",
            "University_ID": "KTU001",
            "Student_Name": "Student One",
            "Branch":
                "Artificial Intelligence and Data Science",
            "Semester": "S3",
            "Subject": "Machine Learning",
            "Attendance": 90,
            "Study_Hours": 4,
            "Internal_Mark": 35,
            "Assignment": 14,
            "Previous_Mark": 50
        },
        {
            "Username": "student001",
            "University_ID": "KTU001",
            "Student_Name": "Student One",
            "Branch":
                "Artificial Intelligence and Data Science",
            "Semester": "S3",
            "Subject":
                "Database Management Systems",
            "Attendance": 85,
            "Study_Hours": 3,
            "Internal_Mark": 30,
            "Assignment": 12,
            "Previous_Mark": 45
        }
    ])

    return template


# ============================================================
# PROCESS UPLOADED CSV
# ============================================================

def process_uploaded_csv(
    uploaded_file,
    tutor_username,
    tutor_branch
):

    df = pd.read_csv(
        uploaded_file
    )

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
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        st.error(
            "Missing columns: "
            + ", ".join(missing)
        )

        return None

    df = df.copy()

    # Tutor's assigned branch is authoritative
    df["Branch"] = tutor_branch

    numeric_columns = [
        "Attendance",
        "Study_Hours",
        "Internal_Mark",
        "Assignment",
        "Previous_Mark"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_columns
    )

    # Apply limits
    df["Attendance"] = df[
        "Attendance"
    ].clip(0, 100)

    df["Study_Hours"] = df[
        "Study_Hours"
    ].clip(0, 6)

    df["Internal_Mark"] = df[
        "Internal_Mark"
    ].clip(0, 40)

    df["Assignment"] = df[
        "Assignment"
    ].clip(0, 15)

    df["Previous_Mark"] = df[
        "Previous_Mark"
    ].clip(0, 60)

    performance = []
    overall = []

    for _, row in df.iterrows():

        level, score = calculate_performance(
            row["Attendance"],
            row["Study_Hours"],
            row["Internal_Mark"],
            row["Assignment"],
            row["Previous_Mark"]
        )

        performance.append(
            level
        )

        overall.append(
            score
        )

    df["Performance"] = performance

    df["Overall_Percentage"] = overall

    df["Submitted_By"] = tutor_username

    df["Submitted_Date"] = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    df = apply_kmeans(
        df
    )

    model = train_random_forest(
        df
    )

    predictions = []
    confidence = []

    for _, row in df.iterrows():

        prediction, conf = predict_rf(
            model,
            row
        )

        predictions.append(
            prediction
        )

        confidence.append(
            conf
        )

    df["RF_Prediction"] = predictions

    df["RF_Confidence"] = confidence

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:

            df[column] = ""

    return df[
        REQUIRED_COLUMNS
    ]


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login_page():

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

    left, right = st.columns(
        2
    )

    with left:

        st.markdown(
            """
            <div class="role-card">

                <div class="role-icon">
                    👨‍🏫
                </div>

                <h2>Tutor</h2>

                <p>
                    Upload and manage student academic data,
                    analyse performance and generate reports.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "👨‍🏫 Tutor Login",
            use_container_width=True
        ):

            st.session_state.login_mode = "tutor"

            st.rerun()

    with right:

        st.markdown(
            """
            <div class="role-card">

                <div class="role-icon">
                    🎓
                </div>

                <h2>Student</h2>

                <p>
                    View your own academic performance,
                    subject details and progress report.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🎓 Student Login",
            use_container_width=True
        ):

            st.session_state.login_mode = "student"

            st.rerun()


# ============================================================
# TUTOR LOGIN
# ============================================================

def tutor_login():

    st.markdown(
        """
        <div class="login-card">

            <div class="login-title">
                👨‍🏫 Tutor Login
            </div>

            <div class="login-subtitle">
                Student Academic Management
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    left, middle, right = st.columns(
        [1, 2, 1]
    )

    with middle:

        username = st.text_input(
            "Tutor Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button(
            "🔐 Login",
            type="primary",
            use_container_width=True
        ):

            if username in TEACHERS:

                if password == TEACHERS[
                    username
                ]["password"]:

                    st.session_state.logged_in = True

                    st.session_state.role = "tutor"

                    st.session_state.username = username

                    st.session_state.branch = TEACHERS[
                        username
                    ]["branch"]

                    st.rerun()

                else:

                    st.error(
                        "Incorrect password."
                    )

            else:

                st.error(
                    "Tutor username not found."
                )

        if st.button(
            "⬅️ Back",
            use_container_width=True
        ):

            st.session_state.login_mode = None

            st.rerun()


# ============================================================
# STUDENT LOGIN
# ============================================================

def student_login():

    st.markdown(
        """
        <div class="login-card">

            <div class="login-title">
                🎓 Student Login
            </div>

            <div class="login-subtitle">
                View your academic performance
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    left, middle, right = st.columns(
        [1, 2, 1]
    )

    with middle:

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
            "Example password: BTECH2007"
        )

        if st.button(
            "🎓 Login",
            type="primary",
            use_container_width=True
        ):

            if len(password) != 9:

                st.error(
                    "Password must contain exactly 9 characters."
                )

                return

            if password[:5].upper() != "BTECH":

                st.error(
                    "Password must start with BTECH."
                )

                return

            year = password[5:]

            if not year.isdigit():

                st.error(
                    "The last 4 characters must be a year."
                )

                return

            year = int(year)

            if year < 2000 or year > 2020:

                st.error(
                    "Use a valid birth year between 2000 and 2020."
                )

                return

            df = load_data()

            student = df[
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

            if student.empty:

                st.error(
                    "Student record not found."
                )

                return

            st.session_state.logged_in = True

            st.session_state.role = "student"

            st.session_state.username = username.strip()

            st.session_state.university_id = (
                university_id.strip()
            )

            st.rerun()

        if st.button(
            "⬅️ Back",
            use_container_width=True
        ):

            st.session_state.login_mode = None

            st.rerun()


# ============================================================
# TUTOR DASHBOARD
# ============================================================

def tutor_dashboard():

    username = st.session_state.username

    branch = st.session_state.branch

    st.title(
        "👨‍🏫 Tutor Dashboard"
    )

    st.success(
        f"Logged in as: {username} | Branch: {branch}"
    )

    if st.button(
        "🚪 Logout"
    ):

        st.session_state.clear()

        st.rerun()

    st.divider()

    df = load_data()

    branch_df = df[
        df["Branch"].astype(str)
        == branch
    ].copy()

    # ========================================================
    # OVERALL ANALYSIS
    # ========================================================

    st.header(
        "📊 Student Data Analysis"
    )

    total_students = (
        branch_df["University_ID"]
        .nunique()
        if not branch_df.empty
        else 0
    )

    total_subject_records = len(
        branch_df
    )

    average_score = (
        branch_df["Overall_Percentage"]
        .astype(float)
        .mean()
        if not branch_df.empty
        else 0
    )

    low_count = (
        (
            branch_df["Performance"]
            == "Low Performance"
        )
        .sum()
        if not branch_df.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👨‍🎓 Total Students",
        total_students
    )

    c2.metric(
        "📚 Subject Records",
        total_subject_records
    )

    c3.metric(
        "📈 Average Score",
        f"{average_score:.2f}%"
    )

    c4.metric(
        "🔴 Low Performance",
        low_count
    )

    # Performance distribution
    if not branch_df.empty:

        st.subheader(
            "Performance Distribution"
        )

        distribution = (
            branch_df[
                "Performance"
            ]
            .value_counts()
            .rename_axis(
                "Performance"
            )
            .reset_index(
                name="Students"
            )
        )

        st.dataframe(
            distribution,
            use_container_width=True
        )

    st.divider()

    # ========================================================
    # CSV TEMPLATE
    # ========================================================

    st.header(
        "📥 Student Data Upload"
    )

    template = create_csv_template()

    st.download_button(
        "📥 Download CSV Template",
        data=template.to_csv(
            index=False
        ),
        file_name="student_marks_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    uploaded = st.file_uploader(
        "Upload Student CSV",
        type=["csv"]
    )

    if uploaded is not None:

        preview = pd.read_csv(
            uploaded
        )

        st.subheader(
            "Uploaded Data Preview"
        )

        st.dataframe(
            preview,
            use_container_width=True
        )

        if st.button(
            "✅ Submit Student Records",
            type="primary",
            use_container_width=True
        ):

            uploaded.seek(0)

            new_data = process_uploaded_csv(
                uploaded,
                username,
                branch
            )

            if new_data is not None:

                current = load_data()

                # Remove same student-subject records
                if not current.empty:

                    for _, new_row in new_data.iterrows():

                        mask = (
                            (current["Username"].astype(str)
                             == str(new_row["Username"]))
                            &
                            (current["University_ID"].astype(str)
                             == str(new_row["University_ID"]))
                            &
                            (current["Subject"].astype(str)
                             == str(new_row["Subject"]))
                            &
                            (current["Branch"].astype(str)
                             == branch)
                        )

                        current = current[
                            ~mask
                        ]

                combined = pd.concat(
                    [
                        current,
                        new_data
                    ],
                    ignore_index=True
                )

                # Recalculate branch clustering
                branch_data = combined[
                    combined["Branch"].astype(str)
                    == branch
                ].copy()

                branch_data = apply_kmeans(
                    branch_data
                )

                combined.loc[
                    combined["Branch"].astype(str) == branch,
                    "Cluster"
                ] = branch_data[
                    "Cluster"
                ].values

                # Random Forest
                model = train_random_forest(
                    branch_data
                )

                for idx in combined.index:

                    if str(
                        combined.loc[idx, "Branch"]
                    ) != branch:

                        continue

                    prediction, confidence = predict_rf(
                        model,
                        combined.loc[idx]
                    )

                    combined.loc[
                        idx,
                        "RF_Prediction"
                    ] = prediction

                    combined.loc[
                        idx,
                        "RF_Confidence"
                    ] = confidence

                save_data(
                    combined
                )

                st.success(
                    "Student records submitted successfully."
                )

                st.rerun()

    st.divider()

    # ========================================================
    # MANUAL ENTRY
    # ========================================================

    st.header(
        "✍️ Manual Student Entry"
    )

    with st.expander(
        "Open Manual Student Entry"
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            username_m = st.text_input(
                "Student Username",
                key="username_manual"
            )

            university_m = st.text_input(
                "University ID",
                key="university_manual"
            )

            name_m = st.text_input(
                "Student Name",
                key="name_manual"
            )

        with c2:

            semester_m = st.selectbox(
                "Semester",
                list(SUBJECTS.keys()),
                key="semester_manual"
            )

            subject_m = st.selectbox(
                "Subject",
                SUBJECTS[semester_m],
                key="subject_manual"
            )

        with c3:

            attendance_m = st.number_input(
                "Attendance (%)",
                0.0,
                100.0,
                80.0,
                1.0,
                key="attendance_manual"
            )

            study_m = st.number_input(
                "Study Hours / Day",
                0.0,
                6.0,
                3.0,
                0.5,
                key="study_manual"
            )

        c4, c5, c6 = st.columns(3)

        with c4:

            internal_m = st.number_input(
                "Internal Mark / 40",
                0.0,
                40.0,
                25.0,
                1.0,
                key="internal_manual"
            )

        with c5:

            assignment_m = st.number_input(
                "Assignment / 15",
                0.0,
                15.0,
                10.0,
                1.0,
                key="assignment_manual"
            )

        with c6:

            previous_m = st.number_input(
                "Previous Mark / 60",
                0.0,
                60.0,
                35.0,
                1.0,
                key="previous_manual"
            )

        if st.button(
            "✅ Submit Manual Record",
            type="primary",
            use_container_width=True
        ):

            if not username_m.strip():

                st.error(
                    "Enter student username."
                )

                return

            if not university_m.strip():

                st.error(
                    "Enter University ID."
                )

                return

            if not name_m.strip():

                st.error(
                    "Enter student name."
                )

                return

            level, score = calculate_performance(
                attendance_m,
                study_m,
                internal_m,
                assignment_m,
                previous_m
            )

            new_row = pd.DataFrame([
                {
                    "Username":
                        username_m.strip(),

                    "University_ID":
                        university_m.strip(),

                    "Student_Name":
                        name_m.strip(),

                    "Branch":
                        branch,

                    "Semester":
                        semester_m,

                    "Subject":
                        subject_m,

                    "Attendance":
                        attendance_m,

                    "Study_Hours":
                        study_m,

                    "Internal_Mark":
                        internal_m,

                    "Assignment":
                        assignment_m,

                    "Previous_Mark":
                        previous_m,

                    "Performance":
                        level,

                    "Overall_Percentage":
                        score,

                    "Cluster":
                        0,

                    "RF_Prediction":
                        level,

                    "RF_Confidence":
                        "",

                    "Submitted_By":
                        username,

                    "Submitted_Date":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                }
            ])

            current = load_data()

            if not current.empty:

                duplicate = (
                    (current["Username"].astype(str)
                     == username_m.strip())
                    &
                    (current["University_ID"].astype(str)
                     == university_m.strip())
                    &
                    (current["Subject"].astype(str)
                     == subject_m)
                    &
                    (current["Branch"].astype(str)
                     == branch)
                )

                current = current[
                    ~duplicate
                ]

            combined = pd.concat(
                [
                    current,
                    new_row
                ],
                ignore_index=True
            )

            branch_data = combined[
                combined["Branch"].astype(str)
                == branch
            ].copy()

            branch_data = apply_kmeans(
                branch_data
            )

            combined.loc[
                combined["Branch"].astype(str) == branch,
                "Cluster"
            ] = branch_data[
                "Cluster"
            ].values

            model = train_random_forest(
                branch_data
            )

            for idx in combined.index:

                if str(
                    combined.loc[idx, "Branch"]
                ) != branch:

                    continue

                prediction, confidence = predict_rf(
                    model,
                    combined.loc[idx]
                )

                combined.loc[
                    idx,
                    "RF_Prediction"
                ] = prediction

                combined.loc[
                    idx,
                    "RF_Confidence"
                ] = confidence

            save_data(
                combined
            )

            st.success(
                "Student record added successfully."
            )

            st.rerun()

    st.divider()

    # ========================================================
    # K-MEANS ANALYSIS
    # ========================================================

    st.header(
        "🔵 K-Means Student Clustering"
    )

    if not branch_df.empty:

        clustered = apply_kmeans(
            branch_df
        )

        cluster_summary = (
            clustered
            .groupby("Cluster")
            .agg(
                Students=(
                    "University_ID",
                    "nunique"
                ),
                Average_Score=(
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
            "Average_Score"
        ] = cluster_summary[
            "Average_Score"
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
            "Cluster numbers are algorithmic group identifiers, "
            "not grades."
        )

    st.divider()

    # ========================================================
    # STUDENT LIST
    # ========================================================

    st.header(
        "👨‍🎓 Manage Students"
    )

    branch_df = load_data()

    branch_df = branch_df[
        branch_df["Branch"].astype(str)
        == branch
    ].copy()

    if branch_df.empty:

        st.info(
            "No student data available."
        )

        return

    students = (
        branch_df[
            [
                "Username",
                "University_ID",
                "Student_Name"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    st.dataframe(
        students,
        use_container_width=True
    )

    # ========================================================
    # SELECT STUDENT
    # ========================================================

    choices = []

    for _, row in students.iterrows():

        choices.append(
            f"{row['Username']} | "
            f"{row['University_ID']} | "
            f"{row['Student_Name']}"
        )

    selected = st.selectbox(
        "Select Student",
        choices
    )

    selected_id = (
        selected
        .split("|")[1]
        .strip()
    )

    selected_student = branch_df[
        branch_df["University_ID"].astype(str)
        == selected_id
    ].copy()

    # ========================================================
    # VIEW STUDENT DATA
    # ========================================================

    st.subheader(
        "📋 Selected Student Records"
    )

    st.dataframe(
        selected_student,
        use_container_width=True
    )

    # ========================================================
    # DOWNLOAD PROGRESS REPORT
    # ========================================================

    st.subheader(
        "📄 Student Progress Report"
    )

    pdf = generate_progress_report(
        selected_student
    )

    st.download_button(
        "📥 Download Selected Student Progress Report",
        data=pdf,
        file_name=(
            f"{selected_id}_Progress_Report.pdf"
        ),
        mime="application/pdf",
        use_container_width=True
    )

    # ========================================================
    # DELETE STUDENT ONE BY ONE
    # ========================================================

    st.subheader(
        "🗑️ Delete Student"
    )

    st.warning(
        "Deleting a student removes all subject records "
        "for that student from the tutor's branch."
    )

    confirm_delete = st.checkbox(
        "I confirm that I want to delete this student's complete data."
    )

    if confirm_delete:

        if st.button(
            "🗑️ Delete Selected Student",
            type="secondary",
            use_container_width=True
        ):

            all_data = load_data()

            mask = (
                (all_data["University_ID"].astype(str)
                 == selected_id)
                &
                (all_data["Branch"].astype(str)
                 == branch)
            )

            all_data = all_data[
                ~mask
            ]

            save_data(
                all_data
            )

            st.success(
                "Student data deleted successfully."
            )

            st.rerun()

    # ========================================================
    # DOWNLOAD BRANCH DATA
    # ========================================================

    st.subheader(
        "📥 Export Branch Data"
    )

    st.download_button(
        "📥 Download Complete Branch CSV",
        data=branch_df.to_csv(
            index=False
        ),
        file_name="branch_student_data.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    username = st.session_state.username

    university_id = (
        st.session_state.university_id
    )

    st.title(
        "🎓 Student Dashboard"
    )

    if st.button(
        "🚪 Logout"
    ):

        st.session_state.clear()

        st.rerun()

    df = load_data()

    student_df = df[
        (
            df["Username"].astype(str).str.lower()
            == username.lower()
        )
        &
        (
            df["University_ID"].astype(str).str.lower()
            == university_id.lower()
        )
    ].copy()

    if student_df.empty:

        st.warning(
            "No academic records found."
        )

        return

    first = student_df.iloc[0]

    student_name = first["Student_Name"]

    branch = first["Branch"]

    semester = first["Semester"]

    st.success(
        f"Welcome {student_name}! 👋"
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

    st.header(
        "📈 Overall Academic Performance"
    )

    overall = (
        student_df[
            "Overall_Percentage"
        ]
        .astype(float)
        .mean()
    )

    if overall < 50:

        overall_level = "Low Performance"

    elif overall < 65:

        overall_level = "Average Performance"

    elif overall < 80:

        overall_level = "Above Average"

    else:

        overall_level = "Good Performance"

    info = PERFORMANCE_INFO[
        overall_level
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "Overall Score",
        f"{overall:.2f}%"
    )

    c2.metric(
        "Overall Level",
        f"{info['circle']} {overall_level}"
    )

    # ========================================================
    # FEEDBACK
    # ========================================================

    if overall_level == "Good Performance":

        st.balloons()

        st.success(
            "🎉 Excellent performance! "
            "Keep up the good work!"
        )

    elif overall_level == "Low Performance":

        st.error(
            "🔴 Performance needs improvement. "
            "Follow the improvement methods shown below."
        )

    elif overall_level == "Average Performance":

        st.warning(
            "🟠 You are at an average level. "
            "Consistent improvement can raise your score."
        )

    else:

        st.info(
            "🟡 You are above average. "
            "Small incremental improvements can help you reach good performance."
        )

    st.divider()

    # ========================================================
    # SUBJECT-WISE
    # ========================================================

    st.header(
        "📚 Subject-wise Performance"
    )

    for _, row in student_df.iterrows():

        level = str(
            row["Performance"]
        )

        info = PERFORMANCE_INFO[
            level
        ]

        with st.expander(
            f"{info['circle']} "
            f"{row['Subject']} — {level}"
        ):

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Overall",
                f'{float(row["Overall_Percentage"]):.2f}%'
            )

            c2.metric(
                "Attendance",
                f'{float(row["Attendance"]):.0f}%'
            )

            c3.metric(
                "Attendance Mark",
                f'{attendance_to_marks(row["Attendance"])} / 5'
            )

            st.markdown(
                f"""
**Internal Mark:** {float(row["Internal_Mark"]):.0f} / 40

**Previous Mark:** {float(row["Previous_Mark"]):.0f} / 60

**Assignment Score:** {float(row["Assignment"]):.0f} / 15

**Study Hours:** {float(row["Study_Hours"]):.1f} / 6 hours per day

**K-Means Cluster:** {row["Cluster"]}

**Random Forest Prediction:** {row["RF_Prediction"]}

**Random Forest Confidence:** {
    str(row["RF_Confidence"]) + "%"
    if str(row["RF_Confidence"]) != ""
    else "Not available"
}
                """
            )

            st.markdown(
                f"""
### {info['circle']} Performance Feedback

{info['message']}

**How to improve:**

{info['advice']}
                """
            )

    st.divider()

    # ========================================================
    # REPORT
    # ========================================================

    st.header(
        "📄 My Progress Report"
    )

    st.write(
        "Your report contains overall marks, "
        "subject-wise marks, performance indicators, "
        "K-Means information and improvement suggestions."
    )

    pdf = generate_progress_report(
        student_df
    )

    st.download_button(
        "📥 Download My Progress Report",
        data=pdf,
        file_name=(
            f"{university_id}_Progress_Report.pdf"
        ),
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

    elif st.session_state.login_mode == "student":

        student_login()

    else:

        show_login_page()

else:

    if st.session_state.role == "tutor":

        tutor_dashboard()

    elif st.session_state.role == "student":

        student_dashboard()
