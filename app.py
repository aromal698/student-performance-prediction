import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
REPORT_FILE = os.path.join(DATA_DIR, "student_reports.csv")

SEMESTERS = [f"S{i}" for i in range(1, 9)]

DEPARTMENTS = [
    "Computer Science and Engineering",
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning",
    "Information Technology",
    "Computer Science and Engineering (AI)",
    "Computer Science and Engineering (Data Science)",
    "Computer Science and Engineering (Cyber Security)",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Electronics and Instrumentation Engineering",
    "Civil Engineering",
    "Mechanical Engineering",
    "Automobile Engineering",
    "Aeronautical Engineering",
    "Biomedical Engineering",
    "Chemical Engineering",
    "Food Technology",
    "Biotechnology",
    "Mechatronics Engineering",
    "Robotics and Automation",
]

# -----------------------------------------------------------------
# Department + semester subject catalogue.
# Keep this dictionary as the SINGLE place to update subjects when
# your college/KTU scheme changes. The app never mixes departments.
# -----------------------------------------------------------------
SUBJECTS_BY_DEPARTMENT = {
    "Computer Science and Engineering": {
        "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics"],
        "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Python Programming", "Basic Electronics", "Professional Communication"],
        "S3": ["Mathematics III", "Data Structures", "Database Management Systems", "Object Oriented Programming", "Computer Organization", "Digital Electronics"],
        "S4": ["Mathematics IV", "Operating Systems", "Computer Networks", "Design and Analysis of Algorithms", "Theory of Computation", "Web Programming"],
        "S5": ["Compiler Design", "Distributed Computing", "Data Mining", "Computer Graphics", "Software Engineering", "Elective I"],
        "S6": ["Cloud Computing", "Information Security", "Data Analytics", "Machine Learning", "Mobile Computing", "Elective II"],
        "S7": ["Artificial Intelligence", "Deep Learning", "Blockchain", "Advanced Networks", "Professional Elective", "Project Phase I"],
        "S8": ["Major Project", "Project Viva", "Seminar", "Professional Elective", "Open Elective", "Comprehensive Viva"],
    },
    "Artificial Intelligence and Data Science": {
        "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics"],
        "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Python Programming", "Basic Electronics", "Professional Communication"],
        "S3": ["Mathematics III", "Data Structures", "Database Management Systems", "Artificial Intelligence", "Probability and Statistics", "Object Oriented Programming"],
        "S4": ["Operating Systems", "Computer Networks", "Machine Learning", "Design and Analysis of Algorithms", "Data Visualization", "Theory of Computation"],
        "S5": ["Deep Learning", "Data Mining", "Big Data Analytics", "Computer Vision", "Natural Language Processing", "Elective I"],
        "S6": ["Cloud Computing", "Data Engineering", "Information Retrieval", "Reinforcement Learning", "Data Analytics Lab", "Elective II"],
        "S7": ["Advanced Machine Learning", "Generative AI", "MLOps", "AI Ethics", "Professional Elective", "Project Phase I"],
        "S8": ["Major Project", "Project Viva", "Seminar", "Professional Elective", "Open Elective", "Comprehensive Viva"],
    },
    "Artificial Intelligence and Machine Learning": {
        "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics"],
        "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Python Programming", "Basic Electronics", "Professional Communication"],
        "S3": ["Mathematics III", "Data Structures", "Database Management Systems", "Artificial Intelligence", "Probability and Statistics", "Object Oriented Programming"],
        "S4": ["Operating Systems", "Computer Networks", "Machine Learning", "Design and Analysis of Algorithms", "Computer Organization", "Theory of Computation"],
        "S5": ["Deep Learning", "Computer Vision", "Natural Language Processing", "Reinforcement Learning", "Data Mining", "Elective I"],
        "S6": ["Generative AI", "MLOps", "Big Data Analytics", "Cloud Computing", "AI Applications", "Elective II"],
        "S7": ["Advanced Machine Learning", "AI Security", "Explainable AI", "AI Ethics", "Professional Elective", "Project Phase I"],
        "S8": ["Major Project", "Project Viva", "Seminar", "Professional Elective", "Open Elective", "Comprehensive Viva"],
    },
    "Information Technology": {
        "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics"],
        "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Python Programming", "Basic Electronics", "Professional Communication"],
        "S3": ["Mathematics III", "Data Structures", "Database Management Systems", "Object Oriented Programming", "Computer Organization", "Digital Electronics"],
        "S4": ["Operating Systems", "Computer Networks", "Design and Analysis of Algorithms", "Theory of Computation", "Web Programming", "Software Engineering"],
        "S5": ["Compiler Design", "Data Mining", "Cloud Computing", "Cyber Security", "Software Testing", "Elective I"],
        "S6": ["Big Data Analytics", "Mobile Computing", "Information Security", "Data Analytics", "Internet of Things", "Elective II"],
        "S7": ["Advanced Web Technologies", "Distributed Systems", "Cyber Security", "Professional Elective", "Open Elective", "Project Phase I"],
        "S8": ["Major Project", "Project Viva", "Seminar", "Professional Elective", "Open Elective", "Comprehensive Viva"],
    },
    "Electronics and Communication Engineering": {
        "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics"],
        "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Basic Electrical Engineering", "Basic Electronics", "Professional Communication"],
        "S3": ["Mathematics III", "Electronic Devices", "Digital Electronics", "Network Theory", "Signals and Systems", "Object Oriented Programming"],
        "S4": ["Mathematics IV", "Analog Circuits", "Microprocessors", "Communication Engineering", "Electromagnetic Theory", "Control Systems"],
        "S5": ["Digital Signal Processing", "VLSI Design", "Microwave Engineering", "Antenna Theory", "Embedded Systems", "Elective I"],
        "S6": ["Optical Communication", "Wireless Communication", "Computer Networks", "Digital Image Processing", "Embedded Systems Lab", "Elective II"],
        "S7": ["Advanced Communication", "RF Engineering", "IoT Systems", "Professional Elective", "Open Elective", "Project Phase I"],
        "S8": ["Major Project", "Project Viva", "Seminar", "Professional Elective", "Open Elective", "Comprehensive Viva"],
    },
    "Electrical and Electronics Engineering": {
        "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics"],
        "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Basic Electrical Engineering", "Basic Electronics", "Professional Communication"],
        "S3": ["Mathematics III", "Electrical Machines I", "Network Theory", "Analog Electronics", "Digital Electronics", "Measurements and Instrumentation"],
        "S4": ["Mathematics IV", "Electrical Machines II", "Power Systems I", "Power Electronics", "Signals and Systems", "Microprocessors"],
        "S5": ["Power Systems II", "Control Systems", "Electrical Drives", "Digital Signal Processing", "Embedded Systems", "Elective I"],
        "S6": ["Power System Protection", "High Voltage Engineering", "Renewable Energy", "Industrial Instrumentation", "Power Electronics Lab", "Elective II"],
        "S7": ["Smart Grid", "Power Quality", "Advanced Control", "Professional Elective", "Open Elective", "Project Phase I"],
        "S8": ["Major Project", "Project Viva", "Seminar", "Professional Elective", "Open Elective", "Comprehensive Viva"],
    },
    "Civil Engineering": {
        "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics"],
        "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Basic Electrical Engineering", "Engineering Mechanics", "Professional Communication"],
        "S3": ["Mathematics III", "Engineering Geology", "Fluid Mechanics", "Strength of Materials", "Surveying", "Construction Materials"],
        "S4": ["Mathematics IV", "Structural Analysis I", "Geotechnical Engineering I", "Hydraulics", "Transportation Engineering I", "Environmental Engineering I"],
        "S5": ["Structural Analysis II", "Geotechnical Engineering II", "Transportation Engineering II", "Water Resources Engineering", "Concrete Technology", "Elective I"],
        "S6": ["Design of Concrete Structures", "Steel Structures", "Environmental Engineering II", "Construction Management", "Hydrology", "Elective II"],
        "S7": ["Advanced Structural Design", "Pavement Engineering", "Remote Sensing and GIS", "Professional Elective", "Open Elective", "Project Phase I"],
        "S8": ["Major Project", "Project Viva", "Seminar", "Professional Elective", "Open Elective", "Comprehensive Viva"],
    },
    "Mechanical Engineering": {
        "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics"],
        "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Basic Electrical Engineering", "Engineering Mechanics", "Professional Communication"],
        "S3": ["Mathematics III", "Thermodynamics", "Fluid Mechanics", "Manufacturing Processes", "Mechanics of Materials", "Material Science"],
        "S4": ["Mathematics IV", "Heat Transfer", "Theory of Machines", "Machine Design I", "Manufacturing Technology", "Metrology"],
        "S5": ["Machine Design II", "Dynamics of Machinery", "Internal Combustion Engines", "Industrial Engineering", "Refrigeration and Air Conditioning", "Elective I"],
        "S6": ["Automobile Engineering", "Finite Element Analysis", "CAD/CAM", "Power Plant Engineering", "Mechatronics", "Elective II"],
        "S7": ["Advanced Manufacturing", "Robotics", "Computational Methods", "Professional Elective", "Open Elective", "Project Phase I"],
        "S8": ["Major Project", "Project Viva", "Seminar", "Professional Elective", "Open Elective", "Comprehensive Viva"],
    },
}

# For departments not explicitly listed, use a clearly marked configurable set.
DEFAULT_SUBJECTS = {
    sem: [f"{sem} Subject {i}" for i in range(1, 7)] for sem in SEMESTERS
}

TEACHERS = {
    "teacher_cse": {"password": "ktutech", "department": "Computer Science and Engineering"},
    "teacher_aids": {"password": "ktutech", "department": "Artificial Intelligence and Data Science"},
    "teacher_aiml": {"password": "ktutech", "department": "Artificial Intelligence and Machine Learning"},
    "teacher_it": {"password": "ktutech", "department": "Information Technology"},
    "teacher_ece": {"password": "ktutech", "department": "Electronics and Communication Engineering"},
    "teacher_eee": {"password": "ktutech", "department": "Electrical and Electronics Engineering"},
    "teacher_ce": {"password": "ktutech", "department": "Civil Engineering"},
    "teacher_me": {"password": "ktutech", "department": "Mechanical Engineering"},
}

REPORT_COLUMNS = [
    "Username", "Student_Name", "University_ID", "Semester", "Department",
    "Subjects_JSON", "Created_Time"
]

# ============================================================
# SESSION STATE
# ============================================================
DEFAULT_STATE = {
    "page": "home", "logged_in": False, "role": None, "username": None,
    "teacher_department": None, "student_report": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# UI
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: Inter, sans-serif; }

    /* FULL-PAGE DYNAMIC 3D BACKGROUND */
    .stApp {
        min-height:100vh;
        color:#eef2ff;
        background:#030712;
        overflow-x:hidden;
    }
    .stApp::before {
        content:""; position:fixed; inset:-20%; z-index:-5; pointer-events:none;
        background:
          radial-gradient(circle at 20% 25%, rgba(34,211,238,.22), transparent 22%),
          radial-gradient(circle at 80% 30%, rgba(139,92,246,.24), transparent 24%),
          radial-gradient(circle at 55% 85%, rgba(59,130,246,.18), transparent 25%),
          linear-gradient(120deg,#020617,#0b1024 45%,#020617);
        animation:bgshift 18s ease-in-out infinite alternate;
        transform:translateZ(0);
    }
    .stApp::after {
        content:""; position:fixed; inset:0; z-index:-4; pointer-events:none; opacity:.22;
        background-image:
          linear-gradient(rgba(125,211,252,.18) 1px, transparent 1px),
          linear-gradient(90deg, rgba(125,211,252,.18) 1px, transparent 1px);
        background-size:55px 55px;
        transform:perspective(650px) rotateX(58deg) scale(1.8);
        transform-origin:center bottom;
        animation:grid3d 9s linear infinite;
    }
    @keyframes bgshift {
        0% { transform:scale(1) rotate(0deg); filter:hue-rotate(0deg); }
        50% { transform:scale(1.08) rotate(1deg); filter:hue-rotate(18deg); }
        100% { transform:scale(1.02) rotate(-1deg); filter:hue-rotate(-12deg); }
    }
    @keyframes grid3d {
        from { background-position:0 0,0 0; }
        to { background-position:0 110px,110px 0; }
    }
    .block-container { max-width:1250px; padding-top:2rem; position:relative; z-index:2; }
    .hero,.login-wrap { position:relative; overflow:hidden; border:1px solid rgba(255,255,255,.14); border-radius:28px; background:linear-gradient(135deg,rgba(5,12,30,.78),rgba(15,23,42,.58)); box-shadow:0 25px 100px rgba(0,0,0,.45); backdrop-filter:blur(14px); }
    .hero { padding:45px 40px; }
    .hero h1 { font-size:clamp(2rem,5vw,4rem); margin:0; font-weight:800; letter-spacing:-2px; }
    .hero p { color:#b7c2d9; font-size:1.05rem; max-width:760px; }
    .login-wrap { min-height:430px; display:flex; align-items:center; justify-content:center; }
    .grid3d { position:absolute; inset:0; opacity:.18; background-image:linear-gradient(rgba(255,255,255,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.12) 1px,transparent 1px); background-size:42px 42px; transform:perspective(500px) rotateX(62deg) scale(1.7); transform-origin:center bottom; animation:gridmove 8s linear infinite; }
    @keyframes gridmove { from{background-position:0 0,0 0} to{background-position:0 84px,84px 0} }
    .orb { position:absolute; width:180px;height:180px;border-radius:50%;opacity:.45;background:radial-gradient(circle at 30% 30%,#67e8f9,#2563eb 55%,transparent 70%);animation:float 7s ease-in-out infinite; filter:blur(.2px); }
    .orb.o1 { right:5%;top:-45px; } .orb.o2 { left:45%;bottom:-110px;width:230px;height:230px;background:radial-gradient(circle at 30% 30%,#c084fc,#7c3aed 55%,transparent 70%);animation-delay:1.5s; }
    @keyframes float { 0%,100%{transform:translate3d(0,0,0) rotate(0deg)} 50%{transform:translate3d(22px,28px,35px) rotate(8deg)} }
    .glass { background:rgba(15,23,42,.68);border:1px solid rgba(255,255,255,.14);border-radius:22px;padding:24px;backdrop-filter:blur(18px); }
    .metric-card { padding:20px;border-radius:20px;border:1px solid rgba(255,255,255,.10);background:linear-gradient(135deg,rgba(30,41,59,.75),rgba(15,23,42,.7)); }

    /* DYNAMIC 3D OBJECT FIELD */
    .stApp .three-d-field { position:fixed; inset:0; z-index:-3; pointer-events:none; overflow:hidden; perspective:900px; }
    .three-d-object { position:absolute; transform-style:preserve-3d; opacity:.55; filter:drop-shadow(0 0 18px rgba(96,165,250,.28)); }
    .cube { width:90px;height:90px; left:8%;top:20%; animation:cubeFloat 12s ease-in-out infinite; }
    .cube::before,.cube::after { content:""; position:absolute; inset:0; border:2px solid rgba(125,211,252,.45); background:rgba(59,130,246,.035); }
    .cube::before { transform:rotateY(60deg); } .cube::after { transform:rotateX(60deg); }
    .ring3d { width:150px;height:150px; right:7%;top:18%; border:3px solid rgba(196,181,253,.45); border-radius:50%; animation:ringFloat 10s linear infinite; box-shadow:0 0 30px rgba(139,92,246,.22); }
    .ring3d::after { content:""; position:absolute; inset:18px; border:1px dashed rgba(125,211,252,.35); border-radius:50%; transform:rotateX(65deg); }
    .sphere3d { width:115px;height:115px; left:45%;bottom:8%; border-radius:50%; background:radial-gradient(circle at 30% 25%,rgba(255,255,255,.65),rgba(59,130,246,.35) 16%,rgba(30,64,175,.12) 58%,transparent 70%); animation:sphereFloat 9s ease-in-out infinite; box-shadow:0 0 45px rgba(59,130,246,.18); }
    .diamond3d { width:70px;height:70px; right:28%;bottom:24%; border:2px solid rgba(74,222,128,.32); transform:rotate(45deg); animation:diamondFloat 11s ease-in-out infinite; }
    @keyframes cubeFloat { 0%,100%{transform:translate3d(0,0,0) rotateX(0) rotateY(0)} 50%{transform:translate3d(80px,-50px,120px) rotateX(180deg) rotateY(220deg)} }
    @keyframes ringFloat { 0%{transform:rotateX(65deg) rotateY(0) translate3d(0,0,0)} 50%{transform:rotateX(110deg) rotateY(180deg) translate3d(-40px,55px,100px)} 100%{transform:rotateX(65deg) rotateY(360deg) translate3d(0,0,0)} }
    @keyframes sphereFloat { 0%,100%{transform:translate3d(0,0,0) scale(1)} 50%{transform:translate3d(-70px,-80px,140px) scale(1.15)} }
    @keyframes diamondFloat { 0%,100%{transform:translate3d(0,0,0) rotate(45deg) rotateX(0)} 50%{transform:translate3d(60px,-70px,90px) rotate(225deg) rotateX(180deg)} }

    .brand-bar { display:flex; align-items:center; gap:14px; margin:0 0 22px; padding:10px 14px; width:max-content; border:1px solid rgba(125,211,252,.20); border-radius:18px; background:rgba(2,6,23,.48); backdrop-filter:blur(12px); box-shadow:0 10px 35px rgba(0,0,0,.22); }
    .brand-mark { width:58px;height:58px; flex:0 0 58px; border-radius:17px; display:grid; place-items:center; background:linear-gradient(135deg,#22d3ee,#6366f1 55%,#a855f7); box-shadow:0 0 28px rgba(99,102,241,.42); }
    .brand-name { font-weight:900; letter-spacing:.4px; font-size:1.05rem; color:#f8fafc; }
    .brand-tag { color:#94a3b8; font-size:.72rem; margin-top:2px; }
    .subject-card { border:1px solid rgba(255,255,255,.10); border-radius:18px; padding:15px; background:rgba(15,23,42,.58); margin:8px 0; }
    .status-dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:7px; box-shadow:0 0 10px currentColor; }

    .metric-card .value{font-size:2rem;font-weight:800}.metric-card .label{color:#94a3b8}
    .good-pop,.sad-pop { padding:25px;border-radius:24px;text-align:center;animation:pop .55s ease-out; }
    .good-pop { background:linear-gradient(135deg,rgba(34,197,94,.20),rgba(16,185,129,.08));border:1px solid rgba(74,222,128,.45); }
    .sad-pop { background:linear-gradient(135deg,rgba(239,68,68,.18),rgba(127,29,29,.08));border:1px solid rgba(248,113,113,.42); }
    @keyframes pop { from{transform:scale(.82);opacity:0} to{transform:scale(1);opacity:1} }
    @media (prefers-reduced-motion: reduce) { .stApp::before,.stApp::after,.grid3d,.orb { animation:none !important; } }
    </style>
    """, unsafe_allow_html=True)

def app_brand():
    st.markdown("""
    <div class="brand-bar">
      <div class="brand-mark">
        <svg width="42" height="42" viewBox="0 0 64 64" aria-label="EduPredict SPP logo">
          <path d="M8 24 32 10l24 14-24 14L8 24Z" fill="none" stroke="white" stroke-width="3"/>
          <path d="M16 29v14c8 8 24 8 32 0V29M32 38v14" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"/>
          <circle cx="48" cy="18" r="5" fill="white" opacity=".9"/>
        </svg>
      </div>
      <div><div class="brand-name">EduPredict SPP</div><div class="brand-tag">Student Performance Prediction • KTU B.Tech</div></div>
    </div>
    <div class="three-d-field" aria-hidden="true">
      <div class="three-d-object cube"></div><div class="three-d-object ring3d"></div>
      <div class="three-d-object sphere3d"></div><div class="three-d-object diamond3d"></div>
    </div>
    """, unsafe_allow_html=True)

inject_css()

# ============================================================
# DATA HELPERS
# ============================================================
def empty_df():
    return pd.DataFrame(columns=REPORT_COLUMNS)


def load_reports():
    if not os.path.exists(REPORT_FILE):
        return empty_df()
    try:
        df = pd.read_csv(REPORT_FILE)
        for col in REPORT_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[REPORT_COLUMNS]
    except Exception:
        return empty_df()


def save_reports(df):
    df.to_csv(REPORT_FILE, index=False)


def upsert_report(report):
    df = load_reports()
    if df.empty:
        df = pd.DataFrame([report], columns=REPORT_COLUMNS)
    else:
        mask = (
            df["University_ID"].astype(str).eq(str(report["University_ID"])) &
            df["Department"].astype(str).eq(str(report["Department"]))
        )
        df = df.loc[~mask].copy()
        df = pd.concat([df, pd.DataFrame([report])], ignore_index=True)
    save_reports(df)


def delete_student(uid, department):
    df = load_reports()
    if df.empty:
        return
    mask = df["University_ID"].astype(str).eq(str(uid)) & df["Department"].astype(str).eq(str(department))
    save_reports(df.loc[~mask].copy())


def get_subjects(department, semester):
    return SUBJECTS_BY_DEPARTMENT.get(department, {}).get(semester, DEFAULT_SUBJECTS[semester])[:6]


def parse_subjects(raw):
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []

# ============================================================
# PERFORMANCE CALCULATION
# ============================================================
def attendance_mark(attendance):
    """Convert attendance percentage to the requested 5-point mark."""
    a = float(attendance)
    if 90 <= a <= 100:
        return 5
    if 80 <= a < 90:
        return 4
    if 70 <= a < 80:
        return 3
    if 60 <= a < 70:
        return 2
    if 10 <= a < 60:
        return 1
    return 0


def performance_circle(level):
    return {
        "Needs Improvement": "🔴",
        "Average": "🟠",
        "Above Average": "🟡",
        "Good": "🟢",
    }.get(level, "⚪")


def normalize_subject(data):
    attendance = max(0.0, min(float(data.get("Attendance", 0)), 100.0))
    study = max(0.0, min(float(data.get("Study_Hours", 0)), 6.0))
    internal = max(0.0, min(float(data.get("Internal", 0)), 40.0))
    assignment = max(0.0, min(float(data.get("Assignment", 0)), 15.0))
    previous = max(0.0, min(float(data.get("Previous", 0)), 60.0))

    # Requested attendance conversion: percentage -> 5-point mark.
    att_mark = attendance_mark(attendance)
    attendance_score = att_mark / 5 * 100
    study_score = min(study / 6 * 100, 100)
    internal_score = internal / 40 * 100
    assignment_score = assignment / 15 * 100
    previous_score = previous / 60 * 100
    overall = float(np.mean([attendance_score, study_score, internal_score, assignment_score, previous_score]))

    if overall >= 80:
        level = "Good"
    elif overall >= 65:
        level = "Above Average"
    elif overall >= 50:
        level = "Average"
    else:
        level = "Needs Improvement"

    recommendations = []
    complements = []
    if level == "Good":
        complements.append("Excellent work! Keep this consistency and continue practising regularly.")
    elif level == "Above Average":
        complements.append("Good progress. A small increase in internal marks and regular practice can move this subject higher.")
    elif level == "Average":
        complements.append("You are progressing. Focus on consistent study and improve internal/assignment marks step by step.")
    else:
        complements.append("Do not worry—this can improve with a simple, consistent study plan and regular class participation.")

    if attendance < 90:
        recommendations.append("Improve attendance; regular class participation can raise the attendance mark.")
    if study < 2:
        recommendations.append("Increase daily study time gradually to at least 2 hours for this subject.")
    if internal < 20:
        recommendations.append("Revise internal-exam topics and practise previous questions to improve internal marks.")
    if assignment < 8:
        recommendations.append("Complete assignments on time and use tutor feedback to improve marks.")
    if previous < 30:
        recommendations.append("Revise previous topics and practise short questions before moving to new chapters.")
    if not recommendations:
        recommendations.append("Maintain the current routine and continue regular revision and practice.")

    return {
        "Subject": data["Subject"],
        "Attendance": attendance,
        "Attendance_Mark": att_mark,
        "Study_Hours": study,
        "Internal": internal,
        "Assignment": assignment,
        "Previous": previous,
        "Overall": overall,
        "Level": level,
        "Circle": performance_circle(level),
        "Recommendations": recommendations,
        "Compliment": complements[0],
    }


def make_report(username, name, uid, semester, department, subjects):
    return {
        "Username": username.strip(),
        "Student_Name": name.strip(),
        "University_ID": uid.strip(),
        "Semester": semester,
        "Department": department,
        "Subjects_JSON": json.dumps(subjects),
        "Created_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def find_student(username, uid):
    df = load_reports()
    if df.empty:
        return None
    found = df[
        df["Username"].astype(str).str.lower().eq(username.strip().lower()) &
        df["University_ID"].astype(str).str.lower().eq(uid.strip().lower())
    ]
    if found.empty:
        return None
    return found.iloc[0].to_dict()

# ============================================================
# PDF
# ============================================================
def create_pdf(report):
    """Detailed student progress report with overall and subject-wise information."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, spaceAfter=12)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)
    story = [Paragraph("Student Performance Progress Report", title)]
    story += [
        Paragraph(f"<b>Name:</b> {report['Student_Name']}", styles["Normal"]),
        Paragraph(f"<b>Username:</b> {report['Username']}", styles["Normal"]),
        Paragraph(f"<b>University ID:</b> {report['University_ID']}", styles["Normal"]),
        Paragraph(f"<b>Semester:</b> {report['Semester']}", styles["Normal"]),
        Paragraph(f"<b>Department:</b> {report['Department']}", styles["Normal"]),
        Spacer(1, 10),
    ]

    subjects = parse_subjects(report["Subjects_JSON"])
    overall = float(np.mean([x["Overall"] for x in subjects])) if subjects else 0
    overall_level = "Good" if overall >= 80 else "Above Average" if overall >= 65 else "Average" if overall >= 50 else "Needs Improvement"
    story.append(Paragraph(f"<b>Overall Performance:</b> {overall:.2f}% &nbsp;&nbsp; <b>Status:</b> {performance_circle(overall_level)} {overall_level}", styles["Heading2"]))
    story.append(Paragraph(f"<b>Subjects:</b> {len(subjects)} &nbsp;&nbsp; <b>Average Attendance:</b> {np.mean([x['Attendance'] for x in subjects]):.1f}% &nbsp;&nbsp; <b>Average Internal:</b> {np.mean([x['Internal'] for x in subjects]):.1f}/40", small))
    story.append(Spacer(1, 10))

    summary = [["Subject", "Attendance", "Att. Mark", "Internal", "Assignment", "Previous", "Study", "Score", "Status"]]
    for x in subjects:
        dot_color = {
            "Needs Improvement": "#dc2626", "Average": "#f97316",
            "Above Average": "#eab308", "Good": "#16a34a"
        }.get(x["Level"], "#64748b")
        status = Paragraph(f'<font color="{dot_color}">●</font> {x["Level"]}', small)
        summary.append([
            x["Subject"], f"{x['Attendance']:.0f}%", f"{x.get('Attendance_Mark', attendance_mark(x['Attendance']))}/5",
            f"{x['Internal']:.0f}/40", f"{x['Assignment']:.0f}/15", f"{x['Previous']:.0f}/60",
            f"{x['Study_Hours']:.1f} h", f"{x['Overall']:.1f}%", status
        ])
    table = Table(summary, repeatRows=1, colWidths=[92, 48, 45, 48, 52, 50, 42, 45, 70])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(Paragraph("Overall Information — All Subjects", styles["Heading2"]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Subject-wise Detailed Information", styles["Heading2"]))
    for i, x in enumerate(subjects, 1):
        dot_color = {"Needs Improvement": "#dc2626", "Average": "#f97316", "Above Average": "#eab308", "Good": "#16a34a"}.get(x["Level"], "#64748b")
        story.append(Paragraph(f'{i}. {x["Subject"]} — <font color="{dot_color}">●</font> {x["Level"]} ({x["Overall"]:.1f}%)', styles["Heading3"]))
        detail = [
            ["Attendance", f"{x['Attendance']:.0f}%", "Attendance Mark", f"{x.get('Attendance_Mark', attendance_mark(x['Attendance']))}/5"],
            ["Internal", f"{x['Internal']:.0f}/40", "Assignment", f"{x['Assignment']:.0f}/15"],
            ["Previous Mark", f"{x['Previous']:.0f}/60", "Study Hours", f"{x['Study_Hours']:.1f} h/day"],
            ["Overall Score", f"{x['Overall']:.1f}%", "Performance", f"{x['Level']}"]
        ]
        dt = Table(detail, colWidths=[95, 75, 95, 75])
        dt.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef2ff")),
            ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#eef2ff")),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(dt)
        story.append(Spacer(1, 5))
        story.append(Paragraph(f"<b>Complement:</b> {x.get('Compliment', '')}", small))
        story.append(Paragraph("<b>How to improve:</b> " + " ".join(x.get("Recommendations", [])), small))
        story.append(Spacer(1, 9))

    story.append(Paragraph("Performance indicators are project-defined indicators and are not official university grades.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def create_kmeans_pdf(cluster_result):
    """Create a separate tutor analysis PDF containing K-Means calculations."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("k_title", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, spaceAfter=12)
    story = [Paragraph("Tutor Dashboard — K-Means Clustering Analysis", title)]
    if not cluster_result.get("available"):
        story.append(Paragraph(cluster_result.get("message", "K-Means analysis unavailable."), styles["Normal"]))
    else:
        story.append(Paragraph(f"<b>Students/subject records analysed:</b> {cluster_result['n_samples']}", styles["Normal"]))
        story.append(Paragraph(f"<b>Number of clusters (K):</b> {cluster_result['k']}", styles["Normal"]))
        story.append(Paragraph("Features used: Attendance Mark, Internal, Assignment, Previous Mark and Study Hours.", styles["Normal"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Cluster Summary", styles["Heading2"]))
        rows = [["Cluster", "Records", "Avg Score", "Interpretation"]]
        for c in cluster_result["summary"]:
            rows.append([str(c["cluster"]), str(c["records"]), f"{c['avg_score']:.2f}%", c["interpretation"]])
        t = Table(rows, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#172554")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))
        story.append(t)
        story.append(Spacer(1, 12))
        story.append(Paragraph("K-Means Calculation / Centroids", styles["Heading2"]))
        cent = [["Cluster", "Attendance Mark", "Internal", "Assignment", "Previous", "Study Hours"]]
        for row in cluster_result["centroids"]:
            cent.append([row["cluster"], f"{row['attendance_mark']:.2f}", f"{row['internal']:.2f}", f"{row['assignment']:.2f}", f"{row['previous']:.2f}", f"{row['study_hours']:.2f}"])
        ct = Table(cent, repeatRows=1)
        ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#334155")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(ct)
        story.append(Spacer(1, 12))
        story.append(Paragraph("The clustering is an analytical grouping for the tutor dashboard. It does not replace the subject performance indicator or official university grading.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# CSV TEMPLATE / IMPORT
# ============================================================
def template_df():
    cols = ["Username", "Student_Name", "University_ID", "Semester", "Department"]
    for i in range(1, 7):
        cols += [f"Subject_{i}", f"Attendance_{i}", f"Internal_{i}", f"Assignment_{i}", f"Previous_{i}"]
    return pd.DataFrame(columns=cols)


def uploaded_to_reports(uploaded_file, tutor_department):
    df = pd.read_csv(uploaded_file)
    required = ["Username", "Student_Name", "University_ID", "Semester"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    reports = []
    for _, row in df.iterrows():
        semester = str(row["Semester"]).strip().upper()
        if semester not in SEMESTERS:
            raise ValueError(f"Invalid semester '{semester}' for {row['University_ID']}")
        subjects = get_subjects(tutor_department, semester)
        values = []
        for i, default_subject in enumerate(subjects, 1):
            sub = str(row.get(f"Subject_{i}", default_subject)).strip() or default_subject
            item = normalize_subject({
                "Subject": sub,
                "Attendance": row.get(f"Attendance_{i}", 0),
                "Study_Hours": 0,
                "Internal": row.get(f"Internal_{i}", 0),
                "Assignment": row.get(f"Assignment_{i}", 0),
                "Previous": row.get(f"Previous_{i}", 0),
            })
            values.append(item)
        reports.append(make_report(row["Username"], row["Student_Name"], row["University_ID"], semester, tutor_department, values))
    return reports

# ============================================================
# LOGIN / PAGES
# ============================================================
def logout():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value
    st.rerun()


def home_page():
    app_brand()
    st.markdown("""
    <div class="hero">
      <div class="grid3d"></div><div class="orb o1"></div><div class="orb o2"></div>
      <div style="position:relative;z-index:2">
        <div style="font-size:3rem">🎓</div>
        <h1>Student Performance Prediction</h1>
        <p>Department-aware and semester-aware student performance tracking for tutors and students.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    if a.button("👨‍🏫 Tutor Login", use_container_width=True, type="primary"):
        st.session_state.page = "teacher_login"; st.rerun()
    if b.button("🎓 Student Login", use_container_width=True):
        st.session_state.page = "student_login"; st.rerun()


def login_shell(title, subtitle):
    app_brand()
    st.markdown(f"""
    <div class="login-wrap"><div class="grid3d"></div><div class="orb o1"></div><div class="orb o2"></div>
      <div style="width:min(560px,90%);position:relative;z-index:2;text-align:center">
        <div style="font-size:3rem">🎓</div><h1>{title}</h1><p style="color:#94a3b8">{subtitle}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def teacher_login():
    login_shell("Tutor Login", "Select the department through the tutor account")
    with st.form("teacher_login_form"):
        username = st.text_input("Tutor Username")
        password = st.text_input("Password", type="password")
        c1, c2 = st.columns(2)
        login = c1.form_submit_button("🔐 Login", use_container_width=True, type="primary")
        back = c2.form_submit_button("← Back", use_container_width=True)
    if back:
        st.session_state.page = "home"; st.rerun()
    if login:
        account = TEACHERS.get(username.strip())
        if account and account["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = "teacher"
            st.session_state.username = username.strip()
            st.session_state.teacher_department = account["department"]
            st.session_state.active_department = account["department"]
            st.session_state.page = "teacher_dashboard"
            st.rerun()
        else:
            st.error("Invalid tutor username or password.")


def valid_student_password(password):
    # Exactly 9 characters: BTECH + year 2000–2022.
    return bool(re.fullmatch(r"BTECH(?:200\d|201\d|202[0-2])", password or ""))


def student_login():
    login_shell("Student Login", "Username + University ID + BTECH year password")
    with st.form("student_login_form"):
        username = st.text_input("Username", placeholder="e.g. student01")
        uid = st.text_input("University ID", placeholder="e.g. UNI001")
        password = st.text_input("Password", type="password", placeholder="Example: BTECH2007", max_chars=9)
        st.caption("Password format: BTECH + year from 2000 to 2022, e.g. BTECH2007")
        c1, c2 = st.columns(2)
        login = c1.form_submit_button("🔐 Login", use_container_width=True, type="primary")
        back = c2.form_submit_button("← Back", use_container_width=True)
    if back:
        st.session_state.page = "home"; st.rerun()
    if login:
        if not valid_student_password(password):
            st.error("Invalid password format. Use BTECH followed by a year from 2000–2022.")
            return
        student = find_student(username, uid)
        if student:
            st.session_state.logged_in = True
            st.session_state.role = "student"
            st.session_state.username = username.strip()
            st.session_state.student_report = student
            st.session_state.page = "student_dashboard"
            st.rerun()
        else:
            st.error("No matching student record found. Check Username and University ID.")

# ============================================================
# K-MEANS TUTOR ANALYSIS
# ============================================================
def kmeans_analysis(department):
    """Robust K-Means tutor analysis. Handles 0, 1 and many records safely."""
    if not SKLEARN_AVAILABLE:
        return {"available": False, "message": "scikit-learn is not installed. Run: pip install scikit-learn"}

    df = load_reports()
    if df.empty:
        return {"available": False, "message": "No student records are available for K-Means analysis."}

    df = df[df["Department"].astype(str).eq(str(department))].copy()
    if df.empty:
        return {"available": False, "message": f"No records found for {department}. Add at least one student first."}

    records = []
    for _, row in df.iterrows():
        for item in parse_subjects(row.get("Subjects_JSON", "[]")):
            try:
                records.append({
                    "University_ID": str(row.get("University_ID", "")),
                    "Student_Name": str(row.get("Student_Name", "")),
                    "Subject": str(item.get("Subject", "Unknown")),
                    "Attendance_Mark": float(item.get("Attendance_Mark", attendance_mark(item.get("Attendance", 0)))),
                    "Internal": float(item.get("Internal", 0)),
                    "Assignment": float(item.get("Assignment", 0)),
                    "Previous": float(item.get("Previous", 0)),
                    "Study_Hours": float(item.get("Study_Hours", 0)),
                    "Overall": float(item.get("Overall", 0)),
                })
            except (TypeError, ValueError):
                continue

    if not records:
        return {"available": False, "message": "Student records were found, but no valid subject data could be read."}

    raw = pd.DataFrame(records)
    features = ["Attendance_Mark", "Internal", "Assignment", "Previous", "Study_Hours"]
    raw[features] = raw[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    raw["Overall"] = pd.to_numeric(raw["Overall"], errors="coerce").fillna(0.0)

    # K cannot be larger than the number of available subject records.
    k = min(4, len(raw))
    X = raw[features].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        raw["Cluster"] = model.fit_predict(X_scaled)
    except Exception as exc:
        return {"available": False, "message": f"K-Means could not be calculated: {exc}"}

    # Sort clusters by average performance only for human-readable interpretation.
    means = raw.groupby("Cluster")["Overall"].mean().sort_values()
    order = {int(cluster): rank for rank, cluster in enumerate(means.index)}
    labels = {0: "Low-performance group", 1: "Average-performance group", 2: "Above-average group", 3: "Good-performance group"}

    summary = []
    for cluster in sorted(raw["Cluster"].unique()):
        group = raw[raw["Cluster"] == cluster]
        rank = order[int(cluster)]
        summary.append({
            "cluster": int(cluster) + 1,
            "records": int(len(group)),
            "avg_score": float(group["Overall"].mean()),
            "interpretation": labels.get(rank, "Performance group"),
        })

    original_centroids = scaler.inverse_transform(model.cluster_centers_)
    centroids = []
    for cluster in range(k):
        rank = order.get(cluster, 0)
        centroids.append({
            "cluster": f"{cluster + 1} ({labels.get(rank, 'Performance group')})",
            "attendance_mark": float(original_centroids[cluster, 0]),
            "internal": float(original_centroids[cluster, 1]),
            "assignment": float(original_centroids[cluster, 2]),
            "previous": float(original_centroids[cluster, 3]),
            "study_hours": float(original_centroids[cluster, 4]),
        })

    raw["Cluster"] = raw["Cluster"].astype(int) + 1
    display_columns = ["University_ID", "Student_Name", "Subject", "Attendance_Mark", "Internal", "Assignment", "Previous", "Study_Hours", "Overall", "Cluster"]
    return {
        "available": True,
        "n_samples": len(raw),
        "k": k,
        "summary": summary,
        "centroids": centroids,
        "records": raw[display_columns].copy(),
        "features": features,
        "inertia": float(model.inertia_),
    }

# ============================================================
# TUTOR WORKFLOW
# ============================================================
def manual_add_form():
    st.subheader("➕ Add One Student")
    tutor_department = st.session_state.get("active_department", st.session_state.teacher_department)
    st.info(f"Active department: **{tutor_department}**")

    with st.form("add_student"):
        c1, c2, c3 = st.columns(3)
        username = c1.text_input("Username")
        name = c2.text_input("Student Name")
        uid = c3.text_input("University ID")
        dcol, scol = st.columns(2)
        department = dcol.selectbox("Department", DEPARTMENTS, index=DEPARTMENTS.index(tutor_department) if tutor_department in DEPARTMENTS else 0)
        semester = scol.selectbox("Semester", SEMESTERS, index=2)

        subjects = get_subjects(department, semester)
        st.success(f"Subjects automatically loaded for **{semester} – {department}**")
        st.dataframe(pd.DataFrame({"No.": range(1, 7), "Subject": subjects}), use_container_width=True, hide_index=True)
        st.caption("Changing Semester or Department automatically changes the six subjects used for this student.")

        st.markdown("### Enter marks for the 6 subjects")
        values = []
        for i, subject in enumerate(subjects):
            st.markdown(f"**{i+1}. {subject}**")
            a, b, c, d = st.columns(4)
            att = a.number_input("Attendance %", 0.0, 100.0, 75.0, 1.0, key=f"att_{i}")
            internal = b.number_input("Internal /40", 0.0, 40.0, 20.0, 1.0, key=f"int_{i}")
            assignment = c.number_input("Assignment /15", 0.0, 15.0, 8.0, 1.0, key=f"asg_{i}")
            previous = d.number_input("Previous /60", 0.0, 60.0, 30.0, 1.0, key=f"prev_{i}")
            values.append(normalize_subject({
                "Subject": subject,
                "Attendance": att,
                "Study_Hours": 0,
                "Internal": internal,
                "Assignment": assignment,
                "Previous": previous,
            }))

        submitted = st.form_submit_button("💾 Submit Student", use_container_width=True, type="primary")

    if submitted:
        if not username.strip() or not name.strip() or not uid.strip():
            st.error("Username, Student Name and University ID are required.")
            return
        report = make_report(username, name, uid, semester, department, values)
        upsert_report(report)
        st.success("✅ Student submitted successfully. You can now enter the next student.")


def teacher_dashboard():
    app_brand()
    assigned_department = st.session_state.teacher_department
    department = st.selectbox(
        "Active Department",
        DEPARTMENTS,
        index=DEPARTMENTS.index(st.session_state.get("active_department", assigned_department)) if st.session_state.get("active_department", assigned_department) in DEPARTMENTS else 0,
        key="active_department"
    )
    c1, c2 = st.columns([5, 1])
    c1.title("👨‍🏫 Tutor Dashboard")
    c2.button("🚪 Logout", on_click=logout, use_container_width=True)
    st.info(f"Tutor account: **{assigned_department}**  •  Active department: **{department}**")

    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Student", "📤 Upload CSV", "👥 Student Records", "📈 K-Means Analysis"])

    with tab1:
        manual_add_form()

    with tab2:
        st.subheader("Upload Multiple Students")
        st.write("Semester and department determine the subject list. The Department column in the CSV is ignored and replaced with the tutor's assigned department.")
        template = template_df().to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV Template", template, "student_marks_template.csv", "text/csv")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded is not None and st.button("🚀 Submit Uploaded Students", type="primary"):
            try:
                reports = uploaded_to_reports(uploaded, department)
                for report in reports:
                    upsert_report(report)
                st.success(f"✅ {len(reports)} student record(s) uploaded successfully.")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    with tab3:
        df = load_reports()
        branch_df = df[df["Department"].astype(str).eq(str(department))].copy()
        st.subheader(f"Student Records — {len(branch_df)}")
        if branch_df.empty:
            st.info("No student records yet.")
            return

        show = branch_df[["Username", "Student_Name", "University_ID", "Semester", "Department", "Created_Time"]]
        st.dataframe(show, use_container_width=True, hide_index=True)

        ids = show["University_ID"].astype(str).tolist()
        selected = st.selectbox("Select University ID", ids)
        row = branch_df[branch_df["University_ID"].astype(str).eq(str(selected))].iloc[0].to_dict()
        subs = parse_subjects(row["Subjects_JSON"])
        overall = float(np.mean([x["Overall"] for x in subs])) if subs else 0
        st.metric("Selected Student Overall", f"{overall:.2f}%")

        rows = [[x["Subject"], x["Attendance"], x["Internal"], x["Assignment"], x["Previous"], x["Overall"], x["Level"]] for x in subs]
        st.dataframe(pd.DataFrame(rows, columns=["Subject", "Attendance", "Internal", "Assignment", "Previous", "Score", "Performance"]), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.download_button("📥 Download Student PDF", create_pdf(row), f"{selected}_Progress_Report.pdf", "application/pdf", use_container_width=True)
        if c2.button("🗑️ Delete Selected Student", use_container_width=True):
            delete_student(selected, department)
            st.success("Student deleted.")
            st.rerun()

    with tab4:
        st.subheader("📈 Tutor Performance Analysis — K-Means")
        st.write("K-Means groups the available student-subject records using attendance mark, internal, assignment, previous mark and study hours.")
        result = kmeans_analysis(department)
        if not result.get("available"):
            st.warning(result.get("message", "Analysis unavailable."))
        else:
            st.metric("Subject records analysed", result["n_samples"])
            st.metric("Clusters", result["k"])
            st.dataframe(pd.DataFrame(result["summary"]), use_container_width=True, hide_index=True)
            st.markdown("### Cluster centroids / calculations")
            st.dataframe(pd.DataFrame(result["centroids"]), use_container_width=True, hide_index=True)
            st.markdown("### Subject records with cluster")
            display = result["records"].copy()
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption(f"K-Means inertia: {result['inertia']:.4f} • Features: {', '.join(result['features'])}")
            st.download_button("📊 Download K-Means Analysis PDF", create_kmeans_pdf(result), f"{department.replace(' ', '_')}_KMeans_Analysis.pdf", "application/pdf", use_container_width=True, type="primary")

# ============================================================
# STUDENT WORKFLOW
# ============================================================
def student_dashboard():
    app_brand()
    student = st.session_state.student_report
    if not student:
        logout(); return

    subjects = parse_subjects(student["Subjects_JSON"])
    # Student enters study hours one subject at a time after login.
    st.title(f"🎓 Welcome, {student['Student_Name']}")
    st.caption(f"University ID: {student['University_ID']}  •  {student['Semester']}  •  {student['Department']}")
    if st.button("🚪 Logout"):
        logout()

    st.subheader("⏱️ Enter Daily Study Hours")
    st.write("Enter your study hours for each subject. The tutor-submitted marks remain unchanged.")

    study_hours = {}
    with st.form("study_hours_form"):
        for i, item in enumerate(subjects):
            study_hours[item["Subject"]] = st.number_input(
                f"{i+1}. {item['Subject']} — hours/day",
                min_value=0.0, max_value=6.0, value=2.0, step=0.5,
                key=f"student_study_{i}"
            )
        calculate = st.form_submit_button("📊 Calculate My Result", use_container_width=True, type="primary")

    if calculate:
        updated = []
        for item in subjects:
            copy = dict(item)
            copy["Study_Hours"] = study_hours.get(item["Subject"], 0)
            updated.append(normalize_subject(copy))
        student["Subjects_JSON"] = json.dumps(updated)
        st.session_state.student_report = student
        subjects = updated

    # Use current stored values if the user has calculated at least once.
    subjects = parse_subjects(st.session_state.student_report["Subjects_JSON"])
    overall = float(np.mean([x["Overall"] for x in subjects])) if subjects else 0

    if any(float(x.get("Study_Hours", 0)) > 0 for x in subjects):
        if overall >= 80:
            st.balloons()
            st.markdown(f'<div class="good-pop"><div style="font-size:4rem">🎉🏆</div><h2>Excellent Performance!</h2><p>Your marks are looking good. Keep the same consistency!</p><p>Overall score: <b>{overall:.2f}%</b></p></div>', unsafe_allow_html=True)
        elif overall < 50:
            st.markdown(f'<div class="sad-pop"><div style="font-size:4rem">😔📉</div><h2>Needs Improvement</h2><p>Don’t give up. Follow the subject-wise improvement methods below.</p><p>Overall score: <b>{overall:.2f}%</b></p></div>', unsafe_allow_html=True)
        else:
            st.info(f"📘 Your marks are available. Overall score: **{overall:.2f}%**. Check each subject below for its individual indicator.")

        a, b, c = st.columns(3)
        a.markdown(f'<div class="metric-card"><div class="label">Overall Score</div><div class="value">{overall:.2f}%</div></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="metric-card"><div class="label">Subjects</div><div class="value">{len(subjects)}</div></div>', unsafe_allow_html=True)
        status = "Good" if overall >= 80 else "Above Average" if overall >= 65 else "Average" if overall >= 50 else "Needs Improvement"
        c.markdown(f'<div class="metric-card"><div class="label">Status</div><div class="value">{status}</div></div>', unsafe_allow_html=True)

        rows = []
        for x in subjects:
            rows.append([x["Subject"], f"{x['Attendance']:.0f}%", f"{x.get('Attendance_Mark', attendance_mark(x['Attendance']))}/5", f"{x['Internal']:.0f}/40", f"{x['Assignment']:.0f}/15", f"{x['Previous']:.0f}/60", f"{x['Study_Hours']:.1f} h", f"{x['Overall']:.1f}%", f"{x.get('Circle', performance_circle(x['Level']))} {x['Level']}"])
        st.subheader("📊 Your Marks — Subject-wise Result")
        st.dataframe(pd.DataFrame(rows, columns=["Subject", "Attendance", "Att. Mark", "Internal", "Assignment", "Previous", "Study", "Score", "Performance"]), use_container_width=True, hide_index=True)

        st.subheader("🎯 Subject-wise Performance & Improvement")
        for x in subjects:
            circle = x.get('Circle', performance_circle(x['Level']))
            with st.expander(f"{circle} {x['Subject']} — {x['Level']} — {x['Overall']:.1f}%"):
                st.markdown(f"**Overall information:** Attendance **{x['Attendance']:.0f}% → {x.get('Attendance_Mark', attendance_mark(x['Attendance']))}/5**, Internal **{x['Internal']:.0f}/40**, Assignment **{x['Assignment']:.0f}/15**, Previous **{x['Previous']:.0f}/60**, Study **{x['Study_Hours']:.1f} h/day**.")
                st.success("💚 " + x.get("Compliment", "Keep progressing!"))
                st.markdown("**How to improve:**")
                for recommendation in x["Recommendations"]:
                    st.write("• " + recommendation)

        st.caption("🔴 Needs Improvement  •  🟠 Average  •  🟡 Above Average  •  🟢 Good")

        st.download_button(
            "📥 Download Progress Report PDF",
            create_pdf(st.session_state.student_report),
            f"{student['University_ID']}_Progress_Report.pdf",
            "application/pdf",
            use_container_width=True,
            type="primary",
        )

# ============================================================
# ROUTER
# ============================================================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "teacher_login":
    teacher_login()
elif st.session_state.page == "student_login":
    student_login()
elif st.session_state.page == "teacher_dashboard" and st.session_state.logged_in and st.session_state.role == "teacher":
    teacher_dashboard()
elif st.session_state.page == "student_dashboard" and st.session_state.logged_in and st.session_state.role == "student":
    student_dashboard()
else:
    st.session_state.page = "home"
    st.rerun()
