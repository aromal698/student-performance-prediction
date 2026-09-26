import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import re
import base64
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
REPORT_FILE = os.path.join(DATA_DIR, "student_reports.csv")
REGISTRATION_FILE = os.path.join(DATA_DIR, "student_registrations.csv")
TUTOR_PROFILE_FILE = os.path.join(DATA_DIR, "tutor_profiles.csv")
TUTOR_ACCOUNT_FILE = os.path.join(DATA_DIR, "tutor_accounts.csv")
SMART_CARD_FILE = os.path.join(DATA_DIR, "smart_card_registrations.csv")
AUDIT_FILE = os.path.join(DATA_DIR, "audit_log.csv")
STUDENT_FILES_DIR = os.path.join(DATA_DIR, "student_files")
os.makedirs(STUDENT_FILES_DIR, exist_ok=True)

SEMESTERS = [f"S{i}" for i in range(1, 9)]

DEPARTMENTS = [
    "Computer Science and Engineering",
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Civil Engineering",
    "Mechanical Engineering",
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

# ============================================================
# STUDENT CREDIT RULES
# ============================================================
# Six subjects: 3 core subjects = 4 credits each; remaining
# subjects = 3, 3 and 2 credits. Credit is awarded ONLY when
# the subject is passed. Project pass threshold: 40%.
SUBJECT_CREDITS = [4, 4, 4, 3, 3, 2]
PASS_MARK = 40.0

def subject_passed(subject):
    try:
        return float(subject.get("Overall", 0)) >= PASS_MARK
    except (TypeError, ValueError):
        return False

def calculate_total_credits(subjects):
    total = 0
    earned = []
    for i, subject in enumerate(subjects):
        credit = SUBJECT_CREDITS[i] if i < len(SUBJECT_CREDITS) else 0
        passed = subject_passed(subject)
        subject["Credit"] = int(credit if passed else 0)
        subject["Pass_Status"] = "PASS" if passed else "NOT PASS"
        if passed:
            total += credit
            earned.append(f"{subject.get('Subject','Subject')} ({credit})")
    return int(total), earned


REPORT_COLUMNS = [
    "Username", "Student_Name", "University_ID", "Semester", "Department",
    "Subjects_JSON", "Created_Time"
]
REGISTRATION_COLUMNS = ["University_ID", "Student_Name", "Department", "Semester", "Tutor_Username", "Registered_Time"]
TUTOR_PROFILE_COLUMNS = ["Tutor_Username", "Tutor_Name", "Department", "Credit_Score", "Updated_Time"]
TUTOR_ACCOUNT_COLUMNS = ["Email", "Tutor_Name", "Department", "Semester", "Tutor_ID", "Created_Time", "Last_Login"]
SMART_CARD_COLUMNS = ["Registration_ID", "University_ID", "Student_Name", "DOB", "Blood_Group", "Address", "PIN_Code", "Studied_College", "Department", "Semester", "CGPA", "University_Name", "Submitted_Time"]
AUDIT_COLUMNS = ["Timestamp", "Role", "Username", "Action", "University_ID", "Department", "Semester", "Details"]

# ============================================================
# SESSION STATE
# ============================================================
DEFAULT_STATE = {
    "page": "home", "logged_in": False, "role": None, "username": None,
    "teacher_department": None, "student_report": None,
    "dashboard_view": "department",
    "student_registration": None,
    "smart_card_registration": None,
    "smart_card_hidden": False,
    "tutor_account_email": "",
    "tutor_wallet_balance": 0.0,
    "show_tutor_create": False,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# UI
# ============================================================
def inject_css():
    minute = datetime.now().minute
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: Inter, sans-serif; }}
    .stApp {{ min-height:100vh; color:#eef2ff; background:#020b1f; overflow-x:hidden; }}
    .stApp::before {{
        content:""; position:fixed; inset:-18%; z-index:-5; pointer-events:none;
        background:
          radial-gradient(circle at 15% 20%, rgba(14,165,233,.28), transparent 22%),
          radial-gradient(circle at 85% 28%, rgba(37,99,235,.26), transparent 24%),
          radial-gradient(circle at 55% 90%, rgba(59,130,246,.20), transparent 26%),
          linear-gradient(125deg,#020817,#08204a 48%,#020817);
        animation:bgshift 16s ease-in-out infinite alternate;
    }}
    .stApp::after {{
        content:""; position:fixed; inset:0; z-index:-4; pointer-events:none; opacity:.18;
        background-image:linear-gradient(rgba(125,211,252,.16) 1px,transparent 1px),linear-gradient(90deg,rgba(125,211,252,.16) 1px,transparent 1px);
        background-size:60px 60px; transform:perspective(700px) rotateX(58deg) scale(1.7);
        transform-origin:center bottom; animation:grid3d 10s linear infinite;
    }}
    .dynamic-3d-wallpaper {{ position:fixed; inset:0; z-index:-3; pointer-events:none; overflow:hidden; perspective:1100px; }}
    .dynamic-3d-wallpaper .orb {{ position:absolute; border-radius:50%; transform-style:preserve-3d; filter:blur(.2px); mix-blend-mode:screen; }}
    .dynamic-3d-wallpaper .orb-a {{ width:240px;height:240px;left:6%;top:14%; background:radial-gradient(circle at 30% 25%,#ffffff 0 3%,#67e8f9 9%,#2563eb 38%,rgba(37,99,235,.05) 70%); box-shadow:0 0 80px rgba(34,211,238,.22); animation:orbA 12s ease-in-out infinite; }}
    .dynamic-3d-wallpaper .orb-b {{ width:320px;height:320px;right:4%;top:8%; background:radial-gradient(circle at 35% 30%,#ffffff 0 2%,#c4b5fd 8%,#7c3aed 35%,rgba(124,58,237,.04) 72%); box-shadow:0 0 100px rgba(139,92,246,.20); animation:orbB 15s ease-in-out infinite; }}
    .dynamic-3d-wallpaper .orb-c {{ width:180px;height:180px;right:25%;bottom:3%; background:radial-gradient(circle at 35% 25%,#ffffff 0 2%,#38bdf8 8%,#0ea5e9 36%,rgba(14,165,233,.03) 72%); box-shadow:0 0 70px rgba(14,165,233,.18); animation:orbC 10s ease-in-out infinite; }}
    .dynamic-3d-wallpaper .ring {{ position:absolute; width:420px;height:420px;border:1px solid rgba(125,211,252,.20);border-radius:50%; transform-style:preserve-3d; animation:ringSpin 18s linear infinite; }}
    .dynamic-3d-wallpaper .ring.one {{ left:-110px;bottom:-170px;transform:rotateX(68deg) rotateY(12deg); }}
    .dynamic-3d-wallpaper .ring.two {{ right:-130px;top:28%;width:520px;height:520px;border-color:rgba(167,139,250,.16);transform:rotateY(68deg) rotateZ(20deg);animation-duration:24s;animation-direction:reverse; }}
    .dynamic-3d-wallpaper .cube {{ position:absolute;width:90px;height:90px;right:24%;top:36%;transform-style:preserve-3d;animation:cubeSpin 14s linear infinite;opacity:.34; }}
    .dynamic-3d-wallpaper .cube i {{ position:absolute;inset:0;border:1px solid rgba(125,211,252,.55);background:rgba(59,130,246,.05);backdrop-filter:blur(2px); }}
    .dynamic-3d-wallpaper .cube i:nth-child(1){{transform:translateZ(45px)}} .dynamic-3d-wallpaper .cube i:nth-child(2){{transform:rotateY(180deg) translateZ(45px)}} .dynamic-3d-wallpaper .cube i:nth-child(3){{transform:rotateY(90deg) translateZ(45px)}} .dynamic-3d-wallpaper .cube i:nth-child(4){{transform:rotateY(-90deg) translateZ(45px)}} .dynamic-3d-wallpaper .cube i:nth-child(5){{transform:rotateX(90deg) translateZ(45px)}} .dynamic-3d-wallpaper .cube i:nth-child(6){{transform:rotateX(-90deg) translateZ(45px)}}
    @keyframes orbA {{ 0%,100%{{transform:translate3d(0,0,0) rotateY(0deg)}} 50%{{transform:translate3d(90px,40px,160px) rotateY(180deg)}} }}
    @keyframes orbB {{ 0%,100%{{transform:translate3d(0,0,0) rotateX(0deg)}} 50%{{transform:translate3d(-80px,80px,-80px) rotateX(180deg)}} }}
    @keyframes orbC {{ 0%,100%{{transform:translate3d(0,0,0)}} 50%{{transform:translate3d(-120px,-50px,120px) scale(1.15)}} }}
    @keyframes ringSpin {{ from{{transform:rotateX(68deg) rotateZ(0deg)}} to{{transform:rotateX(68deg) rotateZ(360deg)}} }}
    @keyframes cubeSpin {{ from{{transform:rotateX(0deg) rotateY(0deg) rotateZ(0deg)}} to{{transform:rotateX(360deg) rotateY(360deg) rotateZ(180deg)}} }}
    @keyframes bgshift {{ 0%{{transform:scale(1)}} 50%{{transform:scale(1.06) rotate(.5deg)}} 100%{{transform:scale(1.02) rotate(-.5deg)}} }}
    @keyframes grid3d {{ from{{background-position:0 0,0 0}} to{{background-position:0 120px,120px 0}} }}
    .block-container {{ max-width:1250px; padding-top:2rem; position:relative; z-index:2; }}
    .hero,.login-wrap {{ position:relative; overflow:hidden; border:1px solid rgba(255,255,255,.13); border-radius:28px; background:linear-gradient(135deg,rgba(5,12,30,.80),rgba(15,23,42,.62)); box-shadow:0 25px 90px rgba(0,0,0,.42); backdrop-filter:blur(14px); }}
    .hero {{ padding:45px 40px; }}
    .hero h1 {{ font-size:clamp(2rem,5vw,4rem); margin:0; font-weight:800; letter-spacing:-2px; }}
    .hero p {{ color:#b7c2d9; font-size:1.05rem; max-width:760px; }}
    .login-wrap {{ min-height:430px; display:flex; align-items:center; justify-content:center; }}
    .login-grid {{ position:absolute; inset:0; opacity:.20; background-image:linear-gradient(rgba(255,255,255,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.12) 1px,transparent 1px); background-size:44px 44px; transform:perspective(520px) rotateX(62deg) scale(1.7); transform-origin:center bottom; animation:gridmove 9s linear infinite; }}
    .login-orb {{ position:absolute; width:170px; height:170px; border-radius:50%; background:radial-gradient(circle at 30% 30%,#67e8f9,#2563eb 55%,transparent 70%); opacity:.38; animation:float 8s ease-in-out infinite; }}
    .login-orb.one {{ right:5%; top:-45px; }} .login-orb.two {{ left:8%; bottom:-100px; width:210px; height:210px; background:radial-gradient(circle at 30% 30%,#a78bfa,#6366f1 55%,transparent 70%); animation-delay:1.4s; }}
    @keyframes gridmove {{ from{{background-position:0 0,0 0}} to{{background-position:0 88px,88px 0}} }}
    @keyframes float {{ 0%,100%{{transform:translate3d(0,0,0)}} 50%{{transform:translate3d(20px,-22px,30px)}} }}
    .login-main-emoji {{ font-size:4rem; filter:drop-shadow(0 0 18px rgba(125,211,252,.35)); animation:emojiFloat 3.5s ease-in-out infinite; }}
    @keyframes emojiFloat {{ 0%,100%{{transform:translateY(0) scale(1)}} 50%{{transform:translateY(-9px) scale(1.04)}} }}
    .minute-badge {{ margin:10px auto 0; padding:7px 12px; width:max-content; border-radius:999px; background:rgba(15,23,42,.58); border:1px solid rgba(255,255,255,.10); color:#cbd5e1; font-size:.78rem; }}
    .glass {{ background:rgba(15,23,42,.68); border:1px solid rgba(255,255,255,.14); border-radius:22px; padding:24px; backdrop-filter:blur(18px); }}
    .metric-card {{ padding:20px; border-radius:20px; border:1px solid rgba(255,255,255,.10); background:linear-gradient(135deg,rgba(30,41,59,.75),rgba(15,23,42,.7)); }}
    .metric-card .value{{font-size:2rem;font-weight:800}} .metric-card .label{{color:#94a3b8}}

    /* STATIC BEAUTIFUL BLUE WALLPAPER - no moving/animated background */
    .stApp::before, .stApp::after,
    .dynamic-3d-wallpaper, .dynamic-3d-wallpaper *,
    .login-grid, .login-orb, .login-main-emoji {{
        animation: none !important;
        transition: none !important;
    }}
    .stApp::before {{
        animation: none !important;
        background:
          radial-gradient(circle at 12% 18%, rgba(56,189,248,.24), transparent 23%),
          radial-gradient(circle at 88% 20%, rgba(37,99,235,.22), transparent 25%),
          radial-gradient(circle at 58% 86%, rgba(59,130,246,.18), transparent 28%),
          linear-gradient(135deg,#020817 0%,#06285d 48%,#0b3b82 72%,#020817 100%) !important;
    }}
    .stApp::after {{
        animation: none !important;
        background-image:
          linear-gradient(rgba(125,211,252,.075) 1px, transparent 1px),
          linear-gradient(90deg, rgba(125,211,252,.075) 1px, transparent 1px) !important;
        background-size: 64px 64px !important;
    }}
    .dynamic-3d-wallpaper {{
        opacity: .95;
        transform: none !important;
    }}
    .dynamic-3d-wallpaper .orb-a {{ left: 7%; top: 13%; transform: none !important; }}
    .dynamic-3d-wallpaper .orb-b {{ right: 4%; top: 8%; transform: none !important; }}
    .dynamic-3d-wallpaper .orb-c {{ right: 25%; bottom: 4%; transform: none !important; }}
    .dynamic-3d-wallpaper .ring.one {{ transform: rotateX(68deg) rotateY(12deg) !important; }}
    .dynamic-3d-wallpaper .ring.two {{ transform: rotateY(68deg) rotateZ(20deg) !important; }}
    .dynamic-3d-wallpaper .cube {{ transform: rotateX(18deg) rotateY(-24deg) rotateZ(8deg) !important; }}
    .login-grid {{ animation: none !important; }}
    .login-orb {{ animation: none !important; transform: none !important; }}
    .login-main-emoji {{ animation: none !important; transform: none !important; }}
    .brand-bar {{ display:flex; align-items:center; gap:14px; margin:0 0 22px; padding:10px 14px; width:max-content; border:1px solid rgba(125,211,252,.20); border-radius:18px; background:rgba(2,6,23,.48); backdrop-filter:blur(12px); }}
    .brand-mark {{ width:58px;height:58px; flex:0 0 58px; border-radius:17px; display:grid; place-items:center; background:linear-gradient(135deg,#22d3ee,#6366f1 55%,#a855f7); box-shadow:0 0 28px rgba(99,102,241,.42); }}
    .brand-name {{ font-weight:900; letter-spacing:.4px; font-size:1.05rem; color:#f8fafc; }} .brand-tag {{ color:#94a3b8; font-size:.72rem; margin-top:2px; }}
    .subject-card {{ border:1px solid rgba(255,255,255,.10); border-radius:18px; padding:15px; background:rgba(15,23,42,.58); margin:8px 0; }}
    .good-pop,.sad-pop {{ padding:25px;border-radius:24px;text-align:center;animation:pop .55s ease-out; }}
    .good-pop {{ background:linear-gradient(135deg,rgba(34,197,94,.20),rgba(16,185,129,.08));border:1px solid rgba(74,222,128,.45); }}
    .sad-pop {{ background:linear-gradient(135deg,rgba(239,68,68,.18),rgba(127,29,29,.08));border:1px solid rgba(248,113,113,.42); }}
    .smart-card {{ width:min(680px,100%); margin:18px auto; padding:28px; border-radius:26px; background:linear-gradient(135deg,#062a6b 0%,#0b63ce 38%,#12a4d8 72%,#083b8f 100%); border:2px solid rgba(255,255,255,.35); box-shadow:0 28px 80px rgba(0,0,0,.42); position:relative; overflow:hidden; }}
    .smart-card:before {{ content:""; position:absolute; inset:-30%; background:radial-gradient(circle,rgba(255,255,255,.18),transparent 35%); animation:cardshine 7s linear infinite; pointer-events:none; }}
    .smart-card .small {{ position:relative; color:#e0f2fe; font-size:.92rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }}
    .smart-card .name {{ position:relative; font-size:2.35rem; font-weight:900; margin:12px 0 5px; color:#fff; text-shadow:0 2px 10px rgba(0,0,0,.25); }}
    .smart-card .uid {{ position:relative; font-family:monospace; font-size:1.35rem; font-weight:800; letter-spacing:.10em; color:#fff; }}
    .smart-card .detail-grid {{ position:relative; display:grid; grid-template-columns:1fr 1fr; gap:9px 22px; margin-top:20px; font-size:1.02rem; line-height:1.45; color:#f0f9ff; }}
    .smart-card .label {{ color:#bae6fd; font-size:.78rem; font-weight:800; text-transform:uppercase; letter-spacing:.06em; }}
    .smart-card .value {{ color:#fff; font-weight:700; word-break:break-word; }}
    @keyframes cardshine {{ from{{transform:translate3d(-25%,-10%,0) rotate(0deg)}} to{{transform:translate3d(25%,10%,0) rotate(360deg)}} }}
    @keyframes pop {{ from{{transform:scale(.82);opacity:0}} to{{transform:scale(1);opacity:1}} }}
    @media (prefers-reduced-motion: reduce) {{ .stApp::before,.stApp::after,.login-grid,.login-orb,.login-main-emoji,.dynamic-3d-wallpaper * {{ animation:none !important; }} }}
    </style>
    """, unsafe_allow_html=True)

def app_brand():
    st.markdown("""
    <div class="dynamic-3d-wallpaper" aria-hidden="true">
      <div class="orb orb-a"></div><div class="orb orb-b"></div><div class="orb orb-c"></div>
      <div class="ring one"></div><div class="ring two"></div>
      <div class="cube"><i></i><i></i><i></i><i></i><i></i><i></i></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="brand-bar">
      <div class="brand-mark">
        <svg width="42" height="42" viewBox="0 0 64 64" aria-label="EduPredict SPP logo">
          <path d="M8 24 32 10l24 14-24 14L8 24Z" fill="none" stroke="white" stroke-width="3"/>
          <path d="M16 29v14c8 8 24 8 32 0V29M32 38v14" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"/>
        </svg>
      </div>
      <div><div class="brand-name">EduPredict SPP</div><div class="brand-tag">Student Performance Prediction • KTU B.Tech</div></div>
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


def _supabase_secret(*names):
    for name in names:
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        if value:
            return str(value).strip()
    return ""


def _get_shared_supabase():
    if not SUPABASE_AVAILABLE:
        return None
    url = _supabase_secret("SUPABASE_URL", "SUPABASE_PROJECT_URL").rstrip("/")
    key = _supabase_secret("SUPABASE_KEY", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    if url.endswith("/rest/v1"):
        url = url[:-8]
    if not url or not key:
        return None
    return create_client(url, key)


def sync_student_to_supabase(report):
    try:
        sb = _get_shared_supabase()
        if sb is None:
            return
        sb.table("students").upsert({
            "university_id": str(report.get("University_ID", "")).strip(),
            "student_name": str(report.get("Student_Name", "")).strip(),
            "department": str(report.get("Department", "")).strip(),
            "semester": str(report.get("Semester", "")).strip().upper(),
            "studied_college": "",
            "registered_by": str(report.get("Username", "")).strip(),
            "active": True,
        }, on_conflict="university_id").execute()
    except Exception:
        pass


def sync_marks_to_supabase(report):
    try:
        sb = _get_shared_supabase()
        if sb is None:
            return
        uid = str(report.get("University_ID", "")).strip()
        subjects = parse_subjects(report.get("Subjects_JSON", "[]"))
        tutor_name = str(st.session_state.get("username", "")).strip()
        sb.table("student_marks").delete().eq("university_id", uid).execute()
        sb.table("student_marks").insert({
            "university_id": uid,
            "department": str(report.get("Department", "")).strip(),
            "semester": str(report.get("Semester", "")).strip().upper(),
            "tutor_name": tutor_name,
            "subjects": subjects,
        }).execute()
    except Exception:
        pass


def sync_tutor_profile_to_supabase(username, tutor_name, department, credit_score):
    try:
        sb = _get_shared_supabase()
        if sb is None:
            return
        sb.table("tutors").upsert({
            "tutor_id": str(username).strip(),
            "tutor_name": str(tutor_name).strip(),
            "department": str(department).strip(),
            "semester": str(st.session_state.get("teacher_semester", "") or "").strip().upper(),
            "credit_score": float(credit_score or 0),
            "active": True,
        }, on_conflict="tutor_id").execute()
    except Exception:
        pass


def sync_audit_to_supabase(action, role, username, uid, department, semester, details):
    try:
        sb = _get_shared_supabase()
        if sb is None:
            return
        sb.table("audit_logs").insert({
            "username": str(username or "").strip(),
            "role": str(role or "").strip(),
            "action": str(action or "").strip(),
            "university_id": str(uid or "").strip(),
            "department": str(department or "").strip(),
            "semester": str(semester or "").strip().upper(),
            "details": str(details or "").strip(),
        }).execute()
    except Exception:
        pass


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
    sync_student_to_supabase(report)
    sync_marks_to_supabase(report)


def delete_student(uid, department="", semester="", tutor_username=""):
    """Permanently remove ONE selected student and all linked local/cloud details.

    The deletion is keyed by University ID only so a tutor's delete action
    removes the student's registration, marks, smart card and files across
    every semester/department record belonging to that University ID.
    """
    uid = str(uid or "").strip()
    if not uid:
        return False

    # Local marks/reports: remove every record for this student.
    df = load_reports()
    if not df.empty and "University_ID" in df.columns:
        save_reports(df.loc[~df["University_ID"].astype(str).eq(uid)].copy())

    # Local registration: remove every registration row for this student.
    reg = load_registrations()
    if not reg.empty and "University_ID" in reg.columns:
        save_registrations(reg.loc[~reg["University_ID"].astype(str).eq(uid)].copy())

    # Remove shared Supabase records. Each operation is isolated so one
    # missing/optional table cannot crash the Tutor + Student portal.
    cloud_errors = []
    try:
        sb = _get_shared_supabase()
        if sb is not None:
            # Remove all previous Tutor/Principal activity for this student so
            # the Principal dashboard no longer shows deleted student records.
            try:
                sb.table("audit_logs").delete().eq("university_id", uid).execute()
            except Exception as exc:
                cloud_errors.append(f"audit_logs: {exc}")
            for table in ("student_marks", "student_files"):
                try:
                    sb.table(table).delete().eq("university_id", uid).execute()
                except Exception as exc:
                    cloud_errors.append(f"{table}: {exc}")
            try:
                sb.table("smart_cards").delete().eq("university_id", uid).execute()
            except Exception as exc:
                cloud_errors.append(f"smart_cards: {exc}")
            try:
                sb.table("students").delete().eq("university_id", uid).execute()
            except Exception as exc:
                cloud_errors.append(f"students: {exc}")
    except Exception as exc:
        cloud_errors.append(str(exc))

    # Remove locally stored profile files.
    try:
        files = get_student_files(uid)
        for path in files:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
    except Exception:
        pass

    audit(
        "Student Permanently Deleted",
        "Tutor",
        tutor_username or st.session_state.get("username", ""),
        uid,
        department,
        semester,
        "Tutor deleted the selected student's registration, marks, Smart Card and linked files."
    )
    return True


def get_subjects(department, semester):
    """Return subjects ONLY for the selected department + semester.
    No cross-department/default fallback is allowed.
    """
    department = str(department).strip()
    semester = str(semester).strip().upper()
    branch = SUBJECTS_BY_DEPARTMENT.get(department, {})
    subjects = branch.get(semester, [])
    if not subjects:
        return []
    return list(dict.fromkeys(subjects))[:6]

def subject_selection_status(department, semester):
    subjects = get_subjects(department, semester)
    if not subjects:
        return f"No subject mapping found for {department} — {semester}."
    return f"{len(subjects)} subjects loaded for {department} — {semester}."


def parse_subjects(raw):
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []

# ============================================================
# REGISTRATION / PROFILE / AUDIT HELPERS
# ============================================================
def _load_generic_csv(path, columns):
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df[columns]
    except Exception:
        return pd.DataFrame(columns=columns)


def load_registrations():
    return _load_generic_csv(REGISTRATION_FILE, REGISTRATION_COLUMNS)


def save_registrations(df):
    df.to_csv(REGISTRATION_FILE, index=False)


def register_student(name, uid, department, semester, tutor_username):
    df = load_registrations()
    uid = uid.strip()
    mask = df["University_ID"].astype(str).str.lower().eq(uid.lower())
    row = {
        "University_ID": uid,
        "Student_Name": name.strip(),
        "Department": department,
        "Semester": semester,
        "Tutor_Username": tutor_username,
        "Registered_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if mask.any():
        df.loc[mask, list(row.keys())] = list(row.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_registrations(df)
    try:
        sb = _get_shared_supabase()
        if sb is not None:
            sb.table("students").upsert({
                "university_id": uid, "student_name": name.strip(),
                "department": department, "semester": str(semester).upper(),
                "studied_college": "", "registered_by": tutor_username, "active": True
            }, on_conflict="university_id").execute()
    except Exception:
        pass
    return row


def find_registration(uid):
    df = load_registrations()
    if df.empty:
        return None
    x = df[df["University_ID"].astype(str).str.lower().eq(str(uid).strip().lower())]
    return x.iloc[0].to_dict() if not x.empty else None


def load_tutor_profiles():
    return _load_generic_csv(TUTOR_PROFILE_FILE, TUTOR_PROFILE_COLUMNS)


def _normalise_tutor_email(email):
    return str(email or "").strip().lower()


def load_tutor_accounts():
    return _load_generic_csv(TUTOR_ACCOUNT_FILE, TUTOR_ACCOUNT_COLUMNS)


def _save_tutor_account_local(email, tutor_name, department, semester, tutor_id=None):
    """Save a local backup of the Tutor account without pandas assignment/type errors."""
    email = _normalise_tutor_email(email)
    tutor_id = _normalise_tutor_email(tutor_id or email)
    row = {
        "Email": email,
        "Tutor_Name": str(tutor_name or "").strip(),
        "Department": str(department or "").strip(),
        "Semester": str(semester or "").strip().upper(),
        "Tutor_ID": tutor_id,
        "Created_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Last_Login": "",
    }
    df = load_tutor_accounts()
    if not df.empty:
        mask = df["Email"].astype(str).str.strip().str.lower().eq(email)
        if mask.any():
            old = df.loc[mask].iloc[0].to_dict()
            row["Created_Time"] = str(old.get("Created_Time") or row["Created_Time"])
            row["Last_Login"] = str(old.get("Last_Login") or "")
            df = df.loc[~mask].copy()
        df = pd.concat([df, pd.DataFrame([row], columns=TUTOR_ACCOUNT_COLUMNS)], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=TUTOR_ACCOUNT_COLUMNS)
    df = df.reindex(columns=TUTOR_ACCOUNT_COLUMNS, fill_value="")
    df.to_csv(TUTOR_ACCOUNT_FILE, index=False)
    return row


def create_tutor_account(email, tutor_name, department, semester):
    email=_normalise_tutor_email(email)
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$",email): return None,"Enter a valid email address."
    if not tutor_name.strip(): return None,"Enter the tutor name."
    sb=_get_shared_supabase()
    try:
        if sb is not None:
            existing=sb.table("tutor_accounts").select("*").eq("email",email).limit(1).execute()
            er=list(getattr(existing,"data",None) or [])
            if er: return er[0],"This email already has a tutor account. Use the email to log in."
            row={"email":email,"tutor_id":email,"tutor_name":tutor_name.strip(),"department":department.strip(),"semester":semester.strip().upper(),"balance":0,"active":True}
            # IMPORTANT: tutor_accounts is a separate salary account.
            # Do NOT insert it into the tutors table; normal Tutor Login remains independent.
            sb.table("tutor_accounts").insert(row).execute()
            return _save_tutor_account_local(email,tutor_name,department,semester,email),"created"
        local=load_tutor_accounts()
        if not local.empty and local["Email"].astype(str).str.lower().eq(email).any(): return local[local["Email"].astype(str).str.lower().eq(email)].iloc[0].to_dict(),"This email already has a tutor account."
        row=_save_tutor_account_local(email,tutor_name,department,semester,email); save_tutor_profile(email,tutor_name,department,0.0); return row,"created"
    except Exception as exc:
        msg = str(exc)
        if "tutor_accounts" in msg and ("PGRST205" in msg or "schema cache" in msg or "relation" in msg):
            return None, ("Supabase table `tutor_accounts` is missing. Run the supplied "
                          "tutor_salary_setup.sql in Supabase SQL Editor, then refresh this app.")
        return None, f"Account creation failed: {msg}"


def get_tutor_account(email):
    email=_normalise_tutor_email(email)
    if not email: return None
    try:
        sb=_get_shared_supabase()
        if sb is not None:
            res=sb.table("tutor_accounts").select("*").eq("email",email).limit(1).execute(); data=list(getattr(res,"data",None) or [])
            if data: return data[0]
    except Exception: pass
    df=load_tutor_accounts()
    if df.empty: return None
    x=df[df["Email"].astype(str).str.lower().eq(email)]
    return x.iloc[0].to_dict() if not x.empty else None


def touch_tutor_account_login(email):
    email=_normalise_tutor_email(email); stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sb=_get_shared_supabase()
        if sb is not None: sb.table("tutor_accounts").update({"last_login":stamp}).eq("email",email).execute()
    except Exception: pass
    df=load_tutor_accounts()
    if not df.empty:
        mask=df["Email"].astype(str).str.lower().eq(email)
        if mask.any(): df.loc[mask,"Last_Login"]=stamp; df.to_csv(TUTOR_ACCOUNT_FILE,index=False)


def get_tutor_wallet(email):
    email=_normalise_tutor_email(email)
    try:
        sb=_get_shared_supabase()
        if sb is not None:
            acct=sb.table("tutor_accounts").select("balance").eq("email",email).limit(1).execute(); ar=list(getattr(acct,"data",None) or [])
            balance=float(ar[0].get("balance") or 0) if ar else 0.0
            tx=sb.table("tutor_salary_transactions").select("*").eq("tutor_id",email).order("sent_at",desc=True).execute()
            return {"balance":balance,"transactions":list(getattr(tx,"data",None) or [])}
    except Exception: pass
    return {"balance":0.0,"transactions":[]}


def save_tutor_profile(username, tutor_name, department, credit_score):
    df=load_tutor_profiles()
    row={"Tutor_Username":username,"Tutor_Name":tutor_name.strip(),"Department":department,"Credit_Score":float(credit_score),"Updated_Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    mask=df["Tutor_Username"].astype(str).eq(username)
    if mask.any(): df.loc[mask,list(row.keys())]=list(row.values())
    else: df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
    df.to_csv(TUTOR_PROFILE_FILE,index=False)
    sync_tutor_profile_to_supabase(username, tutor_name, department, credit_score)

def get_tutor_profile(username):
    df=load_tutor_profiles()
    if df.empty: return {"Tutor_Username":username,"Tutor_Name":"","Department":"","Credit_Score":0.0}
    x=df[df["Tutor_Username"].astype(str).eq(str(username))]
    if x.empty: return {"Tutor_Username":username,"Tutor_Name":"","Department":"","Credit_Score":0.0}
    r=x.iloc[0].to_dict()
    try:r["Credit_Score"]=float(r.get("Credit_Score",0))
    except:r["Credit_Score"]=0.0
    return r

def get_tutor_credit_score(username): return float(get_tutor_profile(username).get("Credit_Score",0.0))

def load_smart_card_registrations():
    """Load local Smart Card records with a stable schema."""
    return _load_generic_csv(SMART_CARD_FILE, SMART_CARD_COLUMNS)


def _clean_smart_card_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _smart_card_cloud_find(sb, university_id):
    """Return the latest cloud Smart Card for a university ID, or None."""
    try:
        rows = (
            sb.table("smart_cards")
            .select("*")
            .eq("university_id", str(university_id).strip())
            .order("submitted_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None
    except Exception:
        return None


def _smart_card_cloud_payload(row):
    """Build a Supabase payload without sending invalid empty values."""
    payload = {
        "registration_id": _clean_smart_card_value(row.get("Registration_ID"))
            or _clean_smart_card_value(row.get("University_ID")),
        "university_id": _clean_smart_card_value(row.get("University_ID")) or None,
        "name": _clean_smart_card_value(row.get("Student_Name")) or "Student",
        "dob": _clean_smart_card_value(row.get("DOB")) or None,
        "blood_group": _clean_smart_card_value(row.get("Blood_Group")) or None,
        "address": _clean_smart_card_value(row.get("Address")) or None,
        "pin_code": _clean_smart_card_value(row.get("PIN_Code")) or None,
        "studied_college": _clean_smart_card_value(row.get("Studied_College")) or None,
        "department": _clean_smart_card_value(row.get("Department")) or None,
        "semester": _clean_smart_card_value(row.get("Semester")) or None,
        "university_name": _clean_smart_card_value(row.get("University_Name")) or None,
    }
    cgpa = _clean_smart_card_value(row.get("CGPA"))
    if cgpa:
        payload["cgpa"] = float(cgpa)
    else:
        payload["cgpa"] = None
    return payload


def save_smart_card_registration(data):
    """Save/update a Smart Card locally and safely sync the same card to Supabase.

    The cloud update first looks up the existing card by University ID. This avoids
    duplicate cards when a student changes the Registration ID later.
    """
    df = load_smart_card_registrations()
    row = {c: _clean_smart_card_value(data.get(c, "")) for c in SMART_CARD_COLUMNS}

    uid = row["University_ID"]
    if not uid:
        raise ValueError("University ID is required for Smart Card registration.")

    if not row["Registration_ID"]:
        row["Registration_ID"] = uid

    row["Submitted_Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # One local card per University ID.
    mask = df["University_ID"].astype(str).str.strip().str.casefold().eq(uid.casefold())
    if mask.any():
        first_index = df.index[mask][0]
        for col in SMART_CARD_COLUMNS:
            df.at[first_index, col] = row.get(col, "")
        # Remove accidental duplicate local records for the same student.
        duplicate_indexes = df.index[mask][1:]
        if len(duplicate_indexes):
            df = df.drop(index=duplicate_indexes)
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(SMART_CARD_FILE, index=False)

    # Cloud sync is deliberately non-fatal: the student must still be able to
    # view the card even if Supabase is temporarily unavailable.
    try:
        sb = _get_shared_supabase()
        if sb is not None:
            payload = _smart_card_cloud_payload(row)
            old = _smart_card_cloud_find(sb, uid)
            if old and old.get("registration_id"):
                sb.table("smart_cards").update(payload).eq(
                    "registration_id", old["registration_id"]
                ).execute()
            else:
                sb.table("smart_cards").upsert(
                    payload, on_conflict="registration_id"
                ).execute()
    except Exception as exc:
        # Do not expose a long Supabase traceback to the student.
        st.info(
            "Smart Card saved successfully on this app. Cloud sync is temporarily "
            "unavailable, so the Principal may see it after Supabase is restored."
        )

    return row


def find_smart_card_registration(uid):
    """Find a Smart Card locally first, then fall back to Supabase."""
    clean_uid = _clean_smart_card_value(uid)
    if not clean_uid:
        return None

    df = load_smart_card_registrations()
    if not df.empty:
        x = df[df["University_ID"].astype(str).str.strip().str.casefold().eq(clean_uid.casefold())]
        if not x.empty:
            return x.iloc[-1].to_dict()

    try:
        sb = _get_shared_supabase()
        if sb is not None:
            cloud = _smart_card_cloud_find(sb, clean_uid)
            if cloud:
                return {
                    "Registration_ID": cloud.get("registration_id", clean_uid) or clean_uid,
                    "University_ID": cloud.get("university_id", clean_uid) or clean_uid,
                    "Student_Name": cloud.get("name", ""),
                    "DOB": cloud.get("dob", "") or "",
                    "Blood_Group": cloud.get("blood_group", "") or "",
                    "Address": cloud.get("address", "") or "",
                    "PIN_Code": cloud.get("pin_code", "") or "",
                    "Studied_College": cloud.get("studied_college", "") or "",
                    "Department": cloud.get("department", "") or "",
                    "Semester": cloud.get("semester", "") or "",
                    "CGPA": "" if cloud.get("cgpa") is None else str(cloud.get("cgpa")),
                    "University_Name": cloud.get("university_name", "") or "",
                    "Submitted_Time": cloud.get("submitted_at", "") or "",
                }
    except Exception:
        pass
    return None

def hide_smart_card(): st.session_state.smart_card_hidden=True

def audit(action, role, username="", uid="", department="", semester="", details=""):
    df = _load_generic_csv(AUDIT_FILE, AUDIT_COLUMNS)
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Role": role, "Username": username, "Action": action,
        "University_ID": uid, "Department": department, "Semester": semester,
        "Details": details,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(AUDIT_FILE, index=False)
    sync_audit_to_supabase(action, role, username, uid, department, semester, details)


def student_file_dir(uid):
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(uid).strip())
    path = os.path.join(STUDENT_FILES_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path


def save_student_file(uploaded_file, uid):
    if uploaded_file is None:
        return None
    folder = student_file_dir(uid)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", uploaded_file.name)
    path = os.path.join(folder, safe_name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def get_student_files(uid):
    folder = student_file_dir(uid)
    return [os.path.join(folder, x) for x in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, x))]


def create_student_card_png(student, output_format="PNG"):
    if not PIL_AVAILABLE:return None
    width,height=1000,630; img=Image.new("RGB",(width,height),(16,24,39)); draw=ImageDraw.Draw(img)
    draw.rounded_rectangle((18,18,width-18,height-18),radius=42,fill=(23,70,120),outline=(148,197,253),width=3)
    draw.rounded_rectangle((38,38,width-38,height-38),radius=34,fill=(12,31,55),outline=(66,153,225),width=2)
    try:
        brand=ImageFont.truetype("DejaVuSans-Bold.ttf",30); title=ImageFont.truetype("DejaVuSans.ttf",19); namef=ImageFont.truetype("DejaVuSans-Bold.ttf",31); uidf=ImageFont.truetype("DejaVuSansMono.ttf",25); normal=ImageFont.truetype("DejaVuSans.ttf",18); small=ImageFont.truetype("DejaVuSans.ttf",15)
    except Exception: brand=title=namef=uidf=normal=small=ImageFont.load_default()
    draw.text((72,70),"EduPredict SPP",font=brand,fill="white"); draw.text((72,112),"STUDENT SMART CARD  •  KTU B.Tech",font=title,fill=(174,214,247))
    draw.rounded_rectangle((72,160,178,232),radius=10,fill=(194,173,83),outline=(245,225,145),width=2)
    for yy in (178,202): draw.line((84,yy,166,yy),fill=(105,92,48),width=2)
    draw.text((205,165),str(student.get("Student_Name","Student"))[:30],font=namef,fill="white"); draw.text((205,207),str(student.get("Registration_ID",student.get("University_ID","")))[:24],font=uidf,fill=(220,235,250))
    left=[("DOB",student.get("DOB","")),("Blood Group",student.get("Blood_Group","")),("Department",student.get("Department","")),("Semester",student.get("Semester","")),("College",student.get("Studied_College","")),("University",student.get("University_Name",""))]
    right=[("University ID",student.get("University_ID","")),("PIN",student.get("PIN_Code","")),("CGPA",student.get("CGPA","") or "—")]
    y=280
    for label,val in left: draw.text((72,y),f"{label}: {str(val)[:42]}",font=normal,fill=(220,229,238)); y+=39
    y=280
    for label,val in right: draw.text((600,y),f"{label}: {str(val)[:28]}",font=normal,fill=(220,229,238)); y+=39
    draw.text((72,520),f"Address: {str(student.get('Address',''))[:78]}",font=small,fill=(190,205,220)); draw.text((72,553),"Valid student identity card • Academic record",font=small,fill=(148,197,253))
    initials="".join([x[:1] for x in str(student.get("Student_Name","S")).split()][:2]).upper() or "S"; draw.ellipse((870,70,930,130),fill=(42,73,105),outline=(148,197,253),width=2); bbox=draw.textbbox((0,0),initials,font=normal); draw.text((900-(bbox[2]-bbox[0])/2,100-(bbox[3]-bbox[1])/2),initials,font=normal,fill="white")
    buf=io.BytesIO(); img.save(buf,format=output_format); buf.seek(0); return buf.getvalue()

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


def tutor_overall_credit_10(subjects):
    """Compress all six tutor-entered subject scores into one student credit out of 10.
    Tutor credit is calculated before student study-hours are entered.
    """
    if not subjects:
        return 0.0
    scores = []
    for item in subjects:
        try:
            scores.append(float(item.get("Overall", 0.0)))
        except Exception:
            pass
    return round(float(np.mean(scores)) / 10.0, 2) if scores else 0.0

def automatic_prediction(credit_10):
    """Project prediction band from the compressed student credit."""
    c = float(credit_10)
    if c >= 8.0:
        return "Good"
    if c >= 6.5:
        return "Above Average"
    if c >= 5.0:
        return "Average"
    return "Needs Improvement"

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


def find_student(username="", uid=""):
    # Student authentication is based on University ID; username is kept only
    # for backward compatibility with older records.
    uid = str(uid).strip()
    reg = find_registration(uid)
    if reg:
        reports = load_reports()
        if not reports.empty:
            found = reports[reports["University_ID"].astype(str).str.lower().eq(uid.lower())]
            if not found.empty:
                return found.iloc[0].to_dict()
        return {
            "Username": "", "Student_Name": reg["Student_Name"], "University_ID": reg["University_ID"],
            "Semester": reg["Semester"], "Department": reg["Department"], "Subjects_JSON": "",
            "Created_Time": reg["Registered_Time"], "_registered_only": True
        }
    # Legacy records without registration entry.
    df = load_reports()
    if df.empty:
        return None
    found = df[df["University_ID"].astype(str).str.lower().eq(uid.lower())]
    return found.iloc[0].to_dict() if not found.empty else None

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
    total_credits, _earned = calculate_total_credits(subjects)
    prediction = str(subjects[0].get("Automatic_Prediction", automatic_prediction(overall/10))) if subjects else automatic_prediction(overall/10)
    story.append(Paragraph(f"<b>Overall Performance:</b> {overall:.2f}% &nbsp;&nbsp; <b>Earned Credits:</b> {total_credits} &nbsp;&nbsp; <b>Prediction:</b> {performance_circle(prediction)} {prediction}", styles["Heading2"]))
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
# ALL-DEPARTMENT ANALYSIS
# ============================================================
def build_all_department_analysis():
    df = load_reports()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0

    rows = []
    for _, row in df.iterrows():
        try:
            subjects = parse_subjects(row["Subjects_JSON"])
        except Exception:
            subjects = []
        if not subjects:
            continue
        scores = [float(x.get("Overall", 0)) for x in subjects]
        attendance = [float(x.get("Attendance", 0)) for x in subjects]
        internal = [float(x.get("Internal", 0)) for x in subjects]
        rows.append({
            "Department": str(row.get("Department", "Unknown")),
            "Semester": str(row.get("Semester", "Unknown")),
            "University_ID": str(row.get("University_ID", "")),
            "Student_Name": str(row.get("Student_Name", "")),
            "Overall": float(np.mean(scores)),
            "Attendance": float(np.mean(attendance)),
            "Internal": float(np.mean(internal)),
        })
    records = pd.DataFrame(rows)
    if records.empty:
        return records, pd.DataFrame(), 0.0

    summary = records.groupby("Department", as_index=False).agg(
        Students=("University_ID", "nunique"),
        Average_Overall=("Overall", "mean"),
        Average_Attendance=("Attendance", "mean"),
        Average_Internal=("Internal", "mean"),
    ).sort_values("Average_Overall", ascending=False)
    return records, summary, float(records["Overall"].mean())


def create_all_department_pdf(records, summary, overall_average):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("all_title", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, spaceAfter=12)
    story = [Paragraph("EduPredict SPP — All Department Analysis", title)]
    if records.empty:
        story.append(Paragraph("No student records are available for analysis.", styles["Normal"]))
    else:
        story.append(Paragraph(f"<b>Total student records:</b> {len(records)}", styles["Normal"]))
        story.append(Paragraph(f"<b>Overall average performance:</b> {overall_average:.2f}%", styles["Normal"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Department Summary", styles["Heading2"]))
        data = [["Department", "Students", "Avg Score", "Avg Attendance", "Avg Internal"]]
        for _, r in summary.iterrows():
            data.append([str(r["Department"]), str(int(r["Students"])), f"{r['Average_Overall']:.2f}%", f"{r['Average_Attendance']:.2f}%", f"{r['Average_Internal']:.2f}/40"])
        t = Table(data, repeatRows=1, colWidths=[210,55,65,75,65])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#172554")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(t)
        story.append(Spacer(1, 12))
        story.append(Paragraph("Student-level Overview", styles["Heading2"]))
        student_data = [["Department", "Semester", "University ID", "Student", "Score"]]
        for _, r in records.sort_values(["Department", "Semester", "Student_Name"]).iterrows():
            student_data.append([str(r["Department"]), str(r["Semester"]), str(r["University_ID"]), str(r["Student_Name"]), f"{r['Overall']:.2f}%"])
        st = Table(student_data, repeatRows=1, colWidths=[145,45,70,150,45])
        st.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#334155")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.5)]))
        story.append(st)
        story.append(Spacer(1, 12))
        story.append(Paragraph("This report is an academic dashboard analysis based on records entered into EduPredict SPP. Performance indicators are project-defined and are not official university grades.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def all_department_dashboard():
    app_brand()
    st.title("🌐 All Department Analysis Dashboard")
    st.caption("Combined academic overview of all departments and semesters stored in the application.")

    c1, c2 = st.columns([1, 1])
    if c1.button("⬅️ Back to Tutor Dashboard", use_container_width=True):
        st.session_state.dashboard_view = "department"
        st.session_state.page = "teacher_dashboard"
        st.rerun()
    if c2.button("🚪 Logout", use_container_width=True):
        logout()

    records, summary, overall_average = build_all_department_analysis()
    if records.empty:
        st.info("No student records are available yet. Add student records from the Tutor Dashboard first.")
        return

    a, b, c = st.columns(3)
    a.metric("Total Students", int(records["University_ID"].nunique()))
    b.metric("Departments", int(records["Department"].nunique()))
    c.metric("Overall Average", f"{overall_average:.2f}%")

    st.subheader("📊 Department-wise Analysis")
    display_summary = summary.copy()
    display_summary["Average_Overall"] = display_summary["Average_Overall"].map(lambda x: f"{x:.2f}%")
    display_summary["Average_Attendance"] = display_summary["Average_Attendance"].map(lambda x: f"{x:.2f}%")
    display_summary["Average_Internal"] = display_summary["Average_Internal"].map(lambda x: f"{x:.2f}/40")
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    st.subheader("👥 All Student Analysis")
    display_records = records.copy()
    display_records["Overall"] = display_records["Overall"].map(lambda x: f"{x:.2f}%")
    display_records["Attendance"] = display_records["Attendance"].map(lambda x: f"{x:.2f}%")
    display_records["Internal"] = display_records["Internal"].map(lambda x: f"{x:.2f}/40")
    st.dataframe(display_records, use_container_width=True, hide_index=True)

    pdf = create_all_department_pdf(records, summary, overall_average)
    st.download_button(
        "📥 Download All Department Analysis PDF",
        pdf,
        "All_Department_Analysis.pdf",
        "application/pdf",
        use_container_width=True,
        type="primary",
    )

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


def minute_visual():
    return ["🎓", "📚", "📊"][datetime.now().minute % 3]


def home_page():
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=60_000, key="home_page_minute_refresh")
    app_brand()
    visual = minute_visual()
    st.markdown(f"""
    <div class="hero">
      <div class="login-grid"></div><div class="login-orb one"></div><div class="login-orb two"></div>
      <div style="position:relative;z-index:2">
        <div class="login-main-emoji">{visual}</div>
        <div class="minute-badge">Live visual • changes every minute</div>
        <h1>Student Performance Prediction</h1>
        <p>KTU B.Tech student registration, marks, performance analysis and smart student card.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    a,b,c=st.columns(3)
    if a.button("👨‍🏫 Tutor Login",use_container_width=True,type="primary"):
        st.session_state.page="teacher_login"; st.rerun()
    if b.button("🎓 Student Login",use_container_width=True):
        st.session_state.page="student_login"; st.rerun()
    if c.button("🪪 Student Smart Card (Self Registration)",use_container_width=True):
        st.session_state.page="smart_card"; st.rerun()

    st.markdown("### 👤 Student Profile — Quick Visit")
    st.caption("Enter your University ID + Student Name to view your registered profile directly. Tutor marks are NOT required.")
    with st.form("home_student_profile_lookup"):
        h1, h2 = st.columns(2)
        home_uid = h1.text_input("🪪 University ID", placeholder="e.g. SNM25CE001")
        home_name = h2.text_input("👤 Student Name", placeholder="Enter your registered name")
        view_profile = st.form_submit_button("🔎 View My Profile", type="primary", use_container_width=True)
    if view_profile:
        if not home_uid.strip() or not home_name.strip():
            st.error("Please enter both University ID and Student Name.")
        else:
            prof = find_student(home_name.strip(), home_uid.strip())
            if prof:
                st.success("✅ Profile found. No tutor marks are required to view this profile.")
                p1,p2,p3,p4 = st.columns(4)
                p1.metric("👤 Name", prof.get("Student_Name", "—"))
                p2.metric("🪪 University ID", prof.get("University_ID", "—"))
                p3.metric("🏫 Department", prof.get("Department", "—"))
                p4.metric("📚 Semester", prof.get("Semester", "—"))
                st.write(f"**Studied College:** {prof.get('Studied_College', '—')}")
                st.write(f"**Registered At:** {prof.get('Registered_Time', '—')}")
            else:
                st.error("Profile not found. Check the University ID and Student Name.")


def login_shell(title, subtitle):
    app_brand()
    visual = minute_visual()
    st.markdown(f"""
    <div class="login-wrap">
      <div class="login-grid"></div><div class="login-orb one"></div><div class="login-orb two"></div>
      <div style="width:min(560px,90%);position:relative;z-index:2;text-align:center">
        <div class="login-main-emoji">{visual}</div>
        <div class="minute-badge">Visual changes every minute</div>
        <h1>{title}</h1><p style="color:#94a3b8">{subtitle}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def tutor_create_account_panel():
    st.markdown("### 🆕 Create Tutor Account")
    st.caption("Create your tutor account once. After that, log in using the registered email ID only.")
    with st.form("tutor_create_account_form"):
        email=st.text_input("📧 Tutor Email",placeholder="teacher_ceS3tutor@gmail.com")
        name=st.text_input("👨‍🏫 Tutor Name",placeholder="Enter your full name")
        department=st.selectbox("🎓 Department",DEPARTMENTS,key="new_tutor_department")
        semester=st.selectbox("📚 Semester",SEMESTERS,key="new_tutor_semester")
        create=st.form_submit_button("✨ Create One-Time Account",type="primary",use_container_width=True)
    if create:
        account,message=create_tutor_account(email,name,department,semester)
        if account and message=="created":
            st.success("✅ Tutor account created successfully. You can now log in with this email ID.")
            st.info(f"📧 Login email: **{email.strip().lower()}**")
            st.session_state.tutor_account_email=email.strip().lower()
        elif account:
            st.warning(f"ℹ️ {message}"); st.info(f"📧 Use **{email.strip().lower()}** on the login form.")
        else: st.error(message)


def tutor_account_login_panel():
    """Separate salary-account login. This NEVER logs the tutor into the Tutor dashboard."""
    st.markdown("### 💳 Tutor Salary Account")
    st.caption("This is separate from Tutor Login. Use the registered email only to view the salary wallet.")
    with st.form("tutor_salary_account_login_form"):
        email = st.text_input(
            "📧 Salary Account Email",
            placeholder="teacher_ceS3tutor@gmail.com",
            key="salary_account_login_email",
        )
        open_account = st.form_submit_button(
            "💳 Open Salary Account", type="primary", use_container_width=True
        )
    if open_account:
        email = _normalise_tutor_email(email)
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            st.error("Enter a valid email address.")
            return
        account = get_tutor_account(email)
        if not account:
            st.error("Salary account not found. Create the one-time account first using the 👤 button.")
            return
        wallet = get_tutor_wallet(email)
        st.session_state["salary_account_view_email"] = email
        st.session_state["salary_account_view"] = True
        st.success(f"✅ Salary account opened for {account.get('tutor_name') or 'Tutor'}.")
        st.metric("💰 Available Salary Balance", f"₹{float(wallet.get('balance') or 0):,.2f}")
        tx = wallet.get("transactions") or []
        if tx:
            txdf = pd.DataFrame([
                {
                    "Date": x.get("sent_at", ""),
                    "Month": x.get("salary_month", ""),
                    "Amount": f"₹{float(x.get('amount') or 0):,.2f}",
                    "Reference": x.get("reference", ""),
                    "Status": x.get("status", "credited"),
                }
                for x in tx
            ])
            st.dataframe(txdf, use_container_width=True, hide_index=True)
        else:
            st.info("No salary has been credited to this account yet.")


def teacher_login():
    """Normal Tutor Login. Kept completely separate from the salary account."""
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=60_000, key="teacher_login_minute_refresh")
    login_shell("Tutor Login", "Use your Tutor Username and Password")

    # Circle icon: account creation / salary-account access only.
    st.markdown("""<style>
    div[data-testid="stButton"] > button.tutor-account-circle{
        border-radius:50%!important;width:58px!important;height:58px!important;
        padding:0!important;font-size:24px!important;position:fixed!important;
        right:28px!important;top:78px!important;z-index:9999!important;
        border:2px solid rgba(125,211,252,.75)!important;
        box-shadow:0 8px 28px rgba(0,0,0,.35)!important;
    }
    </style>""", unsafe_allow_html=True)
    circle = st.button("👤", key="tutor_account_circle", help="Tutor Salary Account")
    if circle:
        st.session_state.show_tutor_create = not st.session_state.get("show_tutor_create", False)
        st.rerun()

    if st.session_state.get("show_tutor_create", False):
        tab_create, tab_account = st.tabs(["🆕 Create Account", "💳 Account Login"])
        with tab_create:
            tutor_create_account_panel()
        with tab_account:
            tutor_account_login_panel()
        st.divider()
        st.markdown("### 🔐 Normal Tutor Login")

    # IMPORTANT: this login is independent of the salary email account.
    with st.form("teacher_login_form"):
        username = st.text_input("👨‍🏫 Tutor Username", placeholder="e.g. teacher_ce")
        password = st.text_input("🔑 Tutor Password", type="password", placeholder="Enter tutor password")
        c1, c2 = st.columns(2)
        login = c1.form_submit_button("🔐 Tutor Login", use_container_width=True, type="primary")
        back = c2.form_submit_button("← Back", use_container_width=True)
    if back:
        st.session_state.page = "home"
        st.rerun()
    if login:
        username = username.strip()
        account = TEACHERS.get(username)
        if not account or account.get("password") != password:
            st.error("Invalid Tutor Username or Password.")
            return

        st.session_state.logged_in = True
        st.session_state.role = "teacher"
        st.session_state.username = username
        st.session_state.tutor_account_email = ""
        st.session_state.teacher_department = account["department"]
        st.session_state.teacher_semester = "S3"
        st.session_state.active_department = account["department"]
        st.session_state.teacher_menu_view = "menu"
        log_action(
            "Tutor", username, "Tutor Login", "", account["department"], "S3",
            "Successful normal tutor login; salary account remains separate"
        )
        st.session_state.show_tutor_create = False
        st.session_state.page = "teacher_dashboard"
        st.rerun()

def student_login():
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=60_000, key="student_login_minute_refresh")
    login_shell("Student Login", "Enter your registered University ID")
    with st.form("student_login_form"):
        uid=st.text_input("University ID",placeholder="e.g. SNM25CE001")
        c1,c2=st.columns(2)
        login=c1.form_submit_button("🎓 Open Dashboard",use_container_width=True,type="primary")
        back=c2.form_submit_button("← Back",use_container_width=True)
    if back:
        st.session_state.page="home"; st.rerun()
    if login:
        student=find_student("",uid)
        if student:
            st.session_state.logged_in=True; st.session_state.role="student"; st.session_state.username=""
            st.session_state.student_report=student
            audit("Student Login","Student","",uid,student.get("Department",""),student.get("Semester",""),"Successful login")
            st.session_state.page="student_dashboard"; st.rerun()
        else: st.error("No registered student found for this University ID.")

    st.divider()
    st.subheader("👤 Quick Student Profile Lookup")
    st.caption("Profile can be viewed using University ID + Student Name even when Tutor marks have not been entered yet.")
    with st.form("quick_profile_lookup"):
        q1,q2=st.columns(2)
        lookup_uid=q1.text_input("University ID", key="quick_profile_uid")
        lookup_name=q2.text_input("Student Name", key="quick_profile_name")
        lookup=q1.form_submit_button("🔎 View Profile", use_container_width=True)
    if lookup:
        prof=find_student(lookup_name.strip(),lookup_uid.strip()) if lookup_uid.strip() and lookup_name.strip() else None
        if prof:
            st.success("✅ Student profile found. Tutor marks are not required to view this profile.")
            st.dataframe(pd.DataFrame([{
                "Student Name": prof.get("Student_Name",""), "University ID": prof.get("University_ID",""),
                "Department": prof.get("Department",""), "Semester": prof.get("Semester",""),
                "Registered Time": prof.get("Registered_Time","")
            }]), use_container_width=True, hide_index=True)
        else:
            st.error("Profile not found. Check the University ID and Student Name.")


def smart_card_page():
    """Student Smart Card registration/view page.

    Works for students already registered by a tutor and also supports a
    University-ID lookup from Supabase/local data. It never requires marks.
    """
    app_brand()
    st.title("🪪 Student Smart Card")
    st.caption("🎓 Student self-service registration • Fill your own complete details • No tutor marks required")

    # IMPORTANT: Smart Card registration is a separate STUDENT self-service
    # workflow. It does NOT require tutor marks or tutor registration.
    logged_student = st.session_state.get("student_report") or {}
    uid = _clean_smart_card_value(logged_student.get("University_ID"))

    if not uid:
        uid = st.text_input("University ID *", placeholder="e.g. SNM25CE001", key="smart_self_uid").strip()

    if not uid:
        st.info("Enter your University ID to start your own Smart Card registration.")
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        return

    # Existing tutor/student profile is only used for optional pre-filling.
    # A student is still allowed to complete the entire Smart Card personally.
    reg = find_registration(uid) or {}
    if not reg:
        try:
            sb = _get_shared_supabase()
            if sb is not None:
                cloud_rows = (sb.table("students")
                    .select("university_id,student_name,department,semester,registered_at")
                    .eq("university_id", uid).limit(1).execute().data or [])
                if cloud_rows:
                    r = cloud_rows[0]
                    reg = {"University_ID": r.get("university_id", uid),
                           "Student_Name": r.get("student_name", ""),
                           "Department": r.get("department", ""),
                           "Semester": r.get("semester", ""),
                           "Registered_Time": r.get("registered_at", "")}
        except Exception:
            pass

    existing = find_smart_card_registration(uid)
    groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

    with st.form("student_smart_card_form", clear_on_submit=False):
        st.subheader("Enter / Update Full Details")

        c1, c2 = st.columns(2)
        registration_id = c1.text_input(
            "Registration ID *",
            value=_clean_smart_card_value((existing or {}).get("Registration_ID")) or uid,
        )
        name = c2.text_input(
            "Student Name *",
            value=_clean_smart_card_value((existing or {}).get("Student_Name"))
            or _clean_smart_card_value(reg.get("Student_Name")),
        )

        c1, c2 = st.columns(2)
        old_dob = _clean_smart_card_value((existing or {}).get("DOB"))
        try:
            dob_default = datetime.strptime(old_dob, "%Y-%m-%d").date() if old_dob else datetime(2007, 1, 1).date()
        except Exception:
            dob_default = datetime(2007, 1, 1).date()
        dob = c1.date_input("Date of Birth *", value=dob_default)

        old_blood = _clean_smart_card_value((existing or {}).get("Blood_Group"))
        blood_index = groups.index(old_blood) if old_blood in groups else 0
        blood = c2.selectbox("Blood Group *", groups, index=blood_index)

        address = st.text_area(
            "Full Address *",
            value=_clean_smart_card_value((existing or {}).get("Address")),
            height=110,
            placeholder="House / Building, Street, Place, District, State",
        )

        c1, c2 = st.columns(2)
        pin = c1.text_input(
            "PIN Code *",
            value=_clean_smart_card_value((existing or {}).get("PIN_Code")),
            max_chars=6,
        )
        college = c2.text_input(
            "Studied College *",
            value=_clean_smart_card_value((existing or {}).get("Studied_College")),
        )

        c1, c2 = st.columns(2)
        department = c1.text_input(
            "Department *",
            value=_clean_smart_card_value((existing or {}).get("Department")) or _clean_smart_card_value(reg.get("Department")),
            placeholder="e.g. Civil Engineering",
        )
        semester = c2.text_input(
            "Semester *",
            value=_clean_smart_card_value((existing or {}).get("Semester")) or _clean_smart_card_value(reg.get("Semester")),
            placeholder="e.g. S3",
        )

        c1, c2 = st.columns(2)
        cgpa = c1.text_input(
            "CGPA (optional)",
            value=_clean_smart_card_value((existing or {}).get("CGPA")),
        )
        university = c2.text_input(
            "University Name *",
            value=_clean_smart_card_value((existing or {}).get("University_Name"))
            or "APJ Abdul Kalam Technological University",
        )

        submit = st.form_submit_button(
            "💾 Save / Update Smart Card",
            type="primary",
            use_container_width=True,
        )

    if submit:
        registration_id = registration_id.strip()
        name = name.strip()
        address = address.strip()
        pin = pin.strip()
        college = college.strip()
        university = university.strip()
        cgpa = cgpa.strip()

        if not all([registration_id, uid, name, address, pin, college, department, semester, university]):
            st.error("Please fill all required fields marked with *.")
            return

        if not pin.isdigit() or len(pin) != 6:
            st.error("PIN Code must contain exactly 6 digits.")
            return

        if cgpa:
            try:
                cgpa_value = float(cgpa)
                if not 0 <= cgpa_value <= 10:
                    raise ValueError
                cgpa = f"{cgpa_value:.2f}".rstrip("0").rstrip(".")
            except Exception:
                st.error("CGPA must be a number from 0 to 10.")
                return

        data = {
            "Registration_ID": registration_id,
            "University_ID": uid,
            "Student_Name": name,
            "DOB": dob.strftime("%Y-%m-%d"),
            "Blood_Group": blood,
            "Address": address,
            "PIN_Code": pin,
            "Studied_College": college,
            "Department": department,
            "Semester": semester,
            "CGPA": cgpa,
            "University_Name": university,
        }

        try:
            row = save_smart_card_registration(data)
            st.session_state.smart_card_registration = row
            st.session_state.smart_card_hidden = False
            audit(
                "Smart Card Registration",
                "Student",
                "",
                uid,
                department,
                semester,
                "Full Smart Card details submitted/updated",
            )
            st.success("✅ Smart Card saved successfully.")
        except Exception as exc:
            st.error(f"Smart Card could not be saved: {exc}")
            return

    card_data = st.session_state.get("smart_card_registration") or find_smart_card_registration(uid)

    if card_data and not st.session_state.get("smart_card_hidden", False):
        details = [
            ("Date of Birth", card_data.get("DOB", "—")),
            ("Blood Group", card_data.get("Blood_Group", "—")),
            ("University ID", card_data.get("University_ID", "—")),
            ("PIN Code", card_data.get("PIN_Code", "—")),
            ("Department", card_data.get("Department", "—")),
            ("Semester", card_data.get("Semester", "—")),
            ("Studied College", card_data.get("Studied_College", "—")),
            ("University", card_data.get("University_Name", "—")),
            ("CGPA", card_data.get("CGPA", "—") or "—"),
            ("Address", card_data.get("Address", "—")),
        ]

        # Escape values before inserting them into HTML.
        from html import escape
        html = (
            "<div class='smart-card'>"
            "<div class='small'>EduPredict SPP • STUDENT SMART CARD</div>"
            f"<div class='name'>{escape(str(card_data.get('Student_Name', 'Student')))}</div>"
            f"<div class='uid'>{escape(str(card_data.get('Registration_ID', uid)))}</div>"
            "<div class='detail-grid'>"
        )
        for label, value in details:
            html += (
                "<div><div class='label'>"
                + escape(str(label))
                + "</div><div class='value'>"
                + escape(str(value or "—"))
                + "</div></div>"
            )
        html += "</div></div>"
        st.markdown(html, unsafe_allow_html=True)

        card = create_student_card_png(card_data)
        if card:
            st.download_button(
                "📥 Download Full Smart Card PNG",
                card,
                f"{re.sub(r'[^A-Za-z0-9_-]', '_', uid)}_Smart_Card.png",
                "image/png",
                type="primary",
                use_container_width=True,
            )

    if st.button(
        "← Back to Student Dashboard" if st.session_state.get("logged_in") else "← Back to Home",
        use_container_width=True,
    ):
        st.session_state.page = (
            "student_dashboard"
            if st.session_state.get("logged_in") and st.session_state.get("role") == "student"
            else "home"
        )
        st.rerun()

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
def tutor_registration_form(department, semester):
    st.subheader("📝 Student Registration")
    st.caption("Register the student first. Username is not required.")
    with st.form("student_registration_form"):
        name = st.text_input("Student Name")
        uid = st.text_input("University ID")
        submit = st.form_submit_button("💾 Register Student", type="primary", use_container_width=True)
    if submit:
        if not name.strip() or not uid.strip():
            st.error("Enter Student Name and University ID.")
            return
        row = register_student(name, uid, department, semester, st.session_state.username)
        audit("Student Registration", "Tutor", st.session_state.username, uid, department, semester, "Student profile registered")
        st.success(f"✅ {name} registered successfully.")
        st.session_state.student_registration = row


def tutor_mark_entry(department, semester):
    st.subheader("📝 Student Mark Entry")
    regs = load_registrations()
    regs = regs[(regs["Department"].astype(str) == str(department)) & (regs["Semester"].astype(str).str.upper() == str(semester).upper())].copy()
    if regs.empty:
        st.info("No registered students are available for this Tutor + Department + Semester. Register a student first.")
        return

    tutor_profile = get_tutor_profile(st.session_state.username)
    default_tutor_name = str(tutor_profile.get("Tutor_Name", "")).strip() or st.session_state.username

    labels = {f"{r['University_ID']} — {r['Student_Name']}": r for _, r in regs.iterrows()}
    selected_label = st.selectbox("Select registered student", list(labels.keys()), key="mark_entry_student")
    selected = labels[selected_label]
    subjects = get_subjects(department, semester)
    existing = load_reports()
    old = existing[existing["University_ID"].astype(str).eq(str(selected["University_ID"]))] if not existing.empty else pd.DataFrame()
    old_subs = parse_subjects(old.iloc[0]["Subjects_JSON"]) if not old.empty else []
    old_map = {x.get("Subject"): x for x in old_subs}

    st.info(f"👨‍🏫 Tutor: **{default_tutor_name}**  •  Department: **{department}**  •  Semester: **{semester}**")
    with st.form("mark_entry_form"):
        tutor_name = st.text_input("👨‍🏫 Tutor Name *", value=default_tutor_name)
        values = []
        for i, subject in enumerate(subjects):
            oldx = old_map.get(subject, {})
            st.markdown(f"**{i+1}. {subject}**")
            a,b,c,d = st.columns(4)
            att = a.number_input("Attendance %", 0.0, 100.0, float(oldx.get("Attendance",75)), 1.0, key=f"m_att_{i}_{selected['University_ID']}")
            internal = b.number_input("Internal /40", 0.0, 40.0, float(oldx.get("Internal",20)), 1.0, key=f"m_int_{i}_{selected['University_ID']}")
            assignment = c.number_input("Assignment /15", 0.0, 15.0, float(oldx.get("Assignment",8)), 1.0, key=f"m_asg_{i}_{selected['University_ID']}")
            previous = d.number_input("Previous /60", 0.0, 60.0, float(oldx.get("Previous",30)), 1.0, key=f"m_prev_{i}_{selected['University_ID']}")
            values.append(normalize_subject({"Subject":subject,"Study_Hours":0,"Attendance":att,"Internal":internal,"Assignment":assignment,"Previous":previous}))

        completed = st.checkbox("✅ Completed — include this Tutor work in Principal salary analysis", value=False, help="Only completed tutor work is counted for salary.")
        submitted = st.form_submit_button("💾 Submit Marks + Predict + Save Result", type="primary", use_container_width=True)

    if submitted:
        if not tutor_name.strip():
            st.error("Tutor Name is required.")
            return
        if len(subjects) != 6:
            st.error("Exactly 6 subjects are required.")
            return

        total_credits, _earned = calculate_total_credits(values)
        prediction = automatic_prediction(float(np.mean([float(v.get("Overall", 0)) for v in values])) / 10.0)
        for item in values:
            item["Tutor_Name"] = tutor_name.strip()
            item.pop("Tutor_Credit_10", None)
            item["Automatic_Prediction"] = prediction
            item["Tutor_Completed"] = bool(completed)
            item["Credit"] = int(item.get("Credit", 0))

        report = make_report("", selected["Student_Name"], selected["University_ID"], semester, department, values)
        upsert_report(report)
        # Explicitly sync tutor identity + credit to the shared Principal portal.
        save_tutor_profile(st.session_state.username, tutor_name.strip(), department, 0.0)
        audit("Student Mark Submission", "Tutor", st.session_state.username, selected["University_ID"], department, semester,
              f"Tutor={tutor_name.strip()}; Automatic Prediction={prediction}; Earned Credits={total_credits}")
        if completed:
            audit("Tutor Work Completed", "Tutor", st.session_state.username, selected["University_ID"], department, semester,
                  f"Tutor={tutor_name.strip()}; Earned Credits={total_credits}; Prediction={prediction}; Salary eligible=Yes")
            st.success(f"✅ Marks saved • Automatic Prediction: **{prediction}** • Earned Credits: **{total_credits}** • 🟢 Tutor work COMPLETED")
        else:
            audit("Tutor Work Saved - Pending Completion", "Tutor", st.session_state.username, selected["University_ID"], department, semester,
                  f"Tutor={tutor_name.strip()}; Earned Credits={total_credits}; Prediction={prediction}; Salary eligible=No")
            st.success(f"✅ Marks saved • Automatic Prediction: **{prediction}** • Earned Credits: **{total_credits}** • 🟡 Pending")
        st.rerun()


def tutor_student_files(department, semester):
    st.subheader("📎 Student Profile File / Photo")
    regs = load_registrations()
    regs = regs[(regs["Department"].astype(str) == str(department)) & (regs["Semester"].astype(str).str.upper() == str(semester).upper())].copy()
    if regs.empty:
        st.info("Register students first.")
        return
    labels = {f"{r['University_ID']} — {r['Student_Name']}": r for _, r in regs.iterrows()}
    label = st.selectbox("Select student", list(labels.keys()), key="file_student")
    student = labels[label]
    uploaded = st.file_uploader("Upload student photo or profile file", type=["png","jpg","jpeg","webp","pdf","doc","docx"], key=f"profile_upload_{student['University_ID']}")
    if uploaded is not None and st.button("📤 Save File", type="primary"):
        path = save_student_file(uploaded, student["University_ID"])
        audit("Student File Upload", "Tutor", st.session_state.username, student["University_ID"], department, semester, uploaded.name)
        st.success(f"✅ Saved: {uploaded.name}")
    files = get_student_files(student["University_ID"])
    if files:
        st.markdown("**Files visible to the student:**")
        for path in files:
            with open(path, "rb") as f:
                data = f.read()
            name = os.path.basename(path)
            st.download_button(f"📎 {name}", data, name, key=f"download_tutor_{student['University_ID']}_{name}")
    else:
        st.info("No file uploaded yet.")


def tutor_profile_tab():
    st.subheader("👨‍🏫 Tutor Profile"); profile=get_tutor_profile(st.session_state.username)
    with st.form("tutor_profile_form"):
        tutor_name=st.text_input("Tutor Name",value=str(profile.get("Tutor_Name","")),placeholder="Enter tutor full name")
        save=st.form_submit_button("💾 Save Tutor Details",type="primary")
    if save:
        if not tutor_name.strip(): st.error("Enter tutor name."); return
        save_tutor_profile(st.session_state.username,tutor_name,st.session_state.teacher_department,0.0)
        audit("Tutor Profile Updated","Tutor",st.session_state.username,"",st.session_state.teacher_department,"",f"Tutor={tutor_name}")
        st.success("✅ Tutor name saved.")

def teacher_dashboard():
    app_brand()
    assigned_department = st.session_state.get("teacher_department") or DEPARTMENTS[0]
    st.title("👨‍🏫 Tutor Dashboard")
    tutor_profile=get_tutor_profile(st.session_state.username)
    tutor_display=tutor_profile.get("Tutor_Name") or st.session_state.username
    st.info(f"Tutor: **{tutor_display}**  •  Department: **{assigned_department}**")
    st.caption("💳 Salary Account is separate. Open it from the 👤 account icon on the Tutor Login page.")
    c1,c2,c3 = st.columns([2.2,1.0,0.8])
    department = c1.selectbox("🎓 B.Tech Department", DEPARTMENTS, index=DEPARTMENTS.index(assigned_department), key="dashboard_department")
    default_sem=str(st.session_state.get("teacher_semester","S3")).upper(); sem_index=SEMESTERS.index(default_sem) if default_sem in SEMESTERS else 2
    semester = c2.selectbox("📚 Semester", SEMESTERS, index=sem_index, key="dashboard_semester")
    if c3.button("➡️ Next Dashboard", use_container_width=True):
        st.session_state.page = "all_department_analysis"; st.rerun()
    subjects = get_subjects(department, semester)
    if len(subjects) != 6:
        st.error(f"No complete 6-subject mapping is configured for **{department} — {semester}**.")
        return
    st.success(f"📖 Active curriculum: **{department} — {semester}**")
    tabs = st.tabs(["📝 Registration","📊 Mark Entry","📎 Student Files","👥 Records","📈 K-Means","👨‍🏫 Tutor Profile","🗑️ Delete Student"])
    with tabs[0]: tutor_registration_form(department, semester)
    with tabs[1]: tutor_mark_entry(department, semester)
    with tabs[2]: tutor_student_files(department, semester)
    with tabs[3]:
        df = load_reports()
        filtered = df[(df["Department"].astype(str)==str(department)) & (df["Semester"].astype(str).str.upper()==str(semester).upper())].copy()
        st.subheader(f"Student Records — {department} — {semester} ({len(filtered)})")
        if filtered.empty:
            st.info("No marks submitted yet. Registered students will appear in the Registration tab.")
        else:
            show=filtered[["Student_Name","University_ID","Semester","Department","Created_Time"]]
            st.dataframe(show,use_container_width=True,hide_index=True)
            ids=show["University_ID"].astype(str).tolist()
            selected=st.selectbox("Select University ID",ids,key="record_student_id")
            row=filtered[filtered["University_ID"].astype(str).eq(str(selected))].iloc[0].to_dict()
            subs=parse_subjects(row["Subjects_JSON"])
            overall=float(np.mean([x["Overall"] for x in subs])) if subs else 0.0
            st.metric("Selected Student Overall",f"{overall:.2f}%")
            rows=[[x["Subject"],x["Attendance"],x["Internal"],x["Assignment"],x["Previous"],x["Overall"],x["Level"]] for x in subs]
            st.dataframe(pd.DataFrame(rows,columns=["Subject","Attendance","Internal","Assignment","Previous","Score","Performance"]),use_container_width=True,hide_index=True)
            c1,c2=st.columns(2)
            c1.download_button("📥 Download Progress PDF",create_pdf(row),f"{selected}_Progress_Report.pdf","application/pdf",use_container_width=True)
            card=create_student_card_png(row)
            if card:
                c2.download_button("🪪 Download Student Card",card,f"{selected}_Student_Card.png","image/png",use_container_width=True)
    with tabs[4]:
        st.subheader(f"📈 K-Means Analysis — {department} — {semester}")
        result=kmeans_analysis(department)
        if not result.get("available"):
            st.warning(result.get("message","Analysis unavailable."))
        else:
            records=result["records"].copy(); semester_students=load_reports()
            ids_for_sem=set(semester_students[(semester_students["Department"].astype(str)==str(department))&(semester_students["Semester"].astype(str).str.upper()==str(semester).upper())]["University_ID"].astype(str))
            selected_records=records[records["University_ID"].astype(str).isin(ids_for_sem)].copy()
            st.metric("Subject records in selected semester",len(selected_records))
            st.dataframe(selected_records,use_container_width=True,hide_index=True)
            st.caption(f"K-Means is calculated from all available records in the selected department. Inertia: {result['inertia']:.4f}")
            st.dataframe(pd.DataFrame(result["summary"]),use_container_width=True,hide_index=True)
            st.dataframe(pd.DataFrame(result["centroids"]),use_container_width=True,hide_index=True)
            st.download_button("📊 Download K-Means Analysis PDF",create_kmeans_pdf(result),f"{department.replace(' ','_')}_KMeans_Analysis.pdf","application/pdf",use_container_width=True,type="primary")
    with tabs[5]: tutor_profile_tab()
    with tabs[6]:
        st.subheader("🗑️ Delete Registered Student")
        st.warning("This is the separate student-deletion page. Select ONE student, then delete that student's complete details. This action removes registration, marks, Smart Card and linked files.")
        reg = load_registrations()
        if reg.empty:
            st.info("No registered students are available to delete.")
        else:
            reg = reg.copy()
            reg["University_ID"] = reg["University_ID"].astype(str)
            # Show all registered students, not only students with marks.
            reg = reg.drop_duplicates(subset=["University_ID"], keep="last")
            labels = {f"{r['University_ID']} — {r['Student_Name']}": r for _, r in reg.iterrows()}
            label = st.selectbox("Select one registered student", list(labels.keys()), key="delete_registered_student")
            selected_student = labels[label]
            st.markdown(f"**Student:** {selected_student['Student_Name']}  \n**University ID:** {selected_student['University_ID']}  \n**Department:** {selected_student['Department']}  \n**Semester:** {selected_student['Semester']}")
            confirm = st.checkbox("I confirm that I want to permanently delete this student's complete details.", key="confirm_delete_student")
            if st.button("🗑️ Delete This Student Completely", type="primary", use_container_width=True, disabled=not confirm):
                uid = str(selected_student["University_ID"])
                delete_student(uid, str(selected_student.get("Department", department)), str(selected_student.get("Semester", semester)), st.session_state.get("username", ""))
                st.success(f"✅ {uid} and all linked student details were deleted.")
                st.rerun()

# ============================================================
# STUDENT WORKFLOW
# ============================================================
def student_dashboard():
    app_brand()
    student=st.session_state.get("student_report")
    if not student:
        logout(); return
    uid=student.get("University_ID","")
    reg=find_registration(uid)
    report=None
    reports=load_reports()
    if not reports.empty:
        x=reports[reports["University_ID"].astype(str).str.lower().eq(str(uid).lower())]
        if not x.empty: report=x.iloc[0].to_dict()
    profile=reg or student
    st.title(f"🎓 Welcome, {profile.get('Student_Name','Student')}")
    st.caption(f"University ID: {uid}  •  {profile.get('Semester','')}  •  {profile.get('Department','')}")
    c1,c2=st.columns(2)
    if c1.button("🪪 Student Smart Card (Self Registration)",use_container_width=True,type="primary"):
        st.session_state.smart_card_registration=find_smart_card_registration(uid); st.session_state.smart_card_hidden=False; st.session_state.page="smart_card"; st.rerun()
    if c2.button("🚪 Logout",use_container_width=True):
        audit("Student Logout","Student","",uid,profile.get("Department",""),profile.get("Semester",""),"Logout")
        logout()

    st.subheader("👤 My Registered Profile")
    p1,p2,p3=st.columns(3)
    p1.metric("Student",profile.get("Student_Name","")); p2.metric("University ID",uid); p3.metric("Semester",profile.get("Semester",""))
    st.write(f"**Department:** {profile.get('Department','')}")

    files=get_student_files(uid)
    if files:
        st.subheader("📎 Files from Tutor")
        for path in files:
            name=os.path.basename(path)
            ext=os.path.splitext(name)[1].lower()
            if ext in {".png",".jpg",".jpeg",".webp"}:
                st.image(path,caption=name,width=240)
            with open(path,"rb") as f:
                st.download_button(f"📥 {name}",f.read(),name,key=f"student_file_{name}")

    if not report or not str(report.get("Subjects_JSON","")).strip() or str(report.get("Subjects_JSON")) in {"[]","nan"}:
        st.info("⏳ Your profile is registered. Marks are not submitted yet. The Calculate My Result section will appear after your tutor submits marks.")
        return

    st.subheader("📊 Calculate My Mark")
    subjects=parse_subjects(report["Subjects_JSON"])
    with st.form("study_hours_form"):
        st.caption("Enter your total daily study hours once. This value is applied consistently to all subjects for the prediction.")
        total_study_hours=st.number_input("⏱️ Total daily study hours",0.0,24.0,2.0,0.5,key="student_total_study_hours")
        calculate=st.form_submit_button("📊 Calculate My Result",use_container_width=True,type="primary")
    if calculate:
        updated=[]
        for item in subjects:
            copy=dict(item); copy["Study_Hours"]=float(total_study_hours); updated.append(normalize_subject(copy))
        report["Subjects_JSON"]=json.dumps(updated); subjects=updated; st.session_state.student_report=report
        audit("Calculate My Mark","Student","",uid,profile.get("Department",""),profile.get("Semester",""),f"Total daily study hours={total_study_hours}")

    subjects=parse_subjects(st.session_state.student_report["Subjects_JSON"])
    overall=float(np.mean([x["Overall"] for x in subjects])) if subjects else 0
    total_credits, earned_credit_subjects = calculate_total_credits(subjects)
    max_credits = sum(SUBJECT_CREDITS[:len(subjects)])
    tutor_prediction=str(subjects[0].get("Automatic_Prediction", automatic_prediction(overall/10))) if subjects else automatic_prediction(overall/10)
    if any(float(x.get("Study_Hours",0))>0 for x in subjects):
        if overall>=80:
            st.balloons(); st.markdown(f'<div class="good-pop"><div style="font-size:3rem">🎉🏆</div><h2>Good Performance</h2><p>Keep your consistency. Overall score: <b>{overall:.2f}%</b></p></div>',unsafe_allow_html=True)
        elif overall<50:
            st.markdown(f'<div class="sad-pop"><div style="font-size:3rem">😔📉</div><h2>Needs Improvement</h2><p>Follow the subject-wise improvement methods. Overall score: <b>{overall:.2f}%</b></p></div>',unsafe_allow_html=True)
        else:
            st.info(f"📘 Overall score: **{overall:.2f}%**")
        a,b,c=st.columns(3)
        a.markdown(f'<div class="metric-card"><div class="label">Overall Score</div><div class="value">{overall:.2f}%</div></div>',unsafe_allow_html=True)
        b.markdown(f'<div class="metric-card"><div class="label">Subjects</div><div class="value">{len(subjects)}</div></div>',unsafe_allow_html=True)
        status="Good" if overall>=80 else "Above Average" if overall>=65 else "Average" if overall>=50 else "Needs Improvement"
        c.markdown(f'<div class="metric-card"><div class="label">Status</div><div class="value">{status}</div></div>',unsafe_allow_html=True)
        d1,d2=st.columns(2)
        d1.markdown(f'<div class="metric-card"><div class="label">🎓 Total Earned Credits</div><div class="value">{total_credits}/{max_credits}</div></div>',unsafe_allow_html=True)
        d2.markdown(f'<div class="metric-card"><div class="label">Automatic Prediction</div><div class="value">{tutor_prediction}</div></div>',unsafe_allow_html=True)
        if earned_credit_subjects:
            st.success("✅ Credits earned: " + ", ".join(earned_credit_subjects))
        else:
            st.warning("⚠️ No subject credit earned yet. A subject gives credit only when its score is 40% or above.")
        rows=[]
        for x in subjects:
            idx = subjects.index(x)
            subject_credit = SUBJECT_CREDITS[idx] if idx < len(SUBJECT_CREDITS) else 0
            passed = subject_passed(x)
            rows.append([x["Subject"],f"{x['Attendance']:.0f}%",f"{x.get('Attendance_Mark',attendance_mark(x['Attendance']))}/5",f"{x['Internal']:.0f}/40",f"{x['Assignment']:.0f}/15",f"{x['Previous']:.0f}/60",f"{x['Study_Hours']:.1f} h",f"{x['Overall']:.1f}%", "PASS" if passed else "NOT PASS", f"{subject_credit if passed else 0} cr",f"{x.get('Circle',performance_circle(x['Level']))} {x['Level']}"])
        st.subheader("📊 Subject-wise Result")
        st.dataframe(pd.DataFrame(rows,columns=["Subject","Attendance","Att. Mark","Internal","Assignment","Previous","Study","Score","Pass","Credit","Performance"]),use_container_width=True,hide_index=True)
        st.subheader("🎯 Improvement")
        for x in subjects:
            with st.expander(f"{x.get('Circle',performance_circle(x['Level']))} {x['Subject']} — {x['Level']} — {x['Overall']:.1f}%"):
                st.write(f"Attendance **{x['Attendance']:.0f}%**, Internal **{x['Internal']:.0f}/40**, Assignment **{x['Assignment']:.0f}/15**, Previous **{x['Previous']:.0f}/60**, Study **{x['Study_Hours']:.1f} h/day**.")
                st.success("💚 "+x.get("Compliment","Keep progressing!"))
                for recommendation in x.get("Recommendations",[]): st.write("• "+recommendation)
        st.caption("🔴 Needs Improvement  •  🟠 Average  •  🟡 Above Average  •  🟢 Good")
        c1,c2=st.columns(2)
        c1.download_button("📥 Download Progress Report PDF",create_pdf(st.session_state.student_report),f"{uid}_Progress_Report.pdf","application/pdf",use_container_width=True,type="primary")
        card=create_student_card_png(profile)
        if card: c2.download_button("🪪 Download Smart Card PNG",card,f"{uid}_Smart_Card.png","image/png",use_container_width=True)

# Clear obsolete widget keys from earlier versions if they exist.
for _old_key in ("active_department", "manual_department_selector", "manual_semester_selector"):
    if _old_key in st.session_state:
        del st.session_state[_old_key]

# ============================================================
# ROUTER
# ============================================================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "teacher_login":
    teacher_login()
elif st.session_state.page == "student_login":
    student_login()
elif st.session_state.page == "smart_card":
    smart_card_page()
elif st.session_state.page == "teacher_dashboard" and st.session_state.logged_in and st.session_state.role == "teacher":
    teacher_dashboard()
elif st.session_state.page == "all_department_analysis" and st.session_state.logged_in and st.session_state.role == "teacher":
    all_department_dashboard()
elif st.session_state.page == "student_dashboard" and st.session_state.logged_in and st.session_state.role == "student":
    student_dashboard()
else:
    st.session_state.page = "home"
    st.rerun()
