import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import re
from datetime import datetime
import time
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
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
    from sklearn.neural_network import MLPClassifier
    SKLEARN_AVAILABLE = True
    ANN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    ANN_AVAILABLE = False

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
# Permanent local archive: every submitted student record is also saved as
# an individual JSON snapshot. Tutor actions never delete archived records.
STUDENT_RECORDS_DIR = os.path.join(DATA_DIR, "student_records")
STUDENT_REGISTRATIONS_FILE = os.path.join(DATA_DIR, "student_registrations.csv")
AUDIT_LOG_FILE = os.path.join(DATA_DIR, "audit_log.csv")
SMART_CARD_FILE = os.path.join(DATA_DIR, "smart_card_registrations.csv")
SMART_CARD_DIR = os.path.join(DATA_DIR, "smart_cards")
os.makedirs(SMART_CARD_DIR, exist_ok=True)
os.makedirs(STUDENT_RECORDS_DIR, exist_ok=True)

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
    "dashboard_view": "department",
    "teacher_menu_view": "menu",
    "intro_seen": False,
    "smart_card_preview": None,
    "smart_card_message": "",
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# UI
# ============================================================
def inject_css():
    """Inject a minute-cycling, CSS-only 3D wallpaper behind the Streamlit UI."""
    minute_key = int(time.time() // 60)
    palettes = [
        ("#22d3ee", "#6366f1", "#a855f7", "#020617"),
        ("#34d399", "#06b6d4", "#3b82f6", "#021014"),
        ("#f59e0b", "#ef4444", "#ec4899", "#16070a"),
        ("#f472b6", "#8b5cf6", "#06b6d4", "#0a0616"),
        ("#60a5fa", "#14b8a6", "#84cc16", "#04100c"),
        ("#fb7185", "#f97316", "#eab308", "#140803"),
    ]
    c1, c2, c3, base = palettes[minute_key % len(palettes)]
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=60_000, key="minute_wallpaper_refresh")
    st.markdown(f"""
    <style>
    :root {{ --c1:{c1}; --c2:{c2}; --c3:{c3}; --base:{base}; }}
    html,body,[class*="css"] {{ font-family:Inter,system-ui,sans-serif; }}
    .stApp {{ min-height:100vh; color:#eef2ff; background:var(--base); overflow-x:hidden; }}
    .stApp::before {{ content:""; position:fixed; inset:-25%; z-index:-10; pointer-events:none;
      background:radial-gradient(circle at 15% 20%,color-mix(in srgb,var(--c1) 24%,transparent),transparent 24%),
      radial-gradient(circle at 82% 28%,color-mix(in srgb,var(--c2) 25%,transparent),transparent 25%),
      radial-gradient(circle at 52% 88%,color-mix(in srgb,var(--c3) 22%,transparent),transparent 27%),
      linear-gradient(135deg,var(--base),#020617 55%,var(--base));
      animation:spacePulse 16s ease-in-out infinite alternate; }}
    .ep-wallpaper {{ position:fixed; inset:0; z-index:-8; pointer-events:none; overflow:hidden; perspective:1200px; }}
    .ep-grid {{ position:absolute; left:-15%; width:130%; height:75%; bottom:-28%; opacity:.20;
      background-image:linear-gradient(rgba(255,255,255,.16) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.16) 1px,transparent 1px);
      background-size:54px 54px; transform:rotateX(64deg) translateZ(-120px); transform-origin:center bottom; animation:gridRun 7s linear infinite; }}
    .ep-orb {{ position:absolute; border-radius:50%; transform-style:preserve-3d; mix-blend-mode:screen; }}
    .ep-orb::after {{ content:""; position:absolute; inset:13%; border-radius:50%; border:1px solid rgba(255,255,255,.35); transform:translateZ(30px); }}
    .o1 {{ width:210px;height:210px;left:5%;top:10%; background:radial-gradient(circle at 28% 24%,#fff 0 4%,var(--c1) 16%,var(--c2) 52%,transparent 72%); box-shadow:0 0 90px color-mix(in srgb,var(--c1) 45%,transparent); animation:orb1 13s ease-in-out infinite; }}
    .o2 {{ width:165px;height:165px;right:9%;top:16%; background:radial-gradient(circle at 30% 22%,#fff 0 3%,var(--c3) 18%,var(--c2) 58%,transparent 73%); box-shadow:0 0 80px color-mix(in srgb,var(--c3) 42%,transparent); animation:orb2 16s ease-in-out infinite; }}
    .o3 {{ width:120px;height:120px;left:43%;bottom:9%; background:radial-gradient(circle at 28% 22%,#fff 0 4%,var(--c2) 18%,var(--c1) 60%,transparent 75%); box-shadow:0 0 65px color-mix(in srgb,var(--c2) 38%,transparent); animation:orb3 11s ease-in-out infinite; }}
    .ep-ring {{ position:absolute; width:210px;height:210px; border:3px solid color-mix(in srgb,var(--c1) 70%,transparent); border-radius:50%; transform-style:preserve-3d; box-shadow:0 0 35px color-mix(in srgb,var(--c1) 30%,transparent), inset 0 0 20px color-mix(in srgb,var(--c2) 18%,transparent); }}
    .r1 {{ right:22%;bottom:18%; animation:ring1 12s linear infinite; }}
    .r2 {{ left:19%;top:48%; width:145px;height:145px; border-color:color-mix(in srgb,var(--c3) 68%,transparent); animation:ring2 9s linear infinite reverse; }}
    .ep-ring::before,.ep-ring::after {{ content:""; position:absolute; inset:17px; border:1px dashed rgba(255,255,255,.26); border-radius:50%; transform:rotateX(70deg); }}
    .ep-ring::after {{ inset:38px; transform:rotateY(70deg); }}
    .ep-cube {{ position:absolute; left:47%;top:34%;width:110px;height:110px;transform-style:preserve-3d; animation:cubeSpin 14s linear infinite; }}
    .ep-cube span {{ position:absolute; inset:0; border:2px solid color-mix(in srgb,var(--c2) 70%,transparent); background:color-mix(in srgb,var(--c2) 4%,transparent); box-shadow:0 0 24px color-mix(in srgb,var(--c2) 20%,transparent); }}
    .front{{transform:translateZ(55px)}} .back{{transform:rotateY(180deg) translateZ(55px)}} .left{{transform:rotateY(-90deg) translateZ(55px)}} .right{{transform:rotateY(90deg) translateZ(55px)}} .top{{transform:rotateX(90deg) translateZ(55px)}} .bottom{{transform:rotateX(-90deg) translateZ(55px)}}
    .ep-particle {{ position:absolute;width:5px;height:5px;border-radius:50%;background:var(--c1);box-shadow:0 0 18px var(--c1); animation:particle 8s ease-in-out infinite; }}
    .p1{{left:30%;top:18%;animation-delay:-1s}} .p2{{left:72%;top:65%;animation-delay:-3s}} .p3{{left:12%;top:78%;animation-delay:-5s}} .p4{{left:84%;top:80%;animation-delay:-7s}}
    @keyframes spacePulse{{50%{{transform:scale(1.08) rotate(1deg);filter:hue-rotate(12deg)}}100%{{transform:scale(1.02) rotate(-1deg);filter:hue-rotate(-8deg)}}}}
    @keyframes gridRun{{to{{background-position:0 108px,108px 0}}}}
    @keyframes orb1{{0%,100%{{transform:translate3d(0,0,0) rotateX(0) rotateY(0)}}50%{{transform:translate3d(100px,70px,180px) rotateX(160deg) rotateY(220deg)}}}}
    @keyframes orb2{{0%,100%{{transform:translate3d(0,0,0)}}50%{{transform:translate3d(-120px,80px,150px) rotateZ(180deg)}}}}
    @keyframes orb3{{0%,100%{{transform:translate3d(0,0,0) scale(1)}}50%{{transform:translate3d(-80px,-90px,220px) scale(1.25)}}}}
    @keyframes ring1{{from{{transform:rotateX(65deg) rotateY(0) rotateZ(0) translateZ(0)}}to{{transform:rotateX(425deg) rotateY(360deg) rotateZ(180deg) translateZ(130px)}}}}
    @keyframes ring2{{from{{transform:rotateX(75deg) rotateY(0)}}to{{transform:rotateX(435deg) rotateY(-360deg)}}}}
    @keyframes cubeSpin{{to{{transform:rotateX(360deg) rotateY(360deg) rotateZ(360deg) translateZ(80px)}}}}
    @keyframes particle{{50%{{transform:translate3d(90px,-80px,180px) scale(2);opacity:.25}}}}
    .block-container{{max-width:1250px;padding-top:2rem;position:relative;z-index:2}}
    .hero,.login-wrap{{position:relative;overflow:hidden;border:1px solid rgba(255,255,255,.14);border-radius:28px;background:linear-gradient(135deg,rgba(5,12,30,.76),rgba(15,23,42,.58));box-shadow:0 25px 100px rgba(0,0,0,.45);backdrop-filter:blur(14px)}}
    .hero{{padding:45px 40px}} .hero h1{{font-size:clamp(2rem,5vw,4rem);margin:0;font-weight:800;letter-spacing:-2px}} .hero p{{color:#b7c2d9;font-size:1.05rem;max-width:760px}}
    .login-wrap{{min-height:430px;display:flex;align-items:center;justify-content:center}}
    .glass{{background:rgba(15,23,42,.68);border:1px solid rgba(255,255,255,.14);border-radius:22px;padding:24px;backdrop-filter:blur(18px)}}
    .metric-card{{padding:20px;border-radius:20px;border:1px solid rgba(255,255,255,.10);background:linear-gradient(135deg,rgba(30,41,59,.75),rgba(15,23,42,.7))}} .metric-card .value{{font-size:2rem;font-weight:800}} .metric-card .label{{color:#94a3b8}}
    .brand-bar{{display:flex;align-items:center;gap:14px;margin:0 0 22px;padding:10px 14px;width:max-content;border:1px solid rgba(125,211,252,.20);border-radius:18px;background:rgba(2,6,23,.48);backdrop-filter:blur(12px);box-shadow:0 10px 35px rgba(0,0,0,.22)}}
    .brand-mark{{width:58px;height:58px;flex:0 0 58px;border-radius:17px;display:grid;place-items:center;background:linear-gradient(135deg,var(--c1),var(--c2) 55%,var(--c3));box-shadow:0 0 28px color-mix(in srgb,var(--c2) 42%,transparent)}}
    .brand-name{{font-weight:900;letter-spacing:.4px;font-size:1.05rem;color:#f8fafc}} .brand-tag{{color:#94a3b8;font-size:.72rem;margin-top:2px}}
    .good-pop,.sad-pop{{padding:25px;border-radius:24px;text-align:center;animation:pop .55s ease-out}} .good-pop{{background:linear-gradient(135deg,rgba(34,197,94,.20),rgba(16,185,129,.08));border:1px solid rgba(74,222,128,.45)}} .sad-pop{{background:linear-gradient(135deg,rgba(239,68,68,.18),rgba(127,29,29,.08));border:1px solid rgba(248,113,113,.42)}} @keyframes pop{{from{{transform:scale(.82);opacity:0}}to{{transform:scale(1);opacity:1}}}}
    .smart-card{{width:min(100%,640px);aspect-ratio:1.586/1;border-radius:24px;padding:28px;position:relative;overflow:hidden;color:white;background:linear-gradient(135deg,var(--c2),var(--c1) 52%,var(--c3));box-shadow:0 25px 70px rgba(0,0,0,.45),inset 0 0 40px rgba(255,255,255,.08)}}
    .smart-card:before{{content:"";position:absolute;inset:-40%;background:repeating-linear-gradient(115deg,rgba(255,255,255,.08) 0 2px,transparent 2px 18px);transform:rotate(8deg);animation:cardShine 8s linear infinite}} @keyframes cardShine{{to{{transform:translateX(120px) rotate(8deg)}}}}
    .smart-chip{{width:58px;height:44px;border-radius:10px;background:linear-gradient(135deg,#f5d98b,#c69b3c);border:1px solid rgba(255,255,255,.55);box-shadow:0 3px 12px rgba(0,0,0,.25)}}
    .intro3d{{position:relative;min-height:520px;display:flex;flex-direction:column;align-items:center;justify-content:center;overflow:hidden;border:1px solid rgba(125,211,252,.18);border-radius:30px;background:radial-gradient(circle at center,rgba(30,41,59,.7),rgba(2,6,23,.92));perspective:900px}}
    @media (prefers-reduced-motion: reduce){{.ep-wallpaper *{{animation:none !important}}}}
    </style>
    <div class="ep-wallpaper" aria-hidden="true">
      <div class="ep-grid"></div><div class="ep-orb o1"></div><div class="ep-orb o2"></div><div class="ep-orb o3"></div>
      <div class="ep-ring r1"></div><div class="ep-ring r2"></div>
      <div class="ep-cube"><span class="front"></span><span class="back"></span><span class="left"></span><span class="right"></span><span class="top"></span><span class="bottom"></span></div>
      <i class="ep-particle p1"></i><i class="ep-particle p2"></i><i class="ep-particle p3"></i><i class="ep-particle p4"></i>
    </div>
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
      <div><div class="brand-name">EduPredict SPP</div><div class="brand-tag">Student Performance Prediction • KTU B.Tech • ANN + K-Means</div></div>
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


REGISTRATION_COLUMNS = ["Username", "University_ID", "Student_Name", "Department", "Semester", "Tutor_Username", "Registered_Time"]
AUDIT_COLUMNS = ["Timestamp", "Role", "Username", "Action", "University_ID", "Department", "Semester", "Details"]

def empty_registration_df():
    return pd.DataFrame(columns=REGISTRATION_COLUMNS)

def load_registrations():
    if not os.path.exists(STUDENT_REGISTRATIONS_FILE):
        return empty_registration_df()
    try:
        df = pd.read_csv(STUDENT_REGISTRATIONS_FILE)
        for col in REGISTRATION_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[REGISTRATION_COLUMNS]
    except Exception:
        return empty_registration_df()

def save_registrations(df):
    df.to_csv(STUDENT_REGISTRATIONS_FILE, index=False)

def log_action(role, username, action, university_id="", department="", semester="", details=""):
    """Local audit trail used by the separate Principal monitoring app."""
    row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Role": role, "Username": username, "Action": action,
        "University_ID": university_id, "Department": department,
        "Semester": semester, "Details": details,
    }], columns=AUDIT_COLUMNS)
    try:
        if os.path.exists(AUDIT_LOG_FILE):
            old = pd.read_csv(AUDIT_LOG_FILE)
            for col in AUDIT_COLUMNS:
                if col not in old.columns: old[col] = ""
            old = old[AUDIT_COLUMNS]
        else:
            old = pd.DataFrame(columns=AUDIT_COLUMNS)
        pd.concat([old, row], ignore_index=True).to_csv(AUDIT_LOG_FILE, index=False)
    except Exception:
        pass

def register_student(username, uid, name, department, semester, tutor_username):
    username, uid, name = username.strip(), uid.strip(), name.strip()
    department, semester, tutor_username = department.strip(), semester.strip().upper(), tutor_username.strip()
    df = load_registrations()
    if not username or not uid or not name or not department or not semester:
        return False, "Enter Username, University ID, Student Name, Department and Semester."
    if department not in DEPARTMENTS or semester not in SEMESTERS:
        return False, "Select a valid B.Tech Department and Semester."
    duplicate = df[
        df["University_ID"].astype(str).str.strip().str.lower().eq(uid.lower()) |
        df["Username"].astype(str).str.strip().str.lower().eq(username.lower())
    ]
    if not duplicate.empty:
        return False, "This Username or University ID is already registered."
    row = pd.DataFrame([{
        "Username": username, "University_ID": uid, "Student_Name": name,
        "Department": department, "Semester": semester,
        "Tutor_Username": tutor_username,
        "Registered_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }], columns=REGISTRATION_COLUMNS)
    save_registrations(pd.concat([df, row], ignore_index=True))
    log_action("Tutor", tutor_username, "Student Registration", uid, department, semester, f"Registered {name}")
    return True, f"Student {name} registered successfully for {department} — {semester}."

def find_registered_student(uid):
    df = load_registrations()
    if df.empty: return None
    found = df[df["University_ID"].astype(str).str.strip().str.lower().eq(str(uid).strip().lower())]
    return found.iloc[-1].to_dict() if not found.empty else None

def find_registered_credentials(username, uid):
    """Find a tutor-registered student using both Username and University ID."""
    df = load_registrations()
    if df.empty: return None
    found = df[
        df["Username"].astype(str).str.strip().str.lower().eq(str(username).strip().lower()) &
        df["University_ID"].astype(str).str.strip().str.lower().eq(str(uid).strip().lower())
    ]
    return found.iloc[-1].to_dict() if not found.empty else None

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


def _safe_folder_name(value):
    """Make a Windows/Linux-safe folder/file component."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "Unknown"


def archive_student_record(report):
    """Write an immutable snapshot of every submitted record to a proper folder.

    Nothing in the tutor UI deletes these snapshots. A new submission creates a
    new timestamped JSON file, so the complete history is retained.
    """
    department_dir = os.path.join(
        STUDENT_RECORDS_DIR, _safe_folder_name(report.get("Department", "Unknown"))
    )
    semester_dir = os.path.join(
        department_dir, _safe_folder_name(report.get("Semester", "Unknown"))
    )
    os.makedirs(semester_dir, exist_ok=True)

    uid = _safe_folder_name(report.get("University_ID", "Unknown"))
    timestamp = re.sub(r"[^0-9]", "", str(report.get("Created_Time", datetime.now().strftime("%Y%m%d%H%M%S"))))
    filename = f"{uid}_{timestamp}_{datetime.now().strftime('%f')}.json"
    path = os.path.join(semester_dir, filename)

    payload = dict(report)
    payload["Archive_Path"] = path.replace("\\", "/")
    payload["Archived_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def upsert_report(report):
    """Append-only storage. Existing student records are NEVER overwritten/deleted.

    The function name is kept for compatibility with the rest of the app, but it
    now appends every submission to the master CSV and creates an archive file.
    """
    df = load_reports()
    new_row = pd.DataFrame([report], columns=REPORT_COLUMNS)
    df = pd.concat([df, new_row], ignore_index=True)
    save_reports(df)
    archive_student_record(report)


def delete_student_record(report):
    """Delete one stored student submission. This function is called only from
    the authenticated Tutor Dashboard. The matching master CSV row and its
    archived JSON snapshot are removed together.
    """
    df = load_reports()
    if df.empty:
        return False, "No stored records found."

    uid = str(report.get("University_ID", "")).strip()
    department = str(report.get("Department", "")).strip()
    semester = str(report.get("Semester", "")).strip()
    created = str(report.get("Created_Time", "")).strip()

    mask = (
        df["University_ID"].astype(str).str.strip().eq(uid) &
        df["Department"].astype(str).str.strip().eq(department) &
        df["Semester"].astype(str).str.strip().str.upper().eq(semester.upper()) &
        df["Created_Time"].astype(str).str.strip().eq(created)
    )
    if not mask.any():
        return False, "The selected record could not be found."

    save_reports(df.loc[~mask].copy())

    # Remove the corresponding archived JSON snapshot(s).
    archive_dir = os.path.join(
        STUDENT_RECORDS_DIR,
        _safe_folder_name(department),
        _safe_folder_name(semester),
    )
    removed_files = 0
    if os.path.isdir(archive_dir):
        timestamp = re.sub(r"[^0-9]", "", created)
        prefix = f"{_safe_folder_name(uid)}_{timestamp}_"
        for filename in os.listdir(archive_dir):
            if filename.startswith(prefix) and filename.lower().endswith(".json"):
                try:
                    os.remove(os.path.join(archive_dir, filename))
                    removed_files += 1
                except OSError:
                    pass

    return True, f"Deleted 1 student submission and {removed_files} archived file(s)."


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


def _ann_features_from_subject(item):
    """Return the five ANN input features used by EduPredict SPP."""
    return [
        float(item.get("Attendance_Mark", attendance_mark(item.get("Attendance", 0)))),
        float(item.get("Internal", 0)),
        float(item.get("Assignment", 0)),
        float(item.get("Previous", 0)),
        float(item.get("Study_Hours", 0)),
    ]


def _ann_level_from_features(row):
    """Project-defined target used only when real labelled records are insufficient."""
    att, internal, assignment, previous, study = row
    score = float(np.mean([
        (att / 5) * 100,
        (min(max(study, 0), 6) / 6) * 100,
        (internal / 40) * 100,
        (assignment / 15) * 100,
        (previous / 60) * 100,
    ]))
    if score >= 80:
        return "Good"
    if score >= 65:
        return "Above Average"
    if score >= 50:
        return "Average"
    return "Needs Improvement"


def train_ann_model():
    """Train the ANN from stored labelled records, with a project-defined seed dataset for first use."""
    if not ANN_AVAILABLE:
        return {"available": False, "message": "scikit-learn is not installed. Run: pip install scikit-learn"}

    X_real, y_real = [], []
    df = load_reports()
    if not df.empty:
        for _, report in df.iterrows():
            for item in parse_subjects(report.get("Subjects_JSON", "[]")):
                try:
                    features = _ann_features_from_subject(item)
                    label = str(item.get("Level", "")).strip()
                    if label not in {"Good", "Above Average", "Average", "Needs Improvement"}:
                        label = _ann_level_from_features(features)
                    X_real.append(features)
                    y_real.append(label)
                except (TypeError, ValueError):
                    continue

    # A first-run ANN needs enough examples and at least two classes. The seed
    # data is generated from the same project-defined scoring rubric; once enough
    # real labelled records exist, those records become the ANN training source.
    use_real = len(X_real) >= 20 and len(set(y_real)) >= 2
    if use_real:
        X, y, source = np.asarray(X_real, dtype=float), np.asarray(y_real), "stored student records"
    else:
        rng = np.random.default_rng(42)
        n = 1600
        X = np.column_stack([
            rng.integers(0, 6, n),
            rng.uniform(0, 40, n),
            rng.uniform(0, 15, n),
            rng.uniform(0, 60, n),
            rng.uniform(0, 6, n),
        ])
        y = np.asarray([_ann_level_from_features(row) for row in X])
        source = "project-defined seed training data (used until 20+ labelled records are available)"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.005,
        max_iter=1200,
        random_state=42,
        early_stopping=False,
    )
    try:
        model.fit(X_scaled, y)
        training_accuracy = float(model.score(X_scaled, y) * 100)
    except Exception as exc:
        return {"available": False, "message": f"ANN training failed: {exc}"}

    return {
        "available": True,
        "model": model,
        "scaler": scaler,
        "training_samples": int(len(X)),
        "training_source": source,
        "training_accuracy": training_accuracy,
        "classes": list(model.classes_),
    }


def predict_with_ann(item, model_result=None):
    """Predict the subject performance level using a small feed-forward ANN."""
    result = model_result if model_result is not None else train_ann_model()
    if not result.get("available"):
        return "", 0.0, result.get("message", "ANN unavailable")
    features = np.asarray([_ann_features_from_subject(item)], dtype=float)
    scaled = result["scaler"].transform(features)
    prediction = str(result["model"].predict(scaled)[0])
    probabilities = result["model"].predict_proba(scaled)[0]
    confidence = float(np.max(probabilities) * 100)
    return prediction, confidence, result


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
    # The newest submission is the active student record, while older
    # submissions remain permanently stored for history/analysis.
    if "Created_Time" in found.columns:
        found = found.sort_values("Created_Time")
    return found.iloc[-1].to_dict()

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
        if x.get("ANN_Prediction"):
            story.append(Paragraph(
                f"<b>ANN Prediction:</b> {x.get('ANN_Prediction')} &nbsp;&nbsp; "
                f"<b>ANN Confidence:</b> {float(x.get('ANN_Confidence', 0)):.1f}%",
                small
            ))
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


def all_department_kmeans_analysis():
    """Run one K-Means analysis across subject records from every department."""
    if not SKLEARN_AVAILABLE:
        return {"available": False, "message": "scikit-learn is not installed. Run: pip install scikit-learn"}

    df = load_reports()
    if df.empty:
        return {"available": False, "message": "No student records are available for K-Means analysis."}

    records = []
    for _, row in df.iterrows():
        for item in parse_subjects(row.get("Subjects_JSON", "[]")):
            try:
                records.append({
                    "Department": str(row.get("Department", "Unknown")),
                    "Semester": str(row.get("Semester", "Unknown")),
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

    k = min(4, len(raw))
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(raw[features].to_numpy(dtype=float))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        raw["Cluster"] = model.fit_predict(X_scaled) + 1
    except Exception as exc:
        return {"available": False, "message": f"K-Means could not be calculated: {exc}"}

    # Give cluster numbers a stable performance interpretation without treating
    # the clusters as official grades.
    means = raw.groupby("Cluster")["Overall"].mean().sort_values()
    rank_by_cluster = {int(cluster): rank for rank, cluster in enumerate(means.index)}
    labels = {
        0: "Low-performance group",
        1: "Average-performance group",
        2: "Above-average group",
        3: "Good-performance group",
    }

    summary = []
    for cluster in sorted(raw["Cluster"].unique()):
        group = raw[raw["Cluster"] == cluster]
        summary.append({
            "Cluster": int(cluster),
            "Records": int(len(group)),
            "Students": int(group["University_ID"].nunique()),
            "Departments": int(group["Department"].nunique()),
            "Average_Overall": float(group["Overall"].mean()),
            "Interpretation": labels.get(rank_by_cluster[int(cluster)], "Performance group"),
        })

    original_centroids = scaler.inverse_transform(model.cluster_centers_)
    centroids = []
    for cluster_zero in range(k):
        cluster_number = cluster_zero + 1
        centroids.append({
            "Cluster": cluster_number,
            "Attendance_Mark": float(original_centroids[cluster_zero, 0]),
            "Internal": float(original_centroids[cluster_zero, 1]),
            "Assignment": float(original_centroids[cluster_zero, 2]),
            "Previous": float(original_centroids[cluster_zero, 3]),
            "Study_Hours": float(original_centroids[cluster_zero, 4]),
            "Interpretation": labels.get(rank_by_cluster.get(cluster_number, 0), "Performance group"),
        })

    department_cluster = (
        raw.groupby(["Department", "Cluster"], as_index=False)
        .agg(Records=("University_ID", "size"), Students=("University_ID", "nunique"),
             Average_Overall=("Overall", "mean"))
        .sort_values(["Department", "Cluster"])
    )

    return {
        "available": True,
        "n_samples": len(raw),
        "n_students": int(raw["University_ID"].nunique()),
        "n_departments": int(raw["Department"].nunique()),
        "k": k,
        "summary": summary,
        "centroids": centroids,
        "department_cluster": department_cluster,
        "records": raw,
        "features": features,
        "inertia": float(model.inertia_),
    }


def create_all_department_kmeans_pdf(result):
    """Create one PDF containing the complete all-department K-Means analysis."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("all_k_title", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, spaceAfter=10)
    small = ParagraphStyle("all_k_small", parent=styles["Normal"], fontSize=7.5, leading=9)
    story = [Paragraph("EduPredict SPP — All Department K-Means Clustering Analysis", title)]

    if not result.get("available"):
        story.append(Paragraph(result.get("message", "K-Means analysis unavailable."), styles["Normal"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    story.append(Paragraph(
        f"<b>Students:</b> {result['n_students']} &nbsp;&nbsp; "
        f"<b>Departments:</b> {result['n_departments']} &nbsp;&nbsp; "
        f"<b>Subject records:</b> {result['n_samples']} &nbsp;&nbsp; "
        f"<b>K:</b> {result['k']} &nbsp;&nbsp; <b>Inertia:</b> {result['inertia']:.4f}", small))
    story.append(Paragraph("Features: Attendance Mark, Internal, Assignment, Previous Mark and Study Hours.", small))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Overall Cluster Summary", styles["Heading2"]))
    data = [["Cluster", "Records", "Students", "Departments", "Avg Overall", "Interpretation"]]
    for r in result["summary"]:
        data.append([str(r["Cluster"]), str(r["Records"]), str(r["Students"]), str(r["Departments"]), f"{r['Average_Overall']:.2f}%", r["Interpretation"]])
    t = Table(data, repeatRows=1, colWidths=[45,48,48,58,65,170])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#172554")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 6.8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Department-wise Cluster Table", styles["Heading2"]))
    data = [["Department", "Cluster", "Records", "Students", "Avg Overall"]]
    for _, r in result["department_cluster"].iterrows():
        data.append([str(r["Department"]), str(int(r["Cluster"])), str(int(r["Records"])), str(int(r["Students"])), f"{r['Average_Overall']:.2f}%"])
    t = Table(data, repeatRows=1, colWidths=[245,55,55,55,65])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#334155")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 6.7),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. K-Means Centroids", styles["Heading2"]))
    data = [["Cluster", "Attendance", "Internal", "Assignment", "Previous", "Study Hrs", "Interpretation"]]
    for r in result["centroids"]:
        data.append([str(r["Cluster"]), f"{r['Attendance_Mark']:.2f}", f"{r['Internal']:.2f}", f"{r['Assignment']:.2f}", f"{r['Previous']:.2f}", f"{r['Study_Hours']:.2f}", r["Interpretation"]])
    t = Table(data, repeatRows=1, colWidths=[42,58,50,58,50,52,155])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#475569")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 6.3),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Student/Subject Cluster Assignment", styles["Heading2"]))
    data = [["Department", "Sem", "University ID", "Student", "Subject", "Overall", "Cluster"]]
    for _, r in result["records"].sort_values(["Department", "Semester", "Student_Name", "Subject"]).iterrows():
        data.append([str(r["Department"]), str(r["Semester"]), str(r["University_ID"]), str(r["Student_Name"]), str(r["Subject"]), f"{r['Overall']:.2f}%", str(int(r["Cluster"]))])
    t = Table(data, repeatRows=1, colWidths=[105,30,65,85,105,45,42])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 5.7),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "K-Means is used here as an analytical clustering method. Cluster labels are project-defined interpretations and do not represent official university grades.",
        styles["Italic"]
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def all_department_dashboard():
    app_brand()
    st.title("🌐 All Department K-Means Analysis Dashboard")
    st.caption("One combined K-Means clustering analysis using subject records from every department and semester.")

    c1, c2 = st.columns([1, 1])
    if c1.button("⬅️ Back to Tutor Dashboard", use_container_width=True):
        st.session_state.dashboard_view = "department"
        st.session_state.page = "teacher_dashboard"
        st.rerun()
    if c2.button("🚪 Logout", use_container_width=True):
        logout()

    result = all_department_kmeans_analysis()
    if not result.get("available"):
        st.info(result.get("message", "No data available for analysis."))
        return

    a, b, c, d = st.columns(4)
    a.metric("Students", result["n_students"])
    b.metric("Departments", result["n_departments"])
    c.metric("Subject Records", result["n_samples"])
    d.metric("Clusters (K)", result["k"])

    st.subheader("📊 All Department K-Means Cluster Summary")
    summary_df = pd.DataFrame(result["summary"]).rename(columns={
        "Cluster": "Cluster", "Records": "Records", "Students": "Students",
        "Departments": "Departments", "Average_Overall": "Average Overall", "Interpretation": "Interpretation"
    })
    summary_df["Average Overall"] = summary_df["Average Overall"].map(lambda x: f"{x:.2f}%")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("🏫 Department-wise K-Means Clustering Table")
    dept_df = result["department_cluster"].copy()
    dept_df["Average_Overall"] = dept_df["Average_Overall"].map(lambda x: f"{x:.2f}%")
    st.dataframe(dept_df, use_container_width=True, hide_index=True)

    st.subheader("🎯 K-Means Centroids")
    st.dataframe(pd.DataFrame(result["centroids"]), use_container_width=True, hide_index=True)

    st.subheader("👥 Student / Subject Cluster Assignment")
    assignment_df = result["records"].copy()
    st.dataframe(
        assignment_df[["Department", "Semester", "University_ID", "Student_Name", "Subject", "Overall", "Cluster"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"K-Means inertia: {result['inertia']:.4f}. Clustering uses standardized attendance, internal, assignment, previous-mark and study-hour features.")

    pdf = create_all_department_kmeans_pdf(result)
    st.download_button(
        "📥 Download Complete All-Department K-Means PDF",
        pdf,
        "All_Department_KMeans_Clustering_Analysis.pdf",
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
# SMART CARD
# ============================================================
SMART_CARD_COLUMNS = [
    "Registration_ID", "Student_Name", "DOB", "Blood_Group", "Address",
    "PIN_Code", "Studied_College", "Department", "Semester", "CGPA",
    "University_Name", "University_ID", "Submitted_Time"
]

# University list used by the Smart Card form.  "Other / Not Listed" lets
# students enter a university that is not in the list.
UNIVERSITY_OPTIONS = [
    "APJ Abdul Kalam Technological University (KTU)",
    "University of Kerala",
    "Mahatma Gandhi University (MGU)",
    "University of Calicut",
    "Kannur University",
    "Cochin University of Science and Technology (CUSAT)",
    "Kerala University of Fisheries and Ocean Studies (KUFOS)",
    "Kerala Agricultural University (KAU)",
    "National University of Advanced Legal Studies (NUALS)",
    "Indian Institute of Technology Palakkad (IIT Palakkad)",
    "Indian Institute of Space Science and Technology (IIST)",
    "Amrita Vishwa Vidyapeetham",
    "Rajagiri School of Engineering & Technology",
    "Federal Institute of Science and Technology (FISAT)",
    "SCMS School of Engineering and Technology",
    "Other / Not Listed",
]

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]


def load_smart_cards():
    if not os.path.exists(SMART_CARD_FILE):
        return pd.DataFrame(columns=SMART_CARD_COLUMNS)
    try:
        df = pd.read_csv(SMART_CARD_FILE, dtype=str).fillna("")
        for col in SMART_CARD_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[SMART_CARD_COLUMNS]
    except Exception:
        # A damaged/empty CSV must not crash the Student site.
        return pd.DataFrame(columns=SMART_CARD_COLUMNS)


def save_smart_card(row):
    """Insert or update a Smart Card record safely."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_smart_cards()
    row = {col: str(row.get(col, "")) for col in SMART_CARD_COLUMNS}
    reg_id = row["Registration_ID"].strip().upper()
    uid = row["University_ID"].strip().upper()

    if not df.empty:
        if reg_id:
            df = df[df["Registration_ID"].astype(str).str.strip().str.upper() != reg_id]
        if uid:
            df = df[df["University_ID"].astype(str).str.strip().str.upper() != uid]

    df = pd.concat([df, pd.DataFrame([row], columns=SMART_CARD_COLUMNS)], ignore_index=True)
    df.to_csv(SMART_CARD_FILE, index=False)


def _font(size, bold=False):
    if not PIL_AVAILABLE:
        return ImageFont.load_default()
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _fit_text(draw, text, font, max_width):
    """Shorten text safely so long university/college names never overflow."""
    text = str(text or "").strip()
    if not text:
        return "—"
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    while len(text) > 3 and draw.textbbox((0, 0), text + "…", font=font)[2] > max_width:
        text = text[:-1]
    return text + "…"


def create_student_card_png(profile):
    """Create a polished, self-contained ATM-style Smart Card PNG."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required. Add Pillow to requirements.txt.")

    W, H = 1200, 760
    img = Image.new("RGB", (W, H), (8, 15, 35))
    draw = ImageDraw.Draw(img)

    # Multi-stop premium gradient background.
    stops = [
        (0, (10, 28, 70)),
        (W // 2, (30, 58, 125)),
        (W, (76, 30, 115)),
    ]
    for x in range(W):
        for i in range(len(stops) - 1):
            if stops[i][0] <= x <= stops[i + 1][0]:
                x0, c0 = stops[i]; x1, c1 = stops[i + 1]
                t = (x - x0) / max(1, x1 - x0)
                c = tuple(int(c0[j] + (c1[j] - c0[j]) * t) for j in range(3))
                break
        draw.line((x, 0, x, H), fill=c)

    # Soft decorative glow circles.
    for cx, cy, r, fill in [
        (1020, 95, 220, (90, 210, 255)),
        (110, 680, 190, (120, 70, 255)),
        (650, 360, 260, (255, 255, 255)),
    ]:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for rr in range(r, 5, -8):
            alpha = max(0, int(2.0 * (r - rr)))
            ld.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(*fill, min(35, alpha)))
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        draw = ImageDraw.Draw(img)

    # Outer card and inner glass panel.
    draw.rounded_rectangle((18, 18, W-18, H-18), radius=42,
                           fill=(255, 255, 255), outline=(255, 255, 255), width=3)
    draw.rounded_rectangle((25, 25, W-25, H-25), radius=38,
                           fill=None, outline=(120, 220, 255), width=2)

    f_title = _font(38, True)
    f_small = _font(22)
    f_bold = _font(28, True)
    f_name = _font(34, True)
    f_value = _font(25)
    f_tiny = _font(18)

    draw.text((62, 52), "EduPredict SPP", font=f_title, fill=(255, 255, 255))
    draw.text((W-365, 61), "STUDENT SMART CARD", font=f_small, fill=(220, 245, 255))
    draw.text((W-365, 92), "ACADEMIC IDENTITY • 2026", font=f_tiny, fill=(180, 220, 245))

    # Contactless-style symbol.
    cx, cy = 1040, 185
    for off in (0, 18, 36):
        draw.arc((cx-55+off, cy-55+off, cx+55-off, cy+55-off), 210, 330,
                 fill=(225, 250, 255), width=5)
    draw.ellipse((cx-8, cy-8, cx+8, cy+8), fill=(225, 250, 255))

    # Smart chip.
    chip = (65, 150, 235, 275)
    draw.rounded_rectangle(chip, radius=18, fill=(229, 194, 93), outline=(255, 245, 190), width=3)
    for yy in (180, 212, 244):
        draw.line((78, yy, 222, yy), fill=(150, 115, 40), width=3)
    draw.line((145, 160, 145, 265), fill=(150, 115, 40), width=3)
    draw.line((78, 228, 222, 228), fill=(150, 115, 40), width=3)

    # Student identity.
    name = _fit_text(draw, profile.get("Student_Name", ""), f_name, 660)
    uid = _fit_text(draw, profile.get("University_ID", "") or "Not Provided", f_value, 660)
    draw.text((285, 145), name, font=f_name, fill=(255, 255, 255))
    draw.text((285, 197), f"University ID  •  {uid}", font=f_value, fill=(225, 242, 255))

    # Information panels.
    dept = _fit_text(draw, profile.get("Department", ""), f_value, 480)
    college = _fit_text(draw, profile.get("Studied_College", ""), f_value, 480)
    university = _fit_text(draw, profile.get("University_Name", ""), f_value, 480)
    semester = str(profile.get("Semester", "") or "—")
    reg = _fit_text(draw, profile.get("Registration_ID", ""), f_value, 480)

    cards = [
        (55, 325, "DEPARTMENT", dept),
        (620, 325, "SEMESTER", semester),
        (55, 420, "COLLEGE", college),
        (620, 420, "REGISTRATION ID", reg),
        (55, 515, "UNIVERSITY", university),
        (620, 515, "BLOOD GROUP", str(profile.get("Blood_Group", "—"))),
    ]
    for x, y, label, value in cards:
        draw.rounded_rectangle((x, y, x+525, y+78), radius=16,
                               fill=(255, 255, 255), outline=(150, 220, 255), width=2)
        draw.text((x+18, y+9), label, font=f_tiny, fill=(75, 105, 145))
        draw.text((x+18, y+36), value, font=f_value, fill=(18, 35, 65))

    # Footer / authenticity line.
    draw.text((55, 650), "EduPredict SPP  •  Student Performance Prediction", font=f_small, fill=(230, 245, 255))
    draw.text((W-355, 650), "SMART ID • VALID RECORD", font=f_tiny, fill=(190, 225, 245))
    draw.line((55, 705, W-55, 705), fill=(150, 220, 255), width=2)
    draw.text((55, 715), "Keep this card for academic identification", font=f_tiny, fill=(205, 230, 250))

    return img


def _clear_card_preview():
    st.session_state.smart_card_preview = None
    st.session_state.smart_card_message = "Smart Card downloaded. Preview removed."


def smart_card_page():
    app_brand()
    st.title("🪪 Student Smart Card")
    st.caption("Create a professional academic Smart Card. Registration details are stored for the Principal portal.")

    if st.session_state.get("smart_card_message"):
        st.success(st.session_state.smart_card_message)
        st.session_state.smart_card_message = ""

    # Never allow a future DOB. The range automatically moves forward each year.
    today = datetime.now().date()
    min_dob = today.replace(year=max(1900, today.year - 100))
    max_dob = today
    default_dob = today.replace(year=max(2000, min(today.year - 18, today.year)))

    with st.form("smart_card_registration_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            reg_id = st.text_input("Registration ID *", placeholder="e.g. REG2026CE001")
            name = st.text_input("Student Name *")
            dob = st.date_input("Date of Birth *", value=default_dob, min_value=min_dob, max_value=max_dob)
            blood = st.selectbox("Blood Group *", BLOOD_GROUPS)
            address = st.text_area("Address *", height=90)
            pin = st.text_input("PIN Code *", max_chars=6, placeholder="6-digit PIN")
        with c2:
            college = st.text_input("Studied College *")
            department = st.selectbox("Department *", DEPARTMENTS)
            semester = st.selectbox("Semester *", SEMESTERS)
            cgpa = st.number_input("CGPA (optional)", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
            university_choice = st.selectbox("University *", UNIVERSITY_OPTIONS)
            other_university = ""
            if university_choice == "Other / Not Listed":
                other_university = st.text_input("Enter University Name *")
            uid = st.text_input("University ID (optional)", placeholder="e.g. KTU24CS001")

        submitted = st.form_submit_button("✨ Create & Register Smart Card", type="primary", use_container_width=True)

    if submitted:
        university = other_university.strip() if university_choice == "Other / Not Listed" else university_choice
        errors = []
        if not reg_id.strip():
            errors.append("Registration ID is required.")
        elif not re.fullmatch(r"[A-Za-z0-9_-]{4,30}", reg_id.strip()):
            errors.append("Registration ID must be 4–30 characters using letters, numbers, _ or - only.")
        if not name.strip(): errors.append("Student Name is required.")
        if not address.strip(): errors.append("Address is required.")
        if not re.fullmatch(r"\d{6}", pin.strip()): errors.append("PIN Code must contain exactly 6 digits.")
        if not college.strip(): errors.append("Studied College is required.")
        if not university: errors.append("University is required.")
        if dob > today: errors.append("Date of Birth cannot be in the future.")
        if not (min_dob <= dob <= max_dob): errors.append("Please select a valid Date of Birth.")

        if errors:
            for err in errors:
                st.error("❌ " + err)
            return

        row = {
            "Registration_ID": reg_id.strip(),
            "Student_Name": name.strip(),
            "DOB": dob.strftime("%Y-%m-%d"),
            "Blood_Group": blood,
            "Address": address.strip(),
            "PIN_Code": pin.strip(),
            "Studied_College": college.strip(),
            "Department": department,
            "Semester": semester,
            "CGPA": f"{cgpa:.2f}" if cgpa else "",
            "University_Name": university,
            "University_ID": uid.strip(),
            "Submitted_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            save_smart_card(row)
            img = create_student_card_png(row)
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", reg_id.strip()) or "student_card"
            path = os.path.join(SMART_CARD_DIR, safe_id + ".png")
            img.save(path, "PNG", optimize=True)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            st.session_state.smart_card_preview = buf.getvalue()
            st.session_state.smart_card_message = "Smart Card created and registered successfully."
            st.rerun()
        except Exception as exc:
            st.error("❌ Smart Card could not be generated.")
            st.exception(exc)

    card_data = st.session_state.get("smart_card_preview")
    if card_data:
        st.divider()
        st.subheader("✨ Your Smart Card")
        st.image(card_data, caption="EduPredict SPP Academic Smart Card", use_container_width=True)
        st.download_button(
            "📥 Download Smart Card (PNG)",
            data=card_data,
            file_name="EduPredict_Student_Smart_Card.png",
            mime="image/png",
            use_container_width=True,
            on_click=_clear_card_preview,
        )
        st.caption("After downloading, the preview is automatically removed from the page.")

    if st.button("⬅️ Back", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ============================================================
# LOGIN / PAGES
# ============================================================
def logout():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value
    st.rerun()


def home_page():
    app_brand()
    if not st.session_state.get("intro_seen", False):
        st.markdown("""
        <div class="intro3d">
          <div class="intro-glow"></div>
          <div class="intro-logo3d">🎓</div>
          <div class="intro-title">EduPredict SPP</div>
          <div class="intro-subtitle">3D Student Performance Prediction • KTU B.Tech</div>
          <div class="intro-cube"><span>ANN</span><span>K-MEANS</span><span>KTU</span></div>
          <div class="intro-loading">INITIALIZING PERFORMANCE ENGINE • • •</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Enter EduPredict SPP", use_container_width=True, type="primary"):
            st.session_state.intro_seen = True
            st.rerun()
        st.caption("The 3D introduction will fade automatically; click Enter to continue immediately.")
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
    st.write("")
    if st.button("🪪 Student Smart Card Registration", use_container_width=True):
        st.session_state.page = "smart_card"; st.rerun()


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
            st.session_state.teacher_menu_view = "menu"
            log_action("Tutor", username.strip(), "Tutor Login", "", account["department"], "", "Successful tutor login")
            st.session_state.page = "teacher_portal"
            st.rerun()
        else:
            st.error("Invalid tutor username or password.")


def student_login():
    login_shell("Student Login", "Use the Username and University ID registered by your tutor")
    st.info("💡 Example: Username **Aromal kv**  •  University ID **SNM25CE001**  •  🔓 No password required")
    with st.form("student_login_form"):
        username = st.text_input("👤 Username", placeholder="e.g. Aromal kv")
        uid = st.text_input("🪪 University ID", placeholder="e.g. SNM25CE001")
        c1, c2 = st.columns(2)
        login = c1.form_submit_button("🎓 Enter Student Portal", use_container_width=True, type="primary")
        back = c2.form_submit_button("← Back", use_container_width=True)
    if back:
        st.session_state.page = "home"; st.rerun()
    if login:
        registered = find_registered_credentials(username, uid)
        if not registered:
            st.error("❌ No tutor-registered profile found. Check your Username and University ID, or contact your tutor.")
            return
        latest_report = find_student(registered["Username"], registered["University_ID"])
        st.session_state.logged_in = True
        st.session_state.role = "student"
        st.session_state.username = str(registered["Username"]).strip()
        st.session_state.student_profile = registered
        st.session_state.student_report = latest_report
        st.session_state.student_result_calculated = False
        log_action("Student", st.session_state.username, "Student Login", registered["University_ID"], registered.get("Department", ""), registered.get("Semester", ""), "Student portal login")
        st.session_state.page = "student_dashboard"
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
def tutor_portal():
    app_brand()
    assigned_department = st.session_state.get("teacher_department")
    st.title("👨‍🏫 Tutor Control Center")
    st.caption(f"Logged in as: **{st.session_state.get('username', '')}**  •  🏫 Department: **{assigned_department}**")
    st.info("🔐 Tutor permissions are department-aware. You can register students in your assigned department and enter marks only for registered students in that department + semester.")
    st.markdown("""
    <div class="glass" style="margin:18px 0;padding:18px">
      <div style="font-size:1.35rem;font-weight:800">🚀 Smart Tutor Workflow</div>
      <div style="color:#cbd5e1;margin-top:8px">🪪 Register → 🏫 Department → 📚 Semester → 📝 Marks → 🧠 ANN → 📊 Result</div>
      <div style="color:#94a3b8;margin-top:6px">🔒 Department + semester permission • 💾 Persistent records • ⚡ Live student portal update</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧑‍🎓 Student Registration")
        st.write("Create the student's official portal profile with department + semester.")
        if st.button("🪪 Open Student Registration", use_container_width=True, type="primary"):
            st.session_state.teacher_menu_view = "registration"; st.rerun()
    with c2:
        st.markdown("### 📝 Student Mark Entry")
        st.write("Enter marks only for students already registered in your department + selected semester.")
        if st.button("📊 Open Student Mark Entry", use_container_width=True, type="primary"):
            st.session_state.teacher_menu_view = "marks"; st.session_state.page = "teacher_dashboard"; st.rerun()

    st.divider()
    if st.session_state.get("teacher_menu_view") == "registration":
        st.subheader("🧑‍🎓 Register Student Profile")
        st.success(f"🏫 Tutor department locked to: **{assigned_department}**")
        st.caption("✨ Every registration creates a portal identity. Students later sign in with Username + University ID only.")
        with st.form("student_registration_form"):
            username = st.text_input("👤 Student Username", placeholder="e.g. Aromal kv")
            uid = st.text_input("🪪 University ID", placeholder="e.g. SNM25CE001")
            name = st.text_input("📛 Student Name", placeholder="e.g. Aromal K V")
            semester = st.selectbox("📚 Student Semester", SEMESTERS)
            st.caption("🔒 Department is controlled by your tutor account.")
            submitted = st.form_submit_button("🚀 Submit Registration", use_container_width=True, type="primary")
        if submitted:
            ok, msg = register_student(username, uid, name, assigned_department, semester, st.session_state.get("username", ""))
            (st.success if ok else st.error)(msg)
        reg = load_registrations()
        mine = reg[reg["Department"].astype(str).eq(str(assigned_department))] if not reg.empty else reg
        st.subheader(f"📋 Registered Students — {assigned_department} ({len(mine)})")
        if mine.empty: st.info("No students registered in your department yet.")
        else: st.dataframe(mine, use_container_width=True, hide_index=True)

    if st.button("🚪 Logout", use_container_width=True):
        log_action("Tutor", st.session_state.get("username", ""), "Tutor Logout")
        logout()

def manual_add_form(department, semester):
    st.subheader("➕ Add One Student")
    st.caption("Subjects are controlled by the Department + Semester selected at the top of the Tutor Dashboard.")

    subjects = get_subjects(department, semester)
    if not subjects:
        st.error(
            f"No subject mapping is configured for {department} — {semester}. "
            "Add the official KTU subjects to SUBJECTS_BY_DEPARTMENT in this file."
        )
        return

    st.success(f"📚 {len(subjects)} subjects loaded: {department} — {semester}")
    st.dataframe(
        pd.DataFrame({"No.": range(1, len(subjects) + 1), "Subject": subjects}),
        use_container_width=True,
        hide_index=True,
    )

    registered = load_registrations()
    if not registered.empty:
        registered = registered[
            registered["Department"].astype(str).str.strip().eq(str(department).strip()) &
            registered["Semester"].astype(str).str.strip().str.upper().eq(str(semester).strip().upper())
        ].reset_index(drop=True)
    if registered.empty:
        st.warning(f"🔒 No registered students found for **{department} — {semester}**. Only department + semester matched registrations can receive marks.")
        return

    registered_display = registered.apply(
        lambda r: f"{r['University_ID']} — {r['Student_Name']} ({r['Username']})", axis=1
    ).tolist()

    with st.form(f"add_student_{re.sub(r'[^a-zA-Z0-9]', '_', department)}_{semester}"):
        selected_student = st.selectbox("Registered Student (University ID — Name)", registered_display)
        selected_idx = registered_display.index(selected_student)
        reg_row = registered.iloc[selected_idx]
        uid = str(reg_row["University_ID"]).strip()
        username = str(reg_row["Username"]).strip()
        name = str(reg_row["Student_Name"]).strip()
        st.success(f"✅ Registered: **{name}**  •  University ID: **{uid}**")

        st.markdown("### Enter marks for the selected subjects")
        values = []
        for i, subject in enumerate(subjects):
            st.markdown(f"**{i + 1}. {subject}**")
            a, b, c, d = st.columns(4)
            prefix = f"{re.sub(r'[^a-zA-Z0-9]', '_', department)}_{semester}_{i}"
            att = a.number_input("Attendance %", 0.0, 100.0, 75.0, 1.0, key=f"att_{prefix}")
            internal = b.number_input("Internal /40", 0.0, 40.0, 20.0, 1.0, key=f"int_{prefix}")
            assignment = c.number_input("Assignment /15", 0.0, 15.0, 8.0, 1.0, key=f"asg_{prefix}")
            previous = d.number_input("Previous /60", 0.0, 60.0, 30.0, 1.0, key=f"prev_{prefix}")
            values.append(normalize_subject({
                "Subject": subject,
                "Attendance": att,
                "Study_Hours": 0,
                "Internal": internal,
                "Assignment": assignment,
                "Previous": previous,
            }))

        submitted = st.form_submit_button(
            "💾 Submit Student",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        reg_check = find_registered_student(uid)
        if not reg_check:
            st.error("This student is not registered. Register the student first.")
            return
        if str(reg_check.get("Department", "")).strip() != str(department).strip() or str(reg_check.get("Semester", "")).strip().upper() != str(semester).strip().upper():
            st.error("🔒 Permission denied: this student registration does not match the selected Department + Semester.")
            return
        if str(st.session_state.get("teacher_department", "")).strip() != str(department).strip():
            st.error("🔒 Permission denied: tutor can enter marks only for the tutor-assigned department.")
            return
        if len(subjects) != 6:
            st.error("This project requires exactly 6 subjects for the selected Semester + Department.")
            return
        report = make_report(username, name, uid, semester, department, values)
        upsert_report(report)
        log_action("Tutor", st.session_state.get("username", ""), "Student Mark Submission", uid, department, semester, f"Submitted marks for {name}")
        st.success(f"✅ {name} saved under {department} — {semester}.")
        st.session_state["last_added_uid"] = uid.strip()
        st.session_state["last_added_department"] = department
        st.rerun()

def teacher_dashboard():
    app_brand()
    assigned_department = st.session_state.get("teacher_department")
    if assigned_department not in DEPARTMENTS:
        assigned_department = DEPARTMENTS[0]

    st.title("📝 Student Mark Entry")
    st.info(f"Tutor account department: **{assigned_department}**  •  Only registered students can receive marks.")
    cback, clog = st.columns(2)
    if cback.button("← Back to Tutor Control Center", use_container_width=True):
        st.session_state.page = "teacher_portal"
        st.session_state.teacher_menu_view = "menu"
        st.rerun()
    if clog.button("🚪 Logout", use_container_width=True):
        logout()

    # IMPORTANT: these selectors are outside every form/tab. Streamlit reruns
    # immediately when either value changes, so the subject list always follows
    # the selected B.Tech department + semester.
    c1, c2, c3 = st.columns([2.2, 1.0, 0.7])
    department = c1.selectbox(
        "🎓 B.Tech Department — Tutor Permission",
        [assigned_department],
        index=0,
        key="dashboard_department",
    )
    semester = c2.selectbox(
        "📚 Semester",
        SEMESTERS,
        index=2,
        key="dashboard_semester",
    )
    if c3.button("➡️ Next Dashboard", use_container_width=True):
        st.session_state.dashboard_view = "all_departments"
        st.session_state.page = "all_department_analysis"
        st.rerun()
    
    st.caption("Next Dashboard: combined analysis for all departments and semesters.")
    
    subjects = get_subjects(department, semester)
    if len(subjects) != 6:
        st.error(
            f"No complete 6-subject mapping is configured for **{department} — {semester}**. "
            "The Tutor Dashboard will not accept marks until all 6 subjects are mapped."
        )
        return

    st.success(f"📖 Active curriculum: **{department} — {semester}**")
    st.info("🔐 Permanent storage enabled: every submitted student record is saved in the master file and archived in `data/student_records/<Department>/<Semester>/`. Only the authenticated Tutor Dashboard can delete a selected record.")
    st.dataframe(
        pd.DataFrame({"No.": range(1, 7), "Subject": subjects}),
        use_container_width=True, hide_index=True
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "➕ Add Student", "📤 Upload CSV", "👥 Student Records", "📈 K-Means Analysis", "🧠 ANN Analysis"
    ])

    with tab1:
        manual_add_form(department, semester)

    with tab2:
        st.subheader("📤 Upload Students")
        st.write(
            "Each CSV row can specify its own Semester. The app ignores manually supplied subject names and "
            "loads the six subjects from the selected B.Tech Department + that row's Semester."
        )
        template = template_df().to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV Template", template, "student_marks_template.csv", "text/csv")
        uploaded = st.file_uploader("Choose CSV", type=["csv"], key="student_csv_upload")
        if uploaded is not None and st.button("🚀 Submit Uploaded Students", type="primary"):
            try:
                reports = uploaded_to_reports(uploaded, department)
                valid_reports = []
                rejected = []
                for report in reports:
                    reg = find_registered_student(report.get("University_ID", ""))
                    if not reg:
                        rejected.append(str(report.get("University_ID", "")))
                        continue
                    if str(reg.get("Username", "")).strip().lower() != str(report.get("Username", "")).strip().lower() or str(reg.get("Student_Name", "")).strip().lower() != str(report.get("Student_Name", "")).strip().lower():
                        rejected.append(str(report.get("University_ID", "")))
                        continue
                    valid_reports.append(report)
                if valid_reports:
                    for report in valid_reports:
                        upsert_report(report)
                    st.success(f"✅ {len(valid_reports)} registered student record(s) uploaded successfully.")
                if rejected:
                    st.warning("⚠️ These rows were skipped because the student is not registered or the Username/Name does not match: " + ", ".join(rejected))
                if not reports:
                    st.warning("No valid student rows were found in the CSV.")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    with tab3:
        df = load_reports()
        filtered = df[
            df["Department"].astype(str).eq(str(department)) &
            df["Semester"].astype(str).str.upper().eq(str(semester).upper())
        ].copy()
        st.subheader(f"Student Records — {department} — {semester} ({len(filtered)})")
        st.caption(f"🔒 {len(filtered)} permanent stored submission(s) in this Department + Semester. Previous submissions are retained even when the same University ID is submitted again.")
        if filtered.empty:
            st.info("No student records for this Department + Semester yet.")
        else:
            show = filtered[["Username", "Student_Name", "University_ID", "Semester", "Department", "Created_Time"]]
            st.dataframe(show, use_container_width=True, hide_index=True)
            ids = show["University_ID"].astype(str).tolist()
            selected = st.selectbox("Select University ID", ids, key="record_student_id")
            row = filtered[filtered["University_ID"].astype(str).eq(str(selected))].iloc[0].to_dict()
            subs = parse_subjects(row["Subjects_JSON"])
            overall = float(np.mean([x["Overall"] for x in subs])) if subs else 0.0
            st.metric("Selected Student Overall", f"{overall:.2f}%")
            rows = [[x["Subject"], x["Attendance"], x["Internal"], x["Assignment"], x["Previous"], x["Overall"], x["Level"]] for x in subs]
            st.dataframe(pd.DataFrame(rows, columns=["Subject", "Attendance", "Internal", "Assignment", "Previous", "Score", "Performance"]), use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            c1.download_button("📥 Download Student PDF", create_pdf(row), f"{selected}_Progress_Report.pdf", "application/pdf", use_container_width=True)
            with c2:
                st.warning("⚠️ Tutor-only action")
                confirm_delete = st.checkbox(
                    "I confirm I want to delete this stored submission",
                    key=f"confirm_delete_{selected}_{row.get('Created_Time','')}"
                )
                if st.button(
                    "🗑️ Delete Selected Student Record",
                    key=f"delete_record_{selected}_{row.get('Created_Time','')}",
                    type="secondary",
                    use_container_width=True,
                    disabled=not confirm_delete,
                ):
                    ok, message = delete_student_record(row)
                    if ok:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(message)

    with tab4:
        st.subheader(f"📈 K-Means Analysis — {department} — {semester}")
        result = kmeans_analysis(department)
        if not result.get("available"):
            st.warning(result.get("message", "Analysis unavailable."))
        else:
            # Show only the selected semester in the analysis table when possible.
            records = result["records"].copy()
            semester_students = load_reports()
            ids_for_sem = set(semester_students[
                semester_students["Department"].astype(str).eq(str(department)) &
                semester_students["Semester"].astype(str).str.upper().eq(str(semester).upper())
            ]["University_ID"].astype(str))
            selected_records = records[records["University_ID"].astype(str).isin(ids_for_sem)].copy()
            st.metric("Subject records in selected semester", len(selected_records))
            if selected_records.empty:
                st.info("No records for the selected Department + Semester. Add students first.")
            else:
                st.dataframe(selected_records, use_container_width=True, hide_index=True)
            st.caption(f"K-Means is calculated from all available records in the selected department. Inertia: {result['inertia']:.4f}")
            st.markdown("### Cluster summary")
            st.dataframe(pd.DataFrame(result["summary"]), use_container_width=True, hide_index=True)
            st.markdown("### Cluster centroids")
            st.dataframe(pd.DataFrame(result["centroids"]), use_container_width=True, hide_index=True)
            st.download_button(
                "📊 Download K-Means Analysis PDF",
                create_kmeans_pdf(result),
                f"{department.replace(' ', '_')}_KMeans_Analysis.pdf",
                "application/pdf", use_container_width=True, type="primary"
            )

    with tab5:
        st.subheader("🧠 Artificial Neural Network (ANN) Analysis")
        st.write("A feed-forward multilayer perceptron uses attendance, internal, assignment, previous mark and study hours as input features.")
        ann = train_ann_model()
        if not ann.get("available"):
            st.warning(ann.get("message", "ANN analysis unavailable."))
        else:
            a, b, c = st.columns(3)
            a.metric("Training Samples", ann["training_samples"])
            b.metric("Training Accuracy", f'{ann["training_accuracy"]:.1f}%')
            c.metric("Output Classes", len(ann["classes"]))
            st.info(f'🧠 Training source: {ann["training_source"]}')
            st.dataframe(pd.DataFrame({"ANN Output Class": ann["classes"]}), use_container_width=True, hide_index=True)
            st.caption("ANN prediction is a project-level academic indicator. It should support, not replace, tutor evaluation.")

# ============================================================
# STUDENT WORKFLOW
# ============================================================
def student_dashboard():
    app_brand()
    profile = st.session_state.get("student_profile")
    if not profile:
        logout(); return

    # Always refresh the latest tutor-entered record. This means a student who
    # logged in before marks were entered will see the marks as soon as the tutor
    # submits them and the student opens/calculates the result again.
    latest_report = find_student(profile.get("Username", ""), profile.get("University_ID", ""))
    st.session_state.student_report = latest_report

    st.title(f"🎓 Welcome, {profile.get('Student_Name', '')}")
    st.caption(f"University ID: {profile.get('University_ID', '')}")

    if st.button("🚪 Logout", use_container_width=False):
        log_action("Student", st.session_state.get("username", ""), "Student Logout", profile.get("University_ID", ""), profile.get("Department", ""), profile.get("Semester", ""), "Student portal logout")
        logout()

    # --------------------------------------------------------
    # Tutor-registered student profile
    # --------------------------------------------------------
    st.subheader("👤 My Tutor-Registered Profile")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(f'<div class="metric-card"><div class="label">👤 Student Name</div><div class="value" style="font-size:1.15rem">{profile.get("Student_Name", "—")}</div></div>', unsafe_allow_html=True)
    p2.markdown(f'<div class="metric-card"><div class="label">🪪 University ID</div><div class="value" style="font-size:1.15rem">{profile.get("University_ID", "—")}</div></div>', unsafe_allow_html=True)
    p3.markdown(f'<div class="metric-card"><div class="label">🏫 Department</div><div class="value" style="font-size:1.05rem">{profile.get("Department", "—")}</div></div>', unsafe_allow_html=True)
    p4.markdown(f'<div class="metric-card"><div class="label">📚 Semester</div><div class="value">{profile.get("Semester", "—")}</div></div>', unsafe_allow_html=True)
    st.write(f"**Registered on:** {profile.get('Registered_Time', '—')}  •  **Tutor:** {profile.get('Tutor_Username', '—')}")

    report = latest_report
    if not report:
        st.warning("⏳ Your profile is registered by the tutor, but marks have not been entered yet.")
        st.info("The **📊 Calculate My Mark** option will become available after your tutor submits your marks.")
        st.button("📊 Calculate My Mark", disabled=True, use_container_width=True)
        return

    # --------------------------------------------------------
    # Marks are visible only after the tutor has submitted them.
    # --------------------------------------------------------
    subjects = parse_subjects(report.get("Subjects_JSON", "[]"))
    if not subjects:
        st.warning("Your tutor record exists, but no subject marks have been submitted yet.")
        st.button("📊 Calculate My Mark", disabled=True, use_container_width=True)
        return

    st.success(f"✅ Tutor marks received for **{report.get('Department', '—')} — {report.get('Semester', '—')}**.")
    st.info("🔐 Your tutor-entered marks are read-only here. You can calculate your result, but you cannot edit the tutor marks.")

    # Study hours are optional inputs used by the ANN/result analysis. The tutor
    # marks themselves remain unchanged.
    st.subheader("📊 Calculate My Mark")
    st.write("Enter your daily study hours and click the button below to calculate your subject-wise result.")
    study_hours = {}
    with st.form("student_calculate_result_form"):
        for i, item in enumerate(subjects):
            study_hours[item["Subject"]] = st.number_input(
                f"{i+1}. {item['Subject']} — daily study hours",
                min_value=0.0, max_value=6.0, value=float(item.get("Study_Hours", 0) or 0), step=0.5,
                key=f"student_study_{i}_{report.get('Created_Time','')}"
            )
        calculate = st.form_submit_button("📊 Calculate My Mark", use_container_width=True, type="primary")

    if calculate:
        updated = []
        ann_model_result = train_ann_model()
        for item in subjects:
            copy = dict(item)
            copy["Study_Hours"] = study_hours.get(item["Subject"], 0)
            normalized = normalize_subject(copy)
            prediction, confidence, ann_info = predict_with_ann(normalized, ann_model_result)
            normalized["ANN_Prediction"] = prediction
            normalized["ANN_Confidence"] = confidence
            updated.append(normalized)

        # Keep tutor-entered values intact while updating only the student's
        # calculation fields in the current session.
        report_view = dict(report)
        report_view["Subjects_JSON"] = json.dumps(updated)
        st.session_state.student_report = report_view
        subjects = updated
        st.session_state.student_result_calculated = True
        log_action("Student", st.session_state.get("username", ""), "Calculate My Mark", profile.get("University_ID", ""), profile.get("Department", ""), profile.get("Semester", ""), "Student calculated performance")

    # Do not show marks until the student has explicitly clicked Calculate.
    if not st.session_state.get("student_result_calculated", False):
        st.info("👆 Click **Calculate My Mark** above to view your marks and performance result.")
        return

    subjects = parse_subjects(st.session_state.student_report.get("Subjects_JSON", "[]"))
    overall = float(np.mean([x["Overall"] for x in subjects])) if subjects else 0

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
        rows.append([x["Subject"], f"{x['Attendance']:.0f}%", f"{x.get('Attendance_Mark', attendance_mark(x['Attendance']))}/5", f"{x['Internal']:.0f}/40", f"{x['Assignment']:.0f}/15", f"{x['Previous']:.0f}/60", f"{x['Study_Hours']:.1f} h", f"{x['Overall']:.1f}%", f"{x.get('Circle', performance_circle(x['Level']))} {x['Level']}", x.get("ANN_Prediction", "—"), f"{float(x.get('ANN_Confidence', 0)):.1f}%" ])
    st.subheader("📊 My Marks — Subject-wise Result")
    st.dataframe(pd.DataFrame(rows, columns=["Subject", "Attendance", "Att. Mark", "Internal", "Assignment", "Previous", "Study", "Score", "Performance", "ANN Prediction", "ANN Confidence"]), use_container_width=True, hide_index=True)

    st.subheader("🎯 Subject-wise Performance & Improvement")
    for x in subjects:
        circle = x.get('Circle', performance_circle(x['Level']))
        with st.expander(f"{circle} {x['Subject']} — {x['Level']} — {x['Overall']:.1f}%"):
            st.markdown(f"**Tutor-entered marks:** Attendance **{x['Attendance']:.0f}% → {x.get('Attendance_Mark', attendance_mark(x['Attendance']))}/5**, Internal **{x['Internal']:.0f}/40**, Assignment **{x['Assignment']:.0f}/15**, Previous **{x['Previous']:.0f}/60**.  **Study:** {x['Study_Hours']:.1f} h/day.")
            if x.get("ANN_Prediction"):
                st.info(f"🧠 ANN Prediction: **{x['ANN_Prediction']}**  •  Confidence: **{float(x.get('ANN_Confidence', 0)):.1f}%**")
            st.success("💚 " + x.get("Compliment", "Keep progressing!"))
            st.markdown("**How to improve:**")
            for recommendation in x["Recommendations"]:
                st.write("• " + recommendation)

    st.caption("🔴 Needs Improvement  •  🟠 Average  •  🟡 Above Average  •  🟢 Good")

    st.download_button(
        "📥 Download Progress Report PDF",
        create_pdf(st.session_state.student_report),
        f"{profile.get('University_ID', 'Student')}_Progress_Report.pdf",
        "application/pdf",
        use_container_width=True,
        type="primary",
    )

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
elif st.session_state.page == "smart_card":
    smart_card_page()
elif st.session_state.page == "student_login":
    student_login()
elif st.session_state.page == "teacher_portal" and st.session_state.logged_in and st.session_state.role == "teacher":
    tutor_portal()
elif st.session_state.page == "teacher_dashboard" and st.session_state.logged_in and st.session_state.role == "teacher":
    teacher_dashboard()
elif st.session_state.page == "all_department_analysis" and st.session_state.logged_in and st.session_state.role == "teacher":
    all_department_dashboard()
elif st.session_state.page == "student_dashboard" and st.session_state.logged_in and st.session_state.role == "student":
    student_dashboard()
else:
    st.session_state.page = "home"
    st.rerun()
