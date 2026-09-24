import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import html
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
REPORT_FILE = os.path.join(DATA_DIR, "student_reports.csv")

BRANCHES = [
    "Artificial Intelligence and Data Science", "Artificial Intelligence and Machine Learning",
    "Computer Science and Engineering", "Computer Science and Engineering (AI)",
    "Computer Science and Engineering (Data Science)", "Computer Science and Engineering (Cyber Security)",
    "Information Technology", "Cyber Security", "Data Science", "Computer Engineering", "Software Engineering",
    "Electronics and Communication Engineering", "Electrical and Electronics Engineering",
    "Electronics and Instrumentation Engineering", "Electronics Engineering", "Instrumentation Engineering",
    "Biomedical Engineering", "Mechanical Engineering", "Automobile Engineering", "Aeronautical Engineering",
    "Civil Engineering", "Chemical Engineering", "Food Technology", "Biotechnology", "Industrial Engineering",
    "Production Engineering", "Mechatronics Engineering", "Robotics and Automation", "Environmental Engineering",
    "Marine Engineering", "Metallurgical Engineering", "Mining Engineering", "Textile Engineering", "Agricultural Engineering"
]

SUBJECTS = {
    "S1": ["Mathematics I", "Physics", "Chemistry", "Engineering Graphics", "Programming in C", "Engineering Mechanics", "Basic Electrical Engineering", "Basic Civil Engineering", "Life Skills", "Workshop Practice", "Design and Engineering", "Engineering Drawing"],
    "S2": ["Mathematics II", "Engineering Physics", "Engineering Chemistry", "Python Programming", "Engineering Mechanics", "Basic Electronics", "Professional Communication", "Environmental Science", "Electrical Engineering", "Design and Engineering", "Programming Lab", "Constitution of India"],
    "S3": ["Mathematics III", "Data Structures", "Database Management Systems", "Artificial Intelligence", "Machine Learning", "Computer Organization", "Object Oriented Programming", "Digital Electronics", "Probability and Statistics", "Operating Systems", "Data Structures Lab", "Microprocessors"],
    "S4": ["Mathematics IV", "Operating Systems", "Computer Networks", "Design and Analysis of Algorithms", "Artificial Intelligence", "Machine Learning", "Software Engineering", "Microprocessors", "Web Programming", "Theory of Computation", "Database Lab", "Computer Networks Lab"],
    "S5": ["Compiler Design", "Computer Networks", "Distributed Computing", "Data Mining", "Deep Learning", "Cloud Computing", "Cyber Security", "Software Testing", "Computer Graphics", "Data Analytics", "Elective I", "Mini Project"],
    "S6": ["Big Data Analytics", "Natural Language Processing", "Computer Vision", "Internet of Things", "Cloud Computing", "Information Security", "Data Science", "Mobile Computing", "Data Mining Lab", "Elective II", "Project Phase I", "Seminar"],
    "S7": ["Advanced Machine Learning", "Deep Learning", "Blockchain", "Artificial Intelligence Applications", "Data Analytics", "Elective III", "Elective IV", "Seminar", "Project Phase II", "Professional Elective", "Major Project", "Industrial Training"],
    "S8": ["Project", "Project Viva", "Comprehensive Viva", "Elective V", "Elective VI", "Industrial Training", "Advanced Artificial Intelligence", "Advanced Data Science", "Professional Elective", "Open Elective", "Technical Seminar", "Major Project Viva"]
}

# Demo teacher accounts. Change before deployment.
TEACHERS = {
    "teacher_aids": {"password": "ktutech", "branch": "Artificial Intelligence and Data Science"},
    "teacher_aiml": {"password": "ktutech", "branch": "Artificial Intelligence and Machine Learning"},
    "teacher_cse": {"password": "ktutech", "branch": "Computer Science and Engineering"},
    "teacher_cseai": {"password": "ktutech", "branch": "Computer Science and Engineering (AI)"},
    "teacher_ds": {"password": "ktutech", "branch": "Computer Science and Engineering (Data Science)"},
    "teacher_cyber": {"password": "ktutech", "branch": "Cyber Security"},
    "teacher_it": {"password": "ktutech", "branch": "Information Technology"},
    "teacher_ece": {"password": "ktutech", "branch": "Electronics and Communication Engineering"},
    "teacher_eee": {"password": "ktutech", "branch": "Electrical and Electronics Engineering"},
    "teacher_me": {"password": "ktutech", "branch": "Mechanical Engineering"},
    "teacher_ce": {"password": "ktutech", "branch": "Civil Engineering"},
    "teacher_auto": {"password": "ktutech", "branch": "Automobile Engineering"},
    "teacher_aero": {"password": "ktutech", "branch": "Aeronautical Engineering"},
    "teacher_bio": {"password": "ktutech", "branch": "Biotechnology"},
    "teacher_biomed": {"password": "ktutech", "branch": "Biomedical Engineering"},
    "teacher_robotics": {"password": "ktutech", "branch": "Robotics and Automation"},
    "teacher_mechatronics": {"password": "ktutech", "branch": "Mechatronics Engineering"},
    "teacher_chemical": {"password": "ktutech", "branch": "Chemical Engineering"},
    "teacher_food": {"password": "ktutech", "branch": "Food Technology"},
}

REPORT_COLUMNS = ["Username", "Student_Name", "University_ID", "Semester", "Branch", "Subjects_JSON", "Created_Time"]

# ============================================================
# SESSION STATE
# ============================================================
for key, default in {
    "page": "home", "logged_in": False, "role": None, "username": None,
    "teacher_branch": None, "student_report": None, "last_student_id": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# MODERN ANIMATED UI
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(circle at 15% 10%, rgba(56,189,248,.12), transparent 25%), radial-gradient(circle at 85% 80%, rgba(139,92,246,.13), transparent 28%), #070b17; color: #eef2ff; }
    .block-container { max-width: 1250px; padding-top: 2rem; }
    header[data-testid="stHeader"] { background: transparent; }
    .hero { position:relative; overflow:hidden; border:1px solid rgba(255,255,255,.12); border-radius:28px; padding:45px 40px; background:linear-gradient(135deg, rgba(15,23,42,.95), rgba(30,41,59,.78)); box-shadow:0 25px 80px rgba(0,0,0,.35); }
    .hero h1 { font-size:clamp(2rem,5vw,4rem); margin:0; font-weight:800; letter-spacing:-2px; }
    .hero p { color:#b7c2d9; font-size:1.05rem; max-width:720px; }
    .orb { position:absolute; width:180px; height:180px; border-radius:50%; filter:blur(1px); opacity:.45; background:radial-gradient(circle at 30% 30%, #67e8f9, #2563eb 55%, transparent 70%); animation:float 7s ease-in-out infinite; }
    .orb.o1 { right:5%; top:-45px; } .orb.o2 { left:45%; bottom:-110px; width:230px; height:230px; background:radial-gradient(circle at 30% 30%, #c084fc, #7c3aed 55%, transparent 70%); animation-delay:1.5s; }
    @keyframes float { 0%,100%{transform:translate3d(0,0,0) rotate(0deg)} 50%{transform:translate3d(18px,25px,0) rotate(12deg)} }
    .grid3d { position:absolute; inset:0; opacity:.18; background-image:linear-gradient(rgba(255,255,255,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.12) 1px,transparent 1px); background-size:42px 42px; transform:perspective(500px) rotateX(62deg) scale(1.7); transform-origin:center bottom; animation:gridmove 8s linear infinite; }
    @keyframes gridmove { from{background-position:0 0,0 0} to{background-position:0 84px,84px 0} }
    .login-wrap { min-height:520px; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; border-radius:30px; border:1px solid rgba(255,255,255,.12); background:linear-gradient(135deg, rgba(2,6,23,.96), rgba(15,23,42,.90)); }
    .login-title { text-align:center; position:relative; z-index:2; margin-bottom:10px; }
    .login-title h1 { font-size:2.2rem; margin:0; } .login-title p{color:#94a3b8}
    .glass { background:rgba(15,23,42,.72); border:1px solid rgba(255,255,255,.14); border-radius:22px; padding:24px; backdrop-filter:blur(18px); box-shadow:0 25px 70px rgba(0,0,0,.35); }
    .metric-card { padding:20px; border-radius:20px; border:1px solid rgba(255,255,255,.10); background:linear-gradient(135deg,rgba(30,41,59,.75),rgba(15,23,42,.7)); }
    .metric-card .value { font-size:2rem; font-weight:800; } .metric-card .label{color:#94a3b8}
    .good-pop { padding:25px; border-radius:24px; text-align:center; background:linear-gradient(135deg,rgba(34,197,94,.20),rgba(16,185,129,.08)); border:1px solid rgba(74,222,128,.45); box-shadow:0 0 45px rgba(34,197,94,.18); animation:pop .55s ease-out; }
    .sad-pop { padding:25px; border-radius:24px; text-align:center; background:linear-gradient(135deg,rgba(239,68,68,.18),rgba(127,29,29,.08)); border:1px solid rgba(248,113,113,.42); box-shadow:0 0 45px rgba(239,68,68,.14); animation:pop .55s ease-out; }
    @keyframes pop { from{transform:scale(.82);opacity:0} to{transform:scale(1);opacity:1} }
    .small-muted { color:#94a3b8; font-size:.88rem; }
    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button { border-radius:12px; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ============================================================
# HELPERS
# ============================================================
def attendance_mark(attendance):
    if attendance >= 90: return 5
    if attendance >= 80: return 4
    if attendance >= 70: return 3
    if attendance >= 60: return 2
    if attendance >= 10: return 1
    return 0

def calculate_performance(attendance, study_hours, internal, assignment, previous):
    attendance_score = (attendance_mark(attendance) / 5) * 100
    study_score = min((study_hours / 6) * 100, 100)
    internal_score = (internal / 40) * 100
    assignment_score = (assignment / 15) * 100
    previous_score = (previous / 60) * 100
    overall = float(np.mean([attendance_score, study_score, internal_score, assignment_score, previous_score]))
    if overall < 50: level = "Low Performance"
    elif overall < 65: level = "Average Performance"
    elif overall < 80: level = "Above Average"
    else: level = "Good Performance"
    return level, overall

def get_feedback(attendance, study_hours, internal, assignment, previous, level):
    methods = []
    if attendance < 75: methods.append("Improve attendance by attending classes regularly.")
    if study_hours < 2: methods.append("Increase daily study time gradually.")
    if internal < 20: methods.append("Prepare more consistently for internal examinations.")
    if assignment < 8: methods.append("Complete assignments on time.")
    if previous < 30: methods.append("Revise previous topics and strengthen fundamentals.")
    if not methods:
        if level == "Good Performance": methods = ["Excellent performance! Keep up the good work.", "Maintain your current study habits and consistency."]
        elif level == "Above Average": methods = ["Good progress. Continue regular revision and improve weaker areas."]
        elif level == "Average Performance": methods = ["Maintain regular revision and improve consistency."]
        else: methods = ["Follow a regular study timetable and revise weak topics."]
    return methods

def normalize_subject(subject):
    attendance = float(subject.get("Attendance", 0) or 0)
    study = float(subject.get("Study_Hours", 0) or 0)
    internal = float(subject.get("Internal", 0) or 0)
    assignment = float(subject.get("Assignment", 0) or 0)
    previous = float(subject.get("Previous", 0) or 0)
    level, overall = calculate_performance(attendance, study, internal, assignment, previous)
    subject.update({
        "Attendance": attendance, "Study_Hours": study, "Internal": internal,
        "Assignment": assignment, "Previous": previous, "Level": level, "Overall": overall,
        "Attendance_Mark": attendance_mark(attendance),
        "Recommendations": get_feedback(attendance, study, internal, assignment, previous, level)
    })
    return subject

def load_reports():
    if not os.path.exists(REPORT_FILE): return pd.DataFrame(columns=REPORT_COLUMNS)
    try:
        df = pd.read_csv(REPORT_FILE, dtype=str)
        for c in REPORT_COLUMNS:
            if c not in df.columns: df[c] = ""
        return df[REPORT_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=REPORT_COLUMNS)

def save_reports(df):
    df.to_csv(REPORT_FILE, index=False)

def upsert_report(report):
    reports = load_reports()
    mask = reports["University_ID"].astype(str).eq(str(report["University_ID"]))
    reports = reports.loc[~mask].copy()
    reports = pd.concat([reports, pd.DataFrame([report])], ignore_index=True)
    save_reports(reports)

def find_student(username, university_id):
    df = load_reports()
    if df.empty: return None
    rows = df[(df["Username"].astype(str).str.strip().str.lower() == username.strip().lower()) & (df["University_ID"].astype(str).str.strip() == university_id.strip())]
    return rows.iloc[0].to_dict() if not rows.empty else None

def parse_subjects(value):
    try:
        data = json.loads(value) if isinstance(value, str) else value
        return [normalize_subject(x) for x in data]
    except Exception:
        return []

def make_report(username, name, uid, semester, branch, subjects):
    return {
        "Username": username.strip(), "Student_Name": name.strip(), "University_ID": uid.strip(),
        "Semester": semester, "Branch": branch,
        "Subjects_JSON": json.dumps(subjects, ensure_ascii=False),
        "Created_Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    }

# ============================================================
# MACHINE LEARNING
# ============================================================
def create_training_data():
    rng = np.random.default_rng(42); n = 400
    attendance = rng.uniform(40, 100, n); study = rng.uniform(0, 6, n)
    internal = rng.uniform(5, 40, n); assignment = rng.uniform(2, 15, n); previous = rng.uniform(10, 60, n)
    score = attendance*.25 + (study/6*100)*.15 + (internal/40*100)*.25 + (assignment/15*100)*.15 + (previous/60*100)*.20
    performance = np.where(score < 50, "Low", np.where(score < 70, "Medium", "High"))
    return pd.DataFrame({"Attendance":attendance,"Study_Hours":study,"Internal_Mark":internal,"Assignment":assignment,"Previous_Mark":previous,"Performance":performance})

@st.cache_resource
def train_models():
    df = create_training_data()
    X = df[["Attendance","Study_Hours","Internal_Mark","Assignment","Previous_Mark"]].copy()
    X["Study_Hours"] = X["Study_Hours"]/6*100; X["Internal_Mark"] = X["Internal_Mark"]/40*100
    X["Assignment"] = X["Assignment"]/15*100; X["Previous_Mark"] = X["Previous_Mark"]/60*100
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10); kmeans.fit(Xs)
    rf = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced"); rf.fit(Xs, df["Performance"])
    return rf, kmeans, scaler

def predict_student(s):
    rf, kmeans, scaler = train_models()
    x = np.array([[s["Attendance"], s["Study_Hours"]/6*100, s["Internal"]/40*100, s["Assignment"]/15*100, s["Previous"]/60*100]])
    xs = scaler.transform(x)
    pred = rf.predict(xs)[0]; probs = rf.predict_proba(xs)[0]
    return pred, float(np.max(probs)*100), int(kmeans.predict(xs)[0])

# ============================================================
# PDF
# ============================================================
def create_pdf(student):
    subjects = parse_subjects(student["Subjects_JSON"])
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=28,leftMargin=28,topMargin=28,bottomMargin=28)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, spaceAfter=12)
    head = ParagraphStyle("h", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=7)
    normal = ParagraphStyle("n", parent=styles["Normal"], fontSize=8.5, leading=11)
    story = [Paragraph("STUDENT PERFORMANCE PROGRESS REPORT", title), Paragraph("AI-Based Student Performance Prediction System", normal), Spacer(1,12)]
    info = [["Student Name",student["Student_Name"]],["Username",student["Username"]],["University ID",student["University_ID"]],["Semester",student["Semester"]],["Branch",student["Branch"]],["Report Generated",student["Created_Time"]]]
    t = Table(info,colWidths=[125,385]); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.grey),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#E8EEF7")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9)])); story += [Paragraph("1. Student Details",head),t,Spacer(1,12)]
    overall = float(np.mean([x["Overall"] for x in subjects])) if subjects else 0
    story += [Paragraph("2. Overall Result",head)]
    summary = [["Subjects",str(len(subjects))],["Overall Score",f"{overall:.2f}%"],["Overall Status", "Good Performance" if overall>=80 else "Above Average" if overall>=65 else "Average Performance" if overall>=50 else "Low Performance"]]
    stbl=Table(summary,colWidths=[200,310]); stbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.grey),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF4EA")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold")]))
    story += [stbl,Spacer(1,12),Paragraph("3. Subject-wise Marks",head)]
    data=[["Subject","Attendance","Study","Internal","Assignment","Previous","Score","Performance"]]
    for x in subjects: data.append([x["Subject"],f'{x["Attendance"]:.0f}%',f'{x["Study_Hours"]:.1f}/6',f'{x["Internal"]:.0f}/40',f'{x["Assignment"]:.0f}/15',f'{x["Previous"]:.0f}/60',f'{x["Overall"]:.1f}%',x["Level"]])
    tbl=Table(data,repeatRows=1,colWidths=[105,48,42,55,58,55,45,102]); cmds=[("GRID",(0,0),(-1,-1),.35,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#D9EAF7")),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),6.8)]
    for i,x in enumerate(subjects,1):
        bg={"Low Performance":"#F4CCCC","Average Performance":"#FCE5CD","Above Average":"#FFF2CC","Good Performance":"#D9EAD3"}[x["Level"]]; cmds.append(("BACKGROUND",(0,i),(-1,i),colors.HexColor(bg)))
    tbl.setStyle(TableStyle(cmds)); story += [tbl,Spacer(1,12),Paragraph("4. Recommendations",head)]
    for x in subjects:
        story.append(Paragraph(f'<b>{html.escape(x["Subject"])}</b>: ' + " ".join("• "+html.escape(m) for m in x["Recommendations"]),normal)); story.append(Spacer(1,4))
    story += [Spacer(1,10),Paragraph("Note: ML results are academic monitoring aids and should not replace teacher evaluation.",normal)]
    doc.build(story); buf.seek(0); return buf.getvalue()

# ============================================================
# CSV TEMPLATE / UPLOAD PARSER
# ============================================================
def template_df():
    row={"Username":"student01","Student_Name":"Sample Student","University_ID":"UNI001","Semester":"S3","Branch":"Computer Science and Engineering"}
    for i in range(1,7):
        row.update({f"Subject_{i}":SUBJECTS["S3"][i-1],f"Attendance_{i}":85,f"Study_Hours_{i}":3,f"Internal_{i}":32,f"Assignment_{i}":12,f"Previous_{i}":48})
    return pd.DataFrame([row])

def uploaded_to_reports(uploaded):
    df=pd.read_csv(uploaded)
    required=["Username","Student_Name","University_ID","Semester","Branch"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError("Missing required columns: "+", ".join(missing))
    reports=[]
    for _,r in df.iterrows():
        subs=[]
        for i in range(1,7):
            subject=str(r.get(f"Subject_{i}","")).strip()
            if not subject or subject.lower()=="nan": continue
            x={"Subject":subject,"Attendance":r.get(f"Attendance_{i}",0),"Study_Hours":r.get(f"Study_Hours_{i}",0),"Internal":r.get(f"Internal_{i}",0),"Assignment":r.get(f"Assignment_{i}",0),"Previous":r.get(f"Previous_{i}",0)}
            subs.append(normalize_subject(x)); pred,conf,cluster=predict_student(subs[-1]); subs[-1].update({"ML_Prediction":pred,"Confidence":conf,"Cluster":cluster})
        if not subs: raise ValueError(f"No subjects found for student {r.get('Student_Name','')}")
        reports.append(make_report(str(r["Username"]),str(r["Student_Name"]),str(r["University_ID"]),str(r["Semester"]),str(r["Branch"]),subs))
    return reports

# ============================================================
# UI COMPONENTS
# ============================================================
def logout():
    for k,v in {"logged_in":False,"role":None,"username":None,"teacher_branch":None,"student_report":None,"page":"home"}.items(): st.session_state[k]=v
    st.rerun()

def home_page():
    st.markdown('<div class="hero"><div class="grid3d"></div><div class="orb o1"></div><div class="orb o2"></div><div style="position:relative;z-index:2"><div class="small-muted">AI • ACADEMIC ANALYTICS • PROGRESS TRACKING</div><h1>🎓 Student Performance<br>Prediction System</h1><p>A modern portal where tutors publish marks and students securely view their subject-wise performance and download a progress report.</p></div></div>',unsafe_allow_html=True)
    st.write("")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="glass"><h2>🎓 Student Portal</h2><p class="small-muted">Login with your Username + University ID. View marks, performance level, feedback and PDF progress report.</p></div>',unsafe_allow_html=True)
        if st.button("Open Student Login →",use_container_width=True,type="primary"): st.session_state.page="student_login"; st.rerun()
    with c2:
        st.markdown('<div class="glass"><h2>👨‍🏫 Tutor Portal</h2><p class="small-muted">Upload a CSV of student details and subject marks, or add a student manually. Submitted records become available to students.</p></div>',unsafe_allow_html=True)
        if st.button("Open Tutor Login →",use_container_width=True): st.session_state.page="teacher_login"; st.rerun()

def login_shell(title,subtitle):
    st.markdown(f'<div class="login-wrap"><div class="grid3d"></div><div class="orb o1"></div><div class="orb o2"></div><div style="width:min(560px,90%);position:relative;z-index:2"><div class="login-title"><div style="font-size:3rem">🎓</div><h1>{title}</h1><p>{subtitle}</p></div></div></div>',unsafe_allow_html=True)

def student_login():
    login_shell("Student Login","Use the Username and University ID supplied by your tutor")
    with st.form("student_login_form"):
        username=st.text_input("Username",placeholder="e.g. student01")
        uid=st.text_input("University ID",placeholder="e.g. UNI001")
        c1,c2=st.columns(2); login=c1.form_submit_button("🔐 Login",use_container_width=True,type="primary"); back=c2.form_submit_button("← Back",use_container_width=True)
    if back: st.session_state.page="home"; st.rerun()
    if login:
        student=find_student(username,uid)
        if student:
            st.session_state.logged_in=True; st.session_state.role="student"; st.session_state.username=username; st.session_state.student_report=student; st.session_state.page="student_dashboard"; st.rerun()
        else: st.error("No matching student record found. Check Username and University ID.")

def teacher_login():
    login_shell("Tutor Login","Upload and manage student performance records")
    with st.form("teacher_login_form"):
        username=st.text_input("Tutor Username")
        password=st.text_input("Password",type="password")
        c1,c2=st.columns(2); login=c1.form_submit_button("🔐 Login",use_container_width=True,type="primary"); back=c2.form_submit_button("← Back",use_container_width=True)
    if back: st.session_state.page="home"; st.rerun()
    if login:
        if username in TEACHERS and TEACHERS[username]["password"]==password:
            st.session_state.logged_in=True; st.session_state.role="teacher"; st.session_state.username=username; st.session_state.teacher_branch=TEACHERS[username]["branch"]; st.session_state.page="teacher_dashboard"; st.rerun()
        else: st.error("Invalid tutor username or password.")

def performance_popup(overall):
    if overall >= 80:
        st.balloons()
        st.markdown(f'<div class="good-pop"><div style="font-size:4rem">🎉🏆</div><h2>Excellent Performance!</h2><p>Your overall score is <b>{overall:.2f}%</b>. Keep up the great work!</p></div>',unsafe_allow_html=True)
    elif overall < 50:
        st.markdown(f'<div class="sad-pop"><div style="font-size:4rem">😔📉</div><h2>Needs Improvement</h2><p>Your overall score is <b>{overall:.2f}%</b>. Don’t give up — follow the feedback and improve step by step.</p></div>',unsafe_allow_html=True)

def student_dashboard():
    student=st.session_state.student_report
    if not student: logout(); return
    subjects=parse_subjects(student["Subjects_JSON"]); overall=float(np.mean([x["Overall"] for x in subjects])) if subjects else 0
    c1,c2=st.columns([5,1]); c1.title(f"🎓 Welcome, {student['Student_Name']}"); c2.button("🚪 Logout",on_click=logout,use_container_width=True)
    st.caption(f"Username: {student['Username']}  •  University ID: {student['University_ID']}  •  {student['Semester']}  •  {student['Branch']}")
    performance_popup(overall)
    a,b,c=st.columns(3)
    for col,label,value in [(a,"Overall Score",f"{overall:.2f}%"),(b,"Subjects",str(len(subjects))),(c,"Status","Good" if overall>=80 else "Above Average" if overall>=65 else "Average" if overall>=50 else "Needs Improvement")]:
        col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>',unsafe_allow_html=True)
    st.divider(); st.subheader("📊 Subject-wise Marks & Progress")
    rows=[]
    for x in subjects: rows.append([x["Subject"],f'{x["Attendance"]:.0f}%',f'{x["Internal"]:.0f}/40',f'{x["Assignment"]:.0f}/15',f'{x["Previous"]:.0f}/60',f'{x["Overall"]:.1f}%',x["Level"]])
    st.dataframe(pd.DataFrame(rows,columns=["Subject","Attendance","Internal","Assignment","Previous Mark","Score","Performance"]),use_container_width=True,hide_index=True)
    st.subheader("💡 Subject Feedback")
    for x in subjects:
        with st.expander(f"{x['Subject']} — {x['Level']} — {x['Overall']:.1f}%"):
            st.write(f"Attendance: **{x['Attendance']:.0f}%** | Study: **{x['Study_Hours']:.1f} h/day** | Internal: **{x['Internal']:.0f}/40** | Assignment: **{x['Assignment']:.0f}/15** | Previous: **{x['Previous']:.0f}/60**")
            for m in x["Recommendations"]: st.write("• "+m)
    pdf=create_pdf(student)
    st.download_button("📥 Download Progress Report PDF",data=pdf,file_name=f"{student['University_ID']}_Progress_Report.pdf",mime="application/pdf",use_container_width=True,type="primary")

def manual_add_form():
    st.subheader("➕ Add Student Manually")
    with st.form("add_student"):
        c1,c2,c3=st.columns(3)
        username=c1.text_input("Username"); name=c2.text_input("Student Name"); uid=c3.text_input("University ID")
        c1,c2=st.columns(2); semester=c1.selectbox("Semester",list(SUBJECTS)); branch=c2.selectbox("Branch",BRANCHES,index=BRANCHES.index(st.session_state.teacher_branch) if st.session_state.teacher_branch in BRANCHES else 0)
        selected=st.multiselect("Select 6 subjects",SUBJECTS[semester],default=SUBJECTS[semester][:6])
        values=[]
        for i,sub in enumerate(selected):
            st.markdown(f"**{i+1}. {sub}**"); a,b,c,d,e=st.columns(5)
            att=a.number_input("Attendance %",0.0,100.0,75.0,1.0,key=f"ma{ i}"); study=b.number_input("Study h/day",0.0,6.0,2.0,.5,key=f"ms{ i}"); internal=c.number_input("Internal /40",0.0,40.0,20.0,1.0,key=f"mi{ i}"); assignment=d.number_input("Assignment /15",0.0,15.0,8.0,1.0,key=f"mp{ i}"); previous=e.number_input("Previous /60",0.0,60.0,30.0,1.0,key=f"mv{ i}")
            values.append(normalize_subject({"Subject":sub,"Attendance":att,"Study_Hours":study,"Internal":internal,"Assignment":assignment,"Previous":previous}))
        submitted=st.form_submit_button("💾 Submit Student Marks",use_container_width=True,type="primary")
    if submitted:
        if not username.strip() or not name.strip() or not uid.strip(): st.error("Username, Student Name and University ID are required."); return
        if len(values)!=6: st.error("Select exactly 6 subjects."); return
        for x in values:
            pred,conf,cluster=predict_student(x); x.update({"ML_Prediction":pred,"Confidence":conf,"Cluster":cluster})
        report=make_report(username,name,uid,semester,branch,values); upsert_report(report); st.success("✅ Student details and marks submitted successfully. The student can now login.")

def teacher_dashboard():
    branch=st.session_state.teacher_branch
    c1,c2=st.columns([5,1]); c1.title("👨‍🏫 Tutor Dashboard"); c2.button("🚪 Logout",on_click=logout,use_container_width=True)
    st.info(f"Assigned branch: **{branch}**")
    tab1,tab2,tab3=st.tabs(["📤 Upload CSV","➕ Manual Entry","👥 Student Records"])
    with tab1:
        st.subheader("Upload Student Details + Marks")
        st.write("Upload one row per student. Each row can contain up to 6 subjects.")
        template=template_df().to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV Template",template,"student_marks_template.csv","text/csv")
        file=st.file_uploader("Choose CSV file",type=["csv"])
        if file is not None and st.button("🚀 Submit Uploaded Students",type="primary"):
            try:
                reports=uploaded_to_reports(file)
                # Tutor may only submit to their assigned branch; override branch for consistency.
                for r in reports: r["Branch"]=branch
                for r in reports: upsert_report(r)
                st.success(f"✅ {len(reports)} student record(s) uploaded successfully.")
            except Exception as e: st.error(f"Upload failed: {e}")
    with tab2: manual_add_form()
    with tab3:
        df=load_reports(); branch_df=df[df["Branch"].astype(str).eq(str(branch))].copy()
        st.subheader(f"Student Records — {len(branch_df)}")
        if branch_df.empty: st.info("No student records yet."); return
        show=branch_df[["Username","Student_Name","University_ID","Semester","Created_Time"]].copy(); st.dataframe(show,use_container_width=True,hide_index=True)
        ids=show["University_ID"].astype(str).tolist(); selected=st.selectbox("Select University ID",ids)
        row=branch_df[branch_df["University_ID"].astype(str).eq(str(selected))].iloc[0].to_dict(); subs=parse_subjects(row["Subjects_JSON"]); overall=float(np.mean([x["Overall"] for x in subs]))
        st.metric("Selected Student Overall",f"{overall:.2f}%")
        rows=[[x["Subject"],x["Attendance"],x["Internal"],x["Assignment"],x["Previous"],x["Overall"],x["Level"]] for x in subs]
        st.dataframe(pd.DataFrame(rows,columns=["Subject","Attendance","Internal","Assignment","Previous","Score","Performance"]),use_container_width=True,hide_index=True)
        st.download_button("📥 Download Selected Student PDF",create_pdf(row),f"{selected}_Progress_Report.pdf","application/pdf",use_container_width=True)
        if st.button("🗑️ Delete Selected Student",type="secondary"):
            all_df=load_reports(); all_df=all_df[~((all_df["University_ID"].astype(str)==str(selected))&(all_df["Branch"].astype(str)==str(branch)))]; save_reports(all_df); st.success("Student record deleted."); st.rerun()

# ============================================================
# ROUTER
# ============================================================
if st.session_state.page=="home": home_page()
elif st.session_state.page=="student_login": student_login()
elif st.session_state.page=="teacher_login": teacher_login()
elif st.session_state.page=="student_dashboard" and st.session_state.logged_in and st.session_state.role=="student": student_dashboard()
elif st.session_state.page=="teacher_dashboard" and st.session_state.logged_in and st.session_state.role=="teacher": teacher_dashboard()
else: st.session_state.page="home"; st.rerun()
