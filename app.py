import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/student_performance.csv")


data = load_data()

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]

# Check columns
required_columns = FEATURES + ["Performance"]

missing = [
    column for column in required_columns
    if column not in data.columns
]

if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

# -------------------------------------------------
# K-MEANS
# -------------------------------------------------

scaler = StandardScaler()

X = data[FEATURES]

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------

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

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -------------------------------------------------
# LOGIN
# -------------------------------------------------

st.title("🎓 Student Performance Prediction System")

st.write(
    "B.Tech S3 – AI & Data Science"
)

st.divider()

role = st.selectbox(
    "Select Login Type",
    ["Student", "Teacher"]
)

# -------------------------------------------------
# STUDENT DASHBOARD
# -------------------------------------------------

if role == "Student":

    st.header("👨‍🎓 Student Dashboard")

    st.info(
        "Enter your academic information to predict your performance."
    )

    student_id = st.text_input(
        "Student ID",
        placeholder="Example: S101"
    )

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter your name"
    )

    st.subheader("📋 Academic Information")

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        80
    )

    study_hours = st.slider(
        "Study Hours / Day",
        0.0,
        12.0,
        3.0
    )

    internal = st.slider(
        "Internal Mark",
        0,
        100,
        65
    )

    assignment = st.slider(
        "Assignment Score",
        0,
        100,
        70
    )

    previous = st.slider(
        "Previous Mark",
        0,
        100,
        65
    )

    if st.button(
        "🔮 Predict My Performance",
        type="primary",
        width="stretch"
    ):

        if not student_id or not student_name:

            st.warning(
                "Please enter your Student ID and Name."
            )

        else:

            student = pd.DataFrame([{
                "Attendance": attendance,
                "Study_Hours": study_hours,
                "Internal_Mark": internal,
                "Assignment": assignment,
                "Previous_Mark": previous
            }])

            prediction = model.predict(student)[0]

            probabilities = model.predict_proba(student)[0]

            confidence = probabilities.max() * 100

            student_scaled = scaler.transform(student)

            cluster = int(
                kmeans.predict(student_scaled)[0]
            )

            st.success(
                f"🎓 Predicted Performance: {prediction}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.1f}%"
                )

            with col2:

                st.metric(
                    "Student Cluster",
                    cluster
                )

            st.subheader("👤 Student Information")

            st.write(f"**Name:** {student_name}")
            st.write(f"**Student ID:** {student_id}")

            st.subheader("📊 Your Academic Details")

            st.dataframe(
                student,
                width="stretch"
            )

            st.subheader("💡 Recommendation")

            if prediction == "High":

                st.success(
                    "Excellent performance. "
                    "Continue your current study pattern."
                )

            elif prediction == "Medium":

                st.warning(
                    "Your performance is moderate. "
                    "Try improving study hours, attendance "
                    "and assignment performance."
                )

            else:

                st.error(
                    "Additional academic support is recommended. "
                    "Focus on attendance, study hours and assignments."
                )

            st.caption(
                "Your input is used only for this prediction "
                "and is not displayed in the student dashboard dataset."
            )

# -------------------------------------------------
# TEACHER DASHBOARD
# -------------------------------------------------

else:

    st.header("👩‍🏫 Teacher Dashboard")

    st.info(
        "Teacher access is required to view student data and analytics."
    )

    teacher_username = st.text_input(
        "Teacher Username"
    )

    teacher_password = st.text_input(
        "Teacher Password",
        type="password"
    )

    if st.button(
        "🔐 Teacher Login",
        type="primary"
    ):

        # DEMO LOGIN
        if (
            teacher_username == "teacher"
            and teacher_password == "teacher123"
        ):

            st.session_state["teacher_logged_in"] = True

        else:

            st.error(
                "Invalid teacher username or password."
            )

    if st.session_state.get(
        "teacher_logged_in",
        False
    ):

        st.success(
            "Teacher login successful."
        )

        st.divider()

        # Statistics
        st.subheader("📊 Student Statistics")

        total_students = len(data)

        high_count = len(
            data[data["Performance"] == "High"]
        )

        medium_count = len(
            data[data["Performance"] == "Medium"]
        )

        low_count = len(
            data[data["Performance"] == "Low"]
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Students",
                total_students
            )

        with col2:

            st.metric(
                "High",
                high_count
            )

        with col3:

            st.metric(
                "Medium",
                medium_count
            )

        with col4:

            st.metric(
                "Low",
                low_count
            )

        st.divider()

        # Model accuracy
        st.subheader("🤖 Model Performance")

        st.metric(
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.write(
            "K-Means Clusters: **3**"
        )

        st.divider()

        # Dataset
        st.subheader("📋 Student Dataset")

        st.dataframe(
            data,
            width="stretch",
            hide_index=True
        )

        st.divider()

        # Performance distribution
        st.subheader(
            "📈 Performance Distribution"
        )

        performance_counts = (
            data["Performance"]
            .value_counts()
            .rename_axis("Performance")
            .reset_index(name="Students")
        )

        st.bar_chart(
            performance_counts.set_index("Performance")
        )

        if st.button("Logout"):

            st.session_state["teacher_logged_in"] = False

            st.rerun()

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Student Performance Prediction using "
    "K-Means Clustering and Random Forest Classification"
)
