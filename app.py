import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/student_performance.csv")


try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found!\n\n"
        "Make sure this file exists:\n"
        "data/student_performance.csv"
    )
    st.stop()


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]

REQUIRED_COLUMNS = FEATURES + ["Performance"]

missing = [
    col for col in REQUIRED_COLUMNS
    if col not in data.columns
]

if missing:
    st.error(f"Missing dataset columns: {missing}")
    st.stop()


# =========================================================
# K-MEANS
# =========================================================

X = data[FEATURES]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X_scaled)


# =========================================================
# RANDOM FOREST
# =========================================================

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


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🎓 Student Performance Prediction")


# =========================================================
# LOGIN TYPE
# =========================================================

if "login_type" not in st.session_state:
    st.session_state.login_type = None

if "student_logged_in" not in st.session_state:
    st.session_state.student_logged_in = False

if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False


# =========================================================
# LOGIN PAGE
# =========================================================

if (
    not st.session_state.student_logged_in
    and not st.session_state.teacher_logged_in
):

    st.subheader("Choose Login")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # STUDENT BUTTON
    # -----------------------------------------------------

    with col1:

        if st.button(
            "🎓 Student",
            width="stretch"
        ):
            st.session_state.login_type = "student"


    # -----------------------------------------------------
    # TEACHER BUTTON
    # -----------------------------------------------------

    with col2:

        if st.button(
            "👨‍🏫 Teacher",
            width="stretch"
        ):
            st.session_state.login_type = "teacher"


    # =====================================================
    # STUDENT LOGIN
    # =====================================================

    if st.session_state.login_type == "student":

        st.divider()

        st.header("🔐 Student Login")

        st.info(
            "Password format: BTECH + 4-digit DOB year\n\n"
            "Example: BTECH2003"
        )

        student_password = st.text_input(
            "Student Password",
            type="password",
            max_chars=9,
            placeholder="BTECH2003"
        )

        if st.button(
            "🔓 Student Login",
            type="primary",
            width="stretch"
        ):

            password = student_password.strip().upper()

            # EXACTLY 9 CHARACTERS
            valid_password = (
                len(password) == 9
                and password[:5] == "BTECH"
                and password[5:].isdigit()
                and 2000 <= int(password[5:]) <= 2020
            )

            if valid_password:

                st.session_state.student_logged_in = True

                st.session_state.dob_year = password[5:]

                st.rerun()

            else:

                st.error(
                    "❌ Invalid password."
                )

                st.warning(
                    "Password must be exactly 9 characters: "
                    "BTECH + 4-digit DOB year."
                )


    # =====================================================
    # TEACHER LOGIN
    # =====================================================

    if st.session_state.login_type == "teacher":

        st.divider()

        st.header("🔐 Teacher Login")

        teacher_password = st.text_input(
            "Teacher Password",
            type="password",
            max_chars=8,
            placeholder="*ktutech"
        )

        if st.button(
            "🔓 Teacher Login",
            type="primary",
            width="stretch"
        ):

            if teacher_password == "*ktutech":

                st.session_state.teacher_logged_in = True

                st.rerun()

            else:

                st.error(
                    "❌ Invalid teacher password."
                )


    st.stop()


# =========================================================
# STUDENT DASHBOARD
# =========================================================

if st.session_state.student_logged_in:

    st.header("🎓 Student Dashboard")

    st.success("Student Login Successful")


    # -----------------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------------

    st.subheader("👤 Student Information")

    col1, col2 = st.columns(2)

    with col1:

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
            ]
        )

    with col2:

        branch = st.selectbox(
            "Branch",
            [
                "AI & Data Science",
                "Computer Science",
                "Information Technology",
                "Electronics & Communication",
                "Electrical & Electronics",
                "Mechanical Engineering",
                "Civil Engineering"
            ]
        )


    st.divider()


    # -----------------------------------------------------
    # SUBJECT NAMES
    # -----------------------------------------------------

    st.subheader("📚 Six Subjects")

    subject_names = []

    columns = st.columns(3)

    for i in range(6):

        with columns[i % 3]:

            subject = st.text_input(
                f"Subject {i + 1}",
                placeholder=f"Enter Subject {i + 1}",
                key=f"subject_{i}"
            )

            subject_names.append(subject)


    st.divider()


    # -----------------------------------------------------
    # SUBJECT INPUTS
    # -----------------------------------------------------

    st.subheader("📊 Academic Information")

    st.write(
        "Enter academic information separately for each subject."
    )

    subject_inputs = []


    for i in range(6):

        subject = subject_names[i].strip()

        if subject == "":
            subject = f"Subject {i + 1}"


        st.markdown(
            f"### 📘 {subject}"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
                key=f"attendance_{i}"
            )

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5,
                key=f"study_{i}"
            )


        with col2:

            internal_mark = st.number_input(
                "Internal Mark",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
                key=f"internal_{i}"
            )

            assignment = st.number_input(
                "Assignment Score",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
                key=f"assignment_{i}"
            )


        with col3:

            previous_mark = st.number_input(
                "Previous Mark",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
                key=f"previous_{i}"
            )


        subject_inputs.append(
            {
                "Subject": subject,
                "Attendance": attendance,
                "Study_Hours": study_hours,
                "Internal_Mark": internal_mark,
                "Assignment": assignment,
                "Previous_Mark": previous_mark
            }
        )


        st.divider()


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Performance",
        type="primary",
        width="stretch"
    ):

        results = []


        for subject_data in subject_inputs:

            input_data = pd.DataFrame(
                [{
                    "Attendance":
                        subject_data["Attendance"],

                    "Study_Hours":
                        subject_data["Study_Hours"],

                    "Internal_Mark":
                        subject_data["Internal_Mark"],

                    "Assignment":
                        subject_data["Assignment"],

                    "Previous_Mark":
                        subject_data["Previous_Mark"]
                }]
            )


            # RANDOM FOREST
            prediction = model.predict(
                input_data
            )[0]


            # CONFIDENCE
            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = (
                probabilities.max() * 100
            )


            # K-MEANS
            scaled_input = scaler.transform(
                input_data
            )

            cluster = int(
                kmeans.predict(
                    scaled_input
                )[0]
            )


            results.append(
                {
                    "Subject":
                        subject_data["Subject"],

                    "Prediction":
                        prediction,

                    "Confidence (%)":
                        round(confidence, 1),

                    "Cluster":
                        cluster
                }
            )


        results_df = pd.DataFrame(results)


        st.success(
            "✅ Prediction completed!"
        )


        # -------------------------------------------------
        # RESULTS TABLE
        # -------------------------------------------------

        st.subheader(
            "📊 Subject-wise Prediction"
        )

        st.dataframe(
            results_df,
            width="stretch",
            hide_index=True
        )


        # -------------------------------------------------
        # INDIVIDUAL RESULTS
        # -------------------------------------------------

        st.subheader(
            "📚 Subject Results"
        )


        for result in results:

            subject = result["Subject"]

            prediction = result["Prediction"]

            confidence = result["Confidence (%)"]


            if prediction == "High":

                st.success(
                    f"📗 {subject} → "
                    f"High Performance | "
                    f"{confidence}% confidence"
                )

            elif prediction == "Medium":

                st.warning(
                    f"📙 {subject} → "
                    f"Medium Performance | "
                    f"{confidence}% confidence"
                )

            else:

                st.error(
                    f"📕 {subject} → "
                    f"Low Performance | "
                    f"{confidence}% confidence"
                )


        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

        st.subheader("💡 Recommendation")

        low_subjects = results_df[
            results_df["Prediction"] == "Low"
        ]["Subject"].tolist()

        medium_subjects = results_df[
            results_df["Prediction"] == "Medium"
        ]["Subject"].tolist()


        if low_subjects:

            st.error(
                "Focus more on: "
                + ", ".join(low_subjects)
            )

        elif medium_subjects:

            st.warning(
                "Consider improving: "
                + ", ".join(medium_subjects)
            )

        else:

            st.success(
                "🎉 Excellent performance predicted "
                "for all subjects!"
            )


    # -----------------------------------------------------
    # STUDENT LOGOUT
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "🚪 Student Logout",
        width="stretch"
    ):

        st.session_state.student_logged_in = False
        st.session_state.login_type = None

        st.rerun()


    st.stop()


# =========================================================
# TEACHER DASHBOARD
# =========================================================

if st.session_state.teacher_logged_in:

    st.header("👨‍🏫 Teacher Dashboard")

    st.success("Teacher Login Successful")


    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Students",
            len(data)
        )

    with col2:
        st.metric(
            "Features",
            len(FEATURES)
        )

    with col3:
        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col4:
        st.metric(
            "Clusters",
            3
        )


    st.divider()


    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    st.subheader("📁 Student Dataset")

    st.dataframe(
        data,
        width="stretch",
        hide_index=True
    )


    st.divider()


    # -----------------------------------------------------
    # PERFORMANCE DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "📈 Performance Distribution"
    )

    performance_counts = (
        data["Performance"]
        .value_counts()
    )

    st.bar_chart(
        performance_counts
    )


    # -----------------------------------------------------
    # CLUSTER DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "🔵 K-Means Cluster Distribution"
    )

    cluster_counts = (
        data["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts
    )


    st.divider()


    # -----------------------------------------------------
    # MODEL ACCURACY
    # -----------------------------------------------------

    st.subheader(
        "🤖 Random Forest Model"
    )

    st.write(
        f"**Model Accuracy:** "
        f"{accuracy * 100:.2f}%"
    )


    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    st.subheader(
        "📋 Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(2),
        width="stretch"
    )


    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader(
        "📌 Feature Importance"
    )

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )


    st.divider()


    # -----------------------------------------------------
    # TEACHER LOGOUT
    # -----------------------------------------------------

    if st.button(
        "🚪 Teacher Logout",
        width="stretch"
    ):

        st.session_state.teacher_logged_in = False
        st.session_state.login_type = None

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Student Performance Prediction System"
)
