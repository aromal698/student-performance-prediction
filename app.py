import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🎓 Student Performance Prediction")


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
        "Dataset not found. Please make sure the file is located at:\n\n"
        "data/student_performance.csv"
    )
    st.stop()


# =========================================================
# REQUIRED FEATURES
# =========================================================

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Mark",
    "Assignment",
    "Previous_Mark"
]

required_columns = FEATURES + ["Performance"]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        f"Missing columns in dataset: {missing_columns}"
    )
    st.stop()


# =========================================================
# K-MEANS CLUSTERING
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
# RANDOM FOREST CLASSIFICATION
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


# =========================================================
# STUDENT LOGIN
# =========================================================

if not st.session_state.get("student_logged_in", False):

    st.header("🔐 Student Login")

    st.write(
        "Enter your 9-character student password."
    )

    st.info(
        "Password format: BTECH + 4-digit DOB year\n\n"
        "Example: BTECH2003"
    )

    password = st.text_input(
        "Student Password",
        type="password",
        max_chars=9,
        placeholder="BTECH2003"
    )

    login_button = st.button(
        "🔓 Login",
        type="primary",
        width="stretch"
    )

    if login_button:

        password_clean = password.strip().upper()

        # =================================================
        # PASSWORD VALIDATION
        # Exactly 9 characters
        # First 5 = BTECH
        # Last 4 = year from 2000 to 2020
        # =================================================

        valid_password = (
            len(password_clean) == 9
            and password_clean[:5] == "BTECH"
            and password_clean[5:].isdigit()
            and 2000 <= int(password_clean[5:]) <= 2020
        )

        if valid_password:

            st.session_state["student_logged_in"] = True

            st.session_state["dob_year"] = (
                password_clean[5:]
            )

            st.rerun()

        else:

            st.error(
                "❌ Invalid password."
            )

            st.warning(
                "Password must be exactly 9 characters: "
                "BTECH + 4-digit DOB year."
            )

    st.stop()


# =========================================================
# STUDENT DASHBOARD
# =========================================================

st.header("👨‍🎓 Student Dashboard")

st.success("✅ Student Login Successful")


# =========================================================
# STUDENT INFORMATION
# =========================================================

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


# =========================================================
# SIX SUBJECTS
# =========================================================

st.subheader("📚 Subject Details")

st.write(
    "Enter the names of your six subjects."
)


subject_names = []

subject_columns = st.columns(3)


for i in range(6):

    with subject_columns[i % 3]:

        subject_name = st.text_input(
            f"Subject {i + 1}",
            placeholder=f"Enter Subject {i + 1}",
            key=f"subject_name_{i}"
        )

        subject_names.append(subject_name)


st.divider()


# =========================================================
# ACADEMIC INPUTS
# =========================================================

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


    # -----------------------------------------------------
    # COLUMN 1
    # -----------------------------------------------------

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
            key=f"study_hours_{i}"
        )


    # -----------------------------------------------------
    # COLUMN 2
    # -----------------------------------------------------

    with col2:

        internal_mark = st.number_input(
            "Internal Mark",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            key=f"internal_mark_{i}"
        )


        assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            key=f"assignment_{i}"
        )


    # -----------------------------------------------------
    # COLUMN 3
    # -----------------------------------------------------

    with col3:

        previous_mark = st.number_input(
            "Previous Mark",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            key=f"previous_mark_{i}"
        )


    # Store subject information

    subject_inputs.append(
        {
            "Subject": subject,
            "Attendance": attendance,
            "Study_Hours": study_hours,
            "Internal_Mark": internal_mark,
            "Assignment": assignment_score,
            "Previous_Mark": previous_mark
        }
    )


    st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔮 Predict Performance",
    type="primary",
    width="stretch"
):

    results = []


    # =====================================================
    # PREDICT EACH SUBJECT
    # =====================================================

    for subject_data in subject_inputs:

        input_data = pd.DataFrame(
            [
                {
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
                }
            ]
        )


        # -----------------------------------------------
        # RANDOM FOREST PREDICTION
        # -----------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # -----------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )[0]

        confidence = (
            probabilities.max() * 100
        )


        # -----------------------------------------------
        # K-MEANS CLUSTER
        # -----------------------------------------------

        scaled_input = scaler.transform(
            input_data
        )

        cluster = int(
            kmeans.predict(
                scaled_input
            )[0]
        )


        # -----------------------------------------------
        # STORE RESULT
        # -----------------------------------------------

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


    # =====================================================
    # RESULT DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(
        results
    )


    st.success(
        "✅ Prediction completed successfully!"
    )


    # =====================================================
    # SUBJECT-WISE RESULTS
    # =====================================================

    st.subheader(
        "📊 Subject-wise Prediction"
    )


    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True
    )


    # =====================================================
    # INDIVIDUAL RESULTS
    # =====================================================

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


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.subheader(
        "💡 Recommendation"
    )


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


# =========================================================
# LOGOUT
# =========================================================

st.divider()


if st.button(
    "🚪 Logout",
    width="stretch"
):

    st.session_state["student_logged_in"] = False

    if "dob_year" in st.session_state:
        del st.session_state["dob_year"]

    st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Student Performance Prediction System"
)
