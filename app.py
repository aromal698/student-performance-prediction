import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎓 Student Performance Prediction")


# =========================================================
# LOAD DATA
# =========================================================

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


required_columns = FEATURES + ["Performance"]


missing = [
    col for col in required_columns
    if col not in data.columns
]


if missing:

    st.error(
        f"Missing columns in dataset: {missing}"
    )

    st.stop()


# =========================================================
# K-MEANS
# =========================================================

scaler = StandardScaler()

X = data[FEATURES]

X_scaled = scaler.fit_transform(X)


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(
    X_scaled
)


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


model.fit(
    X_train,
    y_train
)


y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


# =========================================================
# STUDENT LOGIN
# =========================================================

st.header("🔐 Student Login")

st.write(
    "Enter your details to access your Student Dashboard."
)


student_name = st.text_input(
    "Student Name"
)


university_id = st.text_input(
    "University ID"
)


dob = st.text_input(
    "Date of Birth",
    placeholder="DDMM"
)


password = st.text_input(
    "Password",
    type="password",
    placeholder="Example: ARU*BTECH1508"
)


# =========================================================
# LOGIN BUTTON
# =========================================================

if st.button(
    "🔓 Student Login",
    type="primary",
    width="stretch"
):

    # -----------------------------------------------
    # CREATE EXPECTED PASSWORD
    # -----------------------------------------------

    name_clean = (
        student_name
        .strip()
        .replace(" ", "")
    )

    if len(name_clean) < 3:

        st.error(
            "Student name must contain at least 3 letters."
        )

        st.stop()


    first_three = name_clean[:3].upper()


    dob_clean = (
        dob
        .strip()
        .replace("/", "")
        .replace("-", "")
    )


    expected_password = (
        first_three
        + "*BTECH"
        + dob_clean
    )


    # -----------------------------------------------
    # CHECK PASSWORD
    # -----------------------------------------------

    if (
        university_id
        and student_name
        and dob
        and password == expected_password
    ):

        st.session_state[
            "student_logged_in"
        ] = True

        st.session_state[
            "student_name"
        ] = student_name

        st.session_state[
            "university_id"
        ] = university_id

        st.session_state[
            "student_dob"
        ] = dob


    else:

        st.error(
            "Invalid Student Name, University ID, "
            "DOB or Password."
        )


# =========================================================
# STUDENT DASHBOARD
# =========================================================

if st.session_state.get(
    "student_logged_in",
    False
):

    st.divider()

    st.header(
        "👨‍🎓 Student Dashboard"
    )


    st.success(
        f"Welcome, {st.session_state['student_name']}!"
    )


    # -----------------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------------

    st.subheader(
        "👤 Student Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"**Student Name:** "
            f"{st.session_state['student_name']}"
        )

        st.write(
            f"**University ID:** "
            f"{st.session_state['university_id']}"
        )


    with col2:

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

    st.subheader(
        "📚 Six Subjects"
    )


    subject_names = []


    subject_columns = st.columns(3)


    for i in range(6):

        with subject_columns[i % 3]:

            subject = st.text_input(
                f"Subject {i + 1}",
                key=f"subject_{i}"
            )

            subject_names.append(
                subject
            )


    st.divider()


    # -----------------------------------------------------
    # SUBJECT ACADEMIC INFORMATION
    # -----------------------------------------------------

    st.subheader(
        "📊 Academic Information"
    )


    st.write(
        "Enter academic information separately for each subject."
    )


    subject_inputs = []


    for i in range(6):

        subject = subject_names[i]

        if not subject:

            subject = f"Subject {i + 1}"


        st.markdown(
            f"### 📘 {subject}"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            attendance = st.number_input(
                "Attendance (%)",
                0.0,
                100.0,
                80.0,
                key=f"attendance_{i}"
            )


            study_hours = st.number_input(
                "Study Hours / Day",
                0.0,
                24.0,
                3.0,
                key=f"study_{i}"
            )


        with col2:

            internal = st.number_input(
                "Internal Mark",
                0.0,
                100.0,
                65.0,
                key=f"internal_{i}"
            )


            assignment = st.number_input(
                "Assignment Score",
                0.0,
                100.0,
                70.0,
                key=f"assignment_{i}"
            )


        with col3:

            previous = st.number_input(
                "Previous Mark",
                0.0,
                100.0,
                65.0,
                key=f"previous_{i}"
            )


        subject_inputs.append(
            {
                "Subject": subject,
                "Attendance": attendance,
                "Study_Hours": study_hours,
                "Internal_Mark": internal,
                "Assignment": assignment,
                "Previous_Mark": previous
            }
        )


        st.divider()


    # -----------------------------------------------------
    # PREDICTION
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


            prediction = model.predict(
                input_data
            )[0]


            probabilities = model.predict_proba(
                input_data
            )[0]


            confidence = (
                probabilities.max() * 100
            )


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
                        round(
                            confidence,
                            1
                        ),

                    "Cluster":
                        cluster
                }
            )


        results_df = pd.DataFrame(
            results
        )


        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.success(
            "✅ Prediction completed!"
        )


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

            if result["Prediction"] == "High":

                st.success(
                    f"📗 {result['Subject']} → "
                    f"High Performance | "
                    f"{result['Confidence (%)']}% confidence"
                )


            elif result["Prediction"] == "Medium":

                st.warning(
                    f"📙 {result['Subject']} → "
                    f"Medium Performance | "
                    f"{result['Confidence (%)']}% confidence"
                )


            else:

                st.error(
                    f"📕 {result['Subject']} → "
                    f"Low Performance | "
                    f"{result['Confidence (%)']}% confidence"
                )


        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

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
                "Focus on: "
                + ", ".join(low_subjects)
            )


        elif medium_subjects:

            st.warning(
                "Consider improving: "
                + ", ".join(medium_subjects)
            )


        else:

            st.success(
                "Excellent performance predicted "
                "for all subjects!"
            )


    # -----------------------------------------------------
    # LOGOUT
    # -----------------------------------------------------

    st.divider()


    if st.button(
        "Logout"
    ):

        st.session_state[
            "student_logged_in"
        ] = False

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Student Performance Prediction System"
)
