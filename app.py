import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


st.title("🎓 Student Performance Prediction")


# -----------------------------
# LOAD DATASET
# -----------------------------

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
    st.error(f"Missing columns in dataset: {missing}")
    st.stop()


# -----------------------------
# K-MEANS
# -----------------------------

scaler = StandardScaler()

X = data[FEATURES]

X_scaled = scaler.fit_transform(X)


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X_scaled)


# -----------------------------
# RANDOM FOREST
# -----------------------------

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


# -----------------------------
# STUDENT LOGIN
# -----------------------------

st.header("🔐 Student Login")

st.write("Enter your student password to continue.")

password = st.text_input(
    "Password",
    type="password",
    placeholder="Example: BTECH2003"
)


if st.button(
    "🔓 Login",
    type="primary",
    width="stretch"
):

    password_clean = password.strip().upper()

    # Password format:
    # BTECH + 4 digit DOB year
    # Example: BTECH2003

    if (
        len(password_clean) == 9
        and password_clean[:5] == "BTECH"
        and password_clean[5:].isdigit()
        and 2000 <= int(password_clean[5:]) <= 2020
    ):

        st.session_state["student_logged_in"] = True

        st.session_state["student_dob_year"] = (
            password_clean[5:]
        )

        st.rerun()

    else:

        st.error(
            "Invalid password. "
            "Use BTECH + 4-digit DOB year "
            "(2000–2020). Example: BTECH2003"
        )


# -----------------------------
# STUDENT DASHBOARD
# -----------------------------

if st.session_state.get(
    "student_logged_in",
    False
):

    st.divider()

    st.header("👨‍🎓 Student Dashboard")

    st.success("Login successful!")

    st.write(
        "DOB Year:",
        st.session_state["student_dob_year"]
    )

    st.subheader("📚 Subject Information")

    subject_names = []

    subject_columns = st.columns(3)

    for i in range(6):

        with subject_columns[i % 3]:

            subject = st.text_input(
                f"Subject {i + 1}",
                key=f"subject_{i}"
            )

            subject_names.append(subject)


    st.divider()

    st.subheader("📊 Academic Information")

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


        subject_inputs.append({
            "Subject": subject,
            "Attendance": attendance,
            "Study_Hours": study_hours,
            "Internal_Mark": internal,
            "Assignment": assignment,
            "Previous_Mark": previous
        })


        st.divider()


    # -----------------------------
    # PREDICTION
    # -----------------------------

    if st.button(
        "🔮 Predict Performance",
        type="primary",
        width="stretch"
    ):

        results = []


        for subject_data in subject_inputs:

            input_data = pd.DataFrame([{
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
            }])


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


            results.append({
                "Subject":
                    subject_data["Subject"],

                "Prediction":
                    prediction,

                "Confidence (%)":
                    round(confidence, 1),

                "Cluster":
                    cluster
            })


        results_df = pd.DataFrame(results)


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


    # -----------------------------
    # LOGOUT
    # -----------------------------

    st.divider()

    if st.button("Logout"):

        st.session_state["student_logged_in"] = False

        st.rerun()


st.divider()

st.caption(
    "Student Performance Prediction System"
)
if (
    len(password_clean) == 9
    and password_clean[:5] == "BTECH"
    and password_clean[5:].isdigit()
    and 2000 <= int(password_clean[5:]) <= 2020
):
