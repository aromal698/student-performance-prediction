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
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/student_performance.csv"
    )


data = load_data()


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


missing_columns = [
    column
    for column in REQUIRED_COLUMNS
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
# TITLE
# =========================================================

st.title(
    "🎓 Student Performance Prediction System"
)

st.write(
    "B.Tech S3 – AI & Data Science"
)

st.write(
    "K-Means Clustering + Random Forest Classification"
)

st.divider()


# =========================================================
# ROLE SELECTION
# =========================================================

role = st.selectbox(
    "Select Dashboard",
    [
        "Student Dashboard",
        "Teacher Dashboard"
    ]
)


# =========================================================
# STUDENT DASHBOARD
# =========================================================

if role == "Student Dashboard":

    st.header(
        "👨‍🎓 Student Dashboard"
    )

    st.info(
        "Enter your student information and academic "
        "details for each subject."
    )


    # -----------------------------------------------------
    # STUDENT INFORMATION
    # -----------------------------------------------------

    st.subheader(
        "👤 Student Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input(
            "Student Name"
        )

        university_id = st.text_input(
            "University ID"
        )

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
    # ACADEMIC INFORMATION
    # -----------------------------------------------------

    st.subheader(
        "📊 Academic Information"
    )

    st.write(
        "Enter the following details separately for each subject."
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
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                key=f"attendance_{i}"
            )


            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                key=f"study_{i}"
            )


        with col2:

            internal = st.number_input(
                "Internal Mark",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                key=f"internal_{i}"
            )


            assignment = st.number_input(
                "Assignment Score",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                key=f"assignment_{i}"
            )


        with col3:

            previous = st.number_input(
                "Previous Mark",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
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
    # PREDICTION BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Performance for All Subjects",
        type="primary",
        width="stretch"
    ):


        if not student_name:

            st.warning(
                "Please enter Student Name."
            )

            st.stop()


        if not university_id:

            st.warning(
                "Please enter University ID."
            )

            st.stop()


        results = []


        # -------------------------------------------------
        # PREDICT EACH SUBJECT SEPARATELY
        # -------------------------------------------------

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
        # STUDENT INFORMATION
        # -------------------------------------------------

        st.success(
            "Performance prediction completed!"
        )


        st.subheader(
            "👤 Student Information"
        )


        student_info = pd.DataFrame(
            {
                "Information": [
                    "Student Name",
                    "University ID",
                    "Semester",
                    "Branch"
                ],

                "Value": [
                    student_name,
                    university_id,
                    semester,
                    branch
                ]
            }
        )


        st.dataframe(
            student_info,
            width="stretch",
            hide_index=True
        )


        # -------------------------------------------------
        # SUBJECT-WISE RESULTS
        # -------------------------------------------------

        st.subheader(
            "📊 Subject-wise Performance Prediction"
        )


        st.dataframe(
            results_df,
            width="stretch",
            hide_index=True
        )


        # -------------------------------------------------
        # INDIVIDUAL SUBJECT RESULTS
        # -------------------------------------------------

        st.subheader(
            "📚 Individual Subject Results"
        )


        for result in results:

            prediction = result[
                "Prediction"
            ]


            if prediction == "High":

                st.success(
                    f"📗 {result['Subject']} → "
                    f"High Performance "
                    f"({result['Confidence (%)']}% confidence)"
                )


            elif prediction == "Medium":

                st.warning(
                    f"📙 {result['Subject']} → "
                    f"Medium Performance "
                    f"({result['Confidence (%)']}% confidence)"
                )


            else:

                st.error(
                    f"📕 {result['Subject']} → "
                    f"Low Performance "
                    f"({result['Confidence (%)']}% confidence)"
                )


        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

        st.subheader(
            "💡 Academic Recommendation"
        )


        low_subjects = results_df[
            results_df["Prediction"] == "Low"
        ]["Subject"].tolist()


        medium_subjects = results_df[
            results_df["Prediction"] == "Medium"
        ]["Subject"].tolist()


        if low_subjects:

            st.error(
                "Additional academic attention is "
                "recommended for: "
                + ", ".join(low_subjects)
            )


        elif medium_subjects:

            st.warning(
                "Consider improving academic performance in: "
                + ", ".join(medium_subjects)
            )


        else:

            st.success(
                "Excellent! All subjects are predicted "
                "at High performance level."
            )


        st.caption(
            "The predictions are machine-learning estimates "
            "and should be used as academic guidance."
        )


# =========================================================
# TEACHER DASHBOARD
# =========================================================

else:

    st.header(
        "👩‍🏫 Teacher Dashboard"
    )

    st.info(
        "Teacher access is required to view the student dataset."
    )


    username = st.text_input(
        "Teacher Username"
    )


    password = st.text_input(
        "Teacher Password",
        type="password"
    )


    if st.button(
        "🔐 Teacher Login",
        type="primary"
    ):

        if (
            username == "teacher"
            and password == "teacher123"
        ):

            st.session_state[
                "teacher_logged_in"
            ] = True

        else:

            st.error(
                "Invalid username or password."
            )


    if st.session_state.get(
        "teacher_logged_in",
        False
    ):


        st.success(
            "Teacher login successful."
        )


        st.divider()


        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        st.subheader(
            "📊 Student Statistics"
        )


        total_students = len(data)


        high_count = len(
            data[
                data["Performance"] == "High"
            ]
        )


        medium_count = len(
            data[
                data["Performance"] == "Medium"
            ]
        )


        low_count = len(
            data[
                data["Performance"] == "Low"
            ]
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


        # -------------------------------------------------
        # MODEL PERFORMANCE
        # -------------------------------------------------

        st.subheader(
            "🤖 Model Performance"
        )


        st.metric(
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%"
        )


        st.write(
            "K-Means Clusters: **3**"
        )


        st.divider()


        # -------------------------------------------------
        # DATASET
        # -------------------------------------------------

        st.subheader(
            "📋 Student Dataset"
        )


        st.dataframe(
            data,
            width="stretch",
            hide_index=True
        )


        st.divider()


        # -------------------------------------------------
        # PERFORMANCE CHART
        # -------------------------------------------------

        st.subheader(
            "📈 Performance Distribution"
        )


        performance_counts = (
            data["Performance"]
            .value_counts()
            .rename_axis("Performance")
            .reset_index(
                name="Students"
            )
        )


        st.bar_chart(
            performance_counts.set_index(
                "Performance"
            )
        )


        if st.button(
            "Logout"
        ):

            st.session_state[
                "teacher_logged_in"
            ] = False

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Student Performance Prediction using "
    "K-Means Clustering and Random Forest"
)
