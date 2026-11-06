import streamlit as st
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Student Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Load Dataset
data = pd.read_csv("student.csv")

X = data[["Study_Hours", "Attendance"]]
y = data["Result"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])

model.fit(X_train, y_train)

# Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Initialize Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("🎓 Student Prediction")
st.sidebar.write("Machine Learning Project")
st.sidebar.write("Algorithm: KNN")
st.sidebar.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

# Main Title
st.title("🎓 Student Pass/Fail Prediction")
st.write("Student Performance Prediction System")

# Dashboard
st.subheader("📊 Dashboard")

history = st.session_state.history

total_students = len(history)
pass_students = sum(
    1 for student in history if student["Result"] == "Pass"
)
fail_students = sum(
    1 for student in history if student["Result"] == "Fail"
)

pass_percentage = (
    (pass_students / total_students) * 100
    if total_students > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", total_students)
col2.metric("Pass Students", pass_students)
col3.metric("Fail Students", fail_students)
col4.metric("Pass Percentage", f"{pass_percentage:.2f}%")

st.divider()

# Student Input
st.subheader("📝 Enter Student Details")

name = st.text_input("Student Name")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

# Prediction
if st.button("Predict Result", type="primary"):

    if not name.strip():
        st.warning("Please enter student name.")

    else:
        new_student = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance]
        })

        prediction = model.predict(new_student)[0]

        # Save Record
        record = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Student Name": name.strip(),
            "Study Hours": study_hours,
            "Attendance": attendance,
            "Result": prediction
        }

        st.session_state.history.append(record)

        if prediction == "Pass":
            st.success(f"Congratulations {name}! Student is predicted to PASS 🎉")
        else:
            st.error(f"{name}, student is predicted to FAIL.")

# Prediction History
st.divider()

st.subheader("📋 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, use_container_width=True)

    # Download CSV
    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History CSV",
        data=csv_data,
        file_name="student_prediction_history.csv",
        mime="text/csv"
    )

    # Clear History
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No prediction history available. Predict a student first.")

# Model Performance
st.divider()

st.subheader("📈 Model Performance")

st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Fail", "Pass"]
)

st.write("Confusion Matrix")

st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual Fail", "Actual Pass"],
        columns=["Predicted Fail", "Predicted Pass"]
    )
)

# Original Dataset
with st.expander("View Original Dataset"):
    st.dataframe(data, use_container_width=True)
