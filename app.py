import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# ---- Load & prep data ----
@st.cache_data
def load_data():
    df = pd.read_csv("Student_performance_data _.csv")
    return df

@st.cache_resource
def train_model(df):
    features = ["StudyTimeWeekly", "Tutoring", "ParentalSupport", "Extracurricular", "Absences"]
    X = df[features]
    y = df["GradeClass"].astype("category").cat.codes
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns, acc

df = load_data()
model, model_columns, accuracy = train_model(df)

GRADE_LABELS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}

# ---- UI ----
st.title("🎓 Student Performance Predictor")
st.write(
    "Predicts a student's likely grade class using a Random Forest model "
    f"trained on real student data (test accuracy: **{accuracy:.1%}**)."
)

st.subheader("Enter Student Details")

col1, col2 = st.columns(2)
with col1:
    study_time = st.slider("Weekly Study Time (hours)", 0.0, 20.0, 10.0, 0.5)
    absences = st.slider("Number of Absences", 0, 30, 5)
with col2:
    tutoring = st.selectbox("Receiving Tutoring?", ["No", "Yes"])
    extracurricular = st.selectbox("Extracurricular Activities?", ["No", "Yes"])

parental_support = st.selectbox(
    "Parental Support Level", [0, 1, 2, 3, 4],
    format_func=lambda x: ["None", "Low", "Moderate", "High", "Very High"][x]
)

if st.button("Predict Grade Class", type="primary"):
    input_df = pd.DataFrame([{
        "StudyTimeWeekly": study_time,
        "Tutoring": 1 if tutoring == "Yes" else 0,
        "ParentalSupport": parental_support,
        "Extracurricular": 1 if extracurricular == "Yes" else 0,
        "Absences": absences,
    }])

    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

    grade = GRADE_LABELS.get(prediction, str(prediction))
    st.success(f"### Predicted Grade Class: **{grade}**")

    proba_df = pd.DataFrame({
        "Grade": [GRADE_LABELS.get(i, str(i)) for i in range(len(proba))],
        "Probability": proba
    }).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Grade"))

st.markdown("---")
st.caption("Built by Samuel Isama · Data Analyst & ML Practitioner")
