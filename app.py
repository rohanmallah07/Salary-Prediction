import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# ===========================
# LOAD MODEL & SCALER
# ===========================
model = pickle.load(open(r"C:\Users\rohan\OneDrive\Desktop\Project\model.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\rohan\OneDrive\Desktop\Project\scaler.pkl", "rb"))
model = pickle.load(open(r"C:\Users\rohan\OneDrive\Desktop\Project\model.pkl", "rb"))


# ===========================
# PAGE SETUP + CUSTOM CSS
# ===========================
st.set_page_config(page_title="Salary Prediction App", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0d1117;
}
.main {
    background-color: #161b22;
}
h1, h2, h3, p, label, .css-1kyxreq, .css-q8sbsg {
    color: #e6edf3 !important;
}
.stButton>button {
    background: linear-gradient(45deg, #0072ff, #00c6ff);
    width: 100%;
    border-radius: 10px;
    color: white;
    font-weight: bold;
    font-size: 18px;
}
.css-1d391kg {
    background-color: #21262d !important;
    color: #e6edf3;
    border-radius: 10px;
}
.css-1kyxreq, .css-q8sbsg {
    font-size: 18px;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #21262d;
    box-shadow: 0px 0px 12px rgba(0, 255, 255, 0.15);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# HEADER
# ===========================
st.markdown("<h1 style='text-align: center;'>💼 AI Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Smart Salary estimation based on profile data 🔥</p>",
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===========================
# DROPDOWN OPTIONS
# ===========================
gender_options = {0: "Male", 1: "Female", 2: "Other"}
country_options = {0: "India", 1: "USA", 2: "Germany", 3: "UK", 4: "Canada"}
edu_options = {1: "Graduate", 2: "Masters", 3: "PhD"}
job_options = {1: "Software Engineer", 2: "Java Developer", 3: "Data Analyst", 4: "Web Developer"}
race_options = {0: "Asian", 1: "Black", 2: "White", 3: "Other"}

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("👤 Employee Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", list(gender_options.keys()), format_func=lambda x: gender_options[x])
    country = st.selectbox("Country", list(country_options.keys()), format_func=lambda x: country_options[x])
    race = st.selectbox("Ethnic Category", list(race_options.keys()), format_func=lambda x: race_options[x])

with col2:
    edu = st.selectbox("Education Level", list(edu_options.keys()), format_func=lambda x: edu_options[x])
    job = st.selectbox("Job Title", list(job_options.keys()), format_func=lambda x: job_options[x])
    age = st.number_input("Age", min_value=18, max_value=80)

experience = st.slider("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)
st.markdown("</div>", unsafe_allow_html=True)

history_file = r"C:\Users\rohan\OneDrive\Desktop\Project\salary_history.csv"

# ===========================
# PREDICT BUTTON
# ===========================
if st.button("✨ Predict Salary"):

    numeric_data = np.array([[age, experience, 0]])
    scaled_numeric = scaler.transform(numeric_data)[0]

    final_input = np.array([[gender, edu, job, scaled_numeric[0], scaled_numeric[1], country, race]])

    scaled_salary = model.predict(final_input)[0]
    real_salary = scaler.inverse_transform([[0, 0, scaled_salary]])[0][2]

    if experience == 0:
        real_salary = 10000

    st.markdown(
        f"<div class='card'><h2>💰 Estimated Salary: ₹ {real_salary:,.0f} / year</h2></div>",
        unsafe_allow_html=True)

    history_entry = pd.DataFrame([{
        "Gender": gender_options[gender],
        "Education": edu_options[edu],
        "Job": job_options[job],
        "Country": country_options[country],
        "Race": race_options[race],
        "Age": age,
        "Experience": experience,
        "Predicted Salary": real_salary
    }])

    if os.path.exists(history_file):
        old = pd.read_csv(history_file)
        history_df = pd.concat([old, history_entry], ignore_index=True)
    else:
        history_df = history_entry

    history_df.to_csv(history_file, index=False)
    st.success("📌 Saved to Prediction History!")

# ===========================
# HISTORY DISPLAY
# ===========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📜 Prediction History")
if os.path.exists(history_file):
    st.dataframe(pd.read_csv(history_file))
else:
    st.warning("⚠ No history yet!")
st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# CSV UPLOAD FOR COMPARISON
# ===========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📊 Upload Salary Comparison CSV")

file = st.file_uploader("Upload CSV with 'Job' & 'Predicted Salary'", type=["csv"])

if file:
    df_up = pd.read_csv(file)
    st.dataframe(df_up)

    if "Job" in df_up.columns:
        df_up["Job"] = df_up["Job"].astype(str)

    if "Predicted Salary" in df_up.columns:
        st.bar_chart(df_up, x="Job", y="Predicted Salary")
    else:
        st.error("CSV must contain 'Predicted Salary' column!")
st.markdown("</div>", unsafe_allow_html=True)

