import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
from datetime import datetime
import plotly.express as px

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Neon Salary Predictor", layout="wide", initial_sidebar_state="auto")

# Paths (relative)
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
HISTORY_PATH = "salary_history.csv"

# ---------------------------
# SAFE LOAD FOR MODEL & SCALER
# ---------------------------
def load_pickle_if_exists(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load '{path}': {e}")
            return None
    return None

model = load_pickle_if_exists(MODEL_PATH)
scaler = load_pickle_if_exists(SCALER_PATH)

# ---------------------------
# UI STYLE (Neon Cyber)
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, #071426 0%, #04060a 100%);
        color: #e6edf3;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(2,12,27,0.6), 0 0 30px rgba(0,200,255,0.06);
        border: 1px solid rgba(0,200,255,0.06);
        margin-bottom: 18px;
    }
    .neon-title {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg,#00f0ff,#7b00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 8px rgba(0,255,255,0.12);
    }
    .neon-sub {
        text-align:center;
        color:#9fb8c8;
        margin-top: -8px;
        margin-bottom: 20px;
    }
    .btn-neon>button {
        background: linear-gradient(90deg,#0072ff,#8a00ff);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 8px 12px;
        box-shadow: 0 6px 18px rgba(0,114,255,0.18);
        border: none;
    }
    .small-muted { color:#98a8b9; font-size:13px }
    .footer { color:#7f8b94; font-size:12px; text-align:center; margin-top:18px }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='neon-title'>💼 NEON Salary Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-sub'>Predict • Save • Compare — Modern Neon UI</div>", unsafe_allow_html=True)

# ---------------------------
# MAPPINGS (readable labels)
# ---------------------------
gender_map = {0: "Male", 1: "Female", 2: "Other"}
country_map = {0: "India", 1: "USA", 2: "Germany", 3: "UK", 4: "Canada"}
edu_map = {1: "Graduate", 2: "Masters", 3: "PhD"}
job_map = {1: "Software Engineer", 2: "Java Developer", 3: "Data Analyst", 4: "Web Developer"}
race_map = {0: "Asian", 1: "Black", 2: "White", 3: "Other"}

# ---------------------------
# INPUT FORM
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("👤 Employee Information (Input)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    gender = st.selectbox("Gender", options=list(gender_map.keys()), format_func=lambda x: gender_map[x])
    edu = st.selectbox("Education", options=list(edu_map.keys()), format_func=lambda x: edu_map[x])
with c2:
    country = st.selectbox("Country", options=list(country_map.keys()), format_func=lambda x: country_map[x])
    job = st.selectbox("Job Role", options=list(job_map.keys()), format_func=lambda x: job_map[x])
with c3:
    race = st.selectbox("Race", options=list(race_map.keys()), format_func=lambda x: race_map[x])
    age = st.number_input("Age", min_value=18, max_value=80, value=28)
with c4:
    experience = st.slider("Experience (years)", min_value=0.0, max_value=50.0, step=0.5, value=2.0)
    st.markdown("<div class='small-muted'>Tip: Set experience=0 to check ₹10,000 override</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# MODEL / SCALER UPLOAD (if not auto-found)
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("⚙️ Model & Scaler (auto-load or upload)")

if model is None:
    st.warning("model.pkl not found in app folder.")
    uploaded_model = st.file_uploader("Upload model.pkl (pickle)", type=["pkl"])
    if uploaded_model:
        try:
            model = pickle.load(uploaded_model)
            st.success("Model loaded from upload ✅")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

if scaler is None:
    st.warning("scaler.pkl not found in app folder.")
    uploaded_scaler = st.file_uploader("Upload scaler.pkl (pickle)", type=["pkl"])
    if uploaded_scaler:
        try:
            scaler = pickle.load(uploaded_scaler)
            st.success("Scaler loaded from upload ✅")
        except Exception as e:
            st.error(f"Failed to load uploaded scaler: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# PREDICT BUTTON
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
predict_col1, predict_col2 = st.columns([3,1])
with predict_col1:
    st.markdown("### Ready to predict?")
    st.write("Click the button to run model prediction using the format A:\n`[gender, edu, job, scaled_age, scaled_experience, country, race]`")
with predict_col2:
    predict_clicked = st.button("🚀 Predict Salary", key="predict_btn")

if predict_clicked:
    if model is None or scaler is None:
        st.error("Model or scaler not available. Upload both files or place them in the app folder.")
    else:
        # Scale numeric part (age, experience, dummy salary)
        try:
            numeric_data = np.array([[age, experience, 0]])
            scaled_numeric = scaler.transform(numeric_data)[0]  # expecting [age_scaled, exp_scaled, ...]
        except Exception as e:
            st.warning(f"Scaler transform failed, using raw numeric values. ({e})")
            scaled_numeric = np.array([age, experience, 0.0])

        # final feature vector as per option A
        final_input = np.array([[gender, edu, job, scaled_numeric[0], scaled_numeric[1], country, race]])

        # Predict
        try:
            scaled_salary = model.predict(final_input)[0]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            scaled_salary = 0.0

        # Inverse scale to real salary (assumes scaler.inverse_transform can accept [0,0,scaled_salary])
        try:
            real_salary = scaler.inverse_transform([[0, 0, scaled_salary]])[0][2]
        except Exception as e:
            st.warning(f"Scaler inverse_transform failed: {e}")
            real_salary = float(scaled_salary)

        # Ensure 0-experience rule
        if experience == 0:
            real_salary = 10000.0

        # Show result
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"<h3 style='color:#00f0ff;'>💰 Predicted Salary: ₹ {real_salary:,.0f} / year</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Prediction time: {ts}</div>", unsafe_allow_html=True)

        # Save to history (human readable)
        entry = pd.DataFrame([{
            "Timestamp": ts,
            "Gender": gender_map[gender],
            "Education": edu_map[edu],
            "Job": job_map[job],
            "Country": country_map[country],
            "Race": race_map[race],
            "Age": age,
            "Experience": experience,
            "Predicted Salary": float(real_salary)
        }])

        try:
            if os.path.exists(HISTORY_PATH):
                df_old = pd.read_csv(HISTORY_PATH)
                df_all = pd.concat([df_old, entry], ignore_index=True)
            else:
                df_all = entry
            df_all.to_csv(HISTORY_PATH, index=False)
            st.success("Saved to history ✔")
        except Exception as e:
            st.error(f"Could not save history: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# DASHBOARD: charts and history
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
left, right = st.columns([2,1])

with left:
    st.subheader("📊 Visual Insights")

    # Load history if exists
    hist = pd.DataFrame()
    if os.path.exists(HISTORY_PATH):
        try:
            hist = pd.read_csv(HISTORY_PATH)
        except Exception as e:
            st.warning(f"Could not read history CSV: {e}")
            hist = pd.DataFrame()

    if not hist.empty:
        # KPI row
        avg_sal = hist["Predicted Salary"].mean()
        max_sal = hist["Predicted Salary"].max()
        min_sal = hist["Predicted Salary"].min()
        k1, k2, k3 = st.columns(3)
        k1.metric("Avg Predicted", f"₹ {avg_sal:,.0f}")
        k2.metric("Max Predicted", f"₹ {max_sal:,.0f}")
        k3.metric("Min Predicted", f"₹ {min_sal:,.0f}")

        st.markdown("---")

        # Job-wise averages bar
        try:
            job_group = hist.groupby("Job", as_index=False)["Predicted Salary"].mean().sort_values("Predicted Salary", ascending=False)
            fig = px.bar(job_group, x="Job", y="Predicted Salary", text="Predicted Salary",
                         title="Average Predicted Salary by Job", labels={"Predicted Salary": "Avg Pred Salary (INR)"})
            fig.update_traces(texttemplate='₹ %{text:,.0f}', textposition='outside', marker_color=px.colors.sequential.Blues)
            fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e6edf3")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build job chart: {e}")

        # Experience vs salary scatter
        try:
            fig2 = px.scatter(hist, x="Experience", y="Predicted Salary", color="Job",
                              title="Experience vs Predicted Salary", hover_data=["Age", "Country", "Education"])
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e6edf3")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build scatter: {e}")

        # Distribution
        try:
            fig3 = px.histogram(hist, x="Predicted Salary", nbins=20, title="Predicted Salary Distribution")
            fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e6edf3")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build histogram: {e}")
    else:
        st.info("No history yet. Make predictions to populate charts.")

with right:
    st.subheader("📜 History & Tools")
    if not hist.empty:
        st.dataframe(hist.sort_values("Timestamp", ascending=False).reset_index(drop=True))
        csv_bytes = hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download History CSV", data=csv_bytes, file_name="salary_history.csv", mime="text/csv")
    else:
        st.info("History will appear here after you predict.")

    st.markdown("---")
    st.subheader("📁 Upload CSV to Compare")
    uploaded = st.file_uploader("Upload CSV with 'Job' & 'Predicted Salary' columns", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.markdown("**Uploaded preview:**")
            st.dataframe(df_up.head(50))

            if "Job" in df_up.columns:
                df_up["Job"] = df_up["Job"].astype(str)

            if "Predicted Salary" in df_up.columns:
                figc = px.bar(df_up.sort_values("Predicted Salary", ascending=False), x="Job", y="Predicted Salary",
                              title="Uploaded: Salary Comparison", text="Predicted Salary")
                figc.update_traces(texttemplate='₹ %{text:,.0f}', textposition='outside', marker_color=px.colors.sequential.Viridis)
                figc.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e6edf3")
                st.plotly_chart(figc, use_container_width=True)
            else:
                st.error("Uploaded CSV must include 'Predicted Salary' column.")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("<div class='footer'>Neon Salary Predictor • Made with ❤️ • Keep model.pkl & scaler.pkl in the same folder for auto-load</div>", unsafe_allow_html=True)
