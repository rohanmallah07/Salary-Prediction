import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Neon Salary Predictor", layout="wide", initial_sidebar_state="expanded")

# Paths (update if different)
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
HISTORY_PATH = r"C:\Users\rohan\OneDrive\Desktop\Project\salary_history.csv"

# ---------------------------
# LOAD MODEL & SCALER SAFE
# ---------------------------
def safe_load(path, kind="model"):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        st.error(f"Could not load {kind} from `{path}`. Error: {e}")
        return None

model = safe_load(MODEL_PATH, "model")
scaler = safe_load(SCALER_PATH, "scaler")

# ---------------------------
# NEON CSS
# ---------------------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: radial-gradient( circle at 10% 20%, #071426 0%, #071426 10%, #04060a 100% );
        color: #e6edf3;
    }

    /* Card */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(2,12,27,0.6), 0 0 30px rgba(0, 200, 255, 0.06);
        border: 1px solid rgba(0,200,255,0.06);
    }

    .neon-title {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        background: linear-gradient(90deg,#00f0ff,#7b00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 8px rgba(0,255,255,0.12), 0 0 20px rgba(123,0,255,0.06);
    }

    .neon-sub {
        text-align:center;
        color:#9fb8c8;
        margin-top: -8px;
        margin-bottom: 18px;
    }

    .btn-neon > button {
        background: linear-gradient(90deg,#0072ff,#8a00ff);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 6px 18px rgba(0,114,255,0.18);
        border: none;
    }

    .small-muted {
        color: #98a8b9;
        font-size: 13px;
    }

    .footer {
        color: #7f8b94;
        font-size: 13px;
        text-align: center;
        margin-top: 18px;
    }

    /* make tables stand out a bit */
    .stDataFrame>div {
        background: rgba(255,255,255,0.01) !important;
        border-radius:8px;
    }

    </style>
    """, unsafe_allow_html=True
)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<div class='neon-title'>💼 NEON Salary Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-sub'>Fast. Modern. Visual — Predict & compare salaries with style ✨</div>", unsafe_allow_html=True)

# ---------------------------
# Options / Metadata
# ---------------------------
gender_options = {0: "Male", 1: "Female", 2: "Other"}
country_options = {0: "India", 1: "USA", 2: "Germany", 3: "UK", 4: "Canada"}
edu_options = {1: "Graduate", 2: "Masters", 3: "PhD"}
job_options = {1: "Software Engineer", 2: "Java Developer", 3: "Data Analyst", 4: "Web Developer"}
race_options = {0: "Asian", 1: "Black", 2: "White", 3: "Other"}

# ---------------------------
# LAYOUT - Inputs & Controls
# ---------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 Employee Profile")
    cols = st.columns([1,1,1,1])
    with cols[0]:
        gender = st.selectbox("Gender", list(gender_options.keys()), format_func=lambda x: gender_options[x])
        edu = st.selectbox("Education", list(edu_options.keys()), format_func=lambda x: edu_options[x])
    with cols[1]:
        country = st.selectbox("Country", list(country_options.keys()), format_func=lambda x: country_options[x])
        job = st.selectbox("Job Role", list(job_options.keys()), format_func=lambda x: job_options[x])
    with cols[2]:
        race = st.selectbox("Race", list(race_options.keys()), format_func=lambda x: race_options[x])
        age = st.number_input("Age", min_value=18, max_value=80, value=28)
    with cols[3]:
        experience = st.slider("Experience (years)", min_value=0.0, max_value=50.0, step=0.5, value=2.0)
        st.markdown("<div class='small-muted'>Tip: Set Experience=0 to test minimum salary</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Predict Button (big neon)
# ---------------------------
predict_col1, predict_col2, predict_col3 = st.columns([1,2,1])
with predict_col2:
    if model is None or scaler is None:
        st.warning("Model or scaler not loaded — prediction disabled. Put model.pkl & scaler.pkl in the app directory.")
    else:
        if st.button("🚀 Predict Salary", key="predict", help="Click to predict salary"):
            # Prepare numeric scaled parts
            numeric_data = np.array([[age, experience, 0]])
            try:
                scaled_numeric = scaler.transform(numeric_data)[0]
            except Exception as e:
                st.error(f"Scaler error: {e}")
                scaled_numeric = np.array([age, experience, 0])

            # final feature arrangement: [gender, edu, job, age_scaled, exp_scaled, country, race]
            final_input = np.array([[gender, edu, job, scaled_numeric[0], scaled_numeric[1], country, race]])

            try:
                scaled_salary = model.predict(final_input)[0]
                real_salary = scaler.inverse_transform([[0,0, scaled_salary]])[0][2]
            except Exception as e:
                st.error(f"Model prediction error: {e}")
                real_salary = 0

            # enforce 10000 when experience==0
            if experience == 0:
                real_salary = 10000

            # Show result card
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center; color:#00f0ff;'>💰 Predicted Salary</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center; font-size:32px; color:#ffffff;'>₹ {real_salary:,.0f} / year</h1>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center' class='small-muted'>As of {ts}</div>", unsafe_allow_html=True)

            # Save history entry (human readable strings)
            history_entry = pd.DataFrame([{
                "Timestamp": ts,
                "Gender": gender_options[gender],
                "Education": edu_options[edu],
                "Job": job_options[job],
                "Country": country_options[country],
                "Race": race_options[race],
                "Age": age,
                "Experience": experience,
                "Predicted Salary": float(real_salary)
            }])

            # ensure folder exists and append / create CSV
            try:
                if os.path.exists(HISTORY_PATH):
                    df_old = pd.read_csv(HISTORY_PATH)
                    df_new = pd.concat([df_old, history_entry], ignore_index=True)
                else:
                    df_new = history_entry
                df_new.to_csv(HISTORY_PATH, index=False)
                st.success("📌 Saved to prediction history!")
            except Exception as e:
                st.error(f"Could not save history: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Dashboard: Charts & History
# ---------------------------
st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([2,1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Salary Visualizations")

    # Prepare combined data: history + uploaded (if any)
    history_df = pd.DataFrame()
    if os.path.exists(HISTORY_PATH):
        try:
            history_df = pd.read_csv(HISTORY_PATH)
        except:
            history_df = pd.DataFrame()

    # Show quick KPIs if history present
    if not history_df.empty:
        avg_salary = history_df["Predicted Salary"].mean()
        max_salary = history_df["Predicted Salary"].max()
        min_salary = history_df["Predicted Salary"].min()
        kpi1, kpi2, kpi3 = st.columns([1,1,1])
        kpi1.metric("Avg Predicted", f"₹ {avg_salary:,.0f}")
        kpi2.metric("Max Predicted", f"₹ {max_salary:,.0f}")
        kpi3.metric("Min Predicted", f"₹ {min_salary:,.0f}")

        st.markdown("---")

        # Job-wise aggregated bar chart (gradient)
        try:
            job_map = {str(k): v for k,v in job_options.items()}
            tmp = history_df.copy()
            # If Job stored as numeric, convert to str key to map
            if tmp["Job"].dtype != object:
                tmp["Job"] = tmp["Job"].astype(str)
            # If values are names already, keep them
            # Group by Job name
            job_group = tmp.groupby("Job", as_index=False)["Predicted Salary"].mean()
            job_group = job_group.sort_values("Predicted Salary", ascending=False)
            fig = px.bar(job_group,
                         x="Job",
                         y="Predicted Salary",
                         text="Predicted Salary",
                         labels={"Predicted Salary": "Avg Predicted Salary (INR)"},
                         title="Average Predicted Salary by Job")
            fig.update_traces(marker=dict(line=dict(width=0.5, color='rgba(255,255,255,0.1)')),
                              hovertemplate='<b>%{x}</b><br>Avg: ₹ %{y:,.0f}<extra></extra>',
                              texttemplate='₹ %{y:,.0f}')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font_color="#e6edf3")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render job chart: {e}")

        # Experience vs Predicted Salary scatter with trendline
        try:
            fig2 = px.scatter(history_df, x="Experience", y="Predicted Salary", color="Job",
                              hover_data=["Age", "Education", "Country"],
                              title="Experience vs Predicted Salary",
                              labels={"Predicted Salary": "Pred Salary (INR)"})
            fig2.update_traces(marker=dict(size=10, line=dict(width=0.5, color='#ffffff')))
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e6edf3")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render scatter: {e}")

        # Distribution histogram
        try:
            fig3 = px.histogram(history_df, x="Predicted Salary", nbins=20,
                                title="Predicted Salary Distribution")
            fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e6edf3")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render distribution: {e}")

    else:
        st.info("No prediction history yet — predict a salary to see charts populate.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📜 Prediction History & Tools")

    if os.path.exists(HISTORY_PATH):
        df_hist = pd.read_csv(HISTORY_PATH)
        st.dataframe(df_hist.sort_values("Timestamp", ascending=False).reset_index(drop=True))

        # Download history CSV
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download History CSV", data=csv, file_name="salary_history.csv", mime="text/csv")

        # Show a small insights box
        total = len(df_hist)
        avg = df_hist["Predicted Salary"].mean()
        st.markdown(f"<div class='small-muted'>Records: <b>{total}</b> • Avg: <b>₹ {avg:,.0f}</b></div>", unsafe_allow_html=True)
    else:
        st.warning("No history found. After prediction, your results will be saved here.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Upload CSV for comparison (user-provided)
    st.subheader("📁 Upload CSV (Compare)")
    uploaded = st.file_uploader("Upload CSV with Job & Predicted Salary columns", type=["csv"], key="compare")
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.markdown("**Uploaded preview:**")
            st.dataframe(df_up.head(30))
            # normalize job col
            if "Job" in df_up.columns:
                df_up["Job"] = df_up["Job"].astype(str)
            if "Predicted Salary" in df_up.columns:
                figc = px.bar(df_up.sort_values("Predicted Salary", ascending=False), x="Job", y="Predicted Salary",
                              text="Predicted Salary", title="Uploaded: Salary Comparison")
                figc.update_traces(texttemplate='₹ %{text:,.0f}', textposition='outside')
                figc.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e6edf3")
                st.plotly_chart(figc, use_container_width=True)
            else:
                st.error("Uploaded CSV must contain 'Predicted Salary' column.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("<div class='footer'>Neon Salary Predictor • Made with ❤️ • Tip: Save model.pkl & scaler.pkl in app folder</div>", unsafe_allow_html=True)
