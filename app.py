import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# MEDIQUEUE AI
# AI-Based Healthcare Appointment & Queue Prediction System
# ============================================================

st.set_page_config(
    page_title="MediQueue AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "healthcare_appointments.csv"
QUEUE_MODEL_PATH = BASE_DIR / "models" / "queue_model.pkl"
WAIT_MODEL_PATH = BASE_DIR / "models" / "wait_time_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    queue_model = joblib.load(QUEUE_MODEL_PATH)
    wait_model = joblib.load(WAIT_MODEL_PATH)
    return queue_model, wait_model


df = load_data()
queue_model = None
wait_model = None

try:
    queue_model, wait_model = load_models()
except Exception as e:
    st.error(f"Failed to load AI models: {e}")


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
   <style>

    /* ===== MEDIQUEUE AI PROFESSIONAL THEME ===== */

.stApp {
    background: #f4f8fb;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #073b4c 0%, #0b5268 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main title */
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #073b4c;
    margin-bottom: 2px;
}

.subtitle {
    font-size: 15px;
    color: #6b7c85;
    margin-bottom: 25px;
}

/* Cards */
.dashboard-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5edf2;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
}

.card-label {
    color: #71808a;
    font-size: 14px;
    font-weight: 600;
}

.card-value {
    color: #073b4c;
    font-size: 30px;
    font-weight: 800;
    margin-top: 5px;
}

.card-icon {
    font-size: 25px;
}

/* Section headings */
.section-title {
    color: #073b4c;
    font-size: 22px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 12px;
}

/* Prediction box */
.prediction-box {
    background: linear-gradient(135deg, #073b4c, #087f8c);
    padding: 25px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 6px 20px rgba(7,59,76,0.18);
}

/* Status */
.status-online {
    background: #e8f8f1;
    color: #087f5b;
    padding: 8px 15px;
    border-radius: 20px;
    font-weight: 700;
    text-align: center;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="font-size:25px;">🏥 MediQueue AI</h1>
        <p style="font-size:12px;color:#a9bdcc;">
        Healthcare Appointment Intelligence
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "AI Prediction",
            "Analytics",
            "System Status"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🤖 AI MODEL")

    st.success("Random Forest Online")

    st.caption(
        "Queue and waiting-time prediction engine"
    )


# ============================================================
# HEADER
# ============================================================

header1, header2 = st.columns([5, 1])

with header1:

    st.markdown(
        '<div class="main-title">Healthcare Appointment Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered queue position and waiting-time prediction'
        '</div>',
        unsafe_allow_html=True
    )

with header2:

    st.success("● SYSTEM ONLINE")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    # ==============================
    # WELCOME HEADER
    # ==============================

    st.markdown(
        '<div class="main-title">Good morning 👋</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered healthcare appointment and queue intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    # ==============================
    # KPI CARDS
    # ==============================

    total_patients = len(df)

    avg_wait = round(
        df["Waiting_Time_Minutes"].mean(), 1
    )

    departments = df["Department"].nunique()

    avg_queue = round(
        df["Queue_Number"].mean(), 1
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-icon">👥</div>
                <div class="card-label">TOTAL APPOINTMENTS</div>
                <div class="card-value">{total_patients:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-icon">⏱️</div>
                <div class="card-label">AVERAGE WAIT</div>
                <div class="card-value">{avg_wait} min</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-icon">🏥</div>
                <div class="card-label">DEPARTMENTS</div>
                <div class="card-value">{departments}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-icon">📋</div>
                <div class="card-label">AVG QUEUE</div>
                <div class="card-value">{avg_queue}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    
    # ==============================
    # APPOINTMENT ANALYTICS
    # ==============================

    st.markdown(
        '<div class="section-title">📊 Appointment Activity</div>',
        unsafe_allow_html=True
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown(
            '<div class="card-label">Appointments by Department</div>',
            unsafe_allow_html=True
        )

        department_count = (
            df["Department"]
            .value_counts()
            .sort_values(ascending=True)
        )

        st.bar_chart(
            department_count,
            width="stretch"
        )

    with chart2:

        st.markdown(
            '<div class="card-label">Waiting Time Trend</div>',
            unsafe_allow_html=True
        )

        hourly_wait = (
            df.groupby("Appointment_Hour")[
                "Waiting_Time_Minutes"
            ]
            .mean()
            .round(1)
        )

        st.line_chart(
            hourly_wait,
            width="stretch"
        )

    # ---------------- KPI ----------------

    total = len(df)
    average_wait = df["Waiting_Time_Minutes"].mean()
    departments = df["Department"].nunique()
    average_queue = df["Queue_Number"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">TOTAL APPOINTMENTS</div>
                <div class="card-value">{total:,}</div>
                <div class="card-note">Dataset records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">AVG. WAITING TIME</div>
                <div class="card-value">{average_wait:.1f} min</div>
                <div class="card-note">Historical average</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">DEPARTMENTS</div>
                <div class="card-value">{departments}</div>
                <div class="card-note">Healthcare departments</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">AVG. QUEUE POSITION</div>
                <div class="card-value">{average_queue:.1f}</div>
                <div class="card-note">Historical average</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- CHARTS ----------------

    st.markdown(
        '<div class="section">📊 Appointment Activity</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Historical appointment distribution across departments and time."
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        department_counts = (
            df["Department"]
            .value_counts()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            department_counts.index,
            department_counts.values
        )

        ax.set_xlabel("Appointments")
        ax.set_title(
            "Appointments by Department",
            loc="left",
            fontweight="bold"
        )

        ax.grid(axis="x", alpha=0.2)

        plt.tight_layout()

        st.pyplot(fig, width="stretch")

    with chart2:

        hourly = (
            df.groupby("Appointment_Hour")
            .size()
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            hourly.index,
            hourly.values,
            marker="o",
            linewidth=2
        )

        ax.set_xlabel("Appointment Hour")
        ax.set_ylabel("Appointments")

        ax.set_title(
            "Appointment Activity by Hour",
            loc="left",
            fontweight="bold"
        )

        ax.grid(alpha=0.2)

        plt.tight_layout()

        st.pyplot(fig, width="stretch")

    
# ============================================================
# AI PREDICTION
# ============================================================

elif page == "AI Prediction":

    # ========================================================
    # AI PREDICTION HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">🤖 AI Queue Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter patient appointment details and let MediQueue AI '
        'estimate queue position and waiting time.'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # INPUT + RESULT
    # ========================================================

    input_col, result_col = st.columns(
        [1.35, 1],
        gap="large"
    )

    # ========================================================
    # APPOINTMENT FORM
    # ========================================================

    with input_col:

        st.markdown(
            """
            <div class="section-title">
                📋 Appointment Information
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="small-text">
                Provide the information below for an AI-based prediction.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        appointment_date = st.date_input(
            "📅 Appointment Date"
        )

        c1, c2 = st.columns(2)

        with c1:

            department = st.selectbox(
                "🏥 Department",
                sorted(df["Department"].unique())
            )

        with c2:

            patient_type = st.selectbox(
                "👤 Patient Type",
                sorted(df["Patient_Type"].unique())
            )

        c3, c4 = st.columns(2)

        with c3:

            appointment_hour = st.slider(
                "🕐 Appointment Hour",
                min_value=8,
                max_value=20,
                value=10
            )

        with c4:

            appointment_minute = st.selectbox(
                "Minutes",
                [0, 15, 30, 45]
            )

        c5, c6 = st.columns(2)

        with c5:

            existing_appointments = st.number_input(
                "👥 Existing Appointments",
                min_value=0,
                max_value=100,
                value=10
            )

        with c6:

            previous_visits = st.number_input(
                "🔄 Previous Visits",
                min_value=0,
                max_value=20,
                value=1
            )

        emergency = st.selectbox(
            "🚨 Emergency Status",
            ["No", "Yes"]
        )

        st.write("")

        predict = st.button(
            "🔮  PREDICT QUEUE POSITION",
            width="stretch"
        )

    # ========================================================
    # RESULT PANEL
    # ========================================================

    with result_col:

        st.markdown(
            """
            <div class="section-title">
                🎯 Prediction Result
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="small-text">
                AI-generated prediction based on the entered information.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if predict:

            appointment_day = pd.Timestamp(
                appointment_date
            ).day_name()

            input_data = pd.DataFrame(
                [{
                    "Day": appointment_day,
                    "Department": department,
                    "Patient_Type": patient_type,
                    "Appointment_Hour": appointment_hour,
                    "Appointment_Minute": appointment_minute,
                    "Existing_Appointments": existing_appointments,
                    "Previous_Visits": previous_visits,
                    "Emergency": emergency
                }]
            )

            try:

                queue_prediction = queue_model.predict(
                    input_data
                )[0]

                wait_prediction = wait_model.predict(
                    input_data
                )[0]

                queue_prediction = max(
                    1,
                    int(round(queue_prediction))
                )

                wait_prediction = max(
                    0,
                    int(round(wait_prediction))
                )

                # --------------------------------------------
                # QUEUE RESULT
                # --------------------------------------------

                st.success(
                    f"🎯 Predicted Queue Position: #{queue_prediction}"
                )

                st.metric(
                    "📍 Queue Position",
                    f"#{queue_prediction}"
                )

                st.caption(
                    "Estimated position in today's queue."
                )

                st.write("")

                # --------------------------------------------
                # WAITING TIME
                # --------------------------------------------

                wait_col, status_col = st.columns(2)

                with wait_col:

                    st.metric(
                        "⏱️ Estimated Waiting Time",
                        f"{wait_prediction} min"
                    )

                with status_col:

                    if wait_prediction <= 15:

                        st.success("Low Wait")

                    elif wait_prediction <= 30:

                        st.warning("Moderate Wait")

                    else:

                        st.error("High Wait")

                st.write("")

                # --------------------------------------------
                # APPOINTMENT SUMMARY
                # --------------------------------------------

                st.subheader("📋 Appointment Summary")

                s1, s2 = st.columns(2)

                with s1:

                    st.write(
                        f"**Department:** {department}"
                    )

                    st.write(
                        f"**Patient Type:** {patient_type}"
                    )

                    st.write(
                        f"**Appointment:** "
                        f"{appointment_hour:02d}:{appointment_minute:02d}"
                    )

                with s2:

                    st.write(
                        f"**Existing Appointments:** "
                        f"{existing_appointments}"
                    )

                    st.write(
                        f"**Previous Visits:** "
                        f"{previous_visits}"
                    )

                    st.write(
                        f"**Emergency:** {emergency}"
                    )

                st.success(
                    "✓ AI prediction generated successfully."
                )

            except Exception as error:

                st.error(
                    f"Prediction error: {error}"
                )

        else:

            st.info(
                "🤖 Ready for Prediction"
            )

            st.markdown(
                "Enter the appointment information on the left "
                "and click **Predict Queue Position** to generate "
                "your AI prediction."
            )

            st.write("")

            r1, r2 = st.columns(2)

            with r1:
                st.metric(
                    "Queue Position",
                    "—"
                )

            with r2:
                st.metric(
                    "Waiting Time",
                    "—"
                )

            st.write("")

            st.caption(
                "MediQueue AI uses the trained Random Forest models "
                "to estimate queue position and waiting time."
            )
# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="welcome">📊 Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-text">'
        'Explore historical healthcare appointment patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Minimum Waiting Time",
            f"{df['Waiting_Time_Minutes'].min():.0f} min"
        )

    with c2:

        st.metric(
            "Maximum Waiting Time",
            f"{df['Waiting_Time_Minutes'].max():.0f} min"
        )

    with c3:

        st.metric(
            "Maximum Queue",
            f"{df['Queue_Number'].max():.0f}"
        )

    st.markdown(
        '<div class="section">Waiting Time by Department</div>',
        unsafe_allow_html=True
    )

    department_wait = (
        df.groupby("Department")["Waiting_Time_Minutes"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.barh(
        department_wait.index,
        department_wait.values
    )

    ax.set_xlabel(
        "Average Waiting Time (minutes)"
    )

    ax.set_title(
        "Average Waiting Time by Department",
        loc="left",
        fontweight="bold"
    )

    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig, width="stretch")

    st.markdown(
        '<div class="section">Queue vs Waiting Time</div>',
        unsafe_allow_html=True
    )

    sample = df.sample(
        min(1000, len(df)),
        random_state=42
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.scatter(
        sample["Queue_Number"],
        sample["Waiting_Time_Minutes"],
        alpha=0.35
    )

    ax.set_xlabel("Queue Position")
    ax.set_ylabel("Waiting Time (minutes)")

    ax.set_title(
        "Queue Position vs Waiting Time",
        loc="left",
        fontweight="bold"
    )

    ax.grid(alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig, width="stretch")


# ============================================================
# SYSTEM STATUS
# ============================================================

elif page == "System Status":

    # ========================================================
    # SYSTEM STATUS HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">⚙️ System Status</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor MediQueue AI services, prediction models and dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # ========================================================
    # OVERALL SYSTEM STATUS
    # ========================================================

    st.success(
        "🟢  All Systems Operational — MediQueue AI is running normally."
    )

    st.write("")

    # ========================================================
    # STATUS OVERVIEW
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "🤖 Queue Model",
            "ACTIVE"
        )

    with c2:

        st.metric(
            "⏱️ Waiting-Time Model",
            "ACTIVE"
        )

    with c3:

        st.metric(
            "📊 Dataset",
            f"{len(df):,}"
        )

    with c4:

        st.metric(
            "🏥 Departments",
            df["Department"].nunique()
        )

    st.write("")

    # ========================================================
    # SERVICE STATUS
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 Service Status</div>',
        unsafe_allow_html=True
    )

    s1, s2 = st.columns(2)

    with s1:

        st.markdown("### 🟢 Application")

        st.success("System Online")

        st.caption(
            "MediQueue AI application is running normally."
        )

        st.markdown("### 🟢 Queue Prediction")

        st.success("Model Ready")

        st.caption(
            "Random Forest queue prediction model is loaded."
        )

        st.markdown("### 🟢 Waiting-Time Prediction")

        st.success("Model Ready")

        st.caption(
            "Waiting-time prediction model is loaded."
        )

    with s2:

        st.markdown("### 🟢 Dataset")

        st.success("Connected")

        st.caption(
            f"{len(df):,} healthcare appointment records loaded."
        )

        st.markdown("### 🟢 Analytics")

        st.success("Available")

        st.caption(
            "Historical appointment analytics are available."
        )

        st.markdown("### 🟢 Prediction Engine")

        st.success("Active")

        st.caption(
            "Queue position and waiting-time prediction are available."
        )

    st.write("")

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 AI Model Information</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.info(
            "**Queue Prediction**\n\n"
            "Random Forest\n\n"
            "Predicts patient queue position."
        )

    with m2:

        st.info(
            "**Waiting-Time Prediction**\n\n"
            "Machine Learning Model\n\n"
            "Estimates patient waiting time."
        )

    with m3:

        st.info(
            "**Data Analysis**\n\n"
            "Pandas + Streamlit\n\n"
            "Analyzes historical appointments."
        )

    st.write("")

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 Project Information</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.info(
            "🐍 **Technology**\n\n"
            "Python + Streamlit + Pandas"
        )

    with p2:

        st.info(
            "🤖 **Machine Learning**\n\n"
            "Scikit-learn + Random Forest"
        )

    with p3:

        st.info(
            "🏥 **AI Capabilities**\n\n"
            "Queue + Waiting-Time Prediction"
        )

    st.write("")

    # ========================================================
    # FINAL STATUS
    # ========================================================

    st.success(
        "✅ MediQueue AI is ready for appointment prediction and analytics."
    )
# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MediQueue AI • AI-Based Healthcare Appointment & Queue Prediction System"
)