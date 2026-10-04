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
queue_model, wait_model = load_models()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #0b2239;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        font-size: 30px;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 3px;
    }

    .subtitle {
        font-size: 14px;
        color: #718096;
        margin-bottom: 25px;
    }

    .welcome {
        font-size: 25px;
        font-weight: 800;
        color: #102a43;
    }

    .small-text {
        font-size: 13px;
        color: #718096;
    }

    .card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e3eaf1;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        min-height: 135px;
    }

    .card-title {
        font-size: 12px;
        color: #718096;
        font-weight: 700;
    }

    .card-value {
        font-size: 29px;
        color: #102a43;
        font-weight: 800;
        margin-top: 10px;
    }

    .card-note {
        font-size: 11px;
        color: #159579;
        margin-top: 6px;
    }

    .section {
        font-size: 19px;
        font-weight: 800;
        color: #102a43;
        margin-top: 30px;
        margin-bottom: 5px;
    }

    .info {
        background: #eaf7f5;
        padding: 15px;
        border-radius: 12px;
        color: #176b61;
        font-size: 13px;
    }

    .prediction {
        background: linear-gradient(135deg, #0b2942, #12496a);
        padding: 28px;
        border-radius: 18px;
        color: white;
        text-align: center;
    }

    .prediction-label {
        font-size: 12px;
        color: #b8d0df;
        font-weight: 700;
    }

    .prediction-number {
        font-size: 52px;
        font-weight: 800;
        margin: 8px;
    }

    .prediction-sub {
        font-size: 12px;
        color: #d4e2ea;
    }

    .status {
        padding: 18px;
        background: white;
        border-radius: 14px;
        border: 1px solid #e3eaf1;
        margin-bottom: 12px;
    }

    .online {
        color: #159579;
        font-weight: 800;
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

    st.markdown(
        '<div class="welcome">Good morning 👋</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-text">'
        'Monitor appointment activity and AI-powered queue predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

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

    # ---------------- AI PANEL ----------------

    st.markdown(
        '<div class="section">🤖 AI Prediction Engine</div>',
        unsafe_allow_html=True
    )

    p1, p2 = st.columns([1.5, 1])

    with p1:

        st.markdown(
            """
            <div class="card">

                <h3>Smart Queue Prediction</h3>

                <p class="small-text">
                MediQueue AI uses machine learning to estimate
                a patient's queue position and expected waiting time.
                </p>

                <div class="info">
                The model analyzes department, appointment time,
                patient type, previous visits, existing appointments
                and emergency status.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:

        st.markdown(
            """
            <div class="card">

                <h3>Model Status</h3>

                <p class="online">
                ● Prediction Engine Online
                </p>

                <p class="small-text">
                Queue Model: Loaded
                </p>

                <p class="small-text">
                Waiting-Time Model: Loaded
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# AI PREDICTION
# ============================================================

elif page == "AI Prediction":

    st.markdown(
        '<div class="welcome">🤖 AI Queue Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-text">'
        'Enter appointment details to predict queue position and waiting time.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    input_col, result_col = st.columns([1.4, 1])

    with input_col:

        st.subheader("Appointment Information")

        appointment_date = st.date_input(
            "Appointment Date"
        )

        c1, c2 = st.columns(2)

        with c1:

            department = st.selectbox(
                "Department",
                sorted(df["Department"].unique())
            )

        with c2:

            patient_type = st.selectbox(
                "Patient Type",
                sorted(df["Patient_Type"].unique())
            )

        c3, c4 = st.columns(2)

        with c3:

            appointment_hour = st.slider(
                "Appointment Hour",
                8,
                20,
                10
            )

        with c4:

            appointment_minute = st.selectbox(
                "Appointment Minute",
                [0, 15, 30, 45]
            )

        c5, c6 = st.columns(2)

        with c5:

            existing_appointments = st.number_input(
                "Existing Appointments",
                min_value=0,
                max_value=100,
                value=10
            )

        with c6:

            previous_visits = st.number_input(
                "Previous Visits",
                min_value=0,
                max_value=20,
                value=1
            )

        emergency = st.selectbox(
            "Emergency Status",
            ["No", "Yes"]
        )

        predict = st.button(
            "🔮 Predict Queue Position",
            width="stretch"
        )

    with result_col:

        st.subheader("Prediction Result")

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

                st.markdown(
                    f"""
                    <div class="prediction">

                        <div class="prediction-label">
                        PREDICTED QUEUE POSITION
                        </div>

                        <div class="prediction-number">
                        #{queue_prediction}
                        </div>

                        <div class="prediction-sub">
                        Estimated position in queue
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                st.metric(
                    "Estimated Waiting Time",
                    f"{wait_prediction} minutes"
                )

                st.success(
                    "Prediction generated successfully."
                )

            except Exception as error:

                st.error(
                    f"Prediction error: {error}"
                )

        else:

            st.info(
                "Enter appointment details and click "
                "'Predict Queue Position'."
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

    st.markdown(
        '<div class="welcome">⚙️ System Status</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-text">'
        'MediQueue AI system and model status.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="status">
                <div class="online">
                ● System Online
                </div>
                <div class="small-text">
                MediQueue AI is running normally.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="status">
                <div class="online">
                ● Queue Model Loaded
                </div>
                <div class="small-text">
                Random Forest queue model is available.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="status">
                <div class="online">
                ● Waiting-Time Model Loaded
                </div>
                <div class="small-text">
                Waiting-time prediction model is available.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="status">
                <div class="online">
                ● Dataset Connected
                </div>
                <div class="small-text">
                {len(df):,} records loaded successfully.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="status">
                <div class="online">
                ● Prediction Engine Ready
                </div>
                <div class="small-text">
                Queue and waiting-time prediction available.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="status">
                <div class="online">
                ● Data Analysis Ready
                </div>
                <div class="small-text">
                Historical analytics are available.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">Project Information</div>',
        unsafe_allow_html=True
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        st.info(
            "Python + Streamlit + Pandas"
        )

    with i2:

        st.info(
            "Scikit-learn + Random Forest"
        )

    with i3:

        st.info(
            "Queue + Waiting-Time Prediction"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MediQueue AI • AI-Based Healthcare Appointment & Queue Prediction System"
)