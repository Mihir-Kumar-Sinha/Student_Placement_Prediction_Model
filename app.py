import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #16163a 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #c4b5fd !important;
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }

    /* ── Headers ── */
    h1 {
        background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        color: #c4b5fd !important;
        font-weight: 600 !important;
    }

    /* ── Result Cards ── */
    .result-card {
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-4px);
    }

    .placed-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
    }

    .not-placed-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(248, 113, 113, 0.08));
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15);
    }

    .result-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }

    .placed-title { color: #34d399; }
    .not-placed-title { color: #f87171; }

    .result-confidence {
        font-size: 1.1rem;
        font-weight: 500;
        color: #a5b4fc;
        margin-top: 0.5rem;
    }

    .confidence-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .placed-confidence { color: #6ee7b7; }
    .not-placed-confidence { color: #fca5a5; }

    /* ── Info Cards ── */
    .info-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }

    .info-card h4 {
        color: #a78bfa !important;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .info-card p {
        color: #c4b5fd;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Metric Tiles ── */
    .metric-tile {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.06));
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-tile:hover {
        transform: translateY(-2px);
    }

    .metric-label {
        color: #8b8bbd;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        color: #e0e7ff;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* ── Progress bar ── */
    .progress-container {
        background: rgba(30, 30, 60, 0.6);
        border-radius: 12px;
        height: 20px;
        overflow: hidden;
        margin: 1rem 0;
        border: 1px solid rgba(139, 92, 246, 0.15);
    }

    .progress-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 1s ease-in-out;
    }

    .progress-placed {
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
    }

    .progress-not-placed {
        background: linear-gradient(90deg, #ef4444, #f87171, #fca5a5);
    }

    /* ── Feature bar chart ── */
    .feature-bar-container {
        margin: 0.3rem 0;
    }

    .feature-name {
        color: #c4b5fd;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 2px;
    }

    .feature-bar-bg {
        background: rgba(30, 30, 60, 0.5);
        border-radius: 6px;
        height: 12px;
        overflow: hidden;
    }

    .feature-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s ease;
    }

    .feature-positive {
        background: linear-gradient(90deg, #6366f1, #818cf8);
    }

    .feature-negative {
        background: linear-gradient(90deg, #f87171, #fca5a5);
    }

    /* ── Divider ── */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ── Button styling ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Hide default streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN MODEL (cached — runs once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    """Replicate the exact notebook pipeline and return fitted model + scaler."""

    # Resolve CSV path relative to this script
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Placement_Data_Full_Class.csv")
    data = pd.read_csv(csv_path)

    # Step 1: Drop salary and sl_no
    data_clean = data.drop(columns=['salary', 'sl_no'])

    # Step 2: Strip whitespace from categoricals
    cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t',
                'workex', 'specialisation', 'status']
    for c in cat_cols:
        data_clean[c] = data_clean[c].astype(str).str.strip()

    # Step 3: Create binary target
    data_clean['status_num'] = (data_clean['status'] == 'Placed').astype(int)

    # Step 4: Drop original status
    data_clean = data_clean.drop(columns=['status'])

    # Step 5: Label-encode workex (No→0, Yes→1)
    le = LabelEncoder()
    data_clean['workex'] = le.fit_transform(data_clean['workex'])

    # Step 6: Create academic_average
    data_clean['academic_average'] = data_clean[['ssc_p', 'hsc_p', 'degree_p']].mean(axis=1)

    # Step 7: One-hot encode remaining categoricals
    data_clean = pd.get_dummies(
        data_clean,
        columns=['gender', 'hsc_s', 'degree_t', 'specialisation', 'ssc_b', 'hsc_b'],
        drop_first=True
    )

    # Step 8: Split features / target
    x = data_clean.drop(columns=['status_num'])
    y = data_clean['status_num']
    feature_columns = list(x.columns)

    # Step 9: Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    # Step 10: Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Step 11: Train logistic regression (default params)
    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, feature_columns, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ──────────────────────────────────────────────────────────────────────────────
def predict_placement(model, scaler, feature_columns,
                      gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s,
                      degree_p, degree_t, workex, etest_p,
                      specialisation, mba_p):
    """Transform raw inputs into the 15-feature vector and predict."""

    # Build feature dict
    features = {
        'ssc_p': ssc_p,
        'hsc_p': hsc_p,
        'degree_p': degree_p,
        'workex': 1 if workex == 'Yes' else 0,
        'etest_p': etest_p,
        'mba_p': mba_p,
        'academic_average': np.mean([ssc_p, hsc_p, degree_p]),
        # One-hot encoded features (drop_first)
        'gender_M': 1 if gender == 'M' else 0,
        'hsc_s_Commerce': 1 if hsc_s == 'Commerce' else 0,
        'hsc_s_Science': 1 if hsc_s == 'Science' else 0,
        'degree_t_Others': 1 if degree_t == 'Others' else 0,
        'degree_t_Sci&Tech': 1 if degree_t == 'Sci&Tech' else 0,
        'specialisation_Mkt&HR': 1 if specialisation == 'Mkt&HR' else 0,
        'ssc_b_Others': 1 if ssc_b == 'Others' else 0,
        'hsc_b_Others': 1 if hsc_b == 'Others' else 0,
    }

    # Create dataframe aligned to training column order
    input_df = pd.DataFrame([features])[feature_columns]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    return prediction, probability


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
model, scaler, feature_columns, model_accuracy = train_model()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["M", "F"],
                          format_func=lambda x: "👨 Male" if x == "M" else "👩 Female")

    st.markdown("### 📚 Academic Scores")

    ssc_p = st.slider("SSC Percentage (10th)", 0.0, 100.0, 65.0, 0.1)
    ssc_b = st.selectbox("SSC Board", ["Central", "Others"])

    hsc_p = st.slider("HSC Percentage (12th)", 0.0, 100.0, 65.0, 0.1)
    hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
    hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])

    degree_p = st.slider("Degree Percentage", 0.0, 100.0, 65.0, 0.1)
    degree_t = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])

    st.markdown("### 💼 Professional")

    workex = st.selectbox("Work Experience", ["No", "Yes"],
                          format_func=lambda x: "✅ Yes" if x == "Yes" else "❌ No")
    etest_p = st.slider("E-test Percentage", 0.0, 100.0, 70.0, 0.1)

    st.markdown("### 🎓 MBA")

    specialisation = st.selectbox("MBA Specialisation", ["Mkt&Fin", "Mkt&HR"],
                                  format_func=lambda x: "📊 Mkt & Finance" if x == "Mkt&Fin" else "👥 Mkt & HR")
    mba_p = st.slider("MBA Percentage", 0.0, 100.0, 60.0, 0.1)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Placement Predictor")
st.markdown(
    '<p style="color: #8b8bbd; font-size: 1.05rem; margin-top: -0.8rem;">'
    'Predict campus placement outcomes using Logistic Regression trained on real recruitment data.'
    '</p>',
    unsafe_allow_html=True
)
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# ── Model accuracy tiles ──
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">{model_accuracy * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class="metric-tile">
        <div class="metric-label">Algorithm</div>
        <div class="metric-value" style="font-size: 1.1rem;">Logistic Reg.</div>
    </div>
    """, unsafe_allow_html=True)
with col_c:
    st.markdown("""
    <div class="metric-tile">
        <div class="metric-label">Training Samples</div>
        <div class="metric-value">172</div>
    </div>
    """, unsafe_allow_html=True)
with col_d:
    st.markdown("""
    <div class="metric-tile">
        <div class="metric-label">Features</div>
        <div class="metric-value">15</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


# ── Prediction Results (Live Update) ──
prediction, probability = predict_placement(
    model, scaler, feature_columns,
    gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s,
    degree_p, degree_t, workex, etest_p,
    specialisation, mba_p
)

placed_prob = probability[1] * 100
not_placed_prob = probability[0] * 100
is_placed = prediction == 1

col1, col2 = st.columns([3, 2])

with col1:
    if is_placed:
        st.markdown(f"""
        <div class="result-card placed-card">
            <div class="result-icon">🎉</div>
            <div class="result-title placed-title">PLACED</div>
            <div class="result-confidence">Placement Probability</div>
            <div class="confidence-value placed-confidence">{placed_prob:.1f}%</div>
            <div class="progress-container">
                <div class="progress-bar progress-placed" style="width: {placed_prob}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card not-placed-card">
            <div class="result-icon">📋</div>
            <div class="result-title not-placed-title">NOT PLACED</div>
            <div class="result-confidence">Placement Probability</div>
            <div class="confidence-value not-placed-confidence">{placed_prob:.1f}%</div>
            <div class="progress-container">
                <div class="progress-bar progress-not-placed" style="width: {placed_prob}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Probability Breakdown")
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color: #34d399 !important;">✅ Placed</h4>
        <p style="font-size: 1.8rem; font-weight: 700; color: #6ee7b7; margin: 0;">{placed_prob:.1f}%</p>
    </div>
    <div class="info-card">
        <h4 style="color: #f87171 !important;">❌ Not Placed</h4>
        <p style="font-size: 1.8rem; font-weight: 700; color: #fca5a5; margin: 0;">{not_placed_prob:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Student summary
    acad_avg = np.mean([ssc_p, hsc_p, degree_p])
    st.markdown(f"""
    <div class="info-card" style="margin-top: 0.5rem;">
        <h4>📝 Your Academic Average</h4>
        <p style="font-size: 1.5rem; font-weight: 700; color: #a78bfa; margin: 0;">{acad_avg:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ── About section ──
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    <div class="info-card">
        <h4>📖 About the Model</h4>
        <p>
            This model uses <strong>Logistic Regression</strong> trained on the
            <strong>Campus Recruitment Dataset</strong> (215 students, Kaggle).
            It predicts whether a student will be placed during campus recruitment
            based on academic scores, work experience, and specialisation.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="info-card">
        <h4>🧪 Features Used</h4>
        <p>
            <strong>Academic:</strong> SSC %, HSC %, Degree %, MBA %, E-test %, Academic Average<br>
            <strong>Categorical:</strong> Gender, Board (SSC/HSC), HSC Stream, Degree Type, Specialisation<br>
            <strong>Professional:</strong> Work Experience
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h4>💡 Key Insights from the Model</h4>
    <p>
        <strong>Work experience</strong> is a strong positive predictor — 86.5% placement rate with experience vs 59.6% without.<br>
        <strong>Early academic performance</strong> (SSC, HSC, Degree %) matters more than MBA percentage.<br>
        <strong>Mkt & Finance</strong> specialisation has a higher placement rate (79.2%) than Mkt & HR (55.8%).
    </p>
</div>
""", unsafe_allow_html=True)
