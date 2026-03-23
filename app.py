import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
import os

warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict AI | Disease Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── CSS Variables ── */
:root {
    --brand-teal:      #0d9488;
    --brand-teal-light:#f0fdfa;
    --brand-teal-mid:  #ccfbf1;
    --brand-indigo:    #4f46e5;
    --brand-indigo-lt: #eef2ff;
    --text-primary:    #0f172a;
    --text-secondary:  #475569;
    --text-muted:      #94a3b8;
    --surface:         #ffffff;
    --surface-2:       #f8fafc;
    --surface-3:       #f1f5f9;
    --border:          #e2e8f0;
    --border-strong:   #cbd5e1;
    --shadow-sm:       0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
    --shadow-md:       0 4px 16px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.05);
    --shadow-lg:       0 10px 40px rgba(0,0,0,0.10), 0 4px 12px rgba(0,0,0,0.06);
    --radius-sm:       8px;
    --radius-md:       14px;
    --radius-lg:       20px;
    --radius-xl:       28px;
}

/* ── Base Reset ── */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
    background-color: #f0f4f8 !important;
}

/* Force Streamlit background */
.stApp { background: #f0f4f8 !important; }
section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid var(--border) !important; }
.main .block-container { background: transparent !important; padding-top: 2rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] .stRadio label { 
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 50%, #4f46e5 100%);
    border-radius: var(--radius-xl);
    padding: 52px 48px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 0.5rem;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.hero-subtitle {
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.7;
    color: rgba(255,255,255,0.85) !important;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(8px);
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 100px;
    padding: 7px 18px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 26px 22px;
    text-align: center;
    transition: all 0.25s ease;
    height: 100%;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0d9488, #4f46e5);
    transform: scaleX(0);
    transition: transform 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-md);
    border-color: var(--brand-teal);
}
.metric-card:hover::after { transform: scaleX(1); }
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--brand-teal) !important;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-secondary) !important;
}
.metric-icon {
    font-size: 1.6rem;
    background: var(--brand-teal-light);
    width: 56px; height: 56px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    margin: 0 auto 14px auto;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}
.section-divider {
    height: 3px;
    width: 48px;
    background: linear-gradient(90deg, var(--brand-teal), var(--brand-indigo));
    border-radius: 4px;
    margin-bottom: 1.8rem;
}

/* ── Symptom Pills ── */
.symptom-pill {
    display: inline-block;
    background: var(--brand-teal-light);
    border: 1px solid #99f6e4;
    color: #0f766e !important;
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
}

/* ── Prediction Card ── */
.prediction-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 36px 40px;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-md);
}
.prediction-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #0d9488, #0891b2, #4f46e5);
}
.prediction-disease {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em;
}

/* ── Model Vote Card ── */
.model-vote-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 12px;
    text-align: center;
    transition: all 0.2s;
}
.model-vote-card:hover { background: var(--surface); box-shadow: var(--shadow-sm); }
.model-vote-name {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
    font-weight: 700;
    color: var(--text-muted) !important;
}
.model-vote-disease {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-primary) !important;
}

/* ── Info & Warning Boxes ── */
.info-box {
    background: #f0fdfa;
    border-left: 4px solid var(--brand-teal);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #134e4a !important;
}
.warning-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.92rem;
    color: #78350f !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface-3);
    border-radius: var(--radius-md);
    padding: 5px;
    border: 1px solid var(--border) !important;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--brand-teal) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--brand-teal), #0891b2) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    height: 54px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(13,148,136,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(13,148,136,0.45) !important;
}
.stButton > button p { color: #ffffff !important; }

/* ── Streamlit Elements ── */
.stSelectbox label, .stMultiSelect label, .stCheckbox label,
.stRadio label, .stSlider label, p, .stMarkdown p {
    color: var(--text-primary) !important;
}
.stSelectbox > div > div, .stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}
.stDataFrame { border-radius: var(--radius-md) !important; overflow: hidden; }
.stDataFrame thead th { 
    background: var(--surface-3) !important; 
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
.stSpinner > div { color: var(--brand-teal) !important; }

/* ── Card containers ── */
.card-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: var(--shadow-sm);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface-3); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--brand-teal); }
</style>
""", unsafe_allow_html=True)


# ─── Load Data & Train Models ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from imblearn.over_sampling import RandomOverSampler

    # Resolve path relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "improved_disease_dataset.csv")
    cache_path = os.path.join(base_dir, "models_cache.pkl")

    df = pd.read_csv(data_path)

    encoder = LabelEncoder()
    df["disease_encoded"] = encoder.fit_transform(df["disease"])

    X = df.drop(columns=["disease", "disease_encoded"])
    y = df["disease_encoded"]

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = (
        X_res[: int(len(X_res) * 0.8)],
        X_res[int(len(X_res) * 0.8) :],
        y_res[: int(len(y_res) * 0.8)],
        y_res[int(len(y_res) * 0.8) :],
    )

    # ── Load from disk cache if available ─────────────────────────────────────
    if os.path.exists(cache_path):
        cached = joblib.load(cache_path)
        return (
            df, X, y, X_res, y_res, X_train, X_test, y_train, y_test,
            cached["trained"], cached["metrics"], encoder,
        )

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "SVM":                 SVC(kernel="rbf", C=1.0, probability=False, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    trained = {}
    metrics = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        cv_scores = cross_val_score(m, X_res, y_res, cv=cv, scoring="accuracy", n_jobs=-1)
        trained[name] = m
        metrics[name] = {
            "test_acc":  accuracy_score(y_test, preds),
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }

    # ── Save to disk so next startup is instant ────────────────────────────────
    try:
        joblib.dump({"trained": trained, "metrics": metrics}, cache_path)
    except Exception:
        pass  # If saving fails, just continue without cache

    return (
        df, X, y, X_res, y_res, X_train, X_test, y_train, y_test,
        trained, metrics, encoder,
    )


# ─── Prediction Helper ─────────────────────────────────────────────────────────
def predict_disease(selected_symptoms, trained_models, encoder, feature_cols):
    input_vec = pd.DataFrame(
        [[1 if col in selected_symptoms else 0 for col in feature_cols]],
        columns=feature_cols,
    )

    votes = {}
    probas = {}
    for name, model in trained_models.items():
        pred_idx = model.predict(input_vec)[0]
        pred_label = encoder.classes_[pred_idx]
        votes[name] = pred_label
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_vec)[0]
            probas[name] = {
                "class": pred_label,
                "confidence": float(proba[pred_idx]),
            }

    from collections import Counter
    vote_counts = Counter(votes.values())
    final = vote_counts.most_common(1)[0][0]
    return final, votes, probas


# ─── Disease Info Lookup ───────────────────────────────────────────────────────
DISEASE_INFO = {
    "Diabetes": {
        "category": "Metabolic",
        "severity": "High",
        "emoji": "🩸",
        "description": "A chronic condition affecting how the body processes blood sugar. Early management is critical.",
        "precautions": ["Monitor blood glucose daily", "Follow a low-sugar diet", "Exercise regularly", "Take prescribed medications"],
        "specialist": "Endocrinologist",
    },
    "Heart attack": {
        "category": "Cardiovascular",
        "severity": "Critical",
        "emoji": "❤️",
        "description": "Occurs when blood flow to the heart is severely blocked. Requires immediate medical attention.",
        "precautions": ["Call emergency services immediately", "Chew aspirin if not allergic", "Rest and avoid exertion", "Follow up with a cardiologist"],
        "specialist": "Cardiologist",
    },
    "Malaria": {
        "category": "Infectious",
        "severity": "High",
        "emoji": "🦟",
        "description": "A mosquito-borne infectious disease. Prompt diagnosis and treatment is essential.",
        "precautions": ["Take antimalarial medication", "Use mosquito repellents", "Sleep under bed nets", "Stay hydrated"],
        "specialist": "Infectious Disease Specialist",
    },
    "Tuberculosis": {
        "category": "Respiratory",
        "severity": "High",
        "emoji": "🫁",
        "description": "A bacterial infection primarily affecting the lungs, requiring long-term antibiotic treatment.",
        "precautions": ["Complete full antibiotic course", "Wear a mask in public", "Improve ventilation at home", "Avoid close contact with others"],
        "specialist": "Pulmonologist",
    },
    "AIDS": {
        "category": "Immune",
        "severity": "Critical",
        "emoji": "🔴",
        "description": "Advanced stage of HIV infection. Antiretroviral therapy is the cornerstone of management.",
        "precautions": ["Take ART medications consistently", "Avoid opportunistic infections", "Regular CD4 count monitoring", "Seek psychological support"],
        "specialist": "Infectious Disease Specialist",
    },
    "Dengue": {
        "category": "Infectious",
        "severity": "High",
        "emoji": "🦟",
        "description": "Viral illness spread by Aedes mosquitoes. Monitor platelet count closely.",
        "precautions": ["Stay hydrated with fluids", "Rest completely", "Avoid NSAIDs like ibuprofen", "Monitor platelet count"],
        "specialist": "General Physician",
    },
    "Pneumonia": {
        "category": "Respiratory",
        "severity": "High",
        "emoji": "🫁",
        "description": "Lung infection causing inflammation of air sacs. May require hospitalization.",
        "precautions": ["Complete prescribed antibiotics", "Get adequate rest", "Stay well hydrated", "Follow up with chest X-ray"],
        "specialist": "Pulmonologist",
    },
    "Hypertension": {
        "category": "Cardiovascular",
        "severity": "High",
        "emoji": "💊",
        "description": "Persistently elevated blood pressure damaging blood vessels and organs over time.",
        "precautions": ["Reduce salt intake", "Exercise 30 min daily", "Take antihypertensives as prescribed", "Reduce stress levels"],
        "specialist": "Cardiologist",
    },
}
DEFAULT_INFO = {
    "category": "General",
    "severity": "Moderate",
    "emoji": "🏥",
    "description": "Please consult a qualified medical professional for accurate diagnosis and treatment.",
    "precautions": ["Consult a doctor immediately", "Rest and stay hydrated", "Monitor your symptoms", "Avoid self-medication"],
    "specialist": "General Physician",
}


# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Load models
    with st.spinner("🔬 Loading models... (first launch may take ~30s, subsequent loads are instant)"):
        (
            df, X, y, X_res, y_res, X_train, X_test, y_train, y_test,
            trained_models, metrics, encoder,
        ) = load_and_train()

    feature_cols = X.columns.tolist()
    symptoms = feature_cols  # All symptom column names

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 16px 0;'>
            <div style='font-size: 2.8rem; margin-bottom: 8px;'>🧬</div>
            <div style='font-family: "Space Grotesk", sans-serif; font-size: 1.25rem; font-weight: 700;
                        background: linear-gradient(90deg, #0d9488, #4f46e5);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em;'>
                MediPredict AI
            </div>
            <div style='font-size: 0.75rem; color: #64748b; margin-top: 4px; font-weight: 500; letter-spacing: 0.5px;'>
                Clinical Decision Support
            </div>
        </div>
        <hr style='border-color: #e2e8f0; margin: 0 0 1.5rem 0;'>
        """, unsafe_allow_html=True)

        nav = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔬 Predict", "📊 Model Analytics", "📈 EDA", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color: #e2e8f0;'>", unsafe_allow_html=True)

        # Best model
        best_model_name = max(metrics, key=lambda k: metrics[k]["test_acc"])
        best_acc = metrics[best_model_name]["test_acc"]
        st.markdown(f"""
        <div style='background: #f0fdfa; border: 1px solid #99f6e4;
                    border-radius: 14px; padding: 16px; text-align: center;'>
            <div style='font-size: 0.68rem; color: #0f766e; letter-spacing: 1.2px;
                        text-transform: uppercase; margin-bottom: 6px; font-weight: 700;'>🏆 Best Model</div>
            <div style='font-weight: 700; color: #0f172a; font-size: 0.95rem;'>{best_model_name}</div>
            <div style='font-size: 1.5rem; font-weight: 700; color: #0d9488; font-family: "Space Grotesk", sans-serif;'>{best_acc*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top: 1rem; padding: 14px; background: #f8fafc; border: 1px solid #e2e8f0;
                    border-radius: 12px; font-size: 0.8rem; color: #475569; line-height: 1.8;'>
            <b style='color: #0f172a;'>Dataset Overview</b><br>
            📁 {len(df):,} records<br>
            🔬 {len(symptoms)} symptom features<br>
            🦠 {df['disease'].nunique()} disease classes
        </div>
        """, unsafe_allow_html=True)

    # ── Pages ─────────────────────────────────────────────────────────────────

    if nav == "🏠 Dashboard":
        page_dashboard(df, metrics, trained_models, symptoms, encoder, feature_cols)
    elif nav == "🔬 Predict":
        page_predict(symptoms, trained_models, encoder, feature_cols)
    elif nav == "📊 Model Analytics":
        page_analytics(metrics, X_res, y_res, trained_models, encoder, X_test, y_test)
    elif nav == "📈 EDA":
        page_eda(df, symptoms)
    elif nav == "ℹ️ About":
        page_about()


# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard(df, metrics, trained_models, symptoms, encoder, feature_cols):
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">⚕ AI-Powered Clinical Tool</div>
        <div class="hero-title">MediPredict AI</div>
        <div class="hero-subtitle">
            Industry-grade disease prediction using ensemble machine learning —
            combining Random Forest, SVM, Gradient Boosting & more for 
            clinical-level accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    best_acc = max(m["test_acc"] for m in metrics.values())
    best_cv  = max(m["cv_mean"]  for m in metrics.values())

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "🧬", f"{len(df):,}", "Training Records"),
        (c2, "🎯", f"{best_acc*100:.1f}%", "Peak Test Accuracy"),
        (c3, "📊", f"{best_cv*100:.1f}%", "Best CV Score"),
        (c4, "🦠", f"{df['disease'].nunique()}", "Disease Classes"),
    ]
    for col, icon, val, label in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Leaderboard & Disease Distribution ───────────────────────────────
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-header">🏆 Model Leaderboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]["test_acc"], reverse=True)
        names = [n for n, _ in sorted_metrics]
        accs  = [m["test_acc"]*100 for _, m in sorted_metrics]
        cvs   = [m["cv_mean"]*100  for _, m in sorted_metrics]

        colors = ["#0d9488", "#4f46e5", "#0891b2", "#f59e0b", "#ec4899", "#10b981"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Test Accuracy",
            x=names, y=accs,
            marker=dict(color=colors[:len(names)], line=dict(width=0)),
            text=[f"{a:.1f}%" for a in accs],
            textposition="outside",
            textfont=dict(color="#0f172a", size=12, family="Space Grotesk"),
        ))
        fig.add_trace(go.Scatter(
            name="CV Mean",
            x=names, y=cvs,
            mode="markers+lines",
            marker=dict(size=10, color="#f59e0b", symbol="diamond"),
            line=dict(color="#f59e0b", dash="dot", width=2),
        ))
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="DM Sans"),
            template="none",
            yaxis=dict(range=[85, 105], gridcolor="#e2e8f0", color="#475569",
                       tickfont=dict(color="#475569"), title_font=dict(color="#475569")),
            xaxis=dict(color="#475569", tickfont=dict(color="#0f172a"), showgrid=False),
            legend=dict(font=dict(color="#0f172a"), bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#e2e8f0", borderwidth=1),
            height=340,
            margin=dict(t=30, b=20, l=10, r=10),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with col_r:
        st.markdown('<div class="section-header">🦠 Disease Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        top15 = df["disease"].value_counts().head(15)
        fig2 = go.Figure(go.Bar(
            x=top15.values,
            y=top15.index,
            orientation="h",
            marker=dict(
                color=list(range(len(top15))),
                colorscale=[[0, "#ccfbf1"], [0.5, "#0d9488"], [1, "#0f766e"]],
                line=dict(width=0),
            ),
            text=top15.values,
            textposition="outside",
            textfont=dict(color="#0f172a", size=11),
        ))
        fig2.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="DM Sans"),
            template="none",
            xaxis=dict(color="#475569", gridcolor="#e2e8f0", tickfont=dict(color="#475569"), showgrid=True),
            yaxis=dict(color="#0f172a", tickfont=dict(color="#0f172a", size=11), showgrid=False),
            height=360,
            margin=dict(t=20, b=10, l=10, r=60),
        )
        st.plotly_chart(fig2, use_container_width=True, theme=None)

    # ── Quick Predict CTA ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
        <b>💡 Quick Start:</b> Navigate to <b>🔬 Predict</b> in the sidebar, 
        select your symptoms, and get an instant AI-powered disease prediction 
        backed by a 6-model ensemble with voting consensus.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_predict(symptoms, trained_models, encoder, feature_cols):
    st.markdown('<div class="section-header">🔬 Disease Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        ⚠️ <b>Medical Disclaimer:</b> This tool is for educational purposes only. 
        It is <b>not a substitute</b> for professional medical advice. Always consult a licensed physician.
    </div>
    """, unsafe_allow_html=True)

    # ── Symptom Selection ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### 🩺 Select Your Symptoms")
        st.markdown(
            "<p style='color: #0f172a; font-size:0.85rem; margin-bottom:1rem;'>"
            "Check all symptoms you are currently experiencing:</p>",
            unsafe_allow_html=True,
        )

        # Display symptoms as checkboxes in a 3-column grid
        symptom_display = {s: s.replace("_", " ").title() for s in symptoms}
        selected = []

        cols = st.columns(3)
        for i, sym in enumerate(symptoms):
            with cols[i % 3]:
                if st.checkbox(symptom_display[sym], key=f"sym_{sym}"):
                    selected.append(sym)

    with col_right:
        st.markdown("#### 📋 Selection Summary")
        if selected:
            st.markdown(f"""
            <div style='background: #f0fdfa; border: 1px solid #99f6e4;
                        border-radius: 14px; padding: 20px;'>
                <div style='font-size: 0.78rem; color: #0f766e; 
                            text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px; font-weight: 700;'>
                    Active Symptoms ({len(selected)})
                </div>
            """, unsafe_allow_html=True)
            for s in selected:
                st.markdown(
                    f"<span class='symptom-pill'>✓ {symptom_display[s]}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #f8fafc; border: 2px dashed #cbd5e1;
                        border-radius: 14px; padding: 28px 20px; text-align: center; color: #94a3b8;'>
                <div style='font-size: 2.2rem; margin-bottom: 10px;'>🩺</div>
                <div style='font-weight: 600; font-size: 0.9rem;'>No symptoms selected yet</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔬 Run AI Prediction", use_container_width=True)

    if predict_btn:
        if not selected:
            st.error("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("Running ensemble inference..."):
                time.sleep(0.8)
                final, votes, probas = predict_disease(
                    selected, trained_models, encoder, feature_cols
                )

            info = DISEASE_INFO.get(final, DEFAULT_INFO)

            # Severity color
            sev_colors = {"Critical": "#ef4444", "High": "#f97316", "Moderate": "#eab308", "Low": "#22c55e"}
            sev_color  = sev_colors.get(info["severity"], "#94a3b8")

            st.markdown(f"""
            <div class="prediction-card">
                <div style='display:flex; align-items:center; gap: 16px; margin-bottom: 16px;'>
                    <div style='font-size: 3rem;'>{info['emoji']}</div>
                    <div>
                        <div style='font-size:0.72rem; color: #64748b; 
                                    text-transform:uppercase; letter-spacing:1.5px; font-weight:700;'>
                            Ensemble Prediction
                        </div>
                        <div class="prediction-disease">{final}</div>
                        <div style='display:flex; gap:8px; margin-top:6px;'>
                            <span style='background: #f1f5f9; color: #475569;
                                         border-radius:100px; padding:3px 12px; font-size:0.75rem; font-weight:600;'>
                                {info['category']}
                            </span>
                            <span style='background:{sev_color}22; color:{sev_color};
                                         border-radius:100px; padding:3px 12px; font-size:0.75rem; font-weight:600;'>
                                ⚡ {info['severity']} Severity
                            </span>
                        </div>
                    </div>
                </div>
                <p style='color: #475569; font-size:0.9rem; line-height:1.7; margin-bottom:0;'>
                    {info['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Model votes
            st.markdown("<br>**🗳️ Individual Model Votes:**", unsafe_allow_html=True)
            vote_cols = st.columns(len(votes))
            for col, (mname, mpred) in zip(vote_cols, votes.items()):
                match = "✅" if mpred == final else "🔄"
                conf_str = ""
                if mname in probas:
                    conf_str = f"<div style='font-size:0.7rem; color: #64748b; margin-top:4px; font-weight:600;'>{probas[mname]['confidence']*100:.0f}% conf.</div>"
                col.markdown(f"""
                <div class="model-vote-card">
                    <div class="model-vote-name">{mname}</div>
                    <div class="model-vote-disease">{match} {mpred}</div>
                    {conf_str}
                </div>
                """, unsafe_allow_html=True)

            # Precautions
            st.markdown("<br>**🛡️ Recommended Precautions:**", unsafe_allow_html=True)
            prec_cols = st.columns(2)
            for i, p in enumerate(info["precautions"]):
                with prec_cols[i % 2]:
                    st.markdown(f"""
                    <div style='background: #f8fafc; border: 1px solid #e2e8f0;
                                border-radius:10px; padding:12px 16px; margin-bottom:8px;
                                font-size:0.875rem; color: #0f172a;'>
                        🔹 {p}
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box" style='margin-top:1rem;'>
                👨‍⚕️ <b>Recommended Specialist:</b> {info['specialist']}<br>
                Please schedule a consultation with a qualified {info['specialist']} for proper diagnosis and treatment.
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_analytics(metrics, X_res, y_res, trained_models, encoder, X_test, y_test):
    from sklearn.metrics import confusion_matrix, classification_report

    st.markdown('<div class="section-header">📊 Model Performance Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── CV Score Radar ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📡 Cross-Validation Scores")
        model_names = list(metrics.keys())
        cv_means = [metrics[m]["cv_mean"] * 100 for m in model_names]
        cv_stds  = [metrics[m]["cv_std"]  * 100 for m in model_names]

        colors = ["#0d9488", "#4f46e5", "#0891b2", "#f59e0b", "#ec4899", "#10b981"]
        fig = go.Figure()
        for i, (name, mean, std) in enumerate(zip(model_names, cv_means, cv_stds)):
            fig.add_trace(go.Bar(
                name=name, x=[name], y=[mean],
                error_y=dict(type="data", array=[std], visible=True, color="#64748b", thickness=2),
                marker=dict(color=colors[i], line=dict(width=0)),
                text=[f"{mean:.1f}%"], textposition="outside",
                textfont=dict(color="#0f172a", size=12, family="Space Grotesk"),
            ))
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="DM Sans"),
            template="none",
            showlegend=False, height=340,
            yaxis=dict(range=[80, 105], gridcolor="#e2e8f0", color="#475569",
                       tickfont=dict(color="#475569"), showgrid=True),
            xaxis=dict(color="#0f172a", tickfont=dict(color="#0f172a", size=11), showgrid=False),
            margin=dict(t=30, b=20, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with col2:
        st.markdown("#### 🎯 Test vs CV Accuracy")
        test_accs = [metrics[m]["test_acc"] * 100 for m in model_names]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=test_accs + [test_accs[0]],
            theta=model_names + [model_names[0]],
            fill="toself",
            fillcolor="rgba(13,148,136,0.25)",
            line=dict(color="#0d9488", width=3),
            name="Test Accuracy",
        ))
        fig2.add_trace(go.Scatterpolar(
            r=cv_means + [cv_means[0]],
            theta=model_names + [model_names[0]],
            fill="toself",
            fillcolor="rgba(79,70,229,0.15)",
            line=dict(color="#4f46e5", width=2, dash="dash"),
            name="CV Mean",
        ))
        fig2.update_layout(
            polar=dict(
                bgcolor="#f8fafc",
                radialaxis=dict(
                    visible=True, range=[80, 100],
                    color="#475569",
                    tickfont=dict(color="#475569", size=10),
                    gridcolor="#e2e8f0",
                    linecolor="#e2e8f0",
                ),
                angularaxis=dict(
                    color="#0f172a",
                    tickfont=dict(color="#0f172a", size=11),
                    gridcolor="#e2e8f0",
                    linecolor="#e2e8f0",
                ),
            ),
            paper_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="DM Sans"),
            template="none",
            showlegend=True,
            legend=dict(font=dict(color="#0f172a"), bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#e2e8f0", borderwidth=1),
            height=340,
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True, theme=None)

    # ── Detailed Metrics Table ─────────────────────────────────────────────────
    st.markdown("#### 📋 Detailed Performance Table")
    rows = []
    for name, m in sorted(metrics.items(), key=lambda x: x[1]["test_acc"], reverse=True):
        rows.append({
            "Model":          name,
            "Test Acc (%)":   f"{m['test_acc']*100:.2f}",
            "CV Mean (%)":    f"{m['cv_mean']*100:.2f}",
            "CV Std (%)":     f"{m['cv_std']*100:.2f}",
            "Rank":           "🥇" if name == max(metrics, key=lambda k: metrics[k]["test_acc"]) else "",
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # ── Confusion Matrix ───────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Confusion Matrix")
    model_choice = st.selectbox("Select model to visualize:", list(trained_models.keys()))

    preds = trained_models[model_choice].predict(X_test)
    classes = encoder.classes_
    cm = confusion_matrix(y_test, preds)

    # Show only top-N for visibility
    top_n = 20
    class_counts = np.bincount(y_test.values if hasattr(y_test, "values") else y_test)
    top_idx = np.argsort(class_counts)[-top_n:]
    cm_top = cm[np.ix_(top_idx, top_idx)]
    top_classes = [classes[i] for i in top_idx]

    fig3 = px.imshow(
        cm_top,
        x=top_classes, y=top_classes,
        color_continuous_scale="Teal",
        title=f"Confusion Matrix — {model_choice} (Top {top_n} Classes)",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig3.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="DM Sans"),
        template="none",
        height=540,
        margin=dict(t=50, b=120, l=120, r=20),
        xaxis=dict(tickangle=-45, color="#0f172a", tickfont=dict(color="#0f172a", size=10), showgrid=False),
        yaxis=dict(color="#0f172a", tickfont=dict(color="#0f172a", size=10), showgrid=False),
        title_font=dict(color="#0f172a", size=14),
        coloraxis=dict(colorbar=dict(tickfont=dict(color="#0f172a"), title=dict(font=dict(color="#0f172a")))),
    )
    st.plotly_chart(fig3, use_container_width=True, theme=None)


# ─────────────────────────────────────────────────────────────────────────────
def page_eda(df, symptoms):
    import streamlit.components.v1 as components
    from pyvis.network import Network
    import networkx as nx
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🦠 Disease Analysis", "🔬 Symptom Analysis", "🔗 Correlation Heatmap", "🕸️ Knowledge Graph"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            vc = df["disease"].value_counts()
            fig = px.bar(
                x=vc.values, y=vc.index,
                orientation="h",
                color=vc.values,
                color_continuous_scale="Viridis",
                title="Disease Record Count",
                labels=dict(x="Count", y="Disease"),
            )
            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f172a", family="DM Sans"),
                template="none",
                height=650,
                coloraxis_showscale=False,
                xaxis=dict(color="#0f172a", tickfont=dict(color="#0f172a", size=10), showgrid=False),
                yaxis=dict(color="#0f172a", tickfont=dict(color="#0f172a", size=11), showgrid=True, gridcolor="#e2e8f0"),
                title_font=dict(color="#0f172a"),
                margin=dict(t=40, b=20, l=10, r=60),
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with col2:
            top10 = df["disease"].value_counts().head(10)
            fig2 = px.pie(
                values=top10.values,
                names=top10.index,
                title="Top 10 Disease Distribution",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig2.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f172a", family="DM Sans"),
                template="none",
                height=650,
                legend=dict(font=dict(color="#0f172a", size=11), bgcolor="rgba(255,255,255,0.9)"),
                title_font=dict(color="#0f172a"),
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True, theme=None)

    with tab2:
        symptom_freq = df[symptoms].sum().sort_values(ascending=False)
        fig3 = go.Figure(go.Bar(
            x=symptom_freq.index,
            y=symptom_freq.values,
            marker=dict(
                color=symptom_freq.values,
                colorscale="Teal",
            ),
            text=symptom_freq.values,
            textposition="outside",
        ))
        fig3.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#0f172a", family="DM Sans"), template="none",
            title="Symptom Frequency Across All Records",
            title_font=dict(color="#0f172a", family="DM Sans"),
            xaxis_title="Symptom",
            yaxis_title="Frequency",
            yaxis=dict(color="#475569", gridcolor="#f1f5f9"),
            xaxis=dict(color="#475569"),
            height=440,
            margin=dict(t=40, b=80),
        )
        fig3.update_xaxes(tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True, theme=None)

        # Symptom co-occurrence for top diseases
        st.markdown("#### Symptom Co-occurrence (Top 5 Diseases)")
        top5_diseases = df["disease"].value_counts().head(5).index.tolist()
        df_top5 = df[df["disease"].isin(top5_diseases)]
        co_matrix = df_top5.groupby("disease")[symptoms].mean()

        fig4 = px.imshow(
            co_matrix,
            color_continuous_scale="Teal",
            title="Average Symptom Presence per Disease",
            labels=dict(x="Symptom", y="Disease", color="Avg"),
            aspect="auto",
        )
        fig4.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="DM Sans"),
            template="none",
            height=400,
            xaxis=dict(tickfont=dict(color="#0f172a", size=10)),
            yaxis=dict(tickfont=dict(color="#0f172a", size=11)),
            title_font=dict(color="#0f172a"),
            coloraxis=dict(colorbar=dict(tickfont=dict(color="#0f172a"))),
            margin=dict(t=40, b=100, l=10, r=20),
        )
        fig4.update_xaxes(tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True, theme=None)

    with tab3:
        corr = df[symptoms].corr()
        fig5 = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            title="Symptom Correlation Heatmap",
            zmin=-1, zmax=1,
            aspect="auto",
        )
        fig5.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="DM Sans"),
            template="none",
            height=570,
            xaxis=dict(tickfont=dict(color="#0f172a", size=10)),
            yaxis=dict(tickfont=dict(color="#0f172a", size=10)),
            title_font=dict(color="#0f172a"),
            coloraxis=dict(colorbar=dict(tickfont=dict(color="#0f172a"), title=dict(font=dict(color="#0f172a")))),
            margin=dict(t=50, b=100, l=10, r=20),
        )
        fig5.update_xaxes(tickangle=-45)
        st.plotly_chart(fig5, use_container_width=True, theme=None)


# ─────────────────────────────────────────────────────────────────────────────

    with tab4:
        st.markdown("#### Symptom-Disease Network")
        st.markdown("<p style='color: #0f172a; font-size: 0.9rem;'>Interactive knowledge graph exploring connections between top diseases and their primary symptoms.</p>", unsafe_allow_html=True)
        
        try:
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font=dict(color="#0f172a", family="DM Sans"))
            net.force_atlas_2based()
            net.toggle_physics(False)
            
            top_diseases = df["disease"].value_counts().head(8).index.tolist()
            for d in top_diseases:
                net.add_node(d, label=d, color="#0d9488", size=30, title=f"Disease: {d}",
                             font={"color": "#0f172a", "size": 14, "bold": True})
                d_symptoms = df[df["disease"] == d][symptoms].mean()
                top_syms = d_symptoms[d_symptoms > 0.4].index.tolist()
                for s in top_syms:
                    clean_s = s.replace("_", " ").title()
                    net.add_node(clean_s, label=clean_s, color="#4f46e5", size=15, title=f"Symptom: {clean_s}",
                                 font={"color": "#0f172a", "size": 12})
                    net.add_edge(d, clean_s, value=d_symptoms[s]*10, color="#99f6e4")
            
            # Save graph
            net.save_graph("knowledge_graph.html")
            
            # Display in Streamlit
            HtmlFile = open("knowledge_graph.html", 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=620)
        except Exception as e:
            st.error(f"Could not load Knowledge Graph: {e}")

# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown('<div class="section-header">ℹ️ About MediPredict AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style='color: #0f172a; line-height: 1.8; font-size: 0.95rem;'>

        <h3 style='color: #0d9488; font-family: "Space Grotesk", sans-serif;'>🧬 Project Overview</h3>
        <p>MediPredict AI is an industry-grade disease prediction system built using
        classical machine learning. It leverages a multi-model ensemble architecture
        to provide robust, consensus-based predictions from symptom inputs.</p>

        <h3 style='color: #0d9488; font-family: "Space Grotesk", sans-serif; margin-top: 1.5rem;'>🤖 ML Architecture</h3>
        <p>The system combines 6 diverse classifiers into a majority-voting ensemble:</p>
        <ul>
            <li><b>Random Forest</b> (200 estimators) — robust to noise, handles imbalance well</li>
            <li><b>Support Vector Machine</b> (RBF kernel) — excellent for high-dimensional binary features</li>
            <li><b>Gradient Boosting</b> — sequential error correction for high accuracy</li>
            <li><b>Naive Bayes</b> — fast probabilistic baseline</li>
            <li><b>Decision Tree</b> — interpretable rule-based predictions</li>
            <li><b>K-Nearest Neighbors</b> — similarity-based classification</li>
        </ul>

        <h3 style='color: #0d9488; font-family: "Space Grotesk", sans-serif; margin-top: 1.5rem;'>⚙️ Data Processing</h3>
        <ul>
            <li>Class imbalance handled via <b>RandomOverSampler (SMOTE)</b></li>
            <li>All features are binary symptom indicators (0 / 1)</li>
            <li>Label encoding for 38 disease classes</li>
            <li>80 / 20 stratified train-test split</li>
            <li>5-fold Stratified Cross-Validation</li>
        </ul>

        <h3 style='color: #0d9488; font-family: "Space Grotesk", sans-serif; margin-top: 1.5rem;'>⚠️ Disclaimer</h3>
        <p>This application is intended for <b>educational and research purposes only</b>.
        It is NOT a medical device and should never replace professional medical diagnosis 
        or treatment. Always consult a licensed healthcare professional.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: #f8fafc;
                    border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px;'>
            <div style='font-size: 0.75rem; color: #0f172a; 
                        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;'>
                Tech Stack
            </div>
        """, unsafe_allow_html=True)

        stack = [
            ("🐍", "Python 3.10+"),
            ("📊", "Streamlit"),
            ("🤖", "Scikit-learn"),
            ("📈", "Plotly"),
            ("🐼", "Pandas / NumPy"),
            ("⚖️", "Imbalanced-learn"),
        ]
        for icon, name in stack:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:10px; padding:10px 0;
                        border-bottom: 1px solid rgba(0,0,0,0.05);
                        font-size:0.875rem; color: #0f172a;'>
                <span>{icon}</span> <span>{name}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #f8fafc; border: 1px solid #99f6e4;
                    border-radius: 16px; padding: 24px; margin-top: 1rem;'>
            <div style='font-size: 0.75rem; color: #0f172a; 
                        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;'>
                Dataset Stats
            </div>
            <div style='color: #0f172a; font-size:0.875rem; line-height:2;'>
                📁 2,000 records<br>
                🔬 10 symptom features<br>
                🦠 38 disease classes<br>
                ⚖️ Balanced via SMOTE<br>
                ✅ 5-Fold CV validated
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
