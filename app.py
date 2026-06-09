# ============================================================
# app.py — AI Impact on Students: Prediction Dashboard
# ============================================================


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Handle imports that might fail on Streamlit Cloud
try:
    import joblib
except ImportError:
    from sklearn.utils import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Impact on Students",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Source+Serif+4:wght@400;600&display=swap');

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-title {
        font-family: 'Source Serif 4', serif;
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
    }
    .metric-card h3 {
        font-size: 1.8rem;
        margin: 0;
        font-family: 'DM Serif Display', serif;
    }
    .metric-card p {
        font-size: 0.85rem;
        margin: 0;
        opacity: 0.9;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS AND CONFIG
# ============================================================
@st.cache_resource
def load_models():
    clf = joblib.load('best_classifier.pkl')
    reg = joblib.load('best_regressor.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    return clf, reg, scaler, le

@st.cache_data
def load_config():
    with open('model_config.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('students_ai_engineered.csv')
    except:
        return None

clf_model, reg_model, scaler, label_encoder = load_models()
config = load_config()
df = load_dataset()


# ============================================================
# HEADER
# ============================================================
st.markdown('<h1 class="main-title">AI Impact on Students</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning Dashboard — Predict Academic Outcomes Based on AI Usage Patterns</p>', unsafe_allow_html=True)
st.markdown("---")


# ============================================================
# SIDEBAR — USER INPUT
# ============================================================
st.sidebar.markdown("## Student Profile")
st.sidebar.markdown("Enter the student details below:")

# Demographics
st.sidebar.markdown("### Demographics")
major = st.sidebar.selectbox("Major Category", config['major_categories'])
year = st.sidebar.selectbox("Year of Study", config['year_options'])

# Academic baseline
st.sidebar.markdown("### Academic Baseline")
pre_gpa = st.sidebar.slider("Pre-Semester GPA", 0.0, 4.0, 3.0, 0.01)
traditional_hours = st.sidebar.slider("Traditional Study Hours/Week", 0.0, 40.0, 10.0, 0.5)

# AI usage
st.sidebar.markdown("### AI Usage")
ai_hours = st.sidebar.slider("Weekly GenAI Hours", 0.0, 40.0, 5.0, 0.5)
use_case = st.sidebar.selectbox("Primary AI Use Case", config['use_cases'])
prompt_skill = st.sidebar.selectbox("Prompt Engineering Skill", ['Beginner', 'Intermediate', 'Advanced'])
tool_diversity = st.sidebar.slider("Tool Diversity (number of AI tools)", 0, 10, 3)
paid_sub = st.sidebar.checkbox("Paid AI Subscription", value=False)

# Psychology
st.sidebar.markdown("### Psychology & Well-being")
dependency = st.sidebar.slider("Perceived AI Dependency (1-10)", 1, 10, 5)
anxiety = st.sidebar.slider("Exam Anxiety Level (1-10)", 1, 10, 5)

# Institutional
st.sidebar.markdown("### Institutional Policy")
policy = st.sidebar.selectbox("Institutional AI Policy",
                               ['Strict_Ban', 'Allowed_With_Citation', 'Actively_Encouraged'])

# Predict button
predict_btn = st.sidebar.button("Predict Outcomes", type="primary", use_container_width=True)


# ============================================================
# MODEL PERFORMANCE METRICS (TOP ROW)
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""<div class="metric-card">
        <h3>{config['cls_accuracy']:.1%}</h3>
        <p>Classification Accuracy</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="metric-card">
        <h3>{config['cls_f1']:.1%}</h3>
        <p>F1 Score (Macro)</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="metric-card">
        <h3>{config['reg_r2']:.3f}</h3>
        <p>Regression R²</p>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="metric-card">
        <h3>{config['reg_rmse']:.3f}</h3>
        <p>Regression RMSE</p>
    </div>""", unsafe_allow_html=True)

st.markdown("")


# ============================================================
# ENCODE USER INPUT → FEATURE VECTOR
# ============================================================
def encode_input(major, year, pre_gpa, ai_hours, use_case, prompt_skill,
                 tool_diversity, paid_sub, traditional_hours, dependency,
                 anxiety, policy):
    """Convert user inputs into the feature vector the model expects."""

    eps = 0.001

    year_map = config['year_map']
    skill_map = config['skill_map']
    policy_map = config['policy_map']

    # Ordinal encodings
    year_enc = year_map.get(year, 3)
    skill_enc = skill_map.get(prompt_skill, 2)
    policy_enc = policy_map.get(policy, 2)
    paid_int = int(paid_sub)

    # Engineered features
    gpa_change_placeholder = 0  # unknown at prediction time
    ai_study_ratio = round(ai_hours / (traditional_hours + eps), 3)
    ai_dominant = int(ai_hours > traditional_hours)
    dep_skill_ratio = round(dependency / (skill_enc + eps), 3)
    high_ai = int(ai_hours >= 15)  # approximate threshold

    # Build feature dict
    feature_dict = {
        'Pre_Semester_GPA': pre_gpa,
        'Weekly_GenAI_Hours': ai_hours,
        'Tool_Diversity': tool_diversity,
        'Traditional_Study_Hours': traditional_hours,
        'Perceived_AI_Dependency': dependency,
        'Anxiety_Level_During_Exams': anxiety,
        'Skill_Retention_Score': 50,  # placeholder — model will predict based on pattern
        'Year_of_Study_Encoded': year_enc,
        'Prompt_Skill_Encoded': skill_enc,
        'Policy_Encoded': policy_enc,
        'Burnout_Encoded': 2,  # placeholder
        'Paid_Sub_Int': paid_int,
        'AI_to_Study_Ratio': ai_study_ratio,
        'AI_Dominant': ai_dominant,
        'High_AI_User': high_ai,
        'Dep_Skill_Ratio': dep_skill_ratio,
    }

    # One-hot: Major
    for m in config['major_categories']:
        feature_dict[f'Major_{m}'] = 1 if m == major else 0

    # One-hot: Use Case
    for u in config['use_cases']:
        feature_dict[f'Use_{u}'] = 1 if u == use_case else 0

    # Build DataFrame with correct column order
    feature_cols = config['feature_cols']
    row = pd.DataFrame([feature_dict])

    # Add any missing columns with 0
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    # Reorder to match training
    row = row[feature_cols]

    return row


# ============================================================
# PREDICTIONS
# ============================================================
if predict_btn:
    input_df = encode_input(major, year, pre_gpa, ai_hours, use_case,
                            prompt_skill, tool_diversity, paid_sub,
                            traditional_hours, dependency, anxiety, policy)

    # Classification prediction
    cls_pred = clf_model.predict(input_df)[0]
    cls_label = label_encoder.inverse_transform([cls_pred])[0]
    cls_proba = clf_model.predict_proba(input_df)[0]

    # Regression prediction
    reg_pred = reg_model.predict(input_df)[0]

    # Display results
    st.markdown("## Prediction Results")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("### GPA Direction (Classification)")

        color_map = {'Improved': '#2a9d8f', 'Stable': '#e9c46a', 'Declined': '#e76f51'}
        emoji_map = {'Improved': '📈', 'Stable': '➡️', 'Declined': '📉'}

        st.markdown(f"""
        <div style="background: {color_map.get(cls_label, '#ccc')}22;
                    border: 2px solid {color_map.get(cls_label, '#ccc')};
                    border-radius: 12px; padding: 1.5rem; text-align: center;">
            <h2 style="margin:0; color: {color_map.get(cls_label, '#ccc')};">
                {emoji_map.get(cls_label, '')} {cls_label}
            </h2>
            <p style="margin:0.5rem 0 0 0; font-size: 0.9rem;">
                Predicted GPA direction based on AI usage profile
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("**Class Probabilities:**")
        for label, prob in zip(label_encoder.classes_, cls_proba):
            st.progress(prob, text=f"{label}: {prob:.1%}")

    with res_col2:
        st.markdown("### Exact GPA Change (Regression)")

        arrow = "↑" if reg_pred > 0 else ("↓" if reg_pred < 0 else "→")
        color = "#2a9d8f" if reg_pred > 0 else ("#e76f51" if reg_pred < 0 else "#e9c46a")

        st.markdown(f"""
        <div style="background: {color}22;
                    border: 2px solid {color};
                    border-radius: 12px; padding: 1.5rem; text-align: center;">
            <h2 style="margin:0; color: {color};">
                {arrow} {reg_pred:+.3f} GPA Points
            </h2>
            <p style="margin:0.5rem 0 0 0; font-size: 0.9rem;">
                Predicted change from {pre_gpa:.2f} → {pre_gpa + reg_pred:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge-style display
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(['GPA Change'], [reg_pred], color=color, height=0.4)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlim(-1.5, 1.5)
        ax.set_xlabel("GPA Change")
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)
        plt.close()

    # ============================================================
    # SHAP EXPLANATION FOR THIS PREDICTION
    # ============================================================
    st.markdown("---")
    st.markdown("## Why This Prediction? (SHAP Explanation)")

    try:
        if not SHAP_AVAILABLE:
            st.info("SHAP explanation not available in this deployment.")
        else:
            explainer = shap.TreeExplainer(clf_model)
            shap_values = explainer(input_df)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.waterfall_plot(shap_values[0], max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("""
            <div class="insight-box">
                <strong>How to read this:</strong> Features in <span style="color:#e76f51;">red</span>
                push the prediction higher. Features in <span style="color:#2a9d8f;">blue</span>
                push it lower.
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.info(f"SHAP explanation not available: {e}")

    # ============================================================
    # INPUT SUMMARY
    # ============================================================
    st.markdown("---")
    st.markdown("### Input Summary")

    summary_data = {
        'Category': ['Demographics', 'Demographics', 'Academic', 'AI Usage',
                     'AI Usage', 'AI Usage', 'AI Usage', 'AI Usage',
                     'Well-being', 'Well-being', 'Institutional'],
        'Feature': ['Major', 'Year', 'Pre-Semester GPA', 'AI Hours/Week',
                    'Use Case', 'Prompt Skill', 'Tool Diversity', 'Paid Sub',
                    'AI Dependency', 'Exam Anxiety', 'Policy'],
        'Value': [major, year, f"{pre_gpa:.2f}", f"{ai_hours:.1f}h",
                  use_case, prompt_skill, tool_diversity, "Yes" if paid_sub else "No",
                  f"{dependency}/10", f"{anxiety}/10", policy]
    }
    st.table(pd.DataFrame(summary_data))


# ============================================================
# EXPLORATORY DATA ANALYSIS TAB (always visible)
# ============================================================
else:
    st.markdown("## Dataset Overview")

    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Distribution", "AI Usage Impact", "Burnout & Anxiety", "Feature Correlations"
        ])

        with tab1:
            col_a, col_b = st.columns(2)

            with col_a:
                fig, ax = plt.subplots(figsize=(8, 5))
                df['GPA_Change'] = df['Post_Semester_GPA'] - df['Pre_Semester_GPA']
                ax.hist(df['GPA_Change'], bins=50, color='#667eea', edgecolor='white', alpha=0.85)
                ax.axvline(df['GPA_Change'].mean(), color='red', linestyle='--',
                           label=f"Mean: {df['GPA_Change'].mean():.3f}")
                ax.set_title("GPA Change Distribution", fontweight='bold')
                ax.set_xlabel("GPA Change")
                ax.legend()
                st.pyplot(fig)
                plt.close()

            with col_b:
                fig, ax = plt.subplots(figsize=(8, 5))
                major_means = df.groupby('Major_Category')['GPA_Change'].mean().sort_values()
                major_means.plot(kind='barh', color='#764ba2', ax=ax, edgecolor='white')
                ax.axvline(0, color='red', linewidth=0.8)
                ax.set_title("Avg GPA Change by Major", fontweight='bold')
                ax.set_xlabel("Mean GPA Change")
                st.pyplot(fig)
                plt.close()

        with tab2:
            col_a, col_b = st.columns(2)

            with col_a:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(df['Weekly_GenAI_Hours'], df['GPA_Change'],
                           alpha=0.05, s=8, color='#e76f51')
                z = np.polyfit(df['Weekly_GenAI_Hours'], df['GPA_Change'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(0, df['Weekly_GenAI_Hours'].max(), 100)
                ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
                ax.set_xlabel("Weekly AI Hours")
                ax.set_ylabel("GPA Change")
                ax.set_title("AI Hours vs GPA Change", fontweight='bold')
                ax.legend()
                st.pyplot(fig)
                plt.close()

            with col_b:
                fig, ax = plt.subplots(figsize=(8, 5))
                use_means = df.groupby('Primary_Use_Case')['GPA_Change'].mean().sort_values()
                use_means.plot(kind='barh', color='#2a9d8f', ax=ax, edgecolor='white')
                ax.axvline(0, color='red', linewidth=0.8)
                ax.set_title("GPA Change by AI Use Case", fontweight='bold')
                ax.set_xlabel("Mean GPA Change")
                st.pyplot(fig)
                plt.close()

        with tab3:
            col_a, col_b = st.columns(2)

            with col_a:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df, x='Burnout_Risk_Level', y='Weekly_GenAI_Hours',
                            order=['Low', 'Medium', 'High'], palette='YlOrRd', ax=ax)
                ax.set_title("AI Hours by Burnout Level", fontweight='bold')
                st.pyplot(fig)
                plt.close()

            with col_b:
                fig, ax = plt.subplots(figsize=(8, 5))
                dep_means = df.groupby('Perceived_AI_Dependency')['Anxiety_Level_During_Exams'].mean()
                ax.bar(dep_means.index, dep_means.values, color='#e76f51', edgecolor='white')
                ax.set_xlabel("AI Dependency Level")
                ax.set_ylabel("Mean Exam Anxiety")
                ax.set_title("Dependency vs Exam Anxiety", fontweight='bold')
                st.pyplot(fig)
                plt.close()

        with tab4:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Student_ID' in numeric_cols:
                numeric_cols.remove('Student_ID')

            selected_cols = st.multiselect(
                "Select features for correlation:",
                numeric_cols,
                default=numeric_cols[:8]
            )

            if len(selected_cols) >= 2:
                fig, ax = plt.subplots(figsize=(12, 9))
                corr = df[selected_cols].corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                            center=0, square=True, linewidths=0.5, ax=ax,
                            annot_kws={'size': 9})
                ax.set_title("Feature Correlation Matrix", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    else:
        st.info("Upload `students_ai_engineered.csv` to see dataset visualizations.")

    # ============================================================
    # SIDEBAR: MODEL INFO
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Model Info")
    st.sidebar.markdown(f"""
    **Classification Model:**
    - Algorithm: {config['classification_model_name']}
    - Accuracy: {config['cls_accuracy']:.1%}
    - F1 Score: {config['cls_f1']:.1%}

    **Regression Model:**
    - Algorithm: {config['regression_model_name']}
    - R² Score: {config['reg_r2']:.4f}
    - RMSE: {config['reg_rmse']:.4f}
    """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #aaa; font-size: 0.85rem;">
    Built with Streamlit, Scikit-learn, XGBoost & SHAP | AI Impact on Students Research Project
</div>
""", unsafe_allow_html=True)
